"""
core/lemma_bank.py — 已证引理银行

核心思想 (来自 Seed-Prover)：
  在多次证明尝试中，即使完整证明失败了，其中一些 sub-lemma
  可能已经被 Lean 成功验证。把这些"中间成果"提取出来，
  注入到后续尝试的上下文中，相当于跨 rollout 共享经验。

两个职责：
  1. 从失败/成功的 proof 中提取可复用的 lemma
  2. 管理已积累的 lemma 集合，生成可注入 prompt 的文本

同时，这套 lemma 积累数据天然就是 RL 训练的经验数据——
每条 (state, lemma, verified) 三元组就是一条训练信号。
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from core.models import AttemptStatus

logger = logging.getLogger(__name__)


@dataclass
class ProvedLemma:
    """一条已验证的引理"""
    name: str                  # 引理名
    statement: str             # 完整的 Lean 声明 (lemma xxx : yyy)
    proof: str                 # 证明体
    source_attempt: int        # 在哪次 attempt 中被发现
    source_rollout: int = 0    # 在哪个并行 rollout 中被发现
    verified: bool = True      # 是否被 Lean 单独验证过

    def to_lean(self) -> str:
        """输出可直接插入 Lean 文件的代码"""
        return f"{self.statement} {self.proof}"

    def to_prompt_str(self) -> str:
        """输出适合放入 LLM prompt 的描述"""
        return f"-- Already proved:\n{self.statement} {self.proof}"


class LemmaBank:
    """
    已证引理的累积存储。

    Usage:
        bank = LemmaBank(lean_checker)

        # 每次 rollout 后，从生成的 proof 中提取 lemma
        bank.extract_and_verify(proof_code, attempt_num=1, rollout_id=0)

        # 生成注入下一轮 prompt 的文本
        context = bank.to_prompt_context()

        # 生成注入 Lean 文件的前置代码
        preamble = bank.to_lean_preamble()
    """

    def __init__(self, lean_checker=None):
        self.lean = lean_checker
        self.lemmas: list[ProvedLemma] = []
        self._seen_statements: set[str] = set()  # 去重

    @property
    def count(self) -> int:
        return len(self.lemmas)

    def extract_and_verify(
        self,
        proof_code: str,
        theorem_statement: str,
        attempt_num: int = 0,
        rollout_id: int = 0,
        verify: bool = True,
    ) -> list[ProvedLemma]:
        """
        从一段 proof 代码中提取 lemma/have 块，
        可选地单独验证它们是否能通过 Lean 编译。

        Args:
            proof_code:         完整的 proof 代码
            theorem_statement:  原始 theorem 声明 (用于构建验证文件)
            attempt_num:        当前 attempt 编号
            rollout_id:         当前 rollout 编号
            verify:             是否单独验证每个 lemma

        Returns:
            新提取的已验证 lemma 列表
        """
        candidates = self._parse_lemmas(proof_code)
        new_lemmas = []

        for name, statement, body in candidates:
            # 去重：相同 statement 不重复验证
            norm = _normalize(statement)
            if norm in self._seen_statements:
                continue
            self._seen_statements.add(norm)

            lemma = ProvedLemma(
                name=name,
                statement=statement,
                proof=body,
                source_attempt=attempt_num,
                source_rollout=rollout_id,
                verified=False,
            )

            # 可选：单独验证这个 lemma 能否通过 Lean
            if verify and self.lean:
                verified = self._verify_lemma(lemma, theorem_statement)
                lemma.verified = verified
                if not verified:
                    logger.debug(f"  Lemma '{name}' failed verification, skipping")
                    continue
            else:
                lemma.verified = True  # 无法验证时信任它

            self.lemmas.append(lemma)
            new_lemmas.append(lemma)
            logger.info(f"  Banked lemma: {name}")

        return new_lemmas

    def to_prompt_context(self, max_lemmas: int = 10) -> str:
        """生成可注入 LLM prompt 的已证引理上下文"""
        if not self.lemmas:
            return ""

        recent = self.lemmas[-max_lemmas:]
        parts = [
            "## Already proved lemmas (you can use these directly)",
            "The following lemmas have been verified by the Lean kernel.",
            "You can reference them in your proof.\n",
        ]
        for lem in recent:
            parts.append(f"```lean\n{lem.statement} {lem.proof}\n```\n")

        return "\n".join(parts)

    def to_lean_preamble(self, max_lemmas: int = 20) -> str:
        """生成可插入 Lean 文件的前置引理定义"""
        if not self.lemmas:
            return ""

        recent = self.lemmas[-max_lemmas:]
        lines = [f"\n-- Banked lemmas from previous rollouts"]
        for lem in recent:
            lines.append(lem.to_lean())
            lines.append("")
        return "\n".join(lines)

    def get_rl_experience(self) -> list[dict]:
        """
        导出 RL 训练数据：每条 lemma 就是一条
        (state=theorem_context, action=lemma_proof, reward=verified) 经验。

        这套数据可以直接用于：
          - SFT：把 verified=True 的 lemma 作为正样本
          - DPO：verified=True vs verified=False 作为偏好对
          - RL：reward = 1 if verified else 0
        """
        return [
            {
                "name": lem.name,
                "statement": lem.statement,
                "proof": lem.proof,
                "verified": lem.verified,
                "source_attempt": lem.source_attempt,
                "source_rollout": lem.source_rollout,
            }
            for lem in self.lemmas
        ]

    def clear(self):
        """清空（新题目时调用）"""
        self.lemmas.clear()
        self._seen_statements.clear()

    # ── 内部方法 ──────────────────────────────────────────────

    def _parse_lemmas(self, proof_code: str) -> list[tuple[str, str, str]]:
        """
        从 proof 代码中提取 lemma 和 have 块。

        返回: [(name, statement, proof_body), ...]
        """
        results = []

        # 提取显式 lemma 声明
        # 匹配: lemma name ... := by ...  或  lemma name ... := ...
        lemma_pattern = re.compile(
            r"(lemma\s+(\w+)\s+.*?)\s*(:=.*?)"
            r"(?=\n\s*(?:lemma|theorem|def|end|$)|\Z)",
            re.DOTALL,
        )
        for match in lemma_pattern.finditer(proof_code):
            statement = match.group(1).strip()
            name = match.group(2)
            body = match.group(3).strip()
            if body and name:
                results.append((name, statement, body))

        # 提取 have 块 (命名的中间步骤)
        # 匹配: have name : type := ...  或  have name : type by ...
        have_pattern = re.compile(
            r"have\s+(\w+)\s*:\s*([^:=]+?)\s*:=\s*(.*?)(?=\n\s*(?:have|show|exact|apply|calc|sorry|$))",
            re.DOTALL,
        )
        for match in have_pattern.finditer(proof_code):
            name = match.group(1)
            type_sig = match.group(2).strip()
            body = match.group(3).strip()
            if name and type_sig and body:
                statement = f"lemma {name} : {type_sig}"
                proof = f":= {body}"
                results.append((name, statement, proof))

        return results

    def _verify_lemma(self, lemma: ProvedLemma, theorem_statement: str) -> bool:
        """单独验证一个 lemma 是否能通过 Lean 编译"""
        try:
            status, _, _, _, _ = self.lean.check(
                theorem_statement=lemma.statement,
                proof=lemma.proof,
            )
            return status == AttemptStatus.SUCCESS
        except Exception as e:
            logger.debug(f"  Lemma verification error: {e}")
            return False


def _normalize(s: str) -> str:
    """规范化 statement 字符串用于去重"""
    return re.sub(r'\s+', ' ', s.strip().lower())
