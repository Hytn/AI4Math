"""prover/hilbert/orchestrator.py — Hilbert 式递归分解调度器

参照: Hilbert (arXiv:2509.22819, Apple+UCSD; miniF2F 99.2%,
PutnamBench 462/660) 的 agentic 流程; 官方代码未公开, 工程结构以最
接近的开源复现 Gödel's Poetry (arXiv:2512.14252) 为代码参照。

每个证明节点 (goal) 依序走四条通路:

  1. PROVER 整证:   专用 prover LLM 生成 k 个完整证明 → verifier 逐个判
  2. SHALLOW 浅解:  reasoner LLM 试 m 个"短战术证明" (simp/omega/
                    nlinarith/field_simp 一击型)。Gödel's Poetry 的
                    实验结论: 这条廉价通路显著加速收敛 —— 大量中间
                    子目标其实是平凡的, 不值得动用 prover 大预算。
  3. DECOMPOSE 分解: 仍失败且 depth < D 时, reasoner 把 goal 分解为
                    ≤N 个**自包含** lemma 子目标 + 一个引用它们的
                    主证明 (assembly)。每个子目标递归走 1-3。
  4. ASSEMBLE 组装:  所有子目标证出后, 拼成
                    `lemma h1 ... lemma h2 ... theorem main := assembly`
                    的完整代码块, 交 verifier 终审 (组装结果必须
                    整体可编译, 不存在"纸面成功")。

解耦边界:
  - LLM: 任何带 `async generate(system, user, temperature, max_tokens)`
    → LLMResponse 形状的对象 (agent.brain 的 provider 天然满足);
  - verifier: 注入的 `async (code: str) -> tuple[bool, list[str]]`,
    code 是不含 preamble 的完整声明代码块。run_hilbert.py 把它接到
    AsyncLeanPool.verify_complete; 测试里注入假 verifier。
  - 不 import engine / prover.unified / agent.runtime 的任何模块。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from prover.codegen.code_formatter import (
    extract_proof_body, statement_head)
from prover.hilbert.config import HilbertConfig

logger = logging.getLogger(__name__)

VerifyFn = Callable[[str], Awaitable[tuple[bool, list[str]]]]

# ─── 提示词 ─────────────────────────────────────────────────────────

PROVER_SYSTEM = """\
You are an expert Lean 4 theorem prover using Mathlib. Complete the
proof of the given theorem. Output ONLY the proof term/tactic block
starting with `by` — no theorem header, no imports, no fences, no prose."""

SHALLOW_SYSTEM = """\
You are a Lean 4 expert. The goal below is likely closable by a SHORT
tactic proof (≤ 3 tactics): think simp, omega, norm_num, ring,
nlinarith, positivity, aesop, decide, field_simp, linarith,
exact?-style direct term. Output ONLY the tactic block starting with
`by`. If you believe no short proof exists, output exactly: NO_SHORT_PROOF"""

DECOMPOSE_SYSTEM = """\
You are a Lean 4 proof architect. Decompose the target theorem into at
most {max_subgoals} SELF-CONTAINED helper lemmas plus an assembly proof
of the target that uses them.

Hard rules:
1. Each lemma statement must be closed: it may NOT reference variables
   or hypotheses of the target theorem unless re-bound in its own
   signature. It must compile standalone against Mathlib.
2. Name lemmas aux1, aux2, ... .
3. The assembly proof proves the ORIGINAL theorem and may use aux1...
   by name. Start it with `by`.
4. Reply with ONLY a JSON object, no fences:
{{"subgoals": [{{"name": "aux1", "statement": "lemma aux1 : ... "}}, ...],
  "assembly": "by ..."}}"""

# ─── 数据结构 ───────────────────────────────────────────────────────

@dataclass
class ProofNode:
    statement: str                 # 完整声明头 (theorem/lemma ... ), 不含证明
    depth: int = 0
    status: str = "open"           # open / proved / failed
    method: str = ""               # prover / shallow / decompose
    proof: str = ""                # 成功时: `by ...` 体
    assembled_code: str = ""       # decompose 成功时: 含子 lemma 的完整代码
    children: list["ProofNode"] = field(default_factory=list)
    assembly: str = ""
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "statement": self.statement, "depth": self.depth,
            "status": self.status, "method": self.method,
            "proof": self.proof, "assembly": self.assembly,
            "assembled_code": self.assembled_code,
            "failures": self.failures[-3:],
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class RoleStats:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


# ─── 纯函数: 文本处理 (单测覆盖) ────────────────────────────────────

_FENCE = re.compile(r"```(?:lean4?)?\s*\n?(.*?)```", re.DOTALL)
_BANNED = ("sorry", "admit", "native_decide", "axiom ")


def proof_is_clean(body: str) -> bool:
    low = (body or "").lower()
    return bool(body) and not any(b in low for b in _BANNED)


def parse_decomposition(raw: str, max_subgoals: int) -> dict | None:
    """解析 reasoner 的分解 JSON; 失败返回 None (调用方带反馈重试)。"""
    text = (raw or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        d = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    subs = d.get("subgoals")
    asm = (d.get("assembly") or "").strip()
    if not isinstance(subs, list) or not subs or not asm:
        return None
    out = []
    for sg in subs[:max_subgoals]:
        if not isinstance(sg, dict):
            return None
        stmt = (sg.get("statement") or "").strip()
        name = (sg.get("name") or "").strip()
        if not stmt or not name:
            return None
        if not stmt.startswith(("lemma", "theorem")):
            stmt = f"lemma {name} : {stmt}"
        out.append({"name": name, "statement": statement_head(stmt)})
    if not asm.startswith("by"):
        asm = "by " + asm
    return {"subgoals": out, "assembly": asm}


def assemble(node: ProofNode) -> str:
    """子目标全部证出后, 拼完整可编译代码块。"""
    parts = []
    for c in node.children:
        if c.assembled_code:
            parts.append(c.assembled_code)
        else:
            parts.append(f"{c.statement} := {c.proof}")
    parts.append(f"{statement_head(node.statement)} := {node.assembly}")
    return "\n\n".join(parts)


# ─── 调度器 ─────────────────────────────────────────────────────────

class HilbertOrchestrator:
    def __init__(self, cfg: HilbertConfig, reasoner_llm, prover_llm,
                 verify: VerifyFn):
        self.cfg = cfg
        self.reasoner = reasoner_llm
        self.prover = prover_llm
        self.verify = verify
        self.stats = {"reasoner": RoleStats(), "prover": RoleStats()}
        self._nodes_used = 0

    # ── LLM 调用包装 (统一成本对账) ──────────────────────────────

    async def _call(self, role: str, system: str, user: str) -> str:
        llm = self.reasoner if role == "reasoner" else self.prover
        rm = getattr(self.cfg, role)
        st = self.stats[role]
        st.calls += 1
        try:
            resp = await llm.generate(
                system=system, user=user,
                temperature=rm.temperature, max_tokens=rm.max_tokens)
        except Exception as e:  # noqa: BLE001
            logger.warning("hilbert %s call failed: %s", role, e)
            return ""
        st.tokens_in += int(getattr(resp, "tokens_in", 0) or 0)
        st.tokens_out += int(getattr(resp, "tokens_out", 0) or 0)
        return getattr(resp, "content", "") or ""

    # ── 主入口 ───────────────────────────────────────────────────

    async def prove(self, statement: str) -> ProofNode:
        """对单个题面运行完整递归流程, 返回根节点 (含全树 trace)。"""
        self._nodes_used = 0
        root = ProofNode(statement=statement_head(statement), depth=0)
        t0 = time.time()
        await self._solve(root)
        logger.info("hilbert: %s in %.1fs (%d nodes, reasoner %d calls, "
                    "prover %d calls)", root.status, time.time() - t0,
                    self._nodes_used, self.stats["reasoner"].calls,
                    self.stats["prover"].calls)
        return root

    async def _solve(self, node: ProofNode):
        self._nodes_used += 1
        if self._nodes_used > self.cfg.node_budget:
            node.status = "failed"
            node.failures.append("node budget exhausted")
            return

        # 1) prover 整证
        if await self._try_role(node, "prover", PROVER_SYSTEM,
                                self.cfg.prover_passes):
            node.method = "prover"
            return
        # 2) reasoner 浅解
        if await self._try_shallow(node):
            node.method = "shallow"
            return
        # 3) 分解
        if node.depth < self.cfg.max_depth:
            if await self._try_decompose(node):
                node.method = "decompose"
                return
        node.status = "failed"

    # ── 通路 1/2: 整证与浅解 ─────────────────────────────────────

    async def _try_role(self, node: ProofNode, role: str, system: str,
                        passes: int) -> bool:
        user = (f"Prove this Lean 4 theorem:\n\n{node.statement} := by\n"
                f"  sorry\n\nOutput the proof block only.")
        for _ in range(max(0, passes)):
            body = extract_proof_body(await self._call(role, system, user))
            if not proof_is_clean(body):
                continue
            ok, errs = await self.verify(f"{node.statement} := {body}")
            if ok:
                node.status, node.proof = "proved", body
                return True
            node.failures.append((errs[0] if errs else "verify failed")[:300])
        return False

    async def _try_shallow(self, node: ProofNode) -> bool:
        user = f"Goal:\n\n{node.statement}\n\nShort tactic proof only."
        for _ in range(max(0, self.cfg.shallow_passes)):
            raw = await self._call("reasoner", SHALLOW_SYSTEM, user)
            if "NO_SHORT_PROOF" in raw:
                return False
            body = extract_proof_body(raw)
            if not proof_is_clean(body):
                continue
            ok, errs = await self.verify(f"{node.statement} := {body}")
            if ok:
                node.status, node.proof = "proved", body
                return True
            node.failures.append((errs[0] if errs else "verify failed")[:300])
        return False

    # ── 通路 3/4: 分解与组装 ─────────────────────────────────────

    async def _try_decompose(self, node: ProofNode) -> bool:
        system = DECOMPOSE_SYSTEM.format(max_subgoals=self.cfg.max_subgoals)
        feedback = ""
        for attempt in range(max(1, self.cfg.decompose_attempts)):
            user = (f"Target theorem:\n\n{node.statement}\n{feedback}")
            plan = parse_decomposition(
                await self._call("reasoner", system, user),
                self.cfg.max_subgoals)
            if plan is None:
                feedback = ("\nYour previous reply was not valid JSON in "
                            "the required schema. Reply with ONLY the JSON.")
                continue

            # 子目标声明先行体检: `:= by sorry` 必须可编译,
            # 否则把 verifier 报错喂回 reasoner 重新分解。
            if self.cfg.check_subgoal_statements:
                bad = await self._check_statements(plan["subgoals"])
                if bad:
                    feedback = ("\nThese lemma statements failed to "
                                "compile standalone, fix them:\n" + bad)
                    continue

            children = [ProofNode(statement=sg["statement"],
                                  depth=node.depth + 1)
                        for sg in plan["subgoals"]]
            for c in children:
                await self._solve(c)
            node.children = children
            node.assembly = plan["assembly"]
            if any(c.status != "proved" for c in children):
                unsolved = [c.statement.splitlines()[0][:120]
                            for c in children if c.status != "proved"]
                feedback = ("\nA previous decomposition produced subgoals "
                            "we could not prove; choose easier ones:\n- "
                            + "\n- ".join(unsolved))
                node.children = []
                continue

            code = assemble(node)
            ok, errs = await self.verify(code)
            if ok:
                node.status = "proved"
                node.assembled_code = code
                return True
            node.failures.append(
                ("assembly: " + (errs[0] if errs else "verify failed"))[:300])
            feedback = ("\nThe assembly proof failed to compile with the "
                        "proved lemmas:\n"
                        + (errs[0] if errs else "")[:500])
            node.children = []
        return False

    async def _check_statements(self, subgoals: list[dict]) -> str:
        bad_lines = []
        results = await asyncio.gather(*[
            self.verify(f"{sg['statement']} := by sorry")
            for sg in subgoals])
        for sg, (ok, errs) in zip(subgoals, results):
            if not ok:
                bad_lines.append(
                    f"{sg['statement'][:160]}\n  error: "
                    f"{(errs[0] if errs else '?')[:200]}")
        return "\n".join(bad_lines)
