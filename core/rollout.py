"""
core/rollout.py — 并行 Rollout 策略

核心思想：
  不是 sequential retry，而是每轮并行采样 N 个 proof，
  全部送 Lean 检查，然后从所有结果中提取"经验"（已证 lemma、
  接近成功的 tactic 方向、错误分类统计），注入下一轮采样。

同一套基础设施服务两个目的：
  1. 推理 (inference)：用 rollout 找到正确 proof
  2. RL 数据收集：每条 (prompt, proof, reward) 都是训练样本

关键区别 vs old Orchestrator:
  - 并行采样 (width N) 替代顺序重试
  - LemmaBank 跨 rollout 共享已证引理
  - 固定高温度 + 宽采样替代温度递增
  - 自适应预算分配：容易的题少花，难题追加
  - 每次 rollout 产出结构化经验数据
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.models import (
    BenchmarkProblem, ProofTrace, ProofAttempt,
    AttemptStatus, LeanError, ErrorCategory,
)
from core.lean_checker import LeanChecker
from core.llm_policy import (
    LLMProvider, build_prompt, extract_lean_code, SYSTEM_PROMPT,
)
from core.error_analyzer import analyze_errors, summarize_error_history
from core.retriever import PremiseRetriever
from core.lemma_bank import LemmaBank

logger = logging.getLogger(__name__)


# ── Rollout 配置 ──────────────────────────────────────────────

@dataclass
class RolloutConfig:
    """Rollout 策略配置"""

    # 采样宽度与深度
    samples_per_round: int = 8        # 每轮并行采样数 (width)
    max_rounds: int = 4               # 最大 rollout 轮数 (depth)
    # → 总预算上限 = samples_per_round × max_rounds

    # LLM 参数
    temperature: float = 0.9          # 固定高温，靠 width 获取多样性
    # 不再做 temperature escalation

    # Lean 参数
    lean_timeout: int = 120

    # 经验共享
    enable_lemma_bank: bool = True    # 是否启用跨 rollout 引理积累
    verify_lemmas: bool = False       # 是否单独验证提取的引理 (耗时但更可靠)
    max_banked_lemmas: int = 10       # prompt 中最多注入多少已证引理

    # 前提检索
    top_k_premises: int = 10

    # 自适应预算
    early_stop_on_success: bool = True   # 一旦有 proof 通过就停止
    min_rounds: int = 1                  # 至少跑几轮

    # 并行执行
    max_workers: int = 4              # LLM 调用并行线程数

    # RL 数据收集
    collect_rl_data: bool = True      # 是否收集 RL 训练数据


# ── 单次采样结果 ──────────────────────────────────────────────

@dataclass
class SampleResult:
    """一次采样 (LLM生成 + Lean检查) 的结果"""
    rollout_id: int
    round_num: int
    sample_idx: int               # 在本轮中的编号
    attempt: ProofAttempt         # 完整的 attempt 记录
    proof_code: str = ""
    success: bool = False


# ── RL 经验记录 ───────────────────────────────────────────────

@dataclass
class RolloutExperience:
    """
    一道题的全部 rollout 经验，可用于 RL 训练。

    包含：
      - 所有 (prompt, proof, reward) 三元组
      - 已证 lemma 集合
      - 预算消耗统计
    """
    problem_id: str
    theorem_statement: str

    # 所有采样的 (prompt, generated_proof, reward) 数据
    trajectories: list[dict] = field(default_factory=list)
    # 格式: {"prompt": str, "proof": str, "reward": float,
    #         "lean_errors": list, "round": int, "sample_idx": int}

    # 已证引理
    banked_lemmas: list[dict] = field(default_factory=list)

    # 统计
    total_samples: int = 0
    total_rounds: int = 0
    total_tokens: int = 0
    solved: bool = False


# ── Rollout Engine ────────────────────────────────────────────

class RolloutEngine:
    """
    并行 Rollout 引擎。

    每轮 (round):
      1. 构建 prompt (含已证引理 + 错误摘要)
      2. 并行采样 N 个 proof
      3. 并行送 Lean 检查
      4. 提取已证 lemma → 注入 LemmaBank
      5. 汇总错误分布 → 构建下轮更精准的 prompt
      6. 如果有 proof 通过 → 成功退出

    直到成功或预算耗尽。
    """

    def __init__(
        self,
        lean_checker: LeanChecker,
        llm_provider: LLMProvider,
        retriever: PremiseRetriever,
        config: RolloutConfig = RolloutConfig(),
        on_attempt: Optional[Callable[[ProofAttempt], None]] = None,
        on_round: Optional[Callable[[int, list[SampleResult]], None]] = None,
    ):
        self.lean = lean_checker
        self.llm = llm_provider
        self.retriever = retriever
        self.config = config
        self.on_attempt = on_attempt  # 每个 sample 完成后回调
        self.on_round = on_round      # 每轮完成后回调

    def prove(self, problem: BenchmarkProblem) -> tuple[ProofTrace, RolloutExperience]:
        """
        对一道题执行 rollout 证明。

        Returns:
            (ProofTrace, RolloutExperience)
            前者用于评测和展示，后者用于 RL 训练。
        """
        trace = ProofTrace(
            problem_id=problem.problem_id,
            problem_name=problem.name,
            theorem_statement=problem.theorem_statement,
            natural_language=problem.natural_language,
            config_snapshot={
                "strategy": "parallel_rollout",
                "samples_per_round": self.config.samples_per_round,
                "max_rounds": self.config.max_rounds,
                "temperature": self.config.temperature,
                "llm_model": self.llm.model_name,
                "enable_lemma_bank": self.config.enable_lemma_bank,
            },
        )

        experience = RolloutExperience(
            problem_id=problem.problem_id,
            theorem_statement=problem.theorem_statement,
        )

        # 初始化
        lemma_bank = LemmaBank(self.lean if self.config.verify_lemmas else None)
        premises = self.retriever.retrieve(
            problem.theorem_statement,
            top_k=self.config.top_k_premises,
        )

        # 跨轮积累的错误经验
        round_summaries: list[str] = []  # 每轮的错误摘要
        best_attempts: list[tuple[str, list]] = []  # 最接近成功的尝试

        logger.info(f"Rollout start: {problem.name} "
                     f"(width={self.config.samples_per_round}, "
                     f"max_depth={self.config.max_rounds})")

        attempt_counter = 0

        for round_num in range(1, self.config.max_rounds + 1):
            logger.info(f"  Round {round_num}/{self.config.max_rounds} "
                         f"({self.config.samples_per_round} samples)")

            # ── 构建本轮 prompt ──────────────────────────────
            prompt = self._build_round_prompt(
                theorem_statement=problem.theorem_statement,
                premises=premises,
                lemma_bank=lemma_bank,
                round_summaries=round_summaries,
                best_attempts=best_attempts,
                round_num=round_num,
            )

            # ── 并行采样 + Lean 检查 ─────────────────────────
            results = self._run_parallel_samples(
                prompt=prompt,
                theorem_statement=problem.theorem_statement,
                round_num=round_num,
                start_attempt_num=attempt_counter + 1,
            )

            # ── 处理结果 ─────────────────────────────────────
            round_errors = []
            found_success = False

            for res in results:
                attempt_counter += 1
                res.attempt.attempt_number = attempt_counter

                trace.add_attempt(res.attempt)
                if self.on_attempt:
                    self.on_attempt(res.attempt)

                # 收集 RL 数据
                if self.config.collect_rl_data:
                    experience.trajectories.append({
                        "prompt": prompt,
                        "proof": res.proof_code,
                        "reward": 1.0 if res.success else 0.0,
                        "lean_errors": [
                            {"category": e.category.value, "message": e.message}
                            for e in res.attempt.lean_errors
                        ],
                        "round": round_num,
                        "sample_idx": res.sample_idx,
                    })

                if res.success:
                    found_success = True
                    logger.info(f"  ✓ Sample {res.sample_idx} succeeded!")
                else:
                    round_errors.append((res.proof_code, res.attempt.lean_errors))

                    # 提取 lemma (即使 proof 整体失败)
                    if self.config.enable_lemma_bank and res.proof_code:
                        new_lemmas = lemma_bank.extract_and_verify(
                            proof_code=res.proof_code,
                            theorem_statement=problem.theorem_statement,
                            attempt_num=attempt_counter,
                            rollout_id=res.sample_idx,
                        )
                        if new_lemmas:
                            logger.info(f"    Banked {len(new_lemmas)} lemma(s) "
                                         f"from sample {res.sample_idx}")

            # 回调
            if self.on_round:
                self.on_round(round_num, results)

            # ── 成功则退出 ────────────────────────────────────
            if found_success and self.config.early_stop_on_success:
                break

            # ── 构建下轮的经验摘要 ────────────────────────────
            round_summary = self._summarize_round(round_num, results, round_errors)
            round_summaries.append(round_summary)

            # 保留最"接近成功"的尝试 (错误最少的)
            if round_errors:
                sorted_errors = sorted(round_errors, key=lambda x: len(x[1]))
                best_attempts.append(sorted_errors[0])
                # 只保留最近 3 轮的最佳尝试
                best_attempts = best_attempts[-3:]

        # ── 汇总 ──────────────────────────────────────────────
        experience.total_samples = attempt_counter
        experience.total_rounds = min(round_num, self.config.max_rounds)
        experience.total_tokens = trace.total_tokens
        experience.solved = trace.solved
        experience.banked_lemmas = lemma_bank.get_rl_experience()

        status = "SOLVED" if trace.solved else "FAILED"
        logger.info(
            f"  [{status}] {problem.name} — "
            f"{experience.total_rounds} rounds × "
            f"{self.config.samples_per_round} samples = "
            f"{attempt_counter} total, "
            f"{trace.total_tokens} tokens, "
            f"{lemma_bank.count} lemmas banked"
        )

        return trace, experience

    # ── 内部方法 ──────────────────────────────────────────────

    def _build_round_prompt(
        self,
        theorem_statement: str,
        premises: list[str],
        lemma_bank: LemmaBank,
        round_summaries: list[str],
        best_attempts: list[tuple[str, list]],
        round_num: int,
    ) -> str:
        """构建一轮 rollout 的 prompt，注入已证引理和经验摘要"""

        # 已证引理上下文
        lemma_context = ""
        if self.config.enable_lemma_bank:
            lemma_context = lemma_bank.to_prompt_context(
                max_lemmas=self.config.max_banked_lemmas
            )

        # 错误经验摘要
        error_analysis = ""
        error_history = ""
        if round_summaries:
            error_analysis = round_summaries[-1]  # 最近一轮的摘要
        if best_attempts:
            error_history = summarize_error_history(
                best_attempts, max_history=2
            )

        # 拼装
        prompt = build_prompt(
            theorem_statement=theorem_statement,
            error_analysis=error_analysis,
            error_history=error_history,
            premises=premises,
        )

        # 附加已证引理
        if lemma_context:
            prompt = prompt + "\n\n" + lemma_context

        return prompt

    def _run_parallel_samples(
        self,
        prompt: str,
        theorem_statement: str,
        round_num: int,
        start_attempt_num: int,
    ) -> list[SampleResult]:
        """并行生成 N 个 proof 并分别送 Lean 检查"""

        results: list[SampleResult] = []

        def _single_sample(sample_idx: int) -> SampleResult:
            """单个采样的完整流程"""
            attempt = ProofAttempt(
                attempt_number=start_attempt_num + sample_idx,
            )
            result = SampleResult(
                rollout_id=sample_idx,
                round_num=round_num,
                sample_idx=sample_idx,
                attempt=attempt,
            )

            # ── LLM 生成 ────────────────────────────────
            try:
                llm_response = self.llm.generate(
                    system=SYSTEM_PROMPT,
                    user=prompt,
                    temperature=self.config.temperature,
                )
                proof_code = extract_lean_code(llm_response.content)
                result.proof_code = proof_code

                attempt.generated_proof = proof_code
                attempt.llm_model = llm_response.model
                attempt.llm_tokens_in = llm_response.tokens_in
                attempt.llm_tokens_out = llm_response.tokens_out
                attempt.llm_latency_ms = llm_response.latency_ms
                attempt.prompt_summary = f"round{round_num}_sample{sample_idx}"

            except Exception as e:
                logger.error(f"  LLM error (sample {sample_idx}): {e}")
                attempt.lean_result = AttemptStatus.LLM_ERROR
                attempt.lean_stderr = str(e)
                return result

            # ── 前置检查 ──────────────────────────────────
            if not proof_code.strip():
                attempt.lean_result = AttemptStatus.LLM_ERROR
                attempt.lean_stderr = "Empty proof"
                return result

            # sorry 检测：不再直接跳过，而是仍然送 Lean
            # 因为 sorry 的 proof 里可能有可复用的 lemma

            # ── Lean 编译 ─────────────────────────────────
            try:
                status, errors, stdout, stderr, check_ms = self.lean.check(
                    theorem_statement=theorem_statement,
                    proof=proof_code,
                )
                attempt.lean_result = status
                attempt.lean_errors = errors
                attempt.lean_stdout = stdout
                attempt.lean_stderr = stderr
                attempt.lean_check_ms = check_ms

                if status == AttemptStatus.SUCCESS:
                    result.success = True

            except Exception as e:
                logger.error(f"  Lean error (sample {sample_idx}): {e}")
                attempt.lean_result = AttemptStatus.LEAN_ERROR
                attempt.lean_stderr = str(e)

            return result

        # ── 并行执行 ─────────────────────────────────────
        workers = min(self.config.max_workers, self.config.samples_per_round)

        if workers <= 1:
            # 顺序执行 (调试友好)
            for i in range(self.config.samples_per_round):
                results.append(_single_sample(i))
        else:
            # 线程并行
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_single_sample, i): i
                    for i in range(self.config.samples_per_round)
                }
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"  Sample future error: {e}")

            # 按 sample_idx 排序以保持确定性
            results.sort(key=lambda r: r.sample_idx)

        return results

    def _summarize_round(
        self,
        round_num: int,
        results: list[SampleResult],
        errors: list[tuple[str, list]],
    ) -> str:
        """
        汇总一轮的错误分布，生成给下一轮 prompt 的经验摘要。

        不是简单地罗列每个错误，而是做统计归纳：
        "8 个采样中，5 个死于 tactic_failed，2 个 type_mismatch，
         1 个 unknown_identifier"
        → 这引导 LLM 在下一轮避开高频失败模式。
        """
        if not errors:
            return ""

        # 统计错误类别分布
        category_counts: dict[str, int] = {}
        sample_error_msgs: list[str] = []

        for proof_code, error_list in errors:
            for err in error_list:
                cat = err.category.value if hasattr(err, 'category') else 'other'
                category_counts[cat] = category_counts.get(cat, 0) + 1

        n_total = len(results)
        n_success = sum(1 for r in results if r.success)
        n_fail = n_total - n_success

        parts = [
            f"Round {round_num} results: {n_success}/{n_total} samples succeeded.",
            f"Error distribution across {n_fail} failed samples:",
        ]
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            parts.append(f"  - {cat}: {count} occurrences")

        # 附加最常见错误的具体信息
        if errors:
            most_common_errors = errors[0][1][:2]  # 第一个失败样本的前 2 个错误
            if most_common_errors:
                parts.append("\nMost common error pattern:")
                for err in most_common_errors:
                    msg = err.message if hasattr(err, 'message') else str(err)
                    parts.append(f"  {msg[:120]}")

        parts.append(
            "\nFor the next round: try fundamentally different proof strategies. "
            "Avoid the dominant failure patterns listed above."
        )

        return "\n".join(parts)
