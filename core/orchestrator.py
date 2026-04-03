"""
core/orchestrator.py — 主调度器 (重构版)

支持两种策略：
  1. sequential: 原始顺序重试 (调试用、最简 demo)
  2. rollout:    并行采样 + 经验共享 (生产推荐、RL 数据收集)

推理和 RL 训练共用同一套 rollout 基础设施。
"""

from __future__ import annotations

import logging
from typing import Optional, Callable
from dataclasses import dataclass

from core.models import BenchmarkProblem, ProofTrace, ProofAttempt, AttemptStatus
from core.lean_checker import LeanChecker
from core.llm_policy import (
    LLMProvider, build_prompt, extract_lean_code, SYSTEM_PROMPT,
)
from core.error_analyzer import analyze_errors, summarize_error_history
from core.retriever import PremiseRetriever
from core.rollout import RolloutEngine, RolloutConfig, RolloutExperience

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """调度器配置"""

    # 策略选择
    strategy: str = "rollout"          # "sequential" | "rollout"

    # Sequential 参数 (向后兼容)
    max_attempts: int = 10
    temperature: float = 0.8

    # Rollout 参数
    samples_per_round: int = 8
    max_rounds: int = 4
    rollout_temperature: float = 0.9
    enable_lemma_bank: bool = True
    max_workers: int = 4
    collect_rl_data: bool = True

    # 共享参数
    max_error_history: int = 3
    lean_timeout: int = 120
    top_k_premises: int = 10


class Orchestrator:
    """
    证明智能体的主调度器。

    根据 strategy 配置分发到不同的执行策略。
    """

    def __init__(
        self,
        lean_checker: LeanChecker,
        llm_provider: LLMProvider,
        retriever: PremiseRetriever,
        config: OrchestratorConfig = OrchestratorConfig(),
        on_attempt: Optional[Callable[[ProofAttempt], None]] = None,
    ):
        self.lean = lean_checker
        self.llm = llm_provider
        self.retriever = retriever
        self.config = config
        self.on_attempt = on_attempt

    def prove(self, problem: BenchmarkProblem) -> ProofTrace:
        """执行证明，返回 ProofTrace"""
        if self.config.strategy == "rollout":
            trace, _ = self.prove_with_experience(problem)
            return trace
        else:
            return self._prove_sequential(problem)

    def prove_with_experience(
        self, problem: BenchmarkProblem
    ) -> tuple[ProofTrace, RolloutExperience]:
        """
        执行 rollout 证明，同时返回 RL 经验数据。

        Returns:
            (ProofTrace, RolloutExperience)
        """
        engine = RolloutEngine(
            lean_checker=self.lean,
            llm_provider=self.llm,
            retriever=self.retriever,
            config=RolloutConfig(
                samples_per_round=self.config.samples_per_round,
                max_rounds=self.config.max_rounds,
                temperature=self.config.rollout_temperature,
                lean_timeout=self.config.lean_timeout,
                enable_lemma_bank=self.config.enable_lemma_bank,
                top_k_premises=self.config.top_k_premises,
                max_workers=self.config.max_workers,
                collect_rl_data=self.config.collect_rl_data,
            ),
            on_attempt=self.on_attempt,
        )
        return engine.prove(problem)

    # ── Sequential 策略 (向后兼容) ────────────────────────────

    def _prove_sequential(self, problem: BenchmarkProblem) -> ProofTrace:
        """原始的顺序重试逻辑"""
        trace = ProofTrace(
            problem_id=problem.problem_id,
            problem_name=problem.name,
            theorem_statement=problem.theorem_statement,
            natural_language=problem.natural_language,
            config_snapshot={
                "strategy": "sequential",
                "max_attempts": self.config.max_attempts,
                "temperature": self.config.temperature,
                "llm_model": self.llm.model_name,
            },
        )

        premises = self.retriever.retrieve(
            problem.theorem_statement,
            top_k=self.config.top_k_premises,
        )
        error_history: list[tuple[str, list]] = []

        for attempt_num in range(1, self.config.max_attempts + 1):
            attempt = ProofAttempt(attempt_number=attempt_num)
            attempt.retrieved_premises = premises

            # 构建 prompt
            error_analysis = ""
            history_text = ""
            if error_history:
                error_analysis = analyze_errors(error_history[-1][1])
                history_text = summarize_error_history(
                    error_history, max_history=self.config.max_error_history
                )

            user_prompt = build_prompt(
                theorem_statement=problem.theorem_statement,
                error_analysis=error_analysis,
                error_history=history_text,
                premises=premises,
            )
            attempt.prompt_summary = f"seq_attempt_{attempt_num}"
            attempt.full_prompt = user_prompt

            # LLM 生成
            try:
                llm_response = self.llm.generate(
                    system=SYSTEM_PROMPT,
                    user=user_prompt,
                    temperature=self.config.temperature,
                )
                proof_code = extract_lean_code(llm_response.content)
                attempt.generated_proof = proof_code
                attempt.llm_model = llm_response.model
                attempt.llm_tokens_in = llm_response.tokens_in
                attempt.llm_tokens_out = llm_response.tokens_out
                attempt.llm_latency_ms = llm_response.latency_ms
            except Exception as e:
                attempt.lean_result = AttemptStatus.LLM_ERROR
                attempt.lean_stderr = str(e)
                trace.add_attempt(attempt)
                if self.on_attempt:
                    self.on_attempt(attempt)
                continue

            if not proof_code.strip():
                attempt.lean_result = AttemptStatus.LLM_ERROR
                attempt.lean_stderr = "Empty proof"
                trace.add_attempt(attempt)
                if self.on_attempt:
                    self.on_attempt(attempt)
                continue

            # Lean 检查
            try:
                status, errors, stdout, stderr, check_ms = self.lean.check(
                    theorem_statement=problem.theorem_statement,
                    proof=proof_code,
                )
                attempt.lean_result = status
                attempt.lean_errors = errors
                attempt.lean_stdout = stdout
                attempt.lean_stderr = stderr
                attempt.lean_check_ms = check_ms
            except Exception as e:
                attempt.lean_result = AttemptStatus.LEAN_ERROR
                attempt.lean_stderr = str(e)

            trace.add_attempt(attempt)
            if self.on_attempt:
                self.on_attempt(attempt)

            if attempt.lean_result == AttemptStatus.SUCCESS:
                break
            else:
                error_history.append((proof_code, attempt.lean_errors))

        return trace
