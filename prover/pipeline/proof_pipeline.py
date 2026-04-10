"""prover/pipeline/proof_pipeline.py — 证明管线 (v3 — Lane + Compression + Checkpoint)

Phase 1: 状态机骨架 + 恢复 + 策略引擎
Phase 2: 摘要压缩 — 错误反馈和广播消息在注入 prompt 前压缩
Phase 3: 会话持久化 — 每轮结束后 checkpoint, 支持断点续证

Usage::

    pipeline = ProofPipeline(components)
    trace = pipeline.run(problem)                  # fresh start
    trace = pipeline.run(problem, resume=True)     # resume from checkpoint
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from prover.models import BenchmarkProblem, ProofTrace, ProofAttempt, AttemptStatus
from prover.pipeline._agent_deps import StrategySwitcher
from common.working_memory import WorkingMemory
from common.hook_types import HookEvent, HookContext, HookAction

# Lane system — mandatory imports (no try/except)
from engine.lane.task_state import (
    TaskContext, ProofTaskStateMachine, TaskStatus, ProofFailureClass,
)
from engine.lane.event_bus import ProofEventBus, wire_state_machine_to_bus
from engine.lane.policy import PolicyEngine, PolicyAction, PolicyDecision
from engine.lane.recovery import RecoveryRegistry, RecoveryAction
from engine.lane.dashboard import ProofDashboard
from engine.lane.error_classifier import classify_lean_error, classify_verification_result

# Phase 2: Summary compression
from engine.lane.summary_compressor import (
    compress_lean_errors, compress_feedback, compress_broadcast, compress_for_prompt,
)

# Phase 3: Session persistence
from engine.lane.proof_session_store import (
    ProofSessionStore, ProofSessionSnapshot, build_snapshot, restore_round_context,
)

logger = logging.getLogger(__name__)


@dataclass
class RoundContext:
    """单轮证明的上下文 (在 pipeline stages 之间传递)"""
    problem: BenchmarkProblem
    memory: WorkingMemory
    trace: ProofTrace
    classification: dict = field(default_factory=dict)
    strategy_name: str = "light"

    # Lane integration
    sm: ProofTaskStateMachine = None  # type: ignore[assignment]

    # 当前轮的中间状态
    candidates: list = field(default_factory=list)
    round_number: int = 0

    @property
    def solved(self) -> bool:
        return self.memory.solved

    @property
    def budget_exhausted(self) -> bool:
        return False  # 由 Pipeline 通过 budget 检查


class ProofPipeline:
    """证明管线 v2: 状态机驱动, Lane 系统深度集成

    v1 → v2 关键改变:
    - ProofTaskStateMachine 是必须的, 不是可选装饰
    - PolicyEngine 取代 MetaController 作为主决策引擎
    - RecoveryRegistry 在验证失败时自动尝试恢复
    - 所有状态转换通过 EventBus 广播
    - Dashboard 实时跟踪
    """

    def __init__(self, components: 'EngineComponents',
                 config: dict = None,
                 on_attempt: Callable = None):
        self.comp = components
        self.config = config or {}
        self.on_attempt = on_attempt

        # Lane components — use injected instances or build defaults
        self._event_bus: ProofEventBus = (
            getattr(components, 'event_bus', None) or ProofEventBus())
        self._dashboard: ProofDashboard = (
            getattr(components, 'dashboard', None) or ProofDashboard())
        self._policy: PolicyEngine = (
            getattr(components, 'policy_engine', None) or PolicyEngine.default())
        self._recovery: RecoveryRegistry = (
            getattr(components, 'recovery_registry', None) or RecoveryRegistry())

        # Phase 2: Compression budgets
        self._error_budget = config.get("error_compression_budget", 1200)
        self._feedback_budget = config.get("feedback_compression_budget", 800)
        self._broadcast_budget = config.get("broadcast_compression_budget", 1500)

        # Verification filter: skip candidates below this confidence
        self._verify_min_confidence = config.get("verify_min_confidence", 0.3)

        # Phase 3: Session persistence
        self._session_store: Optional[ProofSessionStore] = (
            getattr(components, 'session_store', None))

    # ═══════════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════════

    def init(self, problem: BenchmarkProblem) -> RoundContext:
        """初始化证明上下文 + 创建状态机"""
        memory = WorkingMemory(
            problem_id=problem.problem_id,
            theorem_statement=problem.theorem_statement)

        # 初始策略 (MetaController 仍负责初始选择, PolicyEngine 负责后续决策)
        strategy_name = (
            self.comp.meta_controller.select_initial_strategy(problem.difficulty)
            if self.comp.meta_controller else "light")
        memory.current_strategy = strategy_name

        trace = ProofTrace(
            problem_id=problem.problem_id,
            problem_name=problem.name,
            theorem_statement=problem.theorem_statement,
            natural_language=problem.natural_language,
            config_snapshot={
                "strategy": strategy_name,
                "max_samples": self.comp.budget.max_samples if self.comp.budget else 128,
                "plugins": self.comp.plugins.list_plugins() if self.comp.plugins else [],
                "hooks": self.comp.hooks.list_hooks() if self.comp.hooks else [],
                "lane_integration": "v2",
            })

        # ── 创建状态机 (Lane 核心) ──
        task_ctx = TaskContext(
            theorem_name=problem.name,
            formal_statement=problem.theorem_statement,
            domain=getattr(problem, 'domain', ""),
            difficulty=getattr(problem, 'difficulty', "unknown"),
        )
        sm = ProofTaskStateMachine(
            task_id=problem.problem_id,
            context=task_ctx,
        )
        # Wire to event bus: all state transitions auto-publish events
        wire_state_machine_to_bus(sm, self._event_bus)
        # Register on dashboard for real-time monitoring
        self._dashboard.register_task(sm)

        # ON_PROBLEM_START 钩子
        classification = {}
        if self.comp.hooks:
            start_result = self.comp.hooks.fire(
                HookEvent.ON_PROBLEM_START,
                HookContext(
                    event=HookEvent.ON_PROBLEM_START,
                    theorem_statement=problem.theorem_statement))
            if start_result.inject_context:
                classification = start_result.inject_context.get(
                    "classification", {})

        return RoundContext(
            problem=problem,
            memory=memory,
            trace=trace,
            classification=classification,
            strategy_name=strategy_name,
            sm=sm)

    def run(self, problem: BenchmarkProblem, resume: bool = False) -> ProofTrace:
        """完整的证明流程 — 状态机驱动, 支持断点续证

        Args:
            problem: 待证明的问题
            resume: True 则尝试从 checkpoint 恢复, False 则全新开始
        """
        from engine.observability import metrics

        start_time = time.time()

        # ── Phase 3: Resume from checkpoint ──
        ctx = None
        if resume and self._session_store:
            ctx = self._try_resume(problem)
            if ctx:
                logger.info(
                    f"[Lane] resumed from checkpoint: round={ctx.round_number}, "
                    f"strategy={ctx.strategy_name}, status={ctx.sm.status.value}")

        if ctx is None:
            ctx = self.init(problem)

        sm = ctx.sm

        metrics.increment("proof_attempts", problem_id=problem.problem_id)

        try:
            while not ctx.solved and not self.comp.budget.is_exhausted():
                if self.comp.confidence and self.comp.confidence.should_abstain(ctx.memory):
                    sm.give_up("confidence below threshold — abstain")
                    break

                # ── Phase 1: Policy-driven pre-round ──
                with metrics.timer("pipeline_pre_round"):
                    self.pre_round(ctx)

                # Check if policy decided to give up
                if sm.status.is_terminal:
                    break

                # ── Phase 2: Generate ──
                sm.transition_to(TaskStatus.GENERATING,
                                 detail=f"round {ctx.round_number}, strategy={ctx.strategy_name}")
                with metrics.timer("pipeline_generate"):
                    self.generate(ctx)

                # ── Phase 3: Verify ──
                sm.transition_to(TaskStatus.VERIFYING,
                                 detail=f"{len(ctx.candidates)} candidates")
                with metrics.timer("pipeline_verify"):
                    self.verify(ctx)

                # ── Phase 4: Post-round ──
                with metrics.timer("pipeline_post_round"):
                    self.post_round(ctx)

                metrics.increment("proof_rounds", strategy=ctx.strategy_name)

                if ctx.solved:
                    sm.succeed(ctx.memory.last_successful_proof
                               if hasattr(ctx.memory, 'last_successful_proof') else "")
                    break

                # Transition back to GENERATING for next round
                if not sm.status.is_terminal and sm.status != TaskStatus.GENERATING:
                    sm.transition_to(TaskStatus.GENERATING,
                                     detail=f"next round {ctx.round_number}")

            # ── Terminal state if loop exited without explicit terminal ──
            if not sm.status.is_terminal:
                if ctx.solved:
                    sm.succeed()
                elif self.comp.budget.is_exhausted():
                    sm.give_up("budget exhausted")
                else:
                    sm.give_up("max rounds or abstain")

        except Exception as e:
            sm.fail(ProofFailureClass.API_ERROR,
                    f"Pipeline exception: {e}",
                    recoverable=False)
            logger.exception(f"Pipeline error for {problem.problem_id}: {e}")

        finally:
            total_ms = int((time.time() - start_time) * 1000)
            ctx.trace.total_duration_ms = total_ms

            # Sync lane metadata to trace
            ctx.trace.config_snapshot["lane_events"] = len(sm.events)
            ctx.trace.config_snapshot["lane_final_status"] = sm.status.value
            ctx.trace.config_snapshot["recovery_attempts"] = sm.recovery_attempts

            metrics.record_time("proof_total", total_ms, solved=str(ctx.solved))
            if ctx.solved:
                metrics.increment("proof_solved")

            if self.comp.hooks:
                self.comp.hooks.fire(
                    HookEvent.ON_PROBLEM_END,
                    HookContext(
                        event=HookEvent.ON_PROBLEM_END,
                        theorem_statement=problem.theorem_statement,
                        metadata={
                            "solved": ctx.solved,
                            "lane_status": sm.status.value,
                        }))

            self._dashboard.unregister_task(problem.problem_id)

            # Phase 3: Clean up checkpoint on terminal state
            if sm.status.is_terminal:
                self._remove_checkpoint(problem.problem_id)

        return ctx.trace

    # ═══════════════════════════════════════════════════════════════════════
    # Pipeline Stages
    # ═══════════════════════════════════════════════════════════════════════

    def pre_round(self, ctx: RoundContext):
        """阶段 1: PolicyEngine 驱动的策略决策

        PolicyEngine 是唯一的决策入口。MetaController 的阈值逻辑
        已迁移到 PolicyRule 中 (BudgetEscalationRule 等)。
        """
        sm = ctx.sm

        # Sync working memory → state machine context
        sm.context.rounds_completed = ctx.round_number
        sm.context.total_samples = getattr(ctx.memory, 'total_samples', 0)
        sm.context.banked_lemmas = (
            [l.get("name", "") if isinstance(l, dict) else str(l)
             for l in ctx.memory.banked_lemmas[:20]]
            if hasattr(ctx.memory, 'banked_lemmas') and ctx.memory.banked_lemmas
            else [])
        sm.context.max_samples = (
            self.comp.budget.max_samples if self.comp.budget else 128)
        sm.context.current_strategy = ctx.strategy_name

        # ── Evaluate policy ──
        decision = self._policy.evaluate(sm)

        logger.info(
            f"[Lane] policy: {decision.action.value} "
            f"(rule={decision.rule_name}, reason={decision.reason[:80]})")

        self._apply_policy_decision(ctx, decision)

    def generate(self, ctx: RoundContext):
        """阶段 2: 生成证明候选"""
        if ctx.strategy_name == "sequential":
            from prover.pipeline.sequential_engine import SequentialEngine
            engine = SequentialEngine(
                self.comp.lean_pool, None, None,
                {**self.config, "max_attempts": 3})
            round_trace = engine.run_round(
                ctx.problem, ctx.memory, self.comp.budget)
            for a in round_trace:
                ctx.trace.add_attempt(a)
                if self.on_attempt:
                    self.on_attempt(a)
                if (hasattr(a, 'lean_result')
                        and a.lean_result == AttemptStatus.SUCCESS):
                    ctx.memory.solved = True
            ctx.candidates = []
        else:
            results = self.comp.hetero_engine.run_round(
                ctx.problem,
                classification=ctx.classification,
                attempt_history=ctx.memory.attempt_history,
                budget=self.comp.budget)
            ctx.candidates = results

    def verify(self, ctx: RoundContext):
        """阶段 3: 验证候选 — 失败时 classify → recover 闭环

        验证失败 → classify_lean_error → sm.fail(ProofFailureClass)
        → RecoveryRegistry 尝试自动恢复 → 成功则继续, 失败则记录
        """
        sm = ctx.sm

        for r in ctx.candidates:
            attempt = self._result_to_attempt(r, ctx.memory)
            ctx.trace.add_attempt(attempt)
            if self.on_attempt:
                self.on_attempt(attempt)

            if not (r.proof_code and r.proof_code.strip()):
                continue

            # Adaptive confidence threshold: lower bar as budget shrinks
            # (when budget is running out, try verifying even low-confidence candidates)
            budget = self.comp.budget
            if budget and hasattr(budget, 'remaining_fraction'):
                remaining = budget.remaining_fraction()
            elif budget and hasattr(budget, 'max_samples'):
                used = getattr(ctx.memory, 'total_samples', len(ctx.memory.attempt_history))
                remaining = max(0.0, 1.0 - used / max(1, budget.max_samples))
            else:
                remaining = 1.0
            min_conf = self._verify_min_confidence * remaining
            if r.confidence < min_conf:
                continue

            success, failure_class = self._verify_and_classify(ctx, r)

            if success:
                ctx.memory.solved = True
                if hasattr(ctx.memory, 'last_successful_proof'):
                    ctx.memory.last_successful_proof = r.proof_code
                break
            elif failure_class:
                error_msg = r.metadata.get("verification", {}).get(
                    "error", failure_class.value)
                sm.fail(failure_class, str(error_msg)[:500], recoverable=True)

                # ── Recovery attempt ──
                recovered = self._try_recovery(ctx, failure_class)
                if recovered:
                    logger.info(f"[Lane] recovered from {failure_class.value}")
                else:
                    # No recovery — transition back to VERIFYING for next candidate
                    if not sm.status.is_terminal and sm.status == TaskStatus.BLOCKED:
                        sm.transition_to(TaskStatus.VERIFYING,
                                         detail="recovery failed, next candidate")

        # Track best attempt
        if ctx.candidates:
            best = max(ctx.candidates, key=lambda r: r.confidence, default=None)
            if best and best.proof_code:
                sm.context.best_attempt_code = best.proof_code[:2000]

    def post_round(self, ctx: RoundContext):
        """阶段 4: 钩子驱动升级, 反思注入, 高级策略"""
        ctx.memory.rounds_completed += 1
        ctx.round_number += 1
        sm = ctx.sm

        if ctx.solved:
            return

        # ON_ROUND_END 钩子
        if self.comp.hooks:
            round_end = self.comp.hooks.fire(
                HookEvent.ON_ROUND_END,
                HookContext(
                    event=HookEvent.ON_ROUND_END,
                    theorem_statement=ctx.problem.theorem_statement,
                    dominant_error=ctx.memory.get_dominant_error(),
                    attempt_count=len(ctx.memory.attempt_history),
                    metadata={
                        "dominant_error_count": self._count_dominant(ctx.memory),
                        "lane_status": sm.status.value,
                    },
                ))

            if round_end.action == HookAction.ESCALATE:
                self._hook_driven_escalation(ctx, round_end)

        # 验证反馈注入 — Phase 2: 压缩后再注入
        last_fb = getattr(ctx.memory, 'last_feedback_text', '')
        if last_fb:
            compressed_fb = compress_feedback(last_fb, budget=self._feedback_budget)
            ctx.classification.setdefault("domain_hints", {}).update({
                "last_verification_feedback": compressed_fb,
            })

        # 高级策略
        strategy_config = StrategySwitcher.get_config(ctx.strategy_name)
        if strategy_config.use_decompose and ctx.memory.rounds_completed >= 2:
            self._try_decompose(ctx)
        if strategy_config.use_conjecture and ctx.memory.rounds_completed >= 3:
            self._try_conjecture(ctx)

        # ── Knowledge decay: periodically age out stale knowledge ──
        if (ctx.round_number % 5 == 0
                and hasattr(self.comp, 'knowledge_evolver')
                and self.comp.knowledge_evolver):
            self._run_knowledge_decay()

        # ── Phase 3: Checkpoint after each round ──
        self._save_checkpoint(ctx)

    # ═══════════════════════════════════════════════════════════════════════
    # Policy Decision Execution
    # ═══════════════════════════════════════════════════════════════════════

    def _apply_policy_decision(self, ctx: RoundContext, decision: PolicyDecision):
        """Execute a policy decision — the single point of strategy control."""
        action = decision.action

        if action == PolicyAction.CONTINUE:
            return

        elif action == PolicyAction.ESCALATE_STRATEGY:
            new_strategy = decision.metadata.get("to", "medium")
            old = ctx.strategy_name
            ctx.strategy_name = new_strategy
            ctx.memory.current_strategy = new_strategy
            ctx.trace.strategy_path.append(new_strategy)
            logger.info(f"[Lane] escalation: {old} → {new_strategy}")
            reflection = self._run_reflection(ctx)
            self._fire_strategy_switch(ctx, reflection)

        elif action == PolicyAction.SWITCH_ROLE:
            old = ctx.strategy_name
            ctx.strategy_name = StrategySwitcher.switch(ctx.strategy_name, "medium")
            ctx.memory.current_strategy = ctx.strategy_name
            ctx.trace.strategy_path.append(ctx.strategy_name)
            logger.info(f"[Lane] role switch: {old} → {ctx.strategy_name}")

        elif action == PolicyAction.INJECT_REFLECTION:
            reflection = self._run_reflection(ctx)
            if reflection:
                ctx.classification.setdefault("domain_hints", {}).update({
                    "periodic_reflection": (
                        f"## Self-reflection after {ctx.round_number} rounds\n"
                        f"{reflection[:800]}\n\n"
                        f"Use this analysis to fundamentally change your approach."
                    ),
                })

        elif action == PolicyAction.TRY_DECOMPOSE:
            self._try_decompose(ctx)
            ctx.sm.context.decompose_attempted = True

        elif action == PolicyAction.TRY_CONJECTURE:
            self._try_conjecture(ctx)

        elif action == PolicyAction.AUTO_RECOVER:
            fc = ctx.sm.last_failure.failure_class if ctx.sm.last_failure else None
            if fc:
                self._try_recovery(ctx, fc)

        elif action == PolicyAction.GIVE_UP:
            ctx.sm.give_up(decision.reason)

        elif action == PolicyAction.ESCALATE_TO_HUMAN:
            ctx.sm.give_up(f"escalation needed: {decision.reason}")

    # ═══════════════════════════════════════════════════════════════════════
    # Recovery
    # ═══════════════════════════════════════════════════════════════════════

    def _try_recovery(self, ctx: RoundContext,
                      failure_class: ProofFailureClass) -> bool:
        """Attempt auto-recovery. Returns True if recovery succeeded."""
        sm = ctx.sm
        recipe = self._recovery.get(failure_class)

        if not recipe or not recipe.attempts_remaining(sm.recovery_attempts):
            logger.info(
                f"[Lane] no recovery for {failure_class.value} "
                f"(attempts={sm.recovery_attempts})")
            return False

        action = recipe.action
        logger.info(f"[Lane] recovery: {action.value} for {failure_class.value}")

        try:
            if action == RecoveryAction.RESTART_REPL:
                if self.comp.lean_pool and hasattr(self.comp.lean_pool, 'restart_session'):
                    self.comp.lean_pool.restart_session()
                if sm.status == TaskStatus.BLOCKED:
                    sm.transition_to(TaskStatus.GENERATING,
                                     detail=f"recovered: {action.value}")
                return True

            elif action == RecoveryAction.RETRY_WITH_BACKOFF:
                # Non-blocking: just record the backoff intent;
                # actual delay is handled by the caller if running async
                logger.info(f"[Lane] recovery: backoff {recipe.backoff_seconds}s before retry")
                if sm.status == TaskStatus.BLOCKED:
                    sm.transition_to(TaskStatus.GENERATING,
                                     detail=f"recovered: retry after {recipe.backoff_seconds}s")
                return True

            elif action == RecoveryAction.RETRY_LARGER_TIMEOUT:
                current_timeout = self.config.get("timeout_seconds", 120)
                self.config["timeout_seconds"] = int(
                    current_timeout * recipe.timeout_multiplier)
                if sm.status == TaskStatus.BLOCKED:
                    sm.transition_to(TaskStatus.GENERATING,
                                     detail=f"recovered: timeout → {self.config['timeout_seconds']}s")
                return True

            elif action == RecoveryAction.REDUCE_CONCURRENCY:
                if self.comp.agent_pool:
                    self.comp.agent_pool.max_workers = max(
                        1, self.comp.agent_pool.max_workers // 2)
                if sm.status == TaskStatus.BLOCKED:
                    sm.transition_to(TaskStatus.GENERATING,
                                     detail="recovered: reduced concurrency")
                return True

            elif action == RecoveryAction.INJECT_NEGATIVE_KNOWLEDGE:
                if sm.last_failure:
                    raw_neg = (
                        f"AVOID: Previous attempt failed with "
                        f"{sm.last_failure.failure_class.value}: "
                        f"{sm.last_failure.message}\n"
                        f"Do NOT repeat this approach."
                    )
                    # Phase 2: compress before injecting
                    compressed_neg = compress_lean_errors(
                        raw_neg, budget=self._error_budget)
                    ctx.classification.setdefault("domain_hints", {}).update({
                        "negative_knowledge": compressed_neg,
                    })
                if sm.status == TaskStatus.BLOCKED:
                    sm.transition_to(TaskStatus.GENERATING,
                                     detail="recovered: injected negative knowledge")
                return True

            elif action == RecoveryAction.SWITCH_ROLE:
                old = ctx.strategy_name
                ctx.strategy_name = StrategySwitcher.switch(
                    ctx.strategy_name, "medium")
                ctx.memory.current_strategy = ctx.strategy_name
                if sm.status == TaskStatus.BLOCKED:
                    sm.transition_to(TaskStatus.GENERATING,
                                     detail=f"recovered: {old} → {ctx.strategy_name}")
                return True

            elif action == RecoveryAction.SWITCH_STRATEGY:
                old = ctx.strategy_name
                ctx.strategy_name = StrategySwitcher.switch(
                    ctx.strategy_name, "heavy")
                ctx.memory.current_strategy = ctx.strategy_name
                if sm.status == TaskStatus.BLOCKED:
                    sm.transition_to(TaskStatus.GENERATING,
                                     detail=f"recovered: {old} → {ctx.strategy_name}")
                return True

            elif action == RecoveryAction.SKIP_AND_CONTINUE:
                if sm.status == TaskStatus.BLOCKED:
                    sm.transition_to(TaskStatus.GENERATING,
                                     detail="recovered: skip and continue")
                return True

            elif action == RecoveryAction.NO_RECOVERY:
                return False

        except Exception as e:
            logger.warning(f"[Lane] recovery {action.value} failed: {e}")
            return False

        return False

    # ═══════════════════════════════════════════════════════════════════════
    # Verification with Classification
    # ═══════════════════════════════════════════════════════════════════════

    def _verify_and_classify(self, ctx, agent_result
                             ) -> tuple[bool, Optional[ProofFailureClass]]:
        """Verify a proof and classify the failure if it fails.

        Returns (success, failure_class). failure_class is None on success.
        Runs sorry/axiom integrity check even after Lean compilation success.
        """
        from prover.verifier.sorry_detector import detect_sorry

        if self.comp.hooks:
            pre = self.comp.hooks.fire(
                HookEvent.PRE_VERIFICATION,
                HookContext(
                    event=HookEvent.PRE_VERIFICATION,
                    theorem_statement=ctx.problem.theorem_statement,
                    proof=agent_result.proof_code))
            if pre.action == HookAction.SKIP:
                from prover.pipeline._agent_deps import ConfidenceEstimator
                agent_result.confidence = ConfidenceEstimator.refine_confidence(
                    agent_result, l0_passed=False)
                return False, ProofFailureClass.INTEGRITY_VIOLATION

        # ── Pre-flight sorry/axiom integrity check ──
        sorry_report = detect_sorry(agent_result.proof_code)
        if sorry_report.has_sorry:
            logger.info(
                f"[Lane] sorry detected pre-verification: "
                f"{sorry_report.locations[:3]}")
            return False, ProofFailureClass.SORRY_DETECTED
        if sorry_report.warnings:
            for w in sorry_report.warnings:
                logger.warning(f"[Lane] integrity warning: {w}")

        try:
            if self.comp.scheduler:
                vr = self.comp.scheduler.verify_complete(
                    theorem=ctx.problem.theorem_statement,
                    proof=agent_result.proof_code,
                    direction="pipeline")

                if vr.feedback:
                    ctx.memory.last_feedback = vr.feedback
                    ctx.memory.last_feedback_text = vr.feedback.to_prompt(1500)

                from prover.pipeline._agent_deps import ConfidenceEstimator
                agent_result.confidence = ConfidenceEstimator.refine_confidence(
                    agent_result,
                    feedback=vr.feedback,
                    l0_passed=vr.l0_passed,
                    l1_passed=(vr.level_reached in ("L1", "L2") and vr.success),
                    l2_passed=vr.l2_verified)

                if vr.success:
                    # Post-success integrity check: reject if axiom/unsafe
                    if sorry_report.warnings:
                        logger.warning(
                            f"[Lane] proof compiles but has integrity warnings: "
                            f"{sorry_report.warnings}")
                        return False, ProofFailureClass.INTEGRITY_VIOLATION
                    return True, None

                fc = classify_verification_result(vr)
                return False, fc

            else:
                from prover.verifier.lean_checker import LeanChecker
                checker = LeanChecker(self.comp.lean_pool)
                status, errors, stderr, ms = checker.check(
                    ctx.problem.theorem_statement, agent_result.proof_code)

                if status == AttemptStatus.SUCCESS:
                    if sorry_report.warnings:
                        logger.warning(
                            f"[Lane] proof compiles but has integrity warnings: "
                            f"{sorry_report.warnings}")
                        return False, ProofFailureClass.INTEGRITY_VIOLATION
                    return True, None

                error_text = "\n".join(errors) if errors else (stderr or "")
                fc = classify_lean_error(error_text)
                return False, fc

        except Exception as e:
            logger.warning(f"Verification error: {e}")
            error_str = str(e).lower()
            if "timeout" in error_str:
                return False, ProofFailureClass.TIMEOUT
            elif "connection" in error_str or "repl" in error_str:
                return False, ProofFailureClass.REPL_CRASH
            else:
                return False, ProofFailureClass.API_ERROR

    # ═══════════════════════════════════════════════════════════════════════
    # Internal Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _run_reflection(self, ctx: RoundContext) -> str:
        try:
            if not self.comp.reflector:
                return ""
            error_summary = ctx.memory.get_dominant_error()
            best_proofs = [a.get("generated_proof", "")[:200]
                           for a in ctx.memory.attempt_history[-3:]
                           if a.get("generated_proof")]
            return self.comp.reflector.reflect(
                ctx.problem.theorem_statement, error_summary, best_proofs)
        except Exception as e:
            logger.debug(f"Reflection failed: {e}")
            return ""

    def _fire_strategy_switch(self, ctx, reflection_text):
        if not self.comp.hooks:
            return
        result = self.comp.hooks.fire(
            HookEvent.ON_STRATEGY_SWITCH,
            HookContext(
                event=HookEvent.ON_STRATEGY_SWITCH,
                theorem_statement=ctx.problem.theorem_statement,
                strategy_name=ctx.strategy_name,
                metadata={"reflection_text": reflection_text}))
        if result.inject_context:
            ctx.classification.setdefault("domain_hints", {}).update(
                result.inject_context)

    def _hook_driven_escalation(self, ctx, round_end):
        if ctx.strategy_name == "light":
            next_level = "medium"
        elif ctx.strategy_name == "medium":
            next_level = "heavy"
        else:
            next_level = "heavy"

        old = ctx.strategy_name
        ctx.strategy_name = StrategySwitcher.switch(ctx.strategy_name, next_level)
        ctx.memory.current_strategy = ctx.strategy_name
        ctx.trace.strategy_path.append(ctx.strategy_name)

        logger.info(
            f"[Lane] hook escalation: {old} → {ctx.strategy_name} "
            f"(reason: {round_end.message[:100]})")

        reflection = self._run_reflection(ctx)
        self._fire_strategy_switch(ctx, reflection)

        if round_end.inject_context:
            ctx.classification.setdefault("domain_hints", {}).update(
                round_end.inject_context)

    def _result_to_attempt(self, r, memory) -> ProofAttempt:
        attempt = ProofAttempt(attempt_number=len(memory.attempt_history) + 1)
        attempt.generated_proof = r.proof_code
        attempt.llm_tokens_in = r.tokens_used // 2
        attempt.llm_tokens_out = r.tokens_used // 2
        attempt.llm_latency_ms = r.latency_ms
        memory.record_attempt({
            "generated_proof": r.proof_code, "errors": [],
            "agent": r.agent_name, "confidence": r.confidence})
        return attempt

    def _count_dominant(self, memory) -> int:
        dom = memory.get_dominant_error()
        if dom == "none":
            return 0
        return sum(1 for a in memory.attempt_history[-6:]
                   if dom in str(a.get("errors", [])))

    def _try_decompose(self, ctx):
        try:
            from prover.decompose.goal_decomposer import GoalDecomposer
            decomposer = GoalDecomposer(None)
            subgoals = decomposer.decompose(ctx.problem.theorem_statement)
            if subgoals:
                for sg in subgoals:
                    ctx.memory.goal_stack.append(sg.statement)
        except Exception as e:
            logger.warning(f"Decompose failed: {e}")

    def _try_conjecture(self, ctx):
        try:
            from prover.conjecture.conjecture_proposer import ConjectureProposer
            proposer = ConjectureProposer(None)
            existing = [l.get("statement", "")
                        for l in ctx.memory.banked_lemmas[:5]]
            conjectures = proposer.propose(
                ctx.problem.theorem_statement,
                existing_lemmas=existing, n=3, verify=False)
            if conjectures:
                for conj in conjectures:
                    ctx.memory.banked_lemmas.append({
                        "name": "conj", "statement": conj,
                        "proof": "", "verified": False})
        except Exception as e:
            logger.warning(f"Conjecture failed: {e}")

    def _run_knowledge_decay(self):
        """Run knowledge decay/GC (async evolver called from sync context)."""
        try:
            import asyncio
            evolver = self.comp.knowledge_evolver
            # If there's a running event loop, schedule as a task;
            # otherwise create a new loop for the decay tick.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(evolver.decay_tick())
            except RuntimeError:
                asyncio.run(evolver.decay_tick())
        except Exception as e:
            logger.debug(f"Knowledge decay failed: {e}")

    def finalize(self, ctx: RoundContext) -> ProofTrace:
        """Legacy-compatible finalize."""
        ctx.trace.total_duration_ms = int(
            (time.time() - ctx.trace.start_time) * 1000
        ) if hasattr(ctx.trace, 'start_time') else 0
        if self.comp.hooks:
            self.comp.hooks.fire(
                HookEvent.ON_PROBLEM_END,
                HookContext(
                    event=HookEvent.ON_PROBLEM_END,
                    theorem_statement=ctx.problem.theorem_statement,
                    metadata={"solved": ctx.solved}))
        return ctx.trace

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3: Checkpoint / Resume
    # ═══════════════════════════════════════════════════════════════════════

    def _save_checkpoint(self, ctx: RoundContext):
        """Save a checkpoint after each round."""
        if not self._session_store:
            return
        try:
            elapsed_ms = int((time.time() - ctx.trace.start_time) * 1000) \
                if hasattr(ctx.trace, 'start_time') else 0
            snapshot = build_snapshot(
                problem_id=ctx.problem.problem_id,
                problem_name=ctx.problem.name,
                theorem_statement=ctx.problem.theorem_statement,
                round_number=ctx.round_number,
                strategy_name=ctx.strategy_name,
                classification=ctx.classification,
                memory=ctx.memory,
                sm=ctx.sm,
                trace=ctx.trace,
                elapsed_ms=elapsed_ms,
                broadcast_bus=self.comp.broadcast if self.comp else None,
            )
            self._session_store.checkpoint(snapshot)
        except Exception as e:
            logger.warning(f"[Lane] checkpoint save failed: {e}")

    def _try_resume(self, problem: BenchmarkProblem) -> Optional[RoundContext]:
        """Try to resume from a checkpoint. Returns RoundContext or None."""
        if not self._session_store:
            return None
        snapshot = self._session_store.load(problem.problem_id)
        if snapshot is None:
            return None
        if snapshot.solved or snapshot.lane_status in ("succeeded", "failed", "given_up"):
            logger.info(f"[Lane] checkpoint is terminal ({snapshot.lane_status}), starting fresh")
            self._session_store.remove(problem.problem_id)
            return None

        try:
            restored = restore_round_context(snapshot, self.comp)
            memory = restored["memory"]
            trace = restored["trace"]
            sm = restored["sm"]

            # Wire SM to event bus and dashboard
            wire_state_machine_to_bus(sm, self._event_bus)
            self._dashboard.register_task(sm)

            # Re-inject saved knowledge context
            classification = restored["classification"]
            if snapshot.last_feedback_text:
                compressed = compress_feedback(
                    snapshot.last_feedback_text, budget=self._feedback_budget)
                classification.setdefault("domain_hints", {}).update({
                    "last_verification_feedback": compressed,
                })
            for neg in snapshot.negative_knowledge:
                compressed_neg = compress_lean_errors(neg, budget=self._error_budget)
                classification.setdefault("domain_hints", {}).update({
                    "negative_knowledge": compressed_neg,
                })

            return RoundContext(
                problem=problem,
                memory=memory,
                trace=trace,
                classification=classification,
                strategy_name=restored["strategy_name"],
                sm=sm,
                round_number=restored["round_number"],
            )
        except Exception as e:
            logger.warning(f"[Lane] resume failed, starting fresh: {e}")
            self._session_store.remove(problem.problem_id)
            return None

    def _remove_checkpoint(self, problem_id: str):
        """Remove checkpoint after terminal state."""
        if self._session_store:
            try:
                self._session_store.remove(problem_id)
            except Exception as _exc:
                logger.debug(f"Suppressed exception: {_exc}")
