"""prover/pipeline/proof_pipeline.py — 证明管线: per-round 逻辑

从 Orchestrator.prove() 中提取的单轮证明逻辑。
每个阶段独立可测, 通过 ProofPipeline 编排。

Pipeline stages:
  1. pre_round:  检查升级/放弃条件, 触发反思, 注入上下文
  2. generate:   调用异构引擎生成候选
  3. verify:     验证候选并更新置信度
  4. post_round: 钩子驱动的升级, 周期性反思, 高级策略 (分解/猜想)

Usage::

    pipeline = ProofPipeline(components)
    trace = pipeline.run(problem)

    # 或手动控制每一轮:
    ctx = pipeline.init(problem)
    while not ctx.solved and not ctx.budget_exhausted:
        pipeline.run_round(ctx)
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

logger = logging.getLogger(__name__)


@dataclass
class RoundContext:
    """单轮证明的上下文 (在 pipeline stages 之间传递)"""
    problem: BenchmarkProblem
    memory: WorkingMemory
    trace: ProofTrace
    classification: dict = field(default_factory=dict)
    strategy_name: str = "light"

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
    """证明管线: 编排单轮证明的各阶段

    将 Orchestrator.prove() 的 ~200 行主循环拆分为
    独立可测试的阶段方法。
    """

    def __init__(self, components: 'EngineComponents',
                 config: dict = None,
                 on_attempt: Callable = None):
        self.comp = components
        self.config = config or {}
        self.on_attempt = on_attempt

    def init(self, problem: BenchmarkProblem) -> RoundContext:
        """初始化证明上下文"""
        memory = WorkingMemory(
            problem_id=problem.problem_id,
            theorem_statement=problem.theorem_statement)

        strategy_name = self.comp.meta_controller.select_initial_strategy(
            problem.difficulty)
        memory.current_strategy = strategy_name

        trace = ProofTrace(
            problem_id=problem.problem_id,
            problem_name=problem.name,
            theorem_statement=problem.theorem_statement,
            natural_language=problem.natural_language,
            config_snapshot={
                "strategy": strategy_name,
                "max_samples": self.comp.budget.max_samples,
                "plugins": self.comp.plugins.list_plugins(),
                "hooks": self.comp.hooks.list_hooks(),
            })

        # ON_PROBLEM_START 钩子
        classification = {}
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
            strategy_name=strategy_name)

    def pre_round(self, ctx: RoundContext):
        """阶段 1: 检查升级条件, 触发反思, 注入上下文

        使用 PolicyEngine (模式6) 进行策略决策,
        保留 MetaController 作为 fallback。
        """
        # ── 模式 6: 优先使用 PolicyEngine ──
        try:
            from engine.lane.policy import PolicyEngine, PolicyAction
            from engine.lane.task_state import (
                TaskContext, ProofTaskStateMachine, TaskStatus,
            )
            # 构建临时状态机用于策略评估
            task_ctx = TaskContext(
                theorem_name=ctx.problem.name,
                formal_statement=ctx.problem.theorem_statement,
                rounds_completed=ctx.round_number,
                total_samples=ctx.memory.total_samples,
                banked_lemmas=[l.name for l in ctx.memory.banked_lemmas]
                    if hasattr(ctx.memory, 'banked_lemmas') and ctx.memory.banked_lemmas else [],
            )
            task_ctx.__dict__["_max_samples"] = (
                self.comp.budget.max_samples
                if hasattr(self.comp, 'budget') else 128)
            task_ctx.__dict__["_current_strategy"] = ctx.strategy_name

            sm = ProofTaskStateMachine(
                task_id=f"policy_{ctx.problem.problem_id}", context=task_ctx)
            sm.transition_to(TaskStatus.GENERATING)

            policy = PolicyEngine.default()
            decision = policy.evaluate(sm)

            if decision.action == PolicyAction.ESCALATE_STRATEGY:
                new_strategy = decision.metadata.get("to", "medium")
                ctx.strategy_name = new_strategy
                ctx.memory.current_strategy = new_strategy
                ctx.trace.strategy_path.append(new_strategy)
                reflection = self._run_reflection(ctx)
                self._fire_strategy_switch(ctx, reflection)
                return
            elif decision.action == PolicyAction.INJECT_REFLECTION:
                self._run_reflection(ctx)
                return
        except Exception:
            pass  # PolicyEngine 不可用时 fallback 到 MetaController

        # ── Fallback: 原 MetaController 逻辑 ──
        escalation = self.comp.meta_controller.should_escalate(ctx.memory)
        if escalation:
            ctx.strategy_name = StrategySwitcher.switch(
                ctx.strategy_name, escalation)
            ctx.memory.current_strategy = ctx.strategy_name
            ctx.trace.strategy_path.append(ctx.strategy_name)

            reflection = self._run_reflection(ctx)
            self._fire_strategy_switch(ctx, reflection)

    def generate(self, ctx: RoundContext):
        """阶段 2: 生成证明候选"""
        strategy_config = StrategySwitcher.get_config(ctx.strategy_name)

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
        """阶段 3: 验证候选并更新状态"""
        for r in ctx.candidates:
            attempt = self._result_to_attempt(r, ctx.memory)
            ctx.trace.add_attempt(attempt)
            if self.on_attempt:
                self.on_attempt(attempt)

            if r.proof_code.strip() and r.confidence > 0.3:
                if self._verify_proof(ctx, r.proof_code, agent_result=r):
                    ctx.memory.solved = True
                    break

    def post_round(self, ctx: RoundContext):
        """阶段 4: 钩子驱动升级, 周期性反思, 高级策略"""
        ctx.memory.rounds_completed += 1
        ctx.round_number += 1

        if ctx.solved:
            return

        # ON_ROUND_END 钩子
        round_end = self.comp.hooks.fire(
            HookEvent.ON_ROUND_END,
            HookContext(
                event=HookEvent.ON_ROUND_END,
                theorem_statement=ctx.problem.theorem_statement,
                dominant_error=ctx.memory.get_dominant_error(),
                attempt_count=len(ctx.memory.attempt_history),
                metadata={"dominant_error_count": self._count_dominant(ctx.memory)},
            ))

        # 钩子驱动的策略升级
        if round_end.action == HookAction.ESCALATE:
            self._hook_driven_escalation(ctx, round_end)

        # 周期性反思
        interval = self.config.get("reflection_interval", 3)
        if (ctx.memory.rounds_completed >= interval
                and ctx.memory.rounds_completed % interval == 0):
            reflection = self._run_reflection(ctx)
            if reflection:
                ctx.classification.setdefault("domain_hints", {}).update({
                    "periodic_reflection": (
                        f"## Self-reflection after {ctx.memory.rounds_completed} rounds\n"
                        f"{reflection[:800]}\n\n"
                        f"Use this analysis to fundamentally change your approach."
                    ),
                })

        # 验证反馈注入
        last_fb = getattr(ctx.memory, 'last_feedback_text', '')
        if last_fb:
            ctx.classification.setdefault("domain_hints", {}).update({
                "last_verification_feedback": last_fb,
            })

        # 高级策略 (分解/猜想)
        strategy_config = StrategySwitcher.get_config(ctx.strategy_name)
        if (strategy_config.use_decompose
                and ctx.memory.rounds_completed >= 2):
            self._try_decompose(ctx)
        if (strategy_config.use_conjecture
                and ctx.memory.rounds_completed >= 3):
            self._try_conjecture(ctx)

    def finalize(self, ctx: RoundContext) -> ProofTrace:
        """收尾: 记录耗时, 触发 ON_PROBLEM_END"""
        ctx.trace.total_duration_ms = int(
            (time.time() - ctx.trace.start_time) * 1000
        ) if hasattr(ctx.trace, 'start_time') else 0

        self.comp.hooks.fire(
            HookEvent.ON_PROBLEM_END,
            HookContext(
                event=HookEvent.ON_PROBLEM_END,
                theorem_statement=ctx.problem.theorem_statement,
                metadata={"solved": ctx.solved}))

        return ctx.trace

    def run(self, problem: BenchmarkProblem) -> ProofTrace:
        """完整的证明流程 (等价于 Orchestrator.prove)"""
        from engine.observability import metrics

        start_time = time.time()
        ctx = self.init(problem)

        metrics.increment("proof_attempts", problem_id=problem.problem_id)

        while not ctx.solved and not self.comp.budget.is_exhausted():
            if self.comp.confidence.should_abstain(ctx.memory):
                break

            with metrics.timer("pipeline_pre_round"):
                self.pre_round(ctx)
            with metrics.timer("pipeline_generate"):
                self.generate(ctx)
            with metrics.timer("pipeline_verify"):
                self.verify(ctx)
            with metrics.timer("pipeline_post_round"):
                self.post_round(ctx)

            metrics.increment("proof_rounds",
                              strategy=ctx.strategy_name)

            if ctx.solved:
                break

        total_ms = int((time.time() - start_time) * 1000)
        ctx.trace.total_duration_ms = total_ms

        metrics.record_time("proof_total", total_ms,
                            solved=str(ctx.solved))
        if ctx.solved:
            metrics.increment("proof_solved")

        self.comp.hooks.fire(
            HookEvent.ON_PROBLEM_END,
            HookContext(
                event=HookEvent.ON_PROBLEM_END,
                theorem_statement=problem.theorem_statement,
                metadata={"solved": ctx.solved}))

        return ctx.trace

    # ── 内部方法 ──

    def _run_reflection(self, ctx: RoundContext) -> str:
        try:
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
            f"Hook-driven escalation: {old} → {ctx.strategy_name} "
            f"(reason: {round_end.message[:100]})")

        reflection = self._run_reflection(ctx)
        self._fire_strategy_switch(ctx, reflection)

        if round_end.inject_context:
            ctx.classification.setdefault("domain_hints", {}).update(
                round_end.inject_context)

    def _verify_proof(self, ctx, proof, agent_result=None) -> bool:
        pre = self.comp.hooks.fire(
            HookEvent.PRE_VERIFICATION,
            HookContext(
                event=HookEvent.PRE_VERIFICATION,
                theorem_statement=ctx.problem.theorem_statement,
                proof=proof))
        if pre.action == HookAction.SKIP:
            if agent_result:
                from prover.pipeline._agent_deps import ConfidenceEstimator
                agent_result.confidence = ConfidenceEstimator.refine_confidence(
                    agent_result, l0_passed=False)
            return False

        try:
            if self.comp.scheduler:
                result = self.comp.scheduler.verify_complete(
                    theorem=ctx.problem.theorem_statement,
                    proof=proof, direction="pipeline")

                if result.feedback:
                    ctx.memory.last_feedback = result.feedback
                    ctx.memory.last_feedback_text = result.feedback.to_prompt(1500)

                if agent_result:
                    from prover.pipeline._agent_deps import ConfidenceEstimator
                    agent_result.confidence = ConfidenceEstimator.refine_confidence(
                        agent_result,
                        feedback=result.feedback,
                        l0_passed=result.l0_passed,
                        l1_passed=(result.level_reached in ("L1", "L2") and result.success),
                        l2_passed=result.l2_verified)

                return result.success
            else:
                from prover.verifier.lean_checker import LeanChecker
                checker = LeanChecker(self.comp.lean_pool)
                status, errors, stderr, ms = checker.check(
                    ctx.problem.theorem_statement, proof)
                return status == AttemptStatus.SUCCESS
        except Exception as e:
            logger.warning(f"Verification error: {e}")
            return False

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
