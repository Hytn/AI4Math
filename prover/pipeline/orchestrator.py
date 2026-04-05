"""prover/pipeline/orchestrator.py — 主调度器 (v2)

集成三大新模块:
  1. HookManager  — 生命周期钩子 (替代散落的 if 判断)
  2. PluginLoader  — 策略插件 (替代硬编码的策略参数)
  3. AgentPool     — 子智能体运行时 (替代同质化并行)
  4. HeterogeneousEngine — 异构并行 (替代 RolloutEngine)

向后兼容: prove() 签名不变, 旧的 RolloutEngine 作为 fallback 保留。
"""
from __future__ import annotations
import logging
import time
from prover.models import BenchmarkProblem, ProofTrace, AttemptStatus, ProofAttempt
from prover.pipeline.rollout_engine import RolloutEngine
from prover.pipeline.sequential_engine import SequentialEngine
from prover.pipeline.heterogeneous_engine import HeterogeneousEngine
from agent.runtime.agent_pool import AgentPool
from agent.runtime.sub_agent import AgentResult
from agent.hooks.hook_manager import HookManager
from agent.hooks.hook_types import HookEvent, HookContext, HookAction
from agent.hooks.builtin_hooks import (
    DomainClassifierHook, RepetitionDetectorHook,
    NatSubSafetyHook, ReflectionCloserHook,
)
from agent.plugins.loader import PluginLoader
from agent.strategy.meta_controller import MetaController
from agent.strategy.strategy_switcher import StrategySwitcher
from agent.strategy.reflection import Reflector
from agent.strategy.budget_allocator import Budget
from agent.strategy.confidence_estimator import ConfidenceEstimator
from agent.memory.working_memory import WorkingMemory
from agent.context.context_window import ContextWindow

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, lean_env, llm_provider, retriever=None,
                 config=None, on_attempt=None):
        self.lean = lean_env
        self.llm = llm_provider
        self.retriever = retriever
        self.config = config or {}
        self.on_attempt = on_attempt

        # ── 原有模块 ──
        self.meta = MetaController(self.config)
        self.reflector = Reflector(llm_provider)
        self.confidence = ConfidenceEstimator()
        self.budget = Budget(
            max_samples=self.config.get("max_samples", 128),
            max_wall_seconds=self.config.get("max_wall_seconds", 3600),
        )

        # ── 新增: 子智能体运行时 ──
        self.pool = AgentPool(
            llm=llm_provider,
            max_workers=self.config.get("max_workers", 4),
        )

        # ── 新增: 钩子系统 ──
        self.hooks = HookManager()
        self._register_builtin_hooks()

        # ── 新增: 插件系统 ──
        plugin_dirs = self.config.get("plugin_dirs", ["plugins/strategies"])
        self.plugins = PluginLoader(plugin_dirs)
        self.plugins.discover()
        self._register_plugin_hooks()

        # ── 新增: 异构引擎 ──
        self.hetero_engine = HeterogeneousEngine(
            pool=self.pool,
            plugin_loader=self.plugins,
            hook_manager=self.hooks,
            retriever=self.retriever,
        )

    def _register_builtin_hooks(self):
        self.hooks.register(HookEvent.ON_PROBLEM_START,
                            DomainClassifierHook(), priority=10)
        self.hooks.register(HookEvent.ON_ROUND_END,
                            RepetitionDetectorHook(threshold=4), priority=20)
        self.hooks.register(HookEvent.PRE_VERIFICATION,
                            NatSubSafetyHook(), priority=30)
        self.hooks.register(HookEvent.ON_STRATEGY_SWITCH,
                            ReflectionCloserHook(), priority=40)

    def _register_plugin_hooks(self):
        for name in self.plugins.list_plugins():
            plugin = self.plugins.get(name)
            if plugin and plugin.hooks:
                self.hooks.register_from_plugin(plugin.hooks)

    def prove(self, problem: BenchmarkProblem) -> ProofTrace:
        """主证明入口 — 集成钩子 + 插件 + 异构并行"""
        start_time = time.time()

        memory = WorkingMemory(
            problem_id=problem.problem_id,
            theorem_statement=problem.theorem_statement)
        strategy_name = self.meta.select_initial_strategy(problem.difficulty)
        memory.current_strategy = strategy_name

        trace = ProofTrace(
            problem_id=problem.problem_id,
            problem_name=problem.name,
            theorem_statement=problem.theorem_statement,
            natural_language=problem.natural_language,
            config_snapshot={
                "strategy": strategy_name,
                "max_samples": self.budget.max_samples,
                "plugins": self.plugins.list_plugins(),
                "hooks": self.hooks.list_hooks(),
            })

        # ── Step 0: ON_PROBLEM_START 钩子 → 领域分类 ──
        classification = {}
        start_result = self.hooks.fire(
            HookEvent.ON_PROBLEM_START,
            HookContext(
                event=HookEvent.ON_PROBLEM_START,
                theorem_statement=problem.theorem_statement,
            ))
        if start_result.inject_context:
            classification = start_result.inject_context.get(
                "classification", {})

        # ── Main loop ──
        while not memory.solved and not self.budget.is_exhausted():
            if self.confidence.should_abstain(memory):
                break

            # 策略升级 (集成反思闭环钩子)
            escalation = self.meta.should_escalate(memory)
            if escalation:
                strategy_name = StrategySwitcher.switch(strategy_name, escalation)
                memory.current_strategy = strategy_name
                trace.strategy_path.append(strategy_name)

                # 反思 → 钩子闭环 → 注入下一轮上下文
                reflection_text = self._run_reflection(problem, memory)
                switch_result = self.hooks.fire(
                    HookEvent.ON_STRATEGY_SWITCH,
                    HookContext(
                        event=HookEvent.ON_STRATEGY_SWITCH,
                        theorem_statement=problem.theorem_statement,
                        strategy_name=strategy_name,
                        metadata={"reflection_text": reflection_text},
                    ))
                if switch_result.inject_context:
                    classification.setdefault("domain_hints", {}).update(
                        switch_result.inject_context)

            strategy_config = StrategySwitcher.get_config(strategy_name)

            if strategy_name == "sequential":
                engine = SequentialEngine(
                    self.lean, self.llm, self.retriever,
                    {**self.config, "max_attempts": 3})
                round_trace = engine.run_round(problem, memory, self.budget)
                for a in round_trace:
                    trace.add_attempt(a)
                    if self.on_attempt:
                        self.on_attempt(a)
                    if hasattr(a, 'lean_result') and a.lean_result == AttemptStatus.SUCCESS:
                        memory.solved = True
            else:
                # ── 核心改进: 异构并行 ──
                results = self.hetero_engine.run_round(
                    problem, classification=classification,
                    attempt_history=memory.attempt_history,
                    budget=self.budget)

                for r in results:
                    attempt = self._agent_result_to_attempt(r, memory)
                    trace.add_attempt(attempt)
                    if self.on_attempt:
                        self.on_attempt(attempt)
                    if r.proof_code.strip() and r.confidence > 0.3:
                        if self._verify_proof(problem, r.proof_code, memory, trace):
                            memory.solved = True
                            break

            memory.rounds_completed += 1
            if memory.solved:
                break

            # ON_ROUND_END 钩子: 检测重复错误 → 主动升级
            round_end = self.hooks.fire(
                HookEvent.ON_ROUND_END,
                HookContext(
                    event=HookEvent.ON_ROUND_END,
                    theorem_statement=problem.theorem_statement,
                    dominant_error=memory.get_dominant_error(),
                    attempt_count=len(memory.attempt_history),
                    metadata={"dominant_error_count": self._count_dominant(memory)},
                ))
            if round_end.action == HookAction.ESCALATE:
                strategy_name = StrategySwitcher.switch(strategy_name, "heavy")
                memory.current_strategy = strategy_name

            if (strategy_config.use_decompose
                    and memory.rounds_completed >= 2 and not memory.solved):
                self._try_decompose(problem, memory)
            if (strategy_config.use_conjecture
                    and not memory.solved and memory.rounds_completed >= 3):
                self._try_conjecture(problem, memory)

        trace.total_duration_ms = int((time.time() - start_time) * 1000)
        self.hooks.fire(HookEvent.ON_PROBLEM_END, HookContext(
            event=HookEvent.ON_PROBLEM_END,
            theorem_statement=problem.theorem_statement,
            metadata={"solved": memory.solved}))
        return trace

    def _run_reflection(self, problem, memory) -> str:
        try:
            error_summary = memory.get_dominant_error()
            best_proofs = [a.get("generated_proof", "")[:200]
                           for a in memory.attempt_history[-3:]
                           if a.get("generated_proof")]
            reflection = self.reflector.reflect(
                problem.theorem_statement, error_summary, best_proofs)
            return reflection  # 返回文本而非只打 log
        except Exception as e:
            logger.debug(f"Reflection failed: {e}")
            return ""

    def _verify_proof(self, problem, proof, memory, trace) -> bool:
        pre = self.hooks.fire(HookEvent.PRE_VERIFICATION,
            HookContext(event=HookEvent.PRE_VERIFICATION,
                        theorem_statement=problem.theorem_statement,
                        proof=proof))
        if pre.action == HookAction.SKIP:
            logger.info("PRE_VERIFICATION hook: SKIP (proof rejected by pre-filter)")
            return False
        if pre.action == HookAction.MODIFY and pre.inject_context:
            # Hook detected an issue (e.g., ℕ subtraction without ≤ guard).
            # Log the warning and store it in memory for downstream repair.
            for key, value in pre.inject_context.items():
                memory.hook_warnings = getattr(memory, 'hook_warnings', {})
                memory.hook_warnings[key] = value
                logger.info(f"PRE_VERIFICATION hook: MODIFY — {key}: {str(value)[:100]}")
        try:
            from prover.verifier.lean_checker import LeanChecker
            checker = LeanChecker(self.lean)
            status, errors, stderr, ms = checker.check(
                problem.theorem_statement, proof)
            return status == AttemptStatus.SUCCESS
        except Exception:
            return False

    def _agent_result_to_attempt(self, r: AgentResult, memory) -> ProofAttempt:
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

    def _try_decompose(self, problem, memory):
        try:
            from prover.decompose.goal_decomposer import GoalDecomposer
            decomposer = GoalDecomposer(self.llm)
            subgoals = decomposer.decompose(problem.theorem_statement)
            if subgoals:
                for sg in subgoals:
                    memory.goal_stack.append(sg.statement)
        except Exception:
            pass

    def _try_conjecture(self, problem, memory):
        try:
            from prover.conjecture.conjecture_proposer import ConjectureProposer
            proposer = ConjectureProposer(self.llm)
            existing = [l.get("statement", "") for l in memory.banked_lemmas[:5]]
            conjectures = proposer.propose(
                problem.theorem_statement, existing_lemmas=existing, n=3, verify=False)
            if conjectures:
                for conj in conjectures:
                    memory.banked_lemmas.append({
                        "name": "conj", "statement": conj,
                        "proof": "", "verified": False})
        except Exception:
            pass
