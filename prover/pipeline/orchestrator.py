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
from prover.pipeline._agent_deps import AgentPool
from prover.pipeline._agent_deps import AgentResult
from prover.pipeline._agent_deps import HookManager
from common.hook_types import HookEvent, HookContext, HookAction
from prover.pipeline._agent_deps import PluginLoader
from prover.pipeline._agent_deps import StrategySwitcher
from common.budget import Budget
from common.working_memory import WorkingMemory
from prover.pipeline._agent_deps import ContextWindow

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, lean_env, llm_provider, retriever=None,
                 config=None, on_attempt=None,
                 components: 'EngineComponents' = None):
        self.lean = lean_env
        self.llm = llm_provider
        self.retriever = retriever
        self.config = config or {}
        self.on_attempt = on_attempt

        # P0-4: 依赖注入 — 接收预构建的组件, 或通过 SystemAssembler 构建
        if components is not None:
            self._components = components
        else:
            from prover.assembly import SystemAssembler
            assembler = SystemAssembler(self.config)
            self._components = assembler.build(
                llm_provider=llm_provider, retriever=retriever)

        # 解包组件引用 (向后兼容)
        self.meta = self._components.meta_controller
        self.reflector = self._components.reflector
        self.confidence = self._components.confidence
        self.budget = self._components.budget
        self.pool = self._components.agent_pool
        self.hooks = self._components.hooks
        self.plugins = self._components.plugins
        self.broadcast = self._components.broadcast
        self.lean_pool = self._components.lean_pool
        self.prefilter = self._components.prefilter
        self.error_intel = self._components.error_intel
        self.scheduler = self._components.scheduler
        self.hetero_engine = self._components.hetero_engine

        # 注册插件钩子
        self._register_plugin_hooks()

        # v3: ProofPipeline (Orchestrator.prove 的重构版本)
        from prover.pipeline.proof_pipeline import ProofPipeline
        self._pipeline = ProofPipeline(
            self._components, self.config, self.on_attempt)

    def _register_plugin_hooks(self):
        for name in self.plugins.list_plugins():
            plugin = self.plugins.get(name)
            if plugin and plugin.hooks:
                self.hooks.register_from_plugin(plugin.hooks)

    def prove(self, problem: BenchmarkProblem) -> ProofTrace:
        """主证明入口

        默认使用 ProofPipeline (推荐路径)。
        设置 config['use_legacy_prove'] = True 可回退到旧的内联逻辑。
        """
        if self.config.get("use_legacy_prove", False):
            return self._prove_legacy(problem)
        return self._pipeline.run(problem)

    def prove_pipeline(self, problem: BenchmarkProblem) -> ProofTrace:
        """使用 ProofPipeline 的证明入口

        等价于 prove() (ProofPipeline 已是默认路径)。
        """
        return self._pipeline.run(problem)

    def _prove_legacy(self, problem: BenchmarkProblem) -> ProofTrace:
        """原有的内联证明逻辑

        .. deprecated::
            使用 ProofPipeline (prove() 的默认路径) 代替。
            本方法将在未来版本中移除。
            设置 config['use_legacy_prove'] = True 可临时启用。
        """
        import warnings
        warnings.warn(
            "_prove_legacy is deprecated; prove() now uses ProofPipeline by default. "
            "Set config['use_legacy_prove']=True only as temporary fallback.",
            DeprecationWarning, stacklevel=2)
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
                        if self._verify_proof(problem, r.proof_code, memory,
                                              trace, agent_result=r):
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

            # ── 处理钩子驱动的策略升级 ──
            # 与 MetaController 的轮数驱动升级不同, 钩子升级基于错误模式
            # 检测 (如 RepetitionDetectorHook 发现同一错误连续出现 N 次)。
            # 完整的升级流程: 渐进升级 → 更新 trace → 触发反思 → 注入上下文。
            if round_end.action == HookAction.ESCALATE:
                # 渐进升级 (而非直接跳到 heavy)
                if strategy_name == "light":
                    next_level = "medium"
                elif strategy_name == "medium":
                    next_level = "heavy"
                else:
                    next_level = "heavy"  # 已在最高级别, 保持不变

                old_strategy = strategy_name
                strategy_name = StrategySwitcher.switch(strategy_name, next_level)
                memory.current_strategy = strategy_name
                trace.strategy_path.append(strategy_name)

                logger.info(
                    f"Hook-driven escalation: {old_strategy} → {strategy_name} "
                    f"(reason: {round_end.message[:100]})")

                # 触发反思: 分析为什么当前策略持续失败
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

                # 注入钩子提供的升级原因和修复提示
                if round_end.inject_context:
                    classification.setdefault("domain_hints", {}).update(
                        round_end.inject_context)

            # ── 常规轮次反思: 连续失败 N 轮后自动触发 ──
            # 不再仅依赖策略升级才触发反思。每 reflection_interval 轮
            # (且仍未解决) 运行一次反思, 将分析结论注入下一轮上下文。
            reflection_interval = self.config.get("reflection_interval", 3)
            if (not memory.solved
                    and memory.rounds_completed >= reflection_interval
                    and memory.rounds_completed % reflection_interval == 0):
                reflection_text = self._run_reflection(problem, memory)
                if reflection_text:
                    classification.setdefault("domain_hints", {}).update({
                        "periodic_reflection": (
                            f"## Self-reflection after {memory.rounds_completed} rounds\n"
                            f"{reflection_text[:800]}\n\n"
                            f"Use this analysis to fundamentally change your approach."
                        ),
                    })
                    logger.info(
                        f"Periodic reflection triggered at round "
                        f"{memory.rounds_completed}")

            # ── 验证反馈注入: 将上一轮的结构化反馈注入下一轮上下文 ──
            last_feedback_text = getattr(memory, 'last_feedback_text', '')
            if last_feedback_text:
                classification.setdefault("domain_hints", {}).update({
                    "last_verification_feedback": last_feedback_text,
                })

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

    def _verify_proof(self, problem, proof, memory, trace,
                      agent_result: 'AgentResult' = None) -> bool:
        """验证证明 (v2: 通过 VerificationScheduler 三级验证)

        改进:
          1. 验证后用 refine_confidence() 更新 agent_result 的置信度
          2. 触发 POST_VERIFICATION 钩子 (使插件的 on_error 规则生效)
        """
        pre = self.hooks.fire(HookEvent.PRE_VERIFICATION,
            HookContext(event=HookEvent.PRE_VERIFICATION,
                        theorem_statement=problem.theorem_statement,
                        proof=proof))
        if pre.action == HookAction.SKIP:
            logger.info("PRE_VERIFICATION hook: SKIP (proof rejected by pre-filter)")
            if agent_result:
                from prover.pipeline._agent_deps import ConfidenceEstimator
                agent_result.confidence = ConfidenceEstimator.refine_confidence(
                    agent_result, l0_passed=False)
            return False
        if pre.action == HookAction.MODIFY and pre.inject_context:
            for key, value in pre.inject_context.items():
                memory.hook_warnings = getattr(memory, 'hook_warnings', {})
                memory.hook_warnings[key] = value
                logger.info(f"PRE_VERIFICATION hook: MODIFY — {key}: {str(value)[:100]}")
        try:
            # ── v2: 使用 VerificationScheduler 三级验证 ──
            if self.scheduler:
                result = self.scheduler.verify_complete(
                    theorem=problem.theorem_statement,
                    proof=proof,
                    direction="orchestrator",
                )
                # 将结构化反馈存入 memory, 供下一轮使用
                if result.feedback:
                    memory.last_feedback = result.feedback
                    memory.last_feedback_text = result.feedback.to_prompt(1500)

                # ── 用验证反馈精化置信度 ──
                if agent_result:
                    from prover.pipeline._agent_deps import ConfidenceEstimator
                    agent_result.confidence = ConfidenceEstimator.refine_confidence(
                        agent_result,
                        feedback=result.feedback,
                        l0_passed=result.l0_passed,
                        l1_passed=(result.level_reached in ("L1", "L2") and result.success),
                        l2_passed=result.l2_verified,
                    )

                # ── 触发 POST_VERIFICATION 钩子 ──
                post_ctx = HookContext(
                    event=HookEvent.POST_VERIFICATION,
                    theorem_statement=problem.theorem_statement,
                    proof=proof,
                    errors=[{"message": result.feedback.error_message,
                             "category": result.feedback.error_category}]
                           if result.feedback.error_message else [],
                    dominant_error=result.feedback.error_category or "",
                    metadata={
                        "level_reached": result.level_reached,
                        "success": result.success,
                    },
                )
                post_result = self.hooks.fire(
                    HookEvent.POST_VERIFICATION, post_ctx)
                if post_result.inject_context:
                    for key, value in post_result.inject_context.items():
                        memory.hook_warnings = getattr(memory, 'hook_warnings', {})
                        memory.hook_warnings[key] = value

                return result.success
            else:
                # Fallback: 原有验证路径
                from prover.verifier.lean_checker import LeanChecker
                checker = LeanChecker(self.lean)
                status, errors, stderr, ms = checker.check(
                    problem.theorem_statement, proof)
                return status == AttemptStatus.SUCCESS
        except Exception as e:
            logger.exception(f"Verification error for {problem.problem_id}: {e}")
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
        except Exception as e:
            logger.warning(f"Decompose failed for {problem.problem_id}: {e}")

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
        except Exception as e:
            logger.warning(f"Conjecture failed for {problem.problem_id}: {e}")

    # ── Context manager + 资源清理 ──

    def close(self):
        """释放所有资源 (P0-4: 通过 EngineComponents 统一管理)"""
        if hasattr(self, '_components') and self._components:
            self._components.close()
        logger.info("Orchestrator: shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
