"""prover/assembly.py — 全系统组装器

将 engine 层、agent 层、prover 层的组件组装为完整的证明系统。

职责分离:
  engine/factory.py  → 只构建 engine 层 (LeanPool, PreFilter, ErrorIntel, Scheduler, Broadcast)
  prover/assembly.py → 在 engine 层之上构建 agent + prover 层 (本文件)

这解决了 engine/ 反向依赖 agent/ 的架构倒置问题:
  旧: engine/factory.py 直接 import agent.hooks, agent.plugins, agent.runtime
  新: engine/factory.py 无 agent 导入; prover/assembly.py 负责跨层组装

Usage::

    from prover.assembly import SystemAssembler

    assembler = SystemAssembler(config)
    components = assembler.build(llm_provider=llm, retriever=retriever)

    orchestrator = Orchestrator(lean_env, llm, components=components)
"""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SystemAssembler:
    """全系统组装器: engine + agent + prover

    替代原 EngineFactory.build() 中的 agent/prover 构建逻辑。
    engine/factory.py 现在只负责 engine 层。
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    def build(self, llm_provider=None, retriever=None,
              overrides: dict = None) -> 'EngineComponents':
        """构建完整系统组件

        1. 委托 EngineFactory 构建 engine 层
        2. 在此基础上构建 agent + prover 层
        """
        from engine.factory import EngineFactory, EngineComponents

        overrides = overrides or {}

        # ── Step 1: 构建 engine 层 ──
        engine_factory = EngineFactory(self.config)
        components = engine_factory.build_engine(
            retriever=retriever, overrides=overrides)

        # ── Step 2: 构建 agent 运行时 ──
        components.hooks = (
            overrides.get('hooks') or self._build_hooks())
        components.plugins = (
            overrides.get('plugins') or self._build_plugins())
        components.agent_pool = (
            overrides.get('agent_pool') or self._build_agent_pool(llm_provider))

        # ── Step 3: 构建策略控制 ──
        components.meta_controller = (
            overrides.get('meta_controller') or self._build_meta())
        components.reflector = (
            overrides.get('reflector') or self._build_reflector(llm_provider))
        components.confidence = (
            overrides.get('confidence') or self._build_confidence())
        components.budget = (
            overrides.get('budget') or self._build_budget())

        # ── Step 4: 构建异构引擎 ──
        components.hetero_engine = (
            overrides.get('hetero_engine') or self._build_hetero(
                components.agent_pool, components.plugins, components.hooks,
                retriever, components.broadcast, components.scheduler))

        # ── Step 5: 构建知识系统 ──
        knowledge_components = (
            overrides.get('knowledge') or self._build_knowledge(
                components.broadcast, components.lean_pool))
        components.knowledge_store = knowledge_components.get('store')
        components.knowledge_writer = knowledge_components.get('writer')
        components.knowledge_reader = knowledge_components.get('reader')
        components.knowledge_broadcaster = knowledge_components.get('broadcaster')
        components.knowledge_evolver = knowledge_components.get('evolver')

        # ── Step 6: 构建 Lane 系统 (claw-code-inspired lifecycle) ──
        components.event_bus = (
            overrides.get('event_bus') or self._build_event_bus())
        components.dashboard = (
            overrides.get('dashboard') or self._build_dashboard())
        components.recovery_registry = (
            overrides.get('recovery_registry') or self._build_recovery_registry())
        components.policy_engine = (
            overrides.get('policy_engine') or self._build_policy_engine(
                components.recovery_registry))
        components.session_store = (
            overrides.get('session_store') or self._build_session_store())

        return components

    # ── Agent 层构建方法 ──

    def _build_hooks(self):
        from agent.hooks.hook_manager import HookManager
        from common.hook_types import HookEvent
        from agent.hooks.builtin_hooks import (
            DomainClassifierHook, RepetitionDetectorHook,
            NatSubSafetyHook, ReflectionCloserHook,
        )
        hooks = HookManager()
        hooks.register(HookEvent.ON_PROBLEM_START,
                       DomainClassifierHook(), priority=10)
        hooks.register(HookEvent.ON_ROUND_END,
                       RepetitionDetectorHook(threshold=4), priority=20)
        hooks.register(HookEvent.PRE_VERIFICATION,
                       NatSubSafetyHook(), priority=30)
        hooks.register(HookEvent.ON_STRATEGY_SWITCH,
                       ReflectionCloserHook(), priority=40)
        return hooks

    def _build_plugins(self):
        from agent.plugins.loader import PluginLoader
        plugin_dirs = self.config.get("plugin_dirs", ["plugins/strategies"])
        plugins = PluginLoader(plugin_dirs)
        plugins.discover()
        return plugins

    def _build_agent_pool(self, llm_provider):
        if not llm_provider:
            return None
        from agent.runtime.agent_pool import AgentPool
        return AgentPool(
            llm=llm_provider,
            max_workers=self.config.get("max_workers", 4))

    # ── 策略层构建方法 ──

    def _build_meta(self):
        from agent.strategy.meta_controller import MetaController
        return MetaController(self.config)

    def _build_reflector(self, llm_provider):
        if not llm_provider:
            return None
        from agent.strategy.reflection import Reflector
        return Reflector(llm_provider)

    def _build_confidence(self):
        from agent.strategy.confidence_estimator import ConfidenceEstimator
        return ConfidenceEstimator()

    def _build_budget(self):
        from common.budget import Budget
        return Budget(
            max_samples=self.config.get("max_samples", 128),
            max_wall_seconds=self.config.get("max_wall_seconds", 3600))

    # ── Prover 层构建方法 ──

    def _build_hetero(self, agent_pool, plugins, hooks, retriever,
                      broadcast, scheduler):
        if not agent_pool:
            return None
        from prover.pipeline.heterogeneous_engine import HeterogeneousEngine
        return HeterogeneousEngine(
            pool=agent_pool, plugin_loader=plugins, hook_manager=hooks,
            retriever=retriever, broadcast=broadcast,
            verification_scheduler=scheduler)

    # ── 知识系统构建方法 ──

    def _build_knowledge(self, broadcast=None, pool=None) -> dict:
        """构建统一知识系统的全部组件

        Returns:
            dict with keys: store, writer, reader, broadcaster, evolver
        """
        try:
            from knowledge.store import UnifiedKnowledgeStore
            from knowledge.writer import KnowledgeWriter
            from knowledge.reader import KnowledgeReader
            from knowledge.broadcaster import KnowledgeBroadcaster
            from knowledge.evolver import KnowledgeEvolver

            db_path = self.config.get("knowledge", {}).get(
                "db_path", ":memory:")

            store = UnifiedKnowledgeStore(db_path)
            writer = KnowledgeWriter(store)
            reader = KnowledgeReader(store)
            broadcaster = KnowledgeBroadcaster(
                store, broadcast, pool, writer=writer)
            evolver = KnowledgeEvolver(
                store,
                decay_rate=self.config.get("knowledge", {}).get(
                    "decay_rate", 0.95),
                stale_threshold=self.config.get("knowledge", {}).get(
                    "stale_threshold", 0.1),
            )

            logger.info(
                f"Knowledge system built (db={db_path})")
            return {
                'store': store,
                'writer': writer,
                'reader': reader,
                'broadcaster': broadcaster,
                'evolver': evolver,
            }
        except Exception as e:
            logger.warning(f"Knowledge system build skipped: {e}")
            return {}

    # ── Lane 系统构建方法 (claw-code-inspired) ──

    def _build_event_bus(self):
        from engine.lane.event_bus import ProofEventBus
        return ProofEventBus()

    def _build_dashboard(self):
        from engine.lane.dashboard import ProofDashboard
        return ProofDashboard()

    def _build_recovery_registry(self):
        from engine.lane.recovery import RecoveryRegistry
        return RecoveryRegistry()

    def _build_policy_engine(self, recovery_registry=None):
        from engine.lane.policy import (
            PolicyEngine, InfraRecoveryRule, ConsecutiveSameErrorRule,
            BudgetEscalationRule, BankedLemmaDecomposeRule, ReflectionRule,
        )
        engine = PolicyEngine()
        engine.add_rule(InfraRecoveryRule(recovery_registry=recovery_registry))
        engine.add_rule(ConsecutiveSameErrorRule())
        engine.add_rule(BudgetEscalationRule())
        engine.add_rule(BankedLemmaDecomposeRule())
        engine.add_rule(ReflectionRule())
        return engine

    def _build_session_store(self):
        from engine.lane.proof_session_store import ProofSessionStore
        session_dir = self.config.get("session_dir", ".proof_sessions")
        return ProofSessionStore(directory=session_dir)
