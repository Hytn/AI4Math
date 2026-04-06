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

        return components

    # ── Agent 层构建方法 ──

    def _build_hooks(self):
        from agent.hooks.hook_manager import HookManager
        from agent.hooks.hook_types import HookEvent
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
        from agent.strategy.budget_allocator import Budget
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
