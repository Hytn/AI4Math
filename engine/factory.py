"""engine/factory.py — 引擎层组件工厂

P0-4 修复: 将 Orchestrator 中内联的 15+ 组件实例化逻辑
抽取到独立的工厂类中, 实现:
  1. 依赖注入 — 组件可替换、可 mock
  2. 原子初始化 — 部分失败时自动清理已启动的资源
  3. 单元可测 — 每个组件可独立构建和测试

Usage::

    factory = EngineFactory(config)
    components = factory.build()

    orchestrator = Orchestrator(
        lean_env=lean_env,
        llm_provider=llm,
        components=components,
    )
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EngineComponents:
    """引擎层所有组件的容器 — Orchestrator 的唯一依赖"""

    # 核心验证组件
    lean_pool: Optional['LeanPool'] = None
    prefilter: Optional['PreFilter'] = None
    error_intel: Optional['ErrorIntelligence'] = None
    scheduler: Optional['VerificationScheduler'] = None
    broadcast: Optional['BroadcastBus'] = None

    # Agent 运行时
    agent_pool: Optional['AgentPool'] = None
    hooks: Optional['HookManager'] = None
    plugins: Optional['PluginLoader'] = None

    # 策略控制
    meta_controller: Optional['MetaController'] = None
    reflector: Optional['Reflector'] = None
    confidence: Optional['ConfidenceEstimator'] = None
    budget: Optional['Budget'] = None

    # 异构引擎
    hetero_engine: Optional['HeterogeneousEngine'] = None

    # 资源管理 (PoolScaler 等后台任务)
    _pool_scaler: Optional['PoolScaler'] = None

    def close(self):
        """释放所有持有资源的组件 (同步版)"""
        errors = []
        # 1. 停止后台调度 (PoolScaler 是 async, 在同步上下文中跳过)
        # 2. 关闭 lean_pool
        if self.lean_pool:
            try:
                self.lean_pool.shutdown()
            except Exception as e:
                errors.append(f"LeanPool: {e}")
        # 3. 清理广播总线
        if self.broadcast:
            try:
                self.broadcast.clear()
            except Exception as e:
                errors.append(f"BroadcastBus: {e}")
        if errors:
            logger.warning(f"EngineComponents close errors: {errors}")

    async def aclose(self):
        """释放所有持有资源的组件 (异步版, 推荐)"""
        errors = []
        # 1. 停止 PoolScaler 后台任务
        if self._pool_scaler:
            try:
                await self._pool_scaler.stop()
            except Exception as e:
                errors.append(f"PoolScaler: {e}")
        # 2. 关闭 lean_pool (支持 sync 和 async 两种)
        if self.lean_pool:
            try:
                if hasattr(self.lean_pool, 'shutdown'):
                    result = self.lean_pool.shutdown()
                    # If it returns a coroutine, await it
                    if hasattr(result, '__await__'):
                        await result
            except Exception as e:
                errors.append(f"LeanPool: {e}")
        # 3. 清理广播总线
        if self.broadcast:
            try:
                self.broadcast.clear()
            except Exception as e:
                errors.append(f"BroadcastBus: {e}")
        if errors:
            logger.warning(f"EngineComponents aclose errors: {errors}")


class EngineFactory:
    """引擎层组件工厂

    只构建 engine 层组件 (LeanPool, PreFilter, ErrorIntelligence,
    VerificationScheduler, BroadcastBus)。不导入 agent/ 或 prover/。

    要构建完整系统 (engine + agent + prover), 使用 prover.assembly.SystemAssembler。
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    def build_engine(self, retriever=None,
                     overrides: dict = None) -> EngineComponents:
        """构建 engine 层组件 (无 agent/prover 依赖)

        Args:
            retriever: 前提检索器 (传给 ErrorIntelligence)
            overrides: 覆盖构建的组件 (用于测试时注入 mock)

        Returns:
            EngineComponents (仅 engine 层字段填充)
        """
        overrides = overrides or {}
        components = EngineComponents()
        cleanup_needed = []

        try:
            # 1. BroadcastBus (无外部依赖)
            components.broadcast = overrides.get('broadcast') or self._build_broadcast()

            # 2. LeanPool (依赖: 项目目录) — 关键组件
            components.lean_pool = overrides.get('lean_pool') or self._build_lean_pool()
            if components.lean_pool is None:
                raise RuntimeError(
                    "LeanPool construction returned None — cannot proceed "
                    "without a verification backend")
            cleanup_needed.append(lambda: components.lean_pool.shutdown())

            # 3. PreFilter (无外部依赖)
            components.prefilter = overrides.get('prefilter') or self._build_prefilter()

            # 4. ErrorIntelligence (依赖: lean_pool, retriever)
            components.error_intel = overrides.get('error_intel') or self._build_error_intel(
                components.lean_pool, retriever)

            # 5. VerificationScheduler (依赖: prefilter, lean_pool, error_intel, broadcast)
            components.scheduler = overrides.get('scheduler') or self._build_scheduler(
                components.prefilter, components.lean_pool,
                components.error_intel, components.broadcast)

            return components

        except Exception as e:
            logger.error(f"EngineFactory build_engine failed: {e}")
            for cleanup in reversed(cleanup_needed):
                try:
                    cleanup()
                except Exception:
                    pass
            raise

    def build(self, llm_provider=None, retriever=None,
              overrides: dict = None) -> EngineComponents:
        """构建完整系统组件 (向后兼容)

        .. deprecated::
            使用 prover.assembly.SystemAssembler.build() 代替。
            本方法保留仅为向后兼容。

        内部委托给 SystemAssembler 完成 agent/prover 层构建。
        """
        from prover.assembly import SystemAssembler
        assembler = SystemAssembler(self.config)
        return assembler.build(
            llm_provider=llm_provider, retriever=retriever,
            overrides=overrides)

    # ── Engine 层构建方法 (无 agent/prover 导入) ──

    def _build_broadcast(self):
        from engine.broadcast import BroadcastBus
        return BroadcastBus()

    def _build_lean_pool(self):
        from engine.async_lean_pool import SyncLeanPool
        pool_size = self.config.get("lean_pool_size", 4)
        project_dir = self.config.get("lean_project_dir", ".")
        pool = SyncLeanPool(pool_size=pool_size, project_dir=project_dir)
        pool.start()
        return pool

    def _build_prefilter(self):
        from engine.prefilter import PreFilter
        return PreFilter()

    def _build_error_intel(self, lean_pool, retriever):
        from engine.error_intelligence import ErrorIntelligence
        return ErrorIntelligence(lean_pool=lean_pool, premise_index=retriever)

    def _build_scheduler(self, prefilter, lean_pool, error_intel, broadcast):
        from engine.verification_scheduler import VerificationScheduler
        project_dir = self.config.get("lean_project_dir", ".")
        return VerificationScheduler(
            prefilter=prefilter, lean_pool=lean_pool,
            error_intel=error_intel, broadcast=broadcast,
            project_dir=project_dir)
