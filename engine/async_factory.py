"""engine/async_factory.py — 异步引擎组件工厂

Phase B 分层解耦: 本模块仅构建 engine 层组件, 不导入 agent/prover。
证明编排逻辑 (async_prove_round) 已迁移到 prover/pipeline/async_prove.py。

Usage::

    factory = AsyncEngineFactory(config)
    components = await factory.build()
"""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AsyncEngineComponents:
    """异步引擎组件容器

    agent_pool 字段保留用于向后兼容, 但不由本工厂构建。
    构建 agent_pool 的职责在 prover.assembly.SystemAssembler。
    """

    lean_pool: Optional['AsyncLeanPool'] = None
    prefilter: Optional['PreFilter'] = None
    error_intel: Optional['ErrorIntelligence'] = None
    scheduler: Optional['AsyncVerificationScheduler'] = None
    broadcast: Optional['BroadcastBus'] = None
    agent_pool: Optional[object] = None  # 由 prover 层注入
<<<<<<< HEAD
=======

    # ── Knowledge system (由 prover 层注入, engine 层不依赖 knowledge/) ──
    knowledge_store: Optional[object] = None   # UnifiedKnowledgeStore
    knowledge_writer: Optional[object] = None  # KnowledgeWriter
    knowledge_reader: Optional[object] = None  # KnowledgeReader
    knowledge_broadcaster: Optional[object] = None  # KnowledgeBroadcaster
>>>>>>> 7a01a9c (infra complete)

    async def close(self):
        if self.lean_pool:
            try:
                await self.lean_pool.shutdown()
            except Exception as e:
                logger.warning(f"AsyncLeanPool shutdown error: {e}")
        if self.broadcast:
            try:
                self.broadcast.clear()
            except Exception as e:
                logger.warning(f"BroadcastBus clear error: {e}")


class AsyncEngineFactory:
    """异步引擎组件工厂 — 仅构建 engine 层组件

    不导入 agent/ 或 prover/, 保持 engine 层独立性。
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    async def build(self, retriever=None,
                    overrides: dict = None,
                    # Deprecated params (ignored, kept for backward compat)
                    async_llm=None) -> AsyncEngineComponents:
        """构建 engine 层异步组件

        Args:
            retriever: 前提检索器
            overrides: 组件覆盖 (测试用)
            async_llm: (deprecated, ignored) 使用 prover.assembly 构建 agent_pool
        """
        overrides = overrides or {}
        comp = AsyncEngineComponents()

        try:
            # 1. BroadcastBus
            from engine.broadcast import BroadcastBus
            comp.broadcast = overrides.get('broadcast') or BroadcastBus()

            # 2. AsyncLeanPool or ElasticPool
            if 'lean_pool' in overrides:
                comp.lean_pool = overrides['lean_pool']
            else:
                pool_size = self.config.get("lean_pool_size", 4)
                project_dir = self.config.get("lean_project_dir", ".")
                remote_workers = self.config.get("remote_workers", [])

<<<<<<< HEAD
=======
                if remote_workers:
                    # Use ElasticPool for mixed local+remote topology
                    from engine.remote_session import ElasticPool
                    pool = ElasticPool(
                        timeout_seconds=self.config.get("timeout", 30))
                    local_count = self.config.get("local_pool_size", pool_size)
                    if local_count > 0:
                        await pool.add_local(local_count,
                                             project_dir=project_dir)
                    if remote_workers:
                        await pool.add_remote(remote_workers)
                    comp.lean_pool = pool
                else:
                    from engine.async_lean_pool import AsyncLeanPool
                    comp.lean_pool = AsyncLeanPool(
                        pool_size=pool_size, project_dir=project_dir)
                    await comp.lean_pool.start()

>>>>>>> 7a01a9c (infra complete)
            # 3. PreFilter
            from engine.prefilter import PreFilter
            comp.prefilter = overrides.get('prefilter') or PreFilter()

            # 4. ErrorIntelligence
            from engine.error_intelligence import ErrorIntelligence
            comp.error_intel = overrides.get('error_intel') or ErrorIntelligence()

            # 5. AsyncVerificationScheduler
            if 'scheduler' in overrides:
                comp.scheduler = overrides['scheduler']
            else:
                from engine.async_verification_scheduler import AsyncVerificationScheduler
                project_dir = self.config.get("lean_project_dir", ".")
                comp.scheduler = AsyncVerificationScheduler(
                    prefilter=comp.prefilter,
                    lean_pool=comp.lean_pool,
                    error_intel=comp.error_intel,
                    broadcast=comp.broadcast,
                    project_dir=project_dir)

            # 6. agent_pool: 由调用者 (prover 层) 注入
            if 'agent_pool' in overrides:
                comp.agent_pool = overrides['agent_pool']

            return comp

        except Exception as e:
            logger.error(f"AsyncEngineFactory build failed: {e}")
            await comp.close()
            raise


# ── 向后兼容 shim: async_prove_round 已迁移到 prover 层 ──

def async_prove_round(*args, **kwargs):
    """Deprecated: 使用 prover.pipeline.async_prove.async_prove_round"""
    from prover.pipeline.async_prove import async_prove_round as _real  # noqa: layer-compat
    return _real(*args, **kwargs)


def run_async_prove_round(*args, **kwargs):
    """Deprecated: 使用 prover.pipeline.async_prove.run_async_prove_round"""
    from prover.pipeline.async_prove import run_async_prove_round as _real  # noqa: layer-compat
    return _real(*args, **kwargs)


def _default_directions(*args, **kwargs):
    """Deprecated: 使用 prover.pipeline.async_prove._default_directions"""
    from prover.pipeline.async_prove import _default_directions as _real  # noqa: layer-compat
    return _real(*args, **kwargs)


def _build_prompt(*args, **kwargs):
    """Deprecated: 使用 prover.pipeline.async_prove._build_prompt"""
    from prover.pipeline.async_prove import _build_prompt as _real  # noqa: layer-compat
    return _real(*args, **kwargs)
