"""prover/pipeline/proof_loop.py — DEPRECATED 兼容 shim

v3 起, ``ProofLoop.single_attempt()`` 内部直接走 ``UnifiedProofRunner`` +
``whole_proof_repair`` profile。本文件保留外部 API 仅为不破坏:

  - ``verification/run_full_verification.py`` 中的烟测
  - ``prover.pipeline.sequential_engine`` / ``rollout_engine`` 旧管线
  - 历史测试

新代码应直接使用::

    from prover.unified import UnifiedProofRunner, get_profile
    runner = UnifiedProofRunner(llm=async_llm, lean_pool=pool)
    result = await runner.run(problem, profile_name="whole_proof_repair")

或通过 ``ProofPipeline`` 的 config 切换 profile::

    pipeline = ProofPipeline(comp, config={"profile": "whole_proof_repair"})

老 API 行为
==========
- ``single_attempt(problem, memory, temperature, attempt_num) -> ProofAttempt``
  仍然返回 ProofAttempt; 内部委托给 ``UnifiedProofRunner.run`` 后用
  ``unified_to_attempt`` 翻译。

- ``self.max_repair_rounds`` 映射到 profile 的 ``max_turns``。
"""
from __future__ import annotations

import asyncio
import logging

from dataclasses import replace
from prover.models import ProofAttempt, AttemptStatus

logger = logging.getLogger(__name__)


class ProofLoop:
    """Deprecated thin shim over ``UnifiedProofRunner``.

    保留构造签名 ``ProofLoop(lean_env, llm, retriever=None, config=None)``
    以兼容 ``SequentialEngine`` / ``RolloutEngine`` / verification scripts。
    """

    def __init__(self, lean_env, llm, retriever=None, config=None):
        self.lean = lean_env
        self.llm = llm
        self.retriever = retriever
        self.config = config or {}
        self.max_repair_rounds = self.config.get("max_repair_rounds", 2)
        # 翻译到 unified profile 的 max_turns
        # max_repair_rounds 是"修复轮数", profile.max_turns 是"总轮数 (含初始)"
        self._max_turns = max(1, int(self.max_repair_rounds) + 1)

    def single_attempt(
        self,
        problem,
        memory,
        temperature: float = 0.7,
        attempt_num: int = 1,
    ) -> ProofAttempt:
        """委托给 UnifiedProofRunner; 用 whole_proof_repair profile。

        失败时回退到 legacy 实现, 确保 verification 烟测路径不挂。
        """
        try:
            return self._run_via_unified(
                problem, memory, temperature, attempt_num)
        except Exception as e:
            logger.warning(
                f"UnifiedProofRunner path failed in ProofLoop "
                f"({type(e).__name__}: {e}); falling back to legacy")
            try:
                from prover.pipeline.proof_loop_legacy import (
                    ProofLoop as LegacyLoop)
                legacy = LegacyLoop(self.lean, self.llm,
                                    self.retriever, self.config)
                return legacy.single_attempt(
                    problem, memory, temperature, attempt_num)
            except Exception as e2:
                logger.error(f"Legacy fallback also failed: {e2}")
                attempt = ProofAttempt(attempt_number=attempt_num)
                attempt.lean_result = AttemptStatus.LLM_ERROR
                attempt.lean_stderr = f"both unified and legacy failed: {e2}"
                return attempt

    # ── internal ──────────────────────────────────────────────────

    def _run_via_unified(self, problem, memory, temperature, attempt_num):
        from prover.unified import UnifiedProofRunner, get_profile
        from prover.unified.adapters import unified_to_attempt

        # 选 profile + override
        if self.max_repair_rounds == 0:
            base = get_profile("whole_proof")
        else:
            base = get_profile("whole_proof_repair")
        profile = replace(
            base, temperature=temperature, max_turns=self._max_turns)

        # 适配 LLM (sync 或 async 都 OK; 内部会自动 wrap)
        llm = self._coerce_async_llm(self.llm)

        runner = UnifiedProofRunner(
            llm=llm,
            lean_pool=getattr(self.lean, "pool", None) or self.lean,
            knowledge_store=None,
            retriever=self.retriever,
            broadcast_bus=None,
        )

        # 同步入口 → 起一个事件循环
        try:
            loop = asyncio.get_event_loop()
            running = loop.is_running()
        except RuntimeError:
            running = False

        if running:
            # 在异步 caller 中调 sync API — 不推荐, 但兜底
            future = asyncio.ensure_future(
                runner.run(problem, profile=profile))
            ur = loop.run_until_complete(future)
        else:
            ur = asyncio.run(runner.run(problem, profile=profile))

        attempt = unified_to_attempt(ur, attempt_number=attempt_num)
        return attempt

    @staticmethod
    def _coerce_async_llm(llm):
        """如果 llm 是 sync, 用 _SyncToAsyncAdapter 包一层。"""
        if llm is None:
            return None
        import inspect
        gen = getattr(llm, "generate", None)
        chat = getattr(llm, "chat", None)
        if (gen and inspect.iscoroutinefunction(gen)) or \
           (chat and inspect.iscoroutinefunction(chat)):
            return llm
        from prover.pipeline.heterogeneous_engine import _SyncToAsyncAdapter
        return _SyncToAsyncAdapter(llm)
