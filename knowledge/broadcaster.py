"""knowledge/broadcaster.py — 知识广播桥接

将知识写入事件自动桥接到 BroadcastBus，实现多方向/多智能体实时共享。

职责：
  1. 验证结果 → 判定是否值得广播 → BroadcastBus.publish()
  2. 已证引理 → 注入 REPL 会话 (Pool.share_lemma) + 广播
  3. 队友发现 → 从 BroadcastBus 读取 → 注入知识库

Usage::

    broadcaster = KnowledgeBroadcaster(store, broadcast_bus, pool)

    # 在验证回调中
    await broadcaster.on_tactic_result(step, direction="automation")

    # 在引理证明后
    await broadcaster.on_lemma_proved(lemma, direction="structured")
"""
from __future__ import annotations

import logging
from typing import Optional

from engine.broadcast import BroadcastBus, BroadcastMessage, MessageType
from engine.proof_context_store import StepDetail
from knowledge.goal_normalizer import normalize_goal_for_key
from knowledge.store import UnifiedKnowledgeStore
from knowledge.types import LemmaRecord
from knowledge.writer import KnowledgeWriter

logger = logging.getLogger(__name__)


class KnowledgeBroadcaster:
    """知识写入 + 广播的桥接器"""

    def __init__(self,
                 store: UnifiedKnowledgeStore,
                 broadcast: Optional[BroadcastBus] = None,
                 pool=None,
                 writer: Optional[KnowledgeWriter] = None):
        self.store = store
        self.broadcast = broadcast
        self.pool = pool  # AsyncLeanPool or ElasticPool
        self.writer = writer or KnowledgeWriter(store)

        # 防重复广播：记录最近已广播的 tactic+goal 组合
        self._recent_broadcasts: set[str] = set()
        self._max_recent = 200

    async def on_tactic_result(
            self, step: StepDetail,
            direction: str = "",
            theorem: str = "",
            domain: str = "",
            trace_id: int = 0) -> None:
        """验证结果回调 — 写入知识库 + 条件广播

        广播条件：
        - 正面发现：首次在某个 goal_pattern 上成功的 tactic
        - 负面知识：在某个 goal_pattern 上累计失败 3+ 次的 tactic
        """
        # 写入知识库
        await self.writer.ingest_step(
            step, theorem=theorem, domain=domain, trace_id=trace_id)

        if not self.broadcast:
            return

        goal_pattern = normalize_goal_for_key(
            " ".join(step.goals_before))
        broadcast_key = f"{step.tactic}::{goal_pattern}"

        # 防重复
        if broadcast_key in self._recent_broadcasts:
            return

        success = (step.env_id_after >= 0 and not step.error_message)

        if success:
            # 查询是否是新发现 (之前没成功过)
            records = await self.store.query_tactic_effectiveness(
                goal_pattern, domain, top_k=1)
            te = records[0] if records else None

            if te and te.successes <= 1:
                # 首次成功 — 值得广播
                self.broadcast.publish(BroadcastMessage.positive(
                    source=direction or "knowledge",
                    discovery=f"`{step.tactic}` works on: "
                              f"{goal_pattern[:80]}"))
                self._track_broadcast(broadcast_key)

        elif step.error_category:
            # 查询累计失败次数
            errors = await self.store.query_error_patterns(
                goal_pattern=goal_pattern,
                tactic=step.tactic.strip(),
                top_k=1)
            if errors and errors[0].frequency >= 3:
                ep = errors[0]
                self.broadcast.publish(BroadcastMessage.negative(
                    source=direction or "knowledge",
                    tactic=step.tactic.strip(),
                    error_category=ep.error_category,
                    reason=f"failed {ep.frequency}x on {goal_pattern[:60]}"
                           + (f", try {ep.typical_fix}" if ep.typical_fix else "")))
                self._track_broadcast(broadcast_key)

    async def on_lemma_proved(
            self, lemma: LemmaRecord,
            direction: str = "") -> None:
        """引理证明回调 — 写入知识库 + 广播 + 注入 REPL"""
        # 写入知识库
        await self.store.add_lemma(lemma)

        # 广播
        if self.broadcast:
            self.broadcast.publish(BroadcastMessage.lemma_proven(
                source=direction or "knowledge",
                name=lemma.name,
                statement=lemma.statement,
                proof=lemma.proof))

        # 注入 REPL 会话
        if self.pool and lemma.verified:
            try:
                await self.pool.share_lemma(lemma.to_lean())
            except Exception as e:
                logger.debug(f"KnowledgeBroadcaster: share_lemma failed: {e}")

    async def on_proof_completed(
            self, context_id: int,
            steps: list[StepDetail],
            success: bool,
            theorem: str = "",
            duration_ms: float = 0.0,
            direction: str = "",
            domain: str = "") -> int:
        """完整证明结果回调 — 写入 + 广播

        Returns:
            trace_id
        """
        trace_id = await self.writer.ingest_proof_result(
            context_id=context_id,
            steps=steps,
            success=success,
            theorem=theorem,
            duration_ms=duration_ms,
            domain=domain)

        if self.broadcast and success:
            # 提取 tactic 序列
            tactics = [s.tactic for s in steps if not s.error_message]
            proof_summary = " → ".join(tactics[:8])
            self.broadcast.publish(BroadcastMessage.partial_proof(
                source=direction or "knowledge",
                proof_so_far=proof_summary,
                remaining_goals=[]))

        return trace_id

    def _track_broadcast(self, key: str):
        self._recent_broadcasts.add(key)
        if len(self._recent_broadcasts) > self._max_recent:
            # 简单清理：丢弃一半
            to_keep = list(self._recent_broadcasts)[self._max_recent // 2:]
            self._recent_broadcasts = set(to_keep)
