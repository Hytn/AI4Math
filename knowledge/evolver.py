"""knowledge/evolver.py — 知识生命周期管理

知识系统的"遗忘、修正与重组"能力。

三个核心职责:
  1. decay_tick()     — 定期衰减所有知识条目的 decay_factor
  2. gc_stale()       — 清理衰减到阈值以下的陈旧知识
  3. changelog audit  — 所有变更写入 knowledge_changelog

设计原则:
  - 衰减是幂等的: 多次调用同一 tick 不会重复衰减
  - gc_stale 只标记 stale=1, 不物理删除 (可恢复)
  - 所有操作通过 run_in_executor 异步化
  - 可独立运行 (定时任务), 也可集成到证明循环中

Usage::

    evolver = KnowledgeEvolver(store)

    # 每 N 轮证明后调用一次
    stats = await evolver.decay_tick()

    # 定期清理
    removed = await evolver.gc_stale()

    # 启动后台定时任务
    await evolver.start_background(interval_seconds=300)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from knowledge.store import UnifiedKnowledgeStore

logger = logging.getLogger(__name__)


@dataclass
class DecayStats:
    """一次衰减 tick 的统计"""
    tactic_rows_decayed: int = 0
    lemma_rows_decayed: int = 0
    strategy_rows_decayed: int = 0
    tick_timestamp: float = 0.0
    duration_ms: float = 0.0


@dataclass
class GCStats:
    """一次垃圾收集的统计"""
    tactics_marked_stale: int = 0
    lemmas_marked_stale: int = 0
    strategies_marked_stale: int = 0
    gc_timestamp: float = 0.0


class KnowledgeEvolver:
    """知识生命周期管理器

    参数:
        store:            UnifiedKnowledgeStore 实例
        decay_rate:       每次 tick 的衰减因子 (0.95 = 每 tick 保留 95%)
        stale_threshold:  decay_factor 低于此值则标记为 stale
        min_samples:      样本数低于此值的条目不参与衰减 (保护新知识)
        gc_grace_hours:   标记 stale 后多少小时内仍可恢复
    """

    def __init__(self, store: UnifiedKnowledgeStore,
                 decay_rate: float = 0.95,
                 stale_threshold: float = 0.1,
                 min_samples: int = 3,
                 gc_grace_hours: float = 24.0):
        self.store = store
        self.decay_rate = decay_rate
        self.stale_threshold = stale_threshold
        self.min_samples = min_samples
        self.gc_grace_hours = gc_grace_hours
        self._last_tick: float = 0.0
        self._background_task: Optional[asyncio.Task] = None

    # ═══════════════════════════════════════════════════════════
    # Core: Decay tick
    # ═══════════════════════════════════════════════════════════

    async def decay_tick(self) -> DecayStats:
        """执行一次衰减 tick

        将所有 decay_factor 乘以 decay_rate, 使老旧知识逐渐淡出。
        新更新的条目 (decay_factor 已被重置为 1.0) 不受影响。

        Returns:
            DecayStats 包含各表受影响的行数
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._decay_tick_sync)

    def _decay_tick_sync(self) -> DecayStats:
        start = time.time()
        now = time.time()
        stats = DecayStats(tick_timestamp=now)

        with self.store._connect() as conn:
            # Layer 1: tactic_effectiveness
            # 只衰减有足够样本且 decay_factor 仍高于阈值的条目
            cur = conn.execute(
                "UPDATE tactic_effectiveness SET decay_factor = decay_factor * ? "
                "WHERE (successes + failures) >= ? AND decay_factor > ?",
                (self.decay_rate, self.min_samples, self.stale_threshold))
            stats.tactic_rows_decayed = cur.rowcount

            # 记录 changelog
            if stats.tactic_rows_decayed > 0:
                self.store._log_change(
                    conn, layer="L1", entity_type="tactic_effectiveness",
                    entity_id=0, action="batch_decay",
                    new_value=f"rate={self.decay_rate}, rows={stats.tactic_rows_decayed}",
                    reason="periodic decay tick")

            # Layer 1: proved_lemmas
            cur = conn.execute(
                "UPDATE proved_lemmas SET decay_factor = decay_factor * ? "
                "WHERE stale = 0 AND decay_factor > ?",
                (self.decay_rate, self.stale_threshold))
            stats.lemma_rows_decayed = cur.rowcount

            if stats.lemma_rows_decayed > 0:
                self.store._log_change(
                    conn, layer="L1", entity_type="proved_lemmas",
                    entity_id=0, action="batch_decay",
                    new_value=f"rate={self.decay_rate}, rows={stats.lemma_rows_decayed}",
                    reason="periodic decay tick")

            # Layer 2: strategy_patterns
            cur = conn.execute(
                "UPDATE strategy_patterns SET decay_factor = decay_factor * ? "
                "WHERE decay_factor > ?",
                (self.decay_rate, self.stale_threshold))
            stats.strategy_rows_decayed = cur.rowcount

            if stats.strategy_rows_decayed > 0:
                self.store._log_change(
                    conn, layer="L2", entity_type="strategy_patterns",
                    entity_id=0, action="batch_decay",
                    new_value=f"rate={self.decay_rate}, rows={stats.strategy_rows_decayed}",
                    reason="periodic decay tick")

        stats.duration_ms = (time.time() - start) * 1000
        self._last_tick = now

        total = (stats.tactic_rows_decayed + stats.lemma_rows_decayed
                 + stats.strategy_rows_decayed)
        if total > 0:
            logger.info(
                f"KnowledgeEvolver: decay tick complete — "
                f"{stats.tactic_rows_decayed} tactics, "
                f"{stats.lemma_rows_decayed} lemmas, "
                f"{stats.strategy_rows_decayed} strategies "
                f"({stats.duration_ms:.1f}ms)")

        return stats

    # ═══════════════════════════════════════════════════════════
    # Core: Garbage collection
    # ═══════════════════════════════════════════════════════════

    async def gc_stale(self) -> GCStats:
        """标记衰减到阈值以下的知识条目为 stale

        不物理删除, 只设置 stale=1。查询时已自动过滤 stale=0。
        在 gc_grace_hours 内可通过 revive() 恢复。

        Returns:
            GCStats 包含各表被标记的行数
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._gc_stale_sync)

    def _gc_stale_sync(self) -> GCStats:
        now = time.time()
        stats = GCStats(gc_timestamp=now)

        with self.store._connect() as conn:
            # Layer 1: tactic_effectiveness — 低衰减 + 低成功率
            # (没有 stale 字段, 直接删除这些噪音数据)
            cur = conn.execute(
                "DELETE FROM tactic_effectiveness "
                "WHERE decay_factor <= ? AND (successes + failures) >= ?",
                (self.stale_threshold, self.min_samples))
            stats.tactics_marked_stale = cur.rowcount

            if stats.tactics_marked_stale > 0:
                self.store._log_change(
                    conn, layer="L1", entity_type="tactic_effectiveness",
                    entity_id=0, action="gc_delete",
                    new_value=f"threshold={self.stale_threshold}, "
                              f"removed={stats.tactics_marked_stale}",
                    reason="decay below threshold")

            # Layer 1: proved_lemmas — 标记 stale (可恢复)
            cur = conn.execute(
                "UPDATE proved_lemmas SET stale = 1 "
                "WHERE stale = 0 AND decay_factor <= ? AND verified = 0",
                (self.stale_threshold,))
            stats.lemmas_marked_stale = cur.rowcount

            if stats.lemmas_marked_stale > 0:
                self.store._log_change(
                    conn, layer="L1", entity_type="proved_lemmas",
                    entity_id=0, action="gc_mark_stale",
                    new_value=f"threshold={self.stale_threshold}, "
                              f"marked={stats.lemmas_marked_stale}",
                    reason="decay below threshold, unverified")

            # Layer 2: strategy_patterns — 低衰减 + 从未成功
            cur = conn.execute(
                "DELETE FROM strategy_patterns "
                "WHERE decay_factor <= ? AND times_succeeded = 0",
                (self.stale_threshold,))
            stats.strategies_marked_stale = cur.rowcount

            if stats.strategies_marked_stale > 0:
                self.store._log_change(
                    conn, layer="L2", entity_type="strategy_patterns",
                    entity_id=0, action="gc_delete",
                    new_value=f"removed={stats.strategies_marked_stale}",
                    reason="never succeeded + decayed")

        total = (stats.tactics_marked_stale + stats.lemmas_marked_stale
                 + stats.strategies_marked_stale)
        if total > 0:
            logger.info(
                f"KnowledgeEvolver: gc complete — "
                f"{stats.tactics_marked_stale} tactics removed, "
                f"{stats.lemmas_marked_stale} lemmas marked stale, "
                f"{stats.strategies_marked_stale} strategies removed")

        return stats

    # ═══════════════════════════════════════════════════════════
    # Revive: 恢复误标记的知识
    # ═══════════════════════════════════════════════════════════

    async def revive_lemma(self, lemma_id: int, reason: str = "") -> bool:
        """恢复被标记为 stale 的引理"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._revive_lemma_sync, lemma_id, reason)

    def _revive_lemma_sync(self, lemma_id: int, reason: str) -> bool:
        with self.store._connect() as conn:
            cur = conn.execute(
                "UPDATE proved_lemmas SET stale = 0, decay_factor = 0.5 "
                "WHERE id = ? AND stale = 1",
                (lemma_id,))
            if cur.rowcount > 0:
                self.store._log_change(
                    conn, layer="L1", entity_type="proved_lemmas",
                    entity_id=lemma_id, action="revive",
                    old_value="stale=1", new_value="stale=0, decay=0.5",
                    reason=reason or "manual revive")
                return True
            return False

    # ═══════════════════════════════════════════════════════════
    # Background task
    # ═══════════════════════════════════════════════════════════

    async def start_background(self, interval_seconds: float = 300.0):
        """启动后台定时衰减+清理任务"""
        if self._background_task and not self._background_task.done():
            logger.warning("KnowledgeEvolver: background task already running")
            return

        self._background_task = asyncio.create_task(
            self._background_loop(interval_seconds))
        logger.info(
            f"KnowledgeEvolver: background task started "
            f"(interval={interval_seconds}s)")

    async def stop_background(self):
        """停止后台任务"""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError as _exc:
                logger.debug(f"Suppressed exception: {_exc}")
            logger.info("KnowledgeEvolver: background task stopped")

    async def _background_loop(self, interval: float):
        """后台循环: 每 interval 秒执行一次 decay + gc"""
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.decay_tick()
                    await self.gc_stale()
                except Exception as e:
                    logger.warning(f"KnowledgeEvolver background error: {e}")
        except asyncio.CancelledError as _exc:
            logger.debug(f"Suppressed exception: {_exc}")

    # ═══════════════════════════════════════════════════════════
    # Stats
    # ═══════════════════════════════════════════════════════════

    async def stats(self) -> dict:
        """获取生命周期管理统计"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._stats_sync)

    def _stats_sync(self) -> dict:
        with self.store._connect() as conn:
            changelog_count = conn.execute(
                "SELECT COUNT(*) as c FROM knowledge_changelog"
            ).fetchone()["c"]

            recent_actions = conn.execute(
                "SELECT action, COUNT(*) as c FROM knowledge_changelog "
                "GROUP BY action ORDER BY c DESC LIMIT 10"
            ).fetchall()

            low_decay_tactics = conn.execute(
                "SELECT COUNT(*) as c FROM tactic_effectiveness "
                "WHERE decay_factor < 0.3"
            ).fetchone()["c"]

            stale_lemmas = conn.execute(
                "SELECT COUNT(*) as c FROM proved_lemmas WHERE stale = 1"
            ).fetchone()["c"]

        return {
            "last_tick": self._last_tick,
            "decay_rate": self.decay_rate,
            "stale_threshold": self.stale_threshold,
            "changelog_entries": changelog_count,
            "recent_actions": {r["action"]: r["c"] for r in recent_actions},
            "low_decay_tactics": low_decay_tactics,
            "stale_lemmas": stale_lemmas,
        }
