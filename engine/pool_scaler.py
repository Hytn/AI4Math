"""engine/pool_scaler.py — REPL 连接池动态伸缩器

根据实时负载自动增减 REPL 会话数量:
  - 所有会话都忙 + 有排队请求 → 扩容 (创建新会话)
  - 空闲会话超过阈值 + 持续空闲 → 缩容 (关闭多余会话)

设计原则:
  - 渐进伸缩: 每次最多增减 1-2 个会话, 避免抖动
  - 冷却期:   扩缩后等待一段时间再次评估, 防止频繁操作
  - 下限保护: 至少保留 min_sessions 个会话 (避免缩到 0)
  - 上限保护: 最多 max_sessions 个会话 (避免耗尽系统资源)

用法::

    pool = AsyncLeanPool(pool_size=2)
    scaler = PoolScaler(pool, min_sessions=1, max_sessions=16)

    # 启动后台伸缩监控
    await scaler.start()

    # ... 使用 pool ...

    await scaler.stop()
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ScaleDecision:
    """伸缩决策"""
    action: str = "hold"     # "scale_up", "scale_down", "hold"
    reason: str = ""
    current_size: int = 0
    target_size: int = 0
    timestamp: float = field(default_factory=time.time)


class PoolScaler:
    """REPL 连接池动态伸缩器

    监控指标:
      - busy_ratio:    忙碌会话 / 总会话数
      - queue_depth:   等待获取会话的协程数 (通过 Condition 的 waiter 估算)
      - idle_duration: 连续空闲的时间

    伸缩策略:
      扩容条件: busy_ratio > scale_up_threshold AND queue_depth > 0
      缩容条件: busy_ratio < scale_down_threshold 持续 idle_cooldown 秒
    """

    def __init__(self, pool: 'AsyncLeanPool',
                 min_sessions: int = 1,
                 max_sessions: int = 16,
                 check_interval: float = 2.0,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 cooldown_seconds: float = 10.0,
                 scale_step: int = 1):
        self._pool = pool
        self.min_sessions = min_sessions
        self.max_sessions = max_sessions
        self._check_interval = check_interval
        self._scale_up_threshold = scale_up_threshold
        self._scale_down_threshold = scale_down_threshold
        self._cooldown = cooldown_seconds
        self._scale_step = scale_step

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_scale_time = 0.0
        self._idle_since: Optional[float] = None
        self._decisions: list[ScaleDecision] = []

    async def start(self):
        """启动后台伸缩监控协程"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"PoolScaler: started (min={self.min_sessions}, "
            f"max={self.max_sessions})")

    async def stop(self):
        """停止监控"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("PoolScaler: stopped")

    def evaluate(self) -> ScaleDecision:
        """评估当前是否需要伸缩 (可手动调用)"""
        stats = self._pool.stats()
        active = stats["active_sessions"]
        busy = stats["busy_sessions"]
        total = max(1, active)
        busy_ratio = busy / total

        now = time.time()

        # 冷却期检查
        if now - self._last_scale_time < self._cooldown:
            return ScaleDecision(
                action="hold", reason="cooldown",
                current_size=active, target_size=active)

        # 扩容判定
        if busy_ratio >= self._scale_up_threshold and active < self.max_sessions:
            target = min(active + self._scale_step, self.max_sessions)
            self._idle_since = None
            return ScaleDecision(
                action="scale_up",
                reason=f"busy_ratio={busy_ratio:.2f} >= {self._scale_up_threshold}",
                current_size=active, target_size=target)

        # 缩容判定
        if busy_ratio <= self._scale_down_threshold and active > self.min_sessions:
            if self._idle_since is None:
                self._idle_since = now
            idle_duration = now - self._idle_since
            if idle_duration >= self._cooldown:
                target = max(active - self._scale_step, self.min_sessions)
                return ScaleDecision(
                    action="scale_down",
                    reason=f"idle for {idle_duration:.0f}s, "
                           f"busy_ratio={busy_ratio:.2f}",
                    current_size=active, target_size=target)
            else:
                return ScaleDecision(
                    action="hold",
                    reason=f"idle {idle_duration:.0f}s < cooldown {self._cooldown}s",
                    current_size=active, target_size=active)
        else:
            self._idle_since = None

        return ScaleDecision(
            action="hold",
            reason=f"busy_ratio={busy_ratio:.2f} within thresholds",
            current_size=active, target_size=active)

    async def apply(self, decision: ScaleDecision):
        """执行伸缩决策"""
        if decision.action == "hold":
            return

        if decision.action == "scale_up":
            await self._scale_up(
                decision.target_size - decision.current_size)
        elif decision.action == "scale_down":
            await self._scale_down(
                decision.current_size - decision.target_size)

        self._last_scale_time = time.time()
        self._decisions.append(decision)
        logger.info(
            f"PoolScaler: {decision.action} "
            f"{decision.current_size} → {decision.target_size} "
            f"({decision.reason})")

    async def _scale_up(self, count: int):
        """添加新会话 (通过 pool 的公开 API)"""
        for _ in range(count):
            await self._pool.add_session()

    async def _scale_down(self, count: int):
        """移除空闲会话 (通过 pool 的公开 API)"""
        for _ in range(count):
            removed = await self._pool.remove_idle_session()
            if not removed:
                break

    async def _monitor_loop(self):
        """后台监控循环"""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                if not self._running:
                    break
                decision = self.evaluate()
                if decision.action != "hold":
                    await self.apply(decision)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"PoolScaler monitor error: {e}")

    def stats(self) -> dict:
        pool_stats = self._pool.stats()
        return {
            "pool_active": pool_stats["active_sessions"],
            "pool_busy": pool_stats["busy_sessions"],
            "min_sessions": self.min_sessions,
            "max_sessions": self.max_sessions,
            "total_scale_events": len(self._decisions),
            "last_decision": (
                self._decisions[-1].action if self._decisions else "none"),
            "idle_since": self._idle_since,
        }

    @property
    def recent_decisions(self) -> list[ScaleDecision]:
        return self._decisions[-20:]
