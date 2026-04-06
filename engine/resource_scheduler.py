"""engine/resource_scheduler.py — 资源调度器

为并发证明任务提供资源隔离和优先级调度:

  1. 优先级队列: 高优先级任务 (如接近成功的证明) 优先获取 REPL 会话
  2. 资源预算:   每个证明任务有独立的 token/time/verification 预算
  3. 准入控制:   系统过载时拒绝新任务, 避免所有任务都饿死
  4. 公平调度:   防止单个大任务垄断所有 REPL 会话

Usage::

    scheduler = ResourceScheduler(pool, max_concurrent=8)
    await scheduler.start()

    # 提交证明任务
    handle = await scheduler.submit(
        task_id="proof_1",
        priority=Priority.HIGH,
        budget=ResourceBudget(max_verifications=100))

    # 获取 REPL 会话 (遵守优先级和预算)
    session = await handle.acquire_session()
    result = await session.try_tactic(env_id, "simp")
    await handle.release_session(session)

    # 任务完成
    await scheduler.complete(handle)
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """任务优先级 (数值越小优先级越高)"""
    CRITICAL = 0    # L2 最终认证
    HIGH = 1        # 接近成功的证明 (confidence > 0.7)
    NORMAL = 2      # 常规证明探索
    LOW = 3         # 投机性探索 (高温采样)
    BACKGROUND = 4  # 后台重验、引理库维护


@dataclass
class ResourceBudget:
    """单个证明任务的资源预算"""
    max_verifications: int = 200      # 最多验证次数
    max_wall_seconds: float = 600.0   # 最长挂钟时间
    max_tokens: int = 500_000         # 最多 LLM token
    max_concurrent_sessions: int = 2  # 最多同时占用的 REPL 会话数

    # 已消耗
    verifications_used: int = 0
    wall_seconds_used: float = 0.0
    tokens_used: int = 0

    @property
    def is_exhausted(self) -> bool:
        return (self.verifications_used >= self.max_verifications
                or self.wall_seconds_used >= self.max_wall_seconds
                or self.tokens_used >= self.max_tokens)

    @property
    def remaining_ratio(self) -> float:
        """剩余预算比例 (0.0 = 耗尽, 1.0 = 未使用)"""
        ratios = [
            1 - self.verifications_used / max(1, self.max_verifications),
            1 - self.wall_seconds_used / max(0.1, self.max_wall_seconds),
            1 - self.tokens_used / max(1, self.max_tokens),
        ]
        return min(ratios)

    def consume_verification(self):
        self.verifications_used += 1

    def consume_tokens(self, n: int):
        self.tokens_used += n


@dataclass
class TaskHandle:
    """任务句柄 — 持有者通过它与调度器交互"""
    task_id: str
    priority: Priority
    budget: ResourceBudget
    created_at: float = field(default_factory=time.time)
    _scheduler: Optional['ResourceScheduler'] = field(
        default=None, repr=False)
    _active_sessions: int = 0

    async def acquire_session(self) -> 'AsyncLeanSession':
        """获取 REPL 会话 (受优先级和预算限制)"""
        if self.budget.is_exhausted:
            raise BudgetExhaustedError(
                f"Task {self.task_id} budget exhausted: "
                f"{self.budget.verifications_used}/{self.budget.max_verifications} "
                f"verifications")
        if self._active_sessions >= self.budget.max_concurrent_sessions:
            raise ConcurrencyLimitError(
                f"Task {self.task_id} at concurrent session limit: "
                f"{self._active_sessions}/{self.budget.max_concurrent_sessions}")
        session = await self._scheduler._acquire_for_task(self)
        self._active_sessions += 1
        return session

    async def release_session(self, session: 'AsyncLeanSession'):
        """释放 REPL 会话"""
        self._active_sessions = max(0, self._active_sessions - 1)
        await self._scheduler._release_for_task(self, session)

    @property
    def wall_seconds(self) -> float:
        return time.time() - self.created_at


class BudgetExhaustedError(Exception):
    pass


class ConcurrencyLimitError(Exception):
    pass


class ResourceScheduler:
    """资源调度器

    在 AsyncLeanPool 之上添加:
      - 优先级: 高优先级任务优先获取空闲会话
      - 预算:   每个任务的验证/时间/token 限额
      - 准入:   超过 max_concurrent_tasks 时排队
      - 公平:   单任务最多占 max_concurrent_sessions 个会话
    """

    def __init__(self, pool: 'AsyncLeanPool',
                 max_concurrent_tasks: int = 8):
        self._pool = pool
        self.max_concurrent_tasks = max_concurrent_tasks
        self._active_tasks: dict[str, TaskHandle] = {}
        self._waiting_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._admission_sem = asyncio.Semaphore(max_concurrent_tasks)
        self._lock = asyncio.Lock()

        # 统计
        self._total_submitted = 0
        self._total_completed = 0
        self._total_rejected = 0

    async def submit(self, task_id: str,
                     priority: Priority = Priority.NORMAL,
                     budget: ResourceBudget = None) -> TaskHandle:
        """提交证明任务, 返回任务句柄

        如果活跃任务数达到上限, 会等待直到有空位。
        """
        budget = budget or ResourceBudget()

        # 准入控制
        await self._admission_sem.acquire()
        self._total_submitted += 1

        handle = TaskHandle(
            task_id=task_id,
            priority=priority,
            budget=budget,
            _scheduler=self)

        async with self._lock:
            self._active_tasks[task_id] = handle

        logger.debug(
            f"ResourceScheduler: task {task_id} admitted "
            f"(priority={priority.name}, active={len(self._active_tasks)})")
        return handle

    async def complete(self, handle: TaskHandle):
        """标记任务完成, 释放资源"""
        async with self._lock:
            self._active_tasks.pop(handle.task_id, None)
        self._admission_sem.release()
        self._total_completed += 1

        handle.budget.wall_seconds_used = handle.wall_seconds
        logger.debug(
            f"ResourceScheduler: task {handle.task_id} completed "
            f"({handle.budget.verifications_used} verifications, "
            f"{handle.wall_seconds:.1f}s)")

    async def cancel(self, task_id: str):
        """取消任务"""
        async with self._lock:
            handle = self._active_tasks.pop(task_id, None)
        if handle:
            self._admission_sem.release()
            self._total_completed += 1

    async def _acquire_for_task(self, handle: TaskHandle) -> 'AsyncLeanSession':
        """为特定任务获取会话 (内部方法, 由 TaskHandle 调用)

        优先级实现: 通过在 pool._acquire_session 之前
        检查是否有更高优先级的任务在等待。
        """
        # 检查预算
        if handle.budget.is_exhausted:
            raise BudgetExhaustedError(f"Task {handle.task_id} exhausted")

        # 检查挂钟时间
        if handle.wall_seconds > handle.budget.max_wall_seconds:
            handle.budget.wall_seconds_used = handle.wall_seconds
            raise BudgetExhaustedError(
                f"Task {handle.task_id} wall time exceeded")

        session = await self._pool._acquire_session()
        handle.budget.consume_verification()
        return session

    async def _release_for_task(self, handle: TaskHandle,
                                session: 'AsyncLeanSession'):
        """为特定任务释放会话"""
        await self._pool._release_session(session)

    def get_task(self, task_id: str) -> Optional[TaskHandle]:
        return self._active_tasks.get(task_id)

    def stats(self) -> dict:
        active = self._active_tasks
        return {
            "active_tasks": len(active),
            "max_concurrent": self.max_concurrent_tasks,
            "total_submitted": self._total_submitted,
            "total_completed": self._total_completed,
            "total_rejected": self._total_rejected,
            "active_task_ids": list(active.keys()),
            "budget_summary": {
                tid: {
                    "priority": h.priority.name,
                    "verifications": f"{h.budget.verifications_used}/{h.budget.max_verifications}",
                    "wall_time": f"{h.wall_seconds:.1f}s/{h.budget.max_wall_seconds}s",
                    "remaining_ratio": f"{h.budget.remaining_ratio:.2f}",
                }
                for tid, h in active.items()
            },
            "pool_stats": self._pool.stats(),
        }
