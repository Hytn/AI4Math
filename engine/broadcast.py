"""engine/broadcast.py — 跨线程实时广播总线

所有并行搜索方向之间的知识共享通道。

三类广播消息:
  1. NegativeKnowledge  — "ring 对 ℕ 减法无效" → 所有方向同时避开这条死路
  2. PositiveDiscovery   — "找到了引理 Nat.sub_add_cancel" → 所有方向的候选集扩充
  3. PartialProof        — "前 3 步已证完, 剩余 goal 是 Y" → 其他方向可 fork 继续

设计要点:
  - 发布-订阅模型, 发布者不阻塞
  - 每个订阅者有独立队列, 不会因慢消费者拖慢快消费者
  - 消息带 TTL, 过期自动丢弃 (防止陈旧信息误导)
  - 线程安全: 所有操作通过 threading.Lock 保护
"""
from __future__ import annotations
import time
import threading
import logging
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """广播消息类型"""
    NEGATIVE_KNOWLEDGE = "negative_knowledge"
    POSITIVE_DISCOVERY = "positive_discovery"
    PARTIAL_PROOF = "partial_proof"
    LEMMA_PROVEN = "lemma_proven"
    GOAL_CLOSED = "goal_closed"
    STRATEGY_INSIGHT = "strategy_insight"


def _deep_freeze(obj):
    """Recursively freeze nested structures.

    dict  → MappingProxyType (read-only view)
    list  → tuple (immutable)
    other → unchanged
    """
    from types import MappingProxyType
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_deep_freeze(item) for item in obj)
    return obj


@dataclass(frozen=True)
class BroadcastMessage:
    """不可变广播消息

    structured 字段在构造时自动 deep-freeze 为 MappingProxyType/tuple,
    确保消费者无法通过引用修改内容。

    使用 __new__ + __init__ 的标准 frozen dataclass 模式:
    所有 freeze 操作在工厂方法中完成, __post_init__ 不需要 object.__setattr__。
    """
    msg_type: MessageType
    source: str              # 发送方的 agent/direction 名称
    content: str             # 人类可读的描述 (供 LLM 消费)
    structured: object = field(default_factory=dict)  # MappingProxyType after freeze
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = 300.0  # 5 分钟过期

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl_seconds

    @staticmethod
    def _freeze_structured(data: dict) -> object:
        """将 dict 深度冻结为 MappingProxyType (供工厂方法使用)"""
        import copy
        if isinstance(data, dict):
            return _deep_freeze(copy.deepcopy(data))
        return data

    # ── 工厂方法: 快速构造常用消息 ──

    @staticmethod
    def negative(source: str, tactic: str, error_category: str,
                 reason: str, goal_type: str = "") -> BroadcastMessage:
        """构造负面知识消息: 某个 tactic 在某种 goal 上失败"""
        return BroadcastMessage(
            msg_type=MessageType.NEGATIVE_KNOWLEDGE,
            source=source,
            content=(
                f"AVOID: `{tactic}` fails on this problem. "
                f"Reason: {reason}"
            ),
            structured=BroadcastMessage._freeze_structured({
                "failed_tactic": tactic,
                "error_category": error_category,
                "reason": reason,
                "goal_type": goal_type,
            }),
        )

    @staticmethod
    def positive(source: str, discovery: str,
                 lemma_name: str = "", lemma_statement: str = "",
                 ) -> BroadcastMessage:
        """构造正面发现消息: 找到有用的引理或洞察"""
        return BroadcastMessage(
            msg_type=MessageType.POSITIVE_DISCOVERY,
            source=source,
            content=f"USEFUL: {discovery}",
            structured=BroadcastMessage._freeze_structured({
                "lemma_name": lemma_name,
                "lemma_statement": lemma_statement,
                "discovery": discovery,
            }),
        )

    @staticmethod
    def partial_proof(source: str, proof_so_far: str,
                      remaining_goals: list[str],
                      env_id: int = -1,
                      goals_closed: int = 0,
                      ) -> BroadcastMessage:
        """构造部分证明消息: 前 N 步成功, 可供其他方向 fork"""
        return BroadcastMessage(
            msg_type=MessageType.PARTIAL_PROOF,
            source=source,
            content=(
                f"PROGRESS: {goals_closed} goal(s) closed. "
                f"Remaining: {len(remaining_goals)} goal(s). "
                f"Proof so far:\n{proof_so_far[:500]}"
            ),
            structured=BroadcastMessage._freeze_structured({
                "proof_so_far": proof_so_far,
                "remaining_goals": remaining_goals,
                "env_id": env_id,
                "goals_closed": goals_closed,
            }),
        )

    @staticmethod
    def lemma_proven(source: str, name: str, statement: str,
                     proof: str, env_id: int = -1,
                     ) -> BroadcastMessage:
        """构造已证引理消息: 一个辅助引理已被验证"""
        return BroadcastMessage(
            msg_type=MessageType.LEMMA_PROVEN,
            source=source,
            content=(
                f"LEMMA PROVEN: {name}\n"
                f"  {statement}\n"
                f"  Proof: {proof[:200]}"
            ),
            structured=BroadcastMessage._freeze_structured({
                "lemma_name": name,
                "lemma_statement": statement,
                "lemma_proof": proof,
                "env_id": env_id,
            }),
        )


class Subscription:
    """单个订阅者的消息队列"""

    def __init__(self, subscriber_id: str, filter_types: set[MessageType] = None,
                 max_queue: int = 100):
        self.subscriber_id = subscriber_id
        self.filter_types = filter_types  # None = 接收所有类型
        self._queue: deque[BroadcastMessage] = deque(maxlen=max_queue)
        self._lock = threading.Lock()

    def push(self, msg: BroadcastMessage):
        """非阻塞推送 (如果队列满, 丢弃最旧的消息)"""
        if msg.source == self.subscriber_id:
            return  # 不接收自己发的消息
        if self.filter_types and msg.msg_type not in self.filter_types:
            return  # 类型过滤
        with self._lock:
            self._queue.append(msg)

    def drain(self) -> list[BroadcastMessage]:
        """取出所有未读消息 (清空队列), 自动过滤过期消息"""
        with self._lock:
            messages = list(self._queue)
            self._queue.clear()
        return [m for m in messages if not m.is_expired]

    def peek(self, n: int = 5) -> list[BroadcastMessage]:
        """查看最新 N 条消息 (不清空)"""
        with self._lock:
            recent = list(self._queue)[-n:]
        return [m for m in recent if not m.is_expired]

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)


class BroadcastBus:
    """发布-订阅广播总线

    核心保证:
      1. publish() 永不阻塞 — O(N) where N = subscriber count
      2. 每个 subscriber 有独立队列 — 慢消费者不影响快消费者
      3. 消息自动过期 — TTL 机制防止陈旧信息
      4. 线程安全 — 所有操作通过锁保护

    Usage::

        bus = BroadcastBus()
        sub_a = bus.subscribe("direction_A")
        sub_b = bus.subscribe("direction_B")

        # Direction A 发现引理
        bus.publish(BroadcastMessage.positive(
            source="direction_A",
            discovery="Nat.sub_add_cancel solves the ℕ subtraction",
            lemma_name="Nat.sub_add_cancel",
        ))

        # Direction B 收到消息
        messages = sub_b.drain()
        # → [BroadcastMessage(POSITIVE_DISCOVERY, ...)]

        # Direction A 不会收到自己的消息
        messages_a = sub_a.drain()
        # → []
    """

    def __init__(self, dedup_window_seconds: float = 30.0):
        self._subscribers: dict[str, Subscription] = {}
        self._lock = threading.Lock()
        self._history: deque[BroadcastMessage] = deque(maxlen=500)
        self._message_count = 0
        self._dedup_count = 0
        self._callbacks: list[Callable[[BroadcastMessage], None]] = []
        # ── 去重机制 ──
        # 在 dedup_window 秒内, 相同 (msg_type, 核心内容) 的消息不重复广播。
        # 避免 4 个方向同时发现 "ring 失败" 时产生 4 条近似消息浪费 prompt 空间。
        self._dedup_window = dedup_window_seconds
        self._recent_fingerprints: deque[tuple[float, str]] = deque(maxlen=200)

    def subscribe(self, subscriber_id: str,
                  filter_types: set[MessageType] = None,
                  ) -> Subscription:
        """注册一个订阅者, 返回其专属 Subscription 对象"""
        with self._lock:
            sub = Subscription(subscriber_id, filter_types)
            self._subscribers[subscriber_id] = sub
            return sub

    def unsubscribe(self, subscriber_id: str):
        """注销订阅者"""
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def publish(self, msg: BroadcastMessage):
        """广播消息到所有订阅者 (非阻塞, 自动去重)

        去重策略: 在 dedup_window 秒内, 相同 (msg_type, 核心关键词) 的消息
        不重复广播。核心关键词根据消息类型提取:
          - NEGATIVE_KNOWLEDGE: failed_tactic + error_category
          - POSITIVE_DISCOVERY: lemma_name 或 discovery 前50字
          - PARTIAL_PROOF: goals_closed 数量
          - 其他: content 前80字
        """
        fingerprint = self._compute_fingerprint(msg)
        now = time.time()

        with self._lock:
            # 清理过期指纹
            while (self._recent_fingerprints
                   and now - self._recent_fingerprints[0][0] > self._dedup_window):
                self._recent_fingerprints.popleft()

            # 检查重复
            if any(fp == fingerprint for _, fp in self._recent_fingerprints):
                self._dedup_count += 1
                logger.debug(f"[Broadcast] dedup: suppressed {msg.msg_type.value} "
                             f"from {msg.source}")
                return

            self._recent_fingerprints.append((now, fingerprint))
            self._message_count += 1
            self._history.append(msg)
            subscribers = list(self._subscribers.values())
            callbacks = list(self._callbacks)

        # 在锁外分发, 避免持锁时间过长
        for sub in subscribers:
            try:
                sub.push(msg)
            except Exception as e:
                logger.warning(f"Broadcast push failed for {sub.subscriber_id}: {e}")

        for cb in callbacks:
            try:
                cb(msg)
            except Exception as e:
                logger.warning(f"Broadcast callback failed: {e}")

        logger.debug(
            f"[Broadcast] {msg.msg_type.value} from {msg.source}: "
            f"{msg.content[:80]}..."
        )

    def on_message(self, callback: Callable[[BroadcastMessage], None]):
        """注册全局回调 (用于日志、监控等)"""
        with self._lock:
            self._callbacks.append(callback)

    def get_recent(self, n: int = 10,
                   msg_type: MessageType = None) -> list[BroadcastMessage]:
        """获取最近 N 条消息 (全局历史, 用于新加入的方向获取上下文)"""
        with self._lock:
            history = list(self._history)
        if msg_type:
            history = [m for m in history if m.msg_type == msg_type]
        return [m for m in history[-n:] if not m.is_expired]

    def render_for_prompt(self, subscriber_id: str,
                          max_messages: int = 10,
                          max_chars: int = 2000) -> str:
        """将未读消息渲染为可直接注入 LLM prompt 的文本

        这是广播系统与 Agent 层的核心接口:
        将结构化的广播消息转化为 LLM 可消费的自然语言上下文。
        """
        sub = self._subscribers.get(subscriber_id)
        if not sub:
            return ""

        messages = sub.drain()
        if not messages:
            return ""

        # 按优先级排序: 已证引理 > 部分证明 > 正面发现 > 负面知识
        priority = {
            MessageType.LEMMA_PROVEN: 0,
            MessageType.PARTIAL_PROOF: 1,
            MessageType.POSITIVE_DISCOVERY: 2,
            MessageType.NEGATIVE_KNOWLEDGE: 3,
            MessageType.GOAL_CLOSED: 1,
            MessageType.STRATEGY_INSIGHT: 2,
        }
        messages.sort(key=lambda m: priority.get(m.msg_type, 5))

        parts = ["## Teammate discoveries (use these)\n"]
        total_chars = 0
        for msg in messages[:max_messages]:
            line = f"- [{msg.source}] {msg.content}\n"
            if total_chars + len(line) > max_chars:
                break
            parts.append(line)
            total_chars += len(line)

        return "".join(parts)

    def stats(self) -> dict:
        """广播总线统计"""
        with self._lock:
            return {
                "total_messages": self._message_count,
                "dedup_suppressed": self._dedup_count,
                "active_subscribers": len(self._subscribers),
                "history_size": len(self._history),
                "pending_per_subscriber": {
                    sid: sub.pending_count
                    for sid, sub in self._subscribers.items()
                },
            }

    def clear(self):
        """重置 (新问题开始时调用)"""
        with self._lock:
            for sub in self._subscribers.values():
                sub.drain()
            self._history.clear()
            self._recent_fingerprints.clear()
            self._message_count = 0
            self._dedup_count = 0

    @staticmethod
    def _compute_fingerprint(msg: BroadcastMessage) -> str:
        """计算消息的去重指纹

        根据消息类型提取核心关键词, 忽略 source 和 timestamp,
        使得不同方向发出的相同发现只广播一次。
        """
        mtype = msg.msg_type.value
        s = msg.structured

        if msg.msg_type == MessageType.NEGATIVE_KNOWLEDGE:
            # 关键词: 失败的 tactic + 错误类别 + goal 类型
            # 修复: 加入 goal_type 区分不同 goal 上的同名 tactic 失败
            # (例如 ring 在 n+0=n 上失败 vs ring 在 x*y=y*x 上失败是不同信息)
            goal_key = s.get('goal_type', '')[:60]
            return f"{mtype}|{s.get('failed_tactic', '')}|{s.get('error_category', '')}|{goal_key}"
        elif msg.msg_type == MessageType.POSITIVE_DISCOVERY:
            # 关键词: 引理名, 或发现描述的前50字
            key = s.get("lemma_name", "") or s.get("discovery", "")[:50]
            return f"{mtype}|{key}"
        elif msg.msg_type == MessageType.LEMMA_PROVEN:
            return f"{mtype}|{s.get('lemma_name', '')}"
        elif msg.msg_type == MessageType.PARTIAL_PROOF:
            # 关键词: 已关闭 goal 数 + 证明代码前100字
            return f"{mtype}|{s.get('goals_closed', 0)}|{s.get('proof_so_far', '')[:100]}"
        else:
            return f"{mtype}|{msg.content[:80]}"
