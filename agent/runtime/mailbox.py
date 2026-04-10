"""agent/runtime/mailbox.py — Inter-agent message passing

Inspired by Claude Code's teammateMailbox.ts: typed messages between
agents with priorities, topics, and delivery guarantees.

Enables the "mathematician society" pattern where agents share discoveries,
warn about dead-ends, and request help from specialists.

Usage::

    mailbox = AgentMailbox()

    # Agent A sends a discovery
    mailbox.send(AgentMessage(
        from_agent="induction_expert",
        to_agent="proof_composer",   # or "*" for broadcast
        topic="lemma_found",
        content="Discovered: Nat.add_comm_succ ...",
        priority=0.8,
    ))

    # Agent B checks its inbox
    messages = mailbox.receive("proof_composer", topic="lemma_found")
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class MessageTopic(str, Enum):
    """Pre-defined message topics for common inter-agent communication."""
    LEMMA_FOUND = "lemma_found"          # Agent found a useful intermediate lemma
    DEAD_END = "dead_end"                # Agent hit a dead end, warn others
    TACTIC_SUCCESS = "tactic_success"    # A tactic worked for a similar goal
    TACTIC_FAILURE = "tactic_failure"    # A tactic failed (avoid repeating)
    HELP_REQUEST = "help_request"        # Agent requests specialist help
    PROOF_FRAGMENT = "proof_fragment"    # Partial proof that might be composable
    STRATEGY_SWITCH = "strategy_switch"  # Suggesting a strategy change
    GOAL_DECOMPOSED = "goal_decomposed"  # Sub-goals identified
    CUSTOM = "custom"


@dataclass
class AgentMessage:
    """A typed message between agents."""
    from_agent: str
    to_agent: str                    # Agent name or "*" for broadcast
    topic: str = MessageTopic.CUSTOM
    content: str = ""
    data: dict = field(default_factory=dict)  # Structured payload
    priority: float = 0.5           # 0.0=low, 1.0=urgent
    timestamp: float = field(default_factory=time.time)
    message_id: str = ""
    delivered: bool = False

    def __post_init__(self):
        if not self.message_id:
            self.message_id = f"msg_{int(self.timestamp * 1000)}_{id(self) % 10000}"


class AgentMailbox:
    """Central message exchange for inter-agent communication.

    Thread-safe. Supports:
      - Point-to-point messaging (to_agent=specific name)
      - Broadcast (to_agent="*")
      - Topic-based filtering
      - Priority ordering
      - Message expiry
    """

    def __init__(self, max_queue_size: int = 1000, ttl_seconds: float = 600.0):
        self._queues: dict[str, list[AgentMessage]] = defaultdict(list)
        self._broadcast: list[AgentMessage] = []
        self._lock = threading.Lock()
        self._max_queue_size = max_queue_size
        self._ttl = ttl_seconds

    def send(self, message: AgentMessage) -> bool:
        """Send a message to an agent or broadcast to all."""
        with self._lock:
            if message.to_agent == "*":
                self._broadcast.append(message)
                if len(self._broadcast) > self._max_queue_size:
                    self._broadcast = self._broadcast[-self._max_queue_size:]
            else:
                queue = self._queues[message.to_agent]
                queue.append(message)
                if len(queue) > self._max_queue_size:
                    self._queues[message.to_agent] = queue[-self._max_queue_size:]

            logger.debug(
                f"Message {message.from_agent} → {message.to_agent}: "
                f"[{message.topic}] {message.content[:80]}")
            return True

    def receive(
        self,
        agent_name: str,
        topic: str = None,
        max_messages: int = 20,
        mark_delivered: bool = True,
    ) -> list[AgentMessage]:
        """Receive messages for an agent, optionally filtered by topic.

        Returns messages sorted by priority (highest first), then timestamp.
        Includes both direct messages and broadcasts.
        """
        now = time.time()
        with self._lock:
            # Collect direct messages
            direct = self._queues.get(agent_name, [])
            # Collect broadcasts
            all_msgs = list(direct) + list(self._broadcast)

            # Filter
            filtered = []
            for msg in all_msgs:
                if msg.delivered and mark_delivered:
                    continue
                if topic and msg.topic != topic:
                    continue
                if now - msg.timestamp > self._ttl:
                    continue
                filtered.append(msg)

            # Sort by priority (desc) then timestamp (asc)
            filtered.sort(key=lambda m: (-m.priority, m.timestamp))
            result = filtered[:max_messages]

            # Mark as delivered
            if mark_delivered:
                for msg in result:
                    msg.delivered = True

            return result

    def peek(self, agent_name: str) -> int:
        """Check how many undelivered messages are waiting."""
        with self._lock:
            direct = [m for m in self._queues.get(agent_name, [])
                      if not m.delivered]
            broadcasts = [m for m in self._broadcast if not m.delivered]
            return len(direct) + len(broadcasts)

    def broadcast(self, from_agent: str, topic: str, content: str,
                  data: dict = None, priority: float = 0.5) -> AgentMessage:
        """Convenience: send a broadcast message."""
        msg = AgentMessage(
            from_agent=from_agent, to_agent="*",
            topic=topic, content=content,
            data=data or {}, priority=priority)
        self.send(msg)
        return msg

    def format_inbox_for_prompt(self, agent_name: str,
                                max_tokens: int = 2000) -> str:
        """Format pending messages as context for LLM prompt injection.

        Returns a structured text block that can be injected into the
        agent's system prompt or context window.
        """
        messages = self.receive(agent_name, max_messages=10)
        if not messages:
            return ""

        lines = ["--- Messages from other agents ---"]
        total_len = 0
        for msg in messages:
            line = f"[From {msg.from_agent}, topic={msg.topic}] {msg.content}"
            if total_len + len(line) > max_tokens * 4:
                lines.append(f"... ({len(messages) - len(lines) + 1} more)")
                break
            lines.append(line)
            total_len += len(line)
        lines.append("--- End messages ---")
        return "\n".join(lines)

    def clear(self, agent_name: str = None):
        """Clear messages for an agent or all messages."""
        with self._lock:
            if agent_name:
                self._queues.pop(agent_name, None)
            else:
                self._queues.clear()
                self._broadcast.clear()

    def stats(self) -> dict:
        """Get mailbox statistics."""
        with self._lock:
            return {
                "queues": len(self._queues),
                "total_direct": sum(len(q) for q in self._queues.values()),
                "total_broadcast": len(self._broadcast),
                "undelivered": sum(
                    sum(1 for m in q if not m.delivered)
                    for q in self._queues.values()
                ) + sum(1 for m in self._broadcast if not m.delivered),
            }
