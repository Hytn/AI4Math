"""engine/lane/event_bus.py — Typed event bus for proof lane events

Inspired by claw-code's LaneEvent system (lane_events.rs):
"Events over scraped prose" — downstream consumers subscribe to
typed events instead of parsing log output.

Replaces AI4Math's current pattern of logging.info() for state changes.

Usage::

    bus = ProofEventBus()

    # Subscribe
    bus.subscribe("task.*", lambda event: print(event))
    bus.subscribe("task.failure.*", knowledge_writer.on_failure)
    bus.subscribe("infra.*", alerting.on_infra_event)

    # Publish (called by state machine transitions)
    bus.publish(task_event)
"""
from __future__ import annotations

import fnmatch
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Any

from engine.lane.task_state import TaskEvent

logger = logging.getLogger(__name__)

EventHandler = Callable[[TaskEvent], None]


class ProofEventBus:
    """Publish-subscribe event bus for proof lane events.

    Thread-safe. Handlers are called synchronously in the publisher's thread.
    For async workflows, handlers should enqueue to an asyncio.Queue.

    Pattern matching uses fnmatch-style globs:
      "task.*"           — all task events
      "task.failure.*"   — all failure events
      "task.succeeded"   — exact match
      "*"                — everything
    """

    def __init__(self, max_log_size: int = 10_000):
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._lock = threading.Lock()
        self._event_log: list[TaskEvent] = []
        self._max_log_size = max_log_size

    def subscribe(self, pattern: str, handler: EventHandler):
        """Subscribe a handler to events matching the glob pattern."""
        with self._lock:
            self._handlers[pattern].append(handler)

    def unsubscribe(self, pattern: str, handler: EventHandler):
        """Remove a handler subscription."""
        with self._lock:
            handlers = self._handlers.get(pattern, [])
            if handler in handlers:
                handlers.remove(handler)

    def publish(self, event: TaskEvent):
        """Publish an event to all matching subscribers.

        Also appends to the internal event log for replay/debugging.
        """
        # Append to log
        with self._lock:
            self._event_log.append(event)
            if len(self._event_log) > self._max_log_size:
                self._event_log = self._event_log[-self._max_log_size:]

            # Collect matching handlers
            matched_handlers: list[EventHandler] = []
            for pattern, handlers in self._handlers.items():
                if fnmatch.fnmatch(event.event_name, pattern):
                    matched_handlers.extend(handlers)

        # Invoke outside lock to avoid deadlock
        for handler in matched_handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    f"Event handler error for {event.event_name}: "
                    f"{handler.__qualname__}")

    def recent_events(self, n: int = 50, pattern: str = None) -> list[TaskEvent]:
        """Get recent events, optionally filtered by pattern."""
        with self._lock:
            events = list(self._event_log)
        if pattern:
            events = [e for e in events if fnmatch.fnmatch(e.event_name, pattern)]
        return events[-n:]

    def events_for_task(self, task_id: str) -> list[TaskEvent]:
        """Get all events for a specific task (by checking metadata)."""
        with self._lock:
            return [e for e in self._event_log
                    if e.metadata.get("task_id") == task_id]

    def clear(self):
        """Clear the event log."""
        with self._lock:
            self._event_log.clear()


# ─── Singleton global bus ────────────────────────────────────────────────────

_global_bus: ProofEventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> ProofEventBus:
    """Get or create the global event bus singleton."""
    global _global_bus
    if _global_bus is None:
        with _bus_lock:
            if _global_bus is None:
                _global_bus = ProofEventBus()
    return _global_bus


# ─── Convenience: connect state machine to bus ───────────────────────────────

def wire_state_machine_to_bus(sm, bus: ProofEventBus = None):
    """Monkey-patch a ProofTaskStateMachine to publish all events to the bus.

    Usage::

        sm = ProofTaskStateMachine(...)
        wire_state_machine_to_bus(sm)
        # Now all sm transitions automatically publish to the global bus
    """
    bus = bus or get_event_bus()
    original_push = sm._push_event

    def publishing_push(*args, **kwargs):
        event = original_push(*args, **kwargs)
        event.metadata["task_id"] = sm.task_id
        bus.publish(event)
        return event

    sm._push_event = publishing_push
