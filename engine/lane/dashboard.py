"""engine/lane/dashboard.py — Machine-readable proof status dashboard

Inspired by claw-code's lane board (ROADMAP Phase 4, item 12):
"a machine-readable board of repos, active claws, worktrees,
 branch freshness, red/green state, current blocker, merge readiness."

Provides a JSON-serializable snapshot of the entire proof system state.

Usage::

    dashboard = ProofDashboard()
    dashboard.register_task(task_sm)
    ...
    snapshot = dashboard.snapshot()
    # → {"total": 488, "succeeded": 12, "in_progress": [...], ...}
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from engine.lane.task_state import ProofTaskStateMachine, TaskStatus


class ProofDashboard:
    """Aggregates state from all active proof tasks.

    Thread-safe. Can be polled for JSON snapshots by monitoring tools.
    """

    def __init__(self):
        self._tasks: dict[str, ProofTaskStateMachine] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()

    def register_task(self, sm: ProofTaskStateMachine):
        with self._lock:
            self._tasks[sm.task_id] = sm

    def unregister_task(self, task_id: str):
        with self._lock:
            self._tasks.pop(task_id, None)

    def get_task(self, task_id: str) -> Optional[ProofTaskStateMachine]:
        with self._lock:
            return self._tasks.get(task_id)

    def snapshot(self) -> dict:
        """Full machine-readable state snapshot."""
        with self._lock:
            tasks = list(self._tasks.values())

        status_counts = Counter(t.status for t in tasks)
        in_progress = [
            t.snapshot() for t in tasks if not t.status.is_terminal
        ]
        recent_successes = [
            t.snapshot() for t in tasks if t.status == TaskStatus.SUCCEEDED
        ][-5:]  # last 5
        recent_failures = [
            t.snapshot() for t in tasks if t.status in (TaskStatus.FAILED, TaskStatus.GIVEN_UP)
        ][-5:]
        blockers = [
            t.snapshot() for t in tasks if t.status == TaskStatus.BLOCKED
        ]

        # Aggregate stats
        total_samples = sum(t.context.total_samples for t in tasks)
        total_tokens = sum(t.context.total_api_tokens for t in tasks)
        total_lemmas = sum(len(t.context.banked_lemmas) for t in tasks)

        return {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
            "summary": {
                "total": len(tasks),
                "succeeded": status_counts.get(TaskStatus.SUCCEEDED, 0),
                "failed": status_counts.get(TaskStatus.FAILED, 0),
                "given_up": status_counts.get(TaskStatus.GIVEN_UP, 0),
                "in_progress": len(in_progress),
                "blocked": status_counts.get(TaskStatus.BLOCKED, 0),
                "pass_rate": (
                    status_counts.get(TaskStatus.SUCCEEDED, 0) /
                    max(1, status_counts.get(TaskStatus.SUCCEEDED, 0) +
                        status_counts.get(TaskStatus.FAILED, 0) +
                        status_counts.get(TaskStatus.GIVEN_UP, 0))
                ),
            },
            "resources": {
                "total_samples_generated": total_samples,
                "total_api_tokens_used": total_tokens,
                "total_banked_lemmas": total_lemmas,
            },
            "in_progress": in_progress,
            "blockers": blockers,
            "recent_successes": recent_successes,
            "recent_failures": recent_failures,
            "status_distribution": {
                status.value: count
                for status, count in status_counts.items()
            },
        }

    def summary_line(self) -> str:
        """One-line status for logging."""
        s = self.snapshot()["summary"]
        return (
            f"[{s['succeeded']}/{s['total']} proved] "
            f"fail={s['failed']} skip={s['given_up']} "
            f"active={s['in_progress']} blocked={s['blocked']} "
            f"rate={s['pass_rate']:.1%}"
        )
