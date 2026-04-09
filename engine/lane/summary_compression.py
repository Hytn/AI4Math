"""engine/lane/summary_compression.py — Compressed proof status summaries

Inspired by claw-code's summary_compression.rs (ROADMAP Phase 2, item 6):
"Collapse noisy event streams into: current phase, last successful
 checkpoint, current blocker, recommended next recovery action."

Replaces the need to scan raw event logs. Produces a compact,
machine-readable + human-readable summary suitable for:
  1. Dashboard display (ProofDashboard.summary_line())
  2. LLM context injection (< 200 tokens)
  3. Monitoring / alerting systems
  4. Logging one-liners

Usage::

    summary = compress_proof_status(sm)
    print(summary.one_liner)
    # → "[VERIFYING r3] best=goals_closed | blocker=type_mismatch×4 | next=switch_role"

    # For LLM injection:
    prompt_text = summary.for_prompt(max_chars=300)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from engine.lane.task_state import (
    ProofTaskStateMachine, TaskStatus, ProofFailureClass, TaskEvent,
)
from engine.lane.green_contract import GreenLevel
from engine.lane.policy import PolicyEngine, PolicyAction, PolicyDecision
from engine.lane.recovery import RecoveryRegistry


@dataclass
class ProofStatusSummary:
    """Compressed, machine-readable proof status.

    All fields are set by compress_proof_status().
    """
    # ── Core status ──
    task_id: str = ""
    theorem_name: str = ""
    status: str = ""               # TaskStatus value
    is_terminal: bool = False

    # ── Progress ──
    current_phase: str = ""        # e.g. "VERIFYING round 3"
    rounds_completed: int = 0
    total_samples: int = 0
    total_api_tokens: int = 0

    # ── Best result so far ──
    best_green_level: str = "none"  # GreenLevel short name
    best_attempt_preview: str = ""  # first 80 chars of best proof code
    banked_lemmas_count: int = 0

    # ── Current blocker ──
    blocker_class: str = ""        # ProofFailureClass value or ""
    blocker_message: str = ""      # short error message
    blocker_streak: int = 0        # consecutive same-error count

    # ── Recommended next action ──
    recommended_action: str = ""   # PolicyAction value
    recommended_reason: str = ""

    # ── Timing ──
    elapsed_seconds: float = 0.0
    recovery_attempts: int = 0

    # ── Event summary ──
    event_count: int = 0
    failure_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def one_liner(self) -> str:
        """Single-line status for logging.

        Format: [STATUS rN] best=LEVEL | blocker=CLASS×COUNT | next=ACTION
        """
        parts = [f"[{self.status.upper()} r{self.rounds_completed}]"]

        if self.best_green_level != "none":
            parts.append(f"best={self.best_green_level}")

        if self.blocker_class:
            streak = f"×{self.blocker_streak}" if self.blocker_streak > 1 else ""
            parts.append(f"blocker={self.blocker_class}{streak}")

        if self.recommended_action and self.recommended_action != "continue":
            parts.append(f"next={self.recommended_action}")

        parts.append(f"{self.elapsed_seconds:.0f}s")

        return " | ".join(parts)

    def for_prompt(self, max_chars: int = 300) -> str:
        """Render for LLM prompt injection.

        Compact enough for context window (< 200 tokens typically).
        """
        lines = []

        # Status line
        lines.append(
            f"Status: {self.status} (round {self.rounds_completed}, "
            f"{self.total_samples} samples, {self.elapsed_seconds:.0f}s)")

        # Best progress
        if self.best_green_level != "none":
            lines.append(f"Best: {self.best_green_level}")
            if self.best_attempt_preview:
                lines.append(f"Preview: {self.best_attempt_preview}")

        # Banked lemmas
        if self.banked_lemmas_count > 0:
            lines.append(f"Banked lemmas: {self.banked_lemmas_count}")

        # Current blocker
        if self.blocker_class:
            lines.append(
                f"Blocker: {self.blocker_class} "
                f"({self.blocker_streak}x): {self.blocker_message}")

        # Failure distribution (top 3)
        if self.failure_distribution:
            top = sorted(self.failure_distribution.items(),
                         key=lambda x: -x[1])[:3]
            dist = ", ".join(f"{k}={v}" for k, v in top)
            lines.append(f"Errors: {dist}")

        # Recommended action
        if self.recommended_action and self.recommended_action != "continue":
            lines.append(
                f"Suggested: {self.recommended_action}"
                + (f" ({self.recommended_reason})"
                   if self.recommended_reason else ""))

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."
        return text

    def to_dict(self) -> dict:
        """Machine-readable JSON-serializable dict."""
        return {
            "task_id": self.task_id,
            "theorem_name": self.theorem_name,
            "status": self.status,
            "is_terminal": self.is_terminal,
            "current_phase": self.current_phase,
            "rounds_completed": self.rounds_completed,
            "total_samples": self.total_samples,
            "best_green_level": self.best_green_level,
            "banked_lemmas_count": self.banked_lemmas_count,
            "blocker_class": self.blocker_class,
            "blocker_streak": self.blocker_streak,
            "recommended_action": self.recommended_action,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "recovery_attempts": self.recovery_attempts,
            "failure_distribution": self.failure_distribution,
        }


def compress_proof_status(
    sm: ProofTaskStateMachine,
    policy: PolicyEngine = None,
) -> ProofStatusSummary:
    """Compress a ProofTaskStateMachine into a compact summary.

    This is the main entry point — call it on any active or completed
    state machine to get a snapshot.

    Args:
        sm: The state machine to summarize
        policy: Optional PolicyEngine to get recommended action
    """
    ctx = sm.context
    events = sm.events

    summary = ProofStatusSummary(
        task_id=sm.task_id,
        theorem_name=ctx.theorem_name,
        status=sm.status.value,
        is_terminal=sm.status.is_terminal,
        rounds_completed=ctx.rounds_completed,
        total_samples=ctx.total_samples,
        total_api_tokens=ctx.total_api_tokens,
        banked_lemmas_count=len(ctx.banked_lemmas),
        recovery_attempts=sm.recovery_attempts,
        event_count=len(events),
    )

    # ── Current phase ──
    last_detail = ""
    for e in reversed(events):
        if e.detail:
            last_detail = e.detail
            break
    summary.current_phase = (
        f"{sm.status.value.upper()}"
        + (f" {last_detail}" if last_detail else ""))

    # ── Timing ──
    if events:
        first_ts = events[0].timestamp
        last_ts = events[-1].timestamp
        summary.elapsed_seconds = last_ts - first_ts

    # ── Best result ──
    if ctx.best_attempt_code:
        summary.best_attempt_preview = ctx.best_attempt_code[:80].replace(
            "\n", " ")
    # Estimate green level from context
    if sm.status == TaskStatus.SUCCEEDED:
        summary.best_green_level = "goals_closed"
    elif ctx.best_attempt_code:
        summary.best_green_level = "syntax_clean"

    # ── Blocker analysis ──
    failure_events = [e for e in events if e.failure is not None]
    if failure_events:
        # Failure distribution
        dist: Counter = Counter()
        for e in failure_events:
            dist[e.failure.failure_class.value] += 1
        summary.failure_distribution = dict(dist)

        # Current blocker (most recent failure)
        last_failure = failure_events[-1].failure
        summary.blocker_class = last_failure.failure_class.value
        summary.blocker_message = last_failure.message[:100]

        # Consecutive streak of same error
        streak = 1
        for e in reversed(failure_events[:-1]):
            if e.failure.failure_class == last_failure.failure_class:
                streak += 1
            else:
                break
        summary.blocker_streak = streak

    # ── Recommended action (via PolicyEngine) ──
    if policy and not sm.status.is_terminal:
        try:
            decision = policy.evaluate(sm)
            summary.recommended_action = decision.action.value
            summary.recommended_reason = decision.reason[:80]
        except Exception:
            summary.recommended_action = "continue"
    elif sm.status.is_terminal:
        summary.recommended_action = sm.status.value

    return summary


def compress_dashboard(
    dashboard,
    policy: PolicyEngine = None,
) -> dict:
    """Compress an entire ProofDashboard into a compact report.

    Returns a dict with:
      - one_liner: single-line global status
      - active_summaries: list of ProofStatusSummary for in-progress tasks
      - global_stats: aggregated statistics
    """
    snapshot = dashboard.snapshot()
    tasks = dashboard._tasks

    active_summaries = []
    for task_id, sm in tasks.items():
        if not sm.status.is_terminal:
            s = compress_proof_status(sm, policy)
            active_summaries.append(s)

    total = snapshot["summary"]["total"]
    succeeded = snapshot["summary"]["succeeded"]
    failed = snapshot["summary"]["failed"] + snapshot["summary"]["given_up"]
    active = snapshot["summary"]["in_progress"]
    blocked = snapshot["summary"]["blocked"]
    rate = snapshot["summary"]["pass_rate"]

    one_liner = (
        f"[{succeeded}/{total} proved ({rate:.0%})] "
        f"active={active} blocked={blocked} failed={failed}")

    return {
        "one_liner": one_liner,
        "active_summaries": [s.to_dict() for s in active_summaries],
        "active_one_liners": [s.one_liner for s in active_summaries],
        "global_stats": snapshot["summary"],
        "resources": snapshot["resources"],
    }
