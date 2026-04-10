"""agent/persistence/replay.py — Event replay for recovery and analysis

Rebuilds proof session state from the event log stored in SessionData.events.
This enables:
  1. Crash recovery: replay events to reconstruct the exact state
  2. Analysis: trace what happened during a proof attempt
  3. Debugging: replay specific segments to understand failures

Integrates with the ProofEventBus from engine/lane/event_bus.py.

Usage::

    replayer = EventReplayer()

    # Load a session and replay its events
    session = store.load("sess_abc123")
    timeline = replayer.replay(session.events)

    # Get state at a specific point
    state_at_turn_5 = replayer.state_at(session.events, turn=5)

    # Generate analysis report
    report = replayer.analyze(session.events)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ReplayState:
    """Reconstructed state at a point in the event timeline."""
    turn: int = 0
    tokens_used: int = 0
    status: str = "in_progress"
    best_proof: str = ""
    best_confidence: float = 0.0
    green_level: str = "NONE"
    errors_seen: list[str] = field(default_factory=list)
    tactics_tried: list[str] = field(default_factory=list)
    lemmas_found: list[str] = field(default_factory=list)
    strategies_used: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    checkpoints: list[dict] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class TimelineEntry:
    """A single entry in the replay timeline."""
    index: int
    event_type: str
    timestamp: float
    summary: str
    data: dict = field(default_factory=dict)
    state_snapshot: Optional[ReplayState] = None


@dataclass
class AnalysisReport:
    """Analysis of a proof session from event replay."""
    total_events: int = 0
    total_turns: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    # Outcome
    final_status: str = ""
    proof_found: bool = False
    # Strategy
    strategies_tried: list[str] = field(default_factory=list)
    strategy_switches: int = 0
    # Tools
    tools_used: dict = field(default_factory=dict)  # tool_name → count
    # Errors
    unique_errors: int = 0
    repeated_errors: int = 0
    error_categories: dict = field(default_factory=dict)
    # Efficiency
    turns_to_first_proof: Optional[int] = None
    checkpoints_created: int = 0
    # Recommendations
    recommendations: list[str] = field(default_factory=list)


class EventReplayer:
    """Replay events to reconstruct state and generate analysis."""

    def replay(self, events: list[dict]) -> list[TimelineEntry]:
        """Replay all events and build a timeline.

        Args:
            events: List of event dicts from SessionData.events

        Returns:
            Chronological timeline with state snapshots at each point
        """
        timeline = []
        state = ReplayState()

        for i, event in enumerate(events):
            etype = event.get("type", "unknown")
            ts = event.get("timestamp", 0)
            state.timestamp = ts

            # Update state based on event type
            self._apply_event(state, event)

            # Create timeline entry
            entry = TimelineEntry(
                index=i,
                event_type=etype,
                timestamp=ts,
                summary=self._summarize_event(event),
                data=event,
                state_snapshot=self._snapshot(state),
            )
            timeline.append(entry)

        return timeline

    def state_at(self, events: list[dict], turn: int = None,
                 index: int = None) -> ReplayState:
        """Get state at a specific turn or event index."""
        state = ReplayState()
        for i, event in enumerate(events):
            if index is not None and i > index:
                break
            self._apply_event(state, event)
            if turn is not None and state.turn >= turn:
                break
        return state

    def analyze(self, events: list[dict]) -> AnalysisReport:
        """Generate analysis report from event log."""
        report = AnalysisReport(total_events=len(events))

        if not events:
            return report

        # Time range
        timestamps = [e.get("timestamp", 0) for e in events if e.get("timestamp")]
        if len(timestamps) >= 2:
            report.duration_seconds = timestamps[-1] - timestamps[0]

        error_counts = defaultdict(int)
        tool_counts = defaultdict(int)
        strategies = []
        first_proof_turn = None

        state = ReplayState()
        for event in events:
            etype = event.get("type", "")
            self._apply_event(state, event)

            if etype == "checkpoint":
                report.checkpoints_created += 1
                report.total_turns = max(
                    report.total_turns, event.get("turn", 0))
                report.total_tokens = max(
                    report.total_tokens, event.get("tokens", 0))

            elif etype == "tool_call":
                tool = event.get("tool", "")
                tool_counts[tool] += 1

            elif etype == "error":
                cat = event.get("category", "other")
                error_counts[cat] += 1

            elif etype == "strategy_switch":
                new_strategy = event.get("new_strategy", "")
                if new_strategy:
                    strategies.append(new_strategy)
                    report.strategy_switches += 1

            elif etype == "proof_found":
                if first_proof_turn is None:
                    first_proof_turn = event.get("turn", 0)
                report.proof_found = True

            elif etype == "finalized":
                report.final_status = (
                    "succeeded" if event.get("success") else "failed")
                report.total_turns = event.get("total_turns", report.total_turns)
                report.total_tokens = event.get("total_tokens", report.total_tokens)

        report.tools_used = dict(tool_counts)
        report.error_categories = dict(error_counts)
        report.unique_errors = len(error_counts)
        report.repeated_errors = sum(
            max(0, c - 1) for c in error_counts.values())
        report.strategies_tried = strategies
        report.turns_to_first_proof = first_proof_turn

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _apply_event(self, state: ReplayState, event: dict):
        """Apply an event to the current state."""
        etype = event.get("type", "")

        if etype == "checkpoint":
            state.turn = event.get("turn", state.turn)
            state.tokens_used = event.get("tokens", state.tokens_used)
            state.status = event.get("status", state.status)
            state.checkpoints.append(event)

        elif etype == "tool_call":
            tool = event.get("tool", "")
            if tool:
                state.tools_called.append(tool)

        elif etype == "error":
            cat = event.get("category", "other")
            msg = event.get("message", "")[:100]
            state.errors_seen.append(f"[{cat}] {msg}")

        elif etype == "tactic_tried":
            tactic = event.get("tactic", "")
            if tactic:
                state.tactics_tried.append(tactic)

        elif etype == "lemma_found":
            lemma = event.get("lemma", "")
            if lemma:
                state.lemmas_found.append(lemma)

        elif etype == "strategy_switch":
            state.strategies_used.append(event.get("new_strategy", ""))

        elif etype == "proof_found":
            state.best_proof = event.get("proof", state.best_proof)
            state.best_confidence = event.get("confidence", state.best_confidence)

        elif etype == "verification":
            state.green_level = event.get("green_level", state.green_level)

        elif etype == "finalized":
            state.status = "succeeded" if event.get("success") else "failed"

        elif etype == "resumed":
            state.status = "in_progress"

    def _snapshot(self, state: ReplayState) -> ReplayState:
        """Create a copy of the current state."""
        return ReplayState(
            turn=state.turn,
            tokens_used=state.tokens_used,
            status=state.status,
            best_proof=state.best_proof,
            best_confidence=state.best_confidence,
            green_level=state.green_level,
            errors_seen=list(state.errors_seen[-5:]),
            tactics_tried=list(state.tactics_tried[-10:]),
            lemmas_found=list(state.lemmas_found),
            strategies_used=list(state.strategies_used),
            tools_called=list(state.tools_called[-10:]),
            checkpoints=list(state.checkpoints[-3:]),
            timestamp=state.timestamp,
        )

    def _summarize_event(self, event: dict) -> str:
        """Generate a one-line summary of an event."""
        etype = event.get("type", "unknown")
        summaries = {
            "checkpoint": lambda e: f"Checkpoint at turn {e.get('turn', '?')} ({e.get('trigger', '')})",
            "tool_call": lambda e: f"Called {e.get('tool', '?')}",
            "error": lambda e: f"Error [{e.get('category', '?')}]: {e.get('message', '')[:60]}",
            "strategy_switch": lambda e: f"Strategy → {e.get('new_strategy', '?')}",
            "proof_found": lambda e: f"Proof found (confidence={e.get('confidence', '?')})",
            "verification": lambda e: f"Verification → {e.get('green_level', '?')}",
            "finalized": lambda e: f"Session {'succeeded' if e.get('success') else 'failed'}",
            "resumed": lambda e: "Session resumed",
        }
        fn = summaries.get(etype, lambda e: f"Event: {etype}")
        return fn(event)

    def _generate_recommendations(self, report: AnalysisReport) -> list[str]:
        """Generate improvement recommendations from the analysis."""
        recs = []

        if report.repeated_errors > 5:
            recs.append(
                "High error repetition detected. Consider adding the "
                "RepetitionDetectorHook or lowering the consecutive error threshold.")

        if report.strategy_switches == 0 and not report.proof_found:
            recs.append(
                "No strategy switches occurred. The PolicyEngine may need "
                "more aggressive escalation rules.")

        if report.strategy_switches > 8:
            recs.append(
                "Excessive strategy switching. Consider increasing the "
                "threshold before switching to allow deeper exploration.")

        tool_total = sum(report.tools_used.values())
        if tool_total == 0 and not report.proof_found:
            recs.append(
                "No tools were used. Enable agentic mode (execute_with_tools) "
                "so agents can search premises and verify intermediate steps.")

        if "premise_search" not in report.tools_used and not report.proof_found:
            recs.append(
                "premise_search was never called. Ensure it's available in "
                "the tool registry — finding relevant lemmas is often critical.")

        if report.turns_to_first_proof and report.turns_to_first_proof > 10:
            recs.append(
                f"First proof found at turn {report.turns_to_first_proof}. "
                "Consider trying simpler tactics first (simp, omega, ring) "
                "before complex strategies.")

        return recs
