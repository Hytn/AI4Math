"""engine/lane/proof_session_store.py — Proof session persistence & checkpoint recovery

Inspired by claw-code's session_store.py and session_control.rs.

Problem: A proof task running for 30+ minutes can be interrupted by REPL timeout,
API disconnect, or OOM kill. Without persistence, all accumulated knowledge
(banked lemmas, error patterns, broadcast history, partial progress) is lost.

Solution: Checkpoint the full proof session state after each round. On restart,
detect the checkpoint and resume from the last known good state.

Stored state:
  - WorkingMemory (attempt history, error patterns, banked lemmas, strategy)
  - TaskStateMachine snapshot (status, events, failures, recovery count)
  - RoundContext metadata (round number, classification, strategy name)
  - Knowledge accumulated (broadcast messages, negative knowledge)
  - ProofTrace (all attempts so far)

Storage: JSON files in a configurable directory (default: .proof_sessions/).
Each session is keyed by problem_id.

Usage::

    store = ProofSessionStore()

    # Checkpoint after each round:
    store.checkpoint(session)

    # Resume on startup:
    session = store.load(problem_id)
    if session:
        ctx = session.to_round_context(components)
        pipeline.run_from(ctx)  # resume

    # Clean up after completion:
    store.remove(problem_id)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Session Data Model
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ProofSessionSnapshot:
    """Complete serializable snapshot of a proof session.

    This is the unit of persistence — everything needed to resume a proof task.
    """
    # Identity
    problem_id: str
    problem_name: str = ""
    theorem_statement: str = ""

    # Round state
    round_number: int = 0
    strategy_name: str = "light"
    classification: dict = field(default_factory=dict)

    # Working memory
    attempt_history: list[dict] = field(default_factory=list)
    error_patterns: dict[str, int] = field(default_factory=dict)
    banked_lemmas: list[dict] = field(default_factory=list)
    goal_stack: list[str] = field(default_factory=list)
    total_samples: int = 0
    solved: bool = False

    # State machine
    lane_status: str = "created"
    lane_events_count: int = 0
    lane_recovery_attempts: int = 0
    lane_last_failure: Optional[dict] = None

    # Trace (serialized attempts)
    trace_id: str = ""
    trace_attempts: list[dict] = field(default_factory=list)
    trace_strategy_path: list[str] = field(default_factory=list)
    trace_error_distribution: dict[str, int] = field(default_factory=dict)
    trace_total_tokens: int = 0
    trace_correct_count: int = 0
    trace_config_snapshot: dict = field(default_factory=dict)

    # Knowledge context
    last_feedback_text: str = ""
    negative_knowledge: list[str] = field(default_factory=list)
    broadcast_history: list[dict] = field(default_factory=list)

    # Metadata
    checkpoint_time: float = field(default_factory=time.time)
    checkpoint_round: int = 0
    total_elapsed_ms: int = 0
    version: str = "1.0"

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> ProofSessionSnapshot:
        """Deserialize from dict."""
        # Handle missing fields gracefully (forward-compatible)
        known_fields = {f.name for f in ProofSessionSnapshot.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return ProofSessionSnapshot(**filtered)


# ═══════════════════════════════════════════════════════════════════════════
# Snapshot Builder — extract session state from live objects
# ═══════════════════════════════════════════════════════════════════════════

def build_snapshot(
    problem_id: str,
    problem_name: str,
    theorem_statement: str,
    round_number: int,
    strategy_name: str,
    classification: dict,
    memory: Any,            # WorkingMemory
    sm: Any,                # ProofTaskStateMachine
    trace: Any,             # ProofTrace
    elapsed_ms: int = 0,
    broadcast_bus: Any = None,
) -> ProofSessionSnapshot:
    """Build a snapshot from live pipeline objects.

    This is the canonical way to create a checkpoint — it handles all
    the extraction and serialization of complex objects.
    """
    # Extract working memory
    attempt_history = list(getattr(memory, 'attempt_history', []))
    error_patterns = dict(getattr(memory, 'error_patterns', {}))
    banked_lemmas = list(getattr(memory, 'banked_lemmas', []))
    goal_stack = list(getattr(memory, 'goal_stack', []))

    # Extract state machine
    lane_status = sm.status.value if sm else "created"
    lane_events_count = len(sm.events) if sm else 0
    lane_recovery_attempts = sm.recovery_attempts if sm else 0
    lane_last_failure = None
    if sm and sm.last_failure:
        lane_last_failure = {
            "failure_class": sm.last_failure.failure_class.value,
            "message": sm.last_failure.message,
            "recoverable": sm.last_failure.recoverable,
        }

    # Extract trace
    trace_attempts = []
    if trace:
        for a in trace.attempts:
            if hasattr(a, 'to_dict'):
                trace_attempts.append(a.to_dict())
            elif isinstance(a, dict):
                trace_attempts.append(a)

    # Extract broadcast history
    broadcast_history = []
    if broadcast_bus:
        try:
            recent = broadcast_bus.get_recent(n=30)
            for msg in recent:
                broadcast_history.append({
                    "msg_type": msg.msg_type.value if hasattr(msg.msg_type, 'value') else str(msg.msg_type),
                    "source": msg.source,
                    "content": msg.content[:500],
                    "timestamp": msg.timestamp,
                })
        except Exception:
            logger.debug("Failed to serialize broadcast messages for snapshot",
                         exc_info=True)

    # Extract knowledge context
    last_feedback_text = getattr(memory, 'last_feedback_text', '')
    negative_knowledge = []
    domain_hints = classification.get("domain_hints", {})
    neg = domain_hints.get("negative_knowledge", "")
    if neg:
        negative_knowledge.append(neg)

    return ProofSessionSnapshot(
        problem_id=problem_id,
        problem_name=problem_name,
        theorem_statement=theorem_statement,
        round_number=round_number,
        strategy_name=strategy_name,
        classification=_safe_serialize(classification),
        attempt_history=attempt_history,
        error_patterns=error_patterns,
        banked_lemmas=banked_lemmas,
        goal_stack=goal_stack,
        total_samples=getattr(memory, 'total_samples', 0),
        solved=getattr(memory, 'solved', False),
        lane_status=lane_status,
        lane_events_count=lane_events_count,
        lane_recovery_attempts=lane_recovery_attempts,
        lane_last_failure=lane_last_failure,
        trace_id=getattr(trace, 'trace_id', ''),
        trace_attempts=trace_attempts,
        trace_strategy_path=list(getattr(trace, 'strategy_path', [])),
        trace_error_distribution=dict(getattr(trace, 'error_distribution', {})),
        trace_total_tokens=getattr(trace, 'total_tokens', 0),
        trace_correct_count=getattr(trace, 'correct_count', 0),
        trace_config_snapshot=dict(getattr(trace, 'config_snapshot', {})),
        last_feedback_text=str(last_feedback_text)[:2000],
        negative_knowledge=negative_knowledge,
        broadcast_history=broadcast_history,
        checkpoint_round=round_number,
        total_elapsed_ms=elapsed_ms,
    )


def _safe_serialize(obj: Any) -> Any:
    """Recursively make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    # Fallback: stringify
    return str(obj)


# ═══════════════════════════════════════════════════════════════════════════
# Snapshot Restorer — rebuild live objects from snapshot
# ═══════════════════════════════════════════════════════════════════════════

def restore_round_context(
    snapshot: ProofSessionSnapshot,
    components: Any,  # EngineComponents
) -> dict:
    """Restore a RoundContext-compatible dict from a snapshot.

    Returns a dict that can be used to reconstruct a RoundContext.
    The caller (ProofPipeline) is responsible for wiring components.

    Returns:
        {
            "memory": WorkingMemory,
            "trace": ProofTrace,
            "round_number": int,
            "strategy_name": str,
            "classification": dict,
            "sm": ProofTaskStateMachine,
        }
    """
    from common.working_memory import WorkingMemory
    from prover.models import ProofTrace, BenchmarkProblem

    # Rebuild WorkingMemory
    memory = WorkingMemory(
        problem_id=snapshot.problem_id,
        theorem_statement=snapshot.theorem_statement,
    )
    memory.attempt_history = list(snapshot.attempt_history)
    memory.error_patterns = dict(snapshot.error_patterns)
    memory.banked_lemmas = list(snapshot.banked_lemmas)
    memory.goal_stack = list(snapshot.goal_stack)
    memory.total_samples = snapshot.total_samples
    memory.current_strategy = snapshot.strategy_name
    memory.rounds_completed = snapshot.round_number
    memory.solved = snapshot.solved

    # Restore feedback text
    if snapshot.last_feedback_text:
        memory.last_feedback_text = snapshot.last_feedback_text

    # Rebuild ProofTrace (partial — attempts are serialized)
    trace = ProofTrace(
        trace_id=snapshot.trace_id,
        problem_id=snapshot.problem_id,
        problem_name=snapshot.problem_name,
        theorem_statement=snapshot.theorem_statement,
    )
    trace.strategy_path = list(snapshot.trace_strategy_path)
    trace.error_distribution = dict(snapshot.trace_error_distribution)
    trace.total_tokens = snapshot.trace_total_tokens
    trace.correct_count = snapshot.trace_correct_count
    trace.total_attempts = len(snapshot.trace_attempts)
    trace.config_snapshot = dict(snapshot.trace_config_snapshot)
    trace.solved = snapshot.solved

    # Rebuild state machine
    from engine.lane.task_state import (
        ProofTaskStateMachine, TaskContext, TaskStatus,
    )
    task_ctx = TaskContext(
        theorem_name=snapshot.problem_name,
        formal_statement=snapshot.theorem_statement,
        rounds_completed=snapshot.round_number,
        total_samples=snapshot.total_samples,
        banked_lemmas=[l.get("name", "") if isinstance(l, dict) else str(l)
                       for l in snapshot.banked_lemmas],
    )
    sm = ProofTaskStateMachine(
        task_id=snapshot.problem_id,
        context=task_ctx,
    )
    # Fast-forward state machine to the saved status
    target_status = TaskStatus(snapshot.lane_status)
    if target_status != TaskStatus.CREATED:
        # Use a simplified path to reach the target state
        _fast_forward_sm(sm, target_status)

    return {
        "memory": memory,
        "trace": trace,
        "round_number": snapshot.round_number,
        "strategy_name": snapshot.strategy_name,
        "classification": snapshot.classification,
        "sm": sm,
    }


def _fast_forward_sm(sm, target: 'TaskStatus'):
    """Fast-forward a state machine to the target status.

    Uses the shortest valid transition path. For terminal states,
    goes through GENERATING → target.
    """
    from engine.lane.task_state import TaskStatus

    # Simple path: CREATED → GENERATING → target (or terminal)
    try:
        if sm.status == TaskStatus.CREATED:
            sm.transition_to(TaskStatus.GENERATING,
                             detail="restored from checkpoint")
        if target == TaskStatus.GENERATING:
            return
        if target == TaskStatus.VERIFYING:
            sm.transition_to(TaskStatus.VERIFYING,
                             detail="restored from checkpoint")
        elif target == TaskStatus.SUCCEEDED:
            sm.transition_to(TaskStatus.VERIFYING,
                             detail="restored from checkpoint")
            sm.succeed("restored from checkpoint")
        elif target == TaskStatus.FAILED:
            sm.transition_to(TaskStatus.VERIFYING,
                             detail="restored from checkpoint")
            from engine.lane.task_state import ProofFailureClass
            sm.fail(ProofFailureClass.BUDGET_EXHAUSTED,
                    "restored from checkpoint — was FAILED", recoverable=False)
        elif target == TaskStatus.GIVEN_UP:
            sm.give_up("restored from checkpoint — was GIVEN_UP")
        elif target == TaskStatus.BLOCKED:
            from engine.lane.task_state import ProofFailureClass
            sm.fail(ProofFailureClass.API_ERROR,
                    "restored from checkpoint — was BLOCKED", recoverable=True)
        # REPAIRING, KNOWLEDGE_LOADING → treat as GENERATING
    except ValueError:
        # Invalid transition — leave at current state
        logger.warning(f"Could not fast-forward SM to {target.value}, "
                       f"staying at {sm.status.value}")


# ═══════════════════════════════════════════════════════════════════════════
# Session Store (File-based)
# ═══════════════════════════════════════════════════════════════════════════

class ProofSessionStore:
    """File-based proof session persistence.

    Each problem gets its own JSON checkpoint file. Files are atomic-written
    (write to temp, then rename) to avoid corruption from crashes.

    Usage::

        store = ProofSessionStore()
        store.checkpoint(snapshot)       # save
        snap = store.load("problem_id")  # load
        store.remove("problem_id")       # clean up
        all_ids = store.list_sessions()  # list all checkpointed sessions
    """

    def __init__(self, directory: str | Path = ".proof_sessions"):
        self._dir = Path(directory)

    def checkpoint(self, snapshot: ProofSessionSnapshot) -> Path:
        """Save a checkpoint. Atomic write (temp + rename)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        target = self._path(snapshot.problem_id)
        tmp = target.with_suffix('.tmp')

        try:
            data = snapshot.to_dict()
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                      default=str))
            tmp.rename(target)
            logger.debug(f"Checkpoint saved: {target.name} "
                         f"(round {snapshot.round_number})")
            return target
        except Exception as e:
            logger.warning(f"Checkpoint failed for {snapshot.problem_id}: {e}")
            if tmp.exists():
                tmp.unlink()
            raise

    def load(self, problem_id: str) -> Optional[ProofSessionSnapshot]:
        """Load a checkpoint. Returns None if not found or corrupt."""
        path = self._path(problem_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            snapshot = ProofSessionSnapshot.from_dict(data)
            logger.info(f"Checkpoint loaded: {path.name} "
                        f"(round {snapshot.round_number}, "
                        f"status={snapshot.lane_status})")
            return snapshot
        except Exception as e:
            logger.warning(f"Checkpoint corrupt for {problem_id}: {e}")
            return None

    def remove(self, problem_id: str) -> bool:
        """Remove a checkpoint."""
        path = self._path(problem_id)
        if path.exists():
            path.unlink()
            logger.debug(f"Checkpoint removed: {path.name}")
            return True
        return False

    def list_sessions(self) -> list[str]:
        """List all checkpointed problem IDs."""
        if not self._dir.exists():
            return []
        return [p.stem for p in self._dir.glob("*.json")]

    def has_checkpoint(self, problem_id: str) -> bool:
        """Check if a checkpoint exists."""
        return self._path(problem_id).exists()

    def list_resumable(self) -> list[ProofSessionSnapshot]:
        """List all non-terminal checkpointed sessions (can be resumed)."""
        result = []
        for pid in self.list_sessions():
            snap = self.load(pid)
            if snap and snap.lane_status not in ("succeeded", "failed", "given_up"):
                result.append(snap)
        return result

    def cleanup_completed(self) -> int:
        """Remove checkpoints for completed (terminal) sessions."""
        removed = 0
        for pid in self.list_sessions():
            snap = self.load(pid)
            if snap and snap.lane_status in ("succeeded", "failed", "given_up"):
                self.remove(pid)
                removed += 1
        return removed

    def _path(self, problem_id: str) -> Path:
        """Generate the checkpoint file path for a problem."""
        # Sanitize problem_id for use as filename
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_"
                          for c in problem_id)
        return self._dir / f"{safe_id}.json"
