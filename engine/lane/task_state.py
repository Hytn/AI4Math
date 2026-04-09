"""engine/lane/task_state.py — Proof task state machine

Inspired by claw-code's WorkerStatus lifecycle (worker_boot.rs).
Every proof task has an explicit state machine with typed transitions.

States::

    Created → KnowledgeLoading → Generating → Verifying
                                      ↑           ↓
                                      └── Repairing
                                            ↓
                          Succeeded / Failed / GivenUp

Each transition emits a ProofTaskEvent for downstream consumption
(knowledge system, monitoring, policy engine).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    """Explicit proof task lifecycle states."""
    CREATED = "created"
    KNOWLEDGE_LOADING = "knowledge_loading"
    GENERATING = "generating"
    VERIFYING = "verifying"
    REPAIRING = "repairing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    GIVEN_UP = "given_up"
    BLOCKED = "blocked"          # infra issue (REPL crash, API error)

    @property
    def is_terminal(self) -> bool:
        return self in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.GIVEN_UP)


# Valid state transitions
_VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.CREATED: {TaskStatus.KNOWLEDGE_LOADING, TaskStatus.GENERATING},
    TaskStatus.KNOWLEDGE_LOADING: {TaskStatus.GENERATING, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.GENERATING: {TaskStatus.VERIFYING, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.GIVEN_UP},
    TaskStatus.VERIFYING: {
        TaskStatus.SUCCEEDED,
        TaskStatus.REPAIRING,
        TaskStatus.GENERATING,   # back to next round
        TaskStatus.BLOCKED,
        TaskStatus.FAILED,
        TaskStatus.GIVEN_UP,
    },
    TaskStatus.REPAIRING: {TaskStatus.VERIFYING, TaskStatus.GENERATING, TaskStatus.BLOCKED, TaskStatus.GIVEN_UP},
    TaskStatus.BLOCKED: {TaskStatus.GENERATING, TaskStatus.VERIFYING, TaskStatus.REPAIRING, TaskStatus.FAILED},
    # Terminal states allow no transitions
    TaskStatus.SUCCEEDED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.GIVEN_UP: set(),
}


class ProofFailureClass(str, Enum):
    """Typed failure taxonomy — inspired by claw-code's LaneFailureClass."""
    SYNTAX_ERROR = "syntax_error"
    TYPE_MISMATCH = "type_mismatch"
    TACTIC_FAILED = "tactic_failed"
    UNKNOWN_IDENTIFIER = "unknown_identifier"
    TIMEOUT = "timeout"
    IMPORT_ERROR = "import_error"
    SORRY_DETECTED = "sorry_detected"
    API_ERROR = "api_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REPL_CRASH = "repl_crash"
    POOL_EXHAUSTED = "pool_exhausted"
    KNOWLEDGE_ERROR = "knowledge_error"
    INTEGRITY_VIOLATION = "integrity_violation"  # sorry/cheat detected


@dataclass
class TaskFailure:
    """Structured failure record — analogous to claw's WorkerFailure."""
    failure_class: ProofFailureClass
    message: str
    recoverable: bool = True
    detail: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class TaskEvent:
    """Typed event emitted on every state transition.

    Analogous to claw-code's WorkerEvent / LaneEvent.
    """
    seq: int
    event_name: str
    prev_status: TaskStatus
    new_status: TaskStatus
    detail: Optional[str] = None
    failure: Optional[TaskFailure] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskContext:
    """Mutable context bag carried through the task lifecycle."""
    theorem_name: str
    formal_statement: str
    domain: str = ""
    difficulty: str = "unknown"
    # Accumulated state
    rounds_completed: int = 0
    total_samples: int = 0
    total_api_tokens: int = 0
    banked_lemmas: list[str] = field(default_factory=list)
    knowledge_injected: bool = False
    best_attempt_code: str = ""
    best_attempt_errors: list[dict] = field(default_factory=list)


class ProofTaskStateMachine:
    """Explicit state machine for a single proof task.

    Mirrors claw-code's Worker struct with typed events and transitions.

    Usage::

        sm = ProofTaskStateMachine(task_id="minif2f_001",
                                   context=TaskContext(theorem_name="...", ...))
        sm.transition_to(TaskStatus.KNOWLEDGE_LOADING)
        sm.transition_to(TaskStatus.GENERATING)
        sm.transition_to(TaskStatus.VERIFYING, detail="round 1, 8 candidates")
        sm.fail(ProofFailureClass.TYPE_MISMATCH, "expected Nat, got Int")
        sm.transition_to(TaskStatus.REPAIRING)
        ...
        # Query
        sm.status          # current status
        sm.events          # full event log
        sm.last_failure    # most recent failure
        sm.snapshot()      # machine-readable dict
    """

    def __init__(self, task_id: str, context: TaskContext):
        self.task_id = task_id
        self.context = context
        self._status = TaskStatus.CREATED
        self._events: list[TaskEvent] = []
        self._last_failure: Optional[TaskFailure] = None
        self._recovery_attempts: int = 0
        self._created_at = time.time()
        self._updated_at = self._created_at

        # Emit creation event
        self._push_event("task.created", TaskStatus.CREATED, TaskStatus.CREATED,
                         detail=f"task created: {context.theorem_name}")

    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def events(self) -> list[TaskEvent]:
        return list(self._events)

    @property
    def last_failure(self) -> Optional[TaskFailure]:
        return self._last_failure

    @property
    def recovery_attempts(self) -> int:
        return self._recovery_attempts

    def transition_to(self, new_status: TaskStatus, *,
                      detail: str = None, metadata: dict = None) -> TaskEvent:
        """Transition to a new state with validation.

        Raises ValueError if the transition is not allowed.
        """
        if new_status not in _VALID_TRANSITIONS.get(self._status, set()):
            raise ValueError(
                f"Invalid transition: {self._status} → {new_status}. "
                f"Allowed: {_VALID_TRANSITIONS.get(self._status, set())}")

        prev = self._status
        self._status = new_status
        self._updated_at = time.time()

        # Clear failure on successful recovery
        if prev == TaskStatus.BLOCKED and new_status not in (TaskStatus.FAILED,):
            self._last_failure = None
            self._recovery_attempts += 1

        event_name = f"task.{new_status.value}"
        return self._push_event(event_name, prev, new_status,
                                detail=detail, metadata=metadata)

    def fail(self, failure_class: ProofFailureClass, message: str, *,
             recoverable: bool = True, detail: str = None) -> TaskEvent:
        """Record a failure and optionally transition to BLOCKED.

        If recoverable=True, transitions to BLOCKED (awaiting recovery).
        If recoverable=False, transitions to FAILED (terminal).
        """
        failure = TaskFailure(
            failure_class=failure_class,
            message=message,
            recoverable=recoverable,
            detail=detail,
        )
        self._last_failure = failure

        if recoverable and not self._status.is_terminal:
            target = TaskStatus.BLOCKED
        else:
            target = TaskStatus.FAILED

        # Force transition (failures bypass normal validation)
        prev = self._status
        self._status = target
        self._updated_at = time.time()

        return self._push_event(
            f"task.failure.{failure_class.value}",
            prev, target,
            detail=message, failure=failure)

    def give_up(self, reason: str = "budget exhausted") -> TaskEvent:
        """Transition to GIVEN_UP (terminal)."""
        prev = self._status
        self._status = TaskStatus.GIVEN_UP
        self._updated_at = time.time()
        return self._push_event("task.given_up", prev, TaskStatus.GIVEN_UP,
                                detail=reason)

    def succeed(self, proof_code: str = "") -> TaskEvent:
        """Transition to SUCCEEDED (terminal)."""
        prev = self._status
        self._status = TaskStatus.SUCCEEDED
        self._updated_at = time.time()
        self.context.best_attempt_code = proof_code
        return self._push_event("task.succeeded", prev, TaskStatus.SUCCEEDED,
                                detail=f"proof found ({len(proof_code)} chars)")

    def snapshot(self) -> dict:
        """Machine-readable state snapshot — for dashboard/monitoring."""
        return {
            "task_id": self.task_id,
            "theorem_name": self.context.theorem_name,
            "status": self._status.value,
            "rounds_completed": self.context.rounds_completed,
            "total_samples": self.context.total_samples,
            "total_api_tokens": self.context.total_api_tokens,
            "banked_lemmas_count": len(self.context.banked_lemmas),
            "recovery_attempts": self._recovery_attempts,
            "last_failure": {
                "class": self._last_failure.failure_class.value,
                "message": self._last_failure.message,
            } if self._last_failure else None,
            "event_count": len(self._events),
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "elapsed_seconds": self._updated_at - self._created_at,
        }

    def _push_event(self, event_name: str, prev: TaskStatus, new: TaskStatus,
                    detail: str = None, failure: TaskFailure = None,
                    metadata: dict = None) -> TaskEvent:
        seq = len(self._events) + 1
        event = TaskEvent(
            seq=seq,
            event_name=event_name,
            prev_status=prev,
            new_status=new,
            detail=detail,
            failure=failure,
            metadata=metadata or {},
        )
        self._events.append(event)
        return event
