"""engine/lane — Proof Lane Runtime

Claw-code-inspired proof task lifecycle management.

Core components:
  - TaskState:          Explicit state machine with typed events
  - EventBus:           Pub-sub event bus replacing log parsing
  - Recovery:           Failure classification + auto-recovery recipes
  - TaskPacket:         Structured task specifications
  - Policy:             Executable strategy rules
  - Dashboard:          Machine-readable system status
  - ErrorClassifier:    Lean error → ProofFailureClass mapping
  - SummaryCompressor:  Error & context compression for LLM prompts
  - SessionStore:       Checkpoint/resume for proof sessions
"""
from engine.lane.task_state import (
    TaskStatus, ProofFailureClass, TaskFailure, TaskEvent,
    TaskContext, ProofTaskStateMachine,
)
from engine.lane.event_bus import ProofEventBus, get_event_bus, wire_state_machine_to_bus
from engine.lane.recovery import RecoveryAction, RecoveryRecipe, RecoveryRegistry
from engine.lane.task_packet import ProofTaskPacket, validate_packet
from engine.lane.policy import PolicyAction, PolicyDecision, PolicyEngine
from engine.lane.dashboard import ProofDashboard
from engine.lane.error_classifier import classify_lean_error, classify_verification_result
from engine.lane.summary_compressor import (
    compress_lean_errors, compress_feedback, compress_broadcast, compress_for_prompt,
    CompressionBudget, CompressionResult,
)
from engine.lane.proof_session_store import (
    ProofSessionStore, ProofSessionSnapshot, build_snapshot, restore_round_context,
)

__all__ = [
    "TaskStatus", "ProofFailureClass", "TaskFailure", "TaskEvent",
    "TaskContext", "ProofTaskStateMachine",
    "ProofEventBus", "get_event_bus", "wire_state_machine_to_bus",
    "RecoveryAction", "RecoveryRecipe", "RecoveryRegistry",
    "ProofTaskPacket", "validate_packet",
    "PolicyAction", "PolicyDecision", "PolicyEngine",
    "ProofDashboard",
    "classify_lean_error", "classify_verification_result",
    "compress_lean_errors", "compress_feedback", "compress_broadcast",
    "compress_for_prompt", "CompressionBudget", "CompressionResult",
    "ProofSessionStore", "ProofSessionSnapshot",
    "build_snapshot", "restore_round_context",
]
