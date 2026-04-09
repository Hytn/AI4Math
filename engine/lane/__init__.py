"""engine/lane — Proof Lane Runtime

Claw-code-inspired proof task lifecycle management.

Core components:
  - TaskState:     Explicit state machine with typed events
  - EventBus:      Pub-sub event bus replacing log parsing
  - Recovery:      Failure classification + auto-recovery recipes
  - TaskPacket:    Structured task specifications
  - Policy:        Executable strategy rules
  - Dashboard:     Machine-readable system status
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

__all__ = [
    "TaskStatus", "ProofFailureClass", "TaskFailure", "TaskEvent",
    "TaskContext", "ProofTaskStateMachine",
    "ProofEventBus", "get_event_bus", "wire_state_machine_to_bus",
    "RecoveryAction", "RecoveryRecipe", "RecoveryRegistry",
    "ProofTaskPacket", "validate_packet",
    "PolicyAction", "PolicyDecision", "PolicyEngine",
    "ProofDashboard",
]
