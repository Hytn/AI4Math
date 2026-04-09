"""engine/lane/recovery.py — Failure auto-recovery recipes

Inspired by claw-code's recovery philosophy (ROADMAP Phase 3):
"one automatic recovery attempt occurs before escalation."

Each ProofFailureClass maps to a RecoveryRecipe that defines
what to try before giving up. Recovery attempts are themselves
emitted as structured events.

Usage::

    registry = RecoveryRegistry()
    recipe = registry.get(ProofFailureClass.REPL_CRASH)
    if recipe and recipe.attempts_remaining(task_sm):
        action = recipe.action  # "restart_repl_session"
        # ... execute recovery ...
        task_sm.transition_to(TaskStatus.GENERATING)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from engine.lane.task_state import ProofFailureClass


class RecoveryAction(str, Enum):
    """Concrete recovery action to take."""
    RESTART_REPL = "restart_repl_session"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    RETRY_LARGER_TIMEOUT = "retry_larger_timeout"
    SWITCH_ROLE = "switch_agent_role"
    SWITCH_STRATEGY = "escalate_strategy"
    SWITCH_MODEL = "switch_model_tier"
    INJECT_NEGATIVE_KNOWLEDGE = "inject_negative_knowledge"
    REDUCE_CONCURRENCY = "reduce_concurrency"
    SKIP_AND_CONTINUE = "skip_and_continue"
    NO_RECOVERY = "no_recovery"


@dataclass
class RecoveryRecipe:
    """A single recovery recipe for a failure class."""
    action: RecoveryAction
    max_attempts: int = 1
    backoff_seconds: float = 1.0
    timeout_multiplier: float = 1.0
    description: str = ""

    def attempts_remaining(self, current_recovery_count: int) -> bool:
        return current_recovery_count < self.max_attempts


# ─── Default recipes ────────────────────────────────────────────────────────

_DEFAULT_RECIPES: dict[ProofFailureClass, RecoveryRecipe] = {
    ProofFailureClass.REPL_CRASH: RecoveryRecipe(
        action=RecoveryAction.RESTART_REPL,
        max_attempts=2,
        backoff_seconds=1.0,
        description="Restart crashed REPL session, retry from last checkpoint",
    ),
    ProofFailureClass.API_ERROR: RecoveryRecipe(
        action=RecoveryAction.RETRY_WITH_BACKOFF,
        max_attempts=3,
        backoff_seconds=2.0,
        description="Retry LLM API call with exponential backoff",
    ),
    ProofFailureClass.TIMEOUT: RecoveryRecipe(
        action=RecoveryAction.RETRY_LARGER_TIMEOUT,
        max_attempts=1,
        timeout_multiplier=2.0,
        description="Retry with doubled timeout",
    ),
    ProofFailureClass.POOL_EXHAUSTED: RecoveryRecipe(
        action=RecoveryAction.REDUCE_CONCURRENCY,
        max_attempts=1,
        description="Reduce parallel workers, retry sequentially",
    ),
    ProofFailureClass.TACTIC_FAILED: RecoveryRecipe(
        action=RecoveryAction.SWITCH_ROLE,
        max_attempts=2,
        description="Switch agent role after repeated tactic failures",
    ),
    ProofFailureClass.TYPE_MISMATCH: RecoveryRecipe(
        action=RecoveryAction.INJECT_NEGATIVE_KNOWLEDGE,
        max_attempts=3,
        description="Inject type mismatch details as negative knowledge, retry generation",
    ),
    ProofFailureClass.UNKNOWN_IDENTIFIER: RecoveryRecipe(
        action=RecoveryAction.INJECT_NEGATIVE_KNOWLEDGE,
        max_attempts=2,
        description="Record unknown identifier, suggest import or name fix",
    ),
    ProofFailureClass.SYNTAX_ERROR: RecoveryRecipe(
        action=RecoveryAction.INJECT_NEGATIVE_KNOWLEDGE,
        max_attempts=1,
        description="L0 syntax fix is cheap, inject feedback and retry once",
    ),
    ProofFailureClass.IMPORT_ERROR: RecoveryRecipe(
        action=RecoveryAction.INJECT_NEGATIVE_KNOWLEDGE,
        max_attempts=1,
        description="Fix import and retry",
    ),
    ProofFailureClass.SORRY_DETECTED: RecoveryRecipe(
        action=RecoveryAction.SWITCH_ROLE,
        max_attempts=1,
        description="Switch to sorry-closer role",
    ),
    ProofFailureClass.INTEGRITY_VIOLATION: RecoveryRecipe(
        action=RecoveryAction.NO_RECOVERY,
        max_attempts=0,
        description="Integrity violation is terminal — reject proof",
    ),
    ProofFailureClass.BUDGET_EXHAUSTED: RecoveryRecipe(
        action=RecoveryAction.NO_RECOVERY,
        max_attempts=0,
        description="Budget exhausted is terminal — give up",
    ),
    ProofFailureClass.KNOWLEDGE_ERROR: RecoveryRecipe(
        action=RecoveryAction.SKIP_AND_CONTINUE,
        max_attempts=1,
        description="Skip knowledge injection, proceed with base prompt",
    ),
}


class RecoveryRegistry:
    """Registry of recovery recipes, extensible at runtime.

    Usage::

        registry = RecoveryRegistry()
        recipe = registry.get(ProofFailureClass.REPL_CRASH)
        if recipe and recipe.attempts_remaining(sm.recovery_attempts):
            execute_recovery(recipe.action)
    """

    def __init__(self):
        self._recipes = dict(_DEFAULT_RECIPES)

    def get(self, failure_class: ProofFailureClass) -> Optional[RecoveryRecipe]:
        return self._recipes.get(failure_class)

    def register(self, failure_class: ProofFailureClass, recipe: RecoveryRecipe):
        self._recipes[failure_class] = recipe

    def should_recover(self, failure_class: ProofFailureClass,
                       current_recovery_count: int) -> bool:
        recipe = self.get(failure_class)
        if recipe is None:
            return False
        return recipe.attempts_remaining(current_recovery_count)

    def get_action(self, failure_class: ProofFailureClass) -> RecoveryAction:
        recipe = self.get(failure_class)
        return recipe.action if recipe else RecoveryAction.NO_RECOVERY
