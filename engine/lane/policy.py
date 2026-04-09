"""engine/lane/policy.py — Executable proof strategy rules

Inspired by claw-code's policy engine concept (ROADMAP Phase 4, item 11):
"doctrine moves from chat instructions into executable rules."

Replaces AI4Math's hardcoded thresholds in meta_controller.py and
confidence_estimator.py with composable, inspectable rules.

Usage::

    engine = PolicyEngine()
    engine.add_rule(ConsecutiveSameErrorRule(threshold=5))
    engine.add_rule(BudgetEscalationRule())

    action = engine.evaluate(task_sm, events)
    # → PolicyAction.SWITCH_ROLE / ESCALATE_STRATEGY / CONTINUE / ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from engine.lane.task_state import (
    ProofTaskStateMachine, TaskEvent, TaskStatus, ProofFailureClass
)
from engine.lane.recovery import RecoveryRegistry, RecoveryAction


class PolicyAction(str, Enum):
    """Actions the policy engine can recommend."""
    CONTINUE = "continue"
    SWITCH_ROLE = "switch_role"
    ESCALATE_STRATEGY = "escalate_strategy"
    SWITCH_MODEL = "switch_model"
    TRY_DECOMPOSE = "try_decompose"
    TRY_CONJECTURE = "try_conjecture"
    INJECT_REFLECTION = "inject_reflection"
    AUTO_RECOVER = "auto_recover"
    GIVE_UP = "give_up"
    ESCALATE_TO_HUMAN = "escalate_to_human"


@dataclass
class PolicyDecision:
    """Result of policy evaluation."""
    action: PolicyAction
    reason: str
    rule_name: str = ""
    metadata: dict = field(default_factory=dict)


class PolicyRule(ABC):
    """A single executable policy rule."""
    name: str = "unnamed_rule"
    priority: int = 100  # lower = evaluated first

    @abstractmethod
    def evaluate(self, sm: ProofTaskStateMachine,
                 events: list[TaskEvent]) -> Optional[PolicyDecision]:
        """Return a PolicyDecision if this rule fires, None otherwise."""
        ...


# ─── Built-in rules ─────────────────────────────────────────────────────────

class ConsecutiveSameErrorRule(PolicyRule):
    """If the same failure class occurs N times in a row, switch role."""
    name = "consecutive_same_error"
    priority = 10

    def __init__(self, threshold: int = 5):
        self.threshold = threshold

    def evaluate(self, sm, events):
        failure_events = [e for e in events if e.failure is not None]
        if len(failure_events) < self.threshold:
            return None
        recent = failure_events[-self.threshold:]
        classes = {e.failure.failure_class for e in recent}
        if len(classes) == 1:
            cls = recent[0].failure.failure_class
            return PolicyDecision(
                action=PolicyAction.SWITCH_ROLE,
                reason=f"{self.threshold} consecutive {cls.value} errors — switch role",
                rule_name=self.name,
                metadata={"failure_class": cls.value, "count": self.threshold},
            )
        return None


class BudgetEscalationRule(PolicyRule):
    """Escalate strategy at budget milestones."""
    name = "budget_escalation"
    priority = 20

    def __init__(self, light_budget_ratio: float = 0.3,
                 medium_budget_ratio: float = 0.7):
        self.light_ratio = light_budget_ratio
        self.medium_ratio = medium_budget_ratio

    def evaluate(self, sm, events):
        ctx = sm.context
        if ctx.total_samples <= 0:
            return None
        # Infer current strategy from metadata (stored by caller)
        current_strategy = sm.context.__dict__.get("_current_strategy", "light")
        # Budget ratio heuristic based on total_samples vs a reasonable max
        max_samples = sm.context.__dict__.get("_max_samples", 128)
        ratio = ctx.total_samples / max_samples

        if current_strategy == "light" and ratio > self.light_ratio:
            return PolicyDecision(
                action=PolicyAction.ESCALATE_STRATEGY,
                reason=f"Budget {ratio:.0%} used in light mode — escalate to medium",
                rule_name=self.name,
                metadata={"from": "light", "to": "medium", "ratio": ratio},
            )
        if current_strategy == "medium" and ratio > self.medium_ratio:
            return PolicyDecision(
                action=PolicyAction.ESCALATE_STRATEGY,
                reason=f"Budget {ratio:.0%} used in medium mode — escalate to heavy",
                rule_name=self.name,
                metadata={"from": "medium", "to": "heavy", "ratio": ratio},
            )
        return None


class BankedLemmaDecomposeRule(PolicyRule):
    """If we have banked lemmas and are stuck, try decompose."""
    name = "banked_lemma_decompose"
    priority = 30

    def __init__(self, min_lemmas: int = 1, min_rounds: int = 3):
        self.min_lemmas = min_lemmas
        self.min_rounds = min_rounds

    def evaluate(self, sm, events):
        ctx = sm.context
        if (len(ctx.banked_lemmas) >= self.min_lemmas
                and ctx.rounds_completed >= self.min_rounds
                and not ctx.__dict__.get("_decompose_attempted", False)):
            return PolicyDecision(
                action=PolicyAction.TRY_DECOMPOSE,
                reason=f"{len(ctx.banked_lemmas)} lemmas banked, "
                       f"{ctx.rounds_completed} rounds — try decomposition",
                rule_name=self.name,
            )
        return None


class ReflectionRule(PolicyRule):
    """Inject reflection after every N rounds."""
    name = "periodic_reflection"
    priority = 50

    def __init__(self, every_n_rounds: int = 3):
        self.every_n = every_n_rounds

    def evaluate(self, sm, events):
        ctx = sm.context
        if ctx.rounds_completed > 0 and ctx.rounds_completed % self.every_n == 0:
            return PolicyDecision(
                action=PolicyAction.INJECT_REFLECTION,
                reason=f"Round {ctx.rounds_completed} — periodic reflection",
                rule_name=self.name,
            )
        return None


class InfraRecoveryRule(PolicyRule):
    """If task is BLOCKED, check recovery registry."""
    name = "infra_recovery"
    priority = 1  # highest priority

    def __init__(self, recovery_registry: RecoveryRegistry = None):
        self.registry = recovery_registry or RecoveryRegistry()

    def evaluate(self, sm, events):
        if sm.status != TaskStatus.BLOCKED or sm.last_failure is None:
            return None
        fc = sm.last_failure.failure_class
        if self.registry.should_recover(fc, sm.recovery_attempts):
            action_map = {
                RecoveryAction.RESTART_REPL: PolicyAction.AUTO_RECOVER,
                RecoveryAction.RETRY_WITH_BACKOFF: PolicyAction.AUTO_RECOVER,
                RecoveryAction.RETRY_LARGER_TIMEOUT: PolicyAction.AUTO_RECOVER,
                RecoveryAction.REDUCE_CONCURRENCY: PolicyAction.AUTO_RECOVER,
                RecoveryAction.SWITCH_ROLE: PolicyAction.SWITCH_ROLE,
                RecoveryAction.SWITCH_STRATEGY: PolicyAction.ESCALATE_STRATEGY,
                RecoveryAction.INJECT_NEGATIVE_KNOWLEDGE: PolicyAction.CONTINUE,
                RecoveryAction.SKIP_AND_CONTINUE: PolicyAction.CONTINUE,
            }
            ra = self.registry.get_action(fc)
            pa = action_map.get(ra, PolicyAction.GIVE_UP)
            return PolicyDecision(
                action=pa,
                reason=f"Auto-recovery for {fc.value}: {ra.value}",
                rule_name=self.name,
                metadata={"failure_class": fc.value, "recovery_action": ra.value},
            )
        return PolicyDecision(
            action=PolicyAction.GIVE_UP,
            reason=f"No recovery available for {fc.value} "
                   f"(attempts: {sm.recovery_attempts})",
            rule_name=self.name,
        )


# ─── Engine ──────────────────────────────────────────────────────────────────

class PolicyEngine:
    """Evaluates proof strategy rules and returns the highest-priority action.

    Inspired by claw-code's policy engine concept.
    """

    def __init__(self):
        self._rules: list[PolicyRule] = []

    def add_rule(self, rule: PolicyRule):
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)

    def evaluate(self, sm: ProofTaskStateMachine,
                 events: list[TaskEvent] = None) -> PolicyDecision:
        """Evaluate all rules in priority order. First match wins."""
        events = events or sm.events
        for rule in self._rules:
            try:
                decision = rule.evaluate(sm, events)
                if decision is not None:
                    return decision
            except Exception:
                logger.warning(
                    f"Policy rule {rule.__class__.__name__} raised an exception",
                    exc_info=True)
                continue  # rule errors don't block evaluation

        return PolicyDecision(
            action=PolicyAction.CONTINUE,
            reason="No rule fired — continue current strategy",
            rule_name="default",
        )

    @staticmethod
    def default() -> PolicyEngine:
        """Create a PolicyEngine with all built-in rules."""
        engine = PolicyEngine()
        engine.add_rule(InfraRecoveryRule())
        engine.add_rule(ConsecutiveSameErrorRule())
        engine.add_rule(BudgetEscalationRule())
        engine.add_rule(BankedLemmaDecomposeRule())
        engine.add_rule(ReflectionRule())
        return engine
