"""engine/lane/green_contract.py — Proof verification grading levels

Inspired by claw-code's GreenContract (green_contract.rs):
    TargetedTests → Package → Workspace → MergeReady

AI4Math adaptation — maps to three-level verification pipeline:

    SyntaxClean   — L0 prefilter passed (no obvious syntax errors)
    TacticValid   — L1 REPL accepted individual tactics
    GoalsClosed   — L1 REPL shows 0 remaining goals
    FullCompile   — L2 full `lake env lean` compile passes
    SorryFree     — L2 passes AND no sorry/admit detected

Each VerificationResult carries a GreenLevel. PolicyEngine rules and
the Dashboard can branch on the level instead of raw booleans.

Usage::

    level = GreenLevel.from_verification_result(vr)
    contract = ProofGreenContract(required=GreenLevel.GOALS_CLOSED)
    outcome = contract.evaluate(level)
    if outcome.satisfied:
        # ready to deposit as proved
"""
from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class GreenLevel(IntEnum):
    """Ordered proof verification levels.

    Higher numeric value = stronger guarantee.
    Uses IntEnum so that level comparisons work naturally:
        GreenLevel.GOALS_CLOSED >= GreenLevel.SYNTAX_CLEAN  # True
    """
    NONE = 0           # nothing verified yet or L0 failed
    SYNTAX_CLEAN = 1   # L0 prefilter passed
    TACTIC_VALID = 2   # L1 REPL accepted (goals may remain)
    GOALS_CLOSED = 3   # L1 REPL shows 0 remaining goals
    FULL_COMPILE = 4   # L2 full compilation passed
    SORRY_FREE = 5     # L2 passed + no sorry/admit

    @classmethod
    def from_verification_result(cls, vr) -> GreenLevel:
        """Extract GreenLevel from a VerificationResult.

        Works with both sync VerificationResult and any object with
        the same fields (duck typing).
        """
        if not getattr(vr, 'l0_passed', False):
            return cls.NONE

        level_reached = getattr(vr, 'level_reached', 'L0')
        success = getattr(vr, 'success', False)

        if level_reached == 'L0':
            return cls.SYNTAX_CLEAN

        if level_reached == 'L1':
            if not success:
                return cls.SYNTAX_CLEAN
            remaining = getattr(vr, 'l1_goals_remaining', None)
            if remaining is not None and len(remaining) == 0:
                return cls.GOALS_CLOSED
            return cls.TACTIC_VALID

        if level_reached == 'L2':
            if not success:
                return cls.SYNTAX_CLEAN
            l2_verified = getattr(vr, 'l2_verified', False)
            if l2_verified:
                has_sorry = getattr(vr, 'has_sorry', True)
                if not has_sorry:
                    return cls.SORRY_FREE
                return cls.FULL_COMPILE
            return cls.GOALS_CLOSED

        return cls.SYNTAX_CLEAN

    @property
    def label(self) -> str:
        """Human-readable label."""
        return {
            GreenLevel.NONE: "⬜ none",
            GreenLevel.SYNTAX_CLEAN: "🟡 syntax_clean",
            GreenLevel.TACTIC_VALID: "🟠 tactic_valid",
            GreenLevel.GOALS_CLOSED: "🔵 goals_closed",
            GreenLevel.FULL_COMPILE: "🟢 full_compile",
            GreenLevel.SORRY_FREE: "✅ sorry_free",
        }.get(self, "? unknown")

    @property
    def short(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class GreenContractOutcome:
    """Result of evaluating a GreenContract."""
    satisfied: bool
    required: GreenLevel
    observed: GreenLevel
    gap: int  # how many levels short (0 = satisfied)

    @property
    def summary(self) -> str:
        if self.satisfied:
            return f"✅ {self.observed.label} >= {self.required.label}"
        return (f"❌ {self.observed.label} < {self.required.label} "
                f"(gap={self.gap})")


@dataclass(frozen=True)
class ProofGreenContract:
    """A contract specifying the minimum verification level required.

    Inspired by claw-code's GreenContract — the policy engine and
    dashboard use this to decide whether a proof is "green enough"
    for the current phase.

    Usage::

        contract = ProofGreenContract(required=GreenLevel.GOALS_CLOSED)
        outcome = contract.evaluate(GreenLevel.SYNTAX_CLEAN)
        outcome.satisfied  # False
        outcome.gap        # 2

        # Default contracts for common scenarios
        ProofGreenContract.for_deposit()      # GOALS_CLOSED
        ProofGreenContract.for_certification() # SORRY_FREE
    """
    required: GreenLevel

    def evaluate(self, observed: GreenLevel) -> GreenContractOutcome:
        """Check whether the observed level satisfies the contract."""
        gap = max(0, self.required - observed)
        return GreenContractOutcome(
            satisfied=observed >= self.required,
            required=self.required,
            observed=observed,
            gap=gap,
        )

    def is_satisfied_by(self, observed: GreenLevel) -> bool:
        """Quick check without creating outcome object."""
        return observed >= self.required

    # ── Factory methods for common contracts ──

    @staticmethod
    def for_quick_filter() -> ProofGreenContract:
        """L0 passed — worth sending to REPL."""
        return ProofGreenContract(required=GreenLevel.SYNTAX_CLEAN)

    @staticmethod
    def for_candidate() -> ProofGreenContract:
        """L1 accepted — valid proof candidate."""
        return ProofGreenContract(required=GreenLevel.TACTIC_VALID)

    @staticmethod
    def for_deposit() -> ProofGreenContract:
        """Goals closed — safe to deposit as proved in knowledge system."""
        return ProofGreenContract(required=GreenLevel.GOALS_CLOSED)

    @staticmethod
    def for_certification() -> ProofGreenContract:
        """L2 sorry-free — certified correct, publication-ready."""
        return ProofGreenContract(required=GreenLevel.SORRY_FREE)
