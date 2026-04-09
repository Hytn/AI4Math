"""agent/strategy/meta_controller.py — 元控制器: PolicyEngine 的向后兼容包装

旧接口::

    mc = MetaController(config)
    strategy = mc.select_initial_strategy(difficulty)
    escalation = mc.should_escalate(memory)
    give_up = mc.should_give_up(memory, budget)

新实现: 上述三个方法仍然可用，但内部全部委托给 PolicyEngine 的
可组合规则链，消除硬编码阈值。

直接使用 PolicyEngine 的新接口（推荐）::

    mc = MetaController(config)
    decision = mc.evaluate(sm)
    # decision.action: CONTINUE / ESCALATE_STRATEGY / GIVE_UP / ...

迁移路径:
  1. 新代码直接使用 mc.evaluate(sm) 或 mc.policy_engine
  2. 旧代码继续调用 select_initial_strategy / should_escalate / should_give_up
  3. 最终移除旧接口
"""
from __future__ import annotations

import logging
from typing import Optional

from common.working_memory import WorkingMemory
from engine.lane.policy import (
    PolicyEngine, PolicyAction, PolicyDecision, PolicyRule,
    ConsecutiveSameErrorRule, BudgetEscalationRule,
    BankedLemmaDecomposeRule, ReflectionRule, InfraRecoveryRule,
)
from engine.lane.task_state import (
    ProofTaskStateMachine, TaskStatus, TaskContext,
    ProofFailureClass,
)

logger = logging.getLogger(__name__)


# ─── Additional policy rules derived from old MetaController logic ──────────

class InitialStrategyRule(PolicyRule):
    """Select the initial strategy based on difficulty.

    Replaces the old ``select_initial_strategy()`` hardcoded if-else.
    This rule only fires when the task is in CREATED or first GENERATING.
    """
    name = "initial_strategy"
    priority = 90  # low priority — runs after recovery/escalation

    DIFFICULTY_MAP = {
        "trivial": "sequential",
        "easy": "sequential",
        "medium": "light",
        "hard": "medium",
        "competition": "heavy",
    }

    def evaluate(self, sm, events):
        # Only fire at the start (round 0)
        if sm.context.rounds_completed > 0:
            return None
        difficulty = sm.context.difficulty
        suggested = self.DIFFICULTY_MAP.get(difficulty, "light")
        current = sm.context.__dict__.get("_current_strategy", "light")
        if suggested != current:
            return PolicyDecision(
                action=PolicyAction.ESCALATE_STRATEGY,
                reason=f"difficulty={difficulty} → start with {suggested}",
                rule_name=self.name,
                metadata={"to": suggested},
            )
        return None


class MaxRoundsGiveUpRule(PolicyRule):
    """Give up if too many rounds for the current strategy level.

    Replaces the old hardcoded max_light_rounds / max_medium_rounds
    thresholds in MetaController.should_escalate() and should_give_up().
    """
    name = "max_rounds_give_up"
    priority = 80

    def __init__(self, max_heavy_rounds: int = 8):
        self.max_heavy_rounds = max_heavy_rounds

    def evaluate(self, sm, events):
        ctx = sm.context
        current = ctx.__dict__.get("_current_strategy", "light")
        if (current == "heavy"
                and ctx.rounds_completed >= self.max_heavy_rounds):
            return PolicyDecision(
                action=PolicyAction.GIVE_UP,
                reason=f"Heavy strategy exhausted after "
                       f"{ctx.rounds_completed} rounds",
                rule_name=self.name,
            )
        return None


# ─── MetaController class ───────────────────────────────────────────────────

class MetaController:
    """元控制器: 策略选择与升级的统一接口。

    内部使用 PolicyEngine，外部保持旧 API 的向后兼容。

    Config keys:
        max_light_rounds (int): Rounds before escalating light → medium (default: 2)
        max_medium_rounds (int): Rounds before escalating medium → heavy (default: 4)
        max_heavy_rounds (int): Rounds before giving up in heavy (default: 8)
        max_samples (int): Global sample budget (default: 128)
        consecutive_error_threshold (int): Same-error count before switch (default: 5)
        reflection_every_n (int): Reflect every N rounds (default: 3)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._policy = self._build_policy_engine()

    @property
    def policy_engine(self) -> PolicyEngine:
        """Direct access to the underlying PolicyEngine (recommended)."""
        return self._policy

    def _build_policy_engine(self) -> PolicyEngine:
        """Build a PolicyEngine with all rules derived from config."""
        engine = PolicyEngine()

        # Highest priority: infra recovery
        engine.add_rule(InfraRecoveryRule())

        # Consecutive same-error detection
        engine.add_rule(ConsecutiveSameErrorRule(
            threshold=self.config.get("consecutive_error_threshold", 5)))

        # Budget-based escalation (light→medium→heavy)
        light_ratio = self.config.get("max_light_rounds", 2)
        medium_ratio = self.config.get("max_medium_rounds", 4)
        # Convert round counts to budget ratios
        max_samples = self.config.get("max_samples", 128)
        # Approximate: each round generates ~4 samples
        approx_samples_per_round = 4
        if max_samples > 0:
            light_budget_ratio = min(
                0.9, (light_ratio * approx_samples_per_round) / max_samples)
            medium_budget_ratio = min(
                0.95, (medium_ratio * approx_samples_per_round) / max_samples)
        else:
            light_budget_ratio = 0.3
            medium_budget_ratio = 0.7

        engine.add_rule(BudgetEscalationRule(
            light_budget_ratio=light_budget_ratio,
            medium_budget_ratio=medium_budget_ratio,
        ))

        # Banked lemma decomposition
        engine.add_rule(BankedLemmaDecomposeRule())

        # Periodic reflection
        engine.add_rule(ReflectionRule(
            every_n_rounds=self.config.get("reflection_every_n", 3)))

        # Initial strategy selection
        engine.add_rule(InitialStrategyRule())

        # Max rounds give-up
        engine.add_rule(MaxRoundsGiveUpRule(
            max_heavy_rounds=self.config.get("max_heavy_rounds", 8)))

        return engine

    # ─── New API (recommended) ───────────────────────────────────────

    def evaluate(self, sm: ProofTaskStateMachine) -> PolicyDecision:
        """Evaluate all policy rules and return the decision.

        This is the recommended API — use this instead of the legacy
        methods below.
        """
        return self._policy.evaluate(sm)

    def add_rule(self, rule: PolicyRule):
        """Add a custom policy rule."""
        self._policy.add_rule(rule)

    # ─── Legacy API (backward-compatible) ────────────────────────────

    def select_initial_strategy(self, difficulty: str) -> str:
        """Legacy: select initial strategy based on difficulty.

        .. deprecated:: Use ``evaluate(sm)`` with InitialStrategyRule.
        """
        return InitialStrategyRule.DIFFICULTY_MAP.get(difficulty, "light")

    def should_escalate(self, memory: WorkingMemory) -> Optional[str]:
        """Legacy: check if strategy should be escalated.

        .. deprecated:: Use ``evaluate(sm)`` with BudgetEscalationRule.

        Returns the new strategy name if escalation is needed, None otherwise.
        """
        # Build a minimal state machine for policy evaluation
        sm = self._memory_to_sm(memory)
        decision = self._policy.evaluate(sm)

        if decision.action == PolicyAction.ESCALATE_STRATEGY:
            new_strategy = decision.metadata.get("to", "medium")
            return new_strategy
        return None

    def should_give_up(self, memory: WorkingMemory,
                       budget: dict = None) -> bool:
        """Legacy: check if we should give up.

        .. deprecated:: Use ``evaluate(sm)`` with MaxRoundsGiveUpRule.
        """
        budget = budget or {}
        max_samples = budget.get("max_samples",
                                 self.config.get("max_samples", 128))
        if memory.total_samples >= max_samples:
            return True

        sm = self._memory_to_sm(memory, max_samples=max_samples)
        decision = self._policy.evaluate(sm)
        return decision.action == PolicyAction.GIVE_UP

    def _memory_to_sm(
        self,
        memory: WorkingMemory,
        max_samples: int = 128,
    ) -> ProofTaskStateMachine:
        """Bridge: convert WorkingMemory to a minimal state machine
        for policy evaluation."""
        ctx = TaskContext(
            theorem_name=memory.problem_id or "",
            formal_statement=memory.theorem_statement or "",
            rounds_completed=memory.rounds_completed,
            total_samples=memory.total_samples,
            banked_lemmas=[
                l.get("name", "") for l in memory.banked_lemmas
            ] if memory.banked_lemmas else [],
        )
        ctx.__dict__["_max_samples"] = max_samples
        ctx.__dict__["_current_strategy"] = memory.current_strategy

        sm = ProofTaskStateMachine(
            task_id=f"compat_{memory.problem_id or 'unknown'}",
            context=ctx,
        )

        # Simulate state based on memory
        if memory.solved:
            sm._status = TaskStatus.SUCCEEDED
        elif memory.total_samples > 0:
            sm._status = TaskStatus.VERIFYING

        return sm
