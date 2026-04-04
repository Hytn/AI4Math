"""prover/repair/repair_strategies.py — 错误修复策略库

根据错误类型选择具体修复策略，支持规则引擎和 LLM 两种模式。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from prover.models import LeanError, ErrorCategory


@dataclass
class RepairStrategy:
    """A concrete repair strategy."""
    name: str
    description: str
    priority: int  # lower = try first
    applicable_errors: list[ErrorCategory]
    tactics_to_try: list[str] = field(default_factory=list)
    code_transforms: list[str] = field(default_factory=list)
    prompt_hint: str = ""


# Built-in repair strategies ordered by priority
STRATEGIES: list[RepairStrategy] = [
    RepairStrategy(
        name="exact_type_cast",
        description="Insert explicit type cast for type mismatch",
        priority=1,
        applicable_errors=[ErrorCategory.TYPE_MISMATCH],
        tactics_to_try=["exact_mod_cast", "push_cast", "norm_cast"],
        code_transforms=["add_cast"],
        prompt_hint="Type mismatch — try adding explicit casts or using norm_cast/push_cast.",
    ),
    RepairStrategy(
        name="fix_identifier",
        description="Fix unknown identifier by searching for correct name",
        priority=1,
        applicable_errors=[ErrorCategory.UNKNOWN_IDENTIFIER],
        tactics_to_try=["exact?", "apply?"],
        prompt_hint="Unknown identifier — use exact? or apply? to find the correct lemma name.",
    ),
    RepairStrategy(
        name="break_into_steps",
        description="Break monolithic tactic into smaller have/suffices steps",
        priority=2,
        applicable_errors=[ErrorCategory.TACTIC_FAILED],
        tactics_to_try=["simp only [...]", "rw [...]", "conv => ..."],
        prompt_hint="Tactic failed — break into smaller have steps or try a different tactic.",
    ),
    RepairStrategy(
        name="fix_syntax",
        description="Fix common syntax errors",
        priority=1,
        applicable_errors=[ErrorCategory.SYNTAX_ERROR],
        code_transforms=["fix_brackets", "fix_commas", "fix_indent"],
        prompt_hint="Syntax error — check brackets, commas, and indentation.",
    ),
    RepairStrategy(
        name="add_import",
        description="Add missing import statement",
        priority=1,
        applicable_errors=[ErrorCategory.IMPORT_ERROR],
        code_transforms=["add_import"],
        prompt_hint="Import error — add the missing import.",
    ),
    RepairStrategy(
        name="simplify_goal",
        description="Reduce elaboration complexity",
        priority=3,
        applicable_errors=[ErrorCategory.TIMEOUT, ErrorCategory.ELABORATION_ERROR],
        tactics_to_try=["simp only", "norm_num"],
        code_transforms=["add_type_annotations", "reduce_implicit_args"],
        prompt_hint="Elaboration timeout — add explicit type annotations and avoid heavy automation.",
    ),
    RepairStrategy(
        name="try_alternative_approach",
        description="Completely different proof approach",
        priority=4,
        applicable_errors=[ErrorCategory.OTHER, ErrorCategory.TACTIC_FAILED],
        prompt_hint="Try a fundamentally different proof strategy.",
    ),
]


def select_strategies(errors: list[LeanError],
                      max_strategies: int = 3) -> list[RepairStrategy]:
    """Select applicable repair strategies based on error types.

    Returns strategies sorted by priority (most promising first).
    """
    if not errors:
        return []

    error_cats = {e.category for e in errors}
    # Primary category = most frequent error
    cat_counts = {}
    for e in errors:
        cat_counts[e.category] = cat_counts.get(e.category, 0) + 1
    primary_cat = max(cat_counts, key=cat_counts.get)

    applicable = []
    for strategy in STRATEGIES:
        # Check if strategy applies to any of the errors
        overlap = set(strategy.applicable_errors) & error_cats
        if overlap:
            # Boost priority if it matches the primary error
            priority = strategy.priority
            if primary_cat in strategy.applicable_errors:
                priority -= 1  # boost
            applicable.append((priority, strategy))

    # Always include the fallback
    fallback = STRATEGIES[-1]  # try_alternative_approach
    if fallback not in [s for _, s in applicable]:
        applicable.append((fallback.priority, fallback))

    applicable.sort(key=lambda x: x[0])
    return [s for _, s in applicable[:max_strategies]]


def build_repair_prompt(errors: list[LeanError], strategies: list[RepairStrategy]) -> str:
    """Build a repair-oriented prompt section from errors and strategies."""
    parts = ["## Error Analysis & Repair Hints\n"]
    for e in errors[:5]:
        loc = f" (line {e.line})" if e.line else ""
        parts.append(f"- [{e.category.value}]{loc}: {e.message[:120]}")
    parts.append("")
    parts.append("## Suggested Strategies")
    for s in strategies:
        parts.append(f"- **{s.name}**: {s.prompt_hint}")
        if s.tactics_to_try:
            parts.append(f"  Tactics to try: {', '.join(s.tactics_to_try)}")
    return "\n".join(parts)
