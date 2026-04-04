"""prover/premise/tactic_suggester.py — 基于规则的 tactic 建议器

根据目标形状和上下文推荐候选 tactic，无需 LLM 调用。
用于 L0 快速过滤阶段。
"""
from __future__ import annotations
import re


# Goal shape → suggested tactics (ordered by priority)
_SHAPE_TACTICS = {
    "forall":       ["intro"],
    "implication":  ["intro"],
    "conjunction":  ["constructor", "exact And.intro"],
    "disjunction":  ["left", "right"],
    "equality":     ["rfl", "ring", "omega", "simp", "rw"],
    "negation":     ["intro", "contradiction"],
    "exists":       ["use", "exact ⟨_, _⟩"],
    "iff":          ["constructor", "intro"],
    "le":           ["omega", "linarith", "norm_num"],
    "lt":           ["omega", "linarith", "norm_num"],
    "nat_expr":     ["omega", "simp", "ring", "induction"],
    "int_expr":     ["omega", "ring", "linarith"],
    "general":      ["simp", "exact?", "apply?", "assumption"],
}


def classify_goal(target: str) -> str:
    """Classify goal target into a shape category."""
    target = target.strip()

    if target.startswith("∀") or target.startswith("(∀"):
        return "forall"
    if "→" in target or "->" in target:
        # Check if it's really an implication (vs a function type)
        return "implication"
    if "∧" in target or "And " in target:
        return "conjunction"
    if "∨" in target or "Or " in target:
        return "disjunction"
    if "↔" in target or "Iff " in target:
        return "iff"
    if "∃" in target or "Exists " in target:
        return "exists"
    if "¬" in target or "Not " in target:
        return "negation"
    if re.search(r'\b\w+\s*=\s*\w+', target):
        return "equality"
    if "≤" in target or "<=" in target or "LE.le" in target:
        return "le"
    if "< " in target or "LT.lt" in target:
        return "lt"
    if "Nat" in target:
        return "nat_expr"
    if "Int" in target:
        return "int_expr"
    return "general"


def suggest_tactics(target: str, hypotheses: list[str] = None,
                    max_suggestions: int = 8) -> list[str]:
    """Suggest tactics based on goal shape and available hypotheses.

    Args:
        target: The goal target as a string (e.g., "a + b = b + a").
        hypotheses: List of hypothesis types in context.
        max_suggestions: Max number of tactics to return.

    Returns:
        Ordered list of tactic strings.
    """
    shape = classify_goal(target)
    base_tactics = list(_SHAPE_TACTICS.get(shape, _SHAPE_TACTICS["general"]))
    hyps = hypotheses or []

    suggestions = []

    # Shape-specific tactics first
    suggestions.extend(base_tactics)

    # If hypotheses match the goal, suggest assumption/exact
    for h in hyps:
        if _type_matches(h, target):
            suggestions.insert(0, "assumption")
            break

    # If any hypothesis is a function type matching conclusion, suggest apply
    for h in hyps:
        if "→" in h or "->" in h:
            parts = re.split(r"→|->", h)
            conclusion = parts[-1].strip()
            if _fuzzy_match(conclusion, target):
                suggestions.insert(1, "apply")
                break

    # Always include fallbacks
    for fallback in ["simp", "assumption", "trivial"]:
        if fallback not in suggestions:
            suggestions.append(fallback)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for t in suggestions:
        if t not in seen:
            seen.add(t)
            result.append(t)

    return result[:max_suggestions]


def _type_matches(hyp_type: str, target: str) -> bool:
    """Check if a hypothesis type approximately matches the target."""
    h = hyp_type.strip().split(":")[-1].strip() if ":" in hyp_type else hyp_type.strip()
    t = target.strip()
    return h == t or _normalize(h) == _normalize(t)


def _fuzzy_match(a: str, b: str) -> bool:
    """Fuzzy structural match between two type strings."""
    return _normalize(a) == _normalize(b)


def _normalize(s: str) -> str:
    """Normalize a type string for comparison."""
    s = re.sub(r'\s+', ' ', s.strip())
    s = re.sub(r'[(){}]', '', s)
    return s.lower()
