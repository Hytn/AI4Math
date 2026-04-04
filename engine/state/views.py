"""Agent-optimized state views — token-efficient representations."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class TargetShape(Enum):
    PROP = "prop"; EQUALITY = "equality"; FORALL = "forall"
    EXISTS = "exists"; CONJUNCTION = "conjunction"; DISJUNCTION = "disjunction"
    IMPLICATION = "implication"; APPLICATION = "application"; OTHER = "other"

@dataclass
class GoalView:
    goal_id: int; num_hypotheses: int; target: str
    relevant_hyps: list = field(default_factory=list)
    depth: int = 0; target_head: Optional[str] = None
    target_shape: TargetShape = TargetShape.OTHER
    is_independent: bool = True

    @staticmethod
    def from_goal(goal, state) -> GoalView:
        target_str = repr(goal.target)
        head = goal.target.get_app_fn_name()
        shape = _classify(goal.target)
        hyps = [{"name": str(d.user_name), "type": repr(d.type_), "relevance": 0.5}
                for d in goal.local_ctx]
        unsolved = state.meta_ctx.unsolved()
        indep = all(state.meta_ctx.are_independent(goal.id, o)
                    for o in unsolved if o != goal.id)
        return GoalView(goal.id.id, len(goal.local_ctx), target_str,
                       hyps[:10], goal.depth, str(head) if head else None, shape, indep)

def _classify(e):
    if e.tag == "sort" and e.level and e.level.to_nat() == 0: return TargetShape.PROP
    if e.tag == "pi":
        return TargetShape.FORALL if (e.name and not e.name.is_anon()) else TargetShape.IMPLICATION
    if e.tag == "app" and e.get_app_fn_name():
        h = str(e.get_app_fn_name())
        if "Eq" in h: return TargetShape.EQUALITY
        if "And" in h: return TargetShape.CONJUNCTION
        if "Or" in h: return TargetShape.DISJUNCTION
        return TargetShape.APPLICATION
    return TargetShape.OTHER
