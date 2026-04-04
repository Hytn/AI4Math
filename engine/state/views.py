"""Agent-optimized state views — token-efficient representations."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TargetShape(Enum):
    PROP = "prop"
    EQUALITY = "equality"
    FORALL = "forall"
    EXISTS = "exists"
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"
    IMPLICATION = "implication"
    APPLICATION = "application"
    OTHER = "other"


@dataclass
class GoalView:
    """Token-efficient representation of a proof goal for LLM consumption."""
    goal_id: int
    num_hypotheses: int
    target: str
    relevant_hyps: list = field(default_factory=list)
    depth: int = 0
    target_head: Optional[str] = None
    target_shape: TargetShape = TargetShape.OTHER
    is_independent: bool = True

    @staticmethod
    def from_goal(goal, state) -> GoalView:
        target_str = repr(goal.target)
        head = goal.target.get_app_fn_name()
        shape = _classify(goal.target)
        hyps = [
            {"name": str(d.user_name), "type": repr(d.type_), "relevance": 0.5}
            for d in goal.local_ctx
        ]
        unsolved = state.meta_ctx.unsolved()
        indep = all(
            state.meta_ctx.are_independent(goal.id, o)
            for o in unsolved if o != goal.id)
        return GoalView(
            goal.id.id, len(goal.local_ctx), target_str,
            hyps[:10], goal.depth,
            str(head) if head else None, shape, indep)

    def to_prompt(self) -> str:
        """Format this goal view for LLM prompt injection.

        Returns a concise, structured representation of the goal state
        suitable for including in a proof generation prompt.
        """
        parts = [f"⊢ {self.target}"]
        if self.relevant_hyps:
            hyp_lines = []
            for h in self.relevant_hyps[:8]:
                hyp_lines.append(f"  {h.get('name', '?')} : {h.get('type', '?')}")
            parts.insert(0, "Hypotheses:\n" + "\n".join(hyp_lines))
        parts.append(f"[shape={self.target_shape.value}, depth={self.depth}]")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to dict for LLM tactic engine consumption."""
        return {
            "target": self.target,
            "shape": self.target_shape.value,
            "hypotheses": self.relevant_hyps,
            "depth": self.depth,
            "target_head": self.target_head,
            "is_independent": self.is_independent,
        }


def format_goal_views_for_prompt(goal_views: list[GoalView],
                                  max_goals: int = 5) -> str:
    """Format multiple GoalViews into a prompt section.

    Used by the proof pipeline to inject structured goal state
    into LLM prompts for targeted tactic generation.
    """
    if not goal_views:
        return ""

    parts = [f"## Current Goal State ({len(goal_views)} goal(s))\n"]

    for i, gv in enumerate(goal_views[:max_goals], 1):
        parts.append(f"### Goal {i}")
        parts.append(gv.to_prompt())
        parts.append("")

    if len(goal_views) > max_goals:
        parts.append(f"... and {len(goal_views) - max_goals} more goals")

    return "\n".join(parts)


def _classify(e):
    if e.tag == "sort" and e.level and e.level.to_nat() == 0:
        return TargetShape.PROP
    if e.tag == "pi":
        if e.name and not e.name.is_anon():
            return TargetShape.FORALL
        return TargetShape.IMPLICATION
    if e.tag == "app" and e.get_app_fn_name():
        h = str(e.get_app_fn_name())
        if "Eq" in h:
            return TargetShape.EQUALITY
        if "And" in h:
            return TargetShape.CONJUNCTION
        if "Or" in h:
            return TargetShape.DISJUNCTION
        if "Exists" in h:
            return TargetShape.EXISTS
        return TargetShape.APPLICATION
    return TargetShape.OTHER
