"""prover/decompose/subgoal_scheduler.py — 子目标调度器

决定子目标的求解顺序，支持依赖感知调度。
"""
from __future__ import annotations
from dataclasses import dataclass
from prover.decompose.goal_decomposer import SubGoal


class SubGoalScheduler:
    """Schedule subgoals for solving in an optimal order.

    Strategies:
        'easy_first':    Solve easier subgoals first (build momentum)
        'hard_first':    Solve hardest subgoal first (fail fast)
        'dependency':    Respect dependency order
    """

    def __init__(self, strategy: str = "easy_first"):
        self.strategy = strategy

    def schedule(self, subgoals: list[SubGoal],
                 dependencies: dict[str, list[str]] = None) -> list[SubGoal]:
        """Return subgoals in optimal solving order.

        Args:
            subgoals: List of subgoals to schedule.
            dependencies: Map from subgoal name → list of prerequisite names.
        """
        deps = dependencies or {}

        if self.strategy == "dependency":
            return self._topo_sort(subgoals, deps)
        elif self.strategy == "hard_first":
            return self._sort_by_difficulty(subgoals, reverse=True)
        else:  # easy_first
            return self._sort_by_difficulty(subgoals, reverse=False)

    def _sort_by_difficulty(self, subgoals: list[SubGoal],
                            reverse: bool = False) -> list[SubGoal]:
        difficulty_order = {"trivial": 0, "easy": 1, "unknown": 2,
                            "medium": 3, "hard": 4, "competition": 5}
        return sorted(subgoals,
                      key=lambda g: difficulty_order.get(g.difficulty, 2),
                      reverse=reverse)

    def _topo_sort(self, subgoals: list[SubGoal],
                   deps: dict[str, list[str]]) -> list[SubGoal]:
        """Topological sort respecting dependencies."""
        name_to_goal = {g.name: g for g in subgoals}
        visited = set()
        result = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            for dep in deps.get(name, []):
                if dep in name_to_goal:
                    visit(dep)
            if name in name_to_goal:
                result.append(name_to_goal[name])

        for g in subgoals:
            visit(g.name)

        # Append any not covered by dependencies
        for g in subgoals:
            if g.name not in visited:
                result.append(g)

        return result

    def mark_solved(self, subgoals: list[SubGoal], name: str,
                    proof: str) -> list[SubGoal]:
        """Mark a subgoal as solved and return updated list."""
        for g in subgoals:
            if g.name == name:
                g.proved = True
                g.proof = proof
        return subgoals

    def unsolved(self, subgoals: list[SubGoal]) -> list[SubGoal]:
        return [g for g in subgoals if not g.proved]

    def progress(self, subgoals: list[SubGoal]) -> float:
        if not subgoals:
            return 1.0
        solved = sum(1 for g in subgoals if g.proved)
        return solved / len(subgoals)
