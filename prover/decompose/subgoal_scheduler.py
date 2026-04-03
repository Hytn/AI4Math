"""prover/decompose/subgoal_scheduler.py — 子目标调度"""
from __future__ import annotations
from prover.decompose.goal_decomposer import SubGoal

def schedule(subgoals: list[SubGoal]) -> list[SubGoal]:
    return sorted(subgoals, key=lambda s: {"easy": 0, "medium": 1, "hard": 2, "unknown": 1}.get(s.difficulty, 1))
