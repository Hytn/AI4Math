"""prover/decompose/composition.py — 子证明合成"""
from __future__ import annotations
from prover.decompose.goal_decomposer import SubGoal

def compose_proof(subgoals: list[SubGoal], main_theorem: str) -> str:
    preamble = "\n\n".join(f"{sg.statement} {sg.proof}" for sg in subgoals if sg.proved and sg.proof)
    return preamble
