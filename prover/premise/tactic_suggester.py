"""prover/premise/tactic_suggester.py — 策略建议器"""
from __future__ import annotations
class TacticSuggester:
    def __init__(self, llm=None): self.llm = llm
    def suggest(self, goal_state: str) -> list[str]:
        return ["simp", "ring", "linarith", "nlinarith", "omega", "norm_num"]
