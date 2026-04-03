"""prover/decompose/goal_decomposer.py — 复杂定理分解为子目标"""
from __future__ import annotations
from dataclasses import dataclass
from agent.brain.roles import AgentRole, ROLE_PROMPTS

@dataclass
class SubGoal:
    name: str; statement: str; difficulty: str = "unknown"; proved: bool = False; proof: str = ""

class GoalDecomposer:
    def __init__(self, llm): self.llm = llm
    def decompose(self, theorem: str, max_subgoals: int = 5) -> list[SubGoal]:
        prompt = f"Decompose this theorem into sub-lemmas:\n```lean\n{theorem}\n```\nOutput each as a Lean 4 lemma statement."
        resp = self.llm.generate(system=ROLE_PROMPTS[AgentRole.DECOMPOSER], user=prompt, temperature=0.5)
        return [SubGoal(name=f"subgoal_{i}", statement=line.strip())
                for i, line in enumerate(resp.content.split("\n")) if "lemma" in line.lower()][:max_subgoals]
