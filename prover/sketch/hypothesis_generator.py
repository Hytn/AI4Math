"""prover/sketch/hypothesis_generator.py — 辅助假设生成"""
from __future__ import annotations
from agent.brain.roles import AgentRole, ROLE_PROMPTS

class HypothesisGenerator:
    def __init__(self, llm): self.llm = llm
    def generate(self, theorem: str, sketch: str = "", max_hypotheses: int = 3) -> list[str]:
        prompt = f"Theorem:\n```lean\n{theorem}\n```\nSketch: {sketch}\n\nPropose {max_hypotheses} auxiliary lemmas."
        resp = self.llm.generate(system=ROLE_PROMPTS[AgentRole.HYPOTHESIS_PROPOSER], user=prompt, temperature=0.7)
        return [line.strip() for line in resp.content.split("\n") if line.strip().startswith("have") or line.strip().startswith("lemma")][:max_hypotheses]
