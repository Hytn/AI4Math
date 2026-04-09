"""prover/sketch/sketch_generator.py — 证明草图生成"""
from __future__ import annotations
from dataclasses import dataclass, field
from common.roles import AgentRole, ROLE_PROMPTS

@dataclass
class ProofSketch:
    strategy: str = ""
    steps: list[str] = field(default_factory=list)
    suggested_lemmas: list[str] = field(default_factory=list)

class SketchGenerator:
    def __init__(self, llm): self.llm = llm
    def generate(self, theorem: str, nl_description: str = "") -> ProofSketch:
        prompt = f"Theorem:\n```lean\n{theorem}\n```\n"
        if nl_description: prompt += f"\nNatural language: {nl_description}\n"
        prompt += "\nProvide a high-level proof plan with strategy, steps, and useful lemmas."
        resp = self.llm.generate(system=ROLE_PROMPTS[AgentRole.PROOF_PLANNER], user=prompt, temperature=0.5)
        return ProofSketch(strategy=resp.content[:200], steps=[resp.content])
