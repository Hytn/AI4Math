"""prover/codegen/scaffold_generator.py — Sorry-based 证明骨架生成"""
from __future__ import annotations
from agent.brain.roles import AgentRole, ROLE_PROMPTS
from agent.brain.response_parser import extract_lean_code

SCAFFOLD_SYSTEM = """\
You are a Lean 4 proof architect. Generate a proof SKELETON with `sorry` placeholders.
Break the proof into `have` steps. Each step should be individually provable.
Mark each unproved step with `sorry`. The overall structure must be logically sound."""

class ScaffoldGenerator:
    def __init__(self, llm): self.llm = llm
    def generate(self, theorem: str, sketch: str = "", premises: list[str] = None) -> str:
        prompt = f"Theorem:\n```lean\n{theorem}\n```\nSketch: {sketch}\n\nGenerate a sorry-skeleton proof."
        resp = self.llm.generate(system=SCAFFOLD_SYSTEM, user=prompt, temperature=0.5)
        return extract_lean_code(resp.content)
