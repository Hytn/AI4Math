"""prover/formalize/autoformalize.py — NL → Lean 4 形式化"""
from __future__ import annotations
from agent.brain.roles import AgentRole, ROLE_PROMPTS
from agent.brain.response_parser import extract_lean_code

class AutoFormalizer:
    def __init__(self, llm): self.llm = llm
    def formalize(self, nl_statement: str) -> str:
        prompt = f"Formalize this mathematical statement in Lean 4 (with Mathlib):\n\n{nl_statement}\n\nOutput a theorem declaration."
        resp = self.llm.generate(system=ROLE_PROMPTS[AgentRole.FORMALIZATION_EXPERT], user=prompt, temperature=0.3)
        return extract_lean_code(resp.content)
