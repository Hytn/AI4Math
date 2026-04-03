"""prover/codegen/tactic_generator.py — Lean 4 tactic 代码生成"""
from __future__ import annotations
from agent.brain.roles import AgentRole, ROLE_PROMPTS
from agent.brain.response_parser import extract_lean_code

class TacticGenerator:
    def __init__(self, llm): self.llm = llm
    def generate(self, theorem: str, sketch: str = "", premises: list[str] = None,
                 banked_lemmas: str = "", temperature: float = 0.7) -> str:
        from agent.brain.prompt_builder import build_prompt
        prompt = build_prompt(theorem, sketch=sketch, premises=premises or [], banked_lemmas=banked_lemmas)
        resp = self.llm.generate(system=ROLE_PROMPTS[AgentRole.PROOF_GENERATOR], user=prompt, temperature=temperature)
        return extract_lean_code(resp.content)
