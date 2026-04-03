"""prover/repair/repair_generator.py — 自动修复方案生成"""
from __future__ import annotations
from agent.brain.roles import AgentRole, ROLE_PROMPTS
from agent.brain.response_parser import extract_lean_code

class RepairGenerator:
    def __init__(self, llm): self.llm = llm
    def generate_repair(self, theorem: str, failed_proof: str, error_analysis: str) -> str:
        prompt = f"Theorem:\n```lean\n{theorem}\n```\n\nFailed proof:\n```lean\n{failed_proof}\n```\n\n{error_analysis}\n\nGenerate corrected proof."
        resp = self.llm.generate(system=ROLE_PROMPTS[AgentRole.REPAIR_AGENT], user=prompt, temperature=0.5)
        return extract_lean_code(resp.content)
