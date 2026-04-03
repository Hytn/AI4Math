"""agent/strategy/reflection.py — 自我反思: 分析失败模式"""
from __future__ import annotations
from agent.brain.llm_provider import LLMProvider
from agent.brain.roles import AgentRole, ROLE_PROMPTS

class Reflector:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def reflect(self, theorem: str, error_summary: str, best_attempts: list[str]) -> str:
        prompt = f"""Theorem: {theorem}

Error summary: {error_summary}

Best attempts so far:
{chr(10).join(best_attempts[:3])}

Analyze: Why are these attempts failing? What fundamentally different approach should we try?"""
        resp = self.llm.generate(system=ROLE_PROMPTS[AgentRole.CRITIC], user=prompt, temperature=0.5)
        return resp.content
