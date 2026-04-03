"""agent/context/compressor.py — 上下文压缩"""
from __future__ import annotations

class ContextCompressor:
    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def compress(self, content: str, target_ratio: float = 0.5) -> str:
        if not self.llm:
            lines = content.split("\n")
            keep = max(1, int(len(lines) * target_ratio))
            return "\n".join(lines[:keep]) + "\n... (compressed)"
        from agent.brain.roles import AgentRole, ROLE_PROMPTS
        resp = self.llm.generate(
            system="Summarize the following proof attempt history concisely.",
            user=content, temperature=0.3, max_tokens=1024)
        return resp.content
