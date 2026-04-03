"""agent/context/context_window.py — 上下文窗口追踪"""
from __future__ import annotations

class ContextWindow:
    def __init__(self, max_tokens: int = 100000):
        self.max_tokens = max_tokens
        self.used_tokens = 0

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def add(self, text: str):
        self.used_tokens += self.estimate_tokens(text)

    def usage_ratio(self) -> float:
        return self.used_tokens / self.max_tokens

    def needs_compression(self, threshold: float = 0.7) -> bool:
        return self.usage_ratio() > threshold
