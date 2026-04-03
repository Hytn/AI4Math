"""prover/premise/selector.py — 统一前提选择入口"""
from __future__ import annotations

class PremiseSelector:
    def __init__(self, config=None):
        self.config = config or {}; self.mode = self.config.get("mode", "none")
    def retrieve(self, theorem: str, top_k: int = 10) -> list[str]:
        if self.mode == "none": return []
        return []  # TODO: integrate BM25/embedding retrievers
