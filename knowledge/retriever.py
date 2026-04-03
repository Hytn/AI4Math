"""knowledge/retriever.py — 统一检索接口"""
from __future__ import annotations

class KnowledgeRetriever:
    def __init__(self, config=None): self.config = config or {}
    def retrieve(self, query: str, top_k: int = 10, mode: str = "hybrid") -> list[str]:
        return []  # TODO: integrate BM25 + FAISS
