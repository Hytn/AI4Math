"""prover/premise/reranker.py — LLM-based 检索结果重排序"""
from __future__ import annotations
class PremiseReranker:
    def __init__(self, llm=None): self.llm = llm
    def rerank(self, theorem: str, candidates: list[str], top_k: int = 10) -> list[str]:
        return candidates[:top_k]
