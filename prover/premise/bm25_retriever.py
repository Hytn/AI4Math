"""prover/premise/bm25_retriever.py — BM25 稀疏检索"""
from __future__ import annotations
class BM25Retriever:
    def __init__(self, index_path: str = ""): self.index_path = index_path
    def retrieve(self, query: str, top_k: int = 10) -> list[str]: return []
