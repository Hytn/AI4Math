"""prover/premise/embedding_retriever.py — 向量语义检索"""
from __future__ import annotations
class EmbeddingRetriever:
    def __init__(self, index_path: str = ""): self.index_path = index_path
    def retrieve(self, query: str, top_k: int = 10) -> list[str]: return []
