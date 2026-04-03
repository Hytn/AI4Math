"""
core/retriever.py — 前提检索 (Premise Selection)

最小实现：让 LLM 自己猜需要的 mathlib 引理。
后续可扩展为：embedding 向量检索、LeanDojo premise selection 等。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PremiseRetriever:
    """
    前提检索器。

    V1 (当前)：空实现，不做独立检索。
       → proof generator 直接在 prompt 里依赖 LLM 自身的 mathlib 知识。
       → 如果需要，可以在 prompt 里加入 `exact?` / `apply?` 的使用提示。

    V2 (规划)：基于 mathlib 名称/签名的 BM25 检索。
    V3 (规划)：基于 embedding 的语义检索 + rerank。
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.mode = self.config.get("mode", "none")
        self._index = None

        if self.mode == "bm25":
            self._init_bm25()
        elif self.mode == "embedding":
            self._init_embedding()

    def retrieve(self, theorem_statement: str, top_k: int = 10) -> list[str]:
        """
        给定 theorem statement，返回 top_k 个可能有用的 mathlib 引理名称。

        Args:
            theorem_statement: Lean 4 theorem 声明
            top_k: 返回数量

        Returns:
            引理名称列表 (如 ["Nat.add_comm", "Int.mul_assoc", ...])
        """
        if self.mode == "none":
            return []
        elif self.mode == "bm25":
            return self._retrieve_bm25(theorem_statement, top_k)
        elif self.mode == "embedding":
            return self._retrieve_embedding(theorem_statement, top_k)
        else:
            return []

    # ── BM25 检索 (V2) ──────────────────────────────────────────

    def _init_bm25(self):
        """加载 mathlib 引理索引并构建 BM25"""
        index_path = self.config.get("index_path", "")
        if not index_path:
            logger.warning("BM25 mode requires index_path in config; falling back to none")
            self.mode = "none"
            return
        # TODO: 加载预构建的 mathlib 引理索引
        # 格式: [{"name": "Nat.add_comm", "type": "...", "doc": "..."}, ...]
        logger.info(f"BM25 index loading from {index_path} — not yet implemented")

    def _retrieve_bm25(self, query: str, top_k: int) -> list[str]:
        # TODO: 实现 BM25 检索
        return []

    # ── Embedding 检索 (V3) ─────────────────────────────────────

    def _init_embedding(self):
        """加载 embedding 模型和向量索引"""
        logger.info("Embedding retriever — not yet implemented")
        self.mode = "none"

    def _retrieve_embedding(self, query: str, top_k: int) -> list[str]:
        # TODO: 实现向量检索
        return []
