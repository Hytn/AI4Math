"""prover/premise/reranker.py — 前提重排序器

对 BM25/Embedding 检索结果进行二次排序:
1. 类型签名匹配度 (与目标的类型结构相似度)
2. Tactic 相关性 (是否适合当前 tactic 策略)
3. 多信号融合 (RRF: Reciprocal Rank Fusion)
"""
from __future__ import annotations
import re
from collections import Counter
from prover.premise.bm25_retriever import tokenize


class PremiseReranker:
    """Rerank retrieved premises using multiple signals."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k  # RRF constant

    def rerank(self, candidates: list[dict], query: str,
               goal_type: str = "", tactic_hint: str = "",
               top_k: int = 10) -> list[dict]:
        """Rerank candidates using reciprocal rank fusion of multiple signals.

        Args:
            candidates: List of dicts with 'name', 'statement', 'score'.
            query: The original query / theorem statement.
            goal_type: The target type to match against (e.g., "Nat → Nat → Prop").
            tactic_hint: If known, the tactic being used (e.g., "apply", "simp").
            top_k: Number of results to return.
        """
        if not candidates:
            return []

        # Signal 1: Original retrieval score (already sorted)
        original_ranks = {c["name"]: i for i, c in enumerate(candidates)}

        # Signal 2: Type signature overlap
        type_ranks = self._rank_by_type_overlap(candidates, goal_type or query)

        # Signal 3: Tactic relevance
        tactic_ranks = self._rank_by_tactic_relevance(candidates, tactic_hint)

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        for name in original_ranks:
            r1 = original_ranks.get(name, len(candidates))
            r2 = type_ranks.get(name, len(candidates))
            r3 = tactic_ranks.get(name, len(candidates))
            rrf_scores[name] = (
                1.0 / (self.rrf_k + r1) +
                1.0 / (self.rrf_k + r2) +
                0.5 / (self.rrf_k + r3)  # tactic signal weighted less
            )

        # Sort by RRF score
        name_to_cand = {c["name"]: c for c in candidates}
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return [
            {**name_to_cand[name], "rrf_score": round(score, 6)}
            for name, score in ranked[:top_k]
            if name in name_to_cand
        ]

    def _rank_by_type_overlap(self, candidates: list[dict],
                               goal: str) -> dict[str, int]:
        """Rank by how much the premise type signature overlaps with the goal."""
        goal_tokens = set(tokenize(goal))
        scored = []
        for c in candidates:
            stmt_tokens = set(tokenize(c["statement"]))
            overlap = len(goal_tokens & stmt_tokens)
            total = len(goal_tokens | stmt_tokens) or 1
            jaccard = overlap / total
            scored.append((c["name"], jaccard))
        scored.sort(key=lambda x: -x[1])
        return {name: rank for rank, (name, _) in enumerate(scored)}

    def _rank_by_tactic_relevance(self, candidates: list[dict],
                                    tactic: str) -> dict[str, int]:
        """Rank premises by relevance to the current tactic."""
        tactic = tactic.lower().strip()
        scored = []
        for c in candidates:
            relevance = 0.0
            stmt = c["statement"].lower()
            name = c["name"].lower()

            if tactic == "simp":
                # simp prefers lemmas with @[simp] or iff/eq in conclusion
                if "simp" in name or "iff" in stmt or "eq" in stmt:
                    relevance += 2.0
            elif tactic == "apply":
                # apply prefers lemmas whose conclusion matches goal
                if "→" in c["statement"] or "->" in c["statement"]:
                    relevance += 1.0
            elif tactic == "rw" or tactic == "rewrite":
                # rewrite prefers equalities
                if "=" in stmt and "≠" not in stmt:
                    relevance += 2.0
            elif tactic == "exact":
                relevance += 0.5  # neutral

            scored.append((c["name"], relevance))
        scored.sort(key=lambda x: -x[1])
        return {name: rank for rank, (name, _) in enumerate(scored)}
