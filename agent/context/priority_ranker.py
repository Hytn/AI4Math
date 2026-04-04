"""agent/context/priority_ranker.py — 上下文优先级排序

为不同类型的上下文信息分配优先级分数。
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RankedItem:
    content: str
    category: str
    priority: float
    tokens: int = 0


class PriorityRanker:
    """Assign priority scores to context items for compression decisions.

    Priority ranges: 0.0 (drop first) to 1.0 (keep always).
    """

    # Default priority by category
    CATEGORY_PRIORITIES = {
        "theorem_statement": 1.0,   # always keep
        "current_goal": 0.95,       # critical
        "latest_error": 0.9,        # very important for repair
        "error_summary": 0.8,       # condensed error info
        "banked_lemma": 0.75,       # proven sub-results
        "premise": 0.7,             # retrieved knowledge
        "recent_attempt": 0.6,      # recent proof attempt
        "proof_template": 0.55,     # suggested templates
        "tactic_hint": 0.5,         # tactic suggestions
        "old_attempt": 0.3,         # older attempts
        "old_error": 0.2,           # older errors
        "metadata": 0.1,            # config/stats
    }

    def __init__(self, custom_priorities: dict = None):
        self.priorities = {**self.CATEGORY_PRIORITIES}
        if custom_priorities:
            self.priorities.update(custom_priorities)

    def rank(self, items: list[dict]) -> list[RankedItem]:
        """Assign priorities and sort items.

        Each item should have 'content' and 'category'.
        Returns sorted list (highest priority first).
        """
        ranked = []
        for item in items:
            category = item.get("category", "metadata")
            base_priority = self.priorities.get(category, 0.5)

            # Adjust by recency (newer items slightly higher priority)
            recency_boost = item.get("recency", 0) * 0.05
            priority = min(1.0, base_priority + recency_boost)

            tokens = self._estimate_tokens(item.get("content", ""))
            ranked.append(RankedItem(
                content=item.get("content", ""),
                category=category,
                priority=priority,
                tokens=tokens,
            ))

        ranked.sort(key=lambda x: -x.priority)
        return ranked

    def filter_by_budget(self, items: list[dict],
                         token_budget: int) -> list[dict]:
        """Filter items to fit within a token budget, prioritized."""
        ranked = self.rank(items)
        result = []
        remaining = token_budget

        for item in ranked:
            if item.tokens <= remaining:
                result.append({
                    "content": item.content,
                    "category": item.category,
                    "priority": item.priority,
                })
                remaining -= item.tokens

        return result

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 3) if text else 0
