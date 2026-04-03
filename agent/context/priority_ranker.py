"""agent/context/priority_ranker.py — 信息优先级排序"""
from __future__ import annotations

def rank_context_items(items: list[dict], budget_tokens: int) -> list[dict]:
    priority_order = ["theorem_statement", "banked_lemmas", "error_analysis",
                      "proof_sketch", "premises", "error_history"]
    ranked = sorted(items, key=lambda x: priority_order.index(x.get("type", ""))
                    if x.get("type", "") in priority_order else 99)
    selected, total = [], 0
    for item in ranked:
        est = len(item.get("content", "")) // 4
        if total + est <= budget_tokens:
            selected.append(item); total += est
    return selected
