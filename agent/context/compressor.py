"""agent/context/compressor.py — 上下文压缩

将过长的对话历史/错误日志压缩到适合 LLM 上下文窗口的大小。
支持多种压缩策略。
"""
from __future__ import annotations
import re


class ContextCompressor:
    """Compress context to fit within token limits.

    Strategies:
        'truncate':   Keep most recent, drop oldest
        'summarize':  Summarize older entries
        'selective':  Keep high-priority items, drop low-priority
    """

    def __init__(self, max_tokens: int = 8000, strategy: str = "selective"):
        self.max_tokens = max_tokens
        self.strategy = strategy

    def compress(self, entries: list[dict],
                 priority_key: str = "priority") -> list[dict]:
        """Compress a list of context entries to fit within token limit.

        Each entry should have 'content' (str) and optionally 'priority' (float).
        Returns filtered/compressed entries.
        """
        if not entries:
            return []

        total = sum(self._estimate_tokens(e.get("content", "")) for e in entries)
        if total <= self.max_tokens:
            return entries

        if self.strategy == "truncate":
            return self._truncate(entries)
        elif self.strategy == "summarize":
            return self._summarize(entries)
        else:
            return self._selective(entries, priority_key)

    def compress_text(self, text: str, max_tokens: int = 0) -> str:
        """Compress a single text block."""
        limit = max_tokens or self.max_tokens
        tokens = self._estimate_tokens(text)
        if tokens <= limit:
            return text

        # Keep first and last portions
        lines = text.split("\n")
        if len(lines) <= 4:
            return text[:limit * 4]  # rough char limit

        keep_start = max(2, len(lines) // 4)
        keep_end = max(2, len(lines) // 4)
        middle_summary = f"\n... ({len(lines) - keep_start - keep_end} lines omitted) ...\n"
        return "\n".join(lines[:keep_start]) + middle_summary + "\n".join(lines[-keep_end:])

    def _truncate(self, entries: list[dict]) -> list[dict]:
        """Keep most recent entries within budget."""
        result = []
        budget = self.max_tokens
        for entry in reversed(entries):
            tokens = self._estimate_tokens(entry.get("content", ""))
            if budget - tokens >= 0:
                result.insert(0, entry)
                budget -= tokens
            else:
                break
        return result

    def _selective(self, entries: list[dict],
                   priority_key: str) -> list[dict]:
        """Keep high-priority entries, drop low-priority ones."""
        # Sort by priority (higher = keep)
        sorted_entries = sorted(entries,
                                key=lambda e: e.get(priority_key, 0.5),
                                reverse=True)

        result = []
        budget = self.max_tokens
        for entry in sorted_entries:
            tokens = self._estimate_tokens(entry.get("content", ""))
            if budget - tokens >= 0:
                result.append(entry)
                budget -= tokens

        # Restore original order
        original_order = {id(e): i for i, e in enumerate(entries)}
        result.sort(key=lambda e: original_order.get(id(e), 0))
        return result

    def _summarize(self, entries: list[dict]) -> list[dict]:
        """Summarize older entries to save space."""
        if len(entries) <= 3:
            return entries

        # Keep recent entries as-is, summarize older ones
        recent = entries[-3:]
        older = entries[:-3]

        summary_content = f"[Summary of {len(older)} earlier entries: "
        keywords = set()
        for e in older:
            words = e.get("content", "").split()[:5]
            keywords.update(w for w in words if len(w) > 3)
        summary_content += ", ".join(list(keywords)[:10]) + "]"

        summary_entry = {"content": summary_content, "priority": 0.3,
                         "is_summary": True}
        return [summary_entry] + recent

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (1 token ≈ 4 chars for English, 2 chars for code)."""
        if not text:
            return 0
        return max(1, len(text) // 3)
