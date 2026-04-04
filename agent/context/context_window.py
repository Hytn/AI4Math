"""agent/context/context_window.py — Context window with entry management

Tracks individual context entries (premises, errors, attempts, etc.)
with priorities and token estimates.  Automatically compresses when
the window approaches its limit.

Usage::

    ctx = ContextWindow(max_tokens=100_000)
    ctx.add_entry("theorem", theorem_text, priority=1.0)
    ctx.add_entry("premise", premise_text, priority=0.7)
    ctx.add_entry("attempt_3_error", error_text, priority=0.3)

    # Before building a prompt, render and auto-compress:
    prompt_text = ctx.render()          # returns compressed text
    remaining  = ctx.remaining_tokens() # how much room is left
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ContextEntry:
    """A single piece of context with metadata."""
    key: str
    content: str
    priority: float = 0.5          # 0.0 = expendable, 1.0 = essential
    category: str = "general"      # theorem, premise, attempt, error, lemma
    tokens: int = 0                # estimated token count
    is_compressible: bool = True   # can this entry be summarized/dropped?

    def __post_init__(self):
        if self.tokens == 0:
            self.tokens = estimate_tokens(self.content)


def estimate_tokens(text: str) -> int:
    """Estimate token count.

    Uses a mixed heuristic: ~1 token per 3.5 chars for code/math
    (tighter than the naive /4 for English prose).
    """
    if not text:
        return 0
    # Code and math tend to have shorter tokens than English prose
    return max(1, int(len(text) / 3.5))


class ContextWindow:
    """Managed context window with auto-compression.

    Entries are stored by key (upsertable).  When ``render()`` or
    ``needs_compression()`` is called, the window evaluates total
    token usage and compresses if necessary.

    Compression strategy (in order):
      1. Drop entries with priority < drop_threshold
      2. Truncate large low-priority entries (keep first/last lines)
      3. Summarize oldest attempt/error entries
    """

    def __init__(self, max_tokens: int = 100_000,
                 compress_threshold: float = 0.75,
                 drop_threshold: float = 0.15):
        self.max_tokens = max_tokens
        self.compress_threshold = compress_threshold
        self.drop_threshold = drop_threshold
        self._entries: dict[str, ContextEntry] = {}
        self._order: list[str] = []  # insertion order

    # ── Entry management ──

    def add_entry(self, key: str, content: str, priority: float = 0.5,
                  category: str = "general",
                  is_compressible: bool = True):
        """Add or update an entry."""
        entry = ContextEntry(
            key=key, content=content, priority=priority,
            category=category, is_compressible=is_compressible)
        if key in self._entries:
            self._entries[key] = entry
        else:
            self._entries[key] = entry
            self._order.append(key)

    def remove_entry(self, key: str):
        """Remove an entry by key."""
        if key in self._entries:
            del self._entries[key]
            self._order = [k for k in self._order if k != key]

    def get_entry(self, key: str) -> Optional[ContextEntry]:
        return self._entries.get(key)

    # ── Token accounting ──

    @property
    def used_tokens(self) -> int:
        return sum(e.tokens for e in self._entries.values())

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def usage_ratio(self) -> float:
        return self.used_tokens / self.max_tokens if self.max_tokens else 0

    def needs_compression(self, threshold: float = None) -> bool:
        t = threshold if threshold is not None else self.compress_threshold
        return self.usage_ratio() > t

    # ── Rendering ──

    def render(self, auto_compress: bool = True) -> str:
        """Render all entries into a single string.

        If auto_compress is True and the window is over threshold,
        compression runs first.
        """
        if auto_compress and self.needs_compression():
            self._compress()
        parts = []
        for key in self._order:
            entry = self._entries.get(key)
            if entry:
                parts.append(entry.content)
        return "\n\n".join(parts)

    def render_entries(self, categories: list[str] = None) -> list[ContextEntry]:
        """Return entries in order, optionally filtered by category."""
        result = []
        for key in self._order:
            entry = self._entries.get(key)
            if entry and (categories is None or entry.category in categories):
                result.append(entry)
        return result

    # ── Compression ──

    def _compress(self):
        """Run compression to bring usage below threshold."""
        target = int(self.max_tokens * self.compress_threshold * 0.9)

        # Phase 1: Drop low-priority compressible entries
        if self.used_tokens > target:
            droppable = sorted(
                [(k, e) for k, e in self._entries.items()
                 if e.is_compressible and e.priority < self.drop_threshold],
                key=lambda x: x[1].priority)
            for key, entry in droppable:
                if self.used_tokens <= target:
                    break
                self.remove_entry(key)

        # Phase 2: Truncate large low-priority entries
        if self.used_tokens > target:
            truncatable = sorted(
                [(k, e) for k, e in self._entries.items()
                 if e.is_compressible and e.tokens > 200
                 and e.priority < 0.6],
                key=lambda x: x[1].priority)
            for key, entry in truncatable:
                if self.used_tokens <= target:
                    break
                truncated = _truncate_text(entry.content, entry.tokens // 3)
                new_tokens = estimate_tokens(truncated)
                self._entries[key] = ContextEntry(
                    key=key, content=truncated, priority=entry.priority,
                    category=entry.category, tokens=new_tokens,
                    is_compressible=entry.is_compressible)

        # Phase 3: Summarize oldest attempt/error entries
        if self.used_tokens > target:
            attempt_keys = [k for k in self._order
                            if k in self._entries
                            and self._entries[k].category in ("attempt", "error")
                            and self._entries[k].is_compressible]
            # Keep most recent 3, summarize the rest
            if len(attempt_keys) > 3:
                to_summarize = attempt_keys[:-3]
                summary_parts = []
                for key in to_summarize:
                    e = self._entries[key]
                    # Extract first line as summary
                    first_line = e.content.split("\n")[0][:100]
                    summary_parts.append(f"[{key}] {first_line}")
                    self.remove_entry(key)

                summary_text = ("Earlier attempts (summarized):\n"
                                + "\n".join(summary_parts))
                self.add_entry("_compressed_history", summary_text,
                               priority=0.2, category="summary",
                               is_compressible=True)

    # ── Convenience ──

    def add(self, text: str):
        """Legacy API: add anonymous text (backward compatible)."""
        key = f"_anon_{len(self._entries)}"
        self.add_entry(key, text, priority=0.5)

    def __len__(self):
        return len(self._entries)

    def __repr__(self):
        return (f"ContextWindow({self.used_tokens}/{self.max_tokens} tokens, "
                f"{len(self._entries)} entries)")


def _truncate_text(text: str, target_tokens: int) -> str:
    """Keep first and last portions of text to fit target tokens."""
    target_chars = max(50, target_tokens * 4)
    if len(text) <= target_chars:
        return text
    half = target_chars // 2
    lines = text.split("\n")
    if len(lines) <= 4:
        return text[:target_chars] + "..."
    # Keep some from start, some from end
    start_lines = []
    start_len = 0
    for line in lines:
        if start_len + len(line) > half:
            break
        start_lines.append(line)
        start_len += len(line) + 1
    end_lines = []
    end_len = 0
    for line in reversed(lines):
        if end_len + len(line) > half:
            break
        end_lines.insert(0, line)
        end_len += len(line) + 1
    omitted = len(lines) - len(start_lines) - len(end_lines)
    return ("\n".join(start_lines)
            + f"\n... ({omitted} lines omitted) ...\n"
            + "\n".join(end_lines))
