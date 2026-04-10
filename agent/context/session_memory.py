"""agent/context/session_memory.py — Cross-task persistent memory

Inspired by Claude Code's session memory system (SessionMemory/):
  - Accumulates key discoveries across multiple proof attempts
  - Persists between tasks within a session
  - Automatically injects relevant memories into new proof contexts
  - Supports decay (old memories fade) and reinforcement (confirmed facts strengthen)

This implements the "knowledge flywheel" at the session level: each proven
theorem deposits its discoveries into session memory, and the next theorem
benefits from accumulated wisdom.

Usage::

    memory = SessionMemory()

    # After proving theorem A
    memory.record_proof(
        problem_id="nat_add_comm",
        theorem="theorem Nat.add_comm ...",
        proof="by induction n with ...",
        tactics_used=["induction", "simp", "rfl"],
        lemmas_used=["Nat.succ_add", "Nat.add_zero"],
    )

    # Before proving theorem B
    relevant = memory.recall(
        theorem="theorem Nat.mul_comm ...",
        domain="number_theory",
        max_tokens=2000,
    )
    # → Returns relevant tactics, lemmas, and patterns from past proofs
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry from a past proof attempt."""
    key: str                        # Unique identifier
    category: str                   # "tactic", "lemma", "pattern", "error", "proof"
    content: str                    # Human-readable content
    domain: str = ""                # Mathematical domain
    relevance_score: float = 0.5    # Current relevance (decays over time)
    use_count: int = 0              # Times this memory was recalled
    success_count: int = 0          # Times the associated tactic/lemma led to success
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    source_problem: str = ""        # Which problem generated this memory
    metadata: dict = field(default_factory=dict)

    def decay(self, factor: float = 0.95):
        """Apply time-based decay to relevance score."""
        age_hours = (time.time() - self.last_accessed) / 3600
        self.relevance_score *= factor ** age_hours

    def reinforce(self, amount: float = 0.1):
        """Reinforce this memory (it was useful)."""
        self.relevance_score = min(1.0, self.relevance_score + amount)
        self.use_count += 1
        self.last_accessed = time.time()


class SessionMemory:
    """Cross-task session memory that accumulates discoveries.

    Memory types:
      - Tactics: Which tactics work for which goal patterns
      - Lemmas: Useful lemmas discovered during proofs
      - Patterns: Proof patterns that transfer across problems
      - Errors: Common errors and their fixes
      - Proofs: Successfully verified proof fragments
    """

    def __init__(self, max_entries: int = 1000, decay_factor: float = 0.98):
        self._entries: dict[str, MemoryEntry] = {}
        self._domain_index: dict[str, list[str]] = defaultdict(list)
        self._category_index: dict[str, list[str]] = defaultdict(list)
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._decay_factor = decay_factor
        self._session_start = time.time()

    # ── Recording ─────────────────────────────────────────────────────────

    def record_proof(
        self,
        problem_id: str,
        theorem: str,
        proof: str,
        success: bool = True,
        tactics_used: list[str] = None,
        lemmas_used: list[str] = None,
        errors_encountered: list[dict] = None,
        domain: str = "",
    ):
        """Record findings from a completed proof attempt."""
        with self._lock:
            # Record the proof itself
            if success and proof:
                self._add_entry(MemoryEntry(
                    key=f"proof:{problem_id}",
                    category="proof",
                    content=f"Proven: {theorem[:200]}\nProof: {proof[:500]}",
                    domain=domain,
                    relevance_score=0.9,
                    source_problem=problem_id,
                ))

            # Record effective tactics
            for tactic in (tactics_used or []):
                key = f"tactic:{tactic}:{domain or 'general'}"
                existing = self._entries.get(key)
                if existing:
                    existing.reinforce(0.15 if success else 0.02)
                    if success:
                        existing.success_count += 1
                else:
                    self._add_entry(MemoryEntry(
                        key=key,
                        category="tactic",
                        content=f"Tactic `{tactic}` {'succeeded' if success else 'attempted'} "
                                f"for {domain or 'general'} problems",
                        domain=domain,
                        relevance_score=0.7 if success else 0.3,
                        success_count=1 if success else 0,
                        source_problem=problem_id,
                    ))

            # Record useful lemmas
            for lemma in (lemmas_used or []):
                key = f"lemma:{lemma}"
                existing = self._entries.get(key)
                if existing:
                    existing.reinforce(0.2)
                    if success:
                        existing.success_count += 1
                else:
                    self._add_entry(MemoryEntry(
                        key=key,
                        category="lemma",
                        content=f"Lemma `{lemma}` useful for {domain or 'general'} proofs",
                        domain=domain,
                        relevance_score=0.8,
                        success_count=1 if success else 0,
                        source_problem=problem_id,
                    ))

            # Record error patterns
            for err in (errors_encountered or []):
                err_cat = err.get("category", "other")
                err_msg = err.get("message", "")[:200]
                fix = err.get("fix", "")[:200]
                key = f"error:{err_cat}:{hash(err_msg) % 100000}"
                if key not in self._entries:
                    self._add_entry(MemoryEntry(
                        key=key,
                        category="error",
                        content=f"Error [{err_cat}]: {err_msg}"
                                + (f"\nFix: {fix}" if fix else ""),
                        domain=domain,
                        relevance_score=0.5,
                        source_problem=problem_id,
                        metadata=err,
                    ))

    def record_discovery(
        self,
        key: str,
        content: str,
        category: str = "pattern",
        domain: str = "",
        relevance: float = 0.6,
        source_problem: str = "",
    ):
        """Record a general discovery."""
        with self._lock:
            self._add_entry(MemoryEntry(
                key=key,
                category=category,
                content=content,
                domain=domain,
                relevance_score=relevance,
                source_problem=source_problem,
            ))

    # ── Recall ────────────────────────────────────────────────────────────

    def recall(
        self,
        theorem: str = "",
        domain: str = "",
        categories: list[str] = None,
        max_entries: int = 15,
        max_tokens: int = 3000,
        min_relevance: float = 0.1,
    ) -> list[MemoryEntry]:
        """Recall relevant memories for a new proof task.

        Uses domain matching, keyword overlap, and relevance scores
        to find the most useful memories.
        """
        with self._lock:
            # Apply decay
            for entry in self._entries.values():
                entry.decay(self._decay_factor)

            # Collect candidates
            candidates = list(self._entries.values())

            # Filter by category
            if categories:
                candidates = [e for e in candidates if e.category in categories]

            # Filter by minimum relevance
            candidates = [e for e in candidates if e.relevance_score >= min_relevance]

            # Score each candidate
            scored = []
            theorem_lower = theorem.lower()
            for entry in candidates:
                score = entry.relevance_score

                # Domain match bonus
                if domain and entry.domain == domain:
                    score += 0.2

                # Keyword overlap bonus
                if theorem_lower:
                    entry_lower = entry.content.lower()
                    keywords = set(theorem_lower.split()) - {
                        "theorem", "lemma", "the", "a", "an", "is", "of", "for",
                        ":", "(", ")", "→", "∀", "∃"}
                    overlap = sum(1 for kw in keywords if kw in entry_lower)
                    score += min(0.3, overlap * 0.05)

                # Success rate bonus
                if entry.use_count > 0:
                    success_rate = entry.success_count / entry.use_count
                    score += success_rate * 0.15

                scored.append((entry, score))

            # Sort and select
            scored.sort(key=lambda x: -x[1])
            result = []
            total_tokens = 0
            for entry, _ in scored[:max_entries]:
                tokens = len(entry.content) // 4
                if total_tokens + tokens > max_tokens:
                    break
                result.append(entry)
                total_tokens += tokens
                entry.last_accessed = time.time()

            return result

    def format_for_prompt(
        self,
        theorem: str = "",
        domain: str = "",
        max_tokens: int = 2000,
    ) -> str:
        """Format recalled memories for injection into LLM prompt."""
        entries = self.recall(
            theorem=theorem, domain=domain, max_tokens=max_tokens)

        if not entries:
            return ""

        sections = defaultdict(list)
        for entry in entries:
            sections[entry.category].append(entry)

        parts = ["--- Session Memory (accumulated knowledge) ---"]

        if sections.get("tactic"):
            parts.append("Effective tactics for similar problems:")
            for e in sections["tactic"][:5]:
                parts.append(f"  - {e.content}")

        if sections.get("lemma"):
            parts.append("Useful lemmas:")
            for e in sections["lemma"][:5]:
                parts.append(f"  - {e.content}")

        if sections.get("pattern"):
            parts.append("Known patterns:")
            for e in sections["pattern"][:3]:
                parts.append(f"  - {e.content}")

        if sections.get("error"):
            parts.append("Common errors to avoid:")
            for e in sections["error"][:3]:
                parts.append(f"  - {e.content}")

        if sections.get("proof"):
            parts.append("Related proven results:")
            for e in sections["proof"][:2]:
                parts.append(f"  - {e.content[:200]}")

        parts.append("--- End Session Memory ---")
        return "\n".join(parts)

    # ── Internal ──────────────────────────────────────────────────────────

    def _add_entry(self, entry: MemoryEntry):
        """Add an entry, evicting lowest-relevance if full."""
        self._entries[entry.key] = entry
        self._domain_index[entry.domain].append(entry.key)
        self._category_index[entry.category].append(entry.key)

        # Evict if over limit
        if len(self._entries) > self._max_entries:
            self._evict()

    def _evict(self):
        """Evict lowest-relevance entries."""
        sorted_entries = sorted(
            self._entries.items(),
            key=lambda x: x[1].relevance_score)
        to_remove = len(self._entries) - self._max_entries + 50  # Remove batch
        for key, _ in sorted_entries[:to_remove]:
            del self._entries[key]

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to dict for persistence."""
        return {
            "entries": {
                k: {
                    "key": e.key, "category": e.category,
                    "content": e.content, "domain": e.domain,
                    "relevance_score": e.relevance_score,
                    "use_count": e.use_count,
                    "success_count": e.success_count,
                    "created_at": e.created_at,
                    "last_accessed": e.last_accessed,
                    "source_problem": e.source_problem,
                    "metadata": e.metadata,
                }
                for k, e in self._entries.items()
            },
            "session_start": self._session_start,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionMemory:
        """Deserialize from dict."""
        memory = cls()
        memory._session_start = data.get("session_start", time.time())
        for key, ed in data.get("entries", {}).items():
            memory._entries[key] = MemoryEntry(**ed)
            memory._domain_index[ed.get("domain", "")].append(key)
            memory._category_index[ed.get("category", "")].append(key)
        return memory

    def __len__(self):
        return len(self._entries)

    def stats(self) -> dict:
        """Return memory statistics."""
        categories = defaultdict(int)
        for e in self._entries.values():
            categories[e.category] += 1
        return {
            "total_entries": len(self._entries),
            "categories": dict(categories),
            "avg_relevance": (
                sum(e.relevance_score for e in self._entries.values())
                / max(1, len(self._entries))),
            "session_age_minutes": (time.time() - self._session_start) / 60,
        }
