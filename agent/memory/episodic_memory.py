"""agent/memory/episodic_memory.py — 情景记忆: 历史解题经验

Storage backends:
  - Legacy: JSONL file (default if no unified store provided)
  - Unified: SQLite via UnifiedKnowledgeStore (preferred)

Migration: pass a UnifiedKnowledgeStore to use SQLite; existing JSONL
data is auto-migrated on first load.
"""
from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    problem_type: str
    difficulty: str
    winning_strategy: str
    key_tactics: list[str]
    key_insight: str
    solve_time_ms: int


class EpisodicMemory:
    def __init__(self, store_path: str = "results/episodic_memory.jsonl",
                 unified_store=None):
        """
        Args:
            store_path: Legacy JSONL file path (used if unified_store is None)
            unified_store: UnifiedKnowledgeStore instance for SQLite backend
        """
        self.store_path = Path(store_path)
        self.episodes: list[Episode] = []
        self._unified = unified_store
        self._load()

    def _load(self):
        # If unified store is available, load from SQLite
        if self._unified:
            self._load_from_unified()
            # Auto-migrate legacy JSONL if it exists
            if self.store_path.exists():
                self._migrate_jsonl_to_unified()
            return
        # Legacy: load from JSONL
        if not self.store_path.exists():
            return
        try:
            for line_num, line in enumerate(
                    self.store_path.read_text().strip().split("\n"), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self.episodes.append(Episode(**data))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"Skipping malformed episode at line {line_num}: {e}")
        except (OSError, IOError) as e:
            logger.warning(f"Could not load episodic memory from {self.store_path}: {e}")

    def _load_from_unified(self):
        """Load episodes from UnifiedKnowledgeStore (SQLite)."""
        try:
            rows = self._unified._query_episodes_sync("", "", 500)
            for r in rows:
                self.episodes.append(Episode(
                    problem_type=r["problem_type"],
                    difficulty=r["difficulty"],
                    winning_strategy=r["winning_strategy"],
                    key_tactics=r["key_tactics"],
                    key_insight=r["key_insight"],
                    solve_time_ms=r["solve_time_ms"]))
        except Exception as e:
            logger.warning(f"Could not load episodes from unified store: {e}")

    def _migrate_jsonl_to_unified(self):
        """One-time migration of legacy JSONL to unified SQLite store."""
        if not self.store_path.exists() or not self._unified:
            return
        try:
            migrated = 0
            for line in self.store_path.read_text().strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._unified._add_episode_sync(
                        data.get("problem_type", ""),
                        data.get("difficulty", ""),
                        data.get("winning_strategy", ""),
                        data.get("key_tactics", []),
                        data.get("key_insight", ""),
                        data.get("solve_time_ms", 0))
                    migrated += 1
                except Exception:
                    pass
            if migrated:
                # Rename old file to mark as migrated
                backup = self.store_path.with_suffix(".jsonl.migrated")
                self.store_path.rename(backup)
                logger.info(
                    f"Migrated {migrated} episodes from JSONL to SQLite; "
                    f"old file renamed to {backup}")
        except Exception as e:
            logger.warning(f"JSONL migration failed (non-fatal): {e}")

    def add(self, episode: Episode):
        self.episodes.append(episode)
        if self._unified:
            try:
                self._unified._add_episode_sync(
                    episode.problem_type, episode.difficulty,
                    episode.winning_strategy, episode.key_tactics,
                    episode.key_insight, episode.solve_time_ms)
            except Exception as e:
                logger.warning(f"Could not persist episode to unified store: {e}")
        else:
            # Legacy: append to JSONL
            try:
                self.store_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.store_path, "a") as f:
                    f.write(json.dumps(episode.__dict__) + "\n")
            except (OSError, IOError) as e:
                logger.warning(f"Could not persist episode: {e}")

    def retrieve_similar(self, problem_type: str, top_k: int = 3) -> list[Episode]:
        if not self.episodes:
            return []
        query_tokens = _tokenize(problem_type)
        scored: list[tuple[float, int, Episode]] = []
        for idx, e in enumerate(self.episodes):
            score = 0.0
            if e.problem_type == problem_type:
                score += 3.0
            elif problem_type in e.problem_type or e.problem_type in problem_type:
                score += 1.5
            insight_tokens = _tokenize(e.key_insight)
            score += len(query_tokens & insight_tokens) * 0.5
            tactic_tokens = set()
            for t in e.key_tactics:
                tactic_tokens |= _tokenize(t)
            score += len(query_tokens & tactic_tokens) * 0.3
            recency = idx / max(1, len(self.episodes))
            score += recency * 0.1
            if score > 0:
                scored.append((score, idx, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, _, e in scored[:top_k]]

    def retrieve_by_difficulty(self, difficulty: str, top_k: int = 3) -> list[Episode]:
        matching = [e for e in reversed(self.episodes) if e.difficulty == difficulty]
        return matching[:top_k]

    def retrieve_by_tactic(self, tactic: str, top_k: int = 3) -> list[Episode]:
        matching = [e for e in reversed(self.episodes)
                    if tactic in e.key_tactics]
        return matching[:top_k]


def _tokenize(text: str) -> set[str]:
    import re
    return set(re.findall(r'[a-z0-9_]+', text.lower())) - _STOP_WORDS


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "and", "or", "but", "not", "no", "if", "then", "than", "that",
    "this", "it", "its", "all", "each", "any", "some",
})
