"""agent/memory/episodic_memory.py — 情景记忆: 历史解题经验"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field

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
    def __init__(self, store_path: str = "results/episodic_memory.jsonl"):
        self.store_path = Path(store_path)
        self.episodes: list[Episode] = []
        self._load()

    def _load(self):
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

    def add(self, episode: Episode):
        self.episodes.append(episode)
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, "a") as f:
                f.write(json.dumps(episode.__dict__) + "\n")
        except (OSError, IOError) as e:
            logger.warning(f"Could not persist episode: {e}")

    def retrieve_similar(self, problem_type: str, top_k: int = 3) -> list[Episode]:
        """Retrieve episodes by relevance score, not just exact problem_type match.

        Scoring: exact problem_type match (3 pts) + keyword overlap in
        key_insight and key_tactics (1 pt per keyword hit).
        Falls back to most-recent if no matches found.
        """
        if not self.episodes:
            return []

        query_tokens = _tokenize(problem_type)

        scored: list[tuple[float, int, Episode]] = []
        for idx, e in enumerate(self.episodes):
            score = 0.0
            # Exact type match
            if e.problem_type == problem_type:
                score += 3.0
            # Partial type match (substring)
            elif problem_type in e.problem_type or e.problem_type in problem_type:
                score += 1.5
            # Keyword overlap with key_insight
            insight_tokens = _tokenize(e.key_insight)
            score += len(query_tokens & insight_tokens) * 0.5
            # Keyword overlap with key_tactics
            tactic_tokens = set()
            for t in e.key_tactics:
                tactic_tokens |= _tokenize(t)
            score += len(query_tokens & tactic_tokens) * 0.3
            # Recency bonus (newer = slightly higher)
            recency = idx / max(1, len(self.episodes))
            score += recency * 0.1

            if score > 0:
                scored.append((score, idx, e))

        scored.sort(key=lambda x: -x[0])
        return [e for _, _, e in scored[:top_k]]

    def retrieve_by_difficulty(self, difficulty: str, top_k: int = 3) -> list[Episode]:
        """Retrieve episodes matching the difficulty level."""
        matching = [e for e in reversed(self.episodes) if e.difficulty == difficulty]
        return matching[:top_k]

    def retrieve_by_tactic(self, tactic: str, top_k: int = 3) -> list[Episode]:
        """Retrieve episodes where a specific tactic was key to the solution."""
        matching = [e for e in reversed(self.episodes)
                    if tactic in e.key_tactics]
        return matching[:top_k]


def _tokenize(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens for keyword matching."""
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
