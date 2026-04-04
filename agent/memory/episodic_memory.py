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
        """Retrieve episodes matching the problem type, most recent first."""
        matching = [e for e in reversed(self.episodes) if e.problem_type == problem_type]
        return matching[:top_k]

    def retrieve_by_difficulty(self, difficulty: str, top_k: int = 3) -> list[Episode]:
        """Retrieve episodes matching the difficulty level."""
        matching = [e for e in reversed(self.episodes) if e.difficulty == difficulty]
        return matching[:top_k]
