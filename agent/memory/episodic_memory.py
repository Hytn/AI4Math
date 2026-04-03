"""agent/memory/episodic_memory.py — 情景记忆: 历史解题经验"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field

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
        if self.store_path.exists():
            for line in self.store_path.read_text().strip().split("\n"):
                if line: self.episodes.append(Episode(**json.loads(line)))

    def add(self, episode: Episode):
        self.episodes.append(episode)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "a") as f:
            f.write(json.dumps(episode.__dict__) + "\n")

    def retrieve_similar(self, problem_type: str, top_k: int = 3) -> list[Episode]:
        return [e for e in self.episodes if e.problem_type == problem_type][:top_k]
