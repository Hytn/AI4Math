"""prover/lemma_bank/bank.py — 已证引理银行（带持久化）"""
from __future__ import annotations
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProvedLemma:
    name: str
    statement: str
    proof: str
    source_attempt: int = 0
    source_rollout: int = 0
    verified: bool = True

    def to_lean(self) -> str:
        return f"{self.statement} {self.proof}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "statement": self.statement,
            "proof": self.proof,
            "source_attempt": self.source_attempt,
            "source_rollout": self.source_rollout,
            "verified": self.verified,
        }

    @staticmethod
    def from_dict(d: dict) -> ProvedLemma:
        return ProvedLemma(
            name=d["name"],
            statement=d["statement"],
            proof=d["proof"],
            source_attempt=d.get("source_attempt", 0),
            source_rollout=d.get("source_rollout", 0),
            verified=d.get("verified", True),
        )


class LemmaBank:
    """Thread-safe lemma bank with optional persistence."""

    def __init__(self, persist_path: str = ""):
        self.lemmas: list[ProvedLemma] = []
        self._seen: set = set()
        self._lock = threading.Lock()
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path:
            self._load()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.lemmas)

    def add(self, lemma: ProvedLemma):
        key = lemma.statement.strip().lower()
        with self._lock:
            if key not in self._seen:
                self._seen.add(key)
                self.lemmas.append(lemma)
                self._save_incremental(lemma)

    def to_prompt_context(self, max_lemmas: int = 10) -> str:
        with self._lock:
            if not self.lemmas:
                return ""
            parts = ["## Already proved lemmas (verified by Lean kernel)\n"]
            for l in self.lemmas[-max_lemmas:]:
                parts.append(f"```lean\n{l.to_lean()}\n```\n")
            return "\n".join(parts)

    def to_lean_preamble(self, max_lemmas: int = 20) -> str:
        with self._lock:
            if not self.lemmas:
                return ""
            return "\n".join(l.to_lean() for l in self.lemmas[-max_lemmas:])

    def get_rl_experience(self) -> list[dict]:
        with self._lock:
            return [l.to_dict() for l in self.lemmas]

    def clear(self):
        with self._lock:
            self.lemmas.clear()
            self._seen.clear()

    def _load(self):
        """Load lemmas from persist file."""
        if self._persist_path and self._persist_path.exists():
            try:
                for line in self._persist_path.read_text().strip().split("\n"):
                    if line.strip():
                        d = json.loads(line)
                        lemma = ProvedLemma.from_dict(d)
                        key = lemma.statement.strip().lower()
                        if key not in self._seen:
                            self._seen.add(key)
                            self.lemmas.append(lemma)
            except (json.JSONDecodeError, KeyError) as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to load lemma bank from {self._persist_path}: {e}")

    def _save_incremental(self, lemma: ProvedLemma):
        """Append a single lemma to persist file."""
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "a") as f:
                f.write(json.dumps(lemma.to_dict()) + "\n")
