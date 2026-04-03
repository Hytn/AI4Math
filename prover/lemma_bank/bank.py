"""prover/lemma_bank/bank.py — 已证引理银行"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class ProvedLemma:
    name: str; statement: str; proof: str; source_attempt: int = 0
    source_rollout: int = 0; verified: bool = True
    def to_lean(self) -> str: return f"{self.statement} {self.proof}"

class LemmaBank:
    def __init__(self): self.lemmas: list[ProvedLemma] = []; self._seen: set = set()
    @property
    def count(self) -> int: return len(self.lemmas)

    def add(self, lemma: ProvedLemma):
        key = lemma.statement.strip().lower()
        if key not in self._seen: self._seen.add(key); self.lemmas.append(lemma)

    def to_prompt_context(self, max_lemmas: int = 10) -> str:
        if not self.lemmas: return ""
        parts = ["## Already proved lemmas (verified by Lean kernel)\n"]
        for l in self.lemmas[-max_lemmas:]:
            parts.append(f"```lean\n{l.to_lean()}\n```\n")
        return "\n".join(parts)

    def to_lean_preamble(self, max_lemmas: int = 20) -> str:
        if not self.lemmas: return ""
        return "\n".join(l.to_lean() for l in self.lemmas[-max_lemmas:])

    def get_rl_experience(self) -> list[dict]:
        return [{"name": l.name, "statement": l.statement, "proof": l.proof,
                 "verified": l.verified} for l in self.lemmas]

    def clear(self): self.lemmas.clear(); self._seen.clear()
