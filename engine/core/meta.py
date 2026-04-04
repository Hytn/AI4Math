"""Metavariable identifier."""
from dataclasses import dataclass

@dataclass(frozen=True)
class MetaId:
    id: int
    def __hash__(self): return hash(self.id)
    def __repr__(self): return f"?m{self.id}"
