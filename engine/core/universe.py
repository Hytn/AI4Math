"""Universe levels."""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Level:
    kind: str  # "zero", "succ", "max", "param"
    value: object = None
    
    @staticmethod
    def zero(): return Level("zero")
    @staticmethod
    def one(): return Level("succ", Level.zero())
    @staticmethod
    def succ(l): return Level("succ", l)
    
    def to_nat(self) -> Optional[int]:
        if self.kind == "zero": return 0
        if self.kind == "succ" and self.value: 
            n = self.value.to_nat()
            return n + 1 if n is not None else None
        return None
    
    def __repr__(self):
        n = self.to_nat()
        return str(n) if n is not None else f"Level({self.kind})"
