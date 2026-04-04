"""Universe levels — proper Level algebra for CIC.

Supports: zero, succ, max, imax, param
With simplification and definitional equality checking.

In Lean4's type theory:
  - Sort 0 = Prop
  - Sort (n+1) = Type n
  - Pi type lives in Sort(imax(level_of_domain, level_of_codomain))
  - imax(a, 0) = 0  (propositions absorb)
  - imax(a, succ(b)) = max(a, succ(b))
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Level:
    kind: str  # "zero", "succ", "max", "imax", "param"
    value: object = None  # Level for succ, (Level, Level) for max/imax, str for param

    # ── Constructors ──

    @staticmethod
    def zero() -> Level:
        return Level("zero")

    @staticmethod
    def one() -> Level:
        return Level("succ", Level.zero())

    @staticmethod
    def succ(l: Level) -> Level:
        return Level("succ", l)

    @staticmethod
    def of_nat(n: int) -> Level:
        result = Level.zero()
        for _ in range(n):
            result = Level.succ(result)
        return result

    @staticmethod
    def max(a: Level, b: Level) -> Level:
        """max(a, b) — the larger of two levels."""
        # Simplify when both are concrete
        an, bn = a.to_nat(), b.to_nat()
        if an is not None and bn is not None:
            return Level.of_nat(max(an, bn))
        # max(a, a) = a
        if a == b:
            return a
        # max(0, b) = b, max(a, 0) = a
        if an == 0: return b
        if bn == 0: return a
        return Level("max", (a, b))

    @staticmethod
    def imax(a: Level, b: Level) -> Level:
        """imax(a, b) — impredicative max for Pi types.

        imax(a, 0) = 0      (propositions absorb — key to impredicativity)
        imax(a, succ(b)) = max(a, succ(b))
        """
        bn = b.to_nat()
        if bn is not None:
            if bn == 0:
                return Level.zero()  # Prop absorbs
            an = a.to_nat()
            if an is not None:
                return Level.of_nat(max(an, bn))
        # If b is definitely not zero (it's succ of something)
        if b.kind == "succ":
            return Level.max(a, b)
        return Level("imax", (a, b))

    @staticmethod
    def param(name: str) -> Level:
        """Universe parameter (polymorphic level variable)."""
        return Level("param", name)

    # ── Queries ──

    def to_nat(self) -> Optional[int]:
        """Try to evaluate to a concrete natural number."""
        if self.kind == "zero":
            return 0
        if self.kind == "succ" and isinstance(self.value, Level):
            n = self.value.to_nat()
            return n + 1 if n is not None else None
        if self.kind == "max" and isinstance(self.value, tuple):
            a, b = self.value
            an, bn = a.to_nat(), b.to_nat()
            if an is not None and bn is not None:
                return max(an, bn)
        if self.kind == "imax" and isinstance(self.value, tuple):
            a, b = self.value
            bn = b.to_nat()
            if bn is not None:
                if bn == 0:
                    return 0
                an = a.to_nat()
                if an is not None:
                    return max(an, bn)
        return None

    @property
    def is_zero(self) -> bool:
        n = self.to_nat()
        return n is not None and n == 0

    @property
    def is_nonzero(self) -> bool:
        """Definitely nonzero (succ of something)."""
        if self.kind == "succ":
            return True
        n = self.to_nat()
        return n is not None and n > 0

    def is_leq(self, other: Level) -> Optional[bool]:
        """Check if self ≤ other. Returns None if undecidable."""
        sn, on = self.to_nat(), other.to_nat()
        if sn is not None and on is not None:
            return sn <= on
        if self == other:
            return True
        return None

    # ── Substitution for universe polymorphism ──

    def subst(self, param_name: str, replacement: Level) -> Level:
        """Substitute a universe parameter with a concrete level."""
        if self.kind == "param" and self.value == param_name:
            return replacement
        if self.kind == "succ" and isinstance(self.value, Level):
            return Level.succ(self.value.subst(param_name, replacement))
        if self.kind in ("max", "imax") and isinstance(self.value, tuple):
            a, b = self.value
            new_a = a.subst(param_name, replacement)
            new_b = b.subst(param_name, replacement)
            return Level.max(new_a, new_b) if self.kind == "max" else Level.imax(new_a, new_b)
        return self

    # ── Equality ──

    def is_equiv(self, other: Level) -> bool:
        """Check definitional equality of levels."""
        if self == other:
            return True
        sn, on = self.to_nat(), other.to_nat()
        if sn is not None and on is not None:
            return sn == on
        return False

    def __repr__(self):
        n = self.to_nat()
        if n is not None:
            return str(n)
        if self.kind == "param":
            return f"u.{self.value}"
        if self.kind == "succ":
            return f"({self.value}+1)"
        if self.kind == "max" and isinstance(self.value, tuple):
            return f"max({self.value[0]}, {self.value[1]})"
        if self.kind == "imax" and isinstance(self.value, tuple):
            return f"imax({self.value[0]}, {self.value[1]})"
        return f"Level({self.kind})"
