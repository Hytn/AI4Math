"""Core expression type — the heart of the type theory.

All expressions are immutable (frozen dataclass).
This enables O(1) structural sharing in persistent proof states.

De Bruijn index convention:
  - bvar(0) refers to the innermost binder
  - instantiate(val) replaces bvar(0) with val (shifting val under binders)
  - abstract(fvar_id) replaces fvar(id) with bvar(0) (closing a scope)
  - lift(n, cutoff) increments bvar indices >= cutoff by n
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
from .name import Name
from .universe import Level
from .meta import MetaId


class BinderInfo(Enum):
    DEFAULT = "default"
    IMPLICIT = "implicit"
    INST_IMPLICIT = "inst_implicit"
    STRICT_IMPLICIT = "strict_implicit"


@dataclass(frozen=True)
class Expr:
    """Immutable expression node."""
    tag: str  # bvar, fvar, mvar, sort, const, app, lam, pi, let, lit
    name: Optional[Name] = None
    level: Optional[Level] = None
    idx: Optional[int] = None
    meta_id: Optional[MetaId] = None
    binder_info: BinderInfo = BinderInfo.DEFAULT
    children: Tuple['Expr', ...] = ()

    # ── Smart constructors ──
    @staticmethod
    def bvar(idx: int) -> Expr:
        return Expr("bvar", idx=idx)

    @staticmethod
    def fvar(fid: int) -> Expr:
        return Expr("fvar", idx=fid)

    @staticmethod
    def mvar(mid: MetaId) -> Expr:
        return Expr("mvar", meta_id=mid)

    @staticmethod
    def sort(level: Level) -> Expr:
        return Expr("sort", level=level)

    @staticmethod
    def prop() -> Expr:
        return Expr.sort(Level.zero())

    @staticmethod
    def type_() -> Expr:
        return Expr.sort(Level.one())

    @staticmethod
    def const(name: Name, levels=()) -> Expr:
        return Expr("const", name=name)

    @staticmethod
    def app(fn: Expr, arg: Expr) -> Expr:
        return Expr("app", children=(fn, arg))

    @staticmethod
    def lam(bi: BinderInfo, name: Name, domain: Expr, body: Expr) -> Expr:
        return Expr("lam", name=name, binder_info=bi, children=(domain, body))

    @staticmethod
    def pi(bi: BinderInfo, name: Name, domain: Expr, body: Expr) -> Expr:
        return Expr("pi", name=name, binder_info=bi, children=(domain, body))

    @staticmethod
    def arrow(a: Expr, b: Expr) -> Expr:
        return Expr.pi(BinderInfo.DEFAULT, Name.anon(), a, b)

    @staticmethod
    def let_(name: Name, type_: Expr, value: Expr, body: Expr) -> Expr:
        return Expr("let", name=name, children=(type_, value, body))

    # ── Predicates ──
    @property
    def is_mvar(self): return self.tag == "mvar"
    @property
    def is_pi(self): return self.tag == "pi"
    @property
    def is_lam(self): return self.tag == "lam"
    @property
    def is_app(self): return self.tag == "app"
    @property
    def is_sort(self): return self.tag == "sort"
    @property
    def is_let(self): return self.tag == "let"

    def get_app_fn(self) -> Expr:
        """Get the head function of an application chain."""
        e = self
        while e.tag == "app" and len(e.children) == 2:
            e = e.children[0]
        return e

    def get_app_fn_name(self) -> Optional[Name]:
        fn = self.get_app_fn()
        return fn.name if fn.tag == "const" else None

    def get_app_args(self) -> list[Expr]:
        """Get all arguments in an application chain."""
        args = []
        e = self
        while e.tag == "app" and len(e.children) == 2:
            args.append(e.children[1])
            e = e.children[0]
        args.reverse()
        return args

    def collect_mvars(self) -> list[MetaId]:
        result = []
        self._collect_mvars(result)
        return result

    def _collect_mvars(self, acc):
        if self.tag == "mvar" and self.meta_id:
            acc.append(self.meta_id)
        for c in self.children:
            c._collect_mvars(acc)

    # ── De Bruijn operations ──

    def has_loose_bvars(self, depth: int = 0) -> bool:
        """Check if expression has unbound de Bruijn variables at given depth."""
        if self.tag == "bvar":
            return self.idx is not None and self.idx >= depth
        if self.tag in ("fvar", "mvar", "sort", "const"):
            return False
        for i, c in enumerate(self.children):
            d = depth
            if self.tag in ("lam", "pi") and i == 1:
                d = depth + 1
            elif self.tag == "let" and i == 2:
                d = depth + 1
            if c.has_loose_bvars(d):
                return True
        return False

    def lift(self, n: int, cutoff: int = 0) -> Expr:
        """Lift (shift) free bvar indices by n, above cutoff.

        lift(n, c): for bvar(i), if i >= c then bvar(i+n) else bvar(i).
        Used to prevent variable capture when substituting under binders.

        When n < 0 (lowering), guards against producing negative indices.
        """
        if n == 0:
            return self
        if self.tag == "bvar":
            if self.idx is not None and self.idx >= cutoff:
                new_idx = self.idx + n
                if new_idx < 0:
                    raise ValueError(
                        f"lift produced negative bvar index: "
                        f"bvar({self.idx}) + {n} = {new_idx}")
                return Expr.bvar(new_idx)
            return self
        if self.tag in ("fvar", "mvar", "sort", "const"):
            return self
        new_children = []
        for i, c in enumerate(self.children):
            co = cutoff
            if self.tag in ("lam", "pi") and i == 1:
                co = cutoff + 1
            elif self.tag == "let" and i == 2:
                co = cutoff + 1
            new_children.append(c.lift(n, co))
        return Expr(self.tag, self.name, self.level, self.idx,
                    self.meta_id, self.binder_info, tuple(new_children))

    def instantiate(self, val: Expr, depth: int = 0) -> Expr:
        """Replace bvar(depth) with val (properly lifting val under binders).

        This is the key operation: when we go under a lam/pi binder,
        depth increases AND val must be lifted to avoid capture.

        Example: (fun x => fun y => #1)(t) should give (fun y => t'),
        where t' = lift(1, 0)(t) to shift past the y binder.
        """
        if self.tag == "bvar":
            if self.idx == depth:
                # Replace with val, lifted to current depth
                return val.lift(depth, 0) if depth > 0 else val
            elif self.idx is not None and self.idx > depth:
                return Expr.bvar(self.idx - 1)
            return self
        if self.tag in ("fvar", "mvar", "sort", "const"):
            return self
        new_children = []
        for i, c in enumerate(self.children):
            d = depth
            if self.tag in ("lam", "pi") and i == 1:
                d = depth + 1
            elif self.tag == "let" and i == 2:
                d = depth + 1
            new_children.append(c.instantiate(val, d))
        return Expr(self.tag, self.name, self.level, self.idx,
                    self.meta_id, self.binder_info, tuple(new_children))

    def abstract(self, fvar_id: int, depth: int = 0) -> Expr:
        """Replace fvar(fvar_id) with bvar(depth) — close a scope.

        This is the inverse of instantiate: converts a free variable
        back into a bound variable at the given depth.
        """
        if self.tag == "fvar" and self.idx == fvar_id:
            return Expr.bvar(depth)
        if self.tag in ("bvar", "mvar", "sort", "const"):
            return self
        new_children = []
        for i, c in enumerate(self.children):
            d = depth
            if self.tag in ("lam", "pi") and i == 1:
                d = depth + 1
            elif self.tag == "let" and i == 2:
                d = depth + 1
            new_children.append(c.abstract(fvar_id, d))
        return Expr(self.tag, self.name, self.level, self.idx,
                    self.meta_id, self.binder_info, tuple(new_children))

    def replace_fvar(self, fvar_id: int, replacement: Expr) -> Expr:
        """Replace all occurrences of fvar(fvar_id) with replacement."""
        if self.tag == "fvar" and self.idx == fvar_id:
            return replacement
        if self.tag in ("bvar", "mvar", "sort", "const"):
            return self
        new_children = tuple(c.replace_fvar(fvar_id, replacement) for c in self.children)
        if new_children == self.children:
            return self
        return Expr(self.tag, self.name, self.level, self.idx,
                    self.meta_id, self.binder_info, new_children)

    def __repr__(self):
        if self.tag == "bvar": return f"#{self.idx}"
        if self.tag == "fvar": return f"f{self.idx}"
        if self.tag == "mvar": return f"{self.meta_id}"
        if self.tag == "sort":
            return "Prop" if self.level and self.level.is_zero else f"Sort {self.level}"
        if self.tag == "const": return str(self.name)
        if self.tag == "app" and len(self.children) == 2:
            return f"({self.children[0]} {self.children[1]})"
        if self.tag == "pi" and len(self.children) == 2:
            if self.name and not self.name.is_anon():
                return f"(∀ ({self.name} : {self.children[0]}), {self.children[1]})"
            return f"({self.children[0]} → {self.children[1]})"
        if self.tag == "lam" and len(self.children) == 2:
            return f"(fun ({self.name} : {self.children[0]}) => {self.children[1]})"
        if self.tag == "let" and len(self.children) == 3:
            return f"(let {self.name} : {self.children[0]} := {self.children[1]} in {self.children[2]})"
        return f"Expr({self.tag})"
