"""Core expression type — the heart of the type theory.

All expressions are immutable (frozen dataclass or tuple-based).
This enables O(1) structural sharing in persistent proof states.
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
    # Payload fields (meaning depends on tag)
    name: Optional[Name] = None
    level: Optional[Level] = None
    idx: Optional[int] = None
    meta_id: Optional[MetaId] = None
    binder_info: BinderInfo = BinderInfo.DEFAULT
    children: Tuple[Expr, ...] = ()
    
    # ── Smart constructors ──
    @staticmethod
    def bvar(idx: int) -> Expr: return Expr("bvar", idx=idx)
    @staticmethod
    def fvar(fid: int) -> Expr: return Expr("fvar", idx=fid)
    @staticmethod
    def mvar(mid: MetaId) -> Expr: return Expr("mvar", meta_id=mid)
    @staticmethod
    def sort(level: Level) -> Expr: return Expr("sort", level=level)
    @staticmethod
    def prop() -> Expr: return Expr.sort(Level.zero())
    @staticmethod
    def type_() -> Expr: return Expr.sort(Level.one())
    @staticmethod
    def const(name: Name, levels=()) -> Expr: return Expr("const", name=name)
    @staticmethod
    def app(fn: Expr, arg: Expr) -> Expr: return Expr("app", children=(fn, arg))
    @staticmethod
    def lam(bi: BinderInfo, name: Name, domain: Expr, body: Expr) -> Expr:
        return Expr("lam", name=name, binder_info=bi, children=(domain, body))
    @staticmethod
    def pi(bi: BinderInfo, name: Name, domain: Expr, body: Expr) -> Expr:
        return Expr("pi", name=name, binder_info=bi, children=(domain, body))
    @staticmethod
    def arrow(a: Expr, b: Expr) -> Expr:
        return Expr.pi(BinderInfo.DEFAULT, Name.anon(), a, b)
    
    # ── Predicates ──
    @property
    def is_mvar(self): return self.tag == "mvar"
    @property
    def is_pi(self): return self.tag == "pi"
    @property
    def is_app(self): return self.tag == "app"
    
    def get_app_fn_name(self) -> Optional[Name]:
        e = self
        while e.tag == "app": e = e.children[0]
        return e.name if e.tag == "const" else None
    
    def collect_mvars(self) -> list[MetaId]:
        result = []
        self._collect_mvars(result)
        return result
    
    def _collect_mvars(self, acc):
        if self.tag == "mvar" and self.meta_id: acc.append(self.meta_id)
        for c in self.children: c._collect_mvars(acc)
    
    def instantiate(self, val: Expr, depth: int = 0) -> Expr:
        if self.tag == "bvar":
            if self.idx == depth: return val
            elif self.idx > depth: return Expr.bvar(self.idx - 1)
            else: return self
        elif self.tag in ("app", "lam", "pi"):
            new_children = []
            for i, c in enumerate(self.children):
                d = depth + 1 if (self.tag in ("lam", "pi") and i == 1) else depth
                new_children.append(c.instantiate(val, d))
            return Expr(self.tag, self.name, self.level, self.idx,
                       self.meta_id, self.binder_info, tuple(new_children))
        return self

    def __repr__(self):
        if self.tag == "bvar": return f"#{self.idx}"
        if self.tag == "fvar": return f"f{self.idx}"
        if self.tag == "mvar": return f"{self.meta_id}"
        if self.tag == "sort": return "Prop" if self.level == Level.zero() else f"Sort {self.level}"
        if self.tag == "const": return str(self.name)
        if self.tag == "app": return f"({self.children[0]} {self.children[1]})"
        if self.tag == "pi":
            if self.name and not self.name.is_anon():
                return f"(∀ ({self.name} : {self.children[0]}), {self.children[1]})"
            return f"({self.children[0]} → {self.children[1]})"
        if self.tag == "lam":
            return f"(fun ({self.name} : {self.children[0]}) => {self.children[1]})"
        return f"Expr({self.tag})"
