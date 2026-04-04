"""Persistent metavariable context with explicit dependency tracking."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pyrsistent import pmap, PMap, pset, PSet
from engine.core import Expr, MetaId, Name, LocalContext

@dataclass
class MetaVarDecl:
    id: MetaId
    local_ctx: LocalContext
    target: Expr
    user_name: Optional[Name] = None
    depth: int = 0

class MetaContext:
    __slots__ = ('_decls', '_assignments', '_deps', '_next_id')
    def __init__(self, decls=pmap(), assignments=pmap(), deps=pmap(), next_id=0):
        object.__setattr__(self, '_decls', decls)
        object.__setattr__(self, '_assignments', assignments)
        object.__setattr__(self, '_deps', deps)
        object.__setattr__(self, '_next_id', next_id)

    def create_meta(self, ctx: LocalContext, target: Expr,
                    user_name=None, depth=0) -> tuple[MetaContext, MetaId]:
        mid = MetaId(self._next_id)
        decl = MetaVarDecl(mid, ctx, target, user_name, depth)
        new_deps = self._deps
        for dep_id in target.collect_mvars():
            existing = new_deps.get(dep_id, pset())
            new_deps = new_deps.set(dep_id, existing.add(mid))
        return MetaContext(self._decls.set(mid, decl), self._assignments,
                          new_deps, self._next_id + 1), mid

    def assign(self, mid: MetaId, value: Expr) -> MetaContext:
        return MetaContext(self._decls, self._assignments.set(mid, value),
                          self._deps, self._next_id)

    def get_decl(self, mid: MetaId) -> Optional[MetaVarDecl]: return self._decls.get(mid)
    def get_assignment(self, mid: MetaId) -> Optional[Expr]: return self._assignments.get(mid)
    def is_assigned(self, mid: MetaId) -> bool: return mid in self._assignments
    @property
    def assignments(self) -> PMap: return self._assignments
    def unsolved(self) -> list[MetaId]:
        return [k for k in self._decls if k not in self._assignments]
    def is_complete(self) -> bool: return len(self.unsolved()) == 0
    def num_unsolved(self) -> int: return len(self.unsolved())
    def are_independent(self, a: MetaId, b: MetaId) -> bool:
        a_deps = self._deps.get(a, pset())
        b_deps = self._deps.get(b, pset())
        return b not in a_deps and a not in b_deps
