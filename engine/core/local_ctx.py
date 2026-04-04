"""Local context with persistent data structure."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterator
from pyrsistent import pmap, PMap
from .name import Name
from .expr import Expr

@dataclass(frozen=True)
class FVarId:
    id: int
    def __hash__(self): return hash(self.id)

@dataclass(frozen=True)
class LocalDecl:
    fvar_id: FVarId
    user_name: Name
    type_: Expr
    value: Optional[Expr] = None
    index: int = 0

class LocalContext:
    """Persistent local context. All mutations return new instances."""
    __slots__ = ('_decls', '_next_idx')
    def __init__(self, decls: PMap = pmap(), next_idx: int = 0):
        object.__setattr__(self, '_decls', decls)
        object.__setattr__(self, '_next_idx', next_idx)
    def push_hyp(self, fvar_id: FVarId, name: Name, ty: Expr) -> LocalContext:
        decl = LocalDecl(fvar_id, name, ty, None, self._next_idx)
        return LocalContext(self._decls.set(fvar_id, decl), self._next_idx + 1)
    def push_let(self, fvar_id: FVarId, name: Name, ty: Expr, val: Expr) -> LocalContext:
        decl = LocalDecl(fvar_id, name, ty, val, self._next_idx)
        return LocalContext(self._decls.set(fvar_id, decl), self._next_idx + 1)
    def get(self, fid: FVarId) -> Optional[LocalDecl]: return self._decls.get(fid)
    def find_by_name(self, name: Name) -> Optional[LocalDecl]:
        for d in self._decls.values():
            if d.user_name == name: return d
        return None
    def __len__(self): return len(self._decls)
    def __iter__(self) -> Iterator[LocalDecl]: return iter(self._decls.values())
