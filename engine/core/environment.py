"""Global environment — shared, read-only knowledge base."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from pyrsistent import pmap, PMap
from .name import Name
from .expr import Expr

@dataclass
class ConstantInfo:
    name: Name
    type_: Expr
    value: Optional[Expr] = None

class Environment:
    def __init__(self):
        self._constants: PMap = pmap()
    def add_const(self, info: ConstantInfo) -> Environment:
        env = Environment()
        env._constants = self._constants.set(info.name, info)
        return env
    def lookup(self, name: Name) -> Optional[ConstantInfo]:
        return self._constants.get(name)
    def __len__(self): return len(self._constants)
