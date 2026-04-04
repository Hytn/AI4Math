"""Global environment — shared, read-only knowledge base.

Supports:
  - Constants (definitions, axioms)
  - Inductive types with constructors and recursors
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from pyrsistent import pmap, PMap
from .name import Name
from .expr import Expr


@dataclass
class ConstantInfo:
    """A constant declaration (theorem, def, axiom)."""
    name: Name
    type_: Expr
    value: Optional[Expr] = None       # None = opaque/axiom
    is_reducible: bool = True          # Can be delta-unfolded?


@dataclass
class ConstructorInfo:
    """A constructor for an inductive type."""
    name: Name
    type_: Expr           # Full type including inductive type args
    inductive_name: Name  # Which inductive this belongs to
    idx: int = 0          # Constructor index (0, 1, ...)


@dataclass
class RecursorInfo:
    """A recursor (eliminator) for an inductive type."""
    name: Name
    type_: Expr
    inductive_name: Name
    num_params: int = 0     # Number of type parameters
    num_indices: int = 0    # Number of indices
    num_motives: int = 1    # Number of motive arguments
    num_minors: int = 0     # Number of minor premises (one per constructor)
    is_prop_elim: bool = False  # True if eliminating into Prop


@dataclass
class InductiveInfo:
    """An inductive type definition."""
    name: Name
    type_: Expr                        # Type of the inductive type itself
    constructors: list[ConstructorInfo] = field(default_factory=list)
    recursor: Optional[RecursorInfo] = None
    num_params: int = 0                # Number of fixed parameters
    num_indices: int = 0               # Number of varying indices
    is_recursive: bool = False
    is_prop: bool = False              # Lives in Prop?


class Environment:
    """Immutable global environment with constants and inductives."""

    def __init__(self):
        self._constants: PMap = pmap()
        self._inductives: PMap = pmap()
        self._constructors: PMap = pmap()   # ctor_name → InductiveInfo
        self._recursors: PMap = pmap()      # rec_name → RecursorInfo

    def add_const(self, info: ConstantInfo) -> Environment:
        env = Environment()
        env._constants = self._constants.set(info.name, info)
        env._inductives = self._inductives
        env._constructors = self._constructors
        env._recursors = self._recursors
        return env

    def add_inductive(self, info: InductiveInfo) -> Environment:
        env = Environment()
        env._constants = self._constants.set(info.name,
            ConstantInfo(info.name, info.type_))
        for ctor in info.constructors:
            env._constants = env._constants.set(ctor.name,
                ConstantInfo(ctor.name, ctor.type_))
        env._inductives = self._inductives.set(info.name, info)
        ctors = self._constructors
        for ctor in info.constructors:
            ctors = ctors.set(ctor.name, info)
        env._constructors = ctors
        recs = self._recursors
        if info.recursor:
            env._constants = env._constants.set(info.recursor.name,
                ConstantInfo(info.recursor.name, info.recursor.type_))
            recs = recs.set(info.recursor.name, info.recursor)
        env._recursors = recs
        return env

    def lookup(self, name: Name) -> Optional[ConstantInfo]:
        return self._constants.get(name)

    def lookup_inductive(self, name: Name) -> Optional[InductiveInfo]:
        return self._inductives.get(name)

    def lookup_constructor_inductive(self, ctor_name: Name) -> Optional[InductiveInfo]:
        """Given a constructor name, find its parent inductive."""
        return self._constructors.get(ctor_name)

    def lookup_recursor(self, name: Name) -> Optional[RecursorInfo]:
        return self._recursors.get(name)

    def is_constructor(self, name: Name) -> bool:
        return name in self._constructors

    def is_recursor(self, name: Name) -> bool:
        return name in self._recursors

    def __len__(self):
        return len(self._constants)
