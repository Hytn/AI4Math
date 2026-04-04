"""Persistent proof state — O(1) fork, O(1) backtrack."""
from __future__ import annotations
import itertools
from typing import Optional
from pyrsistent import pvector
from engine.core import Expr, MetaId, Name, LocalContext, FVarId, Environment
from .meta_ctx import MetaContext
from .goal import Goal

# Thread-safe atomic counter (itertools.count is implemented in C and is
# effectively atomic for single __next__ calls in CPython)
_id_counter = itertools.count(1)

class ProofState:
    """Immutable proof state. Clone is O(1). Fork is O(1)."""
    __slots__ = ('env', 'meta_ctx', 'focus', 'id', 'parent_id', '_next_fvar')

    def __init__(self, env, meta_ctx, focus, parent_id=None, next_fvar=0):
        object.__setattr__(self, 'env', env)
        object.__setattr__(self, 'meta_ctx', meta_ctx)
        object.__setattr__(self, 'focus', focus)
        object.__setattr__(self, 'id', next(_id_counter))
        object.__setattr__(self, 'parent_id', parent_id)
        object.__setattr__(self, '_next_fvar', next_fvar)

    @staticmethod
    def new(env: Environment, goal_type: Expr) -> ProofState:
        mc = MetaContext()
        mc, gid = mc.create_meta(LocalContext(), goal_type, depth=0)
        return ProofState(env, mc, pvector([gid]))

    def main_goal(self) -> Optional[Goal]:
        if not self.focus: return None
        mid = self.focus[0]
        decl = self.meta_ctx.get_decl(mid)
        if not decl or self.meta_ctx.is_assigned(mid): return None
        return Goal(mid, decl.local_ctx, decl.target, decl.depth)

    def goals(self) -> list[Goal]:
        result = []
        for mid in self.focus:
            if self.meta_ctx.is_assigned(mid): continue
            decl = self.meta_ctx.get_decl(mid)
            if decl: result.append(Goal(mid, decl.local_ctx, decl.target, decl.depth))
        return result

    def is_complete(self) -> bool: return self.meta_ctx.is_complete()
    def num_goals(self) -> int: return len(self.goals())

    def fresh_fvar(self) -> tuple[ProofState, FVarId]:
        fid = FVarId(self._next_fvar)
        ns = ProofState(self.env, self.meta_ctx, self.focus, self.id, self._next_fvar + 1)
        return ns, fid

    def assign_goal(self, goal_id: MetaId, proof_term: Expr) -> ProofState:
        new_mc = self.meta_ctx.assign(goal_id, proof_term)
        new_focus = pvector(m for m in self.focus if m != goal_id)
        return ProofState(self.env, new_mc, new_focus, self.id, self._next_fvar)

    def replace_main_goal(self, new_mc: MetaContext, new_goals: list[MetaId]) -> ProofState:
        new_focus = pvector(new_goals) + self.focus[1:]
        return ProofState(self.env, new_mc, new_focus, self.id, self._next_fvar)
