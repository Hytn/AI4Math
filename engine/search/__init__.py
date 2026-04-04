"""Search coordinator: manages tree search and dispatches batch tactics."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
from engine.core import Expr, Name, MetaId
from engine.state import ProofState, SearchTree, NodeId, NodeStatus, GoalView
from engine.tactic import intro, assumption, sorry, exact, apply, TacticResult

@dataclass
class ExpansionResult:
    parent_node: int; tactic: str; success: bool
    child_node: Optional[int] = None
    new_goals: list = field(default_factory=list)
    error: Optional[dict] = None
    elapsed_us: int = 0; is_complete: bool = False

class SearchCoordinator:
    def __init__(self, env, goal_type: Expr):
        self._env = env
        state = ProofState.new(env, goal_type)
        self._tree = SearchTree(state)

    def goal_view(self, node_id: int) -> list[GoalView]:
        node = self._tree.get(NodeId(node_id))
        if not node: return []
        return [GoalView.from_goal(g, node.state) for g in node.state.goals()]

    def try_tactic(self, node_id: int, tactic_str: str) -> ExpansionResult:
        t0 = time.perf_counter_ns()
        node = self._tree.get(NodeId(node_id))
        if not node:
            return ExpansionResult(node_id, tactic_str, False,
                                  error={"kind": "not_found"}, elapsed_us=0)
        result = self._exec(node.state, tactic_str)
        elapsed = int((time.perf_counter_ns() - t0) / 1000)
        if result.success:
            complete = result.state.is_complete()
            goals = [GoalView.from_goal(g, result.state) for g in result.state.goals()]
            self._tree, child_id = self._tree.expand(NodeId(node_id), tactic_str, result.state)
            return ExpansionResult(node_id, tactic_str, True, child_id.id,
                                  goals, elapsed_us=elapsed, is_complete=complete)
        err = None
        if result.error:
            if hasattr(result.error, 'kind'):
                err = {"kind": result.error.kind, "message": result.error.message}
            elif isinstance(result.error, dict):
                err = result.error
        return ExpansionResult(node_id, tactic_str, False, error=err, elapsed_us=elapsed)

    def try_batch(self, node_id: int, tactics: list[str]) -> list[ExpansionResult]:
        return [self.try_tactic(node_id, t) for t in tactics]

    def stats(self):
        return {"total_nodes": self._tree.size(), "open_leaves": len(self._tree.open_leaves()),
                "is_solved": self._tree.root().status == NodeStatus.SOLVED}

    def _exec(self, state: ProofState, tactic_str: str) -> TacticResult:
        parts = tactic_str.strip().split(None, 1)
        name = parts[0]; arg = parts[1] if len(parts) > 1 else ""
        if name == "intro": return intro(state, arg or "h")
        if name == "assumption": return assumption(state)
        if name == "sorry": return sorry(state)
        if name == "exact": return exact(state, arg or "h")
        if name == "apply": return apply(state, arg or "h")
        from engine.tactic import TacticError as TE
        return TacticResult(error=TE("unknown", f"unknown tactic: {name}", name))
