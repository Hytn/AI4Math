"""Pure tactics: ProofState -> TacticResult. No mutation, no side effects."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
from engine.core import Expr, MetaId, Name, BinderInfo, FVarId
from engine.state import ProofState

@dataclass
class TacticError:
    kind: str; message: str; tactic: str = ""; available: list = field(default_factory=list)

@dataclass
class TacticResult:
    state: Optional[ProofState] = None
    error: Optional[TacticError] = None
    elapsed_us: int = 0
    goals_before: int = 0; goals_after: int = 0
    @property
    def success(self): return self.state is not None

def intro(state: ProofState, name: str = "h") -> TacticResult:
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return TacticResult(error=TacticError("no_goals", "no goals", "intro"), goals_before=gb)
    if not goal.target.is_pi:
        return TacticResult(error=TacticError("goal_mismatch", "not a forall", "intro"),
                          elapsed_us=int((time.perf_counter_ns()-t0)/1000), goals_before=gb)
    domain, body = goal.target.children[0], goal.target.children[1]
    ns, fid = state.fresh_fvar()
    new_ctx = goal.local_ctx.push_hyp(fid, Name.from_str(name), domain)
    new_target = body.instantiate(Expr.fvar(fid.id))
    mc = ns.meta_ctx
    mc, new_gid = mc.create_meta(new_ctx, new_target, depth=goal.depth + 1)
    proof = Expr.lam(goal.target.binder_info, Name.from_str(name), domain, Expr.mvar(new_gid))
    mc = mc.assign(goal.id, proof)
    result = ns.replace_main_goal(mc, [new_gid])
    return TacticResult(result, elapsed_us=int((time.perf_counter_ns()-t0)/1000),
                       goals_before=gb, goals_after=result.num_goals())

def assumption(state: ProofState) -> TacticResult:
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return TacticResult(error=TacticError("no_goals", "no goals", "assumption"), goals_before=gb)
    for decl in goal.local_ctx:
        if repr(decl.type_) == repr(goal.target):
            proof = Expr.fvar(decl.fvar_id.id)
            result = state.assign_goal(goal.id, proof)
            return TacticResult(result, elapsed_us=int((time.perf_counter_ns()-t0)/1000),
                              goals_before=gb, goals_after=result.num_goals())
    return TacticResult(error=TacticError("no_match", "no hypothesis matches goal", "assumption"),
                       elapsed_us=int((time.perf_counter_ns()-t0)/1000), goals_before=gb)

def sorry(state: ProofState) -> TacticResult:
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return TacticResult(error=TacticError("no_goals", "no goals", "sorry"), goals_before=gb)
    result = state.assign_goal(goal.id, Expr.const(Name.from_str("APE.sorry")))
    return TacticResult(result, elapsed_us=int((time.perf_counter_ns()-t0)/1000),
                       goals_before=gb, goals_after=result.num_goals())

def exact(state: ProofState, name_str: str) -> TacticResult:
    """Close goal by providing a hypothesis by name."""
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return TacticResult(error=TacticError("no_goals", "no goals", "exact"), goals_before=gb)
    nm = Name.from_str(name_str)
    decl = goal.local_ctx.find_by_name(nm)
    if decl and repr(decl.type_) == repr(goal.target):
        proof = Expr.fvar(decl.fvar_id.id)
        result = state.assign_goal(goal.id, proof)
        return TacticResult(result, elapsed_us=int((time.perf_counter_ns()-t0)/1000),
                          goals_before=gb, goals_after=result.num_goals())
    available = [str(d.user_name) for d in goal.local_ctx]
    return TacticResult(error=TacticError("no_match", f"'{name_str}' doesn't match goal", "exact", available),
                       elapsed_us=int((time.perf_counter_ns()-t0)/1000), goals_before=gb)

def apply(state: ProofState, name_str: str) -> TacticResult:
    """Apply a hypothesis whose conclusion matches the goal."""
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return TacticResult(error=TacticError("no_goals", "no goals", "apply"), goals_before=gb)
    nm = Name.from_str(name_str)
    decl = goal.local_ctx.find_by_name(nm)
    if not decl:
        return TacticResult(error=TacticError("not_found", f"'{name_str}' not in context", "apply"),
                          elapsed_us=int((time.perf_counter_ns()-t0)/1000), goals_before=gb)
    # If hyp type is Pi, create subgoals for premises
    if decl.type_.is_pi:
        # simplified: just try to use its conclusion
        pass
    return TacticResult(error=TacticError("type_mismatch", "cannot apply", "apply"),
                       elapsed_us=int((time.perf_counter_ns()-t0)/1000), goals_before=gb)
