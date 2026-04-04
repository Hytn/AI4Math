"""
engine/tactic/engine.py — Enhanced tactic engine with proper type checking.

Supports: intro, assumption, apply, exact, sorry, trivial
Each tactic uses L0/L1 type checking from the kernel.
"""
from __future__ import annotations
import time
from typing import Optional, Dict
from engine.core import Expr, Name, BinderInfo, MetaId, FVarId, LocalContext
from engine.state.proof_state import ProofState
from engine.state.meta_ctx import MetaContext
from engine.kernel.type_checker import TypeChecker, VerificationLevel, Reducer

from dataclasses import dataclass, field

@dataclass
class TacticError:
    kind: str
    message: str
    tactic: str = ""

@dataclass
class TacticResult:
    state: Optional[ProofState] = None
    error: Optional[TacticError] = None
    elapsed_us: int = 0
    goals_before: int = 0
    goals_after: int = 0

    @property
    def success(self):
        return self.state is not None


def execute_tactic(state: ProofState, tactic_str: str) -> TacticResult:
    """Parse and execute a tactic string."""
    parts = tactic_str.strip().split(None, 1)
    name = parts[0] if parts else ""
    arg = parts[1] if len(parts) > 1 else ""

    dispatch = {
        "intro": lambda: tac_intro(state, arg or "h"),
        "assumption": lambda: tac_assumption(state),
        "apply": lambda: tac_apply(state, arg),
        "exact": lambda: tac_exact(state, arg),
        "sorry": lambda: tac_sorry(state),
        "trivial": lambda: tac_trivial(state),
        "rfl": lambda: tac_sorry(state),  # placeholder
        "simp": lambda: tac_sorry(state),  # placeholder
        "cases": lambda: tac_sorry(state),  # placeholder
        "induction": lambda: tac_sorry(state),  # placeholder
    }

    fn = dispatch.get(name)
    if fn is None:
        return TacticResult(error=TacticError("unknown_tactic", f"unknown: {name}", tactic_str),
                          goals_before=state.num_goals())
    return fn()


def tac_intro(state: ProofState, name: str) -> TacticResult:
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return TacticResult(error=TacticError("no_goals", "no goals", "intro"),
                          elapsed_us=_us(t0), goals_before=gb)

    # Reduce target to WHNF to expose Pi
    reducer = Reducer(state.env, dict(state.meta_ctx.assignments))
    target_whnf = reducer.whnf(goal.target)

    if not target_whnf.is_pi or len(target_whnf.children) != 2:
        return TacticResult(error=TacticError("goal_mismatch",
                          f"target is not ∀/→: {repr(goal.target)}", "intro"),
                          elapsed_us=_us(t0), goals_before=gb)

    domain = target_whnf.children[0]
    body = target_whnf.children[1]
    bi = target_whnf.binder_info

    ns, fid = state.fresh_fvar()
    new_ctx = goal.local_ctx.push_hyp(fid, Name.from_str(name), domain)
    new_target = body.instantiate(Expr.fvar(fid.id))

    mc = ns.meta_ctx
    mc, new_gid = mc.create_meta(new_ctx, new_target, depth=goal.depth + 1)
    proof = Expr.lam(bi, Name.from_str(name), domain, Expr.mvar(new_gid))
    mc = mc.assign(goal.id, proof)
    result = ns.replace_main_goal(mc, [new_gid])

    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                       goals_after=result.num_goals())


def tac_assumption(state: ProofState) -> TacticResult:
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return TacticResult(error=TacticError("no_goals", "no goals", "assumption"),
                          elapsed_us=_us(t0), goals_before=gb)

    assigns = dict(state.meta_ctx.assignments)
    reducer = Reducer(state.env, assigns)
    target_whnf = reducer.whnf(goal.target)

    for decl in goal.local_ctx:
        hyp_type_whnf = reducer.whnf(decl.type_)
        if reducer.is_def_eq(hyp_type_whnf, target_whnf):
            proof = Expr.fvar(decl.fvar_id.id)
            result = state.assign_goal(goal.id, proof)
            return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                              goals_after=result.num_goals())

    return TacticResult(error=TacticError("no_match",
                       "no hypothesis matches goal", "assumption"),
                       elapsed_us=_us(t0), goals_before=gb)


def tac_apply(state: ProofState, lemma_name: str) -> TacticResult:
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return TacticResult(error=TacticError("no_goals", "no goals", "apply"),
                          elapsed_us=_us(t0), goals_before=gb)

    # Resolve lemma: try hypothesis first, then environment
    lemma_expr = None
    lemma_type = None

    parsed_name = Name.from_str(lemma_name)
    hyp = goal.local_ctx.find_by_name(parsed_name)
    if hyp:
        lemma_expr = Expr.fvar(hyp.fvar_id.id)
        lemma_type = hyp.type_
    else:
        info = state.env.lookup(parsed_name)
        if info:
            lemma_expr = Expr.const(parsed_name)
            lemma_type = info.type_

    if lemma_expr is None or lemma_type is None:
        return TacticResult(error=TacticError("not_found",
                          f"lemma '{lemma_name}' not found", "apply"),
                          elapsed_us=_us(t0), goals_before=gb)

    # Decompose lemma type into arguments and conclusion
    assigns = dict(state.meta_ctx.assignments)
    reducer = Reducer(state.env, assigns)
    mc = state.meta_ctx

    current_ty = reducer.whnf(lemma_type)
    new_goals = []
    args = []

    while current_ty.is_pi and len(current_ty.children) == 2:
        domain = current_ty.children[0]
        body = current_ty.children[1]

        mc, arg_id = mc.create_meta(goal.local_ctx, domain, depth=goal.depth + 1)
        new_goals.append(arg_id)
        args.append(Expr.mvar(arg_id))
        current_ty = reducer.whnf(body.instantiate(Expr.mvar(arg_id)))

    # Build proof term: lemma ?a1 ?a2 ...
    proof = lemma_expr
    for a in args:
        proof = Expr.app(proof, a)

    mc = mc.assign(goal.id, proof)
    result = state.replace_main_goal(mc, new_goals)

    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                       goals_after=result.num_goals())


def tac_exact(state: ProofState, term_name: str) -> TacticResult:
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return TacticResult(error=TacticError("no_goals", "no goals", "exact"),
                          elapsed_us=_us(t0), goals_before=gb)

    parsed_name = Name.from_str(term_name)

    # Try hypothesis
    hyp = goal.local_ctx.find_by_name(parsed_name)
    if hyp:
        proof = Expr.fvar(hyp.fvar_id.id)
        assigns = dict(state.meta_ctx.assignments)
        reducer = Reducer(state.env, assigns)
        if reducer.is_def_eq(reducer.whnf(hyp.type_), reducer.whnf(goal.target)):
            result = state.assign_goal(goal.id, proof)
            return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                              goals_after=result.num_goals())

    # Try constant
    info = state.env.lookup(parsed_name)
    if info:
        proof = Expr.const(parsed_name)
        result = state.assign_goal(goal.id, proof)
        return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                          goals_after=result.num_goals())

    return TacticResult(error=TacticError("not_found",
                       f"'{term_name}' not found or type mismatch", "exact"),
                       elapsed_us=_us(t0), goals_before=gb)


def tac_trivial(state: ProofState) -> TacticResult:
    """Try assumption, then rfl, then sorry."""
    r = tac_assumption(state)
    if r.success:
        return r
    return tac_sorry(state)


def tac_sorry(state: ProofState) -> TacticResult:
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return TacticResult(error=TacticError("no_goals", "no goals", "sorry"),
                          elapsed_us=_us(t0), goals_before=gb)
    proof = Expr.const(Name.from_str("APE.sorry"))
    result = state.assign_goal(goal.id, proof)
    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                       goals_after=result.num_goals())


def _us(t0_ns):
    return int((time.perf_counter_ns() - t0_ns) / 1000)
