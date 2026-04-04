"""engine/tactic/engine.py — Complete tactic engine.

Implements 18 tactics with real proof term construction:

  Core:         intro, assumption, apply, exact, have
  Equality:     rfl, symm, trans, rewrite
  Logic:        constructor, cases, contradiction, exfalso
  Automation:   simp, trivial
  Induction:    induction
  Escape hatch: sorry

Each tactic constructs a proper proof term and creates new subgoals.
No tactic ever silently maps to sorry.
"""
from __future__ import annotations
import time
from typing import Optional, Dict
from engine.core import Expr, Name, BinderInfo, MetaId, FVarId, LocalContext
from engine.core.environment import Environment
from engine.state.proof_state import ProofState
from engine.state.meta_ctx import MetaContext
from engine.kernel.type_checker import TypeChecker, VerificationLevel, Reducer

from dataclasses import dataclass, field

BI = BinderInfo.DEFAULT
IMP = BinderInfo.IMPLICIT


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


def _us(t0_ns):
    return int((time.perf_counter_ns() - t0_ns) / 1000)


def _no_goals(tactic: str, t0):
    return TacticResult(error=TacticError("no_goals", "no goals", tactic),
                        elapsed_us=_us(t0))


def _fail(tactic: str, kind: str, msg: str, t0, gb=0):
    return TacticResult(error=TacticError(kind, msg, tactic),
                        elapsed_us=_us(t0), goals_before=gb)


def _mk_reducer(state: ProofState) -> Reducer:
    return Reducer(state.env, dict(state.meta_ctx.assignments))


# ═══════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════

def execute_tactic(state: ProofState, tactic_str: str) -> TacticResult:
    """Parse and execute a tactic string."""
    parts = tactic_str.strip().split(None, 1)
    name = parts[0] if parts else ""
    arg = parts[1] if len(parts) > 1 else ""

    dispatch = {
        "intro":         lambda: tac_intro(state, arg or "h"),
        "assumption":    lambda: tac_assumption(state),
        "apply":         lambda: tac_apply(state, arg),
        "exact":         lambda: tac_exact(state, arg),
        "sorry":         lambda: tac_sorry(state),
        "trivial":       lambda: tac_trivial(state),
        "rfl":           lambda: tac_rfl(state),
        "simp":          lambda: tac_simp(state, arg),
        "cases":         lambda: tac_cases(state, arg),
        "induction":     lambda: tac_induction(state, arg),
        "constructor":   lambda: tac_constructor(state),
        "contradiction": lambda: tac_contradiction(state),
        "exfalso":       lambda: tac_exfalso(state),
        "symm":          lambda: tac_symm(state),
        "trans":         lambda: tac_trans(state, arg),
        "rewrite":       lambda: tac_rewrite(state, arg),
        "rw":            lambda: tac_rewrite(state, arg),
        "have":          lambda: tac_have(state, arg),
    }

    fn = dispatch.get(name)
    if fn is None:
        return TacticResult(
            error=TacticError("unknown_tactic", f"unknown: {name}", tactic_str),
            goals_before=state.num_goals())
    return fn()


# ═══════════════════════════════════════════════════════════════
# Core tactics
# ═══════════════════════════════════════════════════════════════

def tac_intro(state: ProofState, name: str) -> TacticResult:
    """Introduce a universally quantified variable or hypothesis."""
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("intro", t0)

    reducer = _mk_reducer(state)
    target_whnf = reducer.whnf(goal.target)

    if not target_whnf.is_pi or len(target_whnf.children) != 2:
        return _fail("intro", "goal_mismatch",
                     f"target is not ∀/→: {repr(goal.target)}", t0, gb)

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
    """Close goal if a hypothesis has exactly the same type."""
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("assumption", t0)

    reducer = _mk_reducer(state)
    target_whnf = reducer.whnf(goal.target)

    for decl in goal.local_ctx:
        hyp_type_whnf = reducer.whnf(decl.type_)
        if reducer.is_def_eq(hyp_type_whnf, target_whnf):
            proof = Expr.fvar(decl.fvar_id.id)
            result = state.assign_goal(goal.id, proof)
            return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                                goals_after=result.num_goals())

    return _fail("assumption", "no_match",
                 "no hypothesis matches goal", t0, gb)


def tac_apply(state: ProofState, lemma_name: str) -> TacticResult:
    """Apply a lemma/hypothesis whose conclusion unifies with the goal."""
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("apply", t0)

    lemma_expr, lemma_type = _resolve_name(state, goal, lemma_name)
    if lemma_expr is None or lemma_type is None:
        return _fail("apply", "not_found",
                     f"'{lemma_name}' not found", t0, gb)

    reducer = _mk_reducer(state)
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

    # Check conclusion matches goal
    if not reducer.is_def_eq(current_ty, reducer.whnf(goal.target)):
        return _fail("apply", "type_mismatch",
                     f"conclusion {repr(current_ty)} doesn't match goal", t0, gb)

    proof = lemma_expr
    for a in args:
        proof = Expr.app(proof, a)

    mc = mc.assign(goal.id, proof)
    result = state.replace_main_goal(mc, new_goals)

    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                        goals_after=result.num_goals())


def tac_exact(state: ProofState, term_name: str) -> TacticResult:
    """Close goal by providing an exact proof term."""
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("exact", t0)

    proof_expr, proof_type = _resolve_name(state, goal, term_name)
    if proof_expr is None:
        return _fail("exact", "not_found",
                     f"'{term_name}' not found", t0, gb)

    # Type check
    if proof_type is not None:
        reducer = _mk_reducer(state)
        if not reducer.is_def_eq(reducer.whnf(proof_type),
                                  reducer.whnf(goal.target)):
            return _fail("exact", "type_mismatch",
                         f"'{term_name}' has type {repr(proof_type)}, "
                         f"goal is {repr(goal.target)}", t0, gb)

    result = state.assign_goal(goal.id, proof_expr)
    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                        goals_after=result.num_goals())


# ═══════════════════════════════════════════════════════════════
# Equality tactics
# ═══════════════════════════════════════════════════════════════

def tac_rfl(state: ProofState) -> TacticResult:
    """Prove a = a by reflexivity.

    Checks if the goal is an application of Eq where both sides
    are definitionally equal.
    """
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("rfl", t0)

    reducer = _mk_reducer(state)
    target = reducer.whnf(goal.target)

    lhs, rhs = _decompose_eq(target)
    if lhs is None:
        return _fail("rfl", "goal_mismatch",
                     "goal is not an equality (a = b)", t0, gb)

    if not reducer.is_def_eq(lhs, rhs):
        return _fail("rfl", "not_refl",
                     f"sides not definitionally equal: "
                     f"{repr(lhs)} ≠ {repr(rhs)}", t0, gb)

    # Proof term: @Eq.refl _ lhs
    proof = Expr.const(Name.from_str("Eq.refl"))
    proof = Expr.app(proof, lhs)  # simplified: skip implicit type arg
    result = state.assign_goal(goal.id, proof)
    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                        goals_after=result.num_goals())


def tac_symm(state: ProofState) -> TacticResult:
    """Transform goal a = b into b = a."""
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("symm", t0)

    reducer = _mk_reducer(state)
    target = reducer.whnf(goal.target)
    lhs, rhs = _decompose_eq(target)
    if lhs is None:
        return _fail("symm", "goal_mismatch",
                     "goal is not an equality", t0, gb)

    # New goal: rhs = lhs
    new_target = _mk_eq(rhs, lhs)
    ns = state
    mc = ns.meta_ctx
    mc, new_gid = mc.create_meta(goal.local_ctx, new_target, depth=goal.depth)

    # Proof: Eq.symm ?new_goal
    proof = Expr.app(Expr.const(Name.from_str("Eq.symm")), Expr.mvar(new_gid))
    mc = mc.assign(goal.id, proof)
    result = ns.replace_main_goal(mc, [new_gid])
    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                        goals_after=result.num_goals())


def tac_trans(state: ProofState, mid_name: str) -> TacticResult:
    """Transitivity: split a = c into a = ?mid and ?mid = c.

    Usage:
      trans b       — sets the middle term to ``b`` (resolved from context/env)
      trans         — creates a fresh metavariable as the middle term

    Given goal  a = c, produces two subgoals:
      1.  a = mid
      2.  mid = c

    Proof term:  Eq.trans ?sub1 ?sub2
    """
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("trans", t0)

    reducer = _mk_reducer(state)
    target = reducer.whnf(goal.target)
    lhs, rhs = _decompose_eq(target)
    if lhs is None:
        return _fail("trans", "goal_mismatch",
                     "goal is not an equality (a = b)", t0, gb)

    mc = state.meta_ctx

    # Resolve or create the middle term
    if mid_name.strip():
        mid_expr, _ = _resolve_name(state, goal, mid_name.strip())
        if mid_expr is None:
            # Try parsing as a simple type/expression name
            mid_expr = _resolve_type_name(state, goal, mid_name.strip())
        if mid_expr is None:
            return _fail("trans", "not_found",
                         f"'{mid_name}' not found", t0, gb)
    else:
        # Create a fresh metavar as the middle term (unknown, to be unified)
        # Use the same type context; type = same as lhs/rhs
        mc, mid_meta = mc.create_meta(goal.local_ctx, Expr.type_(),
                                        depth=goal.depth + 1)
        mid_expr = Expr.mvar(mid_meta)

    # Subgoal 1:  lhs = mid
    target1 = _mk_eq(lhs, mid_expr)
    mc, gid1 = mc.create_meta(goal.local_ctx, target1, depth=goal.depth + 1)

    # Subgoal 2:  mid = rhs
    target2 = _mk_eq(mid_expr, rhs)
    mc, gid2 = mc.create_meta(goal.local_ctx, target2, depth=goal.depth + 1)

    # Proof:  Eq.trans ?sub1 ?sub2
    proof = Expr.app(
        Expr.app(Expr.const(Name.from_str("Eq.trans")),
                 Expr.mvar(gid1)),
        Expr.mvar(gid2))
    mc = mc.assign(goal.id, proof)

    from pyrsistent import pvector
    result = ProofState(state.env, mc,
                        pvector([gid1, gid2]) + state.focus[1:],
                        state.id, state._next_fvar)
    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                        goals_after=2)


def tac_rewrite(state: ProofState, hyp_name: str) -> TacticResult:
    """Rewrite the goal using an equality hypothesis.

    Given h : a = b, replaces occurrences of a with b in the goal.

    Proof term construction:
      motive = fun x => target[lhs := x]   (abstract lhs out of goal)
      proof  = @Eq.rec _ lhs motive ?new_goal rhs h
      where ?new_goal : motive[rhs] = target[lhs := rhs] = new_target
    """
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("rw", t0)

    hyp_expr, hyp_type = _resolve_name(state, goal, hyp_name)
    if hyp_expr is None or hyp_type is None:
        return _fail("rw", "not_found", f"'{hyp_name}' not found", t0, gb)

    reducer = _mk_reducer(state)
    hyp_type_whnf = reducer.whnf(hyp_type)
    lhs, rhs = _decompose_eq(hyp_type_whnf)
    if lhs is None:
        return _fail("rw", "not_eq",
                     f"'{hyp_name}' is not an equality", t0, gb)

    # Replace lhs with rhs in the goal target
    new_target = _subst_expr(goal.target, lhs, rhs, reducer)
    if new_target == goal.target:
        return _fail("rw", "no_match",
                     f"lhs {repr(lhs)} not found in goal", t0, gb)

    mc = state.meta_ctx
    mc, new_gid = mc.create_meta(goal.local_ctx, new_target, depth=goal.depth)

    # Construct proper proof term via Eq.rec:
    #   motive = fun (x : _) => target[lhs := x]
    #   proof  = Eq.rec motive ?new_goal h
    motive_body = _abstract_subexpr(goal.target, lhs, reducer)
    motive = Expr.lam(BI, Name.from_str("_rw"), Expr.prop(), motive_body)

    eq_rec = Expr.const(Name.from_str("Eq.rec"))
    proof = Expr.app(
        Expr.app(Expr.app(eq_rec, motive), Expr.mvar(new_gid)),
        hyp_expr)

    mc = mc.assign(goal.id, proof)
    result = state.replace_main_goal(mc, [new_gid])
    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                        goals_after=result.num_goals())


# ═══════════════════════════════════════════════════════════════
# Logic tactics
# ═══════════════════════════════════════════════════════════════

def tac_constructor(state: ProofState) -> TacticResult:
    """Split a goal into constructor subgoals.

    For And: creates two subgoals (left and right).
    For Exists: creates witness + property subgoals.
    For Iff: creates forward and backward subgoals.
    """
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("constructor", t0)

    reducer = _mk_reducer(state)
    target = reducer.whnf(goal.target)
    head = target.get_app_fn_name()

    mc = state.meta_ctx

    if head and str(head) in ("And", "Prod"):
        # Goal: A ∧ B → two subgoals: A and B
        args = target.get_app_args()
        if len(args) >= 2:
            a_type, b_type = args[0], args[1]
            mc, gid_a = mc.create_meta(goal.local_ctx, a_type, depth=goal.depth + 1)
            mc, gid_b = mc.create_meta(goal.local_ctx, b_type, depth=goal.depth + 1)
            proof = Expr.app(Expr.app(Expr.const(Name.from_str("And.intro")),
                                       Expr.mvar(gid_a)), Expr.mvar(gid_b))
            mc = mc.assign(goal.id, proof)
            result = state.replace_main_goal(mc, [gid_a, gid_b])
            return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                                goals_after=result.num_goals())

    if head and str(head) in ("Iff",):
        args = target.get_app_args()
        if len(args) >= 2:
            a_type, b_type = args[0], args[1]
            fwd = Expr.arrow(a_type, b_type)
            bwd = Expr.arrow(b_type, a_type)
            mc, gid_fwd = mc.create_meta(goal.local_ctx, fwd, depth=goal.depth + 1)
            mc, gid_bwd = mc.create_meta(goal.local_ctx, bwd, depth=goal.depth + 1)
            proof = Expr.app(Expr.app(Expr.const(Name.from_str("Iff.intro")),
                                       Expr.mvar(gid_fwd)), Expr.mvar(gid_bwd))
            mc = mc.assign(goal.id, proof)
            result = state.replace_main_goal(mc, [gid_fwd, gid_bwd])
            return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                                goals_after=result.num_goals())

    # Try apply the first constructor in the environment
    if head:
        for suffix in [".intro", ".mk"]:
            ctor_name = Name.from_str(str(head) + suffix)
            info = state.env.lookup(ctor_name)
            if info:
                return tac_apply(state, str(ctor_name))

    return _fail("constructor", "no_constructor",
                 "goal doesn't match a known constructor pattern", t0, gb)


def tac_cases(state: ProofState, hyp_name: str) -> TacticResult:
    """Case analysis on a hypothesis.

    For h : A ∨ B → two subgoals with h : A and h : B.
    For h : A ∧ B → extract h.1 : A and h.2 : B into context.
    For h : False → close the goal.
    For h : Bool / inductive → one subgoal per constructor.
    """
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("cases", t0)

    if not hyp_name:
        return _fail("cases", "missing_arg", "cases requires a hypothesis name", t0, gb)

    parsed_name = Name.from_str(hyp_name)
    hyp = goal.local_ctx.find_by_name(parsed_name)
    if not hyp:
        return _fail("cases", "not_found",
                     f"hypothesis '{hyp_name}' not found", t0, gb)

    reducer = _mk_reducer(state)
    hyp_type = reducer.whnf(hyp.type_)
    head = hyp_type.get_app_fn_name()
    mc = state.meta_ctx

    # Case: h : False → close goal
    if head and str(head) == "False":
        proof = Expr.app(Expr.const(Name.from_str("False.elim")),
                         Expr.fvar(hyp.fvar_id.id))
        result = state.assign_goal(goal.id, proof)
        return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                            goals_after=result.num_goals())

    # Case: h : A ∧ B → extract components
    if head and str(head) in ("And", "Prod"):
        args = hyp_type.get_app_args()
        if len(args) >= 2:
            a_type, b_type = args[0], args[1]
            ns1, fid_l = state.fresh_fvar()
            ns2, fid_r = ns1.fresh_fvar()
            new_ctx = goal.local_ctx.push_hyp(
                fid_l, Name.from_str(hyp_name + "_left"), a_type)
            new_ctx = new_ctx.push_hyp(
                fid_r, Name.from_str(hyp_name + "_right"), b_type)
            mc2 = ns2.meta_ctx
            mc2, new_gid = mc2.create_meta(new_ctx, goal.target, depth=goal.depth)

            # Proof term: And.casesOn h (fun left right => ?new_goal)
            # Binds the two components via let-expressions:
            #   let h_left  := And.left h
            #   let h_right := And.right h
            #   ?new_goal
            h_ref = Expr.fvar(hyp.fvar_id.id)
            and_left  = Expr.app(Expr.const(Name.from_str("And.left")), h_ref)
            and_right = Expr.app(Expr.const(Name.from_str("And.right")), h_ref)
            proof = Expr.let_(
                Name.from_str(hyp_name + "_left"), a_type, and_left,
                Expr.let_(
                    Name.from_str(hyp_name + "_right"), b_type, and_right,
                    Expr.mvar(new_gid)))

            mc2 = mc2.assign(goal.id, proof)
            result = ns2.replace_main_goal(mc2, [new_gid])
            return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                                goals_after=result.num_goals())

    # Case: h : A ∨ B → two subgoals
    if head and str(head) in ("Or",):
        args = hyp_type.get_app_args()
        if len(args) >= 2:
            a_type, b_type = args[0], args[1]
            # Case left: context gets h : A
            ns1, fid_l = state.fresh_fvar()
            ctx_l = goal.local_ctx.push_hyp(
                fid_l, Name.from_str(hyp_name), a_type)
            mc_l = ns1.meta_ctx
            mc_l, gid_l = mc_l.create_meta(ctx_l, goal.target, depth=goal.depth + 1)

            # Case right: context gets h : B
            ns2, fid_r = ns1.fresh_fvar()
            ctx_r = goal.local_ctx.push_hyp(
                fid_r, Name.from_str(hyp_name), b_type)
            mc2 = MetaContext(mc_l._decls, mc_l._assignments, mc_l._deps, mc_l._next_id)
            mc2, gid_r = mc2.create_meta(ctx_r, goal.target, depth=goal.depth + 1)

            # Proof: Or.elim h (fun a => ?left) (fun b => ?right)
            proof = Expr.app(
                Expr.app(Expr.app(Expr.const(Name.from_str("Or.elim")),
                                   Expr.fvar(hyp.fvar_id.id)),
                         Expr.lam(BI, parsed_name, a_type, Expr.mvar(gid_l))),
                Expr.lam(BI, parsed_name, b_type, Expr.mvar(gid_r)))
            mc2 = mc2.assign(goal.id, proof)
            result = ProofState(state.env, mc2,
                                state.focus.__class__([gid_l, gid_r]) + state.focus[1:],
                                state.id, state._next_fvar)
            return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                                goals_after=2)

    # Generic: check environment for inductive info
    if head:
        ind_info = state.env.lookup_inductive(head)
        if ind_info and ind_info.constructors:
            # Create one subgoal per constructor
            new_goals = []
            for ctor in ind_info.constructors:
                mc, gid = mc.create_meta(goal.local_ctx, goal.target,
                                          depth=goal.depth + 1)
                new_goals.append(gid)
            if new_goals:
                # Construct recursor proof term:
                #   @T.rec motive ?case1 ?case2 ... h
                # where motive = fun _ => goal.target (constant motive)
                h_ref = Expr.fvar(hyp.fvar_id.id)
                rec_name = str(head) + ".rec"
                rec_expr = Expr.const(Name.from_str(rec_name))
                # Constant motive: fun (_ : T) => goal.target
                motive = Expr.lam(BI, Name.from_str("_"), hyp_type, goal.target)
                proof = Expr.app(rec_expr, motive)
                for gid in new_goals:
                    proof = Expr.app(proof, Expr.mvar(gid))
                proof = Expr.app(proof, h_ref)

                mc = mc.assign(goal.id, proof)
                result = state.replace_main_goal(mc, new_goals)
                return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                                    goals_after=len(new_goals))

    return _fail("cases", "unsupported",
                 f"can't do cases on {repr(hyp_type)}", t0, gb)


def tac_contradiction(state: ProofState) -> TacticResult:
    """Search for contradictory hypotheses to close the goal.

    Looks for: h : False, or h : P and h' : ¬P.
    """
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("contradiction", t0)

    reducer = _mk_reducer(state)

    for decl in goal.local_ctx:
        hyp_type = reducer.whnf(decl.type_)

        # h : False → close immediately
        if (hyp_type.tag == "const" and hyp_type.name
                and str(hyp_type.name) == "False"):
            proof = Expr.app(Expr.const(Name.from_str("False.elim")),
                             Expr.fvar(decl.fvar_id.id))
            result = state.assign_goal(goal.id, proof)
            return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                                goals_after=result.num_goals())

    # Look for P and ¬P
    for d1 in goal.local_ctx:
        t1 = reducer.whnf(d1.type_)
        for d2 in goal.local_ctx:
            if d1.fvar_id == d2.fvar_id:
                continue
            t2 = reducer.whnf(d2.type_)
            # Check if t2 = ¬t1 = t1 → False
            if (t2.is_pi and len(t2.children) == 2):
                domain = reducer.whnf(t2.children[0])
                codomain = reducer.whnf(t2.children[1])
                if (reducer.is_def_eq(domain, t1) and
                        codomain.tag == "const" and
                        codomain.name and str(codomain.name) == "False"):
                    # Apply d2 to d1 to get False, then False.elim
                    false_proof = Expr.app(Expr.fvar(d2.fvar_id.id),
                                            Expr.fvar(d1.fvar_id.id))
                    proof = Expr.app(Expr.const(Name.from_str("False.elim")),
                                     false_proof)
                    result = state.assign_goal(goal.id, proof)
                    return TacticResult(result, elapsed_us=_us(t0),
                                        goals_before=gb,
                                        goals_after=result.num_goals())

    return _fail("contradiction", "no_contradiction",
                 "no contradictory hypotheses found", t0, gb)


def tac_exfalso(state: ProofState) -> TacticResult:
    """Change goal to False."""
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("exfalso", t0)

    false_type = Expr.const(Name.from_str("False"))
    mc = state.meta_ctx
    mc, new_gid = mc.create_meta(goal.local_ctx, false_type, depth=goal.depth)
    proof = Expr.app(Expr.const(Name.from_str("False.elim")), Expr.mvar(new_gid))
    mc = mc.assign(goal.id, proof)
    result = state.replace_main_goal(mc, [new_gid])
    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                        goals_after=result.num_goals())


# ═══════════════════════════════════════════════════════════════
# Induction
# ═══════════════════════════════════════════════════════════════

def tac_induction(state: ProofState, var_name: str) -> TacticResult:
    """Mathematical induction on a variable.

    For n : Nat, creates:
      - Base case: goal with n replaced by Nat.zero
      - Step case: goal with n replaced by Nat.succ m, plus IH
    """
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("induction", t0)

    if not var_name:
        return _fail("induction", "missing_arg",
                     "induction requires a variable name", t0, gb)

    parsed_name = Name.from_str(var_name)
    var_decl = goal.local_ctx.find_by_name(parsed_name)
    if not var_decl:
        return _fail("induction", "not_found",
                     f"variable '{var_name}' not found", t0, gb)

    reducer = _mk_reducer(state)
    var_type = reducer.whnf(var_decl.type_)
    type_head = var_type.get_app_fn_name()

    mc = state.meta_ctx

    # Nat induction
    if type_head and str(type_head) == "Nat":
        nat_zero = Expr.const(Name.from_str("Nat.zero"))
        nat_succ = Expr.const(Name.from_str("Nat.succ"))

        # Base case: replace n with 0
        base_target = goal.target.replace_fvar(var_decl.fvar_id.id, nat_zero)
        mc, gid_base = mc.create_meta(goal.local_ctx, base_target,
                                       depth=goal.depth + 1)

        # Step case: introduce m : Nat and ih : P(m), prove P(succ m)
        ns, fid_m = state.fresh_fvar()
        ns2, fid_ih = ns.fresh_fvar()
        m_expr = Expr.fvar(fid_m.id)

        step_target = goal.target.replace_fvar(var_decl.fvar_id.id,
                                                 Expr.app(nat_succ, m_expr))
        ih_type = goal.target.replace_fvar(var_decl.fvar_id.id, m_expr)

        step_ctx = goal.local_ctx.push_hyp(fid_m, Name.from_str("n✝"), var_type)
        step_ctx = step_ctx.push_hyp(fid_ih, Name.from_str("ih"), ih_type)

        mc2 = MetaContext(mc._decls, mc._assignments, mc._deps, mc._next_id)
        mc2, gid_step = mc2.create_meta(step_ctx, step_target,
                                          depth=goal.depth + 1)

        # Assign original goal with proper recursor proof term:
        #   @Nat.rec (fun n => P n) ?base (fun n✝ ih => ?step) var
        var_ref = Expr.fvar(var_decl.fvar_id.id)
        nat_rec = Expr.const(Name.from_str("Nat.rec"))

        # Motive: fun (n : Nat) => goal.target[var := n]
        motive_body = goal.target.abstract(var_decl.fvar_id.id)
        motive = Expr.lam(BI, Name.from_str("n"), var_type, motive_body)

        # Step case body: fun (n✝ : Nat) (ih : P n✝) => ?step
        step_body = Expr.lam(BI, Name.from_str("n✝"), var_type,
                       Expr.lam(BI, Name.from_str("ih"),
                                goal.target.abstract(var_decl.fvar_id.id),
                                Expr.mvar(gid_step)))

        # proof = @Nat.rec motive ?base step_body var
        proof = Expr.app(
            Expr.app(
                Expr.app(Expr.app(nat_rec, motive), Expr.mvar(gid_base)),
                step_body),
            var_ref)
        mc2 = mc2.assign(goal.id, proof)

        from pyrsistent import pvector
        result = ProofState(state.env, mc2,
                            pvector([gid_base, gid_step]) + state.focus[1:],
                            state.id, ns2._next_fvar)
        return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                            goals_after=2)

    # Generic induction via environment
    if type_head:
        ind_info = state.env.lookup_inductive(type_head)
        if ind_info and ind_info.constructors:
            new_goals = []
            for ctor in ind_info.constructors:
                mc, gid = mc.create_meta(goal.local_ctx, goal.target,
                                          depth=goal.depth + 1)
                new_goals.append(gid)

            # Construct recursor proof term:
            #   @T.rec (fun x => goal.target[var := x]) ?case1 ?case2 ... var
            var_ref = Expr.fvar(var_decl.fvar_id.id)
            rec_name = str(type_head) + ".rec"
            rec_expr = Expr.const(Name.from_str(rec_name))

            # Motive: fun (x : T) => goal.target[var := x]
            motive_body = goal.target.abstract(var_decl.fvar_id.id)
            motive = Expr.lam(BI, Name.from_str("x"), var_type, motive_body)

            proof = Expr.app(rec_expr, motive)
            for gid in new_goals:
                proof = Expr.app(proof, Expr.mvar(gid))
            proof = Expr.app(proof, var_ref)

            mc = mc.assign(goal.id, proof)
            result = state.replace_main_goal(mc, new_goals)
            return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                                goals_after=len(new_goals))

    return _fail("induction", "unsupported",
                 f"can't induct on type {repr(var_type)}", t0, gb)


# ═══════════════════════════════════════════════════════════════
# Automation
# ═══════════════════════════════════════════════════════════════

def tac_simp(state: ProofState, args: str = "", _depth: int = 0) -> TacticResult:
    """Simplification: try a battery of simple tactics.

    Tries in order: rfl, assumption, contradiction, constructor+assumption,
    then known simplification lemmas.

    The ``_depth`` parameter limits recursive intro-then-simp calls to
    avoid unbounded recursion on deeply nested ∀/→ goals.
    """
    _MAX_SIMP_DEPTH = 10

    t0 = time.perf_counter_ns()
    gb = state.num_goals()

    # Try rfl
    r = tac_rfl(state)
    if r.success:
        return TacticResult(r.state, elapsed_us=_us(t0), goals_before=gb,
                            goals_after=r.goals_after)

    # Try assumption
    r = tac_assumption(state)
    if r.success:
        return TacticResult(r.state, elapsed_us=_us(t0), goals_before=gb,
                            goals_after=r.goals_after)

    # Try contradiction
    r = tac_contradiction(state)
    if r.success:
        return TacticResult(r.state, elapsed_us=_us(t0), goals_before=gb,
                            goals_after=r.goals_after)

    # Try intro then recurse (for ∀/→ goals) — with depth limit
    goal = state.main_goal()
    if goal and _depth < _MAX_SIMP_DEPTH:
        reducer = _mk_reducer(state)
        target = reducer.whnf(goal.target)
        if target.is_pi:
            r = tac_intro(state, f"h_simp_{_depth}")
            if r.success:
                r2 = tac_simp(r.state, args, _depth + 1)
                if r2.success:
                    return TacticResult(r2.state, elapsed_us=_us(t0),
                                        goals_before=gb, goals_after=r2.goals_after)

    # Try exact with known lemmas
    if goal:
        for lemma in ["True.intro", "trivial", "rfl"]:
            info = state.env.lookup(Name.from_str(lemma))
            if info:
                reducer = _mk_reducer(state)
                if reducer.is_def_eq(reducer.whnf(info.type_),
                                      reducer.whnf(goal.target)):
                    result = state.assign_goal(goal.id, Expr.const(Name.from_str(lemma)))
                    return TacticResult(result, elapsed_us=_us(t0),
                                        goals_before=gb, goals_after=result.num_goals())

    return _fail("simp", "failed",
                 "simp could not simplify the goal", t0, gb)


def tac_trivial(state: ProofState) -> TacticResult:
    """Try simple closing tactics: assumption, rfl, contradiction, simp."""
    for tac_fn in [tac_assumption, tac_rfl, tac_contradiction]:
        r = tac_fn(state)
        if r.success:
            return r
    return tac_simp(state)


def tac_have(state: ProofState, spec: str) -> TacticResult:
    """Introduce an intermediate goal: have name : type, or have name := term.

    Mode 1 — ``have h : type_name``
      Creates two subgoals:
        (a) prove the type  (new hypothesis goal)
        (b) prove the original goal with h : type added to context
      ``type_name`` is resolved from the environment or local context.

    Mode 2 — ``have h := term_name``
      Looks up *term_name* in the local context or environment, infers its
      type, and adds ``h : inferred_type`` to the context.  Only one new
      subgoal (the original goal with h available).
    """
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("have", t0)

    # ── Mode 2: have name := term ──
    if ":=" in spec:
        parts = spec.split(":=", 1)
        name = parts[0].strip()
        term_name = parts[1].strip()

        if not name or not term_name:
            return _fail("have", "parse_error",
                         "expected 'have name := term'", t0, gb)

        term_expr, term_type = _resolve_name(state, goal, term_name)
        if term_expr is None or term_type is None:
            return _fail("have", "not_found",
                         f"'{term_name}' not found in context or environment",
                         t0, gb)

        # Add h : term_type to context, with value = term_expr
        ns, fid = state.fresh_fvar()
        new_ctx = goal.local_ctx.push_hyp(
            fid, Name.from_str(name), term_type)
        mc = ns.meta_ctx
        mc, new_gid = mc.create_meta(new_ctx, goal.target, depth=goal.depth)

        # Proof: let h := term_expr in ?new_goal
        proof = Expr.let_(
            Name.from_str(name), term_type, term_expr, Expr.mvar(new_gid))
        mc = mc.assign(goal.id, proof)
        result = ns.replace_main_goal(mc, [new_gid])
        return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                            goals_after=result.num_goals())

    # ── Mode 1: have name : type ──
    if ":" not in spec:
        return _fail("have", "parse_error",
                     "expected 'have name : type' or 'have name := term'",
                     t0, gb)

    parts = spec.split(":", 1)
    name = parts[0].strip()
    type_str = parts[1].strip()

    if not name or not type_str:
        return _fail("have", "parse_error",
                     "expected 'have name : type'", t0, gb)

    # Resolve type_str as a known type expression.
    # Supports: environment names (e.g. "Nat", "True"), hypotheses types,
    # and simple applications like "And P Q" via tokenised lookup.
    have_type = _parse_simple_type(state, goal, type_str)
    if have_type is None:
        return _fail("have", "type_resolve",
                     f"cannot resolve type '{type_str}' — "
                     "use names from environment or context",
                     t0, gb)

    # Subgoal 1: prove have_type
    mc = state.meta_ctx
    mc, gid_prove = mc.create_meta(goal.local_ctx, have_type,
                                     depth=goal.depth + 1)

    # Subgoal 2: prove original goal with h : have_type in context
    ns, fid = state.fresh_fvar()
    new_ctx = goal.local_ctx.push_hyp(fid, Name.from_str(name), have_type)
    mc2 = MetaContext(mc._decls, mc._assignments, mc._deps, mc._next_id)
    mc2, gid_cont = mc2.create_meta(new_ctx, goal.target, depth=goal.depth)

    # Proof: let h : T := ?prove in ?cont
    proof = Expr.let_(
        Name.from_str(name), have_type, Expr.mvar(gid_prove),
        Expr.mvar(gid_cont))
    mc2 = mc2.assign(goal.id, proof)

    from pyrsistent import pvector
    result = ProofState(state.env, mc2,
                        pvector([gid_prove, gid_cont]) + state.focus[1:],
                        state.id, ns._next_fvar)
    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                        goals_after=2)


def tac_sorry(state: ProofState) -> TacticResult:
    """Close goal with sorry (axiom). Marks proof as incomplete."""
    t0 = time.perf_counter_ns()
    gb = state.num_goals()
    goal = state.main_goal()
    if not goal:
        return _no_goals("sorry", t0)
    proof = Expr.const(Name.from_str("APE.sorry"))
    result = state.assign_goal(goal.id, proof)
    return TacticResult(result, elapsed_us=_us(t0), goals_before=gb,
                        goals_after=result.num_goals())


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _resolve_name(state: ProofState, goal, name_str: str):
    """Resolve a name to (expr, type) from hypotheses or environment."""
    parsed = Name.from_str(name_str)

    # Try hypothesis
    hyp = goal.local_ctx.find_by_name(parsed)
    if hyp:
        return Expr.fvar(hyp.fvar_id.id), hyp.type_

    # Try environment constant
    info = state.env.lookup(parsed)
    if info:
        return Expr.const(parsed), info.type_

    return None, None


def _decompose_eq(target: Expr):
    """Decompose Eq a b into (a, b). Returns (None, None) if not Eq."""
    # Pattern: app(app(app(Eq, type), lhs), rhs)  or  app(app(Eq, lhs), rhs)
    head = target.get_app_fn_name()
    if head and str(head) == "Eq":
        args = target.get_app_args()
        if len(args) == 3:  # Eq type lhs rhs
            return args[1], args[2]
        if len(args) == 2:  # Eq lhs rhs (implicit type)
            return args[0], args[1]
    # Also check for syntactic pattern: fvar = fvar etc.
    # In APE's representation, equality might just be structural
    return None, None


def _mk_eq(lhs: Expr, rhs: Expr) -> Expr:
    """Construct Eq lhs rhs."""
    return Expr.app(Expr.app(Expr.const(Name.from_str("Eq")), lhs), rhs)


def _subst_expr(target: Expr, old: Expr, new: Expr, reducer: Reducer) -> Expr:
    """Replace occurrences of old with new in target (up to def-eq)."""
    if reducer.is_def_eq(target, old):
        return new
    if not target.children:
        return target
    new_children = tuple(_subst_expr(c, old, new, reducer) for c in target.children)
    if new_children == target.children:
        return target
    return Expr(target.tag, target.name, target.level, target.idx,
                target.meta_id, target.binder_info, new_children)


def _abstract_subexpr(target: Expr, subexpr: Expr, reducer: Reducer,
                      depth: int = 0) -> Expr:
    """Abstract a subexpression out of target, replacing it with bvar(depth).

    This constructs the motive for rewrite: given target = P[subexpr],
    returns P[#depth] so that (fun x => P[x]) is a valid motive.

    Existing bound variables are lifted to avoid capture.
    """
    if reducer.is_def_eq(target, subexpr):
        return Expr.bvar(depth)
    if not target.children:
        # Lift bvars that are >= depth to make room for the new binder
        if target.tag == "bvar" and target.idx is not None and target.idx >= depth:
            return Expr.bvar(target.idx + 1)
        return target
    new_children = []
    for i, c in enumerate(target.children):
        child_depth = depth
        if target.tag in ("lam", "pi") and i == 1:
            child_depth = depth + 1
        elif target.tag == "let" and i == 2:
            child_depth = depth + 1
        new_children.append(_abstract_subexpr(c, subexpr, reducer, child_depth))
    new_children = tuple(new_children)
    if new_children == target.children:
        return target
    return Expr(target.tag, target.name, target.level, target.idx,
                target.meta_id, target.binder_info, new_children)


def _parse_simple_type(state: ProofState, goal, type_str: str) -> Optional[Expr]:
    """Parse a simple type string into an Expr.

    Supports:
      - Single names: "Nat", "True", "P"  → resolved from env/context
      - Arrow types: "A → B", "A -> B"    → Expr.arrow(A, B)
      - Applications: "And P Q", "Or A B" → Expr.app(Expr.app(f, a), b)

    Returns None if the type cannot be resolved.
    """
    s = type_str.strip()

    # Arrow: split on → / ->
    for arrow_tok in (" → ", " -> "):
        if arrow_tok in s:
            parts = s.split(arrow_tok, 1)
            lhs = _parse_simple_type(state, goal, parts[0])
            rhs = _parse_simple_type(state, goal, parts[1])
            if lhs is not None and rhs is not None:
                return Expr.arrow(lhs, rhs)
            return None

    # Multi-token application: "And P Q" → app(app(And, P), Q)
    tokens = s.split()
    if len(tokens) > 1:
        head = _resolve_type_name(state, goal, tokens[0])
        if head is None:
            return None
        result = head
        for tok in tokens[1:]:
            arg = _resolve_type_name(state, goal, tok)
            if arg is None:
                return None
            result = Expr.app(result, arg)
        return result

    # Single token
    return _resolve_type_name(state, goal, s)


def _resolve_type_name(state: ProofState, goal, name_str: str) -> Optional[Expr]:
    """Resolve a single name to a type expression."""
    parsed = Name.from_str(name_str)

    # Try hypothesis type (if h : T, using "h" gives T — the *type* of h)
    hyp = goal.local_ctx.find_by_name(parsed)
    if hyp:
        return hyp.type_

    # Try environment constant (returns the constant itself, e.g. Nat, True)
    info = state.env.lookup(parsed)
    if info:
        return Expr.const(parsed)

    return None
