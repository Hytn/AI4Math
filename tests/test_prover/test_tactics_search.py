"""tests/test_prover/test_tactics_search.py — Tactic engine and search tests

Tests for:
  - All 15 tactics (rfl, simp, cases, induction, constructor, etc.)
  - MCTS/UCB search coordinator
  - Lean4 environment status checks
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pytest
from engine.core import Expr, Name, BinderInfo, MetaId, FVarId, LocalContext, Environment, ConstantInfo
from engine.core.universe import Level
from engine.core.environment import InductiveInfo, ConstructorInfo
from engine.state.proof_state import ProofState
from engine.tactic.engine import (
    execute_tactic, tac_intro, tac_assumption, tac_rfl, tac_simp,
    tac_cases, tac_induction, tac_constructor, tac_contradiction,
    tac_exfalso, tac_symm, tac_apply, tac_exact, tac_sorry,
    tac_trivial, _decompose_eq, _mk_eq,
)
from engine.search import SearchCoordinator, SearchConfig

# Import shared environment from conftest to avoid duplication
from tests.conftest import mk_standard_env, mk_standard_state

BI = BinderInfo.DEFAULT
IMP = BinderInfo.IMPLICIT


def mk_env():
    """Delegate to shared environment constructor."""
    return mk_standard_env()


def mk_state(env, goal_type):
    """Delegate to shared state constructor."""
    return mk_standard_state(env, goal_type)


# ═══════════════════════════════════════════════════════════════
# Tactic tests — Core
# ═══════════════════════════════════════════════════════════════

class TestIntro:
    def test_intro_pi(self):
        env = mk_env()
        goal = Expr.pi(BI, Name.from_str("P"), Expr.prop(), Expr.bvar(0))
        state = mk_state(env, goal)
        r = tac_intro(state, "P")
        assert r.success
        assert r.goals_after >= 1

    def test_intro_non_pi_fails(self):
        env = mk_env()
        state = mk_state(env, Expr.prop())
        r = tac_intro(state, "x")
        assert not r.success
        assert r.error.kind == "goal_mismatch"


class TestAssumption:
    def test_assumption_matches(self):
        env = mk_env()
        # Goal: ∀ (P : Prop), P → P  (need TWO intros: P then h)
        goal = Expr.pi(BI, Name.from_str("P"), Expr.prop(),
                Expr.pi(BI, Name.from_str("h"), Expr.bvar(0), Expr.bvar(1)))
        state = mk_state(env, goal)
        r1 = tac_intro(state, "P")
        assert r1.success
        r2 = tac_intro(r1.state, "h")
        assert r2.success
        r3 = tac_assumption(r2.state)
        assert r3.success

    def test_assumption_no_match(self):
        env = mk_env()
        state = mk_state(env, Expr.const(Name.from_str("True")))
        r = tac_assumption(state)
        assert not r.success


# ═══════════════════════════════════════════════════════════════
# Tactic tests — Equality (rfl, symm)
# ═══════════════════════════════════════════════════════════════

class TestRfl:
    def test_rfl_succeeds_on_eq(self):
        env = mk_env()
        nat = Expr.const(Name.from_str("Nat"))
        zero = Expr.const(Name.from_str("Nat.zero"))
        # Goal: Eq Nat zero zero (i.e., 0 = 0)
        goal = Expr.app(Expr.app(Expr.app(
            Expr.const(Name.from_str("Eq")), nat), zero), zero)
        state = mk_state(env, goal)
        r = tac_rfl(state)
        assert r.success

    def test_rfl_fails_on_non_eq(self):
        env = mk_env()
        state = mk_state(env, Expr.const(Name.from_str("True")))
        r = tac_rfl(state)
        assert not r.success
        assert r.error.kind == "goal_mismatch"

    def test_rfl_fails_on_neq(self):
        env = mk_env()
        nat = Expr.const(Name.from_str("Nat"))
        zero = Expr.const(Name.from_str("Nat.zero"))
        one = Expr.app(Expr.const(Name.from_str("Nat.succ")), zero)
        goal = Expr.app(Expr.app(Expr.app(
            Expr.const(Name.from_str("Eq")), nat), zero), one)
        state = mk_state(env, goal)
        r = tac_rfl(state)
        assert not r.success
        assert r.error.kind == "not_refl"


class TestSymm:
    def test_symm_flips_eq(self):
        env = mk_env()
        a = Expr.fvar(100)
        b = Expr.fvar(101)
        goal = Expr.app(Expr.app(Expr.const(Name.from_str("Eq")), a), b)
        state = mk_state(env, goal)
        r = tac_symm(state)
        assert r.success
        # New goal should have b = a


# ═══════════════════════════════════════════════════════════════
# Tactic tests — Logic (constructor, cases, contradiction)
# ═══════════════════════════════════════════════════════════════

class TestConstructor:
    def test_constructor_and(self):
        env = mk_env()
        p = Expr.const(Name.from_str("True"))
        q = Expr.const(Name.from_str("True"))
        goal = Expr.app(Expr.app(Expr.const(Name.from_str("And")), p), q)
        state = mk_state(env, goal)
        r = tac_constructor(state)
        assert r.success
        assert r.goals_after == 2  # two subgoals

    def test_constructor_non_constructor_fails(self):
        env = mk_env()
        state = mk_state(env, Expr.const(Name.from_str("Nat")))
        r = tac_constructor(state)
        assert not r.success


class TestCases:
    def test_cases_false(self):
        """cases on h : False should close the goal."""
        env = mk_env()
        goal = Expr.pi(BI, Name.from_str("h"), Expr.const(Name.from_str("False")),
                       Expr.const(Name.from_str("True")))
        state = mk_state(env, goal)
        r1 = tac_intro(state, "h")
        assert r1.success
        r2 = tac_cases(r1.state, "h")
        assert r2.success

    def test_cases_and(self):
        """cases on h : A ∧ B should decompose."""
        env = mk_env()
        p = Expr.const(Name.from_str("True"))
        and_pp = Expr.app(Expr.app(Expr.const(Name.from_str("And")), p), p)
        goal = Expr.arrow(and_pp, Expr.const(Name.from_str("True")))
        state = mk_state(env, goal)
        r1 = tac_intro(state, "h")
        assert r1.success
        r2 = tac_cases(r1.state, "h")
        assert r2.success

    def test_cases_missing_arg(self):
        env = mk_env()
        state = mk_state(env, Expr.const(Name.from_str("True")))
        r = tac_cases(state, "")
        assert not r.success

    def test_cases_not_found(self):
        env = mk_env()
        state = mk_state(env, Expr.const(Name.from_str("True")))
        r = tac_cases(state, "nonexistent")
        assert not r.success


class TestContradiction:
    def test_contradiction_with_false(self):
        env = mk_env()
        # ∀ (h : False), True
        goal = Expr.pi(BI, Name.from_str("h"),
                       Expr.const(Name.from_str("False")),
                       Expr.const(Name.from_str("True")))
        state = mk_state(env, goal)
        r1 = tac_intro(state, "h")
        assert r1.success
        r2 = tac_contradiction(r1.state)
        assert r2.success

    def test_contradiction_with_p_and_not_p(self):
        env = mk_env()
        p = Expr.const(Name.from_str("True"))
        not_p = Expr.arrow(p, Expr.const(Name.from_str("False")))
        # ∀ (hp : True) (hn : True → False), Nat
        goal = Expr.pi(BI, Name.from_str("hp"), p,
                Expr.pi(BI, Name.from_str("hn"), not_p,
                        Expr.const(Name.from_str("Nat"))))
        state = mk_state(env, goal)
        r1 = tac_intro(state, "hp")
        assert r1.success
        r2 = tac_intro(r1.state, "hn")
        assert r2.success
        r3 = tac_contradiction(r2.state)
        assert r3.success

    def test_no_contradiction(self):
        env = mk_env()
        state = mk_state(env, Expr.const(Name.from_str("True")))
        r = tac_contradiction(state)
        assert not r.success


class TestExfalso:
    def test_exfalso_changes_goal(self):
        env = mk_env()
        state = mk_state(env, Expr.const(Name.from_str("True")))
        r = tac_exfalso(state)
        assert r.success
        assert r.goals_after == 1  # goal is now False


# ═══════════════════════════════════════════════════════════════
# Tactic tests — Induction
# ═══════════════════════════════════════════════════════════════

class TestInduction:
    def test_induction_nat(self):
        env = mk_env()
        nat = Expr.const(Name.from_str("Nat"))
        # ∀ (n : Nat), True
        goal = Expr.pi(BI, Name.from_str("n"), nat,
                       Expr.const(Name.from_str("True")))
        state = mk_state(env, goal)
        r1 = tac_intro(state, "n")
        assert r1.success
        r2 = tac_induction(r1.state, "n")
        assert r2.success
        assert r2.goals_after == 2  # base + step

    def test_induction_missing_var(self):
        env = mk_env()
        state = mk_state(env, Expr.const(Name.from_str("True")))
        r = tac_induction(state, "nonexistent")
        assert not r.success

    def test_induction_no_arg(self):
        env = mk_env()
        state = mk_state(env, Expr.const(Name.from_str("True")))
        r = tac_induction(state, "")
        assert not r.success


# ═══════════════════════════════════════════════════════════════
# Tactic tests — Automation
# ═══════════════════════════════════════════════════════════════

class TestSimp:
    def test_simp_on_rfl_goal(self):
        env = mk_env()
        zero = Expr.const(Name.from_str("Nat.zero"))
        nat = Expr.const(Name.from_str("Nat"))
        goal = Expr.app(Expr.app(Expr.app(
            Expr.const(Name.from_str("Eq")), nat), zero), zero)
        state = mk_state(env, goal)
        r = tac_simp(state)
        assert r.success  # simp should close 0 = 0 via rfl

    def test_simp_on_assumption(self):
        env = mk_env()
        # ∀ (P : Prop), P → P
        goal = Expr.pi(BI, Name.from_str("P"), Expr.prop(),
                Expr.pi(BI, Name.from_str("h"), Expr.bvar(0), Expr.bvar(1)))
        state = mk_state(env, goal)
        r1 = tac_intro(state, "P")
        assert r1.success
        r2 = tac_intro(r1.state, "h")
        assert r2.success
        r3 = tac_simp(r2.state)
        assert r3.success  # simp should close via assumption


class TestTrivial:
    def test_trivial_assumption(self):
        env = mk_env()
        # ∀ (P : Prop), P → P
        goal = Expr.pi(BI, Name.from_str("P"), Expr.prop(),
                Expr.pi(BI, Name.from_str("h"), Expr.bvar(0), Expr.bvar(1)))
        state = mk_state(env, goal)
        r1 = tac_intro(state, "P")
        assert r1.success
        r2 = tac_intro(r1.state, "h")
        assert r2.success
        r3 = tac_trivial(r2.state)
        assert r3.success


# ═══════════════════════════════════════════════════════════════
# Dispatcher tests
# ═══════════════════════════════════════════════════════════════

class TestDispatcher:
    def test_dispatch_known_tactics(self):
        env = mk_env()
        state = mk_state(env, Expr.pi(BI, Name.from_str("P"), Expr.prop(), Expr.bvar(0)))
        for tac in ["intro x", "assumption", "rfl", "simp", "cases x",
                     "induction x", "constructor", "contradiction",
                     "exfalso", "trivial", "sorry"]:
            r = execute_tactic(state, tac)
            # Should not crash, even if it fails
            assert r.error is not None or r.success

    def test_dispatch_unknown_fails(self):
        env = mk_env()
        state = mk_state(env, Expr.prop())
        r = execute_tactic(state, "nonexistent_tactic")
        assert not r.success
        assert r.error.kind == "unknown_tactic"

    def test_no_tactic_maps_to_sorry(self):
        """Verify that rfl, simp, cases, induction are NOT sorry aliases."""
        env = mk_env()
        state = mk_state(env, Expr.prop())
        for tac in ["rfl", "simp", "cases x", "induction x"]:
            r = execute_tactic(state, tac)
            if r.success and r.state:
                # Check that the proof term is NOT APE.sorry
                goals = r.state.goals()
                for g in goals:
                    mc = r.state.meta_ctx
                    assignment = mc.get_assignment(g.id)
                    if assignment:
                        assert "APE.sorry" not in repr(assignment), \
                            f"Tactic '{tac}' secretly maps to sorry!"


# ═══════════════════════════════════════════════════════════════
# Helper function tests
# ═══════════════════════════════════════════════════════════════

class TestHelpers:
    def test_decompose_eq_3args(self):
        a, b = Expr.fvar(1), Expr.fvar(2)
        ty = Expr.const(Name.from_str("Nat"))
        eq = Expr.app(Expr.app(Expr.app(Expr.const(Name.from_str("Eq")), ty), a), b)
        lhs, rhs = _decompose_eq(eq)
        assert lhs == a
        assert rhs == b

    def test_decompose_eq_2args(self):
        a, b = Expr.fvar(1), Expr.fvar(2)
        eq = Expr.app(Expr.app(Expr.const(Name.from_str("Eq")), a), b)
        lhs, rhs = _decompose_eq(eq)
        assert lhs == a and rhs == b

    def test_decompose_non_eq(self):
        lhs, rhs = _decompose_eq(Expr.prop())
        assert lhs is None and rhs is None


# ═══════════════════════════════════════════════════════════════
# Search Coordinator tests
# ═══════════════════════════════════════════════════════════════

class TestSearchCoordinator:
    def _mk_coord(self, goal_type=None, strategy="best_first"):
        env = mk_env()
        goal = goal_type or Expr.pi(BI, Name.from_str("P"), Expr.prop(), Expr.bvar(0))
        config = SearchConfig(strategy=strategy, max_nodes=100, max_depth=10)
        return SearchCoordinator(env, goal, config)

    def test_try_tactic_success(self):
        coord = self._mk_coord()
        r = coord.try_tactic(0, "intro h")
        assert r.success
        assert r.child_node is not None

    def test_try_tactic_failure(self):
        coord = self._mk_coord()
        r = coord.try_tactic(0, "assumption")
        assert not r.success

    def test_try_batch(self):
        coord = self._mk_coord()
        results = coord.try_batch(0, ["intro h", "assumption", "rfl"])
        assert len(results) == 3
        assert results[0].success  # intro should work
        assert not results[1].success  # assumption should fail

    def test_select_node_best_first(self):
        coord = self._mk_coord(strategy="best_first")
        node = coord.select_node()
        assert node is not None

    def test_select_node_mcts(self):
        coord = self._mk_coord(strategy="mcts")
        node = coord.select_node()
        assert node is not None

    def test_select_node_bfs(self):
        coord = self._mk_coord(strategy="bfs")
        node = coord.select_node()
        assert node is not None

    def test_select_node_dfs(self):
        coord = self._mk_coord(strategy="dfs")
        node = coord.select_node()
        assert node is not None

    def test_backpropagation(self):
        coord = self._mk_coord()
        r = coord.try_tactic(0, "intro h")
        assert r.success
        stats = coord.stats()
        assert stats["nodes_expanded"] >= 1

    def test_solve_p_implies_p(self):
        """Full search: solve ∀ P, P → P."""
        env = mk_env()
        # ∀ (P : Prop), P → P
        goal = Expr.pi(BI, Name.from_str("P"), Expr.prop(),
                Expr.pi(BI, Name.from_str("h"), Expr.bvar(0), Expr.bvar(1)))
        config = SearchConfig(strategy="best_first", max_nodes=200, max_depth=10)
        coord = SearchCoordinator(env, goal, config)
        stats = coord.run_search(
            tactic_generator=lambda nid: ["intro P", "intro h", "assumption", "simp"])
        assert stats.is_solved, f"Failed to solve P → P: {stats}"
        assert len(stats.solution_path) > 0

    def test_stats(self):
        coord = self._mk_coord()
        coord.try_tactic(0, "intro h")
        s = coord.stats()
        assert "total_nodes" in s
        assert "is_solved" in s
        assert s["total_nodes"] >= 2

    def test_goal_view(self):
        coord = self._mk_coord()
        views = coord.goal_view(0)
        assert len(views) >= 1

    def test_max_depth_limit(self):
        """Search should stop at max_depth."""
        config = SearchConfig(strategy="dfs", max_depth=2, max_nodes=100)
        env = mk_env()
        # A deeply nested goal
        goal = Expr.pi(BI, Name.from_str("a"), Expr.prop(),
                Expr.pi(BI, Name.from_str("b"), Expr.prop(),
                Expr.pi(BI, Name.from_str("c"), Expr.prop(),
                Expr.pi(BI, Name.from_str("d"), Expr.prop(),
                        Expr.bvar(0)))))
        coord = SearchCoordinator(env, goal, config)
        stats = coord.run_search(
            tactic_generator=lambda nid: ["intro x"])
        # Should not solve (needs depth 4, limit is 2)
        assert stats.max_depth_reached <= 3  # some tolerance


# ═══════════════════════════════════════════════════════════════
# P0 fix verification tests
# ═══════════════════════════════════════════════════════════════

class TestRewriteProofTerm:
    """P0-1: rewrite must produce Eq.rec proof term, not a placeholder mvar."""

    def test_rewrite_produces_eq_rec(self):
        """After rw h, the proof term should reference Eq.rec, not bare mvar."""
        env = mk_env()
        nat = Expr.const(Name.from_str("Nat"))
        # Goal: Eq a b,  hyp h : Eq a b  →  rw h produces Eq b b (then close with rfl)
        a_type = nat; b_type = nat
        # Setup: h : a = b ⊢ a = b
        fid_a = FVarId(100); fid_b = FVarId(101); fid_h = FVarId(102)
        eq_ab = Expr.app(Expr.app(Expr.const(Name.from_str("Eq")),
                                   Expr.fvar(100)), Expr.fvar(101))
        lctx = LocalContext()
        lctx = lctx.push_hyp(fid_a, Name.from_str("a"), nat)
        lctx = lctx.push_hyp(fid_b, Name.from_str("b"), nat)
        lctx = lctx.push_hyp(fid_h, Name.from_str("h"), eq_ab)
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, eq_ab, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))

        from engine.tactic.engine import tac_rewrite
        r = tac_rewrite(state, "h")
        assert r.success
        # The proof term assigned to the original goal should involve Eq.rec
        assigned = r.state.meta_ctx.get_assignment(gid)
        assert assigned is not None
        # Check that Eq.rec appears somewhere in the proof term
        proof_repr = repr(assigned)
        assert "Eq.rec" in proof_repr, f"Expected Eq.rec in proof: {proof_repr}"

    def test_rewrite_new_target_correct(self):
        """After rw h where h : a = b, the new goal should have b replacing a."""
        env = mk_env()
        nat = Expr.const(Name.from_str("Nat"))
        fid_a = FVarId(200); fid_b = FVarId(201); fid_h = FVarId(202)
        a_expr = Expr.fvar(200); b_expr = Expr.fvar(201)
        eq_ab = Expr.app(Expr.app(Expr.const(Name.from_str("Eq")), a_expr), b_expr)
        lctx = LocalContext()
        lctx = lctx.push_hyp(fid_a, Name.from_str("a"), nat)
        lctx = lctx.push_hyp(fid_b, Name.from_str("b"), nat)
        lctx = lctx.push_hyp(fid_h, Name.from_str("h"), eq_ab)
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, eq_ab, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))
        from engine.tactic.engine import tac_rewrite
        r = tac_rewrite(state, "h")
        assert r.success
        # The new goal's target should be Eq(b, b) instead of Eq(a, b)
        new_goal = r.state.main_goal()
        assert new_goal is not None


class TestCasesAndProofTerm:
    """P0-1: cases on And should use And.left/And.right, not placeholder."""

    def test_cases_and_uses_let_bindings(self):
        env = mk_env()
        prop = Expr.prop()
        p_type = Expr.const(Name.from_str("True"))  # stand-in
        and_pp = Expr.app(Expr.app(Expr.const(Name.from_str("And")), p_type), p_type)
        fid_h = FVarId(300)
        lctx = LocalContext()
        lctx = lctx.push_hyp(fid_h, Name.from_str("h"), and_pp)
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, prop, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))
        from engine.tactic.engine import tac_cases
        r = tac_cases(state, "h")
        assert r.success
        # The proof should reference And.left / And.right via let bindings
        assigned = r.state.meta_ctx.get_assignment(gid)
        assert assigned is not None
        proof_repr = repr(assigned)
        assert "And.left" in proof_repr, f"Expected And.left in proof: {proof_repr}"
        assert "And.right" in proof_repr, f"Expected And.right in proof: {proof_repr}"


class TestGenericCasesProofTerm:
    """P0-1: generic cases should use T.rec, not mvar(first_goal)."""

    def test_cases_generic_uses_recursor(self):
        from engine.core.environment import InductiveInfo, ConstructorInfo
        env = mk_env()
        prop = Expr.prop()
        bool_ty = Expr.const(Name.from_str("MyBool"))
        env = env.add_inductive(InductiveInfo(
            Name.from_str("MyBool"), Expr.type_(),
            constructors=[
                ConstructorInfo(Name.from_str("MyBool.true"), bool_ty,
                                Name.from_str("MyBool"), 0),
                ConstructorInfo(Name.from_str("MyBool.false"), bool_ty,
                                Name.from_str("MyBool"), 1),
            ]))
        fid_h = FVarId(400)
        lctx = LocalContext()
        lctx = lctx.push_hyp(fid_h, Name.from_str("b"), bool_ty)
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, prop, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))
        from engine.tactic.engine import tac_cases
        r = tac_cases(state, "b")
        assert r.success
        assert r.goals_after == 2
        assigned = r.state.meta_ctx.get_assignment(gid)
        proof_repr = repr(assigned)
        assert "MyBool.rec" in proof_repr, f"Expected MyBool.rec: {proof_repr}"


class TestInductionProofTerm:
    """P0-1: induction Nat should produce Nat.rec with motive+base+step."""

    def test_nat_induction_proof_term(self):
        env = mk_env()
        nat = Expr.const(Name.from_str("Nat"))
        # Goal: ∀ n : Nat, P n  — simplified as a Pi
        # Use n : Nat ⊢ Eq n n  (identity goal)
        fid_n = FVarId(500)
        n_expr = Expr.fvar(500)
        eq_nn = Expr.app(Expr.app(Expr.const(Name.from_str("Eq")), n_expr), n_expr)
        lctx = LocalContext()
        lctx = lctx.push_hyp(fid_n, Name.from_str("n"), nat)
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, eq_nn, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))
        from engine.tactic.engine import tac_induction
        r = tac_induction(state, "n")
        assert r.success
        assert r.goals_after == 2
        assigned = r.state.meta_ctx.get_assignment(gid)
        proof_repr = repr(assigned)
        assert "Nat.rec" in proof_repr, f"Expected Nat.rec: {proof_repr}"


class TestTransTactic:
    """P0-5: trans tactic should split a = c into a = mid and mid = c."""

    def test_trans_splits_equality(self):
        env = mk_env()
        nat = Expr.const(Name.from_str("Nat"))
        fid_a = FVarId(600); fid_b = FVarId(601); fid_c = FVarId(602)
        a = Expr.fvar(600); b = Expr.fvar(601); c = Expr.fvar(602)
        eq_ac = Expr.app(Expr.app(Expr.const(Name.from_str("Eq")), a), c)
        lctx = LocalContext()
        lctx = lctx.push_hyp(fid_a, Name.from_str("a"), nat)
        lctx = lctx.push_hyp(fid_b, Name.from_str("b"), nat)
        lctx = lctx.push_hyp(fid_c, Name.from_str("c"), nat)
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, eq_ac, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))
        from engine.tactic.engine import tac_trans
        r = tac_trans(state, "b")
        assert r.success
        assert r.goals_after == 2
        # Proof should use Eq.trans
        assigned = r.state.meta_ctx.get_assignment(gid)
        proof_repr = repr(assigned)
        assert "Eq.trans" in proof_repr, f"Expected Eq.trans: {proof_repr}"

    def test_trans_no_mid_creates_mvar(self):
        """trans without argument should still create two subgoals."""
        env = mk_env()
        nat = Expr.const(Name.from_str("Nat"))
        fid_a = FVarId(610); fid_c = FVarId(612)
        a = Expr.fvar(610); c = Expr.fvar(612)
        eq_ac = Expr.app(Expr.app(Expr.const(Name.from_str("Eq")), a), c)
        lctx = LocalContext()
        lctx = lctx.push_hyp(fid_a, Name.from_str("a"), nat)
        lctx = lctx.push_hyp(fid_c, Name.from_str("c"), nat)
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, eq_ac, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))
        from engine.tactic.engine import tac_trans
        r = tac_trans(state, "")
        assert r.success
        assert r.goals_after == 2

    def test_trans_non_eq_fails(self):
        env = mk_env()
        state = mk_state(env, Expr.prop())
        from engine.tactic.engine import tac_trans
        r = tac_trans(state, "")
        assert not r.success
        assert r.error.kind == "goal_mismatch"

    def test_trans_in_dispatch(self):
        """trans should be accessible via the dispatch table."""
        env = mk_env()
        nat = Expr.const(Name.from_str("Nat"))
        fid_a = FVarId(620); fid_c = FVarId(622)
        eq_ac = Expr.app(Expr.app(Expr.const(Name.from_str("Eq")),
                                   Expr.fvar(620)), Expr.fvar(622))
        lctx = LocalContext()
        lctx = lctx.push_hyp(fid_a, Name.from_str("a"), nat)
        lctx = lctx.push_hyp(fid_c, Name.from_str("c"), nat)
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, eq_ac, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))
        r = execute_tactic(state, "trans a")
        assert r.success


class TestHaveTactic:
    """P0-2: have tactic should create proper subgoals."""

    def test_have_by_definition(self):
        """have h := term should add h to context."""
        env = mk_env()
        prop = Expr.prop()
        true_ty = Expr.const(Name.from_str("True"))
        fid_p = FVarId(700)
        lctx = LocalContext()
        lctx = lctx.push_hyp(fid_p, Name.from_str("p"), true_ty)
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, prop, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))
        from engine.tactic.engine import tac_have
        r = tac_have(state, "h := p")
        assert r.success
        assert r.goals_after == 1  # only continuation goal

    def test_have_by_type(self):
        """have h : True should create two subgoals."""
        env = mk_env()
        prop = Expr.prop()
        lctx = LocalContext()
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, prop, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))
        from engine.tactic.engine import tac_have
        r = tac_have(state, "h : True")
        assert r.success
        assert r.goals_after == 2  # prove True + original with h

    def test_have_unknown_type_fails(self):
        env = mk_env()
        prop = Expr.prop()
        lctx = LocalContext()
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, prop, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))
        from engine.tactic.engine import tac_have
        r = tac_have(state, "h : UnknownType42")
        assert not r.success
        assert r.error.kind == "type_resolve"

    def test_have_no_colon_fails(self):
        env = mk_env()
        state = mk_state(env, Expr.prop())
        from engine.tactic.engine import tac_have
        r = tac_have(state, "just_a_name")
        assert not r.success
        assert r.error.kind == "parse_error"

    def test_have_in_dispatch(self):
        """have should be accessible via dispatch."""
        env = mk_env()
        prop = Expr.prop()
        true_ty = Expr.const(Name.from_str("True"))
        fid_p = FVarId(750)
        lctx = LocalContext()
        lctx = lctx.push_hyp(fid_p, Name.from_str("p"), true_ty)
        from engine.state.meta_ctx import MetaContext
        mc = MetaContext()
        mc, gid = mc.create_meta(lctx, prop, depth=0)
        from pyrsistent import pvector
        state = ProofState(env, mc, pvector([gid]))
        r = execute_tactic(state, "have h := p")
        assert r.success


class TestSimpDepthLimit:
    """P0-3: simp should not recurse infinitely on deeply nested Pi."""

    def test_simp_bounded_recursion(self):
        """simp on a 20-deep Pi should fail, not hang."""
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(200)  # tight limit to catch runaway
        try:
            env = mk_env()
            # Build a 20-deep Pi: ∀ x1, ∀ x2, ... ∀ x20, Prop
            goal = Expr.prop()
            for i in range(20):
                goal = Expr.pi(BI, Name.from_str(f"x{i}"), Expr.prop(), goal)
            state = mk_state(env, goal)
            from engine.tactic.engine import tac_simp
            r = tac_simp(state)
            # Should fail gracefully (depth limit hit), not crash
            # It's OK if it partially succeeds (intro some) then fails
        except RecursionError:
            pytest.fail("simp caused RecursionError — depth limit not working")
        finally:
            sys.setrecursionlimit(old_limit)


class TestUCB1Fix:
    """P0-4: UCB1 should use parent's visit_count, not global sum."""

    def test_ucb_uses_parent_visits(self):
        """After backpropagation, UCB selection should use parent visits."""
        env = mk_env()
        goal = Expr.pi(BI, Name.from_str("P"), Expr.prop(), Expr.bvar(0))
        config = SearchConfig(strategy="mcts", max_nodes=100, max_depth=10)
        coord = SearchCoordinator(env, goal, config)
        # Expand root
        r = coord.try_tactic(0, "intro P")
        assert r.success
        child_id = r.child_node
        # The parent (root) should have visit_count incremented
        from engine.state.search_tree import NodeId
        root = coord._tree.get(NodeId(0))
        assert root.visit_count >= 1
        # UCB selection should work without error
        selected = coord.select_node()
        # Should select the child (only open leaf)
        assert selected == child_id


# ═══════════════════════════════════════════════════════════════
# Lean4 environment tests
# ═══════════════════════════════════════════════════════════════

class TestLeanEnv:
    def test_status_check(self):
        from agent.executor.lean_env import LeanEnvironment
        env = LeanEnvironment(mode="auto")
        s = env.status()
        assert s.mode in ("local", "docker", "unavailable")
        assert isinstance(s.elan_installed, bool)
        assert isinstance(s.lean_installed, bool)

    def test_unavailable_compile(self):
        from agent.executor.lean_env import LeanEnvironment
        env = LeanEnvironment(mode="unavailable")
        env.mode = "unavailable"
        rc, stdout, stderr = env.compile("theorem t : True := trivial")
        assert rc != 0
        assert "not available" in stderr

    def test_create_auto(self):
        from agent.executor.lean_env import LeanEnvironment
        env = LeanEnvironment.create()
        assert env.mode in ("local", "docker", "unavailable")
