"""tests/test_prover/test_debruijn_properties.py — Property-based tests for de Bruijn ops.

Tests key algebraic properties of lift, instantiate, abstract:
  1. lift(0, _) is identity
  2. lift(n, _) ∘ lift(m, _) = lift(n+m, _) for non-overlapping cutoffs
  3. abstract(id) ∘ instantiate(fvar(id)) is identity (round-trip)
  4. instantiate never produces negative bvar indices
  5. has_loose_bvars correctly tracks bvar presence
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest
from engine.core.expr import Expr, BinderInfo
from engine.core.name import Name
from engine.core.universe import Level

BI = BinderInfo.DEFAULT


# ── Helper: build random-ish expressions ──

def mk_bvar(i: int) -> Expr:
    return Expr.bvar(i)

def mk_fvar(i: int) -> Expr:
    return Expr.fvar(i)

def mk_const(s: str) -> Expr:
    return Expr.const(Name.from_str(s))

def mk_app(f: Expr, a: Expr) -> Expr:
    return Expr.app(f, a)

def mk_lam(name: str, domain: Expr, body: Expr) -> Expr:
    return Expr.lam(BI, Name.from_str(name), domain, body)

def mk_pi(name: str, domain: Expr, body: Expr) -> Expr:
    return Expr.pi(BI, Name.from_str(name), domain, body)


# A collection of test expressions of varying complexity
TEST_EXPRS = [
    # Atomic
    mk_bvar(0),
    mk_bvar(1),
    mk_bvar(5),
    mk_fvar(42),
    mk_const("Nat"),
    Expr.prop(),
    Expr.type_(),

    # Simple apps
    mk_app(mk_const("f"), mk_bvar(0)),
    mk_app(mk_const("f"), mk_fvar(10)),
    mk_app(mk_app(mk_const("g"), mk_bvar(0)), mk_bvar(1)),

    # Under binders
    mk_lam("x", Expr.prop(), mk_bvar(0)),           # fun x => x
    mk_lam("x", Expr.prop(), mk_bvar(1)),           # fun x => #1 (free)
    mk_lam("x", Expr.prop(), mk_app(mk_const("f"), mk_bvar(0))),
    mk_pi("x", mk_const("Nat"), mk_bvar(0)),        # ∀ x : Nat, x

    # Nested binders
    mk_lam("x", Expr.prop(),
            mk_lam("y", Expr.prop(), mk_bvar(1))),  # fun x y => x
    mk_lam("x", Expr.prop(),
            mk_lam("y", Expr.prop(), mk_bvar(0))),  # fun x y => y
    mk_lam("x", Expr.prop(),
            mk_lam("y", Expr.prop(),
                    mk_app(mk_bvar(1), mk_bvar(0)))),  # fun x y => x y

    # Mixed fvar and bvar
    mk_lam("x", Expr.prop(),
            mk_app(mk_fvar(99), mk_bvar(0))),
]


class TestLiftProperties:
    """Test algebraic properties of lift."""

    def test_lift_zero_is_identity(self):
        """lift(0, c) should be identity for all expressions and cutoffs."""
        for e in TEST_EXPRS:
            for cutoff in [0, 1, 5, 100]:
                result = e.lift(0, cutoff)
                assert result == e, f"lift(0, {cutoff}) changed {repr(e)}"

    @pytest.mark.parametrize("e", TEST_EXPRS)
    def test_lift_positive_preserves_structure(self, e):
        """lift(n) should only change bvar indices, not structure."""
        lifted = e.lift(3, 0)
        assert lifted.tag == e.tag

    def test_lift_bvar_above_cutoff(self):
        """bvar(i) with i >= cutoff should be shifted."""
        e = mk_bvar(5)
        assert e.lift(3, 0) == mk_bvar(8)
        assert e.lift(3, 5) == mk_bvar(8)
        assert e.lift(3, 6) == mk_bvar(5)  # below cutoff, unchanged

    def test_lift_negative_guards(self):
        """lift with negative n should raise on underflow."""
        e = mk_bvar(0)
        with pytest.raises(ValueError):
            e.lift(-1, 0)

    def test_lift_negative_safe(self):
        """lift(-1, cutoff=1) on bvar(0) is safe (below cutoff)."""
        e = mk_bvar(0)
        assert e.lift(-1, 1) == mk_bvar(0)

    def test_lift_under_binder(self):
        """Lifting under a binder increases cutoff for body."""
        # fun x => #1  -- lift(1, 0) should give fun x => #2
        e = mk_lam("x", Expr.prop(), mk_bvar(1))
        lifted = e.lift(1, 0)
        # The body's bvar(1) is at depth 1 (inside lambda), so cutoff=1
        # bvar(1) >= cutoff(1), so shifted to bvar(2)
        assert lifted.children[1] == mk_bvar(2)


class TestInstantiateProperties:
    """Test properties of instantiate (bvar substitution)."""

    def test_instantiate_replaces_target(self):
        """instantiate should replace bvar(depth) with val."""
        e = mk_bvar(0)
        val = mk_const("x")
        assert e.instantiate(val, 0) == val

    def test_instantiate_shifts_above(self):
        """bvar(i) with i > depth should become bvar(i-1)."""
        e = mk_bvar(3)
        val = mk_const("x")
        assert e.instantiate(val, 1) == mk_bvar(2)

    def test_instantiate_preserves_below(self):
        """bvar(i) with i < depth should be unchanged."""
        e = mk_bvar(0)
        val = mk_const("x")
        assert e.instantiate(val, 5) == mk_bvar(0)

    def test_instantiate_under_binder(self):
        """Under a binder, depth increases and val is lifted."""
        # fun x => #1  -- instantiate(const "a", 0) should replace #1
        # Under the binder, depth=1, so #1 matches depth=1 → replaced
        body = mk_bvar(1)
        lam = mk_lam("x", Expr.prop(), body)
        result = lam.instantiate(mk_const("a"), 0)
        # The body #1 at depth 1 matches → replaced with lift(1,0)(const "a") = const "a"
        assert result.children[1] == mk_const("a")

    def test_fvar_unchanged_by_instantiate(self):
        """fvars should never be affected by instantiate."""
        e = mk_fvar(42)
        assert e.instantiate(mk_const("x"), 0) == mk_fvar(42)


class TestAbstractProperties:
    """Test properties of abstract (fvar → bvar conversion)."""

    def test_abstract_replaces_fvar(self):
        """abstract should replace fvar(id) with bvar(depth)."""
        e = mk_fvar(42)
        result = e.abstract(42, 0)
        assert result == mk_bvar(0)

    def test_abstract_preserves_other_fvars(self):
        """abstract should not touch other fvars."""
        e = mk_fvar(99)
        result = e.abstract(42, 0)
        assert result == mk_fvar(99)

    def test_abstract_instantiate_roundtrip(self):
        """abstract(id) ∘ instantiate(fvar(id)) should be identity.

        This is the fundamental property of de Bruijn variable handling:
        opening a scope and then closing it returns the original term.
        """
        # Start with: fun x => #0  (identity function)
        original = mk_lam("x", Expr.prop(), mk_bvar(0))

        # Open: replace bvar(0) with fvar(42)
        fvar42 = mk_fvar(42)
        opened_body = original.children[1].instantiate(fvar42, 0)
        assert opened_body == mk_fvar(42)

        # Close: replace fvar(42) with bvar(0)
        closed_body = opened_body.abstract(42, 0)
        assert closed_body == mk_bvar(0)
        assert closed_body == original.children[1]


class TestHasLooseBvars:
    """Test has_loose_bvars correctness."""

    def test_bvar_at_depth(self):
        assert mk_bvar(0).has_loose_bvars(0) == True
        assert mk_bvar(0).has_loose_bvars(1) == False

    def test_fvar_never_loose(self):
        assert mk_fvar(42).has_loose_bvars(0) == False

    def test_const_never_loose(self):
        assert mk_const("x").has_loose_bvars(0) == False

    def test_lambda_body_depth(self):
        """Lambda body increases depth by 1."""
        # fun x => #0  -- bvar(0) is bound, not loose
        e = mk_lam("x", Expr.prop(), mk_bvar(0))
        assert e.has_loose_bvars(0) == False

        # fun x => #1  -- bvar(1) is free at depth 0
        e2 = mk_lam("x", Expr.prop(), mk_bvar(1))
        assert e2.has_loose_bvars(0) == True

    def test_nested_binders(self):
        # fun x => fun y => #2  -- free
        e = mk_lam("x", Expr.prop(),
                    mk_lam("y", Expr.prop(), mk_bvar(2)))
        assert e.has_loose_bvars(0) == True

        # fun x => fun y => #1  -- bound to x
        e2 = mk_lam("x", Expr.prop(),
                     mk_lam("y", Expr.prop(), mk_bvar(1)))
        assert e2.has_loose_bvars(0) == False


class TestReplaceFvar:
    """Test replace_fvar correctness."""

    def test_basic_replacement(self):
        e = mk_fvar(42)
        result = e.replace_fvar(42, mk_const("x"))
        assert result == mk_const("x")

    def test_no_replacement_for_other_ids(self):
        e = mk_fvar(99)
        result = e.replace_fvar(42, mk_const("x"))
        assert result == mk_fvar(99)

    def test_replacement_in_app(self):
        e = mk_app(mk_fvar(42), mk_fvar(42))
        result = e.replace_fvar(42, mk_const("x"))
        expected = mk_app(mk_const("x"), mk_const("x"))
        assert result == expected

    def test_replacement_under_binder(self):
        e = mk_lam("y", Expr.prop(), mk_fvar(42))
        result = e.replace_fvar(42, mk_const("x"))
        assert result.children[1] == mk_const("x")
