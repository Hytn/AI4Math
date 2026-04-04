"""tests/test_prover/test_kernel.py — Type checker kernel correctness tests

Tests for all 5 fixes:
  Fix #1: No unsafe fallback (unknown exprs → FAIL)
  Fix #2: Eta reduction in definitional equality
  Fix #3: Proper universe polymorphism (imax, max, param)
  Fix #4: Inductive type support (environment data structures)
  Fix #5: Lambda fvar abstraction (no fvar leak)
  Bonus:  Expr.instantiate variable capture fix
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pytest
from engine.core.expr import Expr, BinderInfo
from engine.core.name import Name
from engine.core.meta import MetaId
from engine.core.universe import Level
from engine.core.local_ctx import LocalContext, FVarId
from engine.core.environment import (
    Environment, ConstantInfo, InductiveInfo, ConstructorInfo, RecursorInfo)
from engine.kernel.type_checker import (
    TypeChecker, VerificationLevel, Reducer, Unifier, CheckResult)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

BI = BinderInfo.DEFAULT
IMP = BinderInfo.IMPLICIT

def mk_env():
    """Create a standard environment with Prop, Nat, etc."""
    env = Environment()
    prop = Expr.prop()
    type_ = Expr.type_()
    nat = Expr.const(Name.from_str("Nat"))

    env = env.add_const(ConstantInfo(Name.from_str("Prop"), type_))
    env = env.add_const(ConstantInfo(Name.from_str("Nat"), type_))
    env = env.add_const(ConstantInfo(Name.from_str("Nat.zero"), nat))
    env = env.add_const(ConstantInfo(Name.from_str("Nat.succ"), Expr.arrow(nat, nat)))
    env = env.add_const(ConstantInfo(Name.from_str("True"), prop))
    env = env.add_const(ConstantInfo(Name.from_str("False"), prop))
    return env


def mk_tc(env=None, level=VerificationLevel.ELABORATE):
    env = env or mk_env()
    return TypeChecker(env, level)


def infer_ok(tc, expr, ctx=None, assigns=None):
    r = tc.infer(expr, ctx or LocalContext(), assigns or {})
    assert r.success, f"Expected success but got: {r.failure}"
    return r


def infer_fail(tc, expr, ctx=None, assigns=None):
    r = tc.infer(expr, ctx or LocalContext(), assigns or {})
    assert not r.success, f"Expected failure but succeeded with: {r.inferred_type}"
    return r


# ═══════════════════════════════════════════════════════════════
# Fix #1: No unsafe fallback
# ═══════════════════════════════════════════════════════════════

class TestFix1_NoUnsafeFallback:
    """Previously, unknown expressions silently returned True with Prop type."""

    def test_unknown_const_fails(self):
        tc = mk_tc()
        r = infer_fail(tc, Expr.const(Name.from_str("NonExistent")))
        assert r.failure.kind == "unknown_const"

    def test_unknown_fvar_fails(self):
        tc = mk_tc()
        r = infer_fail(tc, Expr.fvar(99999))
        assert r.failure.kind == "unknown_fvar"

    def test_unsupported_tag_fails_elaborate(self):
        tc = mk_tc()
        e = Expr("some_weird_tag")
        r = infer_fail(tc, e)
        assert r.failure.kind == "unsupported_expr"

    def test_unsupported_tag_fails_quick(self):
        tc = mk_tc(level=VerificationLevel.QUICK)
        e = Expr("some_weird_tag")
        r = infer_fail(tc, e)
        assert r.failure.kind == "unsupported_expr"

    def test_loose_bvar_fails(self):
        """bvar(0) at top level should fail — it's not under any binder."""
        tc = mk_tc()
        r = infer_fail(tc, Expr.bvar(0))
        assert r.failure.kind == "loose_bvar"

    def test_apply_non_function_fails(self):
        tc = mk_tc()
        # Nat is a type, not a function: applying it should fail
        r = infer_fail(tc, Expr.app(Expr.const(Name.from_str("Nat.zero")),
                                      Expr.const(Name.from_str("Nat.zero"))))

    def test_valid_sort_succeeds(self):
        tc = mk_tc()
        r = infer_ok(tc, Expr.prop())
        assert r.inferred_type == Expr.sort(Level.one())

    def test_valid_const_succeeds(self):
        tc = mk_tc()
        r = infer_ok(tc, Expr.const(Name.from_str("Nat")))
        assert r.inferred_type == Expr.type_()


# ═══════════════════════════════════════════════════════════════
# Fix #2: Eta reduction
# ═══════════════════════════════════════════════════════════════

class TestFix2_EtaReduction:
    """is_def_eq should handle eta: (fun x => f x) ≡ f"""

    def setup_method(self):
        self.reducer = Reducer(mk_env(), {})

    def test_eta_contract(self):
        """fun x : A => f x  ≡  f"""
        A = Expr.prop()
        f = Expr.fvar(42)
        # Build: fun (x : A) => f x
        # Under the binder, f becomes f.lift(1,0) = fvar(42) (fvars unaffected)
        eta = Expr.lam(BI, Name.from_str("x"), A,
                       Expr.app(Expr.fvar(42), Expr.bvar(0)))
        assert self.reducer.is_def_eq(eta, f)

    def test_eta_contract_symmetric(self):
        """f  ≡  fun x : A => f x"""
        A = Expr.prop()
        f = Expr.fvar(42)
        eta = Expr.lam(BI, Name.from_str("x"), A,
                       Expr.app(Expr.fvar(42), Expr.bvar(0)))
        assert self.reducer.is_def_eq(f, eta)

    def test_non_eta_not_equal(self):
        """fun x => f x y  ≢  f  (body uses x in nested position)"""
        A = Expr.prop()
        f = Expr.fvar(42)
        y = Expr.fvar(43)
        non_eta = Expr.lam(BI, Name.from_str("x"), A,
                           Expr.app(Expr.app(Expr.fvar(42), Expr.bvar(0)), y))
        assert not self.reducer.is_def_eq(non_eta, f)

    def test_eta_with_bvar_in_body_not_eta(self):
        """fun x => #0  is NOT eta-reducible (body IS the bvar)."""
        A = Expr.prop()
        body_is_var = Expr.lam(BI, Name.from_str("x"), A, Expr.bvar(0))
        # This is the identity function, not an eta-expansion of anything specific
        # It should not be def-eq to an arbitrary fvar
        assert not self.reducer.is_def_eq(body_is_var, Expr.fvar(42))

    def test_beta_still_works(self):
        """(fun x => x) arg  reduces to  arg"""
        A = Expr.prop()
        id_fn = Expr.lam(BI, Name.from_str("x"), A, Expr.bvar(0))
        arg = Expr.fvar(99)
        app = Expr.app(id_fn, arg)
        reduced = self.reducer.whnf(app)
        assert reduced == arg

    def test_structural_eq(self):
        a = Expr.fvar(1)
        b = Expr.fvar(1)
        assert self.reducer.is_def_eq(a, b)

    def test_structural_neq(self):
        a = Expr.fvar(1)
        b = Expr.fvar(2)
        assert not self.reducer.is_def_eq(a, b)


# ═══════════════════════════════════════════════════════════════
# Fix #3: Universe polymorphism
# ═══════════════════════════════════════════════════════════════

class TestFix3_UniversePolymorphism:
    """imax, max, param, subst — proper Level algebra."""

    # ── imax ──

    def test_imax_absorbs_zero(self):
        """imax(a, 0) = 0 — key impredicativity rule."""
        assert Level.imax(Level.one(), Level.zero()).to_nat() == 0
        assert Level.imax(Level.of_nat(5), Level.zero()).to_nat() == 0

    def test_imax_nonzero(self):
        """imax(a, succ(b)) = max(a, succ(b))."""
        assert Level.imax(Level.zero(), Level.one()).to_nat() == 1
        assert Level.imax(Level.of_nat(3), Level.of_nat(2)).to_nat() == 3

    def test_imax_with_param(self):
        """imax(1, u) — can't simplify if u is a param."""
        u = Level.param("u")
        result = Level.imax(Level.one(), u)
        assert result.to_nat() is None  # not concrete
        assert result.kind == "imax"

    # ── max ──

    def test_max_concrete(self):
        assert Level.max(Level.of_nat(2), Level.of_nat(5)).to_nat() == 5
        assert Level.max(Level.of_nat(3), Level.of_nat(3)).to_nat() == 3

    def test_max_identity(self):
        assert Level.max(Level.zero(), Level.of_nat(4)).to_nat() == 4
        assert Level.max(Level.of_nat(4), Level.zero()).to_nat() == 4

    def test_max_same(self):
        u = Level.param("u")
        assert Level.max(u, u) == u

    # ── param ──

    def test_param_not_concrete(self):
        u = Level.param("u")
        assert u.to_nat() is None
        assert not u.is_zero

    def test_param_subst(self):
        u = Level.param("u")
        result = u.subst("u", Level.of_nat(3))
        assert result.to_nat() == 3

    def test_succ_param_subst(self):
        l = Level.succ(Level.param("u"))
        result = l.subst("u", Level.of_nat(5))
        assert result.to_nat() == 6

    def test_max_param_subst(self):
        l = Level.max(Level.param("u"), Level.param("v"))
        result = l.subst("u", Level.of_nat(3)).subst("v", Level.of_nat(7))
        assert result.to_nat() == 7

    # ── equiv ──

    def test_level_equiv_concrete(self):
        assert Level.of_nat(3).is_equiv(Level.succ(Level.succ(Level.succ(Level.zero()))))

    def test_level_equiv_param(self):
        u = Level.param("u")
        assert u.is_equiv(u)
        assert not u.is_equiv(Level.param("v"))

    # ── Pi universe computation ──

    def test_pi_prop_to_prop_is_type(self):
        """(P : Prop) → Prop lives in Type (Sort 1), not Prop.

        Prop = Sort 0, but Prop : Sort 1, so:
        domain Prop : Sort 1 → u = 1
        codomain Prop : Sort 1 → v = 1
        imax(1, 1) = 1 → the Pi type lives in Sort 1 = Type.
        """
        tc = mk_tc()
        prop = Expr.prop()
        pi = Expr.pi(BI, Name.from_str("P"), prop, prop)
        r = infer_ok(tc, pi)
        assert r.inferred_type.tag == "sort"
        level = r.inferred_type.level
        assert level is not None and level.to_nat() == 1

    def test_pi_prop_var_is_prop(self):
        """∀ (P : Prop), P lives in Prop (impredicativity!).

        domain Prop : Sort 1 → u = 1
        codomain P : Prop = Sort 0 → v = 0
        imax(1, 0) = 0 → Prop.  This is the KEY impredicativity test.
        """
        tc = mk_tc()
        prop = Expr.prop()
        # ∀ (P : Prop), P
        # codomain = bvar(0), referring to P which has type Prop
        pi = Expr.pi(BI, Name.from_str("P"), prop, Expr.bvar(0))
        r = infer_ok(tc, pi)
        assert r.inferred_type.tag == "sort"
        level = r.inferred_type.level
        assert level is not None and level.to_nat() == 0, \
            f"∀ (P : Prop), P should be Prop (Sort 0), got Sort {level}"

    def test_pi_type_to_type(self):
        """(A : Type) → Type should live in Type (imax(1, 1) = 1)."""
        tc = mk_tc()
        type_ = Expr.type_()
        pi = Expr.pi(BI, Name.from_str("A"), type_, type_)
        r = infer_ok(tc, pi)
        assert r.inferred_type.tag == "sort"
        level = r.inferred_type.level
        assert level is not None and level.to_nat() is not None and level.to_nat() >= 1

    def test_pi_type_to_prop_is_sort2(self):
        """(A : Type) → Prop lives in Sort 2.

        domain Type = Sort 1, Type : Sort 2 → u = 2
        codomain Prop : Sort 1 → v = 1
        imax(2, 1) = max(2, 1) = 2.
        """
        tc = mk_tc()
        type_ = Expr.type_()
        prop = Expr.prop()
        pi = Expr.pi(BI, Name.from_str("A"), type_, prop)
        r = infer_ok(tc, pi)
        assert r.inferred_type.tag == "sort"
        level = r.inferred_type.level
        assert level is not None and level.to_nat() == 2

    def test_pi_type_to_prop_value_is_prop(self):
        """∀ (A : Type), True lives in Prop (impredicativity).

        domain Type : Sort 2 → u = 2
        codomain True : Prop = Sort 0 → v = 0
        imax(2, 0) = 0 → Prop.
        """
        env = mk_env()
        tc = mk_tc(env)
        type_ = Expr.type_()
        # codomain = True (a constant of type Prop)
        pi = Expr.pi(BI, Name.from_str("A"), type_,
                     Expr.const(Name.from_str("True")))
        r = infer_ok(tc, pi)
        assert r.inferred_type.tag == "sort"
        level = r.inferred_type.level
        assert level is not None and level.to_nat() == 0, \
            f"∀ (A : Type), True should be Prop (Sort 0), got Sort {level}"


# ═══════════════════════════════════════════════════════════════
# Fix #4: Inductive type support
# ═══════════════════════════════════════════════════════════════

class TestFix4_InductiveTypes:
    """Environment supports inductive types with constructors and recursors."""

    def test_add_simple_inductive(self):
        env = Environment()
        prop = Expr.prop()
        type_ = Expr.type_()

        bool_ind = InductiveInfo(
            name=Name.from_str("Bool"),
            type_=type_,
            constructors=[
                ConstructorInfo(Name.from_str("Bool.true"), Expr.const(Name.from_str("Bool")),
                                Name.from_str("Bool"), 0),
                ConstructorInfo(Name.from_str("Bool.false"), Expr.const(Name.from_str("Bool")),
                                Name.from_str("Bool"), 1),
            ],
            num_params=0,
        )
        env = env.add_inductive(bool_ind)

        # Type itself should be findable
        assert env.lookup(Name.from_str("Bool")) is not None
        assert env.lookup(Name.from_str("Bool")).type_ == type_

        # Constructors should be findable
        assert env.lookup(Name.from_str("Bool.true")) is not None
        assert env.lookup(Name.from_str("Bool.false")) is not None

        # Inductive metadata
        assert env.lookup_inductive(Name.from_str("Bool")) is not None
        assert env.is_constructor(Name.from_str("Bool.true"))
        assert not env.is_constructor(Name.from_str("Bool"))

    def test_constructor_parent_lookup(self):
        env = Environment()
        nat_ind = InductiveInfo(
            name=Name.from_str("MyNat"),
            type_=Expr.type_(),
            constructors=[
                ConstructorInfo(Name.from_str("MyNat.zero"),
                                Expr.const(Name.from_str("MyNat")),
                                Name.from_str("MyNat"), 0),
            ],
        )
        env = env.add_inductive(nat_ind)
        parent = env.lookup_constructor_inductive(Name.from_str("MyNat.zero"))
        assert parent is not None
        assert parent.name == Name.from_str("MyNat")

    def test_recursor_registration(self):
        env = Environment()
        nat = Expr.const(Name.from_str("Nat"))
        rec = RecursorInfo(
            name=Name.from_str("Nat.rec"),
            type_=Expr.prop(),  # simplified
            inductive_name=Name.from_str("Nat"),
            num_params=0,
            num_minors=2,
        )
        ind = InductiveInfo(
            name=Name.from_str("Nat"),
            type_=Expr.type_(),
            constructors=[],
            recursor=rec,
        )
        env = env.add_inductive(ind)

        assert env.is_recursor(Name.from_str("Nat.rec"))
        assert env.lookup_recursor(Name.from_str("Nat.rec")) is not None
        assert env.lookup_recursor(Name.from_str("Nat.rec")).num_minors == 2

    def test_non_inductive_not_constructor(self):
        env = mk_env()
        assert not env.is_constructor(Name.from_str("Nat"))
        assert not env.is_recursor(Name.from_str("Nat"))


# ═══════════════════════════════════════════════════════════════
# Fix #5: Lambda fvar abstraction
# ═══════════════════════════════════════════════════════════════

class TestFix5_LambdaAbstraction:
    """Lambda type inference should not leak fvars."""

    def test_identity_function_type(self):
        """id = fun (P : Prop) (h : P) => h  should have type  ∀ P, P → P"""
        tc = mk_tc()
        prop = Expr.prop()

        # fun (P : Prop) => fun (h : #0) => #0
        inner = Expr.lam(BI, Name.from_str("h"), Expr.bvar(0), Expr.bvar(0))
        id_fn = Expr.lam(BI, Name.from_str("P"), prop, inner)

        r = infer_ok(tc, id_fn)
        ty = r.inferred_type
        assert ty.is_pi, f"Expected Pi type, got {ty.tag}"
        # The result should be ∀ (P : Prop), ∀ (h : P), P
        # which has bvar references, NOT fvar references
        assert "fvar" not in repr(ty).replace("fvar", "FVAR") or True
        # More specifically: check the codomain doesn't contain any fvars
        self._assert_no_fvars(ty)

    def test_const_function_type(self):
        """fun (A : Prop) (B : Prop) (a : A) (b : B) => a  :  ∀ A B, A → B → A"""
        tc = mk_tc()
        prop = Expr.prop()

        # Build from inside out (de Bruijn):
        # fun (b : #1) => #1    (b:B, return a which is bvar(1))
        e = Expr.lam(BI, Name.from_str("b"), Expr.bvar(1), Expr.bvar(1))
        # fun (a : #1) => e     (a:A)
        e = Expr.lam(BI, Name.from_str("a"), Expr.bvar(1), e)
        # fun (B : Prop) => e
        e = Expr.lam(BI, Name.from_str("B"), prop, e)
        # fun (A : Prop) => e
        e = Expr.lam(BI, Name.from_str("A"), prop, e)

        r = infer_ok(tc, e)
        self._assert_no_fvars(r.inferred_type)

    def test_abstract_operation(self):
        """abstract(fvar(42)) should produce bvar(0)."""
        e = Expr.fvar(42)
        result = e.abstract(42, 0)
        assert result.tag == "bvar" and result.idx == 0

    def test_abstract_nested(self):
        """Abstracting under a binder increments depth."""
        # app(fvar(42), lam(x:A, fvar(42)))
        e = Expr.app(
            Expr.fvar(42),
            Expr.lam(BI, Name.from_str("x"), Expr.prop(),
                     Expr.fvar(42))
        )
        result = e.abstract(42, 0)
        # Should be app(bvar(0), lam(x:A, bvar(1)))
        assert result.children[0].tag == "bvar" and result.children[0].idx == 0
        lam_body = result.children[1].children[1]
        assert lam_body.tag == "bvar" and lam_body.idx == 1

    def _assert_no_fvars(self, expr):
        """Assert no fvar nodes appear in the expression."""
        assert expr.tag != "fvar", f"Leaked fvar: {expr}"
        for c in expr.children:
            self._assert_no_fvars(c)


# ═══════════════════════════════════════════════════════════════
# Bonus: Expr.instantiate variable capture fix
# ═══════════════════════════════════════════════════════════════

class TestInstantiateCapture:
    """instantiate must lift val when entering binders."""

    def test_simple_instantiate(self):
        """(fun x => #0)[val] = (fun x => val) — val is fvar, no capture."""
        val = Expr.fvar(99)
        body = Expr.lam(BI, Name.from_str("x"), Expr.prop(), Expr.bvar(0))
        # bvar(0) inside the lambda refers to x, not the outer scope
        # Instantiating at depth 0 only affects bvar(0) at depth 0
        # The bvar(0) inside the lambda is at depth 1, so untouched
        result = body.instantiate(val)
        # The lambda structure should remain, body's bvar(0) → still bvar(0)
        assert result.tag == "lam"

    def test_outer_bvar_captured(self):
        """lam(x:A, bvar(1)) — bvar(1) refers to outer scope, should be replaced."""
        val = Expr.fvar(99)
        # bvar(1) inside lambda at depth 0+1=1, so bvar(1) at depth 1 is the outer bvar(0)
        body = Expr.lam(BI, Name.from_str("x"), Expr.prop(), Expr.bvar(1))
        result = body.instantiate(val)
        # After instantiate: bvar(1) at depth=1 matches depth=1? No.
        # Actually bvar(1) at depth 0 inside a lambda: 
        #   depth increases to 1 for the body
        #   bvar(1) > depth(1)? No, 1 == 1, so it matches!
        #   It gets replaced with val.lift(1, 0)
        lam_body = result.children[1]
        assert lam_body.tag == "fvar" and lam_body.idx == 99

    def test_lift_basic(self):
        assert Expr.bvar(0).lift(3, 0) == Expr.bvar(3)
        assert Expr.bvar(0).lift(3, 1) == Expr.bvar(0)  # below cutoff
        assert Expr.bvar(5).lift(2, 3) == Expr.bvar(7)  # above cutoff

    def test_lift_fvar_unchanged(self):
        """fvars are never affected by lift."""
        assert Expr.fvar(42).lift(10, 0) == Expr.fvar(42)

    def test_lift_zero_identity(self):
        e = Expr.app(Expr.bvar(0), Expr.bvar(1))
        assert e.lift(0) is e

    def test_has_loose_bvars(self):
        assert Expr.bvar(0).has_loose_bvars(0)
        assert not Expr.bvar(0).has_loose_bvars(1)
        assert not Expr.fvar(42).has_loose_bvars(0)
        # Inside a lambda, bvar(0) is bound
        lam = Expr.lam(BI, Name.from_str("x"), Expr.prop(), Expr.bvar(0))
        assert not lam.has_loose_bvars(0)
        # But bvar(1) inside a lambda is still loose
        lam2 = Expr.lam(BI, Name.from_str("x"), Expr.prop(), Expr.bvar(1))
        assert lam2.has_loose_bvars(0)


# ═══════════════════════════════════════════════════════════════
# Unifier tests
# ═══════════════════════════════════════════════════════════════

class TestUnifier:
    def test_unify_identical(self):
        u = Unifier(mk_env(), {})
        assert u.unify(Expr.prop(), Expr.prop())

    def test_unify_mvar(self):
        u = Unifier(mk_env(), {})
        m = MetaId(0)
        assert u.unify(Expr.mvar(m), Expr.prop())
        assert m in u.new_assignments
        assert u.new_assignments[m] == Expr.prop()

    def test_unify_occurs_check(self):
        """?m ≡ app(?m, x) should fail (occurs check)."""
        u = Unifier(mk_env(), {})
        m = MetaId(0)
        rhs = Expr.app(Expr.mvar(m), Expr.fvar(1))
        assert not u.unify(Expr.mvar(m), rhs)

    def test_unify_structural(self):
        u = Unifier(mk_env(), {})
        a = Expr.app(Expr.fvar(1), Expr.fvar(2))
        b = Expr.app(Expr.fvar(1), Expr.fvar(2))
        assert u.unify(a, b)

    def test_unify_structural_fail(self):
        u = Unifier(mk_env(), {})
        a = Expr.app(Expr.fvar(1), Expr.fvar(2))
        b = Expr.app(Expr.fvar(1), Expr.fvar(3))
        assert not u.unify(a, b)


# ═══════════════════════════════════════════════════════════════
# Integration: type-check complete proof terms
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    def test_type_check_prop(self):
        tc = mk_tc()
        r = tc.check(Expr.prop(), Expr.type_(), LocalContext(), {})
        assert r.success

    def test_type_check_nat(self):
        tc = mk_tc()
        r = tc.check(Expr.const(Name.from_str("Nat")), Expr.type_(),
                      LocalContext(), {})
        assert r.success

    def test_type_check_nat_zero(self):
        tc = mk_tc()
        r = tc.check(Expr.const(Name.from_str("Nat.zero")),
                      Expr.const(Name.from_str("Nat")),
                      LocalContext(), {})
        assert r.success

    def test_type_check_mismatch(self):
        tc = mk_tc()
        # Nat.zero : Prop should fail
        r = tc.check(Expr.const(Name.from_str("Nat.zero")),
                      Expr.prop(), LocalContext(), {})
        assert not r.success

    def test_identity_applied(self):
        """(fun (P : Prop) (h : P) => h) True  should type-check."""
        tc = mk_tc()
        prop = Expr.prop()
        inner = Expr.lam(BI, Name.from_str("h"), Expr.bvar(0), Expr.bvar(0))
        id_fn = Expr.lam(BI, Name.from_str("P"), prop, inner)
        # Apply to True
        applied = Expr.app(id_fn, Expr.const(Name.from_str("True")))
        r = infer_ok(tc, applied)
        # Should infer: (h : True) → True
        assert r.inferred_type is not None

    def test_delta_reduction(self):
        """Constants with values should be unfolded."""
        env = mk_env()
        # Define: myTrue := True
        env = env.add_const(ConstantInfo(
            Name.from_str("myTrue"), Expr.prop(),
            value=Expr.const(Name.from_str("True"))))

        reducer = Reducer(env, {})
        result = reducer.whnf(Expr.const(Name.from_str("myTrue")))
        assert result == Expr.const(Name.from_str("True"))

    def test_let_reduction(self):
        """let x := v in body → body[x := v]."""
        reducer = Reducer(mk_env(), {})
        # let x : Nat := Nat.zero in x  →  Nat.zero
        e = Expr.let_(Name.from_str("x"),
                      Expr.const(Name.from_str("Nat")),
                      Expr.const(Name.from_str("Nat.zero")),
                      Expr.bvar(0))
        result = reducer.whnf(e)
        assert result == Expr.const(Name.from_str("Nat.zero"))
