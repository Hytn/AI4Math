"""tests/conftest.py — Shared test fixtures

Provides a shared ``mk_env()`` and ``mk_state()`` used across all
test modules, eliminating the ~3× duplicated environment construction.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from engine.core import Expr, Name, BinderInfo, MetaId, FVarId, LocalContext
from engine.core.universe import Level
from engine.core.environment import (
    Environment, ConstantInfo, InductiveInfo, ConstructorInfo, RecursorInfo,
)
from engine.state.proof_state import ProofState

BI = BinderInfo.DEFAULT
IMP = BinderInfo.IMPLICIT


def mk_standard_env() -> Environment:
    """Create environment with Prop, Nat, Bool, And, Or, Eq, etc.

    This is the canonical shared environment for unit tests.
    Type signatures are as accurate as possible within APE's
    simplified CIC (no full universe polymorphism).

    Key types:
      Eq       : Type → Type → Prop          (simplified: elides implicit type arg)
      Eq.refl  : ∀ {α : Type} (a : α), Eq a a
      Eq.symm  : ∀ {α : Type} {a b : α}, Eq a b → Eq b a
      Eq.trans : ∀ {α : Type} {a b c : α}, Eq a b → Eq b c → Eq a c
      And.left : ∀ {a b : Prop}, a ∧ b → a
      And.right: ∀ {a b : Prop}, a ∧ b → b
      Nat.rec  : ∀ {motive : Nat → Sort u}, motive 0 → (∀ n, motive n → motive (succ n)) → ∀ n, motive n
    """
    env = Environment()
    prop = Expr.prop()
    type_ = Expr.type_()
    nat = Expr.const(Name.from_str("Nat"))

    env = env.add_const(ConstantInfo(Name.from_str("Prop"), type_))
    env = env.add_const(ConstantInfo(Name.from_str("Nat"), type_))
    env = env.add_const(ConstantInfo(Name.from_str("Nat.zero"), nat))
    env = env.add_const(ConstantInfo(
        Name.from_str("Nat.succ"), Expr.arrow(nat, nat)))
    env = env.add_const(ConstantInfo(Name.from_str("True"), prop))
    env = env.add_const(ConstantInfo(Name.from_str("False"), prop))

    # ── Nat inductive info ──
    env = env.add_inductive(InductiveInfo(
        name=Name.from_str("Nat"),
        type_=type_,
        constructors=[
            ConstructorInfo(Name.from_str("Nat.zero"), nat,
                            Name.from_str("Nat"), idx=0),
            ConstructorInfo(Name.from_str("Nat.succ"), Expr.arrow(nat, nat),
                            Name.from_str("Nat"), idx=1),
        ],
        recursor=RecursorInfo(
            name=Name.from_str("Nat.rec"),
            type_=prop,  # simplified
            inductive_name=Name.from_str("Nat"),
            num_params=0, num_motives=1, num_minors=2,
        ),
        is_recursive=True,
    ))

    # ── Bool inductive info ──
    bool_ty = Expr.const(Name.from_str("Bool"))
    env = env.add_const(ConstantInfo(Name.from_str("Bool"), type_))
    env = env.add_inductive(InductiveInfo(
        name=Name.from_str("Bool"),
        type_=type_,
        constructors=[
            ConstructorInfo(Name.from_str("Bool.true"), bool_ty,
                            Name.from_str("Bool"), idx=0),
            ConstructorInfo(Name.from_str("Bool.false"), bool_ty,
                            Name.from_str("Bool"), idx=1),
        ],
        recursor=RecursorInfo(
            name=Name.from_str("Bool.rec"),
            type_=prop,
            inductive_name=Name.from_str("Bool"),
            num_params=0, num_motives=1, num_minors=2,
        ),
    ))

    # ── True.intro : True ──
    env = env.add_const(ConstantInfo(
        Name.from_str("True.intro"),
        Expr.const(Name.from_str("True"))))

    # ── And : Prop → Prop → Prop ──
    env = env.add_const(ConstantInfo(
        Name.from_str("And"),
        Expr.arrow(prop, Expr.arrow(prop, prop))))

    # ── Or : Prop → Prop → Prop ──
    env = env.add_const(ConstantInfo(
        Name.from_str("Or"),
        Expr.arrow(prop, Expr.arrow(prop, prop))))

    # ── Iff : Prop → Prop → Prop ──
    env = env.add_const(ConstantInfo(
        Name.from_str("Iff"),
        Expr.arrow(prop, Expr.arrow(prop, prop))))

    # ── Eq : Type → Type → Prop (simplified, eliding implicit type param) ──
    env = env.add_const(ConstantInfo(
        Name.from_str("Eq"),
        Expr.arrow(type_, Expr.arrow(type_, prop))))

    # ── Eq.refl : ∀ {α : Type} (a : α), Eq a a ──
    # Simplified as: ∀ (a : Type), Eq a a
    eq_refl_ty = Expr.pi(
        IMP, Name.from_str("α"), type_,
        Expr.pi(BI, Name.from_str("a"), Expr.bvar(0),
                Expr.app(Expr.app(
                    Expr.const(Name.from_str("Eq")),
                    Expr.bvar(0)),  # a
                    Expr.bvar(0))))  # a
    env = env.add_const(ConstantInfo(Name.from_str("Eq.refl"), eq_refl_ty))

    # ── Eq.symm : ∀ {α} {a b : α}, Eq a b → Eq b a ──
    eq_symm_ty = Expr.pi(
        IMP, Name.from_str("α"), type_,
        Expr.pi(IMP, Name.from_str("a"), Expr.bvar(0),
        Expr.pi(IMP, Name.from_str("b"), Expr.bvar(1),
        Expr.arrow(
            Expr.app(Expr.app(Expr.const(Name.from_str("Eq")),
                               Expr.bvar(1)), Expr.bvar(0)),  # Eq a b
            Expr.app(Expr.app(Expr.const(Name.from_str("Eq")),
                               Expr.bvar(1)), Expr.bvar(2))   # Eq b a
        ))))
    env = env.add_const(ConstantInfo(Name.from_str("Eq.symm"), eq_symm_ty))

    # ── Eq.trans : ∀ {α} {a b c : α}, Eq a b → Eq b c → Eq a c ──
    eq_trans_ty = Expr.pi(
        IMP, Name.from_str("α"), type_,
        Expr.pi(IMP, Name.from_str("a"), Expr.bvar(0),
        Expr.pi(IMP, Name.from_str("b"), Expr.bvar(1),
        Expr.pi(IMP, Name.from_str("c"), Expr.bvar(2),
        Expr.arrow(
            Expr.app(Expr.app(Expr.const(Name.from_str("Eq")),
                               Expr.bvar(2)), Expr.bvar(1)),  # Eq a b
            Expr.arrow(
                Expr.app(Expr.app(Expr.const(Name.from_str("Eq")),
                                   Expr.bvar(2)), Expr.bvar(1)),  # Eq b c
                Expr.app(Expr.app(Expr.const(Name.from_str("Eq")),
                                   Expr.bvar(4)), Expr.bvar(2))   # Eq a c
            ))))))
    env = env.add_const(ConstantInfo(Name.from_str("Eq.trans"), eq_trans_ty))

    # ── Eq.rec (simplified) ──
    env = env.add_const(ConstantInfo(Name.from_str("Eq.rec"), prop))

    # ── And.intro : ∀ {a b : Prop}, a → b → And a b ──
    and_intro_ty = Expr.pi(
        IMP, Name.from_str("a"), prop,
        Expr.pi(IMP, Name.from_str("b"), prop,
                Expr.arrow(
                    Expr.bvar(1),
                    Expr.arrow(
                        Expr.bvar(1),
                        Expr.app(Expr.app(
                            Expr.const(Name.from_str("And")),
                            Expr.bvar(3)),
                            Expr.bvar(2))))))
    env = env.add_const(ConstantInfo(
        Name.from_str("And.intro"), and_intro_ty))

    # ── And.left : ∀ {a b : Prop}, And a b → a ──
    and_left_ty = Expr.pi(
        IMP, Name.from_str("a"), prop,
        Expr.pi(IMP, Name.from_str("b"), prop,
                Expr.arrow(
                    Expr.app(Expr.app(
                        Expr.const(Name.from_str("And")),
                        Expr.bvar(1)), Expr.bvar(0)),  # And a b
                    Expr.bvar(2))))  # a
    env = env.add_const(ConstantInfo(
        Name.from_str("And.left"), and_left_ty))

    # ── And.right : ∀ {a b : Prop}, And a b → b ──
    and_right_ty = Expr.pi(
        IMP, Name.from_str("a"), prop,
        Expr.pi(IMP, Name.from_str("b"), prop,
                Expr.arrow(
                    Expr.app(Expr.app(
                        Expr.const(Name.from_str("And")),
                        Expr.bvar(1)), Expr.bvar(0)),  # And a b
                    Expr.bvar(1))))  # b
    env = env.add_const(ConstantInfo(
        Name.from_str("And.right"), and_right_ty))

    # ── Or.inl : ∀ {a b : Prop}, a → Or a b ──
    or_inl_ty = Expr.pi(
        IMP, Name.from_str("a"), prop,
        Expr.pi(IMP, Name.from_str("b"), prop,
                Expr.arrow(Expr.bvar(1),
                    Expr.app(Expr.app(
                        Expr.const(Name.from_str("Or")),
                        Expr.bvar(2)), Expr.bvar(1)))))
    env = env.add_const(ConstantInfo(Name.from_str("Or.inl"), or_inl_ty))

    # ── Or.inr : ∀ {a b : Prop}, b → Or a b ──
    or_inr_ty = Expr.pi(
        IMP, Name.from_str("a"), prop,
        Expr.pi(IMP, Name.from_str("b"), prop,
                Expr.arrow(Expr.bvar(0),
                    Expr.app(Expr.app(
                        Expr.const(Name.from_str("Or")),
                        Expr.bvar(2)), Expr.bvar(1)))))
    env = env.add_const(ConstantInfo(Name.from_str("Or.inr"), or_inr_ty))

    # ── Or.elim : ∀ {a b c : Prop}, Or a b → (a → c) → (b → c) → c ──
    or_elim_ty = Expr.pi(
        IMP, Name.from_str("a"), prop,
        Expr.pi(IMP, Name.from_str("b"), prop,
        Expr.pi(IMP, Name.from_str("c"), prop,
        Expr.arrow(
            Expr.app(Expr.app(Expr.const(Name.from_str("Or")),
                               Expr.bvar(2)), Expr.bvar(1)),  # Or a b
            Expr.arrow(
                Expr.arrow(Expr.bvar(3), Expr.bvar(1)),  # a → c
                Expr.arrow(
                    Expr.arrow(Expr.bvar(3), Expr.bvar(2)),  # b → c
                    Expr.bvar(3)))))))  # c
    env = env.add_const(ConstantInfo(Name.from_str("Or.elim"), or_elim_ty))

    # ── Iff.intro : ∀ {a b : Prop}, (a → b) → (b → a) → Iff a b ──
    iff_intro_ty = Expr.pi(
        IMP, Name.from_str("a"), prop,
        Expr.pi(IMP, Name.from_str("b"), prop,
                Expr.arrow(
                    Expr.arrow(Expr.bvar(1), Expr.bvar(0)),  # a → b
                    Expr.arrow(
                        Expr.arrow(Expr.bvar(1), Expr.bvar(2)),  # b → a
                        Expr.app(Expr.app(
                            Expr.const(Name.from_str("Iff")),
                            Expr.bvar(3)), Expr.bvar(2))))))
    env = env.add_const(ConstantInfo(Name.from_str("Iff.intro"), iff_intro_ty))

    # ── False.elim : ∀ {C : Prop}, False → C ──
    env = env.add_const(ConstantInfo(
        Name.from_str("False.elim"),
        Expr.pi(IMP, Name.from_str("C"), prop,
                Expr.arrow(
                    Expr.const(Name.from_str("False")),
                    Expr.bvar(0)))))

    # ── Not : Prop → Prop (defined as α → False) ──
    env = env.add_const(ConstantInfo(
        Name.from_str("Not"),
        Expr.arrow(prop, prop)))

    # ── APE.sorry (escape hatch for sorry tactic) ──
    env = env.add_const(ConstantInfo(Name.from_str("APE.sorry"), prop))

    return env


def mk_standard_state(env: Environment, goal_type: Expr) -> ProofState:
    """Create a proof state with one goal of the given type."""
    return ProofState.new(env, goal_type)
