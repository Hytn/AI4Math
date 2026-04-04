"""
Pre-built Lean4 prelude environment with core types and theorems.

This defines the foundational types (Prop, Nat, Bool, Eq, And, Or, etc.)
that any proof search needs. In production, these would come from .olean files.
"""
from engine.core import Expr, Name, BinderInfo, Environment, ConstantInfo
from engine.core.universe import Level


def build_prelude_env() -> Environment:
    """Build an environment with Lean4 core declarations."""
    env = Environment()
    BI = BinderInfo.DEFAULT
    IMP = BinderInfo.IMPLICIT

    # ── Universes and Sorts ──
    prop = Expr.prop()
    type0 = Expr.type_()

    # ── Core types ──
    # Nat : Type
    env = env.add_const(ConstantInfo(Name.from_str("Nat"), type0))
    nat = Expr.const(Name.from_str("Nat"))

    # Nat.zero : Nat
    env = env.add_const(ConstantInfo(Name.from_str("Nat.zero"), nat))

    # Nat.succ : Nat → Nat
    env = env.add_const(ConstantInfo(Name.from_str("Nat.succ"), Expr.arrow(nat, nat)))

    # Bool : Type
    env = env.add_const(ConstantInfo(Name.from_str("Bool"), type0))

    # True : Prop
    env = env.add_const(ConstantInfo(Name.from_str("True"), prop))

    # False : Prop
    env = env.add_const(ConstantInfo(Name.from_str("False"), prop))

    # ── Logical connectives ──
    # Not : Prop → Prop (defined as P → False)
    env = env.add_const(ConstantInfo(Name.from_str("Not"), Expr.arrow(prop, prop)))

    # And : Prop → Prop → Prop
    env = env.add_const(ConstantInfo(
        Name.from_str("And"),
        Expr.arrow(prop, Expr.arrow(prop, prop))
    ))

    # Or : Prop → Prop → Prop
    env = env.add_const(ConstantInfo(
        Name.from_str("Or"),
        Expr.arrow(prop, Expr.arrow(prop, prop))
    ))

    # And.intro : {a b : Prop} → a → b → And a b
    # Simplified: ∀ (a b : Prop), a → b → And a b
    and_intro_ty = Expr.pi(IMP, Name.from_str("a"), prop,
                   Expr.pi(IMP, Name.from_str("b"), prop,
                   Expr.arrow(Expr.bvar(1),
                   Expr.arrow(Expr.bvar(1),
                   Expr.app(Expr.app(Expr.const(Name.from_str("And")),
                                     Expr.bvar(3)), Expr.bvar(2))))))
    env = env.add_const(ConstantInfo(Name.from_str("And.intro"), and_intro_ty))

    # And.left : {a b : Prop} → And a b → a
    and_left_ty = Expr.pi(IMP, Name.from_str("a"), prop,
                  Expr.pi(IMP, Name.from_str("b"), prop,
                  Expr.arrow(
                      Expr.app(Expr.app(Expr.const(Name.from_str("And")),
                                        Expr.bvar(1)), Expr.bvar(0)),
                      Expr.bvar(2))))
    env = env.add_const(ConstantInfo(Name.from_str("And.left"), and_left_ty))

    # And.right : {a b : Prop} → And a b → b
    and_right_ty = Expr.pi(IMP, Name.from_str("a"), prop,
                   Expr.pi(IMP, Name.from_str("b"), prop,
                   Expr.arrow(
                       Expr.app(Expr.app(Expr.const(Name.from_str("And")),
                                         Expr.bvar(1)), Expr.bvar(0)),
                       Expr.bvar(1))))
    env = env.add_const(ConstantInfo(Name.from_str("And.right"), and_right_ty))

    # ── Equality ──
    # Eq : {α : Sort u} → α → α → Prop
    # Simplified: ∀ (α : Type), α → α → Prop
    env = env.add_const(ConstantInfo(
        Name.from_str("Eq"),
        Expr.pi(IMP, Name.from_str("α"), type0,
        Expr.arrow(Expr.bvar(0),
        Expr.arrow(Expr.bvar(1), prop)))
    ))

    # Eq.refl : {α : Type} → (a : α) → Eq a a
    eq_refl_ty = Expr.pi(IMP, Name.from_str("α"), type0,
                 Expr.pi(BI, Name.from_str("a"), Expr.bvar(0),
                 Expr.app(Expr.app(
                     Expr.app(Expr.const(Name.from_str("Eq")), Expr.bvar(1)),
                     Expr.bvar(0)), Expr.bvar(0))))
    env = env.add_const(ConstantInfo(Name.from_str("Eq.refl"), eq_refl_ty))

    # ── Arithmetic ──
    # Nat.add : Nat → Nat → Nat
    env = env.add_const(ConstantInfo(
        Name.from_str("Nat.add"),
        Expr.arrow(nat, Expr.arrow(nat, nat))
    ))

    # Nat.mul : Nat → Nat → Nat
    env = env.add_const(ConstantInfo(
        Name.from_str("Nat.mul"),
        Expr.arrow(nat, Expr.arrow(nat, nat))
    ))

    # Nat.le : Nat → Nat → Prop
    env = env.add_const(ConstantInfo(
        Name.from_str("Nat.le"),
        Expr.arrow(nat, Expr.arrow(nat, prop))
    ))

    # ── Key lemmas ──
    # Nat.add_zero : ∀ (n : Nat), Nat.add n 0 = n
    nat_add_zero_ty = Expr.pi(BI, Name.from_str("n"), nat, prop)  # simplified
    env = env.add_const(ConstantInfo(Name.from_str("Nat.add_zero"), nat_add_zero_ty))

    # Nat.zero_add : ∀ (n : Nat), Nat.add 0 n = n
    env = env.add_const(ConstantInfo(Name.from_str("Nat.zero_add"), nat_add_zero_ty))

    # Nat.add_comm : ∀ (n m : Nat), Nat.add n m = Nat.add m n
    nat_add_comm_ty = Expr.pi(BI, Name.from_str("n"), nat,
                      Expr.pi(BI, Name.from_str("m"), nat, prop))
    env = env.add_const(ConstantInfo(Name.from_str("Nat.add_comm"), nat_add_comm_ty))

    # Nat.add_assoc
    nat_add_assoc_ty = Expr.pi(BI, Name.from_str("a"), nat,
                       Expr.pi(BI, Name.from_str("b"), nat,
                       Expr.pi(BI, Name.from_str("c"), nat, prop)))
    env = env.add_const(ConstantInfo(Name.from_str("Nat.add_assoc"), nat_add_assoc_ty))

    # ── Iff ──
    env = env.add_const(ConstantInfo(
        Name.from_str("Iff"),
        Expr.arrow(prop, Expr.arrow(prop, prop))
    ))

    # ── Exists ──
    # Exists : {α : Type} → (α → Prop) → Prop
    env = env.add_const(ConstantInfo(
        Name.from_str("Exists"),
        Expr.pi(IMP, Name.from_str("α"), type0,
        Expr.arrow(Expr.arrow(Expr.bvar(0), prop), prop))
    ))

    # ── absurd : {a : Prop} → {b : Prop} → a → Not a → b
    absurd_ty = Expr.pi(IMP, Name.from_str("a"), prop,
                Expr.pi(IMP, Name.from_str("b"), prop,
                Expr.arrow(Expr.bvar(1),
                Expr.arrow(Expr.app(Expr.const(Name.from_str("Not")), Expr.bvar(2)),
                           Expr.bvar(1)))))
    env = env.add_const(ConstantInfo(Name.from_str("absurd"), absurd_ty))

    # trivial : True
    env = env.add_const(ConstantInfo(Name.from_str("trivial"), Expr.const(Name.from_str("True"))))

    return env
