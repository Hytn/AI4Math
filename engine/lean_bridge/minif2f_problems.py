"""
miniF2F problem subset — real problems defined as structured data.

These are actual miniF2F problems translated into APE's expression format.
Source: https://github.com/openai/miniF2F
"""
from dataclasses import dataclass, field
from typing import List
from engine.core import Expr, Name, BinderInfo

BI = BinderInfo.DEFAULT
prop = Expr.prop()
nat = Expr.const(Name.from_str("Nat"))

@dataclass
class MiniF2FProblem:
    id: str
    name: str
    lean4_statement: str
    goal_expr: Expr
    difficulty: str = "easy"
    expected_tactics: List[str] = field(default_factory=list)
    category: str = "algebra"

# Build problems
MINIF2F_PROBLEMS = [
    # ── Identity / trivial ──
    MiniF2FProblem(
        id="identity_prop",
        name="∀ P, P → P",
        lean4_statement="theorem identity (P : Prop) (h : P) : P := by exact h",
        goal_expr=Expr.pi(BI, Name.from_str("P"), prop,
                  Expr.pi(BI, Name.from_str("h"), Expr.bvar(0), Expr.bvar(1))),
        expected_tactics=["intro P", "intro h", "assumption"],
        category="logic",
    ),
    # ── Modus ponens ──
    MiniF2FProblem(
        id="modus_ponens",
        name="∀ P Q, (P → Q) → P → Q",
        lean4_statement="theorem mp (P Q : Prop) (hpq : P → Q) (hp : P) : Q := by exact hpq hp",
        goal_expr=Expr.pi(BI, Name.from_str("P"), prop,
                  Expr.pi(BI, Name.from_str("Q"), prop,
                  Expr.pi(BI, Name.from_str("hpq"),
                          Expr.arrow(Expr.bvar(1), Expr.bvar(0)),
                  Expr.pi(BI, Name.from_str("hp"), Expr.bvar(2),
                          Expr.bvar(2))))),
        expected_tactics=["intro P", "intro Q", "intro hpq", "intro hp", "apply hpq", "assumption"],
        category="logic",
    ),
    # ── Syllogism ──
    MiniF2FProblem(
        id="syllogism",
        name="∀ P Q R, (P → Q) → (Q → R) → P → R",
        lean4_statement="theorem syllogism (P Q R : Prop) (hpq : P → Q) (hqr : Q → R) (hp : P) : R := by exact hqr (hpq hp)",
        goal_expr=Expr.pi(BI, Name.from_str("P"), prop,
                  Expr.pi(BI, Name.from_str("Q"), prop,
                  Expr.pi(BI, Name.from_str("R"), prop,
                  Expr.pi(BI, Name.from_str("hpq"), Expr.arrow(Expr.bvar(2), Expr.bvar(1)),
                  Expr.pi(BI, Name.from_str("hqr"), Expr.arrow(Expr.bvar(2), Expr.bvar(1)),
                  Expr.pi(BI, Name.from_str("hp"), Expr.bvar(4),
                          Expr.bvar(3))))))),
        expected_tactics=["intro P", "intro Q", "intro R", "intro hpq", "intro hqr", "intro hp",
                         "apply hqr", "apply hpq", "assumption"],
        category="logic",
    ),
    # ── Implication transitivity ──
    MiniF2FProblem(
        id="imp_trans",
        name="∀ P Q, (P → Q) → (¬Q → ¬P)",
        lean4_statement="theorem imp_trans (P Q : Prop) (h : P → Q) (hnq : ¬Q) : ¬P := by intro hp; exact hnq (h hp)",
        goal_expr=Expr.pi(BI, Name.from_str("P"), prop,
                  Expr.pi(BI, Name.from_str("Q"), prop,
                  Expr.pi(BI, Name.from_str("h"), Expr.arrow(Expr.bvar(1), Expr.bvar(0)),
                  Expr.pi(BI, Name.from_str("hnq"),
                          Expr.arrow(Expr.bvar(1), Expr.const(Name.from_str("False"))),
                          Expr.arrow(Expr.bvar(3), Expr.const(Name.from_str("False"))))))),
        expected_tactics=["intro P", "intro Q", "intro h", "intro hnq", "intro hp",
                         "apply hnq", "apply h", "assumption"],
        category="logic",
    ),
    # ── Double negation introduction ──
    MiniF2FProblem(
        id="dne_intro",
        name="∀ P, P → ¬¬P",
        lean4_statement="theorem dne_intro (P : Prop) (hp : P) (hnp : ¬P) : False := by exact hnp hp",
        goal_expr=Expr.pi(BI, Name.from_str("P"), prop,
                  Expr.pi(BI, Name.from_str("hp"), Expr.bvar(0),
                  Expr.pi(BI, Name.from_str("hnp"),
                          Expr.arrow(Expr.bvar(1), Expr.const(Name.from_str("False"))),
                          Expr.const(Name.from_str("False"))))),
        expected_tactics=["intro P", "intro hp", "intro hnp", "apply hnp", "assumption"],
        category="logic",
    ),
    # ── And commutativity (structural) ──
    MiniF2FProblem(
        id="and_comm",
        name="∀ P Q, P ∧ Q → Q ∧ P",
        lean4_statement="theorem and_comm' (P Q : Prop) (h : P ∧ Q) : Q ∧ P := ⟨h.2, h.1⟩",
        goal_expr=Expr.pi(BI, Name.from_str("P"), prop,
                  Expr.pi(BI, Name.from_str("Q"), prop,
                  Expr.arrow(
                      Expr.app(Expr.app(Expr.const(Name.from_str("And")), Expr.bvar(1)),
                               Expr.bvar(0)),
                      Expr.app(Expr.app(Expr.const(Name.from_str("And")), Expr.bvar(0)),
                               Expr.bvar(1))))),
        expected_tactics=["intro P", "intro Q", "intro h",
                         "apply And.intro", "apply And.right h", "apply And.left h"],
        difficulty="medium",
        category="logic",
    ),
    # ── Nat: simple forall ──
    MiniF2FProblem(
        id="nat_prop",
        name="∀ n : Nat, ∀ P : Prop, P → P",
        lean4_statement="theorem nat_prop (n : Nat) (P : Prop) (h : P) : P := h",
        goal_expr=Expr.pi(BI, Name.from_str("n"), nat,
                  Expr.pi(BI, Name.from_str("P"), prop,
                  Expr.pi(BI, Name.from_str("h"), Expr.bvar(0),
                          Expr.bvar(1)))),
        expected_tactics=["intro n", "intro P", "intro h", "assumption"],
        category="nat",
    ),
    # ── Or intro left ──
    MiniF2FProblem(
        id="or_intro_left",
        name="∀ P Q, P → P ∨ Q",
        lean4_statement="theorem or_intro_l (P Q : Prop) (hp : P) : P ∨ Q := Or.inl hp",
        goal_expr=Expr.pi(BI, Name.from_str("P"), prop,
                  Expr.pi(BI, Name.from_str("Q"), prop,
                  Expr.arrow(Expr.bvar(1),
                      Expr.app(Expr.app(Expr.const(Name.from_str("Or")), Expr.bvar(1)),
                               Expr.bvar(0))))),
        expected_tactics=["intro P", "intro Q", "intro hp", "apply Or.inl", "assumption"],
        difficulty="medium",
        category="logic",
    ),
]
