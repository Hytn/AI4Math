"""prover/sketch/templates.py — 证明模板库

为常见定理模式提供预设的证明骨架模板。
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ProofTemplate:
    """A proof template for a particular theorem pattern."""
    name: str
    pattern: str          # regex-like description of when to apply
    description: str
    skeleton: str         # Lean4 proof skeleton with {{placeholders}}
    applicable_shapes: list[str]  # goal shapes this applies to


TEMPLATES = [
    ProofTemplate(
        name="induction_nat",
        pattern="∀ (n : Nat), P(n)",
        description="Natural number induction: base case + step",
        skeleton=(
            "by\n"
            "  induction n with\n"
            "  | zero => {{base_case}}\n"
            "  | succ n ih => {{inductive_step}}"
        ),
        applicable_shapes=["nat_expr", "equality"],
    ),
    ProofTemplate(
        name="direct_implication",
        pattern="P → Q",
        description="Direct proof: assume P, show Q",
        skeleton=(
            "by\n"
            "  intro h\n"
            "  {{proof_body}}"
        ),
        applicable_shapes=["implication", "forall"],
    ),
    ProofTemplate(
        name="conjunction_split",
        pattern="P ∧ Q",
        description="Prove both sides of a conjunction",
        skeleton=(
            "by\n"
            "  constructor\n"
            "  · {{left_proof}}\n"
            "  · {{right_proof}}"
        ),
        applicable_shapes=["conjunction"],
    ),
    ProofTemplate(
        name="iff_split",
        pattern="P ↔ Q",
        description="Prove both directions of an iff",
        skeleton=(
            "by\n"
            "  constructor\n"
            "  · intro h\n"
            "    {{forward_proof}}\n"
            "  · intro h\n"
            "    {{backward_proof}}"
        ),
        applicable_shapes=["iff"],
    ),
    ProofTemplate(
        name="contradiction",
        pattern="¬P or False goal",
        description="Proof by contradiction",
        skeleton=(
            "by\n"
            "  intro h\n"
            "  {{derive_contradiction}}\n"
            "  contradiction"
        ),
        applicable_shapes=["negation"],
    ),
    ProofTemplate(
        name="cases_analysis",
        pattern="match or by cases",
        description="Case analysis on a hypothesis",
        skeleton=(
            "by\n"
            "  cases {{hypothesis}} with\n"
            "  | {{case1}} => {{proof1}}\n"
            "  | {{case2}} => {{proof2}}"
        ),
        applicable_shapes=["disjunction"],
    ),
    ProofTemplate(
        name="exists_witness",
        pattern="∃ x, P(x)",
        description="Existential: provide witness",
        skeleton=(
            "by\n"
            "  use {{witness}}\n"
            "  {{verify_property}}"
        ),
        applicable_shapes=["exists"],
    ),
    ProofTemplate(
        name="algebraic_identity",
        pattern="a OP b = c OP d",
        description="Algebraic equality via ring/omega",
        skeleton="by ring",
        applicable_shapes=["equality"],
    ),
    ProofTemplate(
        name="inequality_chain",
        pattern="a ≤ b or a < b",
        description="Inequality via linarith/omega",
        skeleton="by linarith",
        applicable_shapes=["le", "lt"],
    ),
]


def find_templates(goal_shape: str,
                   max_results: int = 3) -> list[ProofTemplate]:
    """Find applicable templates for a given goal shape."""
    applicable = [t for t in TEMPLATES if goal_shape in t.applicable_shapes]
    if not applicable:
        # Return generic templates
        applicable = [t for t in TEMPLATES if "equality" in t.applicable_shapes
                      or "implication" in t.applicable_shapes]
    return applicable[:max_results]


def fill_template(template: ProofTemplate,
                  fillings: dict[str, str]) -> str:
    """Fill in template placeholders with actual tactic code."""
    result = template.skeleton
    for key, value in fillings.items():
        placeholder = "{{" + key + "}}"
        result = result.replace(placeholder, value)
    # Replace unfilled placeholders with sorry
    import re
    result = re.sub(r'\{\{[^}]+\}\}', 'sorry', result)
    return result
