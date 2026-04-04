"""prover/decompose/composition.py — 子证明组合

将已证明的子目标组合成完整证明。
"""
from __future__ import annotations
import re
from prover.decompose.goal_decomposer import SubGoal
from prover.codegen.code_formatter import format_lean_code


def compose_proof(theorem: str, subgoals: list[SubGoal],
                  composition_template: str = "") -> str:
    """Compose sub-proofs into a complete proof.

    Args:
        theorem: The main theorem statement.
        subgoals: List of solved subgoals.
        composition_template: Optional template for how to combine them.

    Returns:
        Complete Lean4 proof code.
    """
    if not subgoals:
        return f"{theorem} := by sorry"

    # Check if all subgoals are proved
    unsolved = [g for g in subgoals if not g.proved]
    solved = [g for g in subgoals if g.proved]

    parts = []

    # Emit each solved subgoal as a 'have' step
    for g in solved:
        proof_body = g.proof.strip()
        if proof_body.startswith(":="):
            proof_body = proof_body[2:].strip()
        if not proof_body.startswith("by"):
            proof_body = f"by {proof_body}"
        parts.append(f"  have {g.name} : {_extract_type(g.statement)} := {proof_body}")

    # If there are unsolved goals, insert sorry
    for g in unsolved:
        parts.append(f"  have {g.name} : {_extract_type(g.statement)} := by sorry")

    # Final step: use composition template or default
    if composition_template:
        parts.append(f"  {composition_template}")
    else:
        # Default: try to close with the available lemmas
        have_names = [g.name for g in subgoals]
        if len(have_names) == 1:
            parts.append(f"  exact {have_names[0]}")
        else:
            parts.append(f"  exact ⟨{', '.join(have_names)}⟩")

    body = "\n".join(parts)
    return f"{theorem} := by\n{body}"


def _extract_type(statement: str) -> str:
    """Extract the type from a lemma statement like 'lemma foo : TYPE'."""
    match = re.search(r':\s*(.+?)(?:\s*:=|\s*$)', statement)
    if match:
        return match.group(1).strip()
    return statement.strip()


def validate_composition(composed_proof: str, subgoals: list[SubGoal]) -> dict:
    """Check that composition references all subgoals and has valid structure.

    Returns dict with 'valid' bool and 'issues' list.
    """
    issues = []

    # Check all subgoal names appear
    for g in subgoals:
        if g.name not in composed_proof:
            issues.append(f"Subgoal '{g.name}' not referenced in composed proof")

    # Check for sorry
    if "sorry" in composed_proof:
        unsolved = [g.name for g in subgoals if not g.proved]
        if unsolved:
            issues.append(f"Contains sorry for unsolved subgoals: {unsolved}")
        else:
            issues.append("Contains sorry but all subgoals are proved — composition issue")

    # Check basic structure
    if ":= by" not in composed_proof and ":= " not in composed_proof:
        issues.append("Missing proof body")

    return {"valid": len(issues) == 0, "issues": issues}
