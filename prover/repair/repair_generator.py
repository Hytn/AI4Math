"""prover/repair/repair_generator.py — 自动修复方案生成

基于错误诊断和修复策略生成修正后的证明。
"""
from __future__ import annotations
from common.roles import AgentRole, ROLE_PROMPTS
from common.response_parser import extract_lean_code
from prover.models import LeanError
from prover.repair.repair_strategies import select_strategies, build_repair_prompt


class RepairGenerator:
    """Generate repaired proofs based on error analysis."""

    def __init__(self, llm):
        self.llm = llm

    def generate_repair(self, theorem: str, failed_proof: str,
                        errors: list[LeanError] = None,
                        error_analysis: str = "",
                        max_repairs: int = 3,
                        temperature: float = 0.5) -> list[str]:
        """Generate one or more repair candidates.

        Args:
            theorem: The theorem statement.
            failed_proof: The proof that failed.
            errors: Structured error list from Lean.
            error_analysis: Free-text error analysis.
            max_repairs: Number of repair candidates to generate.
            temperature: LLM temperature.

        Returns:
            List of repaired proof strings.
        """
        errors = errors or []
        strategies = select_strategies(errors)
        strategy_prompt = build_repair_prompt(errors, strategies)

        prompt = (
            f"Theorem:\n```lean\n{theorem}\n```\n\n"
            f"Failed proof:\n```lean\n{failed_proof}\n```\n\n"
            f"{strategy_prompt}\n\n"
        )
        if error_analysis:
            prompt += f"Additional analysis:\n{error_analysis}\n\n"

        prompt += f"Generate {max_repairs} corrected proof(s), each in a separate ```lean block."

        resp = self.llm.generate(
            system=ROLE_PROMPTS[AgentRole.REPAIR_AGENT],
            user=prompt, temperature=temperature)

        # Extract all lean code blocks
        import re
        blocks = re.findall(r'```lean\s*\n(.*?)```', resp.content, re.DOTALL)
        repairs = [b.strip() for b in blocks if b.strip()]

        # If only one block, still return it
        if not repairs:
            single = extract_lean_code(resp.content)
            if single.strip():
                repairs = [single]

        return repairs[:max_repairs]

    def quick_fix(self, theorem: str, failed_proof: str,
                  errors: list[LeanError]) -> str:
        """Apply rule-based quick fixes without LLM.

        Returns modified proof or original if no fix found.
        """
        proof = failed_proof
        for e in errors:
            if e.category.value == "syntax_error":
                proof = _fix_syntax(proof, e)
            elif e.category.value == "unknown_identifier":
                proof = _fix_identifier(proof, e)
        return proof


def _fix_syntax(proof: str, error: LeanError) -> str:
    """Apply rule-based syntax fixes."""
    import re

    # Fix unmatched round brackets
    open_paren = proof.count("(") - proof.count(")")
    if open_paren > 0:
        proof += ")" * open_paren
    elif open_paren < 0:
        # Remove excess closing parens from end
        for _ in range(-open_paren):
            idx = proof.rfind(")")
            if idx >= 0:
                proof = proof[:idx] + proof[idx+1:]

    # Fix unmatched angle brackets
    open_angle = proof.count("⟨") - proof.count("⟩")
    if open_angle > 0:
        proof += "⟩" * open_angle

    # Common Lean4 syntax fixes
    # Remove stray semicolons at end of tactic blocks
    proof = re.sub(r';\s*$', '', proof, flags=re.MULTILINE)

    # Fix `by\n\n` (empty tactic block) — insert sorry as placeholder
    proof = re.sub(r'by\s*\n\s*\n', 'by\n  sorry\n', proof)

    # Fix missing `by` after `:=`
    proof = re.sub(r':=\s*\n\s+(intro|apply|exact|simp|ring|omega|rfl|cases|induction)',
                   r':= by\n  \1', proof)

    # Fix Lean3-style `begin...end` → `by`
    proof = re.sub(r'\bbegin\b', 'by', proof)
    proof = re.sub(r'\bend\b\s*$', '', proof, flags=re.MULTILINE)

    # Fix `#check` and `#eval` in proofs
    proof = re.sub(r'^\s*#(?:check|eval|print)\s+.*$', '', proof, flags=re.MULTILINE)

    return proof


def _fix_identifier(proof: str, error: LeanError) -> str:
    """Try to fix unknown identifiers with Lean3 → Lean4 common renames."""
    common_fixes = {
        # Nat
        "nat.add_comm": "Nat.add_comm",
        "nat.mul_comm": "Nat.mul_comm",
        "nat.add_assoc": "Nat.add_assoc",
        "nat.mul_assoc": "Nat.mul_assoc",
        "nat.zero_add": "Nat.zero_add",
        "nat.add_zero": "Nat.add_zero",
        "nat.succ": "Nat.succ",
        "nat.zero": "Nat.zero",
        "nat.le_refl": "Nat.le_refl",
        "nat.lt_iff_add_one_le": "Nat.lt_iff_add_one_le",
        "nat.sub_self": "Nat.sub_self",
        "nat.rec_on": "Nat.rec",
        # Int
        "int.add_comm": "Int.add_comm",
        "int.mul_comm": "Int.mul_comm",
        "int.coe_nat": "Int.ofNat",
        # List
        "list.nil": "List.nil",
        "list.cons": "List.cons",
        "list.map": "List.map",
        "list.length": "List.length",
        "list.append": "List.append",
        "list.reverse": "List.reverse",
        # Logic
        "and.intro": "And.intro",
        "and.left": "And.left",
        "and.right": "And.right",
        "and.elim": "And.elim",
        "or.inl": "Or.inl",
        "or.inr": "Or.inr",
        "or.elim": "Or.elim",
        "not.intro": "Not.intro",
        "iff.intro": "Iff.intro",
        "iff.mp": "Iff.mp",
        "iff.mpr": "Iff.mpr",
        "exists.intro": "Exists.intro",
        "classical.em": "Classical.em",
        "classical.by_contradiction": "Classical.byContradiction",
        # Eq
        "eq.refl": "Eq.refl",
        "eq.symm": "Eq.symm",
        "eq.trans": "Eq.trans",
        "eq.subst": "Eq.subst",
        "eq.mp": "Eq.mp",
        "eq.mpr": "Eq.mpr",
        # Finset / Set
        "finset.sum": "Finset.sum",
        "finset.range": "Finset.range",
        "finset.card": "Finset.card",
        "set.mem_union": "Set.mem_union",
        "set.mem_inter": "Set.mem_inter",
        # Tactic renames
        "unfold": "unfold",
        "dsimp": "dsimp",
        "squeeze_simp": "simp?",
        "library_search": "exact?",
        "suggest": "exact?",
        "norm_cast": "push_cast",
        # Other common renames
        "has_add.add": "HAdd.hAdd",
        "has_mul.mul": "HMul.hMul",
        "decidable.em": "Classical.em",
        "function.injective": "Function.Injective",
        "function.surjective": "Function.Surjective",
    }
    for old, new in common_fixes.items():
        if old in error.message.lower() or old in proof.lower():
            # Use word-boundary-aware replacement to avoid partial matches
            import re
            proof = re.sub(r'\b' + re.escape(old) + r'\b', new, proof)
    return proof
