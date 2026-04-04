"""prover/formalize/autoformalize.py — NL → Lean 4 形式化

将自然语言数学命题转换为 Lean4 形式化声明。
"""
from __future__ import annotations
from agent.brain.roles import AgentRole, ROLE_PROMPTS
from agent.brain.response_parser import extract_lean_code
from prover.formalize.statement_verifier import StatementVerifier


class AutoFormalizer:
    """Translate natural language math statements to Lean 4."""

    def __init__(self, llm, lean_env=None):
        self.llm = llm
        self.verifier = StatementVerifier(lean_env, llm)

    def formalize(self, nl_statement: str, hints: list[str] = None,
                  verify: bool = True, max_retries: int = 2) -> str:
        """Formalize a natural language statement.

        Args:
            nl_statement: The natural language math statement.
            hints: Optional Lean4 type/notation hints.
            verify: Whether to verify the output.
            max_retries: Number of retries if verification fails.

        Returns:
            Lean4 theorem declaration string.
        """
        hint_section = ""
        if hints:
            hint_section = (
                "\n\nHints (useful types/notations):\n" +
                "\n".join(f"- {h}" for h in hints)
            )

        prompt = (
            f"Formalize this mathematical statement in Lean 4 (with Mathlib):\n\n"
            f"{nl_statement}\n"
            f"{hint_section}\n\n"
            f"Output a single theorem declaration (no proof, end with := by sorry).\n"
            f"Use standard Mathlib types and notation."
        )

        for attempt in range(max_retries + 1):
            resp = self.llm.generate(
                system=ROLE_PROMPTS[AgentRole.FORMALIZATION_EXPERT],
                user=prompt, temperature=0.3 + attempt * 0.2)

            result = extract_lean_code(resp.content)

            if not result.strip():
                continue

            # Ensure it ends with := by sorry if no proof
            if ":=" not in result:
                result = f"{result.rstrip()} := by sorry"

            if not verify:
                return result

            verification = self.verifier.verify(result, nl_statement)
            if verification.is_valid:
                return result

            # Add error feedback for retry
            issues = "; ".join(verification.issues)
            prompt += f"\n\nPrevious attempt had issues: {issues}. Fix them."

        return result  # return last attempt even if not perfect

    def formalize_batch(self, statements: list[str],
                         verify: bool = True) -> list[dict]:
        """Formalize multiple statements."""
        results = []
        for stmt in statements:
            formal = self.formalize(stmt, verify=verify)
            verification = self.verifier.verify(formal, stmt)
            results.append({
                "natural_language": stmt,
                "formal": formal,
                "valid": verification.is_valid,
                "confidence": verification.confidence,
                "issues": verification.issues,
            })
        return results
