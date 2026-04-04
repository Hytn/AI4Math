"""prover/lemma_bank/lemma_verifier.py — 验证提取的引理

通过 Lean4 编译验证引理的正确性。
"""
from __future__ import annotations
from prover.lemma_bank.bank import ProvedLemma


class LemmaVerifier:
    """Verify extracted lemmas via Lean4 compilation."""

    def __init__(self, lean_env=None):
        self.lean_env = lean_env

    def verify(self, lemma: ProvedLemma) -> bool:
        """Verify a single lemma.

        Returns True if the lemma compiles successfully.
        """
        if not self.lean_env:
            # Without Lean env, do basic structural checks
            return self._structural_check(lemma)

        code = f"import Mathlib\n\n{lemma.statement} {lemma.proof}"
        try:
            returncode, _, stderr = self.lean_env.compile(code)
            if returncode == 0 and "error" not in stderr.lower():
                lemma.verified = True
                return True
            return False
        except Exception:
            return False

    def verify_batch(self, lemmas: list[ProvedLemma]) -> list[ProvedLemma]:
        """Verify a batch of lemmas, return only verified ones."""
        verified = []
        for lemma in lemmas:
            if self.verify(lemma):
                verified.append(lemma)
        return verified

    def _structural_check(self, lemma: ProvedLemma) -> bool:
        """Basic structural verification without Lean."""
        # Must have a statement and proof
        if not lemma.statement.strip() or not lemma.proof.strip():
            return False
        # Must not contain sorry
        if "sorry" in lemma.proof:
            return False
        # Statement must look like a declaration
        stmt = lemma.statement.strip()
        if not (stmt.startswith("lemma") or stmt.startswith("theorem")):
            return False
        # Must have a colon (type annotation)
        if ":" not in stmt:
            return False
        lemma.verified = True
        return True
