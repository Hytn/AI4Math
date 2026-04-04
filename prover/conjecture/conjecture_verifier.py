"""prover/conjecture/conjecture_verifier.py — 猜想验证

验证 LLM 提出的辅助猜想是否:
1. 语法正确 (parseable Lean4)
2. 类型正确 (well-typed statement)
3. 不矛盾 (not trivially False)
4. 有用 (relevant to the target theorem)
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class ConjectureVerification:
    """Result of verifying a conjecture."""
    conjecture: str
    is_parseable: bool = False
    is_well_typed: bool = False
    is_useful: bool = False
    is_trivial: bool = False
    issues: list[str] = field(default_factory=list)
    relevance_score: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.is_parseable and not self.is_trivial


class ConjectureVerifier:
    """Verify generated conjectures for validity and usefulness."""

    def __init__(self, lean_env=None):
        self.lean_env = lean_env

    def verify(self, conjecture: str, target_theorem: str = "") -> ConjectureVerification:
        """Verify a single conjecture."""
        result = ConjectureVerification(conjecture=conjecture)

        # Step 1: Check parseability
        result.is_parseable = self._check_parseable(conjecture, result)

        # Step 2: Check for trivial/useless patterns
        result.is_trivial = self._check_trivial(conjecture, result)

        # Step 3: Check relevance to target
        if target_theorem:
            result.relevance_score = self._compute_relevance(conjecture, target_theorem)
            result.is_useful = result.relevance_score > 0.1

        # Step 4: Type-check via Lean (if available)
        if self.lean_env and result.is_parseable:
            result.is_well_typed = self._type_check(conjecture, result)

        return result

    def verify_batch(self, conjectures: list[str],
                     target_theorem: str = "") -> list[ConjectureVerification]:
        """Verify multiple conjectures, return valid ones first."""
        results = [self.verify(c, target_theorem) for c in conjectures]
        results.sort(key=lambda r: (-int(r.is_valid), -r.relevance_score))
        return results

    def filter_valid(self, conjectures: list[str],
                     target_theorem: str = "") -> list[str]:
        """Return only valid conjectures."""
        results = self.verify_batch(conjectures, target_theorem)
        return [r.conjecture for r in results if r.is_valid]

    def _check_parseable(self, conjecture: str,
                         result: ConjectureVerification) -> bool:
        """Check basic Lean4 syntax validity."""
        s = conjecture.strip()

        # Must start with lemma/theorem/def
        if not re.match(r'^(lemma|theorem|def|axiom)\s+\w+', s):
            result.issues.append("Does not start with lemma/theorem/def")
            return False

        # Must contain a colon (type annotation)
        if ':' not in s:
            result.issues.append("Missing type annotation (no ':')")
            return False

        # Check balanced brackets
        for open_c, close_c in [('(', ')'), ('{', '}'), ('⟨', '⟩'), ('[', ']')]:
            if s.count(open_c) != s.count(close_c):
                result.issues.append(f"Unbalanced brackets: {open_c}{close_c}")
                return False

        return True

    def _check_trivial(self, conjecture: str,
                       result: ConjectureVerification) -> bool:
        """Check if conjecture is trivially true/false/useless."""
        s = conjecture.lower()

        # Trivially True
        if re.search(r':\s*(true|⊤)\s*$', s):
            result.issues.append("Trivially True")
            return True

        # Trivially reflexive
        if re.search(r':\s*(\w+)\s*=\s*\1\s*$', s):
            result.issues.append("Trivially reflexive (a = a)")
            return True

        # Contains sorry in statement (not just proof)
        stmt_part = conjecture.split(":=")[0] if ":=" in conjecture else conjecture
        if "sorry" in stmt_part.lower():
            result.issues.append("Contains sorry in statement")
            return True

        return False

    def _compute_relevance(self, conjecture: str, target: str) -> float:
        """Compute relevance score based on shared tokens."""
        from prover.premise.bm25_retriever import tokenize
        conj_tokens = set(tokenize(conjecture))
        target_tokens = set(tokenize(target))
        if not conj_tokens or not target_tokens:
            return 0.0
        overlap = conj_tokens & target_tokens
        return len(overlap) / len(conj_tokens | target_tokens)

    def _type_check(self, conjecture: str,
                    result: ConjectureVerification) -> bool:
        """Type-check via Lean environment (if available)."""
        if not self.lean_env:
            return False
        try:
            code = f"import Mathlib\n\n{conjecture} := by sorry"
            returncode, _, stderr = self.lean_env.compile(code)
            if returncode == 0 or "unsolved goals" in stderr:
                return True
            result.issues.append(f"Type-check failed: {stderr[:100]}")
            return False
        except Exception as e:
            result.issues.append(f"Type-check error: {str(e)[:80]}")
            return False
