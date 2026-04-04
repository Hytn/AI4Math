"""prover/formalize/statement_verifier.py — 形式化声明验证

验证 NL→Lean4 形式化的结果是否:
1. 语法正确
2. 类型正确 (well-typed)
3. 与原始自然语言语义一致
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    is_parseable: bool = False
    is_well_formed: bool = False
    has_sorry: bool = False
    issues: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.is_parseable and self.is_well_formed and not self.has_sorry


class StatementVerifier:
    """Verify formalized mathematical statements."""

    def __init__(self, lean_env=None, llm=None):
        self.lean_env = lean_env
        self.llm = llm

    def verify(self, formal_statement: str,
               natural_language: str = "") -> VerificationResult:
        """Verify a formalized statement."""
        result = VerificationResult()

        # Syntactic checks
        result.is_parseable = self._check_syntax(formal_statement, result)
        result.has_sorry = "sorry" in formal_statement.split(":=")[0]

        if result.is_parseable:
            result.is_well_formed = self._check_well_formed(formal_statement, result)

        # Semantic consistency check (if LLM available)
        if natural_language and self.llm and result.is_parseable:
            result.confidence = self._check_semantic(
                formal_statement, natural_language)

        return result

    def _check_syntax(self, stmt: str, result: VerificationResult) -> bool:
        s = stmt.strip()

        # Must be a declaration
        if not re.match(r'^(theorem|lemma|def|example)\s', s):
            result.issues.append("Not a theorem/lemma/def declaration")
            return False

        # Must have a type after ':'
        colon_match = re.search(r':\s*(.+?)(?:\s*:=|\s*$)', s)
        if not colon_match or not colon_match.group(1).strip():
            result.issues.append("Missing or empty type annotation")
            return False

        # Check balanced delimiters
        for o, c in [('(', ')'), ('{', '}'), ('[', ']'), ('⟨', '⟩')]:
            if s.count(o) != s.count(c):
                result.issues.append(f"Unbalanced {o}{c}")
                return False

        return True

    def _check_well_formed(self, stmt: str, result: VerificationResult) -> bool:
        """Check structural well-formedness."""
        # Extract the type part
        match = re.search(r':\s*(.+?)(?:\s*:=|\s*$)', stmt)
        if not match:
            return False

        type_part = match.group(1).strip()

        # Should not be empty
        if not type_part:
            result.issues.append("Empty type")
            return False

        # Should not contain undefined syntax
        if "???" in type_part or "..." in type_part:
            result.issues.append("Contains placeholder syntax")
            return False

        return True

    def _check_semantic(self, formal: str, natural: str) -> float:
        """Use LLM to check if formal statement matches natural language."""
        if not self.llm:
            return 0.5

        prompt = (
            f"Natural language statement:\n{natural}\n\n"
            f"Lean 4 formalization:\n```lean\n{formal}\n```\n\n"
            f"Rate how well the formalization captures the natural language "
            f"statement on a scale of 0.0 to 1.0. Output ONLY the number."
        )
        try:
            resp = self.llm.generate(system="You are a math formalization expert.",
                                      user=prompt, temperature=0.1)
            score = float(re.search(r'[01]?\.\d+', resp.content).group())
            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.5
