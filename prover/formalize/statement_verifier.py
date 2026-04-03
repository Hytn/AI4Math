"""prover/formalize/statement_verifier.py — 验证形式化声明合法性"""
from __future__ import annotations

class StatementVerifier:
    def __init__(self, lean_env): self.lean = lean_env
    def verify(self, statement: str) -> bool:
        code = f"import Mathlib\n\n{statement} := by sorry"
        rc, _, _ = self.lean.compile(code)
        return rc == 0
