"""prover/repair/error_diagnostor.py — 错误根因分析"""
from __future__ import annotations
from prover.models import LeanError, ErrorCategory

REPAIR_HINTS = {
    ErrorCategory.TYPE_MISMATCH: "Check type alignment. May need conversion lemma or exact_mod_cast.",
    ErrorCategory.UNKNOWN_IDENTIFIER: "Wrong lemma name — check Mathlib naming. Try exact? or apply?.",
    ErrorCategory.TACTIC_FAILED: "Tactic failed. Break into smaller have steps or try different tactic.",
    ErrorCategory.SYNTAX_ERROR: "Fix syntax: missing commas, unmatched brackets, wrong indentation.",
    ErrorCategory.TIMEOUT: "Elaborator timed out. Simplify or provide more explicit arguments.",
    ErrorCategory.OTHER: "Review raw error message.",
}

def diagnose(errors: list[LeanError]) -> str:
    if not errors: return "No errors."
    parts = [f"{len(errors)} error(s):\n"]
    for i, e in enumerate(errors, 1):
        loc = f" at line {e.line}" if e.line else ""
        parts.append(f"Error {i}{loc} [{e.category.value}]: {e.message}")
        parts.append(f"  Hint: {REPAIR_HINTS.get(e.category, REPAIR_HINTS[ErrorCategory.OTHER])}\n")
    return "\n".join(parts)
