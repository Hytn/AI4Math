"""prover/repair/error_diagnostor.py — 错误根因分析"""
from __future__ import annotations
from prover.models import LeanError, ErrorCategory

REPAIR_HINTS = {
    ErrorCategory.TYPE_MISMATCH: (
        "Type mismatch detected. Possible fixes:\n"
        "  1. Check if implicit arguments need explicit annotation\n"
        "  2. Try exact_mod_cast, push_cast, or norm_cast for numeric types\n"
        "  3. Use `show <expected_type>` before the problematic tactic\n"
        "  4. Check if you need `↑` (coercion) or `.toXxx` conversion"),
    ErrorCategory.UNKNOWN_IDENTIFIER: (
        "Unknown identifier. Possible fixes:\n"
        "  1. Check Lean3→Lean4 naming (nat→Nat, list→List, etc.)\n"
        "  2. Use `exact?` or `apply?` to find the correct lemma name\n"
        "  3. Check if the import is missing\n"
        "  4. Check if the namespace is opened (`open Nat` etc.)"),
    ErrorCategory.TACTIC_FAILED: (
        "Tactic failed on current goal. Possible fixes:\n"
        "  1. Break into smaller `have` steps\n"
        "  2. Try a different tactic (simp→ring, linarith→omega, etc.)\n"
        "  3. Use `conv` to target a specific subexpression\n"
        "  4. Add intermediate `rw` steps before the failing tactic"),
    ErrorCategory.SYNTAX_ERROR: (
        "Syntax error. Check:\n"
        "  1. Matching brackets/parentheses\n"
        "  2. Indentation (Lean4 is indentation-sensitive)\n"
        "  3. Missing commas between tactic arguments\n"
        "  4. `by` keyword before tactic block"),
    ErrorCategory.TIMEOUT: (
        "Elaborator timed out. Possible fixes:\n"
        "  1. Add explicit type annotations to reduce search space\n"
        "  2. Use `simp only [...]` instead of bare `simp`\n"
        "  3. Break complex expressions into named `have` steps\n"
        "  4. Increase `set_option maxHeartbeats` if appropriate"),
    ErrorCategory.ELABORATION_ERROR: (
        "Elaboration failed. Possible fixes:\n"
        "  1. Provide explicit universe levels\n"
        "  2. Add type ascriptions with `(expr : Type)`\n"
        "  3. Check if typeclass instances are missing"),
    ErrorCategory.IMPORT_ERROR: (
        "Import failed. Check:\n"
        "  1. Module name spelling\n"
        "  2. Mathlib availability (run `lake build`)\n"
        "  3. Use `import Mathlib` for full access"),
    ErrorCategory.OTHER: "Review the raw error message for details.",
}


def diagnose(errors: list[LeanError]) -> str:
    """Produce a detailed diagnostic report for LLM consumption."""
    if not errors:
        return "No errors."

    parts = [f"{len(errors)} error(s):\n"]
    for i, e in enumerate(errors, 1):
        loc = f" at line {e.line}" if e.line else ""
        parts.append(f"Error {i}{loc} [{e.category.value}]: {e.message[:200]}")

        # Include structured type info if available
        if e.expected_type and e.actual_type:
            parts.append(f"  Expected type: {e.expected_type[:150]}")
            parts.append(f"  Actual type:   {e.actual_type[:150]}")

        # Include suggestions from Lean4
        if e.suggestions:
            parts.append(f"  Lean4 suggestions: {', '.join(e.suggestions[:3])}")

        hint = REPAIR_HINTS.get(e.category, REPAIR_HINTS[ErrorCategory.OTHER])
        parts.append(f"  Repair hint: {hint}\n")

    # Add summary of error pattern
    cats = {}
    for e in errors:
        cats[e.category.value] = cats.get(e.category.value, 0) + 1
    dominant = max(cats, key=cats.get)
    parts.append(f"Dominant error pattern: {dominant} ({cats[dominant]} occurrences)")
    parts.append(f"Recommendation: Focus repair on {dominant} errors first.")

    return "\n".join(parts)
