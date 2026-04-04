"""prover/codegen/import_resolver.py — 自动推断 import

Analyzes the proof code to determine which Mathlib modules are needed.
Falls back to 'import Mathlib' if specific resolution fails.
"""
from __future__ import annotations
import re

# Map common Lean4/Mathlib identifiers to their import modules.
# This is a best-effort mapping; 'import Mathlib' always works as fallback.
_TACTIC_IMPORTS = {
    "ring": "Mathlib.Tactic.Ring",
    "linarith": "Mathlib.Tactic.Linarith",
    "nlinarith": "Mathlib.Tactic.Linarith",
    "omega": "Mathlib.Tactic.Omega",
    "norm_num": "Mathlib.Tactic.NormNum",
    "positivity": "Mathlib.Tactic.Positivity",
    "polyrith": "Mathlib.Tactic.Polyrith",
    "field_simp": "Mathlib.Tactic.FieldSimp",
    "push_cast": "Mathlib.Tactic.NormCast",
    "gcongr": "Mathlib.Tactic.GCongr",
    "aesop": "Mathlib.Tactic.AesopCat",
    "exact?": "Mathlib.Tactic.LibrarySearch",
    "apply?": "Mathlib.Tactic.LibrarySearch",
}

_TYPE_IMPORTS = {
    "Real": "Mathlib.Data.Real.Basic",
    "Complex": "Mathlib.Data.Complex.Basic",
    "Finset": "Mathlib.Data.Finset.Basic",
    "Polynomial": "Mathlib.Data.Polynomial.Basic",
    "Matrix": "Mathlib.Data.Matrix.Basic",
    "MeasureTheory": "Mathlib.MeasureTheory.Measure.MeasureSpace",
    "Metric": "Mathlib.Topology.MetricSpace.Basic",
}


def resolve_imports(lean_code: str, use_full_mathlib: bool = True) -> str:
    """Determine needed imports for the given Lean4 code.

    Args:
        lean_code: The proof code to analyze.
        use_full_mathlib: If True (default), always use 'import Mathlib'
            for maximum compatibility. Set False to attempt minimal imports
            (faster compilation but may miss dependencies).
    """
    if use_full_mathlib:
        return "import Mathlib\n"

    needed = set()

    # Check for tactic usage
    for tactic, module in _TACTIC_IMPORTS.items():
        if re.search(rf'\b{re.escape(tactic)}\b', lean_code):
            needed.add(module)

    # Check for type usage
    for type_name, module in _TYPE_IMPORTS.items():
        if type_name in lean_code:
            needed.add(module)

    if not needed:
        # Fallback: if nothing specific detected, use full Mathlib
        return "import Mathlib\n"

    return "\n".join(f"import {m}" for m in sorted(needed)) + "\n"


def assemble_lean_file(theorem: str, proof: str, preamble: str = "",
                       extra_imports: list[str] = None) -> str:
    """Assemble a complete Lean4 file from components."""
    parts = [resolve_imports(proof)]
    if extra_imports:
        parts.extend(extra_imports)
    parts.append("")
    if preamble:
        parts.append(preamble)
        parts.append("")

    proof_stripped = proof.strip()
    if not proof_stripped.startswith(":=") and not proof_stripped.startswith("by"):
        if proof_stripped.startswith("by"):
            proof_stripped = f":= {proof_stripped}"
        else:
            proof_stripped = f":= by\n{proof_stripped}"

    parts.append(f"{theorem.rstrip()} {proof_stripped}")
    return "\n".join(parts)
