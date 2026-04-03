"""prover/codegen/import_resolver.py — 自动推断 import"""
from __future__ import annotations

def resolve_imports(lean_code: str) -> str:
    return "import Mathlib\n"  # Default: full Mathlib import

def assemble_lean_file(theorem: str, proof: str, preamble: str = "",
                       extra_imports: list[str] = None) -> str:
    parts = [resolve_imports(proof)]
    if extra_imports: parts.extend(extra_imports)
    parts.append("")
    if preamble: parts.append(preamble); parts.append("")
    proof_stripped = proof.strip()
    if not proof_stripped.startswith(":=") and not proof_stripped.startswith("by"):
        proof_stripped = f":= by\n{proof_stripped}" if not proof_stripped.startswith("by") else f":= {proof_stripped}"
    parts.append(f"{theorem.rstrip()} {proof_stripped}")
    return "\n".join(parts)
