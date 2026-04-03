"""prover/repair/patch_applier.py — 应用修复 patch"""
from __future__ import annotations

def apply_patch(original: str, line_num: int, replacement: str) -> str:
    lines = original.split("\n")
    if 0 < line_num <= len(lines): lines[line_num - 1] = replacement
    return "\n".join(lines)

def replace_sorry(code: str, sorry_line: int, replacement: str) -> str:
    return apply_patch(code, sorry_line, replacement)
