"""prover/codegen/code_formatter.py — Lean 代码格式化"""
from __future__ import annotations
import re

def format_lean(code: str) -> str:
    lines = code.split("\n")
    formatted = []
    for line in lines:
        line = line.rstrip()
        formatted.append(line)
    return "\n".join(formatted)
