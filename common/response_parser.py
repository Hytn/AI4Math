"""common/response_parser.py — LLM response extraction utilities.


v12 时还在 (26 行), 但 0 主路径调用方。
"""
from __future__ import annotations
import re

def extract_lean_code(response: str) -> str:
    """Extract a Lean code block from an LLM response.

    Tries fenced ```lean ... ``` first, then any fenced block. If neither
    matches, returns the whole response with markdown headers stripped.
    """
    for pattern in [r"```lean\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()
    lines = response.strip().split("\n")
    return "\n".join(
        l for l in lines
        if not l.startswith("**") and not l.startswith("##")).strip()
