"""common/response_parser.py — LLM response extraction utilities.


v12 时还在 (26 行), 但 0 主路径调用方。
"""
from __future__ import annotations
import re

_LEAN_START_RE = re.compile(
    r"^\s*(?:"
    r"import\s+|open\s+|namespace\s+|section\b|"
    r"theorem\s+|lemma\s+|example\b|def\s+|"
    r":=\s*by\b|by\b|calc\b|exact\b|refine\b|"
    r"have\b|show\b|apply\b|constructor\b|cases\b|rcases\b|"
    r"intro\b|intros\b|simp\b|rw\b|rfl\b|ring\b|"
    r"norm_num\b|omega\b|linarith\b|nlinarith\b"
    r")",
    re.MULTILINE,
)

def looks_like_lean_code(text: str) -> bool:
    """Return True when unfenced text plausibly starts as Lean code."""
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(_LEAN_START_RE.search(text))

def extract_lean_code(response: str) -> str:
    """Extract a Lean code block from an LLM response.

    Tries fenced ```lean ... ``` first, then any fenced block. If neither
    matches, returns only text that plausibly starts as Lean code. This
    avoids treating explanatory prose as a proof.
    """
    for pattern in [r"```lean\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()
    if not looks_like_lean_code(response):
        return ""
    lines = response.strip().split("\n")
    return "\n".join(
        l for l in lines
        if not l.startswith("**") and not l.startswith("##")).strip()
