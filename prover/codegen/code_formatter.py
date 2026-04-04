"""prover/codegen/code_formatter.py — Lean4 代码格式化与清理

格式化 LLM 生成的 Lean4 代码:
- 缩进修正
- 去除多余空行
- Unicode 符号规范化
- 基本语法修复
"""
from __future__ import annotations
import re


def format_lean_code(code: str, indent: int = 2) -> str:
    """Format and clean Lean4 proof code.

    Args:
        code: Raw Lean4 code (possibly from LLM output).
        indent: Number of spaces per indentation level.
    """
    code = _strip_markdown(code)
    code = _normalize_unicode(code)
    code = _fix_indentation(code, indent)
    code = _remove_trailing_whitespace(code)
    code = _collapse_blank_lines(code)
    return code.strip() + "\n"


def _strip_markdown(code: str) -> str:
    """Remove markdown code fences if present."""
    code = re.sub(r'^```\w*\s*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
    return code.strip()


def _normalize_unicode(code: str) -> str:
    """Normalize ASCII shorthands to their Lean4 Unicode equivalents."""
    replacements = [
        # ASCII → Unicode (order matters: longer patterns first)
        ("->", "→"),
        ("<-", "←"),
        ("=>", "⇒"),
        ("\\lam", "fun"),
        ("\\forall", "∀"),
        ("\\exists", "∃"),
        ("\\and", "∧"),
        ("\\or", "∨"),
        ("\\not", "¬"),
        ("\\ne", "≠"),
        ("\\le", "≤"),
        ("\\ge", "≥"),
        ("\\sub", "⊂"),
        ("\\sup", "⊃"),
        ("\\in", "∈"),
        ("\\notin", "∉"),
        ("\\times", "×"),
        ("\\to", "→"),
        ("\\langle", "⟨"),
        ("\\rangle", "⟩"),
        ("\\alpha", "α"),
        ("\\beta", "β"),
        ("\\gamma", "γ"),
        ("\\epsilon", "ε"),
        ("\\lambda", "λ"),
    ]
    for old, new in replacements:
        code = code.replace(old, new)
    return code


def _fix_indentation(code: str, indent: int) -> str:
    """Fix indentation based on Lean4 block structure."""
    lines = code.split("\n")
    result = []
    level = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue

        # Decrease indent for closing keywords
        if stripped.startswith(("end ", "end\n", "}")) or stripped == "end":
            level = max(0, level - 1)

        result.append(" " * (level * indent) + stripped)

        # Increase indent for opening keywords
        if re.match(r'\b(where|by|do)\s*$', stripped):
            level += 1
        elif stripped.endswith(":=") or stripped.endswith("by"):
            level += 1
        elif stripped.startswith(("namespace ", "section ", "where")):
            level += 1
        elif stripped.startswith(("{")) and not stripped.endswith("}"):
            level += 1

    return "\n".join(result)


def _remove_trailing_whitespace(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.split("\n"))


def _collapse_blank_lines(code: str) -> str:
    """Collapse 3+ consecutive blank lines into 2."""
    return re.sub(r'\n{3,}', '\n\n', code)


def extract_proof_body(full_code: str) -> str:
    """Extract just the proof body from a full theorem declaration.

    E.g., from 'theorem foo : T := by\\n  exact h' → 'exact h'
    """
    match = re.search(r':=\s*by\s*\n(.*)', full_code, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r':=\s*(.*)', full_code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return full_code.strip()


def wrap_proof(theorem_statement: str, proof_body: str) -> str:
    """Combine theorem statement and proof body into a complete declaration."""
    stmt = theorem_statement.rstrip()
    body = proof_body.strip()
    if body.startswith(":="):
        return f"{stmt} {body}"
    if body.startswith("by") or body.startswith("by\n"):
        return f"{stmt} := {body}"
    return f"{stmt} := by\n  {body}"
