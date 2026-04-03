"""
core/error_analyzer.py — Lean 报错结构化分析

职责：把 Lean 编译器的原始错误转换为对 LLM 友好的修复指导。
不只是转发错误文本，而是提供"应该怎么修"的方向性建议。
"""

from __future__ import annotations

from core.models import LeanError, ErrorCategory


# ── 每种错误类型的修复建议模板 ──────────────────────────────────

_REPAIR_HINTS: dict[ErrorCategory, str] = {
    ErrorCategory.TYPE_MISMATCH: (
        "The proof term has a type mismatch. Check that the expression "
        "matches the expected type. You may need to apply a conversion lemma, "
        "use `exact_mod_cast`, or adjust the term to align types."
    ),
    ErrorCategory.UNKNOWN_IDENTIFIER: (
        "An identifier was not found. This often means: (1) wrong lemma name — "
        "check Mathlib naming conventions; (2) missing `open` or `import`; "
        "(3) the lemma was renamed or deprecated in this Mathlib version. "
        "Try searching for a similar name or use `exact?` / `apply?`."
    ),
    ErrorCategory.TACTIC_FAILED: (
        "A tactic failed to close the goal. Review the goal state carefully. "
        "If using `simp`, consider adding specific lemmas: `simp [lemma_name]`. "
        "If using `ring` or `linarith`, check that the goal is in the right form. "
        "You may need intermediate `have` steps to transform the goal."
    ),
    ErrorCategory.SYNTAX_ERROR: (
        "There is a syntax error in the Lean code. Check for: missing commas, "
        "unmatched parentheses/brackets, incorrect indentation, or invalid tokens. "
        "Make sure tactic blocks use proper `by` syntax."
    ),
    ErrorCategory.IMPORT_ERROR: (
        "An import was not found. Use `import Mathlib` for full access, or check "
        "that the specific module path is correct for this Mathlib version."
    ),
    ErrorCategory.ELABORATION_ERROR: (
        "Lean's elaborator failed. This can mean: (1) implicit arguments couldn't "
        "be inferred — try providing them explicitly; (2) a typeclass instance is "
        "missing — check that the right instances are in scope; (3) universe issues."
    ),
    ErrorCategory.TIMEOUT: (
        "The Lean elaborator or a tactic timed out. This usually means the search "
        "space is too large. Try: (1) breaking into smaller `have` steps; "
        "(2) providing more explicit arguments; (3) using `decide` cautiously."
    ),
    ErrorCategory.OTHER: (
        "An unrecognized error occurred. Review the raw error message below."
    ),
}


def analyze_errors(errors: list[LeanError]) -> str:
    """
    将一组 LeanError 转化为适合放入 LLM prompt 的修复指导文本。

    Returns:
        一段结构化的错误分析文本，可直接拼入下一轮 prompt。
    """
    if not errors:
        return "No errors detected."

    sections = []
    sections.append(f"The previous proof attempt produced {len(errors)} error(s):\n")

    for i, err in enumerate(errors, 1):
        loc = f" at line {err.line}" if err.line else ""
        sections.append(f"--- Error {i}{loc} [{err.category.value}] ---")
        sections.append(f"Message: {err.message}")
        sections.append(f"Repair hint: {_REPAIR_HINTS.get(err.category, _REPAIR_HINTS[ErrorCategory.OTHER])}")
        sections.append("")

    sections.append(
        "Based on these errors, generate a corrected proof. "
        "Focus on fixing the specific issues identified above."
    )

    return "\n".join(sections)


def summarize_error_history(
    error_history: list[tuple[str, list[LeanError]]],
    max_history: int = 3,
) -> str:
    """
    把最近 N 轮的 (proof, errors) 压缩为历史摘要，
    避免 prompt 随着重试次数线性膨胀。

    Args:
        error_history: [(generated_proof, [LeanError, ...]), ...]
        max_history:   最多保留几轮历史

    Returns:
        压缩后的历史描述文本
    """
    if not error_history:
        return ""

    recent = error_history[-max_history:]
    parts = ["## Previous Attempts (most recent last)\n"]

    for idx, (proof, errors) in enumerate(recent, 1):
        n = len(error_history) - len(recent) + idx
        # 只保留 proof 的前几行作为摘要
        proof_preview = "\n".join(proof.strip().split("\n")[:5])
        if len(proof.strip().split("\n")) > 5:
            proof_preview += "\n  ... (truncated)"

        error_summary = "; ".join(
            f"[{e.category.value}] {e.message[:80]}" for e in errors[:3]
        )
        if len(errors) > 3:
            error_summary += f"; ... and {len(errors) - 3} more error(s)"

        parts.append(f"### Attempt {n}")
        parts.append(f"```lean\n{proof_preview}\n```")
        parts.append(f"Errors: {error_summary}\n")

    parts.append(
        "Avoid repeating the same mistakes. "
        "Try a fundamentally different approach if previous tactics consistently fail."
    )

    return "\n".join(parts)
