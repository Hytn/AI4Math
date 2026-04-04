"""prover/verifier/error_parser.py — Lean stderr 结构化解析

Extracts structured error information from Lean4 compiler output:
  - Line/column numbers
  - Error categories
  - Expected vs actual types (for type mismatch errors)
  - Suggested fixes (from Lean4's "did you mean" output)
"""
from __future__ import annotations
import re
from prover.models import LeanError, ErrorCategory

_ERROR_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*error:\s*(?P<msg>.+?)(?=\n\S|\Z)",
    re.MULTILINE | re.DOTALL)

_CAT_PATTERNS = [
    (re.compile(r"type mismatch", re.I), ErrorCategory.TYPE_MISMATCH),
    (re.compile(r"unknown (identifier|constant|declaration)", re.I), ErrorCategory.UNKNOWN_IDENTIFIER),
    (re.compile(r"tactic .+ failed|unsolved goals", re.I), ErrorCategory.TACTIC_FAILED),
    (re.compile(r"expected .+ got|unexpected token|expected token", re.I), ErrorCategory.SYNTAX_ERROR),
    (re.compile(r"import .+ not found|unknown package", re.I), ErrorCategory.IMPORT_ERROR),
    (re.compile(r"timeout|deterministic timeout|maxHeartbeats", re.I), ErrorCategory.TIMEOUT),
    (re.compile(r"elaboration|failed to synthesize", re.I), ErrorCategory.ELABORATION_ERROR),
]

# Patterns for extracting structured type info from error messages
_TYPE_MISMATCH_EXPECTED = re.compile(r"expected\s+(?:type)?\s*\n?\s*(.+?)(?:\n|$)", re.I)
_TYPE_MISMATCH_ACTUAL = re.compile(r"(?:has type|but is expected to have type|got)\s*\n?\s*(.+?)(?:\n|$)", re.I)
_DID_YOU_MEAN = re.compile(r"did you mean[:\s]+(.+?)(?:\?|\n|$)", re.I)
_UNKNOWN_ID = re.compile(r"unknown (?:identifier|constant) ['\u2018](.+?)['\u2019]", re.I)


def parse_lean_errors(stderr: str) -> list[LeanError]:
    """Parse Lean4 stderr into structured LeanError objects."""
    errors = []
    for m in _ERROR_RE.finditer(stderr):
        msg = m.group("msg").strip()
        cat = ErrorCategory.OTHER
        for pat, c in _CAT_PATTERNS:
            if pat.search(msg):
                cat = c
                break

        error = LeanError(
            category=cat,
            message=msg[:500],
            line=int(m.group("line")),
            column=int(m.group("col")),
            raw=m.group(0)[:1000],
        )

        # Extract structured type mismatch info
        if cat == ErrorCategory.TYPE_MISMATCH:
            exp = _TYPE_MISMATCH_EXPECTED.search(msg)
            act = _TYPE_MISMATCH_ACTUAL.search(msg)
            if exp:
                error.expected_type = exp.group(1).strip()[:200]
            if act:
                error.actual_type = act.group(1).strip()[:200]

        # Extract "did you mean" suggestions
        for suggestion_match in _DID_YOU_MEAN.finditer(msg):
            suggestions = [s.strip() for s in suggestion_match.group(1).split(",")]
            error.suggestions.extend(suggestions)

        # Extract unknown identifier name for targeted fixes
        if cat == ErrorCategory.UNKNOWN_IDENTIFIER:
            uid = _UNKNOWN_ID.search(msg)
            if uid:
                error.suggestions.append(f"unknown:{uid.group(1)}")

        errors.append(error)

    if not errors and stderr.strip():
        errors.append(LeanError(
            category=ErrorCategory.OTHER,
            message=stderr[:500],
            raw=stderr[:500],
        ))
    return errors


def summarize_errors(errors: list[LeanError]) -> str:
    """Create a concise summary of errors for LLM prompts."""
    if not errors:
        return "No errors."
    parts = []
    for e in errors[:5]:
        loc = f"line {e.line}" if e.line else "unknown location"
        parts.append(f"[{e.category.value}] {loc}: {e.message[:150]}")
        if e.expected_type and e.actual_type:
            parts.append(f"  expected: {e.expected_type[:100]}")
            parts.append(f"  actual:   {e.actual_type[:100]}")
        if e.suggestions:
            parts.append(f"  suggestions: {', '.join(e.suggestions[:3])}")
    if len(errors) > 5:
        parts.append(f"  ... and {len(errors) - 5} more errors")
    return "\n".join(parts)
