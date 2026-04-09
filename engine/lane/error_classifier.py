"""engine/lane/error_classifier.py — Lean error → ProofFailureClass mapping

The bridge between raw Lean 4 compiler output and the lane failure taxonomy.
Used by ProofPipeline to record structured failures on the state machine,
which in turn drives the RecoveryRegistry and PolicyEngine.

Design:
  - Deterministic: same error text → same ProofFailureClass
  - Extensible: add patterns to _PATTERNS without changing call sites
  - Fast: O(n) scan over short pattern list, no regex compilation at call time

Usage::

    from engine.lane.error_classifier import classify_lean_error

    fc = classify_lean_error("unknown identifier 'Nat.sub_add_cancel'")
    # → ProofFailureClass.UNKNOWN_IDENTIFIER

    fc = classify_lean_error("", is_timeout=True)
    # → ProofFailureClass.TIMEOUT
"""
from __future__ import annotations

import re
from typing import Optional

from engine.lane.task_state import ProofFailureClass


# ── Pattern table: (compiled_regex, failure_class) ─────────────────────────
# Order matters: first match wins. More specific patterns go first.

_PATTERNS: list[tuple[re.Pattern, ProofFailureClass]] = [
    # Integrity violations (highest priority — reject immediately)
    (re.compile(r"\bsorry\b", re.IGNORECASE), ProofFailureClass.SORRY_DETECTED),
    (re.compile(r"declaration uses 'sorry'", re.IGNORECASE), ProofFailureClass.SORRY_DETECTED),
    (re.compile(r"\bnative_decide\b.*failed", re.IGNORECASE), ProofFailureClass.INTEGRITY_VIOLATION),

    # Import / namespace errors
    (re.compile(r"unknown (package|namespace|module)", re.IGNORECASE), ProofFailureClass.IMPORT_ERROR),
    (re.compile(r"import.*not found", re.IGNORECASE), ProofFailureClass.IMPORT_ERROR),
    (re.compile(r"could not resolve import", re.IGNORECASE), ProofFailureClass.IMPORT_ERROR),

    # Unknown identifier (before general tactic failure)
    (re.compile(r"unknown (identifier|constant|declaration)", re.IGNORECASE), ProofFailureClass.UNKNOWN_IDENTIFIER),
    (re.compile(r"undeclared (local|universe)", re.IGNORECASE), ProofFailureClass.UNKNOWN_IDENTIFIER),

    # Type mismatch
    (re.compile(r"type mismatch", re.IGNORECASE), ProofFailureClass.TYPE_MISMATCH),
    (re.compile(r"has type.*but is expected to have type", re.IGNORECASE), ProofFailureClass.TYPE_MISMATCH),
    (re.compile(r"application type mismatch", re.IGNORECASE), ProofFailureClass.TYPE_MISMATCH),
    (re.compile(r"failed to synthesize.*instance", re.IGNORECASE), ProofFailureClass.TYPE_MISMATCH),

    # Tactic failures
    (re.compile(r"tactic '.*' failed", re.IGNORECASE), ProofFailureClass.TACTIC_FAILED),
    (re.compile(r"unsolved goals", re.IGNORECASE), ProofFailureClass.TACTIC_FAILED),
    (re.compile(r"goals accomplished", re.IGNORECASE), ProofFailureClass.TACTIC_FAILED),  # partial
    (re.compile(r"simp made no progress", re.IGNORECASE), ProofFailureClass.TACTIC_FAILED),
    (re.compile(r"omega failed", re.IGNORECASE), ProofFailureClass.TACTIC_FAILED),
    (re.compile(r"ring_nf made no progress", re.IGNORECASE), ProofFailureClass.TACTIC_FAILED),
    (re.compile(r"linarith failed", re.IGNORECASE), ProofFailureClass.TACTIC_FAILED),

    # Syntax errors
    (re.compile(r"expected\s+(token|')", re.IGNORECASE), ProofFailureClass.SYNTAX_ERROR),
    (re.compile(r"unexpected (token|end of input)", re.IGNORECASE), ProofFailureClass.SYNTAX_ERROR),
    (re.compile(r"parse error", re.IGNORECASE), ProofFailureClass.SYNTAX_ERROR),

    # Timeout patterns
    (re.compile(r"(deterministic )?timeout", re.IGNORECASE), ProofFailureClass.TIMEOUT),
    (re.compile(r"maximum recursion depth", re.IGNORECASE), ProofFailureClass.TIMEOUT),
    (re.compile(r"(deep|max) recursion", re.IGNORECASE), ProofFailureClass.TIMEOUT),
    (re.compile(r"heartbeat", re.IGNORECASE), ProofFailureClass.TIMEOUT),
]


def classify_lean_error(
    error_text: str,
    *,
    error_category: str = "",
    is_timeout: bool = False,
    is_api_error: bool = False,
    is_repl_crash: bool = False,
) -> ProofFailureClass:
    """Classify a Lean compilation error into a ProofFailureClass.

    Args:
        error_text: Raw Lean compiler error output.
        error_category: Optional pre-classified category from ErrorIntelligence
            (e.g. "type_mismatch", "unknown_identifier"). Used as a hint
            if regex matching is ambiguous.
        is_timeout: True if the error was a timeout (overrides pattern matching).
        is_api_error: True if the error was an LLM API failure.
        is_repl_crash: True if the REPL session crashed.

    Returns:
        The most specific ProofFailureClass that matches.
    """
    # Infrastructure failures take highest precedence
    if is_repl_crash:
        return ProofFailureClass.REPL_CRASH
    if is_api_error:
        return ProofFailureClass.API_ERROR
    if is_timeout:
        return ProofFailureClass.TIMEOUT

    # Try regex patterns on the error text
    if error_text:
        for pattern, fc in _PATTERNS:
            if pattern.search(error_text):
                return fc

    # Fall back to error_category hint from ErrorIntelligence
    if error_category:
        _CATEGORY_MAP = {
            "type_mismatch": ProofFailureClass.TYPE_MISMATCH,
            "unknown_identifier": ProofFailureClass.UNKNOWN_IDENTIFIER,
            "tactic_failed": ProofFailureClass.TACTIC_FAILED,
            "syntax_error": ProofFailureClass.SYNTAX_ERROR,
            "timeout": ProofFailureClass.TIMEOUT,
            "import_error": ProofFailureClass.IMPORT_ERROR,
            "sorry": ProofFailureClass.SORRY_DETECTED,
        }
        fc = _CATEGORY_MAP.get(error_category.lower())
        if fc:
            return fc

    # Default: treat as generic tactic failure
    return ProofFailureClass.TACTIC_FAILED


def classify_verification_result(
    vr: object,
    *,
    is_timeout: bool = False,
) -> Optional[ProofFailureClass]:
    """Classify a VerificationResult (from VerificationScheduler) into a failure class.

    Returns None if verification succeeded (no failure to classify).
    """
    if getattr(vr, 'success', False):
        return None

    error_text = ""
    error_category = ""
    if hasattr(vr, 'feedback') and vr.feedback:
        error_text = getattr(vr.feedback, 'error_message', '') or ""
        error_category = getattr(vr.feedback, 'error_category', '') or ""

    return classify_lean_error(
        error_text,
        error_category=error_category,
        is_timeout=is_timeout,
    )
