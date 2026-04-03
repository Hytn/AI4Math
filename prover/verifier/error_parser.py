"""prover/verifier/error_parser.py — Lean stderr 结构化解析"""
from __future__ import annotations
import re
from prover.models import LeanError, ErrorCategory

_ERROR_RE = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*error:\s*(?P<msg>.+)$", re.MULTILINE)
_CAT_PATTERNS = [
    (re.compile(r"type mismatch", re.I), ErrorCategory.TYPE_MISMATCH),
    (re.compile(r"unknown (identifier|constant)", re.I), ErrorCategory.UNKNOWN_IDENTIFIER),
    (re.compile(r"tactic .+ failed|unsolved goals", re.I), ErrorCategory.TACTIC_FAILED),
    (re.compile(r"expected .+ got|unexpected token", re.I), ErrorCategory.SYNTAX_ERROR),
    (re.compile(r"import .+ not found", re.I), ErrorCategory.IMPORT_ERROR),
    (re.compile(r"timeout|deterministic timeout", re.I), ErrorCategory.TIMEOUT),
]

def parse_lean_errors(stderr: str) -> list[LeanError]:
    errors = []
    for m in _ERROR_RE.finditer(stderr):
        msg = m.group("msg").strip()
        cat = ErrorCategory.OTHER
        for pat, c in _CAT_PATTERNS:
            if pat.search(msg): cat = c; break
        errors.append(LeanError(category=cat, message=msg, line=int(m.group("line")),
                                column=int(m.group("col")), raw=m.group(0)))
    if not errors and stderr.strip():
        errors.append(LeanError(category=ErrorCategory.OTHER, message=stderr[:500], raw=stderr[:500]))
    return errors
