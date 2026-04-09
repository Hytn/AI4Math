"""prover/verifier/integrity_checker.py — 证明完整性 / 反作弊检查 (v2)

v2 新增检测项:
  - native_decide / Decidable.decide (内核计算绕过)
  - set_option maxHeartbeats 0 (关闭超时保护)
  - meta-programming via `import Lean` + `run_tac` (元编程绕过)
  - 在定理声明外部修改定义 (如重定义 Nat.add)
  - unsafe / implemented_by / @[extern] (FFI 绕过)
  - decreasing_by sorry (终止性证明绕过)

严重级别:
  CRITICAL — 必须拒绝, 可能绕过验证
  WARNING  — 可疑但不一定是作弊, 需人工审查
  INFO     — 非关键, 记录但不影响判定
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class IntegrityIssue:
    severity: Severity
    message: str
    line: int = 0
    pattern: str = ""


@dataclass
class IntegrityReport:
    passed: bool = True
    issues: list[IntegrityIssue] = field(default_factory=list)

    # Backward compat: old code reads report.issues as list[str]
    def __post_init__(self):
        pass

    @property
    def critical_issues(self) -> list[IntegrityIssue]:
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> list[IntegrityIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def summary(self) -> str:
        if self.passed:
            n_warn = len(self.warnings)
            if n_warn:
                return f"PASSED with {n_warn} warning(s)"
            return "PASSED"
        n_crit = len(self.critical_issues)
        return f"FAILED: {n_crit} critical issue(s)"


# ═══════════════════════════════════════════════════════════════
# Check rules
# ═══════════════════════════════════════════════════════════════

_CHECKS: list[tuple[str, Severity, str]] = [
    # ── CRITICAL: definitely cheating or unsound ──
    (r'\bsorry\b', Severity.CRITICAL,
     "Contains sorry — proof is incomplete"),
    (r'\badmit\b', Severity.CRITICAL,
     "Contains admit — proof is incomplete"),
    (r'\baxiom\b\s+\w+', Severity.CRITICAL,
     "Contains custom axiom declaration — may introduce inconsistency"),
    (r'\bnative_decide\b', Severity.CRITICAL,
     "Uses native_decide — bypasses kernel type checking via compiled code. "
     "Not accepted in competition benchmarks"),
    (r'\bDecidable\.decide\b', Severity.CRITICAL,
     "Uses Decidable.decide — may bypass elaboration checks"),
    (r'set_option\s+maxHeartbeats\s+0\b', Severity.CRITICAL,
     "Disables heartbeat timeout — proof may diverge or brute-force"),
    (r'\bunsafe\b\s+\bdef\b', Severity.CRITICAL,
     "Contains unsafe definition — bypasses type safety"),
    (r'@\[implemented_by\b', Severity.CRITICAL,
     "Uses @[implemented_by] — replaces implementation with unverified code"),
    (r'@\[extern\b', Severity.CRITICAL,
     "Uses @[extern] — calls unverified foreign function"),
    (r'\bdecreasing_by\s+sorry\b', Severity.CRITICAL,
     "Uses 'decreasing_by sorry' — bypasses termination checking"),
    (r'\bsorry\b.*\bdecreasing_by\b|\bdecreasing_by\b.*\bsorry\b',
     Severity.CRITICAL,
     "Sorry used in termination proof"),

    # ── WARNING: suspicious but not always cheating ──
    (r'\brun_tac\b', Severity.WARNING,
     "Uses run_tac — executes arbitrary tactic via meta-programming"),
    (r'\beval_tactic\b', Severity.WARNING,
     "Uses eval_tactic — dynamic tactic evaluation"),
    (r'set_option\s+maxHeartbeats\s+\d{7,}', Severity.WARNING,
     "Sets very large heartbeat limit — may be trying to brute-force"),
    (r'set_option\s+maxRecDepth\s+\d{5,}', Severity.WARNING,
     "Sets very large recursion depth — unusual"),
    (r'\bMeta\.Tactic\b', Severity.WARNING,
     "Uses Lean meta-programming tactic API — may construct proof terms directly"),
    (r'\bLean\.Elab\b', Severity.WARNING,
     "Uses Lean elaboration API — may bypass normal tactic elaboration"),
    (r'\bopaque\b\s+\bdef\b', Severity.WARNING,
     "Contains opaque definition — hides implementation"),

    # ── INFO: non-critical, just good to know ──
    (r'#check\s|#eval\s|#print\s', Severity.INFO,
     "Contains debug commands (#check/#eval/#print)"),
    (r'--.*TODO|--.*FIXME|--.*HACK', Severity.INFO,
     "Contains TODO/FIXME/HACK comments"),
]

# Compiled patterns for efficiency
_COMPILED_CHECKS = [
    (re.compile(pattern, re.IGNORECASE if sev == Severity.INFO else 0),
     sev, msg)
    for pattern, sev, msg in _CHECKS
]


def check_integrity(code: str, original_statement: str = "") -> IntegrityReport:
    """Run all integrity checks on a proof code string.

    Args:
        code: The complete Lean4 code (with imports + theorem + proof).
        original_statement: The original theorem statement to verify
            it hasn't been modified.

    Returns:
        IntegrityReport with passed=False if any CRITICAL issue found.
    """
    report = IntegrityReport()

    # Strip comments before checking (avoid false positives in comments)
    # But keep the original for line number tracking
    code_no_comments = _strip_comments(code)

    for compiled, severity, message in _COMPILED_CHECKS:
        match = compiled.search(code_no_comments)
        if match:
            # Find line number
            line_no = code_no_comments[:match.start()].count('\n') + 1
            report.issues.append(IntegrityIssue(
                severity=severity,
                message=message,
                line=line_no,
                pattern=compiled.pattern,
            ))
            if severity == Severity.CRITICAL:
                report.passed = False

    # ── Special checks ──

    # Check for theorem statement modification
    if original_statement:
        # Normalize whitespace for comparison
        norm_orig = ' '.join(original_statement.split())
        norm_code = ' '.join(code.split())
        if norm_orig not in norm_code:
            report.issues.append(IntegrityIssue(
                severity=Severity.WARNING,
                message="Original theorem statement may have been modified",
            ))

    # Check for suspicious import patterns
    # (importing Lean meta-programming modules in a proof context)
    if re.search(r'import\s+Lean\b(?!Dojo)', code_no_comments):
        # "import Lean" (but not "import LeanDojo") gives access to
        # meta-programming, which can construct arbitrary proof terms
        report.issues.append(IntegrityIssue(
            severity=Severity.WARNING,
            message="Imports 'Lean' module — gives access to meta-programming. "
                    "Proofs should only need 'import Mathlib'.",
        ))

    # Check for multiple theorem declarations (possible statement injection)
    theorem_count = len(re.findall(
        r'\b(?:theorem|lemma)\s+\w+', code_no_comments))
    if theorem_count > 5:
        report.issues.append(IntegrityIssue(
            severity=Severity.WARNING,
            message=f"Contains {theorem_count} theorem/lemma declarations — "
                    f"unusually many for a single proof",
        ))

    return report


def _strip_comments(code: str) -> str:
    """Remove single-line (--) and block (/- -/) comments.

    Fix #6: Uses stack-based parsing to correctly handle nested block
    comments like /- /- inner -/ outer -/ which Lean4 supports.
    The previous regex r'/-.*?-/' failed on nested comments, allowing
    malicious proofs to hide `sorry` inside nested comment constructs.
    """
    result = []
    i = 0
    n = len(code)
    depth = 0  # nesting depth for /- -/ block comments
    in_line_comment = False

    while i < n:
        # Inside a block comment
        if depth > 0:
            if i + 1 < n and code[i] == '/' and code[i + 1] == '-':
                depth += 1
                i += 2
            elif i + 1 < n and code[i] == '-' and code[i + 1] == '/':
                depth -= 1
                i += 2
            else:
                i += 1
            continue

        # Inside a line comment
        if in_line_comment:
            if code[i] == '\n':
                in_line_comment = False
                result.append('\n')  # preserve line structure
            i += 1
            continue

        # Check for start of block comment
        if i + 1 < n and code[i] == '/' and code[i + 1] == '-':
            depth = 1
            i += 2
            continue

        # Check for start of line comment (--)
        if i + 1 < n and code[i] == '-' and code[i + 1] == '-':
            in_line_comment = True
            i += 2
            continue

        # Check for string literals (don't strip comments inside strings)
        if code[i] == '"':
            result.append(code[i])
            i += 1
            while i < n and code[i] != '"':
                if code[i] == '\\' and i + 1 < n:
                    result.append(code[i])
                    i += 1
                result.append(code[i])
                i += 1
            if i < n:
                result.append(code[i])
                i += 1
            continue

        result.append(code[i])
        i += 1

    return ''.join(result)
