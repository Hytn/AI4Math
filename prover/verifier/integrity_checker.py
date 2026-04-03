"""prover/verifier/integrity_checker.py — 证明完整性 / 反作弊检查"""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass
class IntegrityReport:
    passed: bool = True; issues: list[str] = None
    def __post_init__(self):
        if self.issues is None: self.issues = []

def check_integrity(code: str, original_statement: str = "") -> IntegrityReport:
    report = IntegrityReport()
    if re.search(r'\bsorry\b|\badmit\b', code):
        report.passed = False; report.issues.append("Contains sorry/admit")
    if re.search(r'\baxiom\b\s+\w+', code):
        report.passed = False; report.issues.append("Contains custom axiom declaration")
    if re.search(r'#check\s|#eval\s|#print\s', code):
        report.issues.append("Contains debug commands (non-critical)")
    if original_statement and original_statement not in code:
        report.issues.append("Original theorem statement not found in code (warning)")
    return report
