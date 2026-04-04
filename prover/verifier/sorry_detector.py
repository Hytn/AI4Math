"""prover/verifier/sorry_detector.py — 深度 sorry/admit 检测

不仅匹配关键字，还检测伪装的 sorry 模式:
- native_decide 滥用
- sorry 被重定义
- axiom 注入
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class SorryReport:
    has_sorry: bool = False
    locations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.has_sorry and not self.warnings


def detect_sorry(lean_code: str) -> SorryReport:
    """Scan Lean4 code for sorry, admit, and suspicious axiom patterns.

    Returns a SorryReport with locations and warnings.
    """
    report = SorryReport()
    lines = lean_code.split("\n")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments
        if stripped.startswith("--"):
            continue

        # Direct sorry / admit
        for keyword in ["sorry", "admit"]:
            if re.search(rf'\b{keyword}\b', stripped):
                report.has_sorry = True
                report.locations.append({
                    "line": i, "keyword": keyword,
                    "content": stripped[:100],
                    "kind": "direct",
                })

        # Custom axiom declarations (potential backdoor)
        if re.match(r'\s*axiom\s+\w+', stripped):
            report.warnings.append(
                f"Line {i}: Custom axiom declaration — may introduce inconsistency: "
                f"{stripped[:80]}")

        # native_decide abuse (can hang on large inputs)
        if re.search(r'\bnative_decide\b', stripped):
            report.warnings.append(
                f"Line {i}: native_decide used — verify it terminates: "
                f"{stripped[:80]}")

        # unsafeCoerce (bypasses type checker)
        if re.search(r'\bunsafeCoerce\b|\bunsafe\b', stripped):
            report.warnings.append(
                f"Line {i}: Unsafe coercion detected: {stripped[:80]}")

        # Lean options that weaken checking
        if re.search(r'set_option\s+.*maxHeartbeats\s+0', stripped):
            report.warnings.append(
                f"Line {i}: maxHeartbeats set to 0 — may not terminate")

        # sorry redefinition
        if re.match(r'\s*def\s+sorry\b', stripped) or re.match(r'\s*abbrev\s+sorry\b', stripped):
            report.warnings.append(
                f"Line {i}: sorry redefinition detected: {stripped[:80]}")

    return report


def count_sorries(lean_code: str) -> int:
    """Quick count of sorry occurrences."""
    return len(re.findall(r'\bsorry\b', lean_code))


def extract_sorry_locations(lean_code: str) -> list[dict]:
    """Extract line numbers and context of each sorry."""
    report = detect_sorry(lean_code)
    return report.locations
