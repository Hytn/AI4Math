"""prover/verifier/sorry_detector.py — 检测 sorry/admit 残留"""
from __future__ import annotations
import re

def detect_sorry(code: str) -> list[int]:
    return [i+1 for i, line in enumerate(code.split("\n"))
            if re.search(r'\bsorry\b|\badmit\b', line)]

def is_sorry_free(code: str) -> bool:
    return len(detect_sorry(code)) == 0
