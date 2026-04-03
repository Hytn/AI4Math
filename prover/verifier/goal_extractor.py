"""prover/verifier/goal_extractor.py — 从 REPL/编译输出中提取 goal state"""
from __future__ import annotations
import re

def extract_goals(lean_output: str) -> list[str]:
    goals = re.findall(r"⊢\s*(.+?)(?:\n|$)", lean_output)
    return goals

def extract_unsolved_goals(stderr: str) -> list[str]:
    m = re.search(r"unsolved goals\n(.*?)(?:\n\n|\Z)", stderr, re.DOTALL)
    return [m.group(1).strip()] if m else []
