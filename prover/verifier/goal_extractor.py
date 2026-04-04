"""prover/verifier/goal_extractor.py — 从 Lean4 输出中提取目标状态

解析 Lean4 编译器 / REPL 输出中的 goal state。
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class ExtractedGoal:
    """A single goal extracted from Lean output."""
    index: int
    target: str
    hypotheses: list[dict] = field(default_factory=list)
    case_name: str = ""

    def to_string(self) -> str:
        parts = []
        if self.case_name:
            parts.append(f"case {self.case_name}")
        for h in self.hypotheses:
            parts.append(f"{h['name']} : {h['type']}")
        parts.append(f"⊢ {self.target}")
        return "\n".join(parts)


def extract_goals(lean_output: str) -> list[ExtractedGoal]:
    """Parse goal states from Lean4 compiler or REPL output.

    Supports formats like:
        case name
        h1 : Type1
        h2 : Type2
        ⊢ Target
    """
    goals = []

    # Pattern 1: 'unsolved goals' block
    unsolved_match = re.search(
        r'unsolved goals\s*\n(.*?)(?=\n\S|\Z)', lean_output, re.DOTALL)
    if unsolved_match:
        block = unsolved_match.group(1)
        parsed = _parse_goal_block(block, len(goals))
        if parsed:
            goals.extend(parsed)

    # Pattern 2: Any block containing ⊢
    if not goals and "⊢" in lean_output:
        parsed = _parse_goal_block(lean_output, 0)
        if parsed:
            goals.extend(parsed)

    # Deduplicate by target
    seen = set()
    unique = []
    for g in goals:
        key = g.target.strip()
        if key not in seen:
            seen.add(key)
            unique.append(g)

    return unique


def _parse_goal_block(block: str, start_index: int) -> list[ExtractedGoal]:
    """Parse a single goal block into an ExtractedGoal."""
    lines = block.strip().split("\n")
    if not lines:
        return []

    hypotheses = []
    target = ""
    case_name = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Case name
        case_match = re.match(r'^case\s+(\w+)', line)
        if case_match:
            case_name = case_match.group(1)
            continue

        # Turnstile: target
        if line.startswith("⊢ ") or line.startswith("|- "):
            prefix_len = 2
            target = line[prefix_len:].strip()
            continue

        # Hypothesis: name : type (but not keywords like 'error:', 'unsolved')
        hyp_match = re.match(r'^([a-zA-Z_]\w*)\s*:\s*(?![:=])(.+)$', line)
        if hyp_match:
            name = hyp_match.group(1)
            # Skip if name is a Lean keyword in error context
            if name.lower() not in ("error", "warning", "info", "unsolved"):
                hypotheses.append({
                    "name": name,
                    "type": hyp_match.group(2).strip(),
                })

    if target:
        return [ExtractedGoal(
            index=start_index, target=target,
            hypotheses=hypotheses, case_name=case_name)]
    return []


def format_goal_for_prompt(goal: ExtractedGoal) -> str:
    """Format a goal for inclusion in an LLM prompt."""
    return goal.to_string()
