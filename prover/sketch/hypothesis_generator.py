"""prover/sketch/hypothesis_generator.py — 假说生成器

根据定理结构生成可能有用的中间假说 (have steps)。
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from prover.premise.tactic_suggester import classify_goal


@dataclass
class Hypothesis:
    """A proposed intermediate hypothesis."""
    name: str
    statement: str
    rationale: str
    priority: float = 0.5  # 0-1, higher = more promising


class HypothesisGenerator:
    """Generate intermediate hypotheses for proof planning."""

    def __init__(self, llm=None):
        self.llm = llm

    def generate(self, theorem: str, target: str = "",
                 hypotheses: list[str] = None,
                 max_hypotheses: int = 5) -> list[Hypothesis]:
        """Generate intermediate hypotheses.

        Args:
            theorem: The full theorem statement.
            target: The current goal target.
            hypotheses: Available hypotheses.
            max_hypotheses: Max number to generate.
        """
        if self.llm:
            return self._generate_llm(theorem, target, hypotheses, max_hypotheses)
        return self._generate_rule(theorem, target, hypotheses, max_hypotheses)

    def _generate_rule(self, theorem: str, target: str,
                       hypotheses: list[str] = None,
                       max_hypotheses: int = 5) -> list[Hypothesis]:
        """Rule-based hypothesis generation."""
        shape = classify_goal(target or theorem)
        results = []

        # For equalities: suggest intermediate equality steps
        if shape == "equality":
            match = re.search(r'(.+?)\s*=\s*(.+)', target or theorem)
            if match:
                lhs, rhs = match.group(1).strip(), match.group(2).strip()
                results.append(Hypothesis(
                    name="h_mid", statement=f"{lhs} = sorry",
                    rationale="Intermediate equality step",
                    priority=0.6))

        # For implications: suggest strengthening the hypothesis
        if shape in ("implication", "forall"):
            results.append(Hypothesis(
                name="h_key", statement="sorry",
                rationale="Key intermediate lemma",
                priority=0.5))

        # For inequality goals: suggest bounding
        if shape in ("le", "lt"):
            results.append(Hypothesis(
                name="h_bound", statement="sorry",
                rationale="Intermediate bound",
                priority=0.6))

        # For nat goals: suggest case split on 0/succ
        if shape == "nat_expr":
            results.append(Hypothesis(
                name="h_base", statement="sorry",
                rationale="Base case (n = 0)",
                priority=0.7))
            results.append(Hypothesis(
                name="h_step", statement="sorry",
                rationale="Inductive step",
                priority=0.7))

        return results[:max_hypotheses]

    def _generate_llm(self, theorem: str, target: str,
                      hypotheses: list[str] = None,
                      max_hypotheses: int = 5) -> list[Hypothesis]:
        """LLM-based hypothesis generation."""
        hyps = "\n".join(hypotheses or []) or "(none)"
        prompt = (
            f"Theorem: {theorem}\n"
            f"Current goal: ⊢ {target}\n"
            f"Hypotheses:\n{hyps}\n\n"
            f"Suggest {max_hypotheses} intermediate 'have' steps (as Lean4 types) "
            f"that would help prove this. Format: one per line, 'name : type'."
        )
        resp = self.llm.generate(
            system="You are an expert proof planner.",
            user=prompt, temperature=0.7)

        results = []
        for line in resp.content.strip().split("\n"):
            match = re.match(r'(\w+)\s*:\s*(.+)', line.strip())
            if match:
                results.append(Hypothesis(
                    name=match.group(1), statement=match.group(2).strip(),
                    rationale="LLM-suggested intermediate step",
                    priority=0.6))
        return results[:max_hypotheses]
