"""prover/codegen/tactic_generator.py — Tactic 序列生成

基于目标状态和错误反馈生成 tactic 序列。
支持规则引擎 (fast) 和 LLM (powerful) 两种模式。
"""
from __future__ import annotations
import re
from prover.premise.tactic_suggester import suggest_tactics, classify_goal
from common.roles import AgentRole, ROLE_PROMPTS
from common.response_parser import extract_lean_code


class TacticGenerator:
    """Generate tactic sequences for closing proof goals.

    Supports two modes:
        'rule': Rule-based generation (fast, no LLM needed)
        'llm':  LLM-based generation (powerful, needs API)
    """

    def __init__(self, llm=None, mode: str = "rule"):
        self.llm = llm
        self.mode = mode if (mode == "rule" or llm is not None) else "rule"

    def generate(self, target: str, hypotheses: list[str] = None,
                 max_sequences: int = 5, temperature: float = 0.7) -> list[list[str]]:
        """Generate candidate tactic sequences.

        Args:
            target: The goal target string.
            hypotheses: Available hypotheses.
            max_sequences: Number of candidate sequences to generate.
            temperature: LLM temperature (only for llm mode).

        Returns:
            List of tactic sequences (each is a list of tactic strings).
        """
        if self.mode == "llm" and self.llm is not None:
            return self._generate_llm(target, hypotheses, max_sequences, temperature)
        return self._generate_rule(target, hypotheses, max_sequences)

    def _generate_rule(self, target: str, hypotheses: list[str] = None,
                       max_sequences: int = 5) -> list[list[str]]:
        """Generate tactics using rule-based engine."""
        hyps = hypotheses or []
        base_tactics = suggest_tactics(target, hyps, max_suggestions=10)
        shape = classify_goal(target)

        sequences = []

        # Strategy 1: Single tactic closers
        for t in base_tactics[:3]:
            sequences.append([t])

        # Strategy 2: intro + closer for implications/foralls
        if shape in ("forall", "implication"):
            sequences.append(["intro h", "assumption"])
            sequences.append(["intro h", "exact h"])
            sequences.append(["intro h", "simp"])

        # Strategy 3: Constructor-based for conjunctions
        if shape == "conjunction":
            sequences.append(["constructor", "assumption", "assumption"])
            sequences.append(["exact ⟨‹_›, ‹_›⟩"])

        # Strategy 4: Multi-step for equalities
        if shape == "equality":
            sequences.append(["ring"])
            sequences.append(["simp", "ring"])
            sequences.append(["omega"])
            sequences.append(["norm_num"])

        # Strategy 5: Induction for natural number goals
        if shape == "nat_expr":
            sequences.append(["induction n with", "| zero => simp", "| succ n ih => simp [ih]"])

        # Deduplicate
        seen = set()
        unique = []
        for seq in sequences:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                unique.append(seq)

        return unique[:max_sequences]

    def _generate_llm(self, target: str, hypotheses: list[str] = None,
                      max_sequences: int = 5, temperature: float = 0.7) -> list[list[str]]:
        """Generate tactics using LLM."""
        hyps = hypotheses or []
        hyp_str = "\n".join(f"  {h}" for h in hyps) if hyps else "  (none)"
        prompt = (
            f"Goal target:\n  ⊢ {target}\n\n"
            f"Hypotheses:\n{hyp_str}\n\n"
            f"Generate {max_sequences} different tactic sequences to close this goal.\n"
            f"Each sequence on its own line, tactics separated by '; '.\n"
            f"Output ONLY the tactic lines, no explanations."
        )
        resp = self.llm.generate(
            system=ROLE_PROMPTS[AgentRole.SORRY_CLOSER],
            user=prompt, temperature=temperature)

        sequences = []
        for line in resp.content.strip().split("\n"):
            line = line.strip().lstrip("0123456789.- )")
            if line and not line.startswith("#"):
                tactics = [t.strip() for t in line.split(";") if t.strip()]
                if tactics:
                    sequences.append(tactics)
        return sequences[:max_sequences]
