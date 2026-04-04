"""prover/codegen/scaffold_generator.py — Sorry-based 证明骨架生成

Generates proof skeletons with `sorry` placeholders:
1. Uses proof templates for known goal shapes
2. Falls back to LLM for complex goals
3. Integrates with tactic_suggester for initial tactic hints
"""
from __future__ import annotations
from agent.brain.roles import AgentRole, ROLE_PROMPTS
from agent.brain.response_parser import extract_lean_code
from prover.sketch.templates import find_templates, fill_template
from prover.premise.tactic_suggester import classify_goal


SCAFFOLD_SYSTEM = """\
You are a Lean 4 proof architect. Generate a proof SKELETON with `sorry` placeholders.
Break the proof into `have` steps. Each step should be independently provable.
Mark each unproved step with `sorry`. The overall structure must be logically sound.
Use standard Lean 4 syntax with Mathlib tactics."""


class ScaffoldGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, theorem: str, sketch: str = "",
                 premises: list[str] = None,
                 goal_target: str = "") -> str:
        """Generate a sorry-skeleton proof.

        First tries template-based generation for known patterns,
        then falls back to LLM generation.
        """
        target = goal_target or theorem

        # Try template-based skeleton first
        shape = classify_goal(target)
        templates = find_templates(shape)
        if templates:
            skeleton = fill_template(templates[0], {})
            # Return template-based skeleton if it looks reasonable
            if "sorry" in skeleton and len(skeleton) > 10:
                return skeleton

        # Fall back to LLM
        premise_section = ""
        if premises:
            premise_section = (
                "\nUseful lemmas:\n" +
                "\n".join(f"- {p}" for p in premises[:10]))

        prompt = (
            f"Theorem:\n```lean\n{theorem}\n```\n"
            f"Sketch: {sketch}\n"
            f"{premise_section}\n\n"
            f"Generate a sorry-skeleton proof. Use `have` steps for "
            f"intermediate goals. Each `have` should have an explicit type."
        )
        resp = self.llm.generate(
            system=SCAFFOLD_SYSTEM,
            user=prompt,
            temperature=0.5)
        return extract_lean_code(resp.content)
