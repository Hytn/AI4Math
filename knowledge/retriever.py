"""knowledge/retriever.py — Mathlib 知识检索

统一的知识检索接口，整合 premise retrieval 和证明模板。
"""
from __future__ import annotations
from prover.premise.selector import PremiseSelector
from prover.sketch.templates import find_templates
from prover.premise.tactic_suggester import classify_goal, suggest_tactics


class KnowledgeRetriever:
    """Unified knowledge retrieval for the proof agent.

    Combines:
    - Premise retrieval (BM25 + embedding)
    - Proof templates
    - Tactic suggestions
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._premise_selector = PremiseSelector(self.config.get("premise", {}))

    def retrieve(self, theorem_statement: str, top_k: int = 10) -> list[str]:
        """Retrieve relevant premises as strings (for prompt injection)."""
        results = self._premise_selector.retrieve(theorem_statement, top_k)
        return [f"{r['name']}: {r['statement']}" for r in results]

    def retrieve_full(self, theorem_statement: str,
                      goal_target: str = "",
                      top_k: int = 10) -> dict:
        """Retrieve comprehensive knowledge bundle.

        Returns dict with: premises, templates, tactics.
        """
        target = goal_target or theorem_statement
        shape = classify_goal(target)

        premises = self._premise_selector.retrieve(theorem_statement, top_k)
        templates = find_templates(shape)
        tactics = suggest_tactics(target)

        return {
            "premises": premises,
            "templates": [{"name": t.name, "skeleton": t.skeleton,
                           "description": t.description} for t in templates],
            "tactics": tactics,
            "goal_shape": shape,
        }

    def add_premises(self, premises: list[dict]):
        """Add custom premises to the retriever."""
        self._premise_selector.add_premises(premises)
