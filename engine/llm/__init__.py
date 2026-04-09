"""
engine/llm — LLM Tactic Suggestion Engine

Integrates with Claude API to suggest tactics based on goal state.
The LLM sees a token-efficient GoalView and returns candidate tactics.

Uses agent.brain.claude_provider when available, falls back to heuristic.
"""
from __future__ import annotations
import json
import time
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class LLMSuggestion:
    tactics: List[str]
    reasoning: str = ""
    elapsed_ms: float = 0
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


TACTIC_PROMPT = """You are an expert Lean 4 theorem prover. Given a proof goal, suggest candidate tactics.

RULES:
- Output ONLY a JSON array of tactic strings, nothing else
- Suggest 3-8 tactics ranked by likelihood of success
- Available tactics: intro, assumption, apply, exact, cases, induction, simp, rfl, trivial, ring, omega, linarith, constructor, contradiction
- For `intro`, specify the variable name: "intro x"
- For `apply`, specify the lemma: "apply lemma_name"
- For `exact`, specify the term: "exact term"

GOAL STATE:
{goal_state}

HYPOTHESES:
{hypotheses}

TARGET:
{target}

Respond with ONLY a JSON array like: ["intro x", "apply h", "assumption"]"""


class LLMTacticEngine:
    """Suggests tactics using an LLM.

    Can use either a pre-configured LLMProvider (recommended) or
    fall back to heuristic suggestion.
    """

    def __init__(self, llm_provider=None, use_api: bool = True):
        self._llm = llm_provider
        self.use_api = use_api and (llm_provider is not None)
        self._call_count = 0

    def suggest(self, goal_view: dict, max_suggestions: int = 6) -> LLMSuggestion:
        """Get tactic suggestions from LLM."""
        t0 = time.time()

        if self.use_api and self._llm is not None:
            return self._suggest_api(goal_view, max_suggestions, t0)
        else:
            return self._suggest_heuristic(goal_view, max_suggestions, t0)

    def suggest_from_goal_views(self, goal_views: list,
                                max_suggestions: int = 8) -> list[str]:
        """Suggest tactics from GoalView objects (bridge for SearchCoordinator).

        Args:
            goal_views: List of GoalView objects from engine.state.views.
            max_suggestions: Maximum tactics to return.

        Returns:
            List of tactic strings.
        """
        if not goal_views:
            return ["simp", "assumption", "trivial"]

        # Use the first (main) goal for suggestion
        gv = goal_views[0]
        view_dict = gv.to_dict() if hasattr(gv, 'to_dict') else {
            "target": str(gv.target) if hasattr(gv, 'target') else "?",
            "shape": gv.target_shape.value if hasattr(gv, 'target_shape') else "other",
            "hypotheses": gv.relevant_hyps if hasattr(gv, 'relevant_hyps') else [],
            "depth": gv.depth if hasattr(gv, 'depth') else 0,
        }

        suggestion = self.suggest(view_dict, max_suggestions)
        return suggestion.tactics

    def _suggest_api(self, goal_view: dict, max_suggestions: int,
                     t0: float) -> LLMSuggestion:
        """Call LLM provider for tactic suggestions."""
        try:
            target = goal_view.get("target", "?")
            shape = goal_view.get("shape", "other")
            hyps = goal_view.get("hypotheses", [])
            depth = goal_view.get("depth", 0)

            hyp_str = "\n".join(
                f"  {h.get('name', '?')} : {h.get('type', '?')}"
                for h in hyps
            ) if hyps else "  (none)"

            prompt = TACTIC_PROMPT.format(
                goal_state=f"depth={depth}, shape={shape}",
                hypotheses=hyp_str,
                target=target,
            )

            self._call_count += 1
            response = self._llm.generate(
                system="You are a Lean 4 tactic suggestion engine. "
                       "Output ONLY valid JSON arrays of tactic strings.",
                user=prompt,
                temperature=0.3,
                max_tokens=200)

            tactics = _parse_tactic_response(response.content)
            elapsed = (time.time() - t0) * 1000

            return LLMSuggestion(
                tactics=tactics[:max_suggestions],
                reasoning=response.content,
                elapsed_ms=elapsed,
                model=response.model,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
            )
        except Exception as e:
            # Fallback to heuristic
            return self._suggest_heuristic(goal_view, max_suggestions, t0)

    def _suggest_heuristic(self, goal_view: dict, max_suggestions: int,
                           t0: float) -> LLMSuggestion:
        """Rule-based tactic suggestion (no LLM needed)."""
        target = goal_view.get("target", "")
        shape = goal_view.get("shape", "other")
        hyps = goal_view.get("hypotheses", [])
        depth = goal_view.get("depth", 0)

        tactics = []

        # Shape-based heuristics
        if shape in ("forall", "implication") or "∀" in target or "→" in target or "pi" in target.lower():
            name = f"h{depth}" if depth > 0 else "x"
            tactics.append(f"intro {name}")

        if hyps:
            tactics.append("assumption")
            for h in hyps[:3]:
                hname = h.get("name", "")
                if hname:
                    tactics.append(f"apply {hname}")
                    tactics.append(f"exact {hname}")

        # Generic fallbacks
        tactics.extend(["trivial", "simp", "rfl", "omega"])

        elapsed = (time.time() - t0) * 1000
        return LLMSuggestion(
            tactics=tactics[:max_suggestions],
            reasoning="heuristic",
            elapsed_ms=elapsed,
            model="heuristic",
        )


def _parse_tactic_response(text: str) -> List[str]:
    """Parse a JSON array of tactics from LLM response."""
    import json as json_mod
    text = text.strip()
    # Find JSON array in response
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            arr = json_mod.loads(text[start:end+1])
            if isinstance(arr, list):
                return [str(t) for t in arr if isinstance(t, str)]
        except (ValueError, json_mod.JSONDecodeError):
            pass
    # Fallback: split by lines
    return [line.strip().strip('"').strip("'") for line in text.split("\n")
            if line.strip() and not line.strip().startswith("#")]
