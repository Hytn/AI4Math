"""
engine/llm — LLM Tactic Suggestion Engine

Integrates with Claude API to suggest tactics based on goal state.
The LLM sees a token-efficient GoalView and returns candidate tactics.
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
- Available tactics: intro, assumption, apply, exact, sorry, cases, induction, simp, rfl, trivial
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
    """Suggests tactics using an LLM."""

    def __init__(self, use_api: bool = True):
        self.use_api = use_api
        self._call_count = 0

    def suggest(self, goal_view: dict, max_suggestions: int = 6) -> LLMSuggestion:
        """Get tactic suggestions from LLM."""
        t0 = time.time()

        if self.use_api:
            return self._suggest_api(goal_view, max_suggestions, t0)
        else:
            return self._suggest_heuristic(goal_view, max_suggestions, t0)

    def _suggest_api(self, goal_view: dict, max_suggestions: int,
                     t0: float) -> LLMSuggestion:
        """Call Claude API for tactic suggestions."""
        try:
            import json as json_mod

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

            response = _call_claude_api(prompt)

            # Parse JSON array from response
            tactics = _parse_tactic_response(response.get("text", "[]"))
            elapsed = (time.time() - t0) * 1000

            return LLMSuggestion(
                tactics=tactics[:max_suggestions],
                reasoning=response.get("text", ""),
                elapsed_ms=elapsed,
                model=response.get("model", "claude"),
                tokens_in=response.get("tokens_in", 0),
                tokens_out=response.get("tokens_out", 0),
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
        tactics.extend(["trivial", "simp", "sorry"])

        elapsed = (time.time() - t0) * 1000
        return LLMSuggestion(
            tactics=tactics[:max_suggestions],
            reasoning="heuristic",
            elapsed_ms=elapsed,
            model="heuristic",
        )


def _call_claude_api(prompt: str) -> dict:
    """Call the Anthropic API."""
    import urllib.request
    import json as json_mod

    body = json_mod.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": "",  # handled by environment
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json_mod.loads(resp.read())
            text = data.get("content", [{}])[0].get("text", "[]")
            return {
                "text": text,
                "model": data.get("model", ""),
                "tokens_in": data.get("usage", {}).get("input_tokens", 0),
                "tokens_out": data.get("usage", {}).get("output_tokens", 0),
            }
    except Exception as e:
        return {"text": "[]", "error": str(e)}


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
        except:
            pass
    # Fallback: split by lines
    return [line.strip().strip('"').strip("'") for line in text.split("\n")
            if line.strip() and not line.strip().startswith("#")]
