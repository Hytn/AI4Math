"""agent/tools/builtin/tactic_suggest.py — Suggest tactics for current goal"""
from __future__ import annotations

import json
from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission


class TacticSuggestTool(Tool):
    name = "tactic_suggest"
    description = (
        "Given a Lean4 goal state, suggest tactics that might close it. "
        "Tries exact?, apply?, simp, ring, omega etc. via REPL and reports "
        "which ones succeed."
    )
    permission = ToolPermission.EXTERNAL
    input_schema = {
        "type": "object",
        "properties": {
            "goal_state": {
                "type": "string",
                "description": "Current Lean4 goal state text",
            },
            "partial_proof": {
                "type": "string",
                "description": "Proof code so far (to set up the goal)",
            },
            "tactics_to_try": {
                "type": "array",
                "description": "Specific tactics to try (default: common set)",
            },
        },
        "required": ["goal_state"],
    }

    AUTO_TACTICS = [
        "exact?", "apply?", "simp", "ring", "linarith", "nlinarith",
        "omega", "norm_num", "positivity", "decide", "tauto", "aesop",
        "rfl", "trivial", "contradiction",
    ]

    def __init__(self, lean_pool=None):
        self._pool = lean_pool

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        tactics = input.get("tactics_to_try", self.AUTO_TACTICS)
        goal = input["goal_state"]
        partial = input.get("partial_proof", "")

        if not self._pool:
            # Heuristic mode without REPL
            suggestions = self._heuristic_suggest(goal)
            return ToolResult.success(json.dumps(suggestions, indent=2))

        results = []
        for tactic in tactics:
            try:
                r = self._pool.try_tactic(tactic, context=partial, timeout=10)
                if r and r.get("success"):
                    results.append({
                        "tactic": tactic,
                        "success": True,
                        "remaining_goals": r.get("remaining_goals", 0),
                    })
                else:
                    error = r.get("error", "") if r else ""
                    if error and len(error) < 200:
                        results.append({
                            "tactic": tactic,
                            "success": False,
                            "error_hint": error[:200],
                        })
            except Exception:
                continue

        return ToolResult.success(json.dumps(results, indent=2),
                                 count=len(results))

    def _heuristic_suggest(self, goal: str) -> list[dict]:
        """Heuristic suggestions without REPL."""
        gl = goal.lower()
        suggestions = []
        if "=" in goal and ("+" in goal or "*" in goal):
            suggestions.extend(["ring", "omega", "simp"])
        if "≤" in goal or "≥" in goal or "<" in goal or ">" in goal:
            suggestions.extend(["linarith", "omega"])
        if "∀" in goal or "∃" in goal:
            suggestions.extend(["intro", "use", "constructor"])
        if "¬" in goal or "False" in goal:
            suggestions.extend(["contradiction", "push_neg"])
        if "nat" in gl or "ℕ" in goal:
            suggestions.extend(["omega", "norm_num", "simp"])
        if not suggestions:
            suggestions = ["simp", "exact?", "apply?", "aesop"]
        return [{"tactic": t, "confidence": "heuristic"} for t in suggestions]
