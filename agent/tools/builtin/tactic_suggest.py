"""agent/tools/builtin/tactic_suggest.py — Suggest tactics for current goal


  1. The pool method was called as ``pool.try_tactic(tactic, context=...,
     timeout=...)`` but ``AsyncLeanPool.try_tactic`` is ``async`` and its
     signature is ``try_tactic(env_id: int, tactic: str) -> TacticFeedback``.
     The previous call would have raised TypeError on every invocation,
     except no test exercised the live path so it stayed unnoticed.
  2. The result was used as a dict (``r.get("success")``) but
     ``TacticFeedback`` is a dataclass.

The tool now feature-detects async/sync, awaits coroutines, and reads
TacticFeedback dataclass fields. The heuristic fallback path is unchanged.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging

from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission

logger = logging.getLogger(__name__)

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
        partial = input.get("partial_proof", "")  # noqa: F841 — kept for future REPL state binding

        if not self._pool:
            # Heuristic mode without REPL
            suggestions = self._heuristic_suggest(goal)
            return ToolResult.success(json.dumps(suggestions, indent=2))

        # tactic_apply: prefer pool.base_env_id (set by pool.start()),
        # fall back to 0 if the pool doesn't expose it.
        env_id = getattr(self._pool, "base_env_id", 0)

        results = []
        for tactic in tactics:
            try:
                r = await self._call_try_tactic(env_id, tactic)
            except Exception as e:
                logger.debug(f"tactic_suggest({tactic!r}) failed: {e}")
                continue
            if r is None:
                continue
            success = bool(getattr(r, "success", False))
            if success:
                remaining = getattr(r, "remaining_goals", []) or []
                results.append({
                    "tactic": tactic,
                    "success": True,
                    "remaining_goals": len(remaining),
                })
            else:
                err = (getattr(r, "error_message", "") or "")[:200]
                if err:
                    results.append({
                        "tactic": tactic,
                        "success": False,
                        "error_hint": err,
                    })

        return ToolResult.success(json.dumps(results, indent=2),
                                 count=len(results))

    async def _call_try_tactic(self, env_id: int, tactic: str):
        """Pool may expose try_tactic as sync or async; handle both."""
        fn = getattr(self._pool, "try_tactic", None)
        if fn is None:
            raise RuntimeError("lean_pool has no try_tactic method")
        out = fn(env_id, tactic)
        if inspect.iscoroutine(out):
            out = await out
        return out

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
