"""agent/tools/builtin/lean_auto.py — Try Lean4 automation tactics on current goal"""
from __future__ import annotations

import json
from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission


class LeanAutoTool(Tool):
    name = "lean_auto"
    description = (
        "Try a battery of Lean4 automation tactics (simp, ring, omega, etc.) "
        "on the current goal and report which ones succeed."
    )
    permission = ToolPermission.EXTERNAL
    input_schema = {
        "type": "object",
        "properties": {
            "goal_context": {
                "type": "string",
                "description": "Current proof context / goal state",
            },
        },
        "required": ["goal_context"],
    }

    TACTICS = [
        "exact?", "apply?", "simp", "ring", "linarith", "nlinarith",
        "omega", "norm_num", "positivity", "polyrith", "decide",
        "tauto", "aesop", "rfl",
    ]

    def __init__(self, lean_pool=None):
        self._pool = lean_pool

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        if not self._pool:
            return ToolResult.error("Lean REPL not available")

        successes = []
        for tactic in self.TACTICS:
            try:
                r = self._pool.try_tactic(tactic, timeout=15)
                if r and r.get("success"):
                    successes.append(tactic)
            except Exception:
                continue

        if successes:
            return ToolResult.success(json.dumps({
                "closing_tactics": successes,
                "message": f"Goal can be closed with: {successes[0]}",
            }))
        return ToolResult.success(json.dumps({
            "closing_tactics": [],
            "message": "No automation tactic closed the goal",
        }))
