"""agent/tools/builtin/lean_auto.py — Try Lean4 automation tactics on current goal"""
from __future__ import annotations

import asyncio
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

        proof_state_id = ctx.shared_state.get("proof_state_id")
        if proof_state_id is None:
            return ToolResult.success(json.dumps({
                "closing_tactics": [],
                "mode": "unavailable",
                "message": (
                    "No live proof_state_id is available; automation tactics "
                    "were not executed."),
            }))
        env_id = int(proof_state_id)
        successes = []
        for tactic in self.TACTICS:
            try:
                fn = self._pool.try_tactic
                if asyncio.iscoroutinefunction(fn):
                    r = await fn(env_id, tactic)
                else:
                    r = fn(env_id, tactic)
                ok = bool(getattr(r, "success", False))
                if isinstance(r, dict):
                    ok = bool(r.get("success"))
                if ok:
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
