"""agent/tools/builtin/goal_inspect.py — Inspect current Lean4 proof goal state"""
from __future__ import annotations

import json
from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission


class GoalInspectTool(Tool):
    name = "goal_inspect"
    description = (
        "Inspect the current Lean4 goal state after applying partial tactics. "
        "Returns the remaining goals, local hypotheses, and available context."
    )
    permission = ToolPermission.EXTERNAL
    input_schema = {
        "type": "object",
        "properties": {
            "proof_so_far": {
                "type": "string",
                "description": "Lean4 code up to the point to inspect",
            },
            "theorem_header": {
                "type": "string",
                "description": "Full theorem statement header",
            },
        },
        "required": ["proof_so_far"],
    }

    def __init__(self, lean_pool=None):
        self._pool = lean_pool

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        if not self._pool:
            return ToolResult.error(
                "Lean REPL not available. Cannot inspect goals.")

        proof = input["proof_so_far"]
        header = input.get("theorem_header", ctx.theorem_statement)

        try:
            code = f"{header}\n{proof}"
            result = self._pool.check_proof(code, timeout=15)

            goals = result.get("goals", [])
            hypotheses = result.get("hypotheses", [])
            errors = result.get("errors", [])

            response = {
                "remaining_goals": goals,
                "goal_count": len(goals),
                "hypotheses": hypotheses,
                "errors": [str(e) for e in errors[:5]],
                "all_goals_closed": len(goals) == 0 and not errors,
            }
            return ToolResult.success(json.dumps(response, indent=2))
        except Exception as e:
            return ToolResult.error(f"Goal inspection failed: {e}")
