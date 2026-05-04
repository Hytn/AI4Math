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

        # v10: pool exposes verify_complete(theorem, proof, preamble),
        # not check_proof. Feature-detect to be defensive against any
        # mock-style pools tests might inject.
        try:
            verify = getattr(self._pool, "verify_complete", None)
            if verify is None:
                return ToolResult.error(
                    f"lean_pool does not expose verify_complete; "
                    f"got {type(self._pool).__name__}")
            import inspect as _inspect
            result = verify(header, proof, "")
            if _inspect.iscoroutine(result):
                result = await result

            # FullVerifyResult dataclass fields
            goals = list(getattr(result, "goals_remaining", []) or [])
            errors = list(getattr(result, "errors", []) or [])
            success = bool(getattr(result, "success", False))
            has_sorry = bool(getattr(result, "has_sorry", False))

            response = {
                "remaining_goals": goals,
                "goal_count": len(goals),
                "hypotheses": [],   # Not extractable from verify_complete;
                                    # add when REPL exposes a "context" cmd
                "errors": [str(e)[:300] for e in errors[:5]],
                "all_goals_closed": (success and not has_sorry
                                       and len(goals) == 0
                                       and not errors),
            }
            return ToolResult.success(json.dumps(response, indent=2,
                                                  ensure_ascii=False))
        except Exception as e:
            return ToolResult.error(f"Goal inspection failed: {e}")
