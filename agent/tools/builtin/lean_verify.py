"""agent/tools/builtin/lean_verify.py — Verify a Lean4 proof snippet"""
from __future__ import annotations

import json
from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission


class LeanVerifyTool(Tool):
    name = "lean_verify"
    description = (
        "Submit a complete Lean4 proof for verification. Returns structured "
        "feedback: success/failure, error messages, remaining goals, and "
        "specific repair suggestions."
    )
    permission = ToolPermission.EXTERNAL
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Lean4 code to verify",
            },
            "quick_check": {
                "type": "boolean",
                "description": "If true, use fast L1 check only (default false)",
            },
        },
        "required": ["code"],
    }

    def __init__(self, lean_pool=None):
        self._pool = lean_pool

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        code = input["code"]
        quick = input.get("quick_check", False)

        if not self._pool:
            # Syntax-only check
            from engine.prefilter import SyntaxPrefilter
            pf = SyntaxPrefilter()
            passed, reason = pf.check(code)
            return ToolResult.success(json.dumps({
                "verified": False,
                "syntax_ok": passed,
                "message": reason or "Syntax OK (REPL unavailable for full check)",
            }))

        try:
            timeout = 10 if quick else 30
            result = self._pool.check_proof(code, timeout=timeout)

            response = {
                "verified": result.get("success", False),
                "goals_remaining": result.get("goals", []),
                "errors": [str(e)[:300] for e in result.get("errors", [])[:5]],
                "sorry_free": "sorry" not in code and "admit" not in code,
            }
            return ToolResult.success(json.dumps(response, indent=2))
        except Exception as e:
            return ToolResult.error(f"Verification failed: {e}")
