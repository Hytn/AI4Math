"""agent/tools/builtin/cas_tool.py — Computer Algebra System bridge"""
from __future__ import annotations

import subprocess
from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission


class CASTool(Tool):
    name = "cas_evaluate"
    description = (
        "Evaluate a mathematical expression using SageMath. Useful for "
        "checking numerical conjectures, computing specific values, "
        "or finding counterexamples."
    )
    permission = ToolPermission.EXTERNAL
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "SageMath expression to evaluate",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 30)",
            },
        },
        "required": ["expression"],
    }

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        expr = input["expression"]
        timeout = input.get("timeout", 30)
        try:
            result = subprocess.run(
                ["sage", "-c", f"print({expr})"],
                capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return ToolResult.success(result.stdout.strip())
            return ToolResult.error(result.stderr[:300])
        except FileNotFoundError:
            return ToolResult.error("SageMath not installed")
        except subprocess.TimeoutExpired:
            return ToolResult.error(f"CAS timeout after {timeout}s")
