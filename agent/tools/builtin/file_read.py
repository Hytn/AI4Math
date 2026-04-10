"""agent/tools/builtin/file_read.py — Read Lean4/Mathlib source files"""
from __future__ import annotations

import os
from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission


class FileReadTool(Tool):
    name = "file_read"
    description = (
        "Read a Lean4 source file to understand definitions, lemma statements, "
        "or proof patterns. Useful for inspecting Mathlib internals."
    )
    permission = ToolPermission.READ_ONLY
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the Lean4 file (relative to project root)",
            },
            "line_start": {"type": "integer", "description": "Start line (1-based)"},
            "line_end": {"type": "integer", "description": "End line (1-based)"},
            "search_term": {
                "type": "string",
                "description": "If set, show only lines containing this term",
            },
        },
        "required": ["path"],
    }

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        path = input["path"]

        # Security: prevent path traversal
        if ".." in path or path.startswith("/"):
            return ToolResult.error("Path traversal not allowed")

        full_path = os.path.join(ctx.working_dir or ".", path)
        if not os.path.isfile(full_path):
            return ToolResult.error(f"File not found: {path}")

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            return ToolResult.error(f"Cannot read file: {e}")

        start = input.get("line_start", 1) - 1
        end = input.get("line_end", len(lines))
        search = input.get("search_term", "")

        selected = lines[max(0, start):min(end, len(lines))]

        if search:
            selected = [l for l in selected if search in l]

        content = "".join(selected[:200])  # Cap at 200 lines
        if len(selected) > 200:
            content += f"\n... ({len(selected) - 200} more lines)"

        return ToolResult.success(content)
