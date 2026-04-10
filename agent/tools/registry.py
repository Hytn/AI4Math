"""agent/tools/registry.py — Tool registry with schema discovery and filtering

Rewritten to support the new Tool protocol. Backward-compatible with
legacy function-based tools via LegacyToolAdapter.

Usage::

    registry = ToolRegistry()
    registry.register(PremiseSearchTool())
    registry.register(GoalInspectTool())

    # Get tools available for a specific agent
    schemas = registry.to_claude_tools_schema(
        allowed=["premise_search", "goal_inspect"],
        permission_filter={ToolPermission.READ_ONLY, ToolPermission.EXTERNAL})

    # Execute a tool call
    result = await registry.execute("premise_search",
                                    {"query": "add_comm"}, ctx)
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from agent.tools.base import (
    Tool, ToolContext, ToolResult, ToolPermission,
)

logger = logging.getLogger(__name__)


class LegacyToolAdapter(Tool):
    """Wrap a plain function as a Tool for backward compatibility."""

    def __init__(self, name: str, fn: Callable, description: str,
                 parameters: dict):
        self.name = name
        self.description = description
        self.input_schema = parameters
        self.permission = ToolPermission.EXTERNAL
        self._fn = fn

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        try:
            result = self._fn(**input)
            return ToolResult.success(str(result))
        except Exception as e:
            return ToolResult.error(str(e))


class ToolRegistry:
    """Central registry for all tools.

    Features over the old registry:
    - Tools are proper objects with schema, permissions, validation
    - Supports filtering by name whitelist and permission level
    - Async execution with error handling
    - Schema discovery for LLM tool-use
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    # ── Registration ──

    def register(self, tool: Tool) -> None:
        """Register a Tool instance."""
        if not tool.name:
            raise ValueError(f"Tool must have a name: {tool}")
        if tool.name in self._tools:
            logger.warning(f"Overwriting tool '{tool.name}'")
        self._tools[tool.name] = tool

    def register_function(self, name: str, fn: Callable,
                          description: str = "",
                          parameters: dict = None) -> None:
        """Register a plain function (backward compatibility)."""
        self.register(LegacyToolAdapter(
            name=name, fn=fn,
            description=description or fn.__doc__ or "",
            parameters=parameters or {"type": "object", "properties": {}}))

    def unregister(self, name: str) -> None:
        """Remove a tool."""
        self._tools.pop(name, None)

    # ── Lookup ──

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_by_permission(self, perm: ToolPermission) -> list[Tool]:
        """List tools matching a permission level."""
        return [t for t in self._tools.values() if t.permission == perm]

    # ── Schema generation ──

    def to_claude_tools_schema(
        self,
        allowed: list[str] = None,
        permission_filter: set[ToolPermission] = None,
    ) -> list[dict]:
        """Generate Claude API tool schemas.

        Args:
            allowed: If set, only include these tool names
            permission_filter: If set, only include tools with these permissions

        Returns:
            List of Claude API tool definitions
        """
        schemas = []
        for name, tool in self._tools.items():
            if allowed and name not in allowed:
                continue
            if permission_filter and tool.permission not in permission_filter:
                continue
            schemas.append(tool.to_claude_schema())
        return schemas

    # ── Execution ──

    async def execute(self, name: str, input: dict,
                      ctx: ToolContext) -> ToolResult:
        """Execute a tool by name with full safety checks."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult.error(
                f"Unknown tool '{name}'. Available: {self.list_tools()}")
        return await tool.safe_execute(input, ctx)

    def execute_sync(self, name: str, input: dict) -> str:
        """Synchronous execution (backward compatibility).

        Creates a minimal ToolContext and runs in asyncio.
        """
        import asyncio
        ctx = ToolContext()
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self.execute(name, input, ctx), loop)
            result = future.result(timeout=30)
        except RuntimeError:
            result = asyncio.run(self.execute(name, input, ctx))
        return result.content

    # ── Batch execution ──

    async def execute_tool_calls(
        self,
        tool_calls: list[dict],
        ctx: ToolContext,
    ) -> list[dict]:
        """Execute multiple tool calls, return results in Claude format.

        Args:
            tool_calls: List of {"id": ..., "name": ..., "input": ...}
            ctx: Shared execution context

        Returns:
            List of {"type": "tool_result", "tool_use_id": ..., "content": ...}
        """
        results = []
        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_input = call.get("input", {})
            tool_id = call.get("id", "")

            result = await self.execute(tool_name, tool_input, ctx)
            results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result.content,
                "is_error": result.is_error,
            })
        return results

    def __len__(self):
        return len(self._tools)

    def __repr__(self):
        return f"ToolRegistry({len(self._tools)} tools: {self.list_tools()})"


# ── Global registry (backward compatible) ──

_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(name: str, description: str, parameters: dict):
    """Decorator to register a function as a tool (backward compatible)."""
    def decorator(fn):
        get_registry().register_function(
            name=name, fn=fn, description=description, parameters=parameters)
        return fn
    return decorator
