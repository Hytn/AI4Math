"""agent/tools — Tool system for AI4Math agents

Provides:
  - Tool base class with schema, permissions, validation
  - ToolRegistry for registration and discovery
  - Built-in tools for premise search, goal inspection, verification, etc.
  - Legacy backward-compatible API
"""
from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission
from agent.tools.registry import ToolRegistry, get_registry, register_tool

__all__ = [
    "Tool", "ToolContext", "ToolResult", "ToolPermission",
    "ToolRegistry", "get_registry", "register_tool",
]
