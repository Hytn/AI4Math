"""agent/tools/tool_registry.py — 工具注册中心"""
from __future__ import annotations
from typing import Callable, Any

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, fn: Callable, description: str, parameters: dict):
        self._tools[name] = {"fn": fn, "description": description,
                             "parameters": parameters, "name": name}

    def get(self, name: str) -> Callable:
        return self._tools[name]["fn"]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def to_claude_tools_schema(self) -> list[dict]:
        return [{"name": t["name"], "description": t["description"],
                 "input_schema": t["parameters"]} for t in self._tools.values()]

_global_registry = ToolRegistry()

def register_tool(name: str, description: str, parameters: dict):
    def decorator(fn):
        _global_registry.register(name, fn, description, parameters)
        return fn
    return decorator

def get_registry() -> ToolRegistry:
    return _global_registry
