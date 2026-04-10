"""agent/tools/tool_registry.py — 向后兼容 shim

新代码请直接使用:
    from agent.tools.registry import ToolRegistry

本文件仅保留旧 import 路径的兼容性。
"""
from agent.tools.registry import ToolRegistry, LegacyToolAdapter  # noqa: F401

_global_registry = ToolRegistry()


def register_tool(name: str, description: str, parameters: dict):
    """Decorator to register a plain function as a tool."""
    def decorator(fn):
        adapter = LegacyToolAdapter(name, fn, description, parameters)
        _global_registry.register(adapter)
        return fn
    return decorator


def get_registry() -> ToolRegistry:
    return _global_registry
