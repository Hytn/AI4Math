"""agent/tools/tool_executor.py — 工具调用解析与执行"""
from __future__ import annotations
import logging
from agent.tools.tool_registry import get_registry

logger = logging.getLogger(__name__)

class ToolExecutor:
    def __init__(self):
        self.registry = get_registry()

    def execute(self, tool_name: str, tool_input: dict) -> str:
        try:
            fn = self.registry.get(tool_name)
            result = fn(**tool_input)
            return str(result)
        except KeyError:
            return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error: {e}"

    def execute_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        results = []
        for call in tool_calls:
            result = self.execute(call["name"], call.get("input", {}))
            results.append({"tool_use_id": call.get("id", ""), "content": result})
        return results
