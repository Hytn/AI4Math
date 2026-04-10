"""agent/tools/builtin/__init__.py — Built-in tool collection

Register all built-in tools with a single call:

    from agent.tools.builtin import register_all_builtins
    register_all_builtins(registry, lean_pool=pool, knowledge_store=store)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from agent.tools.registry import ToolRegistry
from agent.tools.builtin.premise_search import PremiseSearchTool
from agent.tools.builtin.goal_inspect import GoalInspectTool
from agent.tools.builtin.tactic_suggest import TacticSuggestTool
from agent.tools.builtin.lean_verify import LeanVerifyTool
from agent.tools.builtin.cas_tool import CASTool
from agent.tools.builtin.lean_auto import LeanAutoTool
from agent.tools.builtin.file_read import FileReadTool

if TYPE_CHECKING:
    from knowledge.store import KnowledgeStore


def register_all_builtins(
    registry: ToolRegistry,
    lean_pool=None,
    knowledge_store=None,
    premise_db_path: str = "",
) -> list[str]:
    """Register all built-in tools and return their names."""
    tools = [
        PremiseSearchTool(
            knowledge_store=knowledge_store,
            premise_db_path=premise_db_path),
        GoalInspectTool(lean_pool=lean_pool),
        TacticSuggestTool(lean_pool=lean_pool),
        LeanVerifyTool(lean_pool=lean_pool),
        LeanAutoTool(lean_pool=lean_pool),
        CASTool(),
        FileReadTool(),
    ]
    names = []
    for tool in tools:
        registry.register(tool)
        names.append(tool.name)
    return names
