"""prover/unified/tool_kits.py — 按 Profile 的 tools 列表组装 ToolRegistry

每个 ToolKit 枚举值对应一个真实工具的注册函数。本模块的责任是:

  Profile.tools=[LEAN_VERIFY, TACTIC_APPLY, ...] →  ToolRegistry 实例

工具的 description 在这里被精心措辞 —— 这是另一个关键的设计杠杆,
因为 LLM 的工具调用倾向几乎完全取决于 description 的措辞。
"""
from __future__ import annotations
import logging

from agent.tools.registry import ToolRegistry
from agent.tools.builtin.lean_verify import LeanVerifyTool
from agent.tools.builtin.goal_inspect import GoalInspectTool
from agent.tools.builtin.tactic_suggest import TacticSuggestTool
from agent.tools.builtin.lean_auto import LeanAutoTool
from agent.tools.builtin.premise_search import PremiseSearchTool
from agent.tools.builtin.cas_tool import CASTool

from prover.unified.profiles import Profile, ToolKit

logger = logging.getLogger(__name__)


def build_tool_registry(profile: Profile, *,
                          lean_pool=None,
                          knowledge_store=None,
                          retriever=None,
                          broadcast_bus=None,
                          search_state=None) -> ToolRegistry:
    """根据 profile.tools 组装一个 ToolRegistry.

    `search_state` 仅在 profile.search.kind != "none" 时由 runner 注入,
    用来让 LLM 通过 tree_view / tree_select 工具看到/操作搜索树。
    """
    registry = ToolRegistry()

    for kit in profile.tools:
        try:
            tool = _build_tool(kit,
                               lean_pool=lean_pool,
                               knowledge_store=knowledge_store,
                               retriever=retriever,
                               broadcast_bus=broadcast_bus,
                               search_state=search_state)
            if tool is not None:
                registry.register(tool)
        except Exception as e:
            logger.warning(
                f"Failed to register tool '{kit.value}' for profile "
                f"'{profile.name}': {e}")

    return registry


def _build_tool(kit: ToolKit, *, lean_pool, knowledge_store,
                retriever, broadcast_bus, search_state):
    """单个 ToolKit → 单个 Tool 实例."""

    if kit == ToolKit.LEAN_VERIFY:
        return LeanVerifyTool(lean_pool=lean_pool)

    if kit == ToolKit.TACTIC_APPLY:
        # 现有项目里没有这个工具 —— 我们在 builtin 下新增了 tactic_apply.py
        from agent.tools.builtin.tactic_apply import TacticApplyTool
        return TacticApplyTool(lean_pool=lean_pool, search_state=search_state)

    if kit == ToolKit.GOAL_INSPECT:
        return GoalInspectTool(lean_pool=lean_pool)

    if kit == ToolKit.TACTIC_SUGGEST:
        return TacticSuggestTool(lean_pool=lean_pool)

    if kit == ToolKit.LEAN_AUTO:
        return LeanAutoTool(lean_pool=lean_pool)

    if kit == ToolKit.PREMISE_SEARCH:
        # PremiseSearchTool signature: (knowledge_store, premise_db_path)
        # `retriever` from runner is typically a PremiseSelector-like object.
        # If it has a knowledge_store attribute, use it; otherwise pass directly.
        ks = knowledge_store
        if ks is None and retriever is not None:
            ks = getattr(retriever, "store", None) or getattr(retriever, "knowledge_store", None)
        return PremiseSearchTool(knowledge_store=ks)

    if kit == ToolKit.CAS:
        return CASTool()

    if kit == ToolKit.DECOMPOSE:
        from prover.unified.tools_extra import DecomposeSubgoalTool
        return DecomposeSubgoalTool()

    if kit == ToolKit.LEMMA_BANK:
        from prover.unified.tools_extra import LemmaBankTool
        return LemmaBankTool(knowledge_store=knowledge_store)

    if kit == ToolKit.BROADCAST:
        from prover.unified.tools_extra import BroadcastTool
        return BroadcastTool(bus=broadcast_bus)

    if kit == ToolKit.TREE_VIEW:
        from prover.unified.tools_extra import TreeViewTool
        return TreeViewTool(search_state=search_state)

    if kit == ToolKit.TREE_SELECT:
        from prover.unified.tools_extra import TreeSelectTool
        return TreeSelectTool(search_state=search_state)

    raise ValueError(f"Unknown ToolKit: {kit}")
