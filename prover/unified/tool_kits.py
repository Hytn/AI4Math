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
                          knowledge_writer=None,
                          world_model=None,
                          retriever=None,
                          broadcast_bus=None,
                          search_state=None,
                          kimina_backend=None,
                          pantograph_backend=None,
                          lookeng_backend=None,
                          persistent_lemma_bank=None,
                          llm=None) -> ToolRegistry:
    """根据 profile.tools 组装一个 ToolRegistry.

    `persistent_lemma_bank` 是 v14 新参数 — 当 profile 含 LEMMA_BANK 或
    CONJECTURE_PROPOSE 时, 把 SQLite + BM25 引理库接到对应 tool。
    LemmaBankTool 走它做"跨问题/跨会话 lemma 检索"的 fallback 路径;
    ConjectureProposeTool 后置写入提议引理。不传则保持 v13 行为
    (LemmaBankTool 仅查 knowledge_store)。
    """
    registry = ToolRegistry()

    for kit in profile.tools:
        try:
            tool = _build_tool(kit,
                               lean_pool=lean_pool,
                               knowledge_store=knowledge_store,
                               knowledge_writer=knowledge_writer,
                               world_model=world_model,
                               retriever=retriever,
                               broadcast_bus=broadcast_bus,
                               search_state=search_state,
                               kimina_backend=kimina_backend,
                               pantograph_backend=pantograph_backend,
                               lookeng_backend=lookeng_backend,
                               persistent_lemma_bank=persistent_lemma_bank,
                               llm=llm,
                               integrity_strict=profile.integrity_strict)
            if tool is not None:
                registry.register(tool)
        except Exception as e:
            logger.warning(
                f"Failed to register tool '{kit.value}' for profile "
                f"'{profile.name}': {e}")

    return registry

def _build_tool(kit: ToolKit, *, lean_pool, knowledge_store,
                retriever, broadcast_bus, search_state,
                knowledge_writer=None,
                world_model=None,
                kimina_backend=None, pantograph_backend=None,
                lookeng_backend=None,
                persistent_lemma_bank=None,
                llm=None,
                integrity_strict: bool = False):
    """单个 ToolKit → 单个 Tool 实例.

    
    ConjectureProposeTool 的提议引理后置写入。"""

    if kit == ToolKit.LEAN_VERIFY:

        # whole-proof submissions get structured AgentFeedback (the
        # README-promised ~100 bits) instead of truncated stderr.
        # ErrorIntelligence works lazily; if its construction fails
        # (e.g. import error in a stripped environment), we fall back
        # to the tool's built-in minimal feedback path.
        ei = None
        try:
            from engine.error_intelligence import ErrorIntelligence
            ei = ErrorIntelligence(lean_pool=lean_pool)
        except Exception as _e:  # noqa: BLE001
            logger.debug(f"ErrorIntelligence unavailable: {_e}")
        return LeanVerifyTool(lean_pool=lean_pool, error_intelligence=ei,
                                integrity_strict=integrity_strict)

    if kit == ToolKit.TACTIC_APPLY:
        # 现有项目里没有这个工具 —— 我们在 builtin 下新增了 tactic_apply.py
        from agent.tools.builtin.tactic_apply import TacticApplyTool

        # best_first/beam) 也参与知识飞轮 —— 每次 tactic 应用都自动落 Layer 1.
        # 取 writer 的两条来源: (a) 调用方显式传入 (RL/eval 路径常用)
        #                       (b) knowledge_store 自己暴露 .writer 属性
        kw = knowledge_writer
        if kw is None and knowledge_store is not None:
            kw = getattr(knowledge_store, "writer", None)
        return TacticApplyTool(lean_pool=lean_pool,
                                search_state=search_state,
                                knowledge_writer=kw,
                                world_model=world_model)

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

        # ConjectureProposeTool. Without this, every call hit
        # ``GoalDecomposer(None).generate(...)``→AttributeError.
        return DecomposeSubgoalTool(llm=llm)

    if kit == ToolKit.LEMMA_BANK:
        from prover.unified.tools_extra import LemmaBankTool
        return LemmaBankTool(knowledge_store=knowledge_store,
                              persistent_bank=persistent_lemma_bank)

    if kit == ToolKit.BROADCAST:
        from prover.unified.tools_extra import BroadcastTool
        return BroadcastTool(bus=broadcast_bus)

    if kit == ToolKit.TREE_VIEW:
        from prover.unified.tools_extra import TreeViewTool
        return TreeViewTool(search_state=search_state)

    if kit == ToolKit.TREE_SELECT:
        from prover.unified.tools_extra import TreeSelectTool
        return TreeSelectTool(search_state=search_state)

    # ─ 基础设施大一统: 社区配套基建工具 ───────────────────
    if kit == ToolKit.BATCH_VERIFY:
        from prover.unified.tools_infra import BatchVerifyTool
        return BatchVerifyTool(kimina_backend=kimina_backend,
                                knowledge_store=knowledge_store)

    if kit == ToolKit.MVAR_FOCUS:
        from prover.unified.tools_infra import MVarFocusTool
        return MVarFocusTool(pantograph_backend=pantograph_backend)

    if kit == ToolKit.DRAFT_HOLE:
        from prover.unified.tools_infra import DraftHoleTool
        return DraftHoleTool(pantograph_backend=pantograph_backend)

    if kit == ToolKit.LEMMA_BY_LEMMA:
        from prover.unified.tools_infra import LemmaByLemmaTool
        return LemmaByLemmaTool(lookeng_backend=lookeng_backend)

    if kit == ToolKit.NL_EXISTENCE:
        from prover.unified.tools_infra import NLExistenceBridgeTool
        return NLExistenceBridgeTool()

    # ─ 
    if kit == ToolKit.CONJECTURE_PROPOSE:
        from prover.unified.tools_extra import ConjectureProposeTool
        # If the runner threaded its LLM through, bind it directly to
        # the tool — that's the cleanest path. If it didn't (older
        # callers, tests), the tool's execute() will pull from ctx
        # metadata at call time.

        # 写入跨问题 SQLite 库 (即使本次没证出来)。
        return ConjectureProposeTool(
            llm=llm, persistent_bank=persistent_lemma_bank)

    raise ValueError(f"Unknown ToolKit: {kit}")
