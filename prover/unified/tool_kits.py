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
                          llm=None) -> ToolRegistry:
    """根据 profile.tools 组装一个 ToolRegistry.

    `search_state` 仅在 profile.search.kind != "none" 时由 runner 注入,
    用来让 LLM 通过 tree_view / tree_select 工具看到/操作搜索树。

    `knowledge_writer` 是 v4 新参数 — 当 profile 含 TACTIC_APPLY 时,
    每次 tactic 应用都会自动 deposit 到 Layer 1. 不传也能跑, 缺省走
    knowledge_store.writer (如果该属性存在).

    `world_model` 是 v4 新参数 — 当 profile 含 TACTIC_APPLY 时,
    每次 tactic 应用前先问 world model "这个 tactic 大概率会失败吗?",
    高置信度 (默认 ≥ 0.85) 预测失败的会被 short-circuit, 不发给 Lean.
    传 ``MockWorldModel`` 就是启发式; 传 ``TrainedWorldModel("x.pkl")``
    就用训练过的 sklearn 分类器; 不传则不做 gating.

    `kimina_backend` / `pantograph_backend` / `lookeng_backend` 由 runner
    在选择对应 backend 时注入, 启用基础设施大一统的工具. 当对应 backend
    未配置时, 工具会以 fallback 模式注册 — 其 execute() 返回结构化错误,
    不会破坏 agent loop。

    `llm` 是 v6 新参数 — 当 profile 含 CONJECTURE_PROPOSE 时, 工具需要
    LLM 调用做猜想生成. 不传也能跑 (执行时会从 ctx.shared_state['llm']
    兜底获取); 显式传入则路径更直接.
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
                               llm=llm)
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
                llm=None):
    """单个 ToolKit → 单个 Tool 实例."""

    if kit == ToolKit.LEAN_VERIFY:
        return LeanVerifyTool(lean_pool=lean_pool)

    if kit == ToolKit.TACTIC_APPLY:
        # 现有项目里没有这个工具 —— 我们在 builtin 下新增了 tactic_apply.py
        from agent.tools.builtin.tactic_apply import TacticApplyTool
        # v4: 连上 KnowledgeWriter, 让步级 profile (reprover/leandojo/mcts/
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

    # ─ v6: 猜想驱动 ──────────────────────────────────────────
    if kit == ToolKit.CONJECTURE_PROPOSE:
        from prover.unified.tools_extra import ConjectureProposeTool
        # If the runner threaded its LLM through, bind it directly to
        # the tool — that's the cleanest path. If it didn't (older
        # callers, tests), the tool's execute() will pull from ctx
        # metadata at call time.
        return ConjectureProposeTool(llm=llm)

    raise ValueError(f"Unknown ToolKit: {kit}")
