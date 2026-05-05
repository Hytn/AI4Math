"""prover/unified/search_driver.py — AgentLoop 形态的搜索包装

搜索代数(节点 / selection / backprop)的唯一权威实现在
``engine/search/``。本文件只提供 prover 主路径专属的两件事:

  * ``SharedSearchState``:在 ``SearchTree`` 之上加几个 LLM/dialog 友好的
    渲染方法 (``render_snapshot`` / ``to_search_tree_dict`` /
    ``solved_path_messages``),被 TacticApplyTool / TreeViewTool / runner
    直接调用。
  * ``make_driver``:把 selection 策略和 ``run_search`` 调度循环组合成
    跟旧 API 同形态的对象 (``await driver.run(expand_one_node=...)``)。

老的类名 ``TreeNode`` / ``BestFirstDriver`` / ``UCBDriver`` / ``BeamDriver``
保留为 alias,给测试与外部代码兼容。
"""
from __future__ import annotations

from typing import Optional, Callable, Awaitable

from engine.search.core import (
    SearchNode, SearchTree,
    extract_solved_path,
)
from engine.search.policies import (
    SelectionPolicy, BestFirstPolicy, UCBPolicy, BeamPolicy, make_policy,
)
from engine.search.runner import run_search

# ═══════════════════════════════════════════════════════════════════════
# 别名 — 旧名字 → 新统一类型
# ═══════════════════════════════════════════════════════════════════════

TreeNode = SearchNode

# ═══════════════════════════════════════════════════════════════════════
# Shared state — SearchTree + LLM/dialog 渲染方法
# ═══════════════════════════════════════════════════════════════════════

class SharedSearchState(SearchTree):
    """在 ``SearchTree`` 上增加 LLM 友好的渲染方法。"""

    def __init__(self, root_env_id: int, root_goals: list[str]):
        super().__init__(root_env_id=root_env_id, root_goals=root_goals)

    # ── 兼容旧 API ────────────────────────────────────────────────────

    def record_failure(self, node_id: int, tactic: str):
        self.mark_failed_tactic(node_id, tactic)

    # set_current 在父类已实现;此处补"找不到节点就抛"的严格版,与历史一致
    def set_current(self, node_id: int):
        if node_id not in self.nodes:
            raise KeyError(f"node {node_id} not in tree")
        self.current_node_id = node_id

    def env_id_for(self, node_id: int) -> int:
        """查指定节点的 Lean REPL env_id;不存在时退到根。"""
        n = self.nodes.get(node_id)
        return n.env_id if n is not None else self.nodes[0].env_id

    def current_env_id(self) -> int:
        """当前节点的 Lean REPL env_id。"""
        return self.env_id_for(self.current_node_id)

    # ── 渲染方法 (LLM 工具直接调) ──────────────────────────────────────

    def render_snapshot(self, *, node_id: Optional[int] = None,
                        depth: int = 3) -> dict:
        """LLM 友好的 JSON 表示。"""
        nid = node_id if node_id is not None else self.current_node_id
        node = self.nodes.get(nid)
        if node is None:
            return {"error": f"node {nid} not found"}
        ancestors = [
            {"node_id": a.id, "tactic": a.tactic, "depth": a.depth}
            for a in self.ancestors(nid)[-depth:]
        ]
        siblings = []
        if node.parent_id is not None:
            parent = self.nodes[node.parent_id]
            for ch_id in parent.children:
                if ch_id == nid:
                    continue
                ch = self.nodes[ch_id]
                siblings.append({
                    "node_id": ch.id, "tactic": ch.tactic,
                    "status": ch.status, "num_goals": len(ch.goals),
                })
        leaves = sorted(self.open_leaves(),
                        key=lambda i: -self.nodes[i].score)[:5]
        return {
            "current_node_id": nid,
            "current_goals": node.goals,
            "num_open_goals": len(node.goals),
            "depth": node.depth,
            "path_from_root": ancestors,
            "sibling_branches": siblings,
            "top_open_leaves": [
                {"node_id": i, "score": self.nodes[i].score,
                 "depth": self.nodes[i].depth}
                for i in leaves
            ],
            "failed_at_this_node": sorted(node.failed_tactics),
        }

    def to_search_tree_dict(self, *, kind: str) -> dict:
        """序列化整棵树到 ``meta.search_tree`` 的形状。"""
        nodes_out = []
        for nid in sorted(self.nodes):
            n = self.nodes[nid]
            nodes_out.append({
                "node_id": n.id,
                "parent_id": n.parent_id,
                "tactic": n.tactic,
                "depth": n.depth,
                "status": n.status,
                "visit_count": n.visit_count,
                "success_count": n.success_count,
                "score": float(n.score),
                "num_goals": len(n.goals),
                "is_complete": bool(n.is_complete),
                "failed_tactics": sorted(n.failed_tactics),
                "messages": list(n.messages),
            })
        max_depth = max((n.depth for n in self.nodes.values()), default=0)
        return {
            "kind": kind,
            "root_node_id": 0,
            "solved_node_id": self.solved_node_id,
            "total_nodes": len(self.nodes),
            "max_depth": max_depth,
            "nodes": nodes_out,
        }

    def solved_path_messages(self) -> list[dict]:
        """根→解节点 messages 链;失败时退而求其次取 (depth, score) 最大节点。"""
        return extract_solved_path(self)

# ═══════════════════════════════════════════════════════════════════════
# Driver — selection policy + run_search 的组合,提供老 API 形状
# ═══════════════════════════════════════════════════════════════════════

class _DriverWrapper:
    """对外保留 ``await driver.run(expand_one_node=...)`` 的形状。

    内部是 selection policy + run_search。
    """

    def __init__(self, state: SharedSearchState,
                 policy: SelectionPolicy, *,
                 max_nodes: int, max_depth: int,
                 expansion_max_turns: int):
        self.state = state
        self.policy = policy
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.expansion_max_turns = expansion_max_turns

    async def run(self, *, expand_one_node: Callable[..., Awaitable]):
        """主循环。``expand_one_node(node_id=, max_turns=)`` 由 runner 提供。"""

        async def _expander(tree: SearchTree, nid: int):
            await expand_one_node(
                node_id=nid, max_turns=self.expansion_max_turns)

        await run_search(
            self.state, self.policy, _expander,
            max_nodes=self.max_nodes,
            max_depth=self.max_depth,
            score_bump_on_success=0.0,    # prover 端用 score_new_node 重打
            rescore_new_nodes=True,
        )
        return self.state

# 老类名作为指向 _DriverWrapper 的工厂别名 — 单元测试/外部代码 import 用
class BestFirstDriver(_DriverWrapper):
    def __init__(self, state, *, max_nodes, max_depth, expansion_max_turns,
                 **_kwargs):
        super().__init__(state, BestFirstPolicy(),
                         max_nodes=max_nodes, max_depth=max_depth,
                         expansion_max_turns=expansion_max_turns)

class UCBDriver(_DriverWrapper):
    def __init__(self, state, *, max_nodes, max_depth, expansion_max_turns,
                 ucb_c: float = 1.414, **_kwargs):
        super().__init__(state, UCBPolicy(c=ucb_c),
                         max_nodes=max_nodes, max_depth=max_depth,
                         expansion_max_turns=expansion_max_turns)

class BeamDriver(_DriverWrapper):
    def __init__(self, state, *, max_nodes, max_depth, expansion_max_turns,
                 beam_width: int = 8, **_kwargs):
        super().__init__(state, BeamPolicy(beam_width=beam_width),
                         max_nodes=max_nodes, max_depth=max_depth,
                         expansion_max_turns=expansion_max_turns)

def make_driver(kind: str, state: SharedSearchState, *,
                max_nodes: int, max_depth: int,
                expansion_max_turns: int,
                beam_width: int = 8,
                ucb_c: float = 1.414):
    """工厂。返回 ``_DriverWrapper`` (旧调用方按 ``await d.run(expand_one_node=)`` 用)。"""
    policy = make_policy(kind, beam_width=beam_width, ucb_c=ucb_c)
    return _DriverWrapper(
        state, policy,
        max_nodes=max_nodes, max_depth=max_depth,
        expansion_max_turns=expansion_max_turns,
    )
