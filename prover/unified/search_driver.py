"""prover/unified/search_driver.py — 搜索算法 = AgentLoop 的外壳

设计原则:
  • AgentLoop 永远是核心. driver 不替换 loop, 只调度它.
  • driver 做"算法部分": selection (UCB/best-first) + backprop + 终止.
  • LLM 只做 expansion: driver 选中节点 → 实例化一个 max_turns 较小的
    AgentLoop, anchor 在该节点 → 让 LLM 提出下一步 tactic.
  • SharedSearchState 是 driver 与 LLM tools (TacticApplyTool / TreeViewTool)
    共享的对象 —— LLM 通过工具调用间接修改它, driver 通过它做 selection.

这样 "主管线 = agent loop" 的统一性成立: 所有算法都跑同一个 loop,
只是 (max_turns, tools, system_prompt) 不同, 外加可选的 driver 调度。
"""
from __future__ import annotations
import asyncio
import heapq
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Shared Search State — 树 + 节点-环境映射
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TreeNode:
    id: int
    parent_id: Optional[int]
    tactic: Optional[str]              # tactic that led here from parent
    env_id: int                        # Lean REPL env corresponding to this node
    goals: list[str] = field(default_factory=list)
    is_complete: bool = False
    depth: int = 0
    children: list[int] = field(default_factory=list)
    visit_count: int = 0
    success_count: int = 0
    score: float = 0.0
    failed_tactics: set = field(default_factory=set)  # 在此节点已失败的 tactic
    status: str = "open"               # open | solved | failed | pruned


class SharedSearchState:
    """Driver 与 Tools 共享的可变状态对象.

    被 TacticApplyTool / TreeViewTool / TreeSelectTool 注入引用. driver
    通过 select() / backprop() 操作它来执行 UCB / best-first 算法。
    """

    def __init__(self, root_env_id: int, root_goals: list[str]):
        root = TreeNode(id=0, parent_id=None, tactic=None,
                        env_id=root_env_id, goals=list(root_goals),
                        depth=0)
        self.nodes: dict[int, TreeNode] = {0: root}
        self._next_id = 1
        self.current_node_id = 0
        self.solved_node_id: Optional[int] = None

    # ── operations called from inside tools ─────────────────────────────

    def expand(self, *, parent_node_id: Optional[int],
                 tactic: str, new_env_id: int,
                 remaining_goals: list[str],
                 is_complete: bool) -> int:
        parent_id = parent_node_id if parent_node_id is not None \
            else self.current_node_id
        parent = self.nodes.get(parent_id)
        if parent is None:
            parent = self.nodes[0]
            parent_id = 0
        node = TreeNode(
            id=self._next_id,
            parent_id=parent_id,
            tactic=tactic,
            env_id=new_env_id,
            goals=list(remaining_goals),
            is_complete=is_complete,
            depth=parent.depth + 1,
            status="solved" if is_complete else "open",
        )
        self.nodes[node.id] = node
        parent.children.append(node.id)
        self._next_id += 1
        if is_complete and self.solved_node_id is None:
            self.solved_node_id = node.id
        return node.id

    def env_id_for(self, node_id: int) -> int:
        return self.nodes[node_id].env_id

    def current_env_id(self) -> int:
        return self.nodes[self.current_node_id].env_id

    def set_current(self, node_id: int):
        if node_id not in self.nodes:
            raise KeyError(f"node {node_id} not in tree")
        self.current_node_id = node_id

    def record_failure(self, node_id: int, tactic: str):
        node = self.nodes.get(node_id)
        if node:
            node.failed_tactics.add(tactic)

    # ── helpers used by drivers ─────────────────────────────────────────

    def open_leaves(self) -> list[int]:
        return [n.id for n in self.nodes.values()
                if n.status == "open" and not n.children]

    def ancestors(self, node_id: int) -> list[TreeNode]:
        path = []
        cur = self.nodes.get(node_id)
        while cur is not None:
            path.append(cur)
            cur = self.nodes.get(cur.parent_id) if cur.parent_id is not None else None
        return list(reversed(path))

    def render_snapshot(self, *, node_id: Optional[int] = None,
                          depth: int = 3) -> dict:
        """LLM 友好的 JSON 表示."""
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


# ═══════════════════════════════════════════════════════════════════════
# Drivers
# ═══════════════════════════════════════════════════════════════════════

class _BaseDriver:
    """Common loop: select node → run AgentLoop for expansion → backprop → repeat."""

    def __init__(self, state: SharedSearchState, *,
                  max_nodes: int, max_depth: int,
                  expansion_max_turns: int):
        self.state = state
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.expansion_max_turns = expansion_max_turns

    def _select(self) -> Optional[int]:        # override
        raise NotImplementedError

    def _backprop(self, node_id: int, success: bool):
        cur = self.state.nodes.get(node_id)
        while cur is not None:
            cur.visit_count += 1
            if success:
                cur.success_count += 1
            cur = self.state.nodes.get(cur.parent_id) \
                if cur.parent_id is not None else None

    def _score_new_node(self, node_id: int):
        n = self.state.nodes[node_id]
        # Higher = more promising. Goal reduction + closeness to depth budget.
        parent = self.state.nodes.get(n.parent_id) if n.parent_id is not None else None
        parent_goals = len(parent.goals) if parent else 1
        reduction = (parent_goals - len(n.goals)) / max(1, parent_goals)
        n.score = reduction * 2.0 \
                  + (10.0 if n.is_complete else 0.0) \
                  - 0.05 * n.depth

    async def run(self, *, expand_one_node: Callable):
        """Main driver loop. `expand_one_node(node_id)` is supplied by runner —
        it instantiates an AgentLoop anchored at `node_id` and returns its
        LoopResult (we don't care about the result here; the tool calls
        inside the loop will have mutated `state` directly).
        """
        start = time.time()
        while True:
            if self.state.solved_node_id is not None:
                logger.info(
                    f"[search] solved at node {self.state.solved_node_id}")
                return self.state
            if len(self.state.nodes) >= self.max_nodes:
                logger.info(f"[search] max_nodes={self.max_nodes} reached")
                return self.state

            nid = self._select()
            if nid is None:
                logger.info("[search] frontier empty")
                return self.state

            # Expansion: hand off to AgentLoop. The loop's tactic_apply tool
            # will mutate `state` (adding child nodes) as a side effect.
            children_before = set(self.state.nodes.keys())
            self.state.set_current(nid)
            try:
                await expand_one_node(node_id=nid,
                                       max_turns=self.expansion_max_turns)
            except Exception as e:
                logger.warning(f"expansion at node {nid} failed: {e}")
                self.state.nodes[nid].status = "failed"

            # Backprop based on whether expansion produced a solved child
            new_children = [i for i in self.state.nodes
                             if i not in children_before]
            for child_id in new_children:
                self._score_new_node(child_id)
                self._backprop(
                    child_id,
                    success=self.state.nodes[child_id].is_complete)

            elapsed = time.time() - start
            logger.debug(
                f"[search] expanded node {nid} → {len(new_children)} children, "
                f"total nodes={len(self.state.nodes)}, t={elapsed:.1f}s")


class BestFirstDriver(_BaseDriver):
    """优先扩展 score 最高的开放叶子."""

    def _select(self) -> Optional[int]:
        leaves = self.state.open_leaves()
        if not leaves:
            return None
        leaves = [
            i for i in leaves
            if self.state.nodes[i].depth < self.max_depth
        ]
        if not leaves:
            return None
        return max(leaves, key=lambda i: self.state.nodes[i].score)


class UCBDriver(_BaseDriver):
    """MCTS-UCB1 selection."""

    def __init__(self, state, *, max_nodes, max_depth,
                  expansion_max_turns, ucb_c: float = 1.414):
        super().__init__(state, max_nodes=max_nodes, max_depth=max_depth,
                          expansion_max_turns=expansion_max_turns)
        self.c = ucb_c

    def _select(self) -> Optional[int]:
        leaves = self.state.open_leaves()
        if not leaves:
            return None
        best = None
        best_ucb = -math.inf
        for nid in leaves:
            n = self.state.nodes[nid]
            if n.depth >= self.max_depth:
                continue
            visits = max(1, n.visit_count)
            wins = n.success_count
            parent_visits = 1
            if n.parent_id is not None:
                p = self.state.nodes.get(n.parent_id)
                if p:
                    parent_visits = max(1, p.visit_count)
            exploit = wins / visits
            explore = self.c * math.sqrt(math.log(parent_visits + 1) / visits)
            ucb = exploit + explore + 0.3 * n.score / (1 + visits)
            if ucb > best_ucb:
                best_ucb = ucb
                best = nid
        return best


class BeamDriver(_BaseDriver):
    """每深度只保留 top-W 节点."""

    def __init__(self, state, *, max_nodes, max_depth,
                  expansion_max_turns, beam_width: int = 8):
        super().__init__(state, max_nodes=max_nodes, max_depth=max_depth,
                          expansion_max_turns=expansion_max_turns)
        self.W = beam_width
        self._current_depth = 0
        self._depth_buckets: dict[int, list[int]] = {0: [0]}

    def _select(self) -> Optional[int]:
        # Keep top-W open leaves at the current frontier depth
        leaves = [i for i in self.state.open_leaves()
                   if self.state.nodes[i].depth < self.max_depth]
        if not leaves:
            return None
        leaves.sort(key=lambda i: -self.state.nodes[i].score)
        # Prune all but top W
        for i in leaves[self.W:]:
            self.state.nodes[i].status = "pruned"
        return leaves[0] if leaves[:self.W] else None


def make_driver(kind: str, state: SharedSearchState, *,
                  max_nodes: int, max_depth: int,
                  expansion_max_turns: int,
                  beam_width: int = 8,
                  ucb_c: float = 1.414):
    """Factory."""
    if kind == "best_first":
        return BestFirstDriver(state, max_nodes=max_nodes,
                                max_depth=max_depth,
                                expansion_max_turns=expansion_max_turns)
    if kind == "ucb":
        return UCBDriver(state, max_nodes=max_nodes, max_depth=max_depth,
                          expansion_max_turns=expansion_max_turns,
                          ucb_c=ucb_c)
    if kind == "beam":
        return BeamDriver(state, max_nodes=max_nodes, max_depth=max_depth,
                           expansion_max_turns=expansion_max_turns,
                           beam_width=beam_width)
    raise ValueError(f"unknown search.kind: {kind}")
