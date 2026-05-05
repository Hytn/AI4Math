"""engine/search/core.py — 搜索节点 / 树容器 / 通用回传与打分

两个调用方共用一套节点数据。差异通过可选字段表达:

  * prover 端用到 ``env_id`` ``goals`` ``messages`` (Lean REPL + dialog)
  * sampler 端用到 ``observation`` ``cumulative_reward`` ``reward_at_step``
    ``action_token_ids`` ``action_log_probs`` (RL 训练数据)

所有字段默认值合理,任何一端不需要的字段就空着。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SearchNode:
    """搜索树节点。两端共用一份数据。"""
    id: int
    parent_id: Optional[int]
    tactic: Optional[str]                 # 从 parent 走到这里的 tactic
    depth: int = 0

    # ── 树结构 ──
    children: list[int] = field(default_factory=list)

    # ── 状态 ──
    status: str = "open"                  # open | solved | failed | pruned
    is_complete: bool = False             # 等价于 status == "solved",方便代码读
    visit_count: int = 0
    success_count: int = 0
    score: float = 0.0
    failed_tactics: set = field(default_factory=set)

    # ── prover 端字段 (Lean REPL + dialog) ──
    env_id: int = 0                       # Lean REPL 环境编号
    goals: list[str] = field(default_factory=list)
    messages: list = field(default_factory=list)  # 节点 expansion 期间的对话

    # ── sampler 端字段 (RL 训练数据) ──
    observation: str = ""                 # 进入此节点前的观察
    cumulative_reward: float = 0.0        # 从根到此节点的累计 reward
    reward_at_step: float = 0.0           # 进入此节点的单步 reward
    is_terminal: bool = False             # ProofEnv 说 done
    success: bool = False                 # is_terminal AND reward > 0
    action_token_ids: list[int] = field(default_factory=list)
    action_log_probs: list[float] = field(default_factory=list)
    observation_token_ids: list[int] = field(default_factory=list)

class SearchTree:
    """树容器 + 共享状态。

    被 driver/runner 持有,被 LLM 工具或 policy 通过 ``expand()`` 间接修改。
    线程安全不是这一层的责任:prover 主路径是单 asyncio 任务串行,sampler
    并行隔离在 problem 粒度(每 problem 一棵树)。
    """

    def __init__(self, root_env_id: int = 0,
                 root_goals: Optional[list[str]] = None,
                 root_observation: str = ""):
        root = SearchNode(
            id=0, parent_id=None, tactic=None, depth=0,
            env_id=root_env_id,
            goals=list(root_goals or []),
            observation=root_observation,
        )
        self.nodes: dict[int, SearchNode] = {0: root}
        self._next_id = 1
        self.current_node_id = 0
        self.solved_node_id: Optional[int] = None

    # ── 操作 ─────────────────────────────────────────────────────────

    def expand(self, *, parent_node_id: Optional[int],
               tactic: str,
               new_env_id: int = 0,
               remaining_goals: Optional[list[str]] = None,
               is_complete: bool = False,
               observation: str = "",
               reward: float = 0.0,
               is_terminal: bool = False,
               success: bool = False,
               action_token_ids: Optional[list[int]] = None,
               action_log_probs: Optional[list[float]] = None) -> int:
        """新增子节点。返回新节点 id。

        prover 端关心 ``new_env_id`` ``remaining_goals`` ``is_complete``。
        sampler 端关心 ``observation`` ``reward`` ``is_terminal`` ``success``
        和 token 数据。其他字段空着不影响。
        """
        parent_id = parent_node_id if parent_node_id is not None \
            else self.current_node_id
        parent = self.nodes.get(parent_id)
        if parent is None:
            parent = self.nodes[0]
            parent_id = 0

        node = SearchNode(
            id=self._next_id,
            parent_id=parent_id,
            tactic=tactic,
            depth=parent.depth + 1,
            env_id=new_env_id,
            goals=list(remaining_goals or []),
            is_complete=is_complete,
            status="solved" if is_complete else "open",
            observation=observation,
            cumulative_reward=parent.cumulative_reward + reward,
            reward_at_step=reward,
            is_terminal=is_terminal,
            success=success,
            action_token_ids=list(action_token_ids or []),
            action_log_probs=list(action_log_probs or []),
        )
        self.nodes[self._next_id] = node
        parent.children.append(self._next_id)
        if is_complete and self.solved_node_id is None:
            self.solved_node_id = self._next_id
        self._next_id += 1
        return node.id

    def set_current(self, node_id: int):
        if node_id in self.nodes:
            self.current_node_id = node_id

    def mark_failed_tactic(self, node_id: int, tactic: str):
        n = self.nodes.get(node_id)
        if n:
            n.failed_tactics.add(tactic)

    def append_messages(self, node_id: int, messages: list):
        n = self.nodes.get(node_id)
        if n:
            n.messages.extend(messages)

    # ── 读取 ─────────────────────────────────────────────────────────

    def open_leaves(self) -> list[int]:
        """所有 status=open 且无子节点的叶子。"""
        return [
            i for i, n in self.nodes.items()
            if n.status == "open" and not n.children
        ]

    def all_leaves(self) -> list[int]:
        return [i for i, n in self.nodes.items() if not n.children]

    def ancestors(self, node_id: int) -> list[SearchNode]:
        """根→该节点的路径(含两端)。"""
        path: list[SearchNode] = []
        cur_id: Optional[int] = node_id
        while cur_id is not None and cur_id in self.nodes:
            n = self.nodes[cur_id]
            path.append(n)
            cur_id = n.parent_id
        return list(reversed(path))

# ═══════════════════════════════════════════════════════════════════════
# 通用算法 (selection 之外的部分)
# ═══════════════════════════════════════════════════════════════════════

def backprop_visit(tree: SearchTree, node_id: int, success: bool,
                   score_bump: float = 0.0):
    """从给定节点回传到根,累加 visit / success,可选累加分数。

    prover 端调用方传 ``score_bump=0`` (打分用 score_new_node)
    sampler 端调用方传 ``score_bump=0.5`` 累加,直到 5.0 上限。
    """
    cur = tree.nodes.get(node_id)
    while cur is not None:
        cur.visit_count += 1
        if success:
            cur.success_count += 1
            if score_bump > 0 and cur.score < 5.0:
                cur.score += score_bump
        cur = tree.nodes.get(cur.parent_id) \
            if cur.parent_id is not None else None

def score_new_node(tree: SearchTree, node_id: int):
    """启发式打分:子目标减少 + 完成奖励 - 深度惩罚。

    与 prover 端原 _BaseDriver._score_new_node 等价。sampler 端原 _Node 的
    score 默认为 0,通过 backprop 时累加,所以这个函数对 sampler 端也兼容
    (调用方不调即可)。
    """
    n = tree.nodes.get(node_id)
    if n is None:
        return
    parent = tree.nodes.get(n.parent_id) if n.parent_id is not None else None
    parent_goals = len(parent.goals) if parent and parent.goals else 1
    reduction = (parent_goals - len(n.goals)) / max(1, parent_goals)
    n.score = (
        reduction * 2.0
        + (10.0 if n.is_complete else 0.0)
        - 0.05 * n.depth
    )

def extract_solved_path(tree: SearchTree) -> list[dict]:
    """抽取根→解节点的扁平 messages 列表,塞入 dialog['messages']。

    没解出来时退而求其次:走 (depth, score) 字典序最大的节点。
    """
    target = tree.solved_node_id
    if target is None:
        if not tree.nodes:
            return []
        target = max(
            tree.nodes,
            key=lambda i: (tree.nodes[i].depth, tree.nodes[i].score))
    out: list[dict] = []
    for n in tree.ancestors(target):
        out.extend(n.messages)
    return out
