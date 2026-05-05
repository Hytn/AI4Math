"""engine/search/policies.py — Selection 策略 (best-first / UCB / beam)

纯算法,只看 ``SearchTree.nodes`` 数据。两端调用一致。
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

from engine.search.core import SearchTree

class SelectionPolicy(ABC):
    """选择下一个要扩展的节点。"""

    @abstractmethod
    def select(self, tree: SearchTree, *, max_depth: int) -> Optional[int]:
        ...

class BestFirstPolicy(SelectionPolicy):
    """选 score 最高的开放叶子。"""

    def select(self, tree: SearchTree, *, max_depth: int) -> Optional[int]:
        leaves = [
            i for i in tree.open_leaves()
            if tree.nodes[i].depth < max_depth
        ]
        if not leaves:
            return None
        return max(leaves, key=lambda i: tree.nodes[i].score)

class UCBPolicy(SelectionPolicy):
    """MCTS-UCB1。"""

    def __init__(self, c: float = 1.414):
        self.c = c

    def select(self, tree: SearchTree, *, max_depth: int) -> Optional[int]:
        leaves = [
            i for i in tree.open_leaves()
            if tree.nodes[i].depth < max_depth
        ]
        if not leaves:
            return None
        best, best_ucb = None, -math.inf
        for nid in leaves:
            n = tree.nodes[nid]
            visits = max(1, n.visit_count)
            wins = n.success_count
            parent_visits = 1
            if n.parent_id is not None:
                p = tree.nodes.get(n.parent_id)
                if p:
                    parent_visits = max(1, p.visit_count)
            exploit = wins / visits
            explore = self.c * math.sqrt(
                math.log(parent_visits + 1) / visits)
            # 把 score 作为先验小幅注入,初次访问时 visit=1 才参考
            ucb = exploit + explore + 0.3 * n.score / (1 + visits)
            if ucb > best_ucb:
                best_ucb = ucb
                best = nid
        return best

class BeamPolicy(SelectionPolicy):
    """每层只保留 top-W 节点,选最高分。"""

    def __init__(self, beam_width: int = 8):
        self.W = beam_width

    def select(self, tree: SearchTree, *, max_depth: int) -> Optional[int]:
        leaves = [
            i for i in tree.open_leaves()
            if tree.nodes[i].depth < max_depth
        ]
        if not leaves:
            return None
        leaves.sort(key=lambda i: -tree.nodes[i].score)
        # 剪枝 top-W 之外的节点 (副作用,持久化到树状态)
        for i in leaves[self.W:]:
            tree.nodes[i].status = "pruned"
        return leaves[0] if leaves[:self.W] else None

def make_policy(kind: str, *, beam_width: int = 8,
                ucb_c: float = 1.414) -> SelectionPolicy:
    """工厂。"""
    if kind == "best_first":
        return BestFirstPolicy()
    if kind == "ucb":
        return UCBPolicy(c=ucb_c)
    if kind == "beam":
        return BeamPolicy(beam_width=beam_width)
    raise ValueError(f"unknown selection policy: {kind!r}")
