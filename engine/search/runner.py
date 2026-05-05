"""engine/search/runner.py — 搜索调度循环

接受一个 ``expander`` callback 完成节点扩展。这是两端唯一的差异点:

  * prover 端 expander = 跑一次 AgentLoop (LLM + tools)
  * sampler 端 expander = 调一次 policy_fn (RL 策略)

调度本身在两端是同一段代码。
"""
from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable

from engine.search.core import (
    SearchTree, backprop_visit, score_new_node,
)
from engine.search.policies import SelectionPolicy

logger = logging.getLogger(__name__)

# expander 签名: async def expander(tree, node_id) -> None
# 调用方在 expander 内部应该:
#   1. 用合适的方式产生候选 tactic / action
#   2. 通过 tree.expand(...) 写入新节点
#   3. (可选) 设置 node.is_complete=True 触发 solved_node_id
ExpanderFn = Callable[[SearchTree, int], Awaitable[None]]

async def run_search(
    tree: SearchTree,
    policy: SelectionPolicy,
    expander: ExpanderFn,
    *,
    max_nodes: int = 200,
    max_depth: int = 25,
    score_bump_on_success: float = 0.0,
    rescore_new_nodes: bool = True,
) -> SearchTree:
    """主调度循环。

    Args:
        tree:             共享树状态
        policy:           selection 策略
        expander:         扩展回调,调用方提供
        max_nodes:        总节点上限
        max_depth:        最大深度
        score_bump_on_success: backprop 时给祖先累加的分数 (sampler 端=0.5,
                          prover 端=0,因为 prover 端用 score_new_node 重算)
        rescore_new_nodes: 是否对每个新生子节点跑一次 score_new_node
                          (prover 端=True,sampler 端可关因为 RL reward 已含信号)

    Returns:
        同一棵 ``tree``,run 完毕状态。
    """
    start = time.time()
    while True:
        if tree.solved_node_id is not None:
            logger.info(
                "[search] solved at node %d", tree.solved_node_id)
            return tree
        if len(tree.nodes) >= max_nodes:
            logger.info("[search] max_nodes=%d reached", max_nodes)
            return tree

        nid = policy.select(tree, max_depth=max_depth)
        if nid is None:
            logger.info("[search] frontier empty")
            return tree

        children_before = set(tree.nodes.keys())
        tree.set_current(nid)

        try:
            await expander(tree, nid)
        except Exception as e:
            logger.warning("expansion at node %d failed: %s", nid, e)
            tree.nodes[nid].status = "failed"

        new_children = [i for i in tree.nodes if i not in children_before]
        for child_id in new_children:
            if rescore_new_nodes:
                score_new_node(tree, child_id)
            backprop_visit(
                tree, child_id,
                success=tree.nodes[child_id].is_complete
                        or tree.nodes[child_id].success,
                score_bump=score_bump_on_success,
            )

        elapsed = time.time() - start
        logger.debug(
            "[search] expanded node %d → %d children, total=%d, t=%.1fs",
            nid, len(new_children), len(tree.nodes), elapsed)
