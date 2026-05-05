"""engine/search/ — 唯一权威的搜索代数 (节点 / 选择 / 回传 / 提取)

这个包把 ``prover/unified/`` 和 ``sampler/`` 两边各自实现的搜索算法
合并到一处。两个调用方的差异只在 expansion 那一步:

  * prover 端 expansion = 跑一次 AgentLoop (LLM + tools)
  * sampler 端 expansion = 调一次 policy_fn (裸的 RL 策略)

所以这里只做"算法"部分:

  * ``SearchNode``                  — 节点数据(两端共用)
  * ``SearchTree``                  — 树容器 + ancestors / open_leaves
  * ``BestFirstPolicy/UCBPolicy/BeamPolicy``
                                    — selection 算法
  * ``backprop_visit``              — 回传访问数 / 成功数
  * ``score_new_node``              — 启发式打分(子目标减少 + 完成奖励 - 深度惩罚)
  * ``extract_solved_path``         — 抽取根→解节点的消息链
  * ``run_search``                  — 调度循环(接受 expander callback)

调用方按需要包一层:
  * ``prover.unified.search_driver`` 提供 ``AgentLoopExpander``
  * ``sampler.tree_rollout_sampler`` 提供 ``PolicyFnExpander``

主路径与 RL 路径用同一段算法。
"""
from engine.search.core import (
    SearchNode,
    SearchTree,
    backprop_visit,
    score_new_node,
    extract_solved_path,
)
from engine.search.policies import (
    SelectionPolicy,
    BestFirstPolicy,
    UCBPolicy,
    BeamPolicy,
    make_policy,
)
from engine.search.runner import run_search

__all__ = [
    "SearchNode", "SearchTree",
    "backprop_visit", "score_new_node", "extract_solved_path",
    "SelectionPolicy", "BestFirstPolicy", "UCBPolicy", "BeamPolicy",
    "make_policy",
    "run_search",
]
