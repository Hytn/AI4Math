"""sampler/tree_rollout_sampler.py — 树形 RL rollout

每个 problem 跑一棵搜索树,产出 N 条 root→leaf 的 trajectory,根 prompt
相同就构成天然的 GRPO group。

搜索代数(节点 / selection / backprop)走 ``engine/search/``,与 prover
主路径同一份实现。差异只在 expansion:这里调 ``policy_fn`` + ``ProofEnv.step``,
prover 端调 AgentLoop。

用法::

    cfg = TreeRolloutConfig(
        env_config=ProofEnvConfig(backend="kimina", backend_url="..."),
        search_kind="best_first",     # 或 "ucb" / "beam"
        branching_factor=4,
        max_nodes=128,
        max_paths_per_problem=8,
    )
    sampler = TreeRolloutSampler(cfg, policy_fn=my_policy)
    await sampler.setup()
    trajectories = await sampler.collect_rollouts(problems)
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

from engine.search.core import SearchTree, SearchNode, backprop_visit
from engine.search.policies import make_policy

from sampler.base_sampler import BaseSampler, PolicyFn, SamplerConfig
from sampler.proof_env import ProofEnv
from sampler.trajectory import (
    RewardInfo, TerminationReason, Trajectory, Turn,
)

logger = logging.getLogger(__name__)

# 兼容旧测试 / 外部代码:_Node 是 SearchNode 的别名,字段一致
_Node = SearchNode

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TreeRolloutConfig(SamplerConfig):
    """树形 rollout 配置。"""

    # 搜索形状
    search_kind: str = "best_first"   # "best_first" | "ucb" | "beam"
    branching_factor: int = 4
    max_nodes: int = 128
    max_depth: int = 16
    ucb_c: float = 1.414
    beam_width: int = 4

    # 每 problem 输出多少条 trajectory
    max_paths_per_problem: int = 8

    # GRPO/REINFORCE++ 辅助
    group_normalize_rewards: bool = False

# ═══════════════════════════════════════════════════════════════════════
# Sampler
# ═══════════════════════════════════════════════════════════════════════

class TreeRolloutSampler(BaseSampler):
    """树形 RL rollout sampler。

    每 problem 跑一棵 ``SearchTree``,然后抽 K 条 root→leaf 路径。
    """

    def __init__(self, config: TreeRolloutConfig = None,
                 policy_fn: PolicyFn = None):
        super().__init__(config or TreeRolloutConfig())
        self.tcfg: TreeRolloutConfig = self.config  # type alias
        self._policy_fn = policy_fn

    async def generate_action(self, observation, problem_id, turn_idx):
        """BaseSampler 抽象方法 — 仅线性 rollout 路径用。这里委托给 policy_fn。"""
        if self._policy_fn:
            return await self._policy_fn(observation)
        raise RuntimeError(
            "TreeRolloutSampler.generate_action called without a policy_fn")

    # ── 主入口 ─────────────────────────────────────────────────────────

    async def collect_rollouts(
        self, problems: list[dict[str, Any]],
        policy_fn: PolicyFn = None,
    ) -> list[Trajectory]:
        """每 problem 跑一棵树,产出最多 K 条 trajectory。"""
        if not self._setup_done:
            await self.setup()
        if policy_fn is not None:
            self._policy_fn = policy_fn

        sem = asyncio.Semaphore(self.config.max_concurrent_problems)

        async def _run_one(problem: dict) -> list[Trajectory]:
            async with sem:
                env: ProofEnv = await self._env_queue.get()
                try:
                    return await self._tree_rollout(env, problem)
                finally:
                    self._env_queue.put_nowait(env)

        nested = await asyncio.gather(
            *[_run_one(p) for p in problems], return_exceptions=True)
        out: list[Trajectory] = []
        for i, group in enumerate(nested):
            if isinstance(group, Exception):
                logger.error("tree rollout %d failed: %s", i, group)
                continue
            out.extend(group)
        return out

    # ── 单个 problem 跑树 ──────────────────────────────────────────────

    async def _tree_rollout(self, env: ProofEnv,
                            problem: dict) -> list[Trajectory]:
        cfg = self.tcfg

        root_obs = await env.reset(problem)
        tree = SearchTree(root_observation=root_obs)

        policy = make_policy(
            cfg.search_kind, beam_width=cfg.beam_width, ucb_c=cfg.ucb_c)

        # ProofEnv 是有状态的:展开非根节点要先 reset → replay 路径。
        # Pantograph backend 之后可以替换为原生 snapshot 优化。
        async def _expander(_t: SearchTree, target_id: int):
            target = tree.nodes[target_id]
            await env.reset(problem)
            path = tree.ancestors(target_id)
            replay_failed = False
            for ancestor in path[1:]:  # 跳过根
                _, _, done, _ = await env.step(ancestor.tactic or "")
                if done:
                    replay_failed = True
                    break
            if replay_failed:
                target.is_terminal = True
                target.status = "failed"
                return

            obs_here = target.observation or root_obs
            candidates = await self._sample_k_candidates(
                obs_here, problem["problem_id"], target.depth,
                cfg.branching_factor)

            for cand_idx, (action, tok_ids, log_probs) in enumerate(
                    candidates):
                if len(tree.nodes) >= cfg.max_nodes:
                    break

                # 第二个起的候选:每个都要 reset 回 target 再施加,保证正确性
                if cand_idx > 0:
                    await env.reset(problem)
                    for ancestor in path[1:]:
                        await env.step(ancestor.tactic or "")

                obs_after, reward_info, done, _info = await env.step(action)
                tree.expand(
                    parent_node_id=target_id,
                    tactic=action,
                    observation=obs_after,
                    reward=reward_info.scalar,
                    is_terminal=done,
                    success=(done and reward_info.scalar > 0),
                    is_complete=(done and reward_info.scalar > 0),
                    action_token_ids=tok_ids,
                    action_log_probs=log_probs,
                )
                # 找到完整证明就停止此节点的扩展
                child_id = max(tree.nodes)
                if tree.nodes[child_id].success:
                    break

        # 调度循环 — 与 prover 主路径走同一段代码
        # 上界用 nodes 增量来近似,因为 sampler 端可能在一次扩展中产生多个候选
        from engine.search.runner import run_search
        await run_search(
            tree, policy, _expander,
            max_nodes=cfg.max_nodes,
            max_depth=cfg.max_depth,
            score_bump_on_success=0.5,    # 与历史行为一致
            rescore_new_nodes=False,      # RL 端用 reward,自己已打分
        )

        return self._extract_trajectories(tree, problem, root_obs)

    # ── helpers ───────────────────────────────────────────────────────

    async def _sample_k_candidates(
        self, observation: str, problem_id: str, depth: int, k: int,
    ) -> list[tuple[str, list[int], list[float]]]:
        if self._policy_fn is None:
            raise RuntimeError(
                "TreeRolloutSampler requires a policy_fn (got None)")

        out = []
        for _ in range(k):
            try:
                out.append(await self._policy_fn(observation))
            except Exception as e:
                logger.warning("policy sample failed at depth %d: %s",
                               depth, e)
        return out

    def _extract_trajectories(
        self, tree_or_nodes, problem: dict, root_obs: str,
    ) -> list[Trajectory]:
        """走完树,每条 root→leaf 输出一条 Trajectory(top-K 按 success / score)。

        ``tree_or_nodes`` 兼容 ``SearchTree`` 实例和 ``dict[int, SearchNode]``。
        """
        cfg = self.tcfg

        if isinstance(tree_or_nodes, SearchTree):
            tree = tree_or_nodes
        else:
            # 老调用方:dict[id]→Node,从中重建一个轻量 SearchTree 包装,
            # 复用 ancestors / all_leaves
            nodes_dict = tree_or_nodes
            tree = SearchTree(root_observation=root_obs)
            tree.nodes = nodes_dict
            tree._next_id = max(nodes_dict) + 1 if nodes_dict else 1
            for n in nodes_dict.values():
                if n.is_complete or n.success:
                    tree.solved_node_id = tree.solved_node_id or n.id

        leaves = [tree.nodes[i] for i in tree.all_leaves()]

        solved = sorted([n for n in leaves if n.success],
                        key=lambda n: n.depth)
        unsolved = sorted([n for n in leaves if not n.success],
                          key=lambda n: -n.score)

        chosen = (solved + unsolved)[:cfg.max_paths_per_problem]
        trajectories: list[Trajectory] = []

        for leaf in chosen:
            t = Trajectory(
                problem_id=problem["problem_id"],
                theorem_statement=problem.get("theorem_statement", ""),
            )
            t.metadata["search_kind"] = cfg.search_kind
            t.metadata["leaf_node_id"] = leaf.id
            t.metadata["tree_size"] = len(tree.nodes)

            path = tree.ancestors(leaf.id)
            for i, n in enumerate(path[1:], start=0):
                parent = path[i]
                turn = Turn(
                    turn_idx=i,
                    observation=parent.observation if i > 0 else root_obs,
                    action=n.tactic or "",
                    reward=RewardInfo(
                        scalar=n.reward_at_step,
                        is_terminal=n.is_terminal,
                        verification_level="L1",
                    ),
                    action_token_ids=list(n.action_token_ids),
                    action_log_probs=list(n.action_log_probs),
                    action_mask=[1] * len(n.action_token_ids),
                )
                t.add_turn(turn)

            if leaf.success:
                t.success = True
                t.termination = TerminationReason.SUCCESS
            else:
                t.termination = TerminationReason.MAX_TURNS
            trajectories.append(t)

        # GRPO-style group normalisation (可选)
        if cfg.group_normalize_rewards and len(trajectories) >= 2:
            rewards = [t.total_reward for t in trajectories]
            mu = sum(rewards) / len(rewards)
            var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
            sd = math.sqrt(var) if var > 1e-9 else 1.0
            for t in trajectories:
                t.metadata["group_advantage"] = (t.total_reward - mu) / sd

        return trajectories

    # ── search_tree dict (与 prover/unified 同 schema) ─────────────────

    @staticmethod
    def to_search_tree_dict(tree_or_nodes, kind: str) -> dict:
        """Trajectory 之外可单独序列化树到 ``meta.search_tree`` 形态,
        和 prover/unified 主路径产出的 dialog.json 同 schema。

        参数兼容两种形态:
          - ``SearchTree`` 实例
          - ``dict[int, SearchNode]`` (老调用方,直接给节点字典)
        """
        if isinstance(tree_or_nodes, SearchTree):
            nodes_dict = tree_or_nodes.nodes
            solved = tree_or_nodes.solved_node_id
        else:
            nodes_dict = tree_or_nodes
            solved = next(
                (n.id for n in nodes_dict.values()
                 if n.success or n.is_complete), None)
        max_depth = max((n.depth for n in nodes_dict.values()), default=0)
        return {
            "kind": kind,
            "root_node_id": 0,
            "solved_node_id": solved,
            "total_nodes": len(nodes_dict),
            "max_depth": max_depth,
            "nodes": [
                {
                    "node_id": n.id,
                    "parent_id": n.parent_id,
                    "tactic": n.tactic,
                    "depth": n.depth,
                    "status": ("solved"
                               if (n.success or n.is_complete)
                               else ("failed" if n.is_terminal
                                     else (n.status or "open"))),
                    "visit_count": n.visit_count,
                    "score": n.score,
                    "is_complete": n.is_complete or n.success,
                    "reward_at_step": n.reward_at_step,
                }
                for n in sorted(nodes_dict.values(), key=lambda x: x.id)
            ],
        }
