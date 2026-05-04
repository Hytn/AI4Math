"""sampler/tree_rollout_sampler.py — Tree-search RL rollouts (v7)

Closes the gap noted in the v6 evaluation: ``BaseSampler`` runs strictly
linear multi-turn episodes, so the MCTS / best_first / beam profiles in
``prover/unified/profiles.py`` could not be used as RL roll-out units.
That left a whole family of SOTA methods (HyperTree, AlphaProof-style
self-play, Goedel-Prover-V2's verifier-guided search) outside the
training pipeline.

This module reuses ``prover.unified.search_driver.SharedSearchState``'s
tree data structure but layers it on top of ``ProofEnv`` instead of the
full ``AgentLoop`` machinery — so the search becomes a pure
*policy + verifier* loop with the exact same observation/reward
interface every other RL roll-out uses.

What you get
------------

For each problem, ``TreeRolloutSampler`` produces *N* trajectories —
one per root→leaf path in the search tree. Sharing the root makes each
problem a natural GRPO group: the prompt is identical, the multiple
completions branch via the search, and the per-trajectory reward
comes from ``ProofEnv``'s verification feedback (success at the leaf,
goal-progress along the way).

How it integrates
-----------------

  sampler/                                 prover/unified/
  ├── proof_env.py        (verification + reward)
  ├── base_sampler.py     (linear roll-outs)        ─┐
  ├── verl_sampler.py     (verl integration)         │
  ├── slime_sampler.py    (slime integration)        │
  ├── trajectory.py       (rollout data)             │
  └── tree_rollout_sampler.py  ◄── this file        ─┤
                                                      │
                                          search_driver.py
                                          (we reuse SharedSearchState
                                           but not the BaseDriver, since
                                           BaseDriver is bound to AgentLoop)

Usage::

    cfg = TreeRolloutConfig(
        env_config=ProofEnvConfig(backend="kimina", backend_url="..."),
        search_kind="best_first",     # or "ucb" for MCTS
        branching_factor=4,            # k candidates per node
        max_nodes=128,
        max_paths_per_problem=8,       # how many trajectories to emit
    )
    sampler = TreeRolloutSampler(cfg, policy_fn=my_policy)
    await sampler.setup()
    trajectories = await sampler.collect_rollouts(problems)
    # Each problem contributes up to `max_paths_per_problem` trajectories,
    # all sharing the same prompt — feed straight into GRPO/REINFORCE++.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from sampler.base_sampler import BaseSampler, PolicyFn, SamplerConfig
from sampler.proof_env import ProofEnv
from sampler.trajectory import (
    RewardInfo, TerminationReason, Trajectory, Turn,
)

logger = logging.getLogger(__name__)


@dataclass
class TreeRolloutConfig(SamplerConfig):
    """Configuration for the tree-rollout sampler."""

    # Search shape
    search_kind: str = "best_first"   # "best_first" | "ucb" | "beam"
    branching_factor: int = 4          # k candidate tactics per node expansion
    max_nodes: int = 128               # global tree node cap
    max_depth: int = 16                # tree depth cap
    ucb_c: float = 1.414               # exploration constant for UCB
    beam_width: int = 4                # for beam search

    # How many trajectories to emit per problem. Top-K paths by score
    # (or root→solved + top-(K-1) explored paths if a solved leaf exists).
    max_paths_per_problem: int = 8

    # GRPO/REINFORCE++ helpers
    group_normalize_rewards: bool = False
    """If True, the per-trajectory total_reward is centred and divided
    by the std-dev within each problem's group. Many GRPO implementations
    expect this to be done at the trainer; flip on if your trainer expects
    pre-normalised advantages."""


@dataclass
class _Node:
    """One tree node. Shape kept compatible with
    ``prover.unified.search_driver.TreeNode`` so the search_tree dict can
    be serialised into ``meta.search_tree`` if a caller wants to. We do
    NOT directly import that class because we want this module to remain
    importable without the full prover stack."""
    id: int
    parent_id: Optional[int]
    tactic: Optional[str]              # tactic that created this node
    observation: str                    # observation BEFORE expanding here
    depth: int = 0
    children: list[int] = field(default_factory=list)
    is_terminal: bool = False           # ProofEnv said done
    success: bool = False               # is_terminal AND reward > 0
    visit_count: int = 0
    score: float = 0.0
    cumulative_reward: float = 0.0      # sum of rewards along root→here
    reward_at_step: float = 0.0          # the step reward into this node

    # Per-step token data (from policy_fn)
    action_token_ids: list[int] = field(default_factory=list)
    action_log_probs: list[float] = field(default_factory=list)
    observation_token_ids: list[int] = field(default_factory=list)


class TreeRolloutSampler(BaseSampler):
    """RL sampler that produces tree-shaped rollouts from a single root.

    This is what unblocks training MCTS / best-first prover policies on
    top of verl/slime: each problem becomes a *group* of trajectories
    sharing the same prompt — exactly the input shape GRPO expects.

    We do NOT subclass ``BaseDriver`` from ``prover.unified.search_driver``
    because that driver is glued to ``AgentLoop`` (which owns its own LLM
    call). Instead we reimplement the (tiny) selection + expansion loop
    on top of ``ProofEnv`` so the policy is whatever ``policy_fn`` gives
    us — typically the verl/slime-managed model.
    """

    def __init__(self, config: TreeRolloutConfig = None,
                 policy_fn: PolicyFn = None):
        super().__init__(config or TreeRolloutConfig())
        self.tcfg: TreeRolloutConfig = self.config  # type alias
        self._policy_fn = policy_fn

    async def generate_action(self, observation, problem_id, turn_idx):
        """BaseSampler abstract method — only used by the linear rollout
        path. TreeRolloutSampler overrides ``collect_rollouts`` so this
        is unreachable in practice; we still implement it to satisfy the
        ABC, delegating to the supplied ``policy_fn`` if any."""
        if self._policy_fn:
            return await self._policy_fn(observation)
        raise RuntimeError(
            "TreeRolloutSampler.generate_action called without a policy_fn")

    # ── core: tree rollout per problem ────────────────────────────────

    async def collect_rollouts(
        self, problems: list[dict[str, Any]],
        policy_fn: PolicyFn = None,
    ) -> list[Trajectory]:
        """Run a tree search per problem; emit up to K trajectories each.

        Trajectory is exactly the same dataclass other samplers produce
        — ``to_verl_format()`` / ``to_slime_episodes()`` work as-is.
        Trajectories from the same problem all start with the identical
        root observation, which lets verl/slime treat them as a GRPO
        group (key on ``problem_id``).
        """
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

        # Each problem produces a *list* of trajectories. Flatten.
        nested = await asyncio.gather(
            *[_run_one(p) for p in problems], return_exceptions=True)
        out: list[Trajectory] = []
        for i, group in enumerate(nested):
            if isinstance(group, Exception):
                logger.error("tree rollout %d failed: %s", i, group)
                continue
            out.extend(group)
        return out

    async def _tree_rollout(self, env: ProofEnv,
                              problem: dict) -> list[Trajectory]:
        """Drive one search tree on one problem and return K trajectories."""
        cfg = self.tcfg

        # Root: ProofEnv.reset gives us the initial observation.
        root_obs = await env.reset(problem)
        nodes: dict[int, _Node] = {
            0: _Node(id=0, parent_id=None, tactic=None,
                       observation=root_obs, depth=0)
        }

        # ProofEnv is stateful — applying a tactic at node N advances
        # internal state from N. To explore siblings/cousins we have to
        # reset the env back to the parent's snapshot before expanding
        # each candidate. We therefore execute the search *path-by-path*:
        # pick a leaf → reset env → walk down ancestors applying their
        # tactics → expand at that leaf with k candidates → recurse.
        #
        # This is slower than a real backtracking REPL (Pantograph would
        # let us snapshot-and-restore in O(1)) but it's correct on every
        # backend. When backend == "pantograph", a future optimisation
        # can swap this in for native snapshots.
        node_count = 1
        while (self._has_open_frontier(nodes)
               and node_count < cfg.max_nodes):
            target = self._select_leaf(nodes, cfg)
            if target is None:
                break

            # Replay path-from-root in env to land on `target`.
            await env.reset(problem)
            path = self._ancestors(nodes, target)
            replay_failed = False
            for ancestor in path[1:]:  # skip root (its tactic is None)
                _, _, done, _ = await env.step(ancestor.tactic or "")
                if done:
                    # Replay shouldn't terminate before we reach target;
                    # if it does, mark the target as unreachable.
                    replay_failed = True
                    break
            if replay_failed:
                nodes[target.id].is_terminal = True
                continue

            # Expand `target` with up to k candidates from policy_fn.
            obs_here = self._observation_at(nodes, target.id, root_obs)
            candidates = await self._sample_k_candidates(
                obs_here, problem["problem_id"], target.depth,
                cfg.branching_factor)

            # Each candidate is a (action, token_ids, log_probs) tuple.
            # We can't apply more than one to the *same* env without
            # rolling back; so we apply the first directly here and
            # save the rest for replay-then-apply on later iterations.
            # Simpler model: re-reset env per candidate. Costs a reset
            # per child but keeps the code obviously correct.
            for cand_idx, (action, tok_ids, log_probs) in enumerate(
                    candidates):
                if node_count >= cfg.max_nodes:
                    break

                # Restore env to target's state before applying this candidate.
                if cand_idx > 0:  # first candidate: env still at target
                    await env.reset(problem)
                    for ancestor in path[1:]:
                        await env.step(ancestor.tactic or "")

                obs_after, reward_info, done, _info = await env.step(action)
                child_id = node_count
                node_count += 1
                child = _Node(
                    id=child_id,
                    parent_id=target.id,
                    tactic=action,
                    observation=obs_after,
                    depth=target.depth + 1,
                    is_terminal=done,
                    success=(done and reward_info.scalar > 0),
                    reward_at_step=reward_info.scalar,
                    cumulative_reward=(
                        nodes[target.id].cumulative_reward
                        + reward_info.scalar),
                    action_token_ids=tok_ids,
                    action_log_probs=log_probs,
                )
                # Score = goal-reduction-style heuristic (matches
                # search_driver._score_new_node behaviour qualitatively).
                child.score = (reward_info.scalar
                                  + (10.0 if child.success else 0.0)
                                  - 0.05 * child.depth)
                nodes[child_id] = child
                nodes[target.id].children.append(child_id)
                self._backprop(nodes, child_id,
                                  success=child.success)

                if child.success:
                    # Found a complete proof — stop expanding deeper but
                    # keep collecting paths for the next iterations.
                    break

        return self._extract_trajectories(nodes, problem, root_obs)

    # ── helpers ─────────────────────────────────────────────────────────

    def _has_open_frontier(self, nodes: dict[int, _Node]) -> bool:
        for n in nodes.values():
            if not n.is_terminal and n.depth < self.tcfg.max_depth:
                return True
        return False

    def _select_leaf(self, nodes: dict[int, _Node],
                       cfg: TreeRolloutConfig) -> Optional[_Node]:
        """Pick the next node to expand, per ``search_kind``."""
        candidates = [
            n for n in nodes.values()
            if not n.is_terminal
            and n.depth < cfg.max_depth
            and not n.children   # only expand un-expanded leaves
        ]
        # Fallback: also allow re-expansion of leaves that have children
        # but didn't solve (gives UCB room to explore deeper).
        if not candidates:
            candidates = [
                n for n in nodes.values()
                if not n.is_terminal and n.depth < cfg.max_depth
                and not any(nodes[c].success for c in n.children)
            ]
        if not candidates:
            return None

        if cfg.search_kind == "best_first":
            return max(candidates, key=lambda n: n.score)

        if cfg.search_kind == "beam":
            # Pick the top-N by score at the current frontier depth
            depth_grouped: dict[int, list[_Node]] = {}
            for n in candidates:
                depth_grouped.setdefault(n.depth, []).append(n)
            shallowest = min(depth_grouped)
            beam = sorted(
                depth_grouped[shallowest],
                key=lambda n: -n.score)[:cfg.beam_width]
            return beam[0] if beam else None

        if cfg.search_kind == "ucb":
            # Standard UCB1: argmax(value/visits + c·sqrt(ln(parent_visits)/visits))
            def _ucb(n: _Node) -> float:
                if n.visit_count == 0:
                    return math.inf
                parent = nodes.get(n.parent_id) if n.parent_id is not None else None
                pv = parent.visit_count if parent else 1
                exploit = n.score / max(1, n.visit_count)
                explore = cfg.ucb_c * math.sqrt(
                    math.log(max(1, pv)) / n.visit_count)
                return exploit + explore
            return max(candidates, key=_ucb)

        # Unknown kind — fall back to best_first
        return max(candidates, key=lambda n: n.score)

    def _backprop(self, nodes: dict[int, _Node], leaf_id: int,
                    success: bool):
        cur = nodes.get(leaf_id)
        while cur is not None:
            cur.visit_count += 1
            if success and cur.score < 5.0:
                cur.score += 0.5
            cur = nodes.get(cur.parent_id) \
                if cur.parent_id is not None else None

    def _ancestors(self, nodes: dict[int, _Node],
                     leaf: _Node) -> list[_Node]:
        path: list[_Node] = [leaf]
        cur = leaf
        while cur.parent_id is not None:
            cur = nodes[cur.parent_id]
            path.append(cur)
        return list(reversed(path))

    def _observation_at(self, nodes: dict[int, _Node],
                          node_id: int, root_obs: str) -> str:
        node = nodes[node_id]
        return node.observation if node.observation else root_obs

    async def _sample_k_candidates(
        self, observation: str, problem_id: str, depth: int, k: int,
    ) -> list[tuple[str, list[int], list[float]]]:
        """Ask the policy for k distinct tactics from the same observation.

        We call ``policy_fn`` k times; if the policy doesn't return
        diverse samples (e.g. greedy) the search collapses to a chain,
        which is fine and detected by the tree's structure.
        """
        if self._policy_fn is None:
            raise RuntimeError(
                "TreeRolloutSampler requires a policy_fn (got None)")

        async def _one() -> tuple[str, list[int], list[float]]:
            return await self._policy_fn(observation)

        # Sequential sampling. Parallelism is the *trainer's* job —
        # for vLLM/SGLang policies, a batched call is cleaner; we
        # leave that as an extension point.
        out = []
        for _ in range(k):
            try:
                out.append(await _one())
            except Exception as e:
                logger.warning("policy sample failed at depth %d: %s",
                                  depth, e)
        return out

    def _extract_trajectories(
        self, nodes: dict[int, _Node], problem: dict, root_obs: str,
    ) -> list[Trajectory]:
        """Walk the tree, emit one Trajectory per top-K leaf path.

        Selection priority:
          1. All solved leaves (success=True), sorted by depth (shorter first).
          2. Top-K-remaining un-solved leaves by score.

        Trimmed to ``max_paths_per_problem``. Each Trajectory has the
        full root→leaf chain as ``turns``.
        """
        cfg = self.tcfg
        leaves = [n for n in nodes.values() if not n.children]

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
            t.metadata["tree_size"] = len(nodes)

            path = self._ancestors(nodes, leaf)
            for i, n in enumerate(path[1:], start=0):
                # path[0] is root (no tactic / no reward) — turn 0
                # corresponds to the first tactic, i.e. path[1].
                parent = path[i]  # the node *before* this tactic
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

        # Optional GRPO-style group normalisation
        if cfg.group_normalize_rewards and len(trajectories) >= 2:
            rewards = [t.total_reward for t in trajectories]
            mu = sum(rewards) / len(rewards)
            var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
            sd = math.sqrt(var) if var > 1e-9 else 1.0
            for t in trajectories:
                t.metadata["group_advantage"] = (t.total_reward - mu) / sd

        return trajectories

    # ── search_tree dict (V3 dialog.json compatible) ────────────────────

    @staticmethod
    def to_search_tree_dict(nodes: dict[int, _Node],
                              kind: str) -> dict:
        """Serialise the tree to ``meta.search_tree`` shape so callers can
        embed the rollout's tree structure in a dialog.json — matching
        the schema produced by ``prover.unified.search_driver``.

        Useful for offline analysis / SFT data extraction.
        """
        max_depth = max((n.depth for n in nodes.values()), default=0)
        solved = next(
            (n.id for n in nodes.values() if n.success), None)
        return {
            "kind": kind,
            "root_node_id": 0,
            "solved_node_id": solved,
            "total_nodes": len(nodes),
            "max_depth": max_depth,
            "nodes": [
                {
                    "node_id": n.id,
                    "parent_id": n.parent_id,
                    "tactic": n.tactic,
                    "depth": n.depth,
                    "status": ("solved" if n.success
                                 else ("failed" if n.is_terminal
                                         else "open")),
                    "visit_count": n.visit_count,
                    "score": n.score,
                    "is_complete": n.success,
                    "reward_at_step": n.reward_at_step,
                }
                for n in sorted(nodes.values(), key=lambda x: x.id)
            ],
        }
