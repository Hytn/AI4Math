"""engine/async_search.py — 异步证明搜索协调器

将 SearchCoordinator 的 REPL 交互路径从同步升级为异步,
使搜索引擎可以与 AsyncLeanPool / ElasticPool 在同一事件循环中协作。

关键改进:
  - async_try_tactic: REPL 调用期间不阻塞事件循环
  - async_try_batch:  asyncio.gather 实现 N 路 tactic 真正并行验证
  - async_run_search: 异步搜索主循环, 可与 LLM 调用交替执行

与 SearchCoordinator 的关系:
  继承其搜索树管理、评分、UCB 选择等纯 CPU 逻辑,
  仅替换涉及 I/O 的 REPL 交互路径。

Usage::

    from engine.async_search import AsyncSearchCoordinator

    coord = AsyncSearchCoordinator(env, goal_type, async_pool=pool)

    # 异步搜索
    stats = await coord.async_run_search(tactic_generator)

    # 或手动控制
    node_id = coord.select_node()
    results = await coord.async_try_batch(node_id, ["simp", "omega"])
"""
from __future__ import annotations
import asyncio
import heapq
import logging
import time
from typing import Callable, Awaitable, Optional

from engine.core import Expr
from engine.state import NodeId
from engine.search import (
    SearchCoordinator, SearchConfig, SearchStats, ExpansionResult,
)
from engine.state import GoalView
from engine.tactic import execute_tactic

logger = logging.getLogger(__name__)


class AsyncSearchCoordinator(SearchCoordinator):
    """异步证明搜索协调器

    继承 SearchCoordinator 的搜索树/评分/选择逻辑,
    将 REPL 交互替换为 async I/O。

    async_pool 接受 AsyncLeanPool 或 ElasticPool (鸭子类型),
    需满足:
      - async try_tactic(env_id, tactic) -> TacticFeedback
      - async try_tactics_parallel(env_id, tactics) -> list[TacticFeedback]
      - base_env_id property
    """

    def __init__(self, env, goal_type: Expr,
                 config: SearchConfig = None,
                 async_pool=None,
                 lean_pool=None):
        """
        Args:
            env: Lean environment for local tactic engine.
            goal_type: Root goal type expression.
            config: Search configuration.
            async_pool: Async REPL pool (AsyncLeanPool or ElasticPool).
                        If provided, REPL verification uses async I/O.
            lean_pool: Sync pool (fallback, passed to parent).
        """
        super().__init__(env, goal_type, config=config, lean_pool=lean_pool)
        self._async_pool = async_pool
        if async_pool and not lean_pool:
            # Bind root node to async pool's base env_id
            self._node_env_map[0] = async_pool.base_env_id

    async def async_try_tactic(self, node_id: int, tactic_str: str,
                               prior: float = 0.5) -> ExpansionResult:
        """异步尝试单条 tactic — REPL 调用不阻塞事件循环

        如果 async_pool 可用且 node_id 有对应的 env_id 映射,
        通过异步 REPL 验证。否则回退到本地 tactic 引擎 (同步)。
        """
        # If no async pool or no env_id mapping, fall back to sync
        if not self._async_pool or node_id not in self._node_env_map:
            return self.try_tactic(node_id, tactic_str, prior)

        t0 = time.perf_counter_ns()
        node = self._tree.get(NodeId(node_id))
        if not node:
            return ExpansionResult(node_id, tactic_str, False,
                                  error={"kind": "not_found"}, elapsed_us=0)

        env_id = self._node_env_map[node_id]
        repl_result = await self._async_pool.try_tactic(env_id, tactic_str)
        elapsed = int((time.perf_counter_ns() - t0) / 1000)
        self._stats.nodes_expanded += 1

        if repl_result.success:
            complete = repl_result.is_proof_complete
            goals = [{"target": g} for g in repl_result.remaining_goals]

            # Build child state: prefer local engine, fallback to REPL info
            local_result = execute_tactic(node.state, tactic_str)
            if local_result.success:
                child_state = local_result.state
            else:
                child_state = self._build_state_from_repl(
                    node.state, repl_result.remaining_goals, complete)

            result = self._handle_tactic_success(
                node_id, tactic_str, child_state, prior, elapsed,
                complete, goals)

            # Bind new env_id to child node
            if result.child_node is not None:
                self._node_env_map[result.child_node] = \
                    repl_result.new_env_id

            return result

        # REPL failure
        self._stats.l1_filtered += 1
        err = {"kind": repl_result.error_category,
               "message": repl_result.error_message}
        return ExpansionResult(node_id, tactic_str, False,
                               error=err, elapsed_us=elapsed)

    async def async_try_batch(self, node_id: int, tactics: list[str],
                              priors: list[float] = None
                              ) -> list[ExpansionResult]:
        """异步批量尝试多条 tactic — asyncio.gather 真正并行

        当 async_pool 可用时, 所有 tactic 并发发送到 REPL,
        I/O 等待期间事件循环可处理其他协程 (如 LLM 调用)。
        """
        if priors is None:
            priors = [0.5] * len(tactics)

        if not self._async_pool or node_id not in self._node_env_map:
            # Sync fallback
            return self.try_batch(node_id, tactics, priors)

        # For async REPL: we must serialize expansions to maintain
        # correct tree state, because each expansion mutates the tree.
        # However, we can parallelize the REPL calls themselves and
        # then apply results sequentially.
        env_id = self._node_env_map[node_id]

        # Phase 1: parallel REPL calls
        repl_results = await self._async_pool.try_tactics_parallel(
            env_id, tactics)

        # Phase 2: sequential tree updates
        results = []
        for tactic, prior_val, repl_r in zip(tactics, priors, repl_results):
            t0 = time.perf_counter_ns()
            node = self._tree.get(NodeId(node_id))
            if not node:
                results.append(ExpansionResult(
                    node_id, tactic, False,
                    error={"kind": "not_found"}, elapsed_us=0))
                continue

            self._stats.nodes_expanded += 1
            elapsed = int((time.perf_counter_ns() - t0) / 1000) + \
                repl_r.elapsed_ms * 1000  # approximate

            if repl_r.success:
                complete = repl_r.is_proof_complete
                goals = [{"target": g} for g in repl_r.remaining_goals]

                local_result = execute_tactic(node.state, tactic)
                if local_result.success:
                    child_state = local_result.state
                else:
                    child_state = self._build_state_from_repl(
                        node.state, repl_r.remaining_goals, complete)

                result = self._handle_tactic_success(
                    node_id, tactic, child_state, prior_val, elapsed,
                    complete, goals)

                if result.child_node is not None:
                    self._node_env_map[result.child_node] = \
                        repl_r.new_env_id
                results.append(result)
            else:
                self._stats.l1_filtered += 1
                err = {"kind": repl_r.error_category,
                       "message": repl_r.error_message}
                results.append(ExpansionResult(
                    node_id, tactic, False,
                    error=err, elapsed_us=elapsed))

        self.release_virtual_loss(node_id)
        return results

    async def async_run_search(
            self,
            tactic_generator: Callable[[int], list[str]],
            prior_generator: Callable[[int], list[float]] = None,
            async_tactic_generator: Callable[
                [int], Awaitable[list[str]]] = None,
    ) -> SearchStats:
        """异步搜索主循环

        Args:
            tactic_generator: Sync function: node_id -> tactics.
            prior_generator: Sync function: node_id -> priors.
            async_tactic_generator: Async function: node_id -> tactics.
                If provided, used instead of tactic_generator.
                Useful when tactic generation involves LLM calls.
        """
        iteration = 0
        while True:
            node_id = self.select_node()
            if node_id is None:
                logger.debug(
                    f"Async search exhausted "
                    f"(expanded={self._stats.nodes_expanded})")
                break

            # Generate tactics (possibly via async LLM)
            if async_tactic_generator:
                tactics = await async_tactic_generator(node_id)
            else:
                tactics = tactic_generator(node_id)

            priors = (prior_generator(node_id) if prior_generator
                      else [0.5] * len(tactics))

            results = await self.async_try_batch(
                node_id, tactics, priors)

            if any(r.is_complete for r in results):
                logger.info(
                    f"Async search solved in "
                    f"{self._stats.nodes_expanded} expansions, "
                    f"depth={self._stats.solution_depth}")
                break

            iteration += 1
            if iteration % 50 == 0:
                logger.debug(
                    f"Async search progress: "
                    f"{self._stats.nodes_expanded} expanded, "
                    f"{len(self._tree.open_leaves())} open, "
                    f"max_depth={self._stats.max_depth_reached}")

            # Yield control to event loop periodically
            if iteration % 10 == 0:
                await asyncio.sleep(0)

        self._stats.time_ms = \
            (time.perf_counter() - self._start_time) * 1000
        return self._stats
