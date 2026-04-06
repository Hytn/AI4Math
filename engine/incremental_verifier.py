"""engine/incremental_verifier.py — 增量证明验证器

核心思想: 利用 Lean4 REPL 的 env_id 不可变快照语义,
将"完整重编译"变为"从分叉点开始的增量验证"。

对比:
  传统方式: 修改第 5 步 → 重新编译 import + theorem + 全部 5 步 (2-12s)
  增量方式: 修改第 5 步 → 从第 4 步的 env_id fork + 只验证第 5 步 (50-200ms)

关键前提: ProofSessionManager 维护了完整的 env_id 状态树,
每个步骤对应一个不可变的 env_id 快照。

典型场景:
  1. Agent 生成 5 步 proof → 第 3 步失败 → 修改第 3 步
     → 从 step 2 的 env_id fork → 只验证新的 step 3 + step 4 + step 5
  2. 探索分叉: 在 step 2 处同时尝试 3 条不同的 tactic
     → 3 路 fork 从同一个 env_id, 并行验证

Usage::

    verifier = IncrementalVerifier(pool, session_manager)

    # 验证完整 tactic 脚本
    result = await verifier.verify_script(
        theorem="theorem t : 1+1=2 := by",
        tactics=["norm_num"])

    # 修改某一步后增量验证
    result = await verifier.verify_edit(
        session_id="proof_0",
        edit_step=3,
        new_tactic="omega")

    # 并行探索多条路径
    results = await verifier.explore_alternatives(
        session_id="proof_0",
        at_step=2,
        alternatives=["simp", "ring", "omega"])
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from engine.lean_pool import TacticFeedback
from engine.proof_session import (
    ProofSessionManager, ProofSession, ProofSessionState, EnvNode,
)

logger = logging.getLogger(__name__)


@dataclass
class IncrementalResult:
    """增量验证的结果"""
    success: bool
    steps_verified: int              # 实际验证的步骤数 (vs 完整重编译的步数)
    steps_reused: int                # 复用的步骤数 (从缓存的 env_id)
    total_steps: int
    proof_complete: bool = False
    failed_step: int = -1            # 失败的步骤索引 (-1 = 全部成功)
    failed_tactic: str = ""
    error_message: str = ""
    remaining_goals: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    proof_script: str = ""           # 成功时的完整 proof script

    # 每步的详细结果
    step_results: list[TacticFeedback] = field(default_factory=list)

    @property
    def speedup(self) -> float:
        """相对于完整验证的加速比"""
        if self.total_steps == 0:
            return 1.0
        return self.total_steps / max(1, self.steps_verified)


class IncrementalVerifier:
    """增量证明验证器

    三种验证模式:
      1. verify_script: 完整 tactic 脚本, 逐步验证并缓存 env_id
      2. verify_edit:   编辑某一步后, 从分叉点增量验证
      3. explore_alternatives: 在某一步并行尝试多条 tactic
    """

    def __init__(self, pool: 'AsyncLeanPool',
                 session_manager: ProofSessionManager = None):
        self._pool = pool
        self._mgr = session_manager or ProofSessionManager(pool)
        self._stats = {
            "total_verifications": 0,
            "total_steps_verified": 0,
            "total_steps_reused": 0,
        }

    async def verify_script(self, theorem: str,
                            tactics: list[str],
                            session_id: str = "") -> IncrementalResult:
        """验证完整 tactic 脚本, 逐步执行并缓存每步 env_id

        这是最基本的验证模式:
        - 第一次验证: 逐步执行, 缓存 N 个 env_id
        - 后续修改: 可通过 verify_edit() 从任意步骤增量验证

        Returns:
            IncrementalResult 包含每步详细结果
        """
        t0 = time.time()
        self._stats["total_verifications"] += 1

        session = await self._mgr.begin_proof(theorem, session_id=session_id)
        step_results = []
        steps_verified = 0

        for i, tactic in enumerate(tactics):
            result = await session.try_step(tactic)
            step_results.append(result)
            steps_verified += 1

            if not result.success:
                elapsed = int((time.time() - t0) * 1000)
                self._stats["total_steps_verified"] += steps_verified
                return IncrementalResult(
                    success=False,
                    steps_verified=steps_verified,
                    steps_reused=0,
                    total_steps=len(tactics),
                    failed_step=i,
                    failed_tactic=tactic,
                    error_message=result.error_message,
                    remaining_goals=result.remaining_goals,
                    elapsed_ms=elapsed,
                    step_results=step_results)

            if result.is_proof_complete:
                break

        elapsed = int((time.time() - t0) * 1000)
        self._stats["total_steps_verified"] += steps_verified

        is_complete = session.is_solved
        proof = session.get_proof_script() if is_complete else ""

        return IncrementalResult(
            success=True,
            steps_verified=steps_verified,
            steps_reused=0,
            total_steps=len(tactics),
            proof_complete=is_complete,
            remaining_goals=session.current_goals,
            elapsed_ms=elapsed,
            proof_script=proof,
            step_results=step_results)

    async def verify_edit(self, session_id: str,
                          edit_step: int,
                          new_tactic: str,
                          subsequent_tactics: list[str] = None,
                          ) -> IncrementalResult:
        """编辑某一步后增量验证

        核心逻辑:
          1. 回退到 edit_step - 1 的 env_id (零成本: env_id 是不可变快照)
          2. 执行新的 tactic (仅验证这一步)
          3. 如果有后续 tactic, 继续验证

        加速比: O(1) 回退 + O(M) 验证 (M = 从编辑点到末尾的步数)
        vs 完整重编译: O(N) (N = 全部步数)
        """
        t0 = time.time()
        self._stats["total_verifications"] += 1

        session = self._mgr.get_session(session_id)
        if not session:
            return IncrementalResult(
                success=False, steps_verified=0, steps_reused=0,
                total_steps=0,
                error_message=f"Session '{session_id}' not found")

        # 回退到编辑点之前 (零成本)
        current_depth = session.current_depth
        rewind_steps = max(0, current_depth - edit_step)
        if rewind_steps > 0:
            session.rewind(steps=rewind_steps)

        steps_reused = edit_step  # 编辑点之前的步骤复用
        steps_verified = 0
        step_results = []

        # 执行新的 tactic
        result = await session.try_step(new_tactic)
        step_results.append(result)
        steps_verified += 1

        if not result.success:
            elapsed = int((time.time() - t0) * 1000)
            self._stats["total_steps_verified"] += steps_verified
            self._stats["total_steps_reused"] += steps_reused
            return IncrementalResult(
                success=False,
                steps_verified=steps_verified,
                steps_reused=steps_reused,
                total_steps=steps_reused + 1 + len(subsequent_tactics or []),
                failed_step=edit_step,
                failed_tactic=new_tactic,
                error_message=result.error_message,
                remaining_goals=result.remaining_goals,
                elapsed_ms=int((time.time() - t0) * 1000),
                step_results=step_results)

        # 执行后续 tactic (如果有)
        for i, tactic in enumerate(subsequent_tactics or []):
            if result.is_proof_complete:
                break
            result = await session.try_step(tactic)
            step_results.append(result)
            steps_verified += 1
            if not result.success:
                elapsed = int((time.time() - t0) * 1000)
                self._stats["total_steps_verified"] += steps_verified
                self._stats["total_steps_reused"] += steps_reused
                total = steps_reused + steps_verified
                return IncrementalResult(
                    success=False,
                    steps_verified=steps_verified,
                    steps_reused=steps_reused,
                    total_steps=total,
                    failed_step=edit_step + 1 + i,
                    failed_tactic=tactic,
                    error_message=result.error_message,
                    remaining_goals=result.remaining_goals,
                    elapsed_ms=int((time.time() - t0) * 1000),
                    step_results=step_results)

        elapsed = int((time.time() - t0) * 1000)
        self._stats["total_steps_verified"] += steps_verified
        self._stats["total_steps_reused"] += steps_reused

        total = steps_reused + steps_verified
        return IncrementalResult(
            success=True,
            steps_verified=steps_verified,
            steps_reused=steps_reused,
            total_steps=total,
            proof_complete=session.is_solved,
            remaining_goals=session.current_goals,
            elapsed_ms=elapsed,
            proof_script=session.get_proof_script() if session.is_solved else "",
            step_results=step_results)

    async def explore_alternatives(self, session_id: str,
                                   at_step: int,
                                   alternatives: list[str],
                                   ) -> list[IncrementalResult]:
        """在指定步骤并行尝试多条 tactic

        每条 tactic 从同一个 env_id fork, 互不影响。
        用于 beam search / MCTS 宽度搜索。

        Returns:
            每条 tactic 的 IncrementalResult 列表
        """
        t0 = time.time()
        session = self._mgr.get_session(session_id)
        if not session:
            return [IncrementalResult(
                success=False, steps_verified=0, steps_reused=0,
                total_steps=0,
                error_message=f"Session '{session_id}' not found")]

        # 确保在正确的步骤
        current_depth = session.current_depth
        if current_depth != at_step:
            rewind = max(0, current_depth - at_step)
            if rewind > 0:
                session.rewind(steps=rewind)

        # 并行尝试所有替代 (不改变 session.current)
        results = await session.try_alternatives(alternatives)

        # 将 TacticFeedback 转为 IncrementalResult
        incremental_results = []
        for r in results:
            elapsed = int((time.time() - t0) * 1000)
            incremental_results.append(IncrementalResult(
                success=r.success,
                steps_verified=1,
                steps_reused=at_step,
                total_steps=at_step + 1,
                proof_complete=r.is_proof_complete if r.success else False,
                failed_step=at_step if not r.success else -1,
                failed_tactic=r.tactic if not r.success else "",
                error_message=r.error_message if not r.success else "",
                remaining_goals=r.remaining_goals,
                elapsed_ms=elapsed,
                step_results=[r]))

        return incremental_results

    def stats(self) -> dict:
        s = self._stats
        total_steps = s["total_steps_verified"] + s["total_steps_reused"]
        return {
            **s,
            "total_steps": total_steps,
            "reuse_rate": round(
                s["total_steps_reused"] / max(1, total_steps), 3),
            "avg_speedup": round(
                total_steps / max(1, s["total_steps_verified"]), 2),
        }
