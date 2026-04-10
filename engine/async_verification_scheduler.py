"""engine/async_verification_scheduler.py — 异步三级验证调度器

与同步 VerificationScheduler 共享 VerificationResult 数据类型,
将 L1 REPL 调用和 L2 编译改为 async。

L0 仍然同步 (纯 CPU, <10μs, 不值得异步化)。
"""
from __future__ import annotations
import asyncio
import logging
import os
import shutil
import time
from typing import Optional

from engine.prefilter import PreFilter, FilterResult
from engine.async_lean_pool import AsyncLeanPool
from engine.lean_pool import TacticFeedback, FullVerifyResult
from engine.error_intelligence import ErrorIntelligence, AgentFeedback
from engine.broadcast import BroadcastBus, BroadcastMessage
from engine.verification_scheduler import VerificationResult

logger = logging.getLogger(__name__)


class AsyncVerificationScheduler:
    """异步三级验证调度器

    性能改进:
      - verify_tactics_parallel: L0 同步过滤 + L1 asyncio.gather
      - verify_complete: L1 async REPL, L2 async subprocess
      - 多个证明候选可真正并行验证, 不阻塞 LLM 调用

    lean_pool 可以是 AsyncLeanPool 或 ElasticPool (鸭子类型, 需满足
    try_tactic / try_tactics_parallel / verify_complete / stats 接口)。
    """

    def __init__(self, prefilter: PreFilter = None,
                 lean_pool=None,
                 error_intel: ErrorIntelligence = None,
                 broadcast: BroadcastBus = None,
                 project_dir: str = "."):
        self.prefilter = prefilter or PreFilter()
        self.pool = lean_pool
        self.error_intel = error_intel or ErrorIntelligence()
        self.broadcast = broadcast
        self.project_dir = project_dir
        self._stats = {
            "total": 0, "l0_rejected": 0,
            "l1_passed": 0, "l1_failed": 0,
            "l2_verified": 0, "l2_failed": 0,
        }

    async def verify_tactic(self, env_id: int, tactic: str,
                            goals_before: int = 1,
                            direction: str = "") -> VerificationResult:
        """验证单条 tactic (L0 + L1)"""
        t0 = time.time()
        self._stats["total"] += 1

        # L0: 同步 (纯 CPU, <10μs)
        l0_start = time.perf_counter_ns()
        l0_result = self.prefilter.check(tactic)
        l0_us = int((time.perf_counter_ns() - l0_start) / 1000)

        if not l0_result.passed:
            self._stats["l0_rejected"] += 1
            total_ms = int((time.time() - t0) * 1000)
            self._broadcast_negative(direction, tactic,
                                     l0_result.rule_name, l0_result.reason)
            return VerificationResult(
                success=False, level_reached="L0",
                l0_passed=False,
                l0_reject_reason=l0_result.reason,
                l0_fix_hint=l0_result.fix_hint,
                l0_us=l0_us, total_ms=total_ms,
                feedback=AgentFeedback.from_failure(
                    tactic, l0_result.reason, l0_result.rule_name))

        # L1: 异步 REPL
        if not self.pool:
            total_ms = int((time.time() - t0) * 1000)
            return VerificationResult(
                success=False, level_reached="L0",
                l0_passed=True, l0_us=l0_us, total_ms=total_ms,
                feedback=AgentFeedback.from_failure(
                    tactic, "No Lean4 REPL available", "no_backend"))

        tactic_result = await self.pool.try_tactic(env_id, tactic)
        feedback = self.error_intel.analyze(
            tactic_result, goals_before, parent_env_id=env_id)
        total_ms = int((time.time() - t0) * 1000)

        if tactic_result.success:
            self._stats["l1_passed"] += 1
            if tactic_result.is_proof_complete:
                self._broadcast_positive(
                    direction, f"Proof completed using `{tactic}`!")
            elif tactic_result.goals_closed > 0:
                self._broadcast_partial(
                    direction, tactic,
                    tactic_result.remaining_goals,
                    tactic_result.new_env_id,
                    tactic_result.goals_closed)
            return VerificationResult(
                success=True, level_reached="L1",
                feedback=feedback,
                l0_passed=True, l0_us=l0_us,
                l1_ms=tactic_result.elapsed_ms,
                l1_env_id=tactic_result.new_env_id,
                l1_goals_remaining=tactic_result.remaining_goals,
                total_ms=total_ms)
        else:
            self._stats["l1_failed"] += 1
            if tactic_result.error_category in (
                    "type_mismatch", "unknown_identifier", "tactic_failed"):
                self._broadcast_negative(
                    direction, tactic,
                    tactic_result.error_category,
                    tactic_result.error_message[:100])
            return VerificationResult(
                success=False, level_reached="L1",
                feedback=feedback,
                l0_passed=True, l0_us=l0_us,
                l1_ms=tactic_result.elapsed_ms,
                total_ms=total_ms)

    async def verify_complete(self, theorem: str, proof: str,
                              direction: str = "",
                              require_l2: bool = False) -> VerificationResult:
        """验证完整证明 (L0 + async L1, 可选 async L2)"""
        t0 = time.time()
        self._stats["total"] += 1

        # L0
        l0_start = time.perf_counter_ns()
        l0_result = self.prefilter.check(proof, theorem)
        l0_us = int((time.perf_counter_ns() - l0_start) / 1000)

        if not l0_result.passed:
            self._stats["l0_rejected"] += 1
            total_ms = int((time.time() - t0) * 1000)
            self._broadcast_negative(direction, proof[:50],
                                     l0_result.rule_name, l0_result.reason)
            return VerificationResult(
                success=False, level_reached="L0",
                l0_passed=False,
                l0_reject_reason=l0_result.reason,
                l0_fix_hint=l0_result.fix_hint,
                l0_us=l0_us, total_ms=total_ms,
                feedback=AgentFeedback.from_failure(
                    proof[:50], l0_result.reason, l0_result.rule_name))

        # L1: async REPL
        if self.pool:
            l1_result = await self.pool.verify_complete(theorem, proof)
            l1_ms = l1_result.elapsed_ms

            if l1_result.success:
                self._stats["l1_passed"] += 1
                if not require_l2:
                    total_ms = int((time.time() - t0) * 1000)
                    self._broadcast_positive(direction, "Complete proof verified!")
                    return VerificationResult(
                        success=True, level_reached="L1",
                        l0_passed=True, l0_us=l0_us,
                        l1_ms=l1_ms, l1_env_id=l1_result.env_id,
                        total_ms=total_ms,
                        feedback=AgentFeedback(
                            is_proof_complete=True, progress_score=1.0))
            else:
                self._stats["l1_failed"] += 1
                if not require_l2:
                    total_ms = int((time.time() - t0) * 1000)
                    feedback = AgentFeedback.from_failure(
                        proof[:50], l1_result.stderr[:300], "lean_error",
                        goals=l1_result.goals_remaining)
                    return VerificationResult(
                        success=False, level_reached="L1",
                        feedback=feedback,
                        l0_passed=True, l0_us=l0_us,
                        l1_ms=l1_ms, total_ms=total_ms)

        # L2: async full compile
        if require_l2:
            l2_result = await self._l2_full_compile(theorem, proof)
            l2_ms = l2_result.elapsed_ms
            total_ms = int((time.time() - t0) * 1000)

            if l2_result.success:
                self._stats["l2_verified"] += 1
                self._broadcast_positive(direction, "L2 CERTIFIED!")
                return VerificationResult(
                    success=True, level_reached="L2",
                    l2_verified=True,
                    l0_passed=True, l0_us=l0_us,
                    l2_ms=l2_ms, total_ms=total_ms,
                    feedback=AgentFeedback(
                        is_proof_complete=True, progress_score=1.0))
            else:
                self._stats["l2_failed"] += 1
                return VerificationResult(
                    success=False, level_reached="L2",
                    l0_passed=True, l0_us=l0_us,
                    l2_ms=l2_ms, total_ms=total_ms,
                    feedback=AgentFeedback.from_failure(
                        proof[:50], l2_result.stderr[:300], "lean_error"))

        total_ms = int((time.time() - t0) * 1000)
        return VerificationResult(
            success=False, level_reached="L0",
            l0_passed=True, l0_us=l0_us, total_ms=total_ms,
            feedback=AgentFeedback.from_failure(
                proof[:50], "No backend available", "no_backend"))

    async def verify_tactics_parallel(self, env_id: int,
                                      tactics: list[str],
                                      goals_before: int = 1,
                                      direction: str = "") -> list[VerificationResult]:
        """并行验证多条 tactic — L0 同步过滤 + L1 asyncio.gather"""
        passed = []
        results = [None] * len(tactics)

        # L0 过滤 (同步, <10μs/条)
        for i, tactic in enumerate(tactics):
            l0 = self.prefilter.check(tactic)
            if not l0.passed:
                self._stats["l0_rejected"] += 1
                results[i] = VerificationResult(
                    success=False, level_reached="L0",
                    l0_passed=False,
                    l0_reject_reason=l0.reason,
                    l0_fix_hint=l0.fix_hint,
                    feedback=AgentFeedback.from_failure(
                        tactic, l0.reason, l0.rule_name))
            else:
                passed.append((i, tactic))

        # L1 并行 (async)
        if passed and self.pool:
            tactic_list = [t for _, t in passed]
            l1_results = await self.pool.try_tactics_parallel(
                env_id, tactic_list)

            for (orig_idx, tactic), l1_result in zip(passed, l1_results):
                feedback = self.error_intel.analyze(
                    l1_result, goals_before,
                    use_search_tactics=False, parent_env_id=env_id)
                if l1_result.success:
                    self._stats["l1_passed"] += 1
                    if l1_result.is_proof_complete:
                        self._broadcast_positive(
                            direction, f"`{tactic}` completed the proof!")
                    results[orig_idx] = VerificationResult(
                        success=True, level_reached="L1",
                        feedback=feedback, l0_passed=True,
                        l1_ms=l1_result.elapsed_ms,
                        l1_env_id=l1_result.new_env_id,
                        l1_goals_remaining=l1_result.remaining_goals,
                        total_ms=l1_result.elapsed_ms)
                else:
                    self._stats["l1_failed"] += 1
                    results[orig_idx] = VerificationResult(
                        success=False, level_reached="L1",
                        feedback=feedback, l0_passed=True,
                        l1_ms=l1_result.elapsed_ms,
                        total_ms=l1_result.elapsed_ms)
        elif passed:
            for orig_idx, tactic in passed:
                results[orig_idx] = VerificationResult(
                    success=False, level_reached="L0",
                    l0_passed=True,
                    feedback=AgentFeedback.from_failure(
                        tactic, "No REPL available", "no_backend"))

        self._stats["total"] += len(tactics)
        return results

    async def _l2_full_compile(self, theorem: str, proof: str,
                                preamble: str = "import Mathlib",
                                timeout: int = 120) -> FullVerifyResult:
        """L2 异步编译 — asyncio.create_subprocess_exec + temp file"""
        import re
        import tempfile

        parts = [preamble, ""]
        full_stmt = theorem.strip()
        if proof.strip():
            if ":=" not in full_stmt:
                full_stmt += f" {proof.strip()}"
        parts.append(full_stmt)
        full_code = "\n".join(parts)

        tmp_dir = os.path.join(self.project_dir, ".ai4math_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.lean', dir=tmp_dir,
                    delete=False) as f:
                f.write(full_code)
                tmp_path = f.name

            lakefile = os.path.join(self.project_dir, "lakefile.lean")
            lake_bin = shutil.which("lake")
            lean_bin = shutil.which("lean")

            if lake_bin and os.path.isfile(lakefile):
                cmd = [lake_bin, "env", "lean", tmp_path]
            elif lean_bin:
                cmd = [lean_bin, tmp_path]
            else:
                return FullVerifyResult(
                    success=False,
                    stderr="L2: neither lake nor lean found")

            t0 = time.time()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_dir)

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout)

            elapsed = int((time.time() - t0) * 1000)
            stderr_text = stderr.decode(errors="replace")
            combined = stderr_text + stdout.decode(errors="replace")

            _ERROR_RE = re.compile(
                r'(?:^|\n)\S+:\d+:\d+:\s*error\b|'
                r"\bdeclaration uses 'sorry'\b|"
                r'\bunsolved goals\b',
                re.IGNORECASE)
            has_error = bool(_ERROR_RE.search(combined))
            success = proc.returncode == 0 and not has_error

            return FullVerifyResult(
                success=success, stderr=stderr_text,
                elapsed_ms=elapsed,
                has_sorry="sorry" in combined.lower())

        except asyncio.TimeoutError:
            return FullVerifyResult(
                success=False,
                stderr=f"L2 compile timed out after {timeout}s")
        except Exception as e:
            return FullVerifyResult(
                success=False, stderr=f"L2 compile error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError as _exc:
                    logger.debug(f"Suppressed exception: {_exc}")

    # ── 广播辅助 ──

    def _broadcast_negative(self, direction, tactic, error_cat, reason):
        if self.broadcast and direction:
            self.broadcast.publish(BroadcastMessage.negative(
                source=direction, tactic=tactic,
                error_category=error_cat, reason=reason))

    def _broadcast_positive(self, direction, discovery):
        if self.broadcast and direction:
            self.broadcast.publish(BroadcastMessage.positive(
                source=direction, discovery=discovery))

    def _broadcast_partial(self, direction, proof, remaining, env_id, closed):
        if self.broadcast and direction:
            self.broadcast.publish(BroadcastMessage.partial_proof(
                source=direction, proof_so_far=proof,
                remaining_goals=remaining,
                env_id=env_id, goals_closed=closed))

    def stats(self) -> dict:
        s = self._stats
        total = max(1, s["total"])
        return {
            **s,
            "l0_filter_rate": round(s["l0_rejected"] / total, 3),
            "l1_pass_rate": round(s["l1_passed"] / total, 3),
            "pool_stats": self.pool.stats() if self.pool else {},
        }
