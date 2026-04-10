"""engine/verification_scheduler.py — 自适应验证调度器

将验证请求智能路由到 L0/L1/L2 三级验证层:
  L0 PreFilter:    纯语法检查, ~1μs,  过滤 ~90% 明显无效输出
  L1 REPL Pool:    Lean4 增量验证, ~50-500ms, 精确的类型检查
  L2 Full Compile: 完整 Lean4 编译, ~2-12s, 最终可信认证

调度策略:
  - 所有请求先过 L0 (零成本, 无理由跳过)
  - L0 通过后进入 L1 (REPL 池, 增量验证)
  - L1 通过且需要最终认证时进入 L2 (完整编译)

信息流:
  - L0 的拒绝原因 → 直接反馈给 Agent (含修复建议)
  - L1 的结果 → 通过 ErrorIntelligence 丰富化后反馈
  - L2 的结果 → 最终裁决, 决定是否计入 pass@k

与广播总线集成:
  - L1 发现有效 tactic → 广播 POSITIVE_DISCOVERY
  - L1 发现必然失败 → 广播 NEGATIVE_KNOWLEDGE
  - L1 部分成功 → 广播 PARTIAL_PROOF
"""
from __future__ import annotations
import shutil
import os
import subprocess
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from engine.prefilter import PreFilter, FilterResult
from engine.lean_pool import LeanPool, TacticFeedback, FullVerifyResult
from engine.error_intelligence import ErrorIntelligence, AgentFeedback
from engine.broadcast import BroadcastBus, BroadcastMessage, MessageType
from engine.observability import metrics

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """三级验证的统一结果"""
    # 最终判定
    success: bool
    level_reached: str  # "L0", "L1", "L2"

    # 结构化反馈 (始终存在, 即使成功)
    feedback: AgentFeedback = field(default_factory=AgentFeedback)

    # L0 相关
    l0_passed: bool = True
    l0_reject_reason: str = ""
    l0_fix_hint: str = ""

    # L1 相关 (REPL)
    l1_env_id: int = -1
    l1_goals_remaining: list[str] = field(default_factory=list)

    # L2 相关 (完整编译)
    l2_verified: bool = False

    # 性能
    l0_us: int = 0       # 微秒
    l1_ms: int = 0       # 毫秒
    l2_ms: int = 0       # 毫秒
    total_ms: int = 0

    # 广播消息 (调度器自动发出的)
    broadcast_sent: list[str] = field(default_factory=list)


class VerificationScheduler:
    """自适应三级验证调度器

    核心逻辑: L0 过滤 → L1 增量验证 → L2 最终认证

    与传统方案的对比:
      传统: LLM → Lean4 完整编译 (2-12s) → pass/fail (1 bit)
      本方案: LLM → L0 (1μs) → L1 REPL (50ms) → 结构化反馈 (~100 bits)
             仅有 ~2% 的候选需要进入 L2 完整编译

    Usage::

        scheduler = VerificationScheduler(
            prefilter=PreFilter(),
            lean_pool=pool,
            error_intel=ErrorIntelligence(pool),
            broadcast=bus,
        )

        # 验证单条 tactic
        result = scheduler.verify_tactic(env_id=0, tactic="simp",
                                         direction="structured")

        # 验证完整证明
        result = scheduler.verify_complete(
            theorem="theorem t : 1 + 1 = 2",
            proof=":= by norm_num",
            direction="automation")
    """

    def __init__(self, prefilter: PreFilter = None,
                 lean_pool: LeanPool = None,
                 error_intel: ErrorIntelligence = None,
                 broadcast: BroadcastBus = None,
                 project_dir: str = "."):
        self.prefilter = prefilter or PreFilter()
        self.pool = lean_pool
        self.error_intel = error_intel or ErrorIntelligence(lean_pool)
        self.broadcast = broadcast
        self.project_dir = project_dir

        # 指标收集
        from engine.observability import metrics as _metrics
        self._metrics = _metrics

        # 统计
        self._stats = {
            "total": 0,
            "l0_rejected": 0,
            "l1_passed": 0,
            "l1_failed": 0,
            "l2_verified": 0,
            "l2_failed": 0,
        }

    def verify_tactic(self, env_id: int, tactic: str,
                      goals_before: int = 1,
                      direction: str = "",
                      ) -> VerificationResult:
        """验证单条 tactic (L0 + L1)

        这是 Agent 搜索过程中最频繁的操作。
        每次调用完成后, 自动通过广播总线通知其他方向。
        """
        t0 = time.time()
        self._stats["total"] += 1

        # ── L0: 语法预过滤 ──
        l0_start = time.perf_counter_ns()
        l0_result = self.prefilter.check(tactic)
        l0_us = int((time.perf_counter_ns() - l0_start) / 1000)

        if not l0_result.passed:
            self._stats["l0_rejected"] += 1
            total_ms = int((time.time() - t0) * 1000)

            # 广播负面知识
            self._broadcast_negative(direction, tactic,
                                     l0_result.rule_name, l0_result.reason)

            return VerificationResult(
                success=False, level_reached="L0",
                l0_passed=False,
                l0_reject_reason=l0_result.reason,
                l0_fix_hint=l0_result.fix_hint,
                l0_us=l0_us, total_ms=total_ms,
                feedback=AgentFeedback.from_failure(
                    tactic, l0_result.reason, l0_result.rule_name),
            )

        # ── L1: REPL 增量验证 ──
        if not self.pool:
            total_ms = int((time.time() - t0) * 1000)
            return VerificationResult(
                success=False, level_reached="L0",
                l0_passed=True, l0_us=l0_us, total_ms=total_ms,
                feedback=AgentFeedback.from_failure(
                    tactic, "No Lean4 REPL available", "no_backend"),
            )

        tactic_result = self.pool.try_tactic(env_id, tactic)
        feedback = self.error_intel.analyze(
            tactic_result, goals_before, parent_env_id=env_id)
        total_ms = int((time.time() - t0) * 1000)

        if tactic_result.success:
            self._stats["l1_passed"] += 1

            # 广播正面发现
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
                total_ms=total_ms,
            )
        else:
            self._stats["l1_failed"] += 1

            # 广播负面知识 (但只在确定性失败时)
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
                total_ms=total_ms,
            )

    def verify_complete(self, theorem: str, proof: str,
                        direction: str = "",
                        require_l2: bool = False,
                        ) -> VerificationResult:
        """验证完整证明 (L0 + L1, 可选 L2)

        Args:
            theorem: 定理声明
            proof: 证明代码
            direction: 发起验证的方向名称 (用于广播标记)
            require_l2: 是否要求 L2 完整编译认证

        Returns:
            VerificationResult
        """
        t0 = time.time()
        self._stats["total"] += 1

        # ── L0: 语法预过滤 ──
        l0_start = time.perf_counter_ns()
        l0_result = self.prefilter.check(proof, theorem)
        l0_us = int((time.perf_counter_ns() - l0_start) / 1000)
        metrics.record_time("verify.l0_prefilter", l0_us / 1000)  # convert to ms

        if not l0_result.passed:
            self._stats["l0_rejected"] += 1
            metrics.increment("verify.l0_rejected")
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
                    proof[:50], l0_result.reason, l0_result.rule_name),
            )

        # ── L1: REPL 验证 ──
        if self.pool:
            with metrics.timer("verify.l1_repl"):
                l1_result = self.pool.verify_complete(theorem, proof)
            l1_ms = l1_result.elapsed_ms

            if l1_result.success:
                self._stats["l1_passed"] += 1
                metrics.increment("verify.l1_passed")

                if not require_l2:
                    total_ms = int((time.time() - t0) * 1000)
                    self._broadcast_positive(
                        direction, f"Complete proof verified!")
                    return VerificationResult(
                        success=True, level_reached="L1",
                        l0_passed=True, l0_us=l0_us,
                        l1_ms=l1_ms, l1_env_id=l1_result.env_id,
                        total_ms=total_ms,
                        feedback=AgentFeedback(
                            is_proof_complete=True,
                            progress_score=1.0),
                    )
                # 继续到 L2
            else:
                self._stats["l1_failed"] += 1
                metrics.increment("verify.l1_failed")
                total_ms = int((time.time() - t0) * 1000)
                feedback = AgentFeedback.from_failure(
                    proof[:50],
                    l1_result.stderr[:300],
                    "lean_error",
                    goals=l1_result.goals_remaining,
                )
                # 生成修复候选
                if self.error_intel:
                    tac_fb = TacticFeedback(
                        success=False, tactic=proof[:50],
                        error_message=l1_result.stderr[:300],
                        error_category="lean_error",
                        new_env_id=l1_result.env_id,
                    )
                    feedback = self.error_intel.analyze(tac_fb, 1, False)
                return VerificationResult(
                    success=False, level_reached="L1",
                    feedback=feedback,
                    l0_passed=True, l0_us=l0_us,
                    l1_ms=l1_ms, total_ms=total_ms,
                )

        # ── L2: 独立完整编译 (最终认证, 不复用 REPL 池) ──
        if require_l2:
            l2_t0 = time.time()
            l2_result = self._l2_full_compile(theorem, proof)
            l2_ms = int((time.time() - l2_t0) * 1000)
            total_ms = int((time.time() - t0) * 1000)

            if l2_result.success:
                self._stats["l2_verified"] += 1
                self._broadcast_positive(direction, "L2 CERTIFIED: proof is valid!")
                return VerificationResult(
                    success=True, level_reached="L2",
                    l2_verified=True,
                    l0_passed=True, l0_us=l0_us,
                    l2_ms=l2_ms, total_ms=total_ms,
                    feedback=AgentFeedback(
                        is_proof_complete=True, progress_score=1.0),
                )
            else:
                self._stats["l2_failed"] += 1
                return VerificationResult(
                    success=False, level_reached="L2",
                    l0_passed=True, l0_us=l0_us,
                    l2_ms=l2_ms, total_ms=total_ms,
                    feedback=AgentFeedback.from_failure(
                        proof[:50], l2_result.stderr[:300], "lean_error"),
                )

        total_ms = int((time.time() - t0) * 1000)
        return VerificationResult(
            success=False, level_reached="L0",
            l0_passed=True, l0_us=l0_us, total_ms=total_ms,
            feedback=AgentFeedback.from_failure(
                proof[:50], "No verification backend available", "no_backend"),
        )

    def verify_tactics_parallel(self, env_id: int, tactics: list[str],
                                goals_before: int = 1,
                                direction: str = "",
                                ) -> list[VerificationResult]:
        """并行验证多条 tactic (MCTS 搜索树展开)

        每条 tactic 走 L0 → L1 管线。
        成功/失败的结果自动广播给所有方向。

        Returns:
            list[VerificationResult], 与 tactics 一一对应
        """
        # L0 预过滤
        passed = []
        results = [None] * len(tactics)

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
                        tactic, l0.reason, l0.rule_name),
                )
            else:
                passed.append((i, tactic))

        # L1 并行验证 (通过 REPL 池)
        if passed and self.pool:
            tactic_list = [t for _, t in passed]
            l1_results = self.pool.try_tactics_parallel(env_id, tactic_list)

            for (orig_idx, tactic), l1_result in zip(passed, l1_results):
                feedback = self.error_intel.analyze(l1_result, goals_before,
                                                    use_search_tactics=False,
                                                    parent_env_id=env_id)
                if l1_result.success:
                    self._stats["l1_passed"] += 1
                    if l1_result.is_proof_complete:
                        self._broadcast_positive(
                            direction, f"`{tactic}` completed the proof!")
                    results[orig_idx] = VerificationResult(
                        success=True, level_reached="L1",
                        feedback=feedback,
                        l0_passed=True,
                        l1_ms=l1_result.elapsed_ms,
                        l1_env_id=l1_result.new_env_id,
                        l1_goals_remaining=l1_result.remaining_goals,
                        total_ms=l1_result.elapsed_ms,
                    )
                else:
                    self._stats["l1_failed"] += 1
                    results[orig_idx] = VerificationResult(
                        success=False, level_reached="L1",
                        feedback=feedback,
                        l0_passed=True,
                        l1_ms=l1_result.elapsed_ms,
                        total_ms=l1_result.elapsed_ms,
                    )
        elif passed:
            # 无 REPL 池, 返回未验证状态
            for orig_idx, tactic in passed:
                results[orig_idx] = VerificationResult(
                    success=False, level_reached="L0",
                    l0_passed=True,
                    feedback=AgentFeedback.from_failure(
                        tactic, "No REPL available", "no_backend"),
                )

        self._stats["total"] += len(tactics)
        return results

    # ── L2: 独立完整编译 (不复用 REPL 池) ──

    def _l2_full_compile(self, theorem: str, proof: str,
                         preamble: str = "import Mathlib",
                         timeout: int = 120) -> FullVerifyResult:
        """L2 最终认证: 在独立 subprocess 中从零编译完整 Lean4 文件。

        P1-6 修复:
          - 写入临时 .lean 文件 (而非 stdin pipe), 避免 --run 误用
          - 优先使用 `lake env lean` (正确的编译命令)
          - 更精确的错误判定: 使用正则匹配避免误判
        """
        import tempfile
        import re
        from engine._core import assemble_code as _assemble_code

        # 使用统一的 assemble_code (与 LeanSession 保持一致)
        full_code = _assemble_code(theorem, proof, preamble)

        # 使用 TemporaryDirectory 确保完整清理 (包括目录本身)
        try:
            with tempfile.TemporaryDirectory(
                    prefix="ai4math_l2_",
                    dir=self.project_dir) as tmp_dir:
                tmp_path = os.path.join(tmp_dir, "verify.lean")
                with open(tmp_path, 'w') as f:
                    f.write(full_code)

                # 选择编译命令 (优先 lake env lean)
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
                        stderr="L2 full compile unavailable: neither lake nor lean found. "
                               "Install elan + lean4 to enable L2 certification.",
                    )

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.project_dir,
                )
                stderr = result.stderr or ""
                stdout = result.stdout or ""
                combined = stderr + stdout

                # P1-6: 精确判定 — 使用 Lean4 错误输出格式匹配
                _ERROR_RE = re.compile(
                    r'(?:^|\n)\S+:\d+:\d+:\s*error\b|'
                    r"\bdeclaration uses 'sorry'\b|"
                    r'\bunsolved goals\b',
                    re.IGNORECASE
                )
                has_error = bool(_ERROR_RE.search(combined))
                has_sorry = "sorry" in combined.lower()
                success = result.returncode == 0 and not has_error

                return FullVerifyResult(
                    success=success,
                    stderr=stderr,
                    has_sorry=has_sorry,
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"L2: lean compile timed out after {timeout}s")
            return FullVerifyResult(
                success=False,
                stderr=f"L2 compile timed out after {timeout}s",
            )
        except Exception as e:
            logger.error(f"L2: lean compile failed: {e}")
            return FullVerifyResult(
                success=False,
                stderr=f"L2 compile error: {e}",
            )

    # ── 广播辅助 ──

    def _broadcast_negative(self, direction: str, tactic: str,
                            error_cat: str, reason: str):
        if self.broadcast and direction:
            self.broadcast.publish(BroadcastMessage.negative(
                source=direction, tactic=tactic,
                error_category=error_cat, reason=reason))

    def _broadcast_positive(self, direction: str, discovery: str):
        if self.broadcast and direction:
            self.broadcast.publish(BroadcastMessage.positive(
                source=direction, discovery=discovery))

    def _broadcast_partial(self, direction: str, proof: str,
                           remaining: list[str], env_id: int, closed: int):
        if self.broadcast and direction:
            self.broadcast.publish(BroadcastMessage.partial_proof(
                source=direction, proof_so_far=proof,
                remaining_goals=remaining,
                env_id=env_id, goals_closed=closed))

    def stats(self) -> dict:
        s = self._stats
        total = max(1, s["total"])
        result = {
            **s,
            "l0_filter_rate": round(s["l0_rejected"] / total, 3),
            "l1_pass_rate": round(s["l1_passed"] / total, 3),
            "pool_stats": self.pool.stats() if self.pool else {},
        }
        # 附加 observability metrics (如果有)
        try:
            metrics_snapshot = self._metrics.snapshot()
            if metrics_snapshot:
                result["metrics"] = metrics_snapshot
        except Exception as _exc:
            logger.debug(f"Suppressed exception: {_exc}")
        return result
