"""prover/unified/adapters.py — Unified ↔ Legacy 数据桥

让 ``UnifiedProofRunner`` 的输出 (``UnifiedResult`` + ``LoopResult``)
能无缝喂给现有的 ``ProofPipeline`` / ``HeterogeneousEngine`` /
``ResultFuser`` 等组件 —— 它们仍然消费 ``ProofAttempt`` / ``AgentResult``。

数据形状对应::

    LoopResult.messages         ──┐
    UnifiedResult.profile_name    ├─→ dialog.json (持久化主格式)
    UnifiedResult.search_summary ──┘

    UnifiedResult.success         ──┐
    UnifiedResult.proof_code       ├─→ ProofAttempt   (ProofPipeline 内部)
    LoopResult.total_tokens       ──┘

    UnifiedResult.success         ──┐
    UnifiedResult.proof_code       ├─→ AgentResult    (HeterogeneousEngine fan-in)
    UnifiedResult.profile_name    ──┘

所有重构对外都不破坏旧 API: 旧测试、旧持久化、旧 SFT 导出脚本继续工作。
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from prover.models import (
    ProofAttempt, AttemptStatus, LeanError, ErrorCategory,
)
from prover.unified.runner import UnifiedResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# UnifiedResult → ProofAttempt
# ══════════════════════════════════════════════════════════════════════

def unified_to_attempt(
    result: UnifiedResult,
    *,
    attempt_number: int = 1,
) -> ProofAttempt:
    """``UnifiedResult`` → ``ProofAttempt`` (ProofPipeline 兼容路径)。

    错误从 dialog.messages 中的 tool_result 抽取; 如果 LoopResult 不可用
    则保留空错误列表。
    """
    loop = result.loop_result
    proof_code = result.proof_code or (loop.proof_code if loop else "")

    if result.success:
        status = AttemptStatus.SUCCESS
    elif loop and loop.stopped_reason == "timeout":
        status = AttemptStatus.TIMEOUT
    elif loop and loop.stopped_reason.startswith("error"):
        status = AttemptStatus.LLM_ERROR
    else:
        status = AttemptStatus.LEAN_ERROR

    lean_errors, stderr_blob = _extract_lean_errors_from_loop(loop)

    return ProofAttempt(
        attempt_number=attempt_number,
        generated_proof=proof_code,
        prompt_summary=(
            f"[unified.{result.profile_name}] "
            f"turns={loop.turns_used if loop else 0} "
            f"tools={','.join(loop.tools_called) if loop else ''}"
        ),
        llm_model=_get_loop_model(loop),
        llm_tokens_in=0,
        llm_tokens_out=loop.total_tokens if loop else 0,
        llm_latency_ms=loop.total_latency_ms if loop else result.total_duration_ms,
        lean_result=status,
        lean_errors=lean_errors,
        lean_stderr=stderr_blob[:2000],
        lean_check_ms=0,
        retrieved_premises=[],
        started_at=time.time() - (result.total_duration_ms or 0) / 1000.0,
        finished_at=time.time(),
        repair_rounds=max(0, (loop.turns_used - 1) if loop else 0),
    )


# ══════════════════════════════════════════════════════════════════════
# UnifiedResult → AgentResult (heterogeneous engine fan-in)
# ══════════════════════════════════════════════════════════════════════

def unified_to_agent_result(result: UnifiedResult, *, agent_name: str = ""):
    """``UnifiedResult`` → ``AgentResult`` (异构方向并行兼容)。

    ``HeterogeneousEngine`` 的下游 (ResultFuser / BroadcastBus / confidence
    sort) 全部消费 AgentResult, 这个 adapter 把统一 runtime 的输出"装扮"成
    旧 SubAgent 的输出形状。
    """
    from prover.pipeline._agent_deps import AgentResult
    from common.roles import AgentRole

    loop = result.loop_result
    name = agent_name or f"unified.{result.profile_name}"
    proof_code = result.proof_code or (loop.proof_code if loop else "")
    confidence = _confidence_heuristic(result)

    return AgentResult(
        agent_name=name,
        role=AgentRole.PROOF_GENERATOR,
        content=loop.content if loop else "",
        proof_code=proof_code,
        tool_calls=[{"tools_used": loop.tools_called if loop else []}],
        tokens_used=loop.total_tokens if loop else 0,
        latency_ms=loop.total_latency_ms if loop else result.total_duration_ms,
        confidence=confidence,
        success=result.success,
        error=("" if result.success
               else (loop.stopped_reason if loop else "no_result")),
        metadata={
            "profile_name": result.profile_name,
            "stopped_reason": loop.stopped_reason if loop else "",
            "turns_used": loop.turns_used if loop else 0,
            "search_summary": result.search_summary,
            "verification": {
                "success": result.success,
                "level": "L2" if result.success else "L1",
            },
        },
    )


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════

def _extract_lean_errors_from_loop(loop) -> tuple[list, str]:
    """从 LoopResult.messages 的 tool_result 内容里翻出 Lean 错误。"""
    if loop is None or not loop.messages:
        return [], ""

    errors = []
    stderr_parts = []

    for msg in loop.messages:
        # LoopMessage.tool_results 是字符串列表
        results = getattr(msg, "tool_results", None) or []
        for r in results:
            text = str(r) if r is not None else ""
            if not text:
                continue
            stderr_parts.append(text[:500])
            low = text.lower()
            if ("error" in low) or ("failed" in low):
                errors.append(LeanError(
                    category=_classify_category(low),
                    message=text[:300],
                ))

    return errors[:10], "\n---\n".join(stderr_parts[:5])


def _classify_category(text_lower: str) -> ErrorCategory:
    if "type mismatch" in text_lower or "expected type" in text_lower:
        return ErrorCategory.TYPE_MISMATCH
    if "unknown identifier" in text_lower or "unknown constant" in text_lower:
        return ErrorCategory.UNKNOWN_IDENTIFIER
    if "tactic" in text_lower and "failed" in text_lower:
        return ErrorCategory.TACTIC_FAILED
    if "syntax" in text_lower or "expected" in text_lower:
        return ErrorCategory.SYNTAX_ERROR
    if "timeout" in text_lower:
        return ErrorCategory.TIMEOUT
    if "import" in text_lower:
        return ErrorCategory.IMPORT_ERROR
    return ErrorCategory.OTHER


def _get_loop_model(loop) -> str:
    if loop is None:
        return ""
    # LoopResult 没有 model 字段; 从 messages 找最后一条 assistant 推断
    return ""


def _confidence_heuristic(result: UnifiedResult) -> float:
    """启发式: 验证通过 → 高; 仅产出代码 → 中; 失败 → 低。"""
    if result.success:
        return 0.95
    if result.proof_code:
        loop = result.loop_result
        if loop and loop.stopped_reason == "text_only":
            return 0.5
        return 0.4
    if result.loop_result is not None:
        if result.loop_result.stopped_reason in ("timeout", "token_budget"):
            return 0.1
    return 0.2
