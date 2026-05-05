"""prover/unified/adapters.py — Unified → ProofAttempt adapter

把 ``UnifiedResult`` 转成 ``ProofAttempt`` (旧 ProofTrace 数据模型) ，让
``run_eval.py`` 在 unified 路径下能继续用 ProofTrace 累积 pass@k 统计。

历史: v8 之前还有 ``unified_to_agent_result`` 喂给 ``HeterogeneousEngine``
和 ``ResultFuser``, 但那两个组件 (依赖 SubAgent / AgentResult) 已在 v9
随 ``agent/runtime/{sub_agent,async_agent_pool,result_fuser}.py`` 一起
删除。本文件只剩 ``unified_to_attempt`` 一个函数。
"""
from __future__ import annotations

import logging
import time

from prover.models import (
    ProofAttempt, AttemptStatus, LeanError, ErrorCategory,
)
from prover.unified.runner import UnifiedResult

logger = logging.getLogger(__name__)

def unified_to_attempt(
    result: UnifiedResult,
    *,
    attempt_number: int = 1,
) -> ProofAttempt:
    """``UnifiedResult`` → ``ProofAttempt`` (ProofTrace 兼容路径)。

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
        llm_model="",
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

def _extract_lean_errors_from_loop(loop) -> tuple[list, str]:
    """从 LoopResult.messages 的 tool_result 内容里翻出 Lean 错误。"""
    if loop is None or not loop.messages:
        return [], ""

    errors = []
    stderr_parts = []

    for msg in loop.messages:
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
    """
    with LeanVerifyTool / ErrorIntelligence. The string returned by
    classify_error is mapped onto the legacy ErrorCategory enum."""
    try:
        from engine._core import classify_error
        cat = classify_error(text_lower or "")
    except Exception:
        cat = ""
    if cat in ("type_mismatch", "app_type_mismatch"):
        return ErrorCategory.TYPE_MISMATCH
    if cat == "unknown_identifier":
        return ErrorCategory.UNKNOWN_IDENTIFIER
    if cat == "tactic_failed":
        return ErrorCategory.TACTIC_FAILED
    if cat == "syntax_error":
        return ErrorCategory.SYNTAX_ERROR
    if cat == "timeout":
        return ErrorCategory.TIMEOUT
    # Unrecognised / "other" / classifier failed → fall back to the
    # original keyword scan as a last resort.
    if "import" in text_lower:
        return ErrorCategory.IMPORT_ERROR
    return ErrorCategory.OTHER

