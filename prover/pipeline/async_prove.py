"""prover/pipeline/async_prove.py — 异步证明轮次编排

从 engine/async_factory.py 迁移而来 (Phase B 分层解耦)。
此模块属于 prover 层, 可自由导入 agent 层组件。

engine 层只负责验证 (LeanPool, VerificationScheduler),
证明编排逻辑 (方向规划、置信度估计、AgentPool 调度) 属于 prover 层。
"""
from __future__ import annotations
import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.async_factory import AsyncEngineComponents
    from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)


async def async_prove_round(
        problem: 'BenchmarkProblem',
        components: 'AsyncEngineComponents',
        directions: list = None,
        attempt_history: list = None,
        classification: dict = None,
) -> list:
    """异步执行一轮异构并行证明

    1. 构建 N 个方向的 (spec, task) 对
    2. asyncio.gather 并行生成证明候选
    3. asyncio.gather 并行验证所有候选
    4. 广播发现, 返回排序结果
    """
    from prover.pipeline._agent_deps import AgentSpec, AgentTask, AgentResult, ContextItem
    from common.roles import AgentRole
    from prover.pipeline._agent_deps import ConfidenceEstimator

    if not components.agent_pool:
        return []

    classification = classification or {}
    attempt_history = attempt_history or []

    if not directions:
        directions = _default_directions(
            problem, classification, attempt_history)

    bus = components.broadcast
    subscriptions = {}
    for d in directions:
        sub = bus.subscribe(d.name)
        subscriptions[d.name] = sub
        for msg in bus.get_recent(n=15):
            sub.push(msg)

    specs_and_tasks = []
    for d in directions:
        spec = AgentSpec(
            name=d.name, role=d.role,
            model=d.model, temperature=d.temperature,
            few_shot_override=d.few_shot_override,
            tools=d.allowed_tools)

        context_items = [
            ContextItem("theorem", problem.theorem_statement, 1.0,
                        "theorem_statement"),
        ]
        if d.strategic_hint:
            context_items.append(
                ContextItem("strategy", d.strategic_hint, 0.9, "tactic_hint"))

        broadcast_text = bus.render_for_prompt(
            d.name, max_messages=8,
            current_goal=problem.theorem_statement)
        if broadcast_text:
            context_items.append(
                ContextItem("teammate_discoveries", broadcast_text,
                            0.95, "premise"))

        # ── Knowledge injection (Gap 0 fix) ──
        if components.knowledge_reader:
            try:
                knowledge_text = await components.knowledge_reader.render_for_prompt(
                    goal=problem.theorem_statement,
                    theorem=problem.theorem_statement,
                    max_chars=1200)
                if knowledge_text:
                    context_items.append(
                        ContextItem("proof_knowledge", knowledge_text,
                                    0.92, "premise"))
            except Exception:
                pass  # graceful degradation

        for hk, hv in classification.get("domain_hints", {}).items():
            context_items.append(
                ContextItem(hk, str(hv), 0.85, "premise"))

        task = AgentTask(
            description=_build_prompt(d, problem),
            injected_context=context_items,
            theorem_statement=problem.theorem_statement,
            metadata={"direction": d.name})

        specs_and_tasks.append((spec, task))

    results = await components.agent_pool.run_parallel(specs_and_tasks)

    if components.scheduler:
        pool_stats = components.scheduler.pool.stats() if components.scheduler.pool else {}
        has_real_repl = not pool_stats.get("all_fallback", True)

        if has_real_repl:
            verify_tasks = []
            verify_indices = []
            for i, (result, direction) in enumerate(zip(results, directions)):
                if result.proof_code and result.proof_code.strip():
                    verify_tasks.append(
                        components.scheduler.verify_complete(
                            theorem=problem.theorem_statement,
                            proof=result.proof_code,
                            direction=direction.name))
                    verify_indices.append(i)

            if verify_tasks:
                verify_results = await asyncio.gather(
                    *verify_tasks, return_exceptions=True)

                for idx, vr in zip(verify_indices, verify_results):
                    if isinstance(vr, Exception):
                        logger.warning(f"Verification error: {vr}")
                        continue
                    result = results[idx]
                    result.confidence = ConfidenceEstimator.refine_confidence(
                        result,
                        feedback=vr.feedback,
                        l0_passed=vr.l0_passed,
                        l1_passed=(vr.level_reached in ("L1", "L2") and vr.success),
                        l2_passed=vr.l2_verified)
                    result.metadata["verification"] = {
                        "success": vr.success,
                        "level": vr.level_reached,
                    }
                    if vr.success:
                        result.success = True

    results.sort(key=lambda r: -r.confidence)

    for name in subscriptions:
        bus.unsubscribe(name)

    return results


def run_async_prove_round(problem, components, **kwargs):
    """同步入口 — 内部用 asyncio.run() 驱动异步管线"""
    return asyncio.run(async_prove_round(problem, components, **kwargs))


# ── 辅助函数 ──

def _default_directions(problem, classification, attempt_history):
    """默认的 4 方向规划"""
    from prover.pipeline.heterogeneous_engine import ProofDirection
    from common.roles import AgentRole

    directions = [
        ProofDirection(
            name="automation",
            role=AgentRole.PROOF_GENERATOR,
            model="claude-sonnet-4-20250514",
            temperature=0.2,
            strategic_hint=(
                "Try to solve this with simple automation ONLY. "
                "Attempt: decide, norm_num, simp, omega, ring, aesop.")),
        ProofDirection(
            name="structured",
            role=AgentRole.PROOF_GENERATOR,
            temperature=0.7,
            strategic_hint=(
                "Plan the proof structure carefully. "
                "Use `have` statements with explicit types.")),
        ProofDirection(
            name="alternative",
            role=AgentRole.PROOF_PLANNER,
            temperature=0.9,
            strategic_hint=(
                "Try a fundamentally DIFFERENT approach. "
                "Consider casting to ℤ, using `conv`, or direct Mathlib lemmas.")),
    ]

    if len(attempt_history) >= 2:
        recent_errors = []
        for a in attempt_history[-3:]:
            errs = a.get("errors", [])
            for e in errs[:2]:
                msg = e.get("message", str(e)) if isinstance(e, dict) else str(e)
                recent_errors.append(msg[:100])
        directions.append(ProofDirection(
            name="repair_rethink",
            role=AgentRole.CRITIC,
            temperature=0.5,
            strategic_hint=(
                f"Previous {len(attempt_history)} attempts failed. "
                f"Recent errors:\n" +
                "\n".join(f"  - {e}" for e in recent_errors) +
                "\n\nPropose a completely different proof strategy.")))

    return directions


def _build_prompt(direction, problem) -> str:
    parts = [
        f"Prove the following Lean 4 theorem:\n"
        f"```lean\n{problem.theorem_statement}\n```"]
    if direction.strategic_hint:
        parts.append(f"\n## Strategy guidance\n{direction.strategic_hint}")
    if problem.natural_language:
        parts.append(f"\n## Natural language\n{problem.natural_language}")
    parts.append(
        "\nGenerate a complete proof. Output ONLY the proof body "
        "(starting with `:= by`) inside a single ```lean block.")
    return "\n".join(parts)
