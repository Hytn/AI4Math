"""engine/async_factory.py — 异步引擎组件工厂 + 编排集成

提供两个关键入口:
  1. AsyncEngineFactory — 构建异步版所有组件
  2. async_prove()      — 异步证明入口 (可被 Orchestrator.prove() 内部调用)

设计原则:
  - 外部接口保持同步兼容: Orchestrator.prove() 仍是同步方法
  - 内部用 asyncio.run() 驱动异步管线
  - 所有异步组件与同步组件共享数据类型, 结果可互操作
"""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AsyncEngineComponents:
    """异步引擎组件容器"""

    lean_pool: Optional['AsyncLeanPool'] = None
    prefilter: Optional['PreFilter'] = None
    error_intel: Optional['ErrorIntelligence'] = None
    scheduler: Optional['AsyncVerificationScheduler'] = None
    broadcast: Optional['BroadcastBus'] = None
    agent_pool: Optional['AsyncAgentPool'] = None

    async def close(self):
        if self.lean_pool:
            try:
                await self.lean_pool.shutdown()
            except Exception as e:
                logger.warning(f"AsyncLeanPool shutdown error: {e}")
        if self.broadcast:
            try:
                self.broadcast.clear()
            except Exception as e:
                logger.warning(f"BroadcastBus clear error: {e}")


class AsyncEngineFactory:
    """异步引擎组件工厂"""

    def __init__(self, config: dict = None):
        self.config = config or {}

    async def build(self, async_llm=None, retriever=None,
                    overrides: dict = None) -> AsyncEngineComponents:
        """构建所有异步组件

        Args:
            async_llm: AsyncLLMProvider 实例
            retriever: 前提检索器
            overrides: 组件覆盖 (测试用)
        """
        overrides = overrides or {}
        comp = AsyncEngineComponents()

        try:
            # 1. BroadcastBus (同步组件, 异步版复用)
            from engine.broadcast import BroadcastBus
            comp.broadcast = overrides.get('broadcast') or BroadcastBus()

            # 2. AsyncLeanPool
            if 'lean_pool' in overrides:
                comp.lean_pool = overrides['lean_pool']
            else:
                from engine.async_lean_pool import AsyncLeanPool
                pool_size = self.config.get("lean_pool_size", 4)
                project_dir = self.config.get("lean_project_dir", ".")
                comp.lean_pool = AsyncLeanPool(
                    pool_size=pool_size, project_dir=project_dir)
                await comp.lean_pool.start()

            # 3. PreFilter (同步, CPU-only)
            from engine.prefilter import PreFilter
            comp.prefilter = overrides.get('prefilter') or PreFilter()

            # 4. ErrorIntelligence (同步分析, 不需要异步)
            from engine.error_intelligence import ErrorIntelligence
            comp.error_intel = overrides.get('error_intel') or ErrorIntelligence()

            # 5. AsyncVerificationScheduler
            if 'scheduler' in overrides:
                comp.scheduler = overrides['scheduler']
            else:
                from engine.async_verification_scheduler import AsyncVerificationScheduler
                project_dir = self.config.get("lean_project_dir", ".")
                comp.scheduler = AsyncVerificationScheduler(
                    prefilter=comp.prefilter,
                    lean_pool=comp.lean_pool,
                    error_intel=comp.error_intel,
                    broadcast=comp.broadcast,
                    project_dir=project_dir)

            # 6. AsyncAgentPool
            if 'agent_pool' in overrides:
                comp.agent_pool = overrides['agent_pool']
            elif async_llm:
                from agent.runtime.async_agent_pool import AsyncAgentPool
                comp.agent_pool = AsyncAgentPool(
                    llm=async_llm,
                    max_workers=self.config.get("max_workers", 4))

            return comp

        except Exception as e:
            logger.error(f"AsyncEngineFactory build failed: {e}")
            await comp.close()
            raise


async def async_prove_round(
        problem: 'BenchmarkProblem',
        components: AsyncEngineComponents,
        directions: list['ProofDirection'] = None,
        attempt_history: list = None,
        classification: dict = None,
) -> list['AgentResult']:
    """异步执行一轮异构并行证明

    这是异步架构的核心编排函数:
    1. 构建 N 个方向的 (spec, task) 对
    2. asyncio.gather 并行生成证明候选
    3. asyncio.gather 并行验证所有候选
    4. 广播发现, 返回排序结果

    关键并行模式:
      同步版: [LLM A] [LLM B] [LLM C] [LLM D] [Verify A] [Verify B] ...
      异步版: [LLM A ──────]                    ← 4 路 LLM 并行
              [LLM B ──────]
              [LLM C ──────]
              [LLM D ──────]
              [Verify A/B/C/D ──]                ← 4 路验证并行
    """
    from agent.runtime.sub_agent import AgentSpec, AgentTask, AgentResult, ContextItem
    from agent.brain.roles import AgentRole

    if not components.agent_pool:
        return []

    classification = classification or {}
    attempt_history = attempt_history or []

    # 1. 构建方向 (复用同步版 _plan_directions 的逻辑)
    if not directions:
        directions = _default_directions(
            problem, classification, attempt_history)

    # 2. 为每个方向注册广播订阅 + 注入历史
    bus = components.broadcast
    subscriptions = {}
    for d in directions:
        sub = bus.subscribe(d.name)
        subscriptions[d.name] = sub
        for msg in bus.get_recent(n=15):
            sub.push(msg)

    # 3. 构建 specs_and_tasks
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

        # 注入广播消息
        broadcast_text = bus.render_for_prompt(d.name, max_messages=8)
        if broadcast_text:
            context_items.append(
                ContextItem("teammate_discoveries", broadcast_text,
                            0.95, "premise"))

        # 注入 domain hints
        for hk, hv in classification.get("domain_hints", {}).items():
            context_items.append(
                ContextItem(hk, str(hv), 0.85, "premise"))

        task = AgentTask(
            description=_build_prompt(d, problem),
            injected_context=context_items,
            theorem_statement=problem.theorem_statement,
            metadata={"direction": d.name})

        specs_and_tasks.append((spec, task))

    # 4. 并行生成证明候选 (N 路 LLM 同时调用)
    results = await components.agent_pool.run_parallel(specs_and_tasks)

    # 5. 并行验证所有候选 (N 路 REPL 同时验证)
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
                    from agent.strategy.confidence_estimator import ConfidenceEstimator
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

    # 6. 排序
    results.sort(key=lambda r: -r.confidence)

    # 7. 清理订阅
    for name in subscriptions:
        bus.unsubscribe(name)

    return results


def run_async_prove_round(problem, components, **kwargs):
    """同步入口 — 内部用 asyncio.run() 驱动异步管线

    供 Orchestrator.prove() 调用, 保持外部接口不变。
    """
    return asyncio.run(async_prove_round(problem, components, **kwargs))


# ── 辅助函数 ──

def _default_directions(problem, classification, attempt_history):
    """默认的 4 方向规划 (与 HeterogeneousEngine._plan_directions 对齐)"""
    from prover.pipeline.heterogeneous_engine import ProofDirection
    from agent.brain.roles import AgentRole

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
