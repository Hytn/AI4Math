"""prover/pipeline/async_orchestrator.py — 异步流水线编排器

核心改进: LLM 生成和 REPL 验证真正交替运行, 互不阻塞。

架构:
  ┌─────────────┐     ┌───────────────┐     ┌──────────────┐
  │ LLM Producer│────►│ asyncio.Queue │────►│ REPL Consumer│
  │ (N 路并发)  │     │ (候选缓冲)     │     │ (M 路并发)   │
  └─────────────┘     └───────────────┘     └──────┬───────┘
                                                   │
                                            BroadcastBus
                                                   │
                                            ┌──────▼───────┐
                                            │ Result Sink  │
                                            │ (trace 聚合) │
                                            └──────────────┘

同步 Orchestrator 的问题:
  [LLM gen ────][wait][Verify ──][LLM gen ────][wait][Verify ──]
  LLM 和 REPL 轮流闲置, 利用率 ~50%

异步版:
  LLM-0:  [gen ──────][gen ──────][gen ──────]...
  LLM-1:  [gen ──────][gen ──────]...
  REPL-0: ···[verify ─][verify ─][verify ─]...
  REPL-1: ···[verify ─][verify ─]...
  利用率 ~90%+

Usage:
    orchestrator = AsyncOrchestrator(components)
    trace = await orchestrator.prove(problem)

    # 或从同步上下文:
    trace = orchestrator.prove_sync(problem)
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from prover.models import (
    BenchmarkProblem, ProofTrace, ProofAttempt, AttemptStatus,
)
from prover.pipeline._agent_deps import MetaController
from prover.pipeline._agent_deps import ConfidenceEstimator
from common.budget import Budget
from common.working_memory import WorkingMemory
from prover.pipeline._agent_deps import HookManager
from common.hook_types import HookEvent, HookContext, HookAction

logger = logging.getLogger(__name__)


@dataclass
class ProofCandidate:
    """从 LLM 生成的待验证证明候选"""
    proof_code: str
    direction: str
    confidence: float = 0.5
    metadata: dict = field(default_factory=dict)
    agent_result: Optional[object] = None  # AgentResult
    timestamp: float = field(default_factory=time.time)


@dataclass
class VerifiedResult:
    """经过 REPL 验证的结果"""
    candidate: ProofCandidate
    success: bool
    verification_level: str = ""  # L0, L1, L2
    feedback_text: str = ""
    elapsed_ms: int = 0


class AsyncOrchestrator:
    """异步流水线编排器

    生产者-消费者模型:
      - N 个 LLM producer 协程持续生成候选
      - M 个 REPL consumer 协程持续验证候选
      - 通过 asyncio.Queue 解耦, 实现 LLM 和 REPL 的重叠执行
    """

    def __init__(self, components: 'AsyncEngineComponents',
                 config: dict = None,
                 hooks: Optional[HookManager] = None,
                 on_attempt=None):
        self.comp = components
        self.config = config or {}
        self.hooks = hooks or HookManager()
        self.on_attempt = on_attempt

        self.meta = MetaController(self.config)
        self.confidence = ConfidenceEstimator()
        self.budget = Budget(
            max_samples=self.config.get("max_samples", 128),
            max_wall_seconds=self.config.get("max_wall_seconds", 3600))

        # Pipeline tuning
        self._queue_size = self.config.get("pipeline_queue_size", 16)
        self._num_verifiers = self.config.get("num_verifiers", 4)

    async def prove(self, problem: BenchmarkProblem) -> ProofTrace:
        """主证明入口 — 异步流水线"""
        start_time = time.time()

        memory = WorkingMemory(
            problem_id=problem.problem_id,
            theorem_statement=problem.theorem_statement)
        memory.current_strategy = self.meta.select_initial_strategy(
            problem.difficulty)

        trace = ProofTrace(
            problem_id=problem.problem_id,
            problem_name=problem.name,
            theorem_statement=problem.theorem_statement,
            natural_language=problem.natural_language,
            config_snapshot={"strategy": memory.current_strategy,
                             "mode": "async_pipeline"})

        # Shared state
        candidate_queue: asyncio.Queue[Optional[ProofCandidate]] = \
            asyncio.Queue(maxsize=self._queue_size)
        solved_event = asyncio.Event()
        winning_proof: list[str] = []  # mutable container for result

        # Fire hooks
        self.hooks.fire(HookEvent.ON_PROBLEM_START, HookContext(
            event=HookEvent.ON_PROBLEM_START,
            theorem_statement=problem.theorem_statement))

        # Launch pipeline
        producers = [
            asyncio.create_task(
                self._producer(problem, memory, candidate_queue,
                               solved_event, direction_idx=i))
            for i in range(self._num_producers(memory))
        ]
        consumers = [
            asyncio.create_task(
                self._consumer(problem, memory, trace, candidate_queue,
                               solved_event, winning_proof))
            for _ in range(self._num_verifiers)
        ]

        # Wait for solved or budget exhaustion
        try:
            # Producers will terminate when solved or budget exhausted,
            # then send None sentinels to stop consumers
            await asyncio.gather(*producers, return_exceptions=True)

            # Send sentinel for each consumer
            for _ in consumers:
                await candidate_queue.put(None)

            await asyncio.gather(*consumers, return_exceptions=True)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            # Cancel any still-running tasks
            for t in producers + consumers:
                if not t.done():
                    t.cancel()

        memory.solved = solved_event.is_set()
        trace.total_duration_ms = int((time.time() - start_time) * 1000)

        self.hooks.fire(HookEvent.ON_PROBLEM_END, HookContext(
            event=HookEvent.ON_PROBLEM_END,
            theorem_statement=problem.theorem_statement,
            metadata={"solved": memory.solved}))

        return trace

    def prove_sync(self, problem: BenchmarkProblem) -> ProofTrace:
        """同步入口 — 内部驱动 asyncio 事件循环"""
        return asyncio.run(self.prove(problem))

    # ── Producer: LLM 生成 ──

    async def _producer(self, problem, memory, queue, solved_event,
                        direction_idx: int):
        """持续生成证明候选, 推入队列"""
        from prover.pipeline.async_prove import _default_directions, _build_prompt
        from agent.runtime.sub_agent import AgentSpec, AgentTask, ContextItem
        from common.roles import AgentRole

        round_num = 0
        while not solved_event.is_set() and not self.budget.is_exhausted():
            # Plan direction for this round
            directions = _default_directions(
                problem, {}, memory.attempt_history)
            idx = direction_idx % len(directions)
            d = directions[idx]

            # Inject broadcast knowledge
            broadcast_text = ""
            if self.comp.broadcast:
                broadcast_text = self.comp.broadcast.render_for_prompt(
                    d.name, max_messages=8)

            # ── Knowledge injection (Gap 0 fix) ──
            # Retrieve accumulated knowledge from the unified store
            knowledge_text = ""
            if self.comp.knowledge_reader:
                try:
                    knowledge_text = await self.comp.knowledge_reader.render_for_prompt(
                        goal=problem.theorem_statement,
                        theorem=problem.theorem_statement,
                        max_chars=1200)
                except Exception as e:
                    logger.debug(f"Knowledge retrieval skipped: {e}")

            spec = AgentSpec(
                name=f"{d.name}_r{round_num}",
                role=d.role,
                temperature=d.temperature,
                few_shot_override=d.few_shot_override,
                tools=d.allowed_tools)

            context_items = [
                ContextItem("theorem", problem.theorem_statement, 1.0,
                            "theorem_statement"),
            ]
            if d.strategic_hint:
                context_items.append(
                    ContextItem("strategy", d.strategic_hint, 0.9,
                                "tactic_hint"))
            if knowledge_text:
                context_items.append(
                    ContextItem("proof_knowledge", knowledge_text,
                                0.92, "premise"))
            if broadcast_text:
                context_items.append(
                    ContextItem("teammate_discoveries", broadcast_text,
                                0.95, "premise"))

            task = AgentTask(
                description=_build_prompt(d, problem),
                injected_context=context_items,
                theorem_statement=problem.theorem_statement,
                metadata={"direction": d.name, "round": round_num})

            try:
                results = await self.comp.agent_pool.run_parallel(
                    [(spec, task)])
                for r in results:
                    if r.proof_code and r.proof_code.strip():
                        candidate = ProofCandidate(
                            proof_code=r.proof_code,
                            direction=d.name,
                            confidence=r.confidence,
                            agent_result=r)
                        await queue.put(candidate)
                        self.budget.record_sample()
            except Exception as e:
                logger.warning(f"Producer {direction_idx} error: {e}")

            round_num += 1
            memory.rounds_completed = max(
                memory.rounds_completed, round_num)

    # ── Consumer: REPL 验证 ──

    async def _consumer(self, problem, memory, trace, queue,
                        solved_event, winning_proof):
        """从队列取候选, 验证, 广播结果"""
        while True:
            candidate = await queue.get()
            if candidate is None:
                break  # Sentinel: producer is done
            if solved_event.is_set():
                continue  # Drain queue but skip verification

            try:
                vr = await self._verify_candidate(problem, candidate)
                attempt = self._to_attempt(candidate, vr, memory)
                trace.add_attempt(attempt)
                if self.on_attempt:
                    self.on_attempt(attempt)

                if vr.success:
                    winning_proof.append(candidate.proof_code)
                    solved_event.set()
                    memory.solved = True
                    logger.info(
                        f"SOLVED by {candidate.direction}: "
                        f"{candidate.proof_code[:100]}...")

            except Exception as e:
                logger.warning(f"Consumer error: {e}")
            finally:
                queue.task_done()

    async def _verify_candidate(self, problem, candidate) -> VerifiedResult:
        """通过 AsyncVerificationScheduler 验证候选"""
        if not self.comp.scheduler:
            return VerifiedResult(candidate=candidate, success=False,
                                 verification_level="none")

        try:
            result = await self.comp.scheduler.verify_complete(
                theorem=problem.theorem_statement,
                proof=candidate.proof_code,
                direction=candidate.direction)

            # ── Knowledge ingestion (Gap 0 fix) ──
            # Use KnowledgeBroadcaster if available (handles both
            # knowledge store writing AND broadcast publishing)
            if self.comp.knowledge_broadcaster:
                from engine.proof_context_store import StepDetail
                step = StepDetail(
                    step_index=0,
                    tactic=candidate.proof_code[:200],
                    env_id_before=0,
                    env_id_after=1 if result.success else -1,
                    goals_before=[problem.theorem_statement],
                    goals_after=[] if result.success else [problem.theorem_statement],
                    error_message=result.feedback.error_message if result.feedback else "",
                    error_category=result.feedback.error_category if result.feedback else "",
                    elapsed_ms=float(result.total_ms),
                    is_proof_complete=result.success,
                )
                try:
                    await self.comp.knowledge_broadcaster.on_tactic_result(
                        step,
                        direction=candidate.direction,
                        theorem=problem.theorem_statement)
                except Exception as e:
                    logger.debug(f"Knowledge ingestion skipped: {e}")

            # Broadcast discoveries (fallback for when knowledge_broadcaster
            # is not configured — preserves backward compatibility)
            elif self.comp.broadcast:
                from engine.broadcast import BroadcastMessage
                if result.success:
                    self.comp.broadcast.publish(
                        BroadcastMessage.positive(
                            source=candidate.direction,
                            discovery=f"Proof found: {candidate.proof_code[:80]}"))
                elif result.feedback and result.feedback.error_category:
                    self.comp.broadcast.publish(
                        BroadcastMessage.negative(
                            source=candidate.direction,
                            tactic=candidate.proof_code[:40],
                            error_category=result.feedback.error_category,
                            reason=result.feedback.error_message[:100]))

            return VerifiedResult(
                candidate=candidate,
                success=result.success,
                verification_level=result.level_reached,
                feedback_text=result.feedback.error_message if result.feedback else "",
                elapsed_ms=result.total_ms)

        except Exception as e:
            logger.warning(f"Verification error: {e}")
            return VerifiedResult(candidate=candidate, success=False,
                                 verification_level="error")

    # ── Helpers ──

    def _num_producers(self, memory) -> int:
        """根据策略决定并发生成数"""
        s = memory.current_strategy
        if s == "sequential":
            return 1
        if s == "light":
            return 2
        if s == "medium":
            return 3
        return 4  # heavy

    def _to_attempt(self, candidate, vr, memory) -> ProofAttempt:
        """将验证结果转为 ProofAttempt"""
        status = AttemptStatus.SUCCESS if vr.success else AttemptStatus.LEAN_ERROR
        attempt = ProofAttempt(
            generated_proof=candidate.proof_code,
            lean_result=status,
            check_ms=vr.elapsed_ms,
            metadata={
                "direction": candidate.direction,
                "verification_level": vr.verification_level,
            })
        memory.record_attempt({
            "generated_proof": candidate.proof_code,
            "status": status.value,
            "direction": candidate.direction,
            "errors": [{"message": vr.feedback_text}] if vr.feedback_text else [],
        })
        memory.total_samples += 1
        return attempt
