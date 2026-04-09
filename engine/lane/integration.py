"""engine/lane/integration.py — Production wiring: lane ↔ proof pipeline

Connects the lane runtime (state machine, event bus, policy, recovery)
into the real proof pipeline (AsyncAgentPool, AsyncVerificationScheduler,
KnowledgeReader, KnowledgeBroadcaster).

This is NO LONGER a reference implementation — it is the primary execution
path for lane-aware proofs.

Usage::

    from engine.lane.integration import LaneProofRunner

    runner = LaneProofRunner(
        agent_pool=async_pool,
        scheduler=async_scheduler,
        knowledge_reader=reader,
        knowledge_writer=writer,
        knowledge_broadcaster=broadcaster,
    )
    sm = await runner.run(packet)
    # sm.status == TaskStatus.SUCCEEDED / FAILED / GIVEN_UP
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, TYPE_CHECKING

from engine.lane.task_state import (
    TaskStatus, ProofFailureClass, TaskContext, ProofTaskStateMachine,
)
from engine.lane.event_bus import ProofEventBus, wire_state_machine_to_bus
from engine.lane.recovery import RecoveryRegistry, RecoveryAction
from engine.lane.task_packet import ProofTaskPacket, validate_packet
from engine.lane.policy import PolicyEngine, PolicyAction, PolicyDecision
from engine.lane.dashboard import ProofDashboard

if TYPE_CHECKING:
    from agent.runtime.async_agent_pool import AsyncAgentPool
    from agent.runtime.sub_agent import AgentSpec, AgentTask, AgentResult, ContextItem
    from agent.strategy.direction_planner import DirectionPlanner, ProofDirection
    from engine.async_verification_scheduler import AsyncVerificationScheduler
    from engine.verification_scheduler import VerificationResult
    from knowledge.reader import KnowledgeReader
    from knowledge.writer import KnowledgeWriter
    from knowledge.broadcaster import KnowledgeBroadcaster
    from common.budget import Budget
    from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)


# ─── Bridge: map existing error categories to ProofFailureClass ──────────────

_ERROR_CATEGORY_MAP: dict[str, ProofFailureClass] = {
    "type_mismatch": ProofFailureClass.TYPE_MISMATCH,
    "app_type_mismatch": ProofFailureClass.TYPE_MISMATCH,
    "unknown_identifier": ProofFailureClass.UNKNOWN_IDENTIFIER,
    "tactic_failed": ProofFailureClass.TACTIC_FAILED,
    "syntax_error": ProofFailureClass.SYNTAX_ERROR,
    "timeout": ProofFailureClass.TIMEOUT,
    "import_error": ProofFailureClass.IMPORT_ERROR,
    "sorry": ProofFailureClass.SORRY_DETECTED,
    "unsolved_goals": ProofFailureClass.TACTIC_FAILED,
    "elaboration_error": ProofFailureClass.TYPE_MISMATCH,
    "instance_not_found": ProofFailureClass.UNKNOWN_IDENTIFIER,
    "recursion_limit": ProofFailureClass.TIMEOUT,
    "ambiguous": ProofFailureClass.SYNTAX_ERROR,
    "other": ProofFailureClass.TACTIC_FAILED,
}


def map_error_to_failure_class(error_category: str) -> ProofFailureClass:
    """Map error category string to ProofFailureClass."""
    return _ERROR_CATEGORY_MAP.get(
        error_category, ProofFailureClass.TACTIC_FAILED)


def map_verification_result_to_failure_class(vr) -> ProofFailureClass:
    """Extract the primary failure class from a VerificationResult."""
    if vr.success:
        return ProofFailureClass.TACTIC_FAILED  # shouldn't be called on success

    # L0 rejection → syntax
    if not vr.l0_passed:
        return ProofFailureClass.SYNTAX_ERROR

    # L1/L2 failure → extract from feedback
    feedback = vr.feedback
    if hasattr(feedback, 'error_category') and feedback.error_category:
        return map_error_to_failure_class(feedback.error_category)

    # Fallback: parse reject reason
    if vr.l0_reject_reason:
        return ProofFailureClass.SYNTAX_ERROR

    return ProofFailureClass.TACTIC_FAILED


# ─── Direction → AgentSpec/Task bridge ───────────────────────────────────────

def _direction_to_spec_and_task(
    direction,
    problem_statement: str,
    knowledge_text: str = "",
    broadcast_context: str = "",
    dead_ends_text: str = "",
    domain_hints: dict = None,
):
    """Convert a ProofDirection + context into (AgentSpec, AgentTask) pair."""
    from agent.runtime.sub_agent import AgentSpec, AgentTask, ContextItem
    from agent.strategy.direction_planner import build_direction_prompt
    from prover.models import BenchmarkProblem

    spec = AgentSpec(
        name=direction.name,
        role=direction.role,
        model=direction.model,
        temperature=direction.temperature,
        few_shot_override=direction.few_shot_override,
        tools=direction.allowed_tools,
    )

    context_items = [
        ContextItem("theorem", problem_statement, 1.0, "theorem_statement"),
    ]

    if direction.strategic_hint:
        context_items.append(
            ContextItem("strategy", direction.strategic_hint, 0.9,
                        "tactic_hint"))

    if direction.selected_premises:
        premises_text = "\n".join(
            f"- {p}" for p in direction.selected_premises[:15])
        context_items.append(
            ContextItem("premises", premises_text, 0.7, "premise"))

    if knowledge_text:
        context_items.append(
            ContextItem("knowledge", knowledge_text, 0.95, "premise"))

    if broadcast_context:
        context_items.append(
            ContextItem("teammate_discoveries", broadcast_context, 0.9,
                        "premise"))

    if dead_ends_text:
        context_items.append(
            ContextItem("known_dead_ends", dead_ends_text, 0.85, "premise"))

    if domain_hints:
        for hk, hv in domain_hints.items():
            context_items.append(
                ContextItem(hk, str(hv), 0.85, "premise"))

    # Create a minimal BenchmarkProblem for build_direction_prompt
    problem = BenchmarkProblem(
        problem_id="", name="", theorem_statement=problem_statement)
    task_description = build_direction_prompt(direction, problem)

    task = AgentTask(
        description=task_description,
        injected_context=context_items,
        theorem_statement=problem_statement,
        metadata={"direction": direction.name},
    )

    return spec, task


# ═════════════════════════════════════════════════════════════════════════════
# LaneProofRunner — the primary lane-aware proof execution engine
# ═════════════════════════════════════════════════════════════════════════════

class LaneProofRunner:
    """Run a single proof task through the full lane lifecycle.

    Replaces the placeholder ``run_proof_task_with_lane()`` with a real
    implementation that connects to all subsystems.

    Components (all optional — gracefully degrade if absent):

    =========  =====================  ====================================
    Component  Type                   Purpose
    =========  =====================  ====================================
    agent_pool AsyncAgentPool         Generates proof candidates
    scheduler  AsyncVerifScheduler    L0/L1/L2 verification
    planner    DirectionPlanner       Plans exploration directions
    reader     KnowledgeReader        Injects knowledge into prompts
    writer     KnowledgeWriter        Records proof traces
    broadcaster KnowledgeBroadcaster  Cross-agent sharing
    broadcast  BroadcastBus           Real-time discovery sharing
    event_bus  ProofEventBus          Typed event publishing
    dashboard  ProofDashboard         Aggregate monitoring
    policy     PolicyEngine           Strategy rules
    =========  =====================  ====================================
    """

    def __init__(
        self,
        agent_pool: Optional[AsyncAgentPool] = None,
        scheduler: Optional[AsyncVerificationScheduler] = None,
        direction_planner=None,
        knowledge_reader: Optional[KnowledgeReader] = None,
        knowledge_writer: Optional[KnowledgeWriter] = None,
        knowledge_broadcaster: Optional[KnowledgeBroadcaster] = None,
        broadcast=None,
        event_bus: Optional[ProofEventBus] = None,
        dashboard: Optional[ProofDashboard] = None,
        policy: Optional[PolicyEngine] = None,
        budget: Optional[Budget] = None,
    ):
        self.agent_pool = agent_pool
        self.scheduler = scheduler
        self.knowledge_reader = knowledge_reader
        self.knowledge_writer = knowledge_writer
        self.knowledge_broadcaster = knowledge_broadcaster
        self.policy = policy or PolicyEngine.default()
        self.event_bus = event_bus
        self.dashboard = dashboard
        self.recovery_reg = RecoveryRegistry()

        # Direction planner: use provided or create default
        if direction_planner:
            self.direction_planner = direction_planner
        else:
            from agent.strategy.direction_planner import DirectionPlanner
            self.direction_planner = DirectionPlanner()

        # Broadcast bus: use provided, or extract from scheduler, or new
        if broadcast:
            self.broadcast = broadcast
        elif (scheduler and hasattr(scheduler, 'broadcast')
              and scheduler.broadcast):
            self.broadcast = scheduler.broadcast
        else:
            from engine.broadcast import BroadcastBus
            self.broadcast = BroadcastBus()

        self.budget = budget

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    async def run(self, packet: ProofTaskPacket) -> ProofTaskStateMachine:
        """Execute a single proof task through the full lane lifecycle.

        Returns the ProofTaskStateMachine with full event history.
        """
        packet = validate_packet(packet)

        # ── Setup state machine ──────────────────────────────────────
        ctx = TaskContext(
            theorem_name=packet.theorem_name,
            formal_statement=packet.formal_statement,
            domain=packet.domain,
            difficulty=packet.difficulty,
        )
        sm = ProofTaskStateMachine(
            task_id=f"lane_{packet.theorem_name}", context=ctx)

        if self.event_bus:
            wire_state_machine_to_bus(sm, self.event_bus)
        if self.dashboard:
            self.dashboard.register_task(sm)

        # Store budget info for policy rules
        ctx.__dict__["_max_samples"] = packet.max_samples
        ctx.__dict__["_current_strategy"] = packet.initial_strategy
        ctx.__dict__["_domain_hints"] = {}
        ctx.__dict__["_start_time"] = time.time()

        start_time = time.time()
        knowledge_text = ""
        attempt_history: list[dict] = []

        try:
            # ── Phase 1: Knowledge Loading ───────────────────────────
            knowledge_text = await self._load_knowledge(sm, packet, ctx)

            # ── Phase 2: Main prove loop ─────────────────────────────
            if sm.status.is_terminal:
                return sm  # knowledge error with GIVE_UP policy

            # Only transition if not already in GENERATING
            # (knowledge failure recovery may have already moved us there)
            if sm.status != TaskStatus.GENERATING:
                sm.transition_to(
                    TaskStatus.GENERATING,
                    detail=f"strategy={packet.initial_strategy}")

            while not sm.status.is_terminal:
                loop_result = await self._run_one_round(
                    sm, ctx, packet, knowledge_text,
                    attempt_history, start_time)
                if loop_result:  # True means proof found or terminal
                    break

            # ── Phase 3: Knowledge deposit (failure) ─────────────────
            if sm.status != TaskStatus.SUCCEEDED:
                await self._deposit_knowledge_failure(
                    packet, ctx, attempt_history)

        except Exception as e:
            logger.exception(f"[{sm.task_id}] Unexpected error: {e}")
            sm.fail(ProofFailureClass.API_ERROR,
                    f"Unexpected error: {e}", recoverable=False)

        return sm

    # ─────────────────────────────────────────────────────────────────
    # Phase 1: Knowledge loading
    # ─────────────────────────────────────────────────────────────────

    async def _load_knowledge(
        self,
        sm: ProofTaskStateMachine,
        packet: ProofTaskPacket,
        ctx: TaskContext,
    ) -> str:
        """Load knowledge and return the prompt text (or empty string)."""
        if not (packet.inject_knowledge and self.knowledge_reader):
            return ""

        sm.transition_to(TaskStatus.KNOWLEDGE_LOADING)
        try:
            text = await self.knowledge_reader.render_for_prompt(
                goal=packet.formal_statement,
                theorem=packet.formal_statement,
                domain=packet.domain,
                max_chars=1500,
            )
            ctx.knowledge_injected = True
            logger.info(
                f"[{sm.task_id}] Knowledge loaded: {len(text)} chars")
            return text

        except Exception as e:
            logger.warning(
                f"[{sm.task_id}] Knowledge loading failed: {e}")
            sm.fail(ProofFailureClass.KNOWLEDGE_ERROR, str(e))
            decision = self.policy.evaluate(sm)
            if decision.action in (PolicyAction.CONTINUE,
                                   PolicyAction.AUTO_RECOVER):
                sm.transition_to(TaskStatus.GENERATING)
                return ""
            else:
                # Terminal — caller checks sm.status.is_terminal
                return ""

    # ─────────────────────────────────────────────────────────────────
    # Phase 2: Single round of generate → verify → policy
    # ─────────────────────────────────────────────────────────────────

    async def _run_one_round(
        self,
        sm: ProofTaskStateMachine,
        ctx: TaskContext,
        packet: ProofTaskPacket,
        knowledge_text: str,
        attempt_history: list[dict],
        start_time: float,
    ) -> bool:
        """Run one generate→verify→policy round.

        Returns True if the loop should stop (proof found or terminal).
        """
        # ── Budget / wall-time checks ────────────────────────────────
        elapsed = time.time() - start_time
        if elapsed > packet.max_wall_seconds:
            sm.give_up(
                f"wall time exceeded "
                f"({elapsed:.0f}s > {packet.max_wall_seconds}s)")
            return True

        if ctx.total_samples >= packet.max_samples:
            sm.give_up(
                f"sample budget exhausted "
                f"({ctx.total_samples}/{packet.max_samples})")
            return True

        if self.budget and self.budget.is_exhausted():
            sm.give_up("global budget exhausted")
            return True

        # ── Generate candidates ──────────────────────────────────────
        candidates = await self._generate_candidates(
            packet, ctx, knowledge_text, attempt_history)

        n_candidates = len(candidates)
        ctx.total_samples += max(n_candidates, 1)
        ctx.total_api_tokens += sum(
            getattr(c, 'tokens_used', 0) for c in candidates)

        if self.budget and candidates:
            self.budget.add_samples(n_candidates)
            self.budget.add_tokens(
                sum(getattr(c, 'tokens_used', 0) for c in candidates))

        if not candidates:
            ctx.rounds_completed += 1
            decision = self.policy.evaluate(sm)
            if decision.action == PolicyAction.GIVE_UP:
                sm.give_up(decision.reason)
                return True
            return False  # continue to next round

        # ── Verify candidates ────────────────────────────────────────
        sm.transition_to(
            TaskStatus.VERIFYING,
            detail=f"round {ctx.rounds_completed + 1}, "
                   f"{n_candidates} candidates")

        found_proof, best_vr = await self._verify_all_candidates(
            sm, ctx, packet, candidates, attempt_history)

        if found_proof:
            return True

        # ── Record best failure on state machine ─────────────────────
        if best_vr and not best_vr.success:
            fc = map_verification_result_to_failure_class(best_vr)
            error_msg = best_vr.l0_reject_reason or "verification failed"
            if hasattr(best_vr.feedback, 'error_message'):
                error_msg = best_vr.feedback.error_message or error_msg
            sm.fail(fc, error_msg, recoverable=True)

        # ── Post-round policy ────────────────────────────────────────
        ctx.rounds_completed += 1
        return await self._post_round_policy(
            sm, ctx, packet, attempt_history)

    # ─────────────────────────────────────────────────────────────────
    # Generate candidates
    # ─────────────────────────────────────────────────────────────────

    async def _generate_candidates(
        self,
        packet: ProofTaskPacket,
        ctx: TaskContext,
        knowledge_text: str,
        attempt_history: list[dict],
    ) -> list:
        """Generate proof candidates via AsyncAgentPool + DirectionPlanner.

        Returns a list of AgentResult with non-empty proof_code.
        """
        if not self.agent_pool:
            logger.warning(
                f"[lane_{ctx.theorem_name}] No agent pool — "
                f"cannot generate candidates")
            return []

        from prover.models import BenchmarkProblem

        problem = BenchmarkProblem(
            problem_id=ctx.theorem_name,
            name=ctx.theorem_name,
            theorem_statement=ctx.formal_statement,
            difficulty=ctx.difficulty,
        )

        # Plan exploration directions
        classification = ctx.__dict__.get("_domain_hints", {})
        directions = self.direction_planner.plan(
            problem,
            classification=classification,
            attempt_history=attempt_history,
        )

        # Build broadcast context (cross-agent knowledge)
        broadcast_context = ""
        if self.broadcast:
            try:
                broadcast_context = self.broadcast.render_for_prompt(
                    subscriber_name="lane_runner",
                    max_messages=8, max_chars=1500,
                    current_goal=ctx.formal_statement,
                )
            except Exception:
                pass  # broadcast rendering is best-effort

        # Build dead-ends text from error intelligence
        dead_ends_text = ""
        if (self.scheduler
                and hasattr(self.scheduler, 'error_intel')
                and self.scheduler.error_intel):
            try:
                dead_ends_text = (
                    self.scheduler.error_intel.get_accumulated_knowledge(5))
            except Exception:
                pass

        # Convert directions → (spec, task) pairs
        specs_and_tasks = []
        for direction in directions:
            spec, task = _direction_to_spec_and_task(
                direction,
                problem_statement=ctx.formal_statement,
                knowledge_text=knowledge_text,
                broadcast_context=broadcast_context,
                dead_ends_text=dead_ends_text,
                domain_hints=classification,
            )
            specs_and_tasks.append((spec, task))

        # Run agents in parallel
        try:
            results = await self.agent_pool.run_parallel(
                specs_and_tasks, budget=self.budget)
            return [r for r in results
                    if r.proof_code and r.proof_code.strip()]
        except Exception as e:
            logger.error(
                f"[lane_{ctx.theorem_name}] Agent pool error: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────
    # Verify all candidates
    # ─────────────────────────────────────────────────────────────────

    async def _verify_all_candidates(
        self,
        sm: ProofTaskStateMachine,
        ctx: TaskContext,
        packet: ProofTaskPacket,
        candidates: list,
        attempt_history: list[dict],
    ) -> tuple[bool, object]:
        """Verify all candidates. Returns (found_proof, best_vr)."""
        best_vr = None
        best_result = None
        require_l2 = (packet.verification_level == "l2_full_compile")

        for candidate in candidates:
            if not (candidate.proof_code and candidate.proof_code.strip()):
                continue

            vr = await self._verify_single(
                ctx, candidate, require_l2)
            if vr is None:
                continue

            # Track best
            if best_vr is None or (vr.success and not best_vr.success):
                best_result = candidate
                best_vr = vr

            if vr.success:
                # ── PROOF FOUND ──────────────────────────────────────
                sm.succeed(candidate.proof_code)
                await self._deposit_knowledge_success(
                    packet, candidate, ctx)
                return True, vr

            # Record failure in attempt history
            fc = map_verification_result_to_failure_class(vr)
            feedback_text = ""
            if hasattr(vr.feedback, 'to_prompt'):
                try:
                    feedback_text = vr.feedback.to_prompt(max_chars=200)
                except Exception:
                    feedback_text = str(vr.l0_reject_reason or "")

            attempt_history.append({
                "proof_code": candidate.proof_code[:500],
                "direction": candidate.metadata.get("direction", ""),
                "errors": [{
                    "category": fc.value,
                    "message": feedback_text,
                }],
            })

        # Update best attempt on context
        if best_result:
            ctx.best_attempt_code = best_result.proof_code

        return False, best_vr

    async def _verify_single(self, ctx, candidate, require_l2):
        """Verify a single candidate through the scheduler."""
        if not self.scheduler:
            logger.warning(
                f"[lane_{ctx.theorem_name}] No scheduler — "
                f"cannot verify")
            return None

        direction_name = candidate.metadata.get("direction", "")
        try:
            return await self.scheduler.verify_complete(
                theorem=ctx.formal_statement,
                proof=candidate.proof_code,
                direction=direction_name,
                require_l2=require_l2,
            )
        except Exception as e:
            logger.warning(
                f"[lane_{ctx.theorem_name}] Verification error "
                f"({direction_name}): {e}")
            return None

    # ─────────────────────────────────────────────────────────────────
    # Post-round policy evaluation
    # ─────────────────────────────────────────────────────────────────

    async def _post_round_policy(
        self,
        sm: ProofTaskStateMachine,
        ctx: TaskContext,
        packet: ProofTaskPacket,
        attempt_history: list[dict],
    ) -> bool:
        """Evaluate policy after a round.

        Returns True if proof loop should stop (terminal state reached).
        """
        decision = self.policy.evaluate(sm)

        if decision.action == PolicyAction.GIVE_UP:
            sm.give_up(decision.reason)
            return True

        if decision.action == PolicyAction.AUTO_RECOVER:
            await self._execute_recovery(sm, ctx)
            # After recovery, transition back to GENERATING
            if not sm.status.is_terminal:
                if sm.status == TaskStatus.BLOCKED:
                    sm.transition_to(TaskStatus.GENERATING,
                                     detail="post-recovery")
            return sm.status.is_terminal

        # Strategy switches — all transition back to GENERATING
        detail = decision.reason
        new_status = TaskStatus.GENERATING

        if decision.action == PolicyAction.SWITCH_ROLE:
            ctx.__dict__["_current_role"] = "repair"
            if sm.status == TaskStatus.VERIFYING:
                sm.transition_to(TaskStatus.REPAIRING, detail=detail)
            # Then to GENERATING for next round

        elif decision.action == PolicyAction.ESCALATE_STRATEGY:
            new_strategy = decision.metadata.get("to", "medium")
            ctx.__dict__["_current_strategy"] = new_strategy
            logger.info(
                f"[{sm.task_id}] Escalating to {new_strategy}")
            detail = f"escalated to {new_strategy}"

        elif decision.action == PolicyAction.TRY_DECOMPOSE:
            ctx.__dict__["_decompose_attempted"] = True
            logger.info(f"[{sm.task_id}] Attempting decomposition")
            detail = "decompose"

        elif decision.action == PolicyAction.INJECT_REFLECTION:
            logger.info(f"[{sm.task_id}] Injecting reflection")
            detail = "reflection"

        # Transition to GENERATING for next round
        if not sm.status.is_terminal and sm.status != TaskStatus.GENERATING:
            sm.transition_to(new_status, detail=detail)

        return False

    # ─────────────────────────────────────────────────────────────────
    # Recovery execution
    # ─────────────────────────────────────────────────────────────────

    async def _execute_recovery(
        self,
        sm: ProofTaskStateMachine,
        ctx: TaskContext,
    ):
        """Execute a recovery action based on the last failure."""
        if not sm.last_failure:
            return

        fc = sm.last_failure.failure_class
        recipe = self.recovery_reg.get(fc)
        if not recipe:
            return

        action = recipe.action
        logger.info(f"[{sm.task_id}] Recovery: {action.value}")

        if action == RecoveryAction.RESTART_REPL:
            if (self.scheduler and hasattr(self.scheduler, 'pool')
                    and self.scheduler.pool):
                try:
                    if hasattr(self.scheduler.pool,
                               'restart_crashed_sessions'):
                        await self.scheduler.pool.restart_crashed_sessions()
                    elif hasattr(self.scheduler.pool, 'restart_all'):
                        await self.scheduler.pool.restart_all()
                except Exception as e:
                    logger.warning(f"REPL restart failed: {e}")

        elif action == RecoveryAction.RETRY_WITH_BACKOFF:
            await asyncio.sleep(recipe.backoff_seconds)

        elif action == RecoveryAction.RETRY_LARGER_TIMEOUT:
            ctx.__dict__["_timeout_multiplier"] = recipe.timeout_multiplier

        elif action == RecoveryAction.REDUCE_CONCURRENCY:
            ctx.__dict__["_reduced_concurrency"] = True

        elif action == RecoveryAction.SWITCH_ROLE:
            ctx.__dict__["_current_role"] = "repair"

        elif action == RecoveryAction.SWITCH_STRATEGY:
            current = ctx.__dict__.get("_current_strategy", "light")
            escalation = {"light": "medium", "medium": "heavy"}
            ctx.__dict__["_current_strategy"] = escalation.get(
                current, "heavy")

    # ─────────────────────────────────────────────────────────────────
    # Knowledge deposit
    # ─────────────────────────────────────────────────────────────────

    async def _deposit_knowledge_success(
        self,
        packet: ProofTaskPacket,
        candidate,
        ctx: TaskContext,
    ):
        """Deposit knowledge after successful proof."""
        if not packet.deposit_knowledge:
            return

        # Write via broadcaster if available
        if self.knowledge_broadcaster:
            try:
                from engine.proof_context_store import StepDetail
                step = StepDetail(
                    tactic=candidate.proof_code[:200],
                    goals_before=[ctx.formal_statement],
                    goals_after=[],
                    env_id_after=0,
                    error_message="",
                    elapsed_ms=0,
                )
                await self.knowledge_broadcaster.on_tactic_result(
                    step=step,
                    direction=candidate.metadata.get("direction", ""),
                    theorem=ctx.formal_statement,
                    domain=ctx.domain,
                )
            except Exception as e:
                logger.debug(
                    f"Knowledge deposit (broadcast/success) failed: {e}")

        # Also write via writer for persistent record
        if self.knowledge_writer:
            try:
                elapsed = time.time() - ctx.__dict__.get(
                    "_start_time", time.time())
                await self.knowledge_writer.ingest_proof_result(
                    context_id=0,
                    steps=[],
                    success=True,
                    theorem=ctx.formal_statement,
                    duration_ms=int(elapsed * 1000),
                )
            except Exception as e:
                logger.debug(
                    f"Knowledge deposit (writer/success) failed: {e}")

    async def _deposit_knowledge_failure(
        self,
        packet: ProofTaskPacket,
        ctx: TaskContext,
        attempt_history: list[dict],
    ):
        """Deposit negative knowledge after failed proof."""
        if not packet.deposit_knowledge:
            return

        if self.knowledge_writer:
            try:
                elapsed = time.time() - ctx.__dict__.get(
                    "_start_time", time.time())
                await self.knowledge_writer.ingest_proof_result(
                    context_id=0,
                    steps=[],
                    success=False,
                    theorem=ctx.formal_statement,
                    duration_ms=int(elapsed * 1000),
                )
            except Exception as e:
                logger.debug(
                    f"Knowledge deposit (writer/failure) failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# Convenience: lane-aware eval runner
# ═════════════════════════════════════════════════════════════════════════════

async def run_eval_with_lanes(
    problems: list,
    runner: LaneProofRunner,
    config: dict = None,
) -> tuple[list[ProofTaskStateMachine], dict]:
    """Run evaluation across multiple problems using lane runtime.

    Drop-in replacement for run_eval_async.py.

    Args:
        problems: List of BenchmarkProblem instances
        runner: Pre-configured LaneProofRunner
        config: Optional overrides for packet creation

    Returns:
        (list of state machines, dashboard snapshot dict)
    """
    from engine.lane.task_packet import packet_from_benchmark_problem

    config = config or {}
    bus = runner.event_bus or ProofEventBus()
    dashboard = runner.dashboard or ProofDashboard()
    runner.event_bus = bus
    runner.dashboard = dashboard

    results: list[ProofTaskStateMachine] = []
    for problem in problems:
        packet = packet_from_benchmark_problem(problem, config)
        sm = await runner.run(packet)
        results.append(sm)
        logger.info(
            f"[{sm.task_id}] {sm.status.value} | "
            f"{dashboard.summary_line()}")

    final = dashboard.snapshot()
    logger.info(
        f"Evaluation complete: "
        f"{final['summary']['succeeded']}/{final['summary']['total']} "
        f"({final['summary']['pass_rate']:.1%})")

    return results, final
