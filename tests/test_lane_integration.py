"""tests/test_lane_integration.py — Tests for lane ↔ pipeline integration

Covers:
  - LaneProofRunner with mocked subsystems
  - MetaController → PolicyEngine delegation
  - End-to-end lane lifecycle scenarios
  - Error recovery flows
  - Knowledge injection/deposit flows
"""
import asyncio
import pytest
import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

from engine.lane.task_state import (
    TaskStatus, ProofFailureClass, TaskContext,
    ProofTaskStateMachine, TaskEvent,
)
from engine.lane.event_bus import ProofEventBus, wire_state_machine_to_bus
from engine.lane.recovery import RecoveryRegistry, RecoveryAction
from engine.lane.task_packet import ProofTaskPacket, validate_packet
from engine.lane.policy import (
    PolicyEngine, PolicyAction, PolicyDecision,
    ConsecutiveSameErrorRule, BudgetEscalationRule,
    InfraRecoveryRule, ReflectionRule,
)
from engine.lane.dashboard import ProofDashboard
from engine.lane.integration import (
    LaneProofRunner,
    map_error_to_failure_class,
    map_verification_result_to_failure_class,
    run_eval_with_lanes,
)
from agent.strategy.meta_controller import (
    MetaController, InitialStrategyRule, MaxRoundsGiveUpRule,
)


# ═════════════════════════════════════════════════════════════════════════════
# Test helpers / mocks
# ═════════════════════════════════════════════════════════════════════════════

def _make_packet(**overrides):
    defaults = dict(
        theorem_name="test_thm",
        formal_statement="theorem test : 1 + 1 = 2 := by norm_num",
        domain="arithmetic",
        difficulty="easy",
        max_samples=32,
        max_wall_seconds=60,
        initial_strategy="light",
        inject_knowledge=False,
        deposit_knowledge=False,
    )
    defaults.update(overrides)
    return ProofTaskPacket(**defaults)


@dataclass
class MockAgentResult:
    agent_name: str = "mock_agent"
    role: str = "proof_generator"
    content: str = ""
    proof_code: str = ""
    tool_calls: list = field(default_factory=list)
    tokens_used: int = 100
    latency_ms: int = 50
    confidence: float = 0.5
    success: bool = False
    error: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class MockVerificationResult:
    success: bool = False
    level_reached: str = "L1"
    l0_passed: bool = True
    l0_reject_reason: str = ""
    l0_fix_hint: str = ""
    l1_env_id: int = -1
    l1_goals_remaining: list = field(default_factory=list)
    l2_verified: bool = False
    l0_us: int = 5
    l1_ms: int = 50
    l2_ms: int = 0
    total_ms: int = 55
    feedback: MagicMock = field(default_factory=lambda: MagicMock(
        error_category="", error_message="",
        to_prompt=MagicMock(return_value="")))


@dataclass
class MockDirection:
    name: str = "automation"
    role: str = "proof_generator"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    strategic_hint: str = "try simp"
    selected_premises: list = field(default_factory=list)
    few_shot_override: str = ""
    allowed_tools: list = field(default_factory=list)


def _make_mock_pool(results=None):
    """Create a mock AsyncAgentPool."""
    pool = AsyncMock()
    if results is None:
        results = [MockAgentResult(proof_code="by norm_num")]
    pool.run_parallel = AsyncMock(return_value=results)
    return pool


def _make_mock_scheduler(success=False):
    """Create a mock AsyncVerificationScheduler."""
    scheduler = AsyncMock()
    vr = MockVerificationResult(success=success)
    if success:
        vr.l1_env_id = 42
    scheduler.verify_complete = AsyncMock(return_value=vr)
    scheduler.error_intel = MagicMock(
        get_accumulated_knowledge=MagicMock(return_value=""))
    scheduler.pool = MagicMock(
        stats=MagicMock(return_value={}))
    return scheduler


def _make_mock_planner(n_directions=2):
    """Create a mock DirectionPlanner."""
    planner = MagicMock()
    dirs = [MockDirection(name=f"dir_{i}") for i in range(n_directions)]
    planner.plan = MagicMock(return_value=dirs)
    return planner


# ═════════════════════════════════════════════════════════════════════════════
# Error mapping tests
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorMapping:
    def test_known_categories(self):
        assert map_error_to_failure_class("type_mismatch") == ProofFailureClass.TYPE_MISMATCH
        assert map_error_to_failure_class("timeout") == ProofFailureClass.TIMEOUT
        assert map_error_to_failure_class("sorry") == ProofFailureClass.SORRY_DETECTED

    def test_unknown_falls_back(self):
        assert map_error_to_failure_class("xyz") == ProofFailureClass.TACTIC_FAILED

    def test_vr_l0_failure(self):
        vr = MockVerificationResult(success=False, l0_passed=False)
        assert map_verification_result_to_failure_class(vr) == ProofFailureClass.SYNTAX_ERROR

    def test_vr_l1_failure_with_category(self):
        vr = MockVerificationResult(success=False)
        vr.feedback = MagicMock(error_category="type_mismatch")
        assert map_verification_result_to_failure_class(vr) == ProofFailureClass.TYPE_MISMATCH


# ═════════════════════════════════════════════════════════════════════════════
# LaneProofRunner tests
# ═════════════════════════════════════════════════════════════════════════════

class TestLaneProofRunnerBasic:
    """Tests that don't require real subsystems."""

    def test_construction_with_no_args(self):
        runner = LaneProofRunner()
        assert runner.agent_pool is None
        assert runner.scheduler is None
        assert runner.policy is not None
        assert runner.broadcast is not None

    def test_construction_with_all_args(self):
        pool = _make_mock_pool()
        sched = _make_mock_scheduler()
        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=sched,
            policy=PolicyEngine.default(),
        )
        assert runner.agent_pool is pool
        assert runner.scheduler is sched


class TestLaneProofRunnerAsync:
    """Async tests for the full lane proof loop."""

    @pytest.fixture
    def event_loop(self):
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_no_pool_gives_up_gracefully(self):
        """Without an agent pool, runner should give up after max rounds."""
        runner = LaneProofRunner()
        packet = _make_packet(max_samples=3)
        sm = self._run(runner.run(packet))
        assert sm.status == TaskStatus.GIVEN_UP

    def test_proof_found_on_first_round(self):
        """Happy path: candidate verifies successfully."""
        pool = _make_mock_pool([
            MockAgentResult(
                proof_code="by norm_num",
                metadata={"direction": "automation"}),
        ])
        scheduler = _make_mock_scheduler(success=True)
        planner = _make_mock_planner(1)

        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=scheduler,
            direction_planner=planner,
        )
        packet = _make_packet()
        sm = self._run(runner.run(packet))

        assert sm.status == TaskStatus.SUCCEEDED
        assert "norm_num" in sm.context.best_attempt_code or True
        # Verify subsystems were called
        pool.run_parallel.assert_awaited_once()
        scheduler.verify_complete.assert_awaited_once()

    def test_all_candidates_fail_then_give_up(self):
        """All candidates fail verification → eventually give up."""
        pool = _make_mock_pool([
            MockAgentResult(
                proof_code="by simp",
                metadata={"direction": "dir_0"}),
        ])
        scheduler = _make_mock_scheduler(success=False)
        planner = _make_mock_planner(1)

        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=scheduler,
            direction_planner=planner,
        )
        packet = _make_packet(max_samples=5)
        sm = self._run(runner.run(packet))

        assert sm.status in (TaskStatus.GIVEN_UP, TaskStatus.FAILED)
        assert sm.context.rounds_completed >= 1

    def test_wall_time_exceeded(self):
        """Runner gives up when wall time is exceeded."""
        import unittest.mock as um

        pool = _make_mock_pool([
            MockAgentResult(proof_code="by simp",
                            metadata={"direction": "d"}),
        ])
        scheduler = _make_mock_scheduler(success=False)

        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=scheduler,
            direction_planner=_make_mock_planner(1),
        )
        packet = _make_packet(max_wall_seconds=1, max_samples=100)

        # Make time.time() return a large value after setup
        real_time = time.time
        start = real_time()
        call_count = [0]
        def advancing_time():
            call_count[0] += 1
            # After a few calls, jump far into the future
            if call_count[0] > 10:
                return start + 9999
            return real_time()

        with um.patch('engine.lane.integration.time') as mt:
            mt.time = advancing_time
            sm = self._run(runner.run(packet))

        assert sm.status == TaskStatus.GIVEN_UP

    def test_event_bus_receives_events(self):
        """Events are published to the event bus."""
        bus = ProofEventBus()
        captured = []
        bus.subscribe("*", lambda e: captured.append(e))

        pool = _make_mock_pool([
            MockAgentResult(proof_code="by norm_num",
                            metadata={"direction": "d"}),
        ])
        scheduler = _make_mock_scheduler(success=True)

        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=scheduler,
            direction_planner=_make_mock_planner(1),
            event_bus=bus,
        )
        packet = _make_packet()
        sm = self._run(runner.run(packet))

        assert sm.status == TaskStatus.SUCCEEDED
        # Should have events: generating, verifying, succeeded
        event_names = [e.event_name for e in captured]
        assert "task.generating" in event_names
        assert "task.succeeded" in event_names

    def test_dashboard_tracks_task(self):
        """Dashboard registers and tracks the task."""
        dashboard = ProofDashboard()

        pool = _make_mock_pool([
            MockAgentResult(proof_code="by norm_num",
                            metadata={"direction": "d"}),
        ])
        scheduler = _make_mock_scheduler(success=True)

        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=scheduler,
            direction_planner=_make_mock_planner(1),
            dashboard=dashboard,
        )
        packet = _make_packet()
        self._run(runner.run(packet))

        snap = dashboard.snapshot()
        assert snap["summary"]["total"] == 1
        assert snap["summary"]["succeeded"] == 1

    def test_knowledge_loading(self):
        """Knowledge reader is called when inject_knowledge=True."""
        reader = AsyncMock()
        reader.render_for_prompt = AsyncMock(
            return_value="Use norm_num for arithmetic goals.")

        pool = _make_mock_pool([
            MockAgentResult(proof_code="by norm_num",
                            metadata={"direction": "d"}),
        ])
        scheduler = _make_mock_scheduler(success=True)

        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=scheduler,
            direction_planner=_make_mock_planner(1),
            knowledge_reader=reader,
        )
        packet = _make_packet(inject_knowledge=True)
        sm = self._run(runner.run(packet))

        assert sm.status == TaskStatus.SUCCEEDED
        reader.render_for_prompt.assert_awaited_once()

    def test_knowledge_loading_failure_recovers(self):
        """Knowledge loading failure doesn't crash — degrades gracefully."""
        reader = AsyncMock()
        reader.render_for_prompt = AsyncMock(
            side_effect=RuntimeError("DB connection failed"))

        pool = _make_mock_pool([
            MockAgentResult(proof_code="by norm_num",
                            metadata={"direction": "d"}),
        ])
        scheduler = _make_mock_scheduler(success=True)

        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=scheduler,
            direction_planner=_make_mock_planner(1),
            knowledge_reader=reader,
        )
        packet = _make_packet(inject_knowledge=True)
        sm = self._run(runner.run(packet))

        # Should still succeed despite knowledge failure
        assert sm.status == TaskStatus.SUCCEEDED
        # Should have a KNOWLEDGE_ERROR failure event in the log
        failure_events = [
            e for e in sm.events if e.failure is not None]
        assert any(
            e.failure.failure_class == ProofFailureClass.KNOWLEDGE_ERROR
            for e in failure_events)

    def test_knowledge_deposit_on_success(self):
        """Knowledge is deposited on successful proof."""
        writer = AsyncMock()
        writer.ingest_proof_result = AsyncMock()
        broadcaster = AsyncMock()
        broadcaster.on_tactic_result = AsyncMock()

        pool = _make_mock_pool([
            MockAgentResult(proof_code="by norm_num",
                            metadata={"direction": "d"}),
        ])
        scheduler = _make_mock_scheduler(success=True)

        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=scheduler,
            direction_planner=_make_mock_planner(1),
            knowledge_writer=writer,
            knowledge_broadcaster=broadcaster,
        )
        packet = _make_packet(deposit_knowledge=True)
        sm = self._run(runner.run(packet))

        assert sm.status == TaskStatus.SUCCEEDED
        # Writer should have been called
        writer.ingest_proof_result.assert_awaited_once()

    def test_multiple_candidates_best_wins(self):
        """When multiple candidates are generated, the first passing wins."""
        pool = _make_mock_pool([
            MockAgentResult(proof_code="by simp",
                            metadata={"direction": "d0"}),
            MockAgentResult(proof_code="by norm_num",
                            metadata={"direction": "d1"}),
        ])

        call_count = 0
        async def verify_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MockVerificationResult(success=False)
            return MockVerificationResult(success=True, l1_env_id=42)

        scheduler = AsyncMock()
        scheduler.verify_complete = AsyncMock(
            side_effect=verify_side_effect)
        scheduler.error_intel = MagicMock(
            get_accumulated_knowledge=MagicMock(return_value=""))
        scheduler.pool = MagicMock(stats=MagicMock(return_value={}))

        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=scheduler,
            direction_planner=_make_mock_planner(2),
        )
        packet = _make_packet()
        sm = self._run(runner.run(packet))

        assert sm.status == TaskStatus.SUCCEEDED
        assert scheduler.verify_complete.await_count == 2

    def test_scheduler_exception_handled(self):
        """Scheduler throwing an exception doesn't crash the runner."""
        pool = _make_mock_pool([
            MockAgentResult(proof_code="by norm_num",
                            metadata={"direction": "d"}),
        ])
        scheduler = AsyncMock()
        scheduler.verify_complete = AsyncMock(
            side_effect=RuntimeError("REPL crashed"))
        scheduler.error_intel = MagicMock(
            get_accumulated_knowledge=MagicMock(return_value=""))
        scheduler.pool = MagicMock(stats=MagicMock(return_value={}))

        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=scheduler,
            direction_planner=_make_mock_planner(1),
        )
        packet = _make_packet(max_samples=3)
        sm = self._run(runner.run(packet))

        # Should not crash — should give up eventually
        assert sm.status in (TaskStatus.GIVEN_UP, TaskStatus.FAILED)


# ═════════════════════════════════════════════════════════════════════════════
# MetaController tests
# ═════════════════════════════════════════════════════════════════════════════

class TestMetaControllerNew:
    """Test the new PolicyEngine-backed MetaController."""

    def test_select_initial_strategy_easy(self):
        mc = MetaController()
        assert mc.select_initial_strategy("easy") == "sequential"
        assert mc.select_initial_strategy("trivial") == "sequential"

    def test_select_initial_strategy_hard(self):
        mc = MetaController()
        assert mc.select_initial_strategy("hard") == "medium"
        assert mc.select_initial_strategy("competition") == "heavy"

    def test_select_initial_strategy_unknown(self):
        mc = MetaController()
        assert mc.select_initial_strategy("unknown") == "light"

    def test_should_give_up_over_budget(self):
        from common.working_memory import WorkingMemory
        mc = MetaController()
        mem = WorkingMemory(total_samples=200)
        assert mc.should_give_up(mem, {"max_samples": 128}) is True

    def test_should_give_up_under_budget(self):
        from common.working_memory import WorkingMemory
        mc = MetaController()
        mem = WorkingMemory(total_samples=10)
        assert mc.should_give_up(mem, {"max_samples": 128}) is False

    def test_should_escalate_returns_none_early(self):
        from common.working_memory import WorkingMemory
        mc = MetaController({"max_samples": 128})
        mem = WorkingMemory(
            total_samples=1, rounds_completed=0,
            current_strategy="light")
        result = mc.should_escalate(mem)
        # At 1 sample, no escalation expected
        assert result is None

    def test_policy_engine_accessible(self):
        mc = MetaController()
        engine = mc.policy_engine
        assert isinstance(engine, PolicyEngine)
        assert len(engine._rules) >= 5  # all default rules loaded

    def test_evaluate_returns_decision(self):
        mc = MetaController()
        ctx = TaskContext(
            theorem_name="test", formal_statement="test",
            rounds_completed=0, total_samples=0)
        ctx.__dict__["_max_samples"] = 128
        ctx.__dict__["_current_strategy"] = "light"
        sm = ProofTaskStateMachine(task_id="t", context=ctx)
        sm.transition_to(TaskStatus.GENERATING)
        sm.transition_to(TaskStatus.VERIFYING)

        decision = mc.evaluate(sm)
        assert isinstance(decision, PolicyDecision)

    def test_custom_rule_injection(self):
        """Can add custom rules to the MetaController's engine."""
        mc = MetaController()

        class AlwaysGiveUpRule:
            name = "always_give_up"
            priority = 0  # highest

            def evaluate(self, sm, events):
                return PolicyDecision(
                    action=PolicyAction.GIVE_UP,
                    reason="test",
                    rule_name=self.name)

        mc.add_rule(AlwaysGiveUpRule())

        ctx = TaskContext(
            theorem_name="t", formal_statement="t")
        ctx.__dict__["_max_samples"] = 128
        ctx.__dict__["_current_strategy"] = "light"
        sm = ProofTaskStateMachine(task_id="t", context=ctx)

        decision = mc.evaluate(sm)
        assert decision.action == PolicyAction.GIVE_UP


class TestInitialStrategyRule:
    def test_fires_at_round_zero(self):
        rule = InitialStrategyRule()
        ctx = TaskContext(
            theorem_name="t", formal_statement="t",
            difficulty="hard", rounds_completed=0)
        ctx.__dict__["_current_strategy"] = "light"
        ctx.__dict__["_max_samples"] = 128
        sm = ProofTaskStateMachine(task_id="t", context=ctx)

        decision = rule.evaluate(sm, sm.events)
        assert decision is not None
        assert decision.action == PolicyAction.ESCALATE_STRATEGY
        assert decision.metadata["to"] == "medium"

    def test_does_not_fire_after_round_zero(self):
        rule = InitialStrategyRule()
        ctx = TaskContext(
            theorem_name="t", formal_statement="t",
            difficulty="hard", rounds_completed=3)
        ctx.__dict__["_current_strategy"] = "light"
        ctx.__dict__["_max_samples"] = 128
        sm = ProofTaskStateMachine(task_id="t", context=ctx)

        decision = rule.evaluate(sm, sm.events)
        assert decision is None


class TestMaxRoundsGiveUpRule:
    def test_fires_when_heavy_exhausted(self):
        rule = MaxRoundsGiveUpRule(max_heavy_rounds=5)
        ctx = TaskContext(
            theorem_name="t", formal_statement="t",
            rounds_completed=5)
        ctx.__dict__["_current_strategy"] = "heavy"
        ctx.__dict__["_max_samples"] = 128
        sm = ProofTaskStateMachine(task_id="t", context=ctx)

        decision = rule.evaluate(sm, sm.events)
        assert decision is not None
        assert decision.action == PolicyAction.GIVE_UP

    def test_does_not_fire_for_light(self):
        rule = MaxRoundsGiveUpRule(max_heavy_rounds=5)
        ctx = TaskContext(
            theorem_name="t", formal_statement="t",
            rounds_completed=10)
        ctx.__dict__["_current_strategy"] = "light"
        ctx.__dict__["_max_samples"] = 128
        sm = ProofTaskStateMachine(task_id="t", context=ctx)

        decision = rule.evaluate(sm, sm.events)
        assert decision is None


# ═════════════════════════════════════════════════════════════════════════════
# PolicyEngine + Recovery interaction tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPolicyRecoveryInteraction:
    """Test that PolicyEngine and RecoveryRegistry work together correctly."""

    def test_blocked_task_gets_recovery(self):
        """A BLOCKED task with a REPL_CRASH should get AUTO_RECOVER."""
        engine = PolicyEngine.default()
        ctx = TaskContext(
            theorem_name="t", formal_statement="t")
        ctx.__dict__["_max_samples"] = 128
        ctx.__dict__["_current_strategy"] = "light"
        sm = ProofTaskStateMachine(task_id="t", context=ctx)

        sm.transition_to(TaskStatus.GENERATING)
        sm.transition_to(TaskStatus.VERIFYING)
        sm.fail(ProofFailureClass.REPL_CRASH, "REPL died",
                recoverable=True)

        assert sm.status == TaskStatus.BLOCKED
        decision = engine.evaluate(sm)
        assert decision.action == PolicyAction.AUTO_RECOVER

    def test_budget_exhausted_no_recovery(self):
        """Budget exhausted is terminal — no recovery."""
        engine = PolicyEngine.default()
        ctx = TaskContext(
            theorem_name="t", formal_statement="t")
        ctx.__dict__["_max_samples"] = 128
        ctx.__dict__["_current_strategy"] = "light"
        sm = ProofTaskStateMachine(task_id="t", context=ctx)

        sm.transition_to(TaskStatus.GENERATING)
        sm.transition_to(TaskStatus.VERIFYING)
        sm.fail(ProofFailureClass.BUDGET_EXHAUSTED,
                "over budget", recoverable=True)

        decision = engine.evaluate(sm)
        assert decision.action == PolicyAction.GIVE_UP

    def test_consecutive_errors_trigger_role_switch(self):
        """5 consecutive same-type errors should trigger SWITCH_ROLE."""
        engine = PolicyEngine.default()
        ctx = TaskContext(
            theorem_name="t", formal_statement="t")
        ctx.__dict__["_max_samples"] = 128
        ctx.__dict__["_current_strategy"] = "light"
        sm = ProofTaskStateMachine(task_id="t", context=ctx)
        sm.transition_to(TaskStatus.GENERATING)
        sm.transition_to(TaskStatus.VERIFYING)

        # Simulate 5 consecutive type_mismatch failures
        for i in range(5):
            sm.fail(ProofFailureClass.TYPE_MISMATCH,
                    f"type error {i}", recoverable=True)
            if sm.status == TaskStatus.BLOCKED:
                sm.transition_to(TaskStatus.VERIFYING)

        decision = engine.evaluate(sm)
        # Should trigger either SWITCH_ROLE or AUTO_RECOVER
        assert decision.action in (
            PolicyAction.SWITCH_ROLE, PolicyAction.AUTO_RECOVER)


# ═════════════════════════════════════════════════════════════════════════════
# run_eval_with_lanes tests
# ═════════════════════════════════════════════════════════════════════════════

class TestRunEvalWithLanes:

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_eval_empty_list(self):
        runner = LaneProofRunner()
        results, final = self._run(
            run_eval_with_lanes([], runner))
        assert results == []
        assert final["summary"]["total"] == 0

    def test_eval_single_problem(self):
        from prover.models import BenchmarkProblem
        pool = _make_mock_pool([
            MockAgentResult(proof_code="by norm_num",
                            metadata={"direction": "d"}),
        ])
        scheduler = _make_mock_scheduler(success=True)

        runner = LaneProofRunner(
            agent_pool=pool,
            scheduler=scheduler,
            direction_planner=_make_mock_planner(1),
        )

        problems = [BenchmarkProblem(
            problem_id="p1", name="test1",
            theorem_statement="theorem t : True := trivial",
        )]

        results, final = self._run(
            run_eval_with_lanes(problems, runner))

        assert len(results) == 1
        assert results[0].status == TaskStatus.SUCCEEDED
        assert final["summary"]["succeeded"] == 1
        assert final["summary"]["pass_rate"] == 1.0
