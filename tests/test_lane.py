"""tests/test_lane.py — Tests for engine/lane proof lane runtime

Covers: state machine, event bus, recovery, task packet, policy, dashboard.
"""
import pytest
import time

from engine.lane.task_state import (
    TaskStatus, ProofFailureClass, TaskContext,
    ProofTaskStateMachine, TaskFailure,
)
from engine.lane.event_bus import ProofEventBus, wire_state_machine_to_bus
from engine.lane.recovery import (
    RecoveryRegistry, RecoveryAction, RecoveryRecipe,
)
from engine.lane.task_packet import (
    ProofTaskPacket, validate_packet, packet_from_benchmark_problem,
)
from engine.lane.policy import (
    PolicyEngine, PolicyAction, PolicyDecision,
    ConsecutiveSameErrorRule, BudgetEscalationRule,
    BankedLemmaDecomposeRule, InfraRecoveryRule, ReflectionRule,
)
from engine.lane.dashboard import ProofDashboard


# ─── TaskState tests ─────────────────────────────────────────────────────────

class TestTaskStateMachine:
    def _make_sm(self, name="test_thm"):
        ctx = TaskContext(theorem_name=name, formal_statement=f"theorem {name} : True := trivial")
        return ProofTaskStateMachine(task_id=f"task_{name}", context=ctx)

    def test_initial_state_is_created(self):
        sm = self._make_sm()
        assert sm.status == TaskStatus.CREATED
        assert len(sm.events) == 1
        assert sm.events[0].event_name == "task.created"

    def test_happy_path_lifecycle(self):
        sm = self._make_sm()
        sm.transition_to(TaskStatus.KNOWLEDGE_LOADING)
        sm.transition_to(TaskStatus.GENERATING, detail="round 1")
        sm.transition_to(TaskStatus.VERIFYING, detail="8 candidates")
        sm.succeed("theorem test : True := trivial")
        assert sm.status == TaskStatus.SUCCEEDED
        assert sm.context.best_attempt_code == "theorem test : True := trivial"
        assert len(sm.events) == 5

    def test_repair_loop(self):
        sm = self._make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        sm.transition_to(TaskStatus.VERIFYING)
        sm.transition_to(TaskStatus.REPAIRING)
        sm.transition_to(TaskStatus.VERIFYING)
        sm.succeed("proof")
        assert sm.status == TaskStatus.SUCCEEDED

    def test_invalid_transition_raises(self):
        sm = self._make_sm()
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition_to(TaskStatus.VERIFYING)

    def test_terminal_state_blocks_transitions(self):
        sm = self._make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        sm.give_up("too hard")
        assert sm.status == TaskStatus.GIVEN_UP
        with pytest.raises(ValueError):
            sm.transition_to(TaskStatus.GENERATING)

    def test_fail_recoverable_goes_to_blocked(self):
        sm = self._make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        sm.fail(ProofFailureClass.REPL_CRASH, "process died", recoverable=True)
        assert sm.status == TaskStatus.BLOCKED
        assert sm.last_failure.failure_class == ProofFailureClass.REPL_CRASH

    def test_fail_unrecoverable_goes_to_failed(self):
        sm = self._make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        sm.fail(ProofFailureClass.INTEGRITY_VIOLATION, "sorry detected",
                recoverable=False)
        assert sm.status == TaskStatus.FAILED

    def test_recovery_from_blocked(self):
        sm = self._make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        sm.fail(ProofFailureClass.REPL_CRASH, "crash", recoverable=True)
        assert sm.status == TaskStatus.BLOCKED
        sm.transition_to(TaskStatus.GENERATING)
        assert sm.status == TaskStatus.GENERATING
        assert sm.last_failure is None
        assert sm.recovery_attempts == 1

    def test_snapshot_is_serializable(self):
        sm = self._make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        snap = sm.snapshot()
        assert snap["status"] == "generating"
        assert snap["task_id"] == "task_test_thm"
        assert isinstance(snap["elapsed_seconds"], float)


# ─── EventBus tests ──────────────────────────────────────────────────────────

class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = ProofEventBus()
        received = []
        bus.subscribe("task.*", lambda e: received.append(e))

        sm = ProofTaskStateMachine(
            task_id="t1",
            context=TaskContext(theorem_name="x", formal_statement="x"))
        wire_state_machine_to_bus(sm, bus)

        sm.transition_to(TaskStatus.GENERATING)
        assert len(received) >= 1
        assert any(e.event_name == "task.generating" for e in received)

    def test_pattern_filtering(self):
        bus = ProofEventBus()
        failures_only = []
        bus.subscribe("task.failure.*", lambda e: failures_only.append(e))

        sm = ProofTaskStateMachine(
            task_id="t2",
            context=TaskContext(theorem_name="y", formal_statement="y"))
        wire_state_machine_to_bus(sm, bus)

        sm.transition_to(TaskStatus.GENERATING)
        sm.fail(ProofFailureClass.API_ERROR, "rate limited")

        assert len(failures_only) == 1
        assert failures_only[0].failure.failure_class == ProofFailureClass.API_ERROR

    def test_recent_events(self):
        bus = ProofEventBus()
        from engine.lane.task_state import TaskEvent
        for i in range(10):
            bus.publish(TaskEvent(
                seq=i, event_name=f"task.test_{i}",
                prev_status=TaskStatus.CREATED, new_status=TaskStatus.CREATED))
        assert len(bus.recent_events(5)) == 5


# ─── Recovery tests ──────────────────────────────────────────────────────────

class TestRecovery:
    def test_repl_crash_has_recipe(self):
        reg = RecoveryRegistry()
        recipe = reg.get(ProofFailureClass.REPL_CRASH)
        assert recipe is not None
        assert recipe.action == RecoveryAction.RESTART_REPL
        assert recipe.max_attempts == 2

    def test_budget_exhausted_is_terminal(self):
        reg = RecoveryRegistry()
        assert not reg.should_recover(ProofFailureClass.BUDGET_EXHAUSTED, 0)

    def test_recovery_respects_max_attempts(self):
        reg = RecoveryRegistry()
        assert reg.should_recover(ProofFailureClass.API_ERROR, 0)
        assert reg.should_recover(ProofFailureClass.API_ERROR, 2)
        assert not reg.should_recover(ProofFailureClass.API_ERROR, 3)

    def test_custom_recipe_registration(self):
        reg = RecoveryRegistry()
        reg.register(ProofFailureClass.TIMEOUT, RecoveryRecipe(
            action=RecoveryAction.SKIP_AND_CONTINUE, max_attempts=5))
        assert reg.get(ProofFailureClass.TIMEOUT).max_attempts == 5


# ─── TaskPacket tests ────────────────────────────────────────────────────────

class TestTaskPacket:
    def test_valid_packet(self):
        p = ProofTaskPacket(
            theorem_name="nat_add_comm",
            formal_statement="theorem nat_add_comm (n m : Nat) : n + m = m + n",
        )
        validated = validate_packet(p)
        assert validated.theorem_name == "nat_add_comm"

    def test_empty_name_raises(self):
        p = ProofTaskPacket(theorem_name="", formal_statement="x")
        with pytest.raises(ValueError, match="theorem_name is required"):
            validate_packet(p)

    def test_invalid_strategy_raises(self):
        p = ProofTaskPacket(
            theorem_name="x", formal_statement="x",
            initial_strategy="turbo")
        with pytest.raises(ValueError, match="invalid initial_strategy"):
            validate_packet(p)

    def test_invalid_temperature_raises(self):
        p = ProofTaskPacket(
            theorem_name="x", formal_statement="x",
            temperature=5.0)
        with pytest.raises(ValueError, match="temperature"):
            validate_packet(p)

    def test_from_benchmark_problem(self):
        class FakeProblem:
            name = "test"
            formal_statement = "theorem test : True := trivial"
            domain = "logic"
            difficulty = "easy"
        p = packet_from_benchmark_problem(FakeProblem())
        assert p.theorem_name == "test"
        assert p.domain == "logic"


# ─── Policy tests ────────────────────────────────────────────────────────────

class TestPolicy:
    def _make_sm(self):
        ctx = TaskContext(theorem_name="t", formal_statement="t")
        return ProofTaskStateMachine(task_id="p1", context=ctx)

    def test_default_engine_returns_continue(self):
        engine = PolicyEngine.default()
        sm = self._make_sm()
        decision = engine.evaluate(sm)
        assert decision.action == PolicyAction.CONTINUE

    def test_consecutive_error_triggers_switch_role(self):
        rule = ConsecutiveSameErrorRule(threshold=3)
        sm = self._make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        for _ in range(3):
            sm.fail(ProofFailureClass.TACTIC_FAILED, "omega failed", recoverable=True)
            sm.transition_to(TaskStatus.GENERATING)

        decision = rule.evaluate(sm, sm.events)
        assert decision is not None
        assert decision.action == PolicyAction.SWITCH_ROLE

    def test_infra_recovery_rule_fires_on_blocked(self):
        rule = InfraRecoveryRule()
        sm = self._make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        sm.fail(ProofFailureClass.REPL_CRASH, "crash", recoverable=True)
        assert sm.status == TaskStatus.BLOCKED

        decision = rule.evaluate(sm, sm.events)
        assert decision is not None
        assert decision.action == PolicyAction.AUTO_RECOVER

    def test_decompose_rule(self):
        rule = BankedLemmaDecomposeRule(min_lemmas=1, min_rounds=2)
        sm = self._make_sm()
        sm.context.banked_lemmas = ["lemma h1"]
        sm.context.rounds_completed = 3

        decision = rule.evaluate(sm, sm.events)
        assert decision is not None
        assert decision.action == PolicyAction.TRY_DECOMPOSE

    def test_reflection_rule_fires_every_n(self):
        rule = ReflectionRule(every_n_rounds=2)
        sm = self._make_sm()
        sm.context.rounds_completed = 4
        decision = rule.evaluate(sm, sm.events)
        assert decision is not None
        assert decision.action == PolicyAction.INJECT_REFLECTION


# ─── Dashboard tests ─────────────────────────────────────────────────────────

class TestDashboard:
    def test_empty_dashboard(self):
        d = ProofDashboard()
        snap = d.snapshot()
        assert snap["summary"]["total"] == 0

    def test_register_and_snapshot(self):
        d = ProofDashboard()
        ctx = TaskContext(theorem_name="t1", formal_statement="t1")
        sm1 = ProofTaskStateMachine(task_id="d1", context=ctx)
        sm1.transition_to(TaskStatus.GENERATING)
        d.register_task(sm1)

        ctx2 = TaskContext(theorem_name="t2", formal_statement="t2")
        sm2 = ProofTaskStateMachine(task_id="d2", context=ctx2)
        sm2.transition_to(TaskStatus.GENERATING)
        sm2.transition_to(TaskStatus.VERIFYING)
        sm2.succeed("proof")
        d.register_task(sm2)

        snap = d.snapshot()
        assert snap["summary"]["total"] == 2
        assert snap["summary"]["succeeded"] == 1
        assert snap["summary"]["in_progress"] == 1

    def test_summary_line(self):
        d = ProofDashboard()
        line = d.summary_line()
        assert "proved" in line
