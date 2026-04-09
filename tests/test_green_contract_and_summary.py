"""tests/test_green_contract_and_summary.py — Tests for Phase 2 components"""
import pytest

from engine.lane.green_contract import (
    GreenLevel, ProofGreenContract, GreenContractOutcome,
)
from engine.lane.summary_compression import (
    compress_proof_status, ProofStatusSummary,
)
from engine.lane.task_state import (
    TaskStatus, ProofFailureClass, TaskContext, ProofTaskStateMachine,
)
from engine.lane.policy import PolicyEngine
from engine.lane.dashboard import ProofDashboard
from engine.lane.summary_compression import compress_dashboard
from unittest.mock import MagicMock


# ═══════════════════════════════════════════════════════════════════════
# GreenLevel tests
# ═══════════════════════════════════════════════════════════════════════

class TestGreenLevel:
    def test_ordering(self):
        assert GreenLevel.NONE < GreenLevel.SYNTAX_CLEAN
        assert GreenLevel.SYNTAX_CLEAN < GreenLevel.TACTIC_VALID
        assert GreenLevel.TACTIC_VALID < GreenLevel.GOALS_CLOSED
        assert GreenLevel.GOALS_CLOSED < GreenLevel.FULL_COMPILE
        assert GreenLevel.FULL_COMPILE < GreenLevel.SORRY_FREE

    def test_from_vr_l0_fail(self):
        vr = MagicMock(l0_passed=False)
        assert GreenLevel.from_verification_result(vr) == GreenLevel.NONE

    def test_from_vr_l0_pass(self):
        vr = MagicMock(l0_passed=True, level_reached='L0', success=False)
        assert GreenLevel.from_verification_result(vr) == GreenLevel.SYNTAX_CLEAN

    def test_from_vr_l1_goals_closed(self):
        vr = MagicMock(l0_passed=True, level_reached='L1',
                       success=True, l1_goals_remaining=[])
        assert GreenLevel.from_verification_result(vr) == GreenLevel.GOALS_CLOSED

    def test_from_vr_l1_goals_remain(self):
        vr = MagicMock(l0_passed=True, level_reached='L1',
                       success=True, l1_goals_remaining=["⊢ n = n"])
        assert GreenLevel.from_verification_result(vr) == GreenLevel.TACTIC_VALID

    def test_from_vr_l2_sorry_free(self):
        vr = MagicMock(l0_passed=True, level_reached='L2',
                       success=True, l2_verified=True, has_sorry=False)
        assert GreenLevel.from_verification_result(vr) == GreenLevel.SORRY_FREE

    def test_from_vr_l2_with_sorry(self):
        vr = MagicMock(l0_passed=True, level_reached='L2',
                       success=True, l2_verified=True, has_sorry=True)
        assert GreenLevel.from_verification_result(vr) == GreenLevel.FULL_COMPILE

    def test_label_and_short(self):
        assert "syntax" in GreenLevel.SYNTAX_CLEAN.short
        assert "✅" in GreenLevel.SORRY_FREE.label


class TestProofGreenContract:
    def test_satisfied(self):
        c = ProofGreenContract(required=GreenLevel.GOALS_CLOSED)
        outcome = c.evaluate(GreenLevel.SORRY_FREE)
        assert outcome.satisfied is True
        assert outcome.gap == 0

    def test_not_satisfied(self):
        c = ProofGreenContract(required=GreenLevel.GOALS_CLOSED)
        outcome = c.evaluate(GreenLevel.SYNTAX_CLEAN)
        assert outcome.satisfied is False
        assert outcome.gap == 2

    def test_exact_match(self):
        c = ProofGreenContract(required=GreenLevel.TACTIC_VALID)
        assert c.is_satisfied_by(GreenLevel.TACTIC_VALID) is True

    def test_factory_for_deposit(self):
        c = ProofGreenContract.for_deposit()
        assert c.required == GreenLevel.GOALS_CLOSED

    def test_factory_for_certification(self):
        c = ProofGreenContract.for_certification()
        assert c.required == GreenLevel.SORRY_FREE

    def test_outcome_summary(self):
        c = ProofGreenContract.for_deposit()
        outcome = c.evaluate(GreenLevel.SYNTAX_CLEAN)
        assert "❌" in outcome.summary
        outcome2 = c.evaluate(GreenLevel.GOALS_CLOSED)
        assert "✅" in outcome2.summary


# ═══════════════════════════════════════════════════════════════════════
# SummaryCompression tests
# ═══════════════════════════════════════════════════════════════════════

def _make_sm(name="test"):
    ctx = TaskContext(
        theorem_name=name, formal_statement=f"theorem {name} : True",
        rounds_completed=3, total_samples=12)
    ctx.__dict__["_max_samples"] = 128
    ctx.__dict__["_current_strategy"] = "light"
    return ProofTaskStateMachine(task_id=f"lane_{name}", context=ctx)


class TestCompressProofStatus:
    def test_basic_fields(self):
        sm = _make_sm()
        s = compress_proof_status(sm)
        assert s.task_id == "lane_test"
        assert s.theorem_name == "test"
        assert s.rounds_completed == 3
        assert s.total_samples == 12

    def test_terminal_state(self):
        sm = _make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        sm.transition_to(TaskStatus.VERIFYING)
        sm.succeed("by trivial")
        s = compress_proof_status(sm)
        assert s.is_terminal is True
        assert s.status == "succeeded"

    def test_failure_tracking(self):
        sm = _make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        sm.transition_to(TaskStatus.VERIFYING)
        sm.fail(ProofFailureClass.TYPE_MISMATCH, "expected Nat", recoverable=True)
        sm.transition_to(TaskStatus.VERIFYING)
        sm.fail(ProofFailureClass.TYPE_MISMATCH, "expected Int", recoverable=True)

        s = compress_proof_status(sm)
        assert s.blocker_class == "type_mismatch"
        assert s.blocker_streak == 2
        assert s.failure_distribution["type_mismatch"] == 2

    def test_one_liner(self):
        sm = _make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        sm.transition_to(TaskStatus.VERIFYING)
        sm.fail(ProofFailureClass.TACTIC_FAILED, "simp failed", recoverable=True)

        s = compress_proof_status(sm)
        line = s.one_liner
        assert "BLOCKED" in line.upper() or "blocked" in line
        assert "tactic_failed" in line

    def test_for_prompt_within_budget(self):
        sm = _make_sm()
        s = compress_proof_status(sm)
        text = s.for_prompt(max_chars=500)
        assert len(text) <= 500
        assert "round 3" in text

    def test_policy_recommendation(self):
        sm = _make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        sm.transition_to(TaskStatus.VERIFYING)
        policy = PolicyEngine.default()
        s = compress_proof_status(sm, policy=policy)
        assert s.recommended_action != ""

    def test_to_dict_serializable(self):
        import json
        sm = _make_sm()
        s = compress_proof_status(sm)
        d = s.to_dict()
        json.dumps(d)  # should not raise

    def test_best_green_level_on_success(self):
        sm = _make_sm()
        sm.transition_to(TaskStatus.GENERATING)
        sm.transition_to(TaskStatus.VERIFYING)
        sm.succeed("by norm_num")
        s = compress_proof_status(sm)
        assert s.best_green_level == "goals_closed"


class TestCompressDashboard:
    def test_empty_dashboard(self):
        dashboard = ProofDashboard()
        result = compress_dashboard(dashboard)
        assert "0/0" in result["one_liner"]

    def test_dashboard_with_tasks(self):
        dashboard = ProofDashboard()
        sm1 = _make_sm("thm1")
        sm1.transition_to(TaskStatus.GENERATING)
        sm1.transition_to(TaskStatus.VERIFYING)
        sm1.succeed("done")
        dashboard.register_task(sm1)

        sm2 = _make_sm("thm2")
        sm2.transition_to(TaskStatus.GENERATING)
        dashboard.register_task(sm2)

        result = compress_dashboard(dashboard)
        assert "1/2" in result["one_liner"]
        assert len(result["active_one_liners"]) == 1  # sm2 is active
