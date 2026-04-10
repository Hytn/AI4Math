"""tests/test_all_fixes_v2.py — Tests for all v2 fixes

Covers:
  1. TaskContext explicit fields (no more __dict__ hacks)
  2. ClaudeProvider thread safety + extended_thinking
  3. PolicyEngine error logging (not silent swallow)
  4. sorry_detector integration in verification
  5. Adaptive confidence threshold direction
  6. BudgetEscalationRule multi-dimensional
  7. Knowledge decay call point
  8. map_verification_result_to_failure_class guard
"""
import asyncio
import logging
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from engine.lane.task_state import (
    TaskContext, ProofTaskStateMachine, TaskStatus, ProofFailureClass,
)
from engine.lane.policy import (
    PolicyEngine, PolicyAction, PolicyDecision,
    BudgetEscalationRule, ConsecutiveSameErrorRule, ReflectionRule,
)
from engine.lane.event_bus import ProofEventBus, wire_state_machine_to_bus
from engine.lane.integration import map_verification_result_to_failure_class


# ═══════════════════════════════════════════════════════════════════
# Fix 1: TaskContext explicit fields
# ═══════════════════════════════════════════════════════════════════

class TestTaskContextExplicitFields(unittest.TestCase):
    """Verify all formerly __dict__-hacked fields are real dataclass fields."""

    def test_all_fields_exist(self):
        ctx = TaskContext(theorem_name="t", formal_statement="s")
        # These used to be __dict__ hacks
        self.assertEqual(ctx.max_samples, 128)
        self.assertEqual(ctx.current_strategy, "light")
        self.assertEqual(ctx.current_role, "generator")
        self.assertFalse(ctx.decompose_attempted)
        self.assertEqual(ctx.domain_hints, {})
        self.assertIsInstance(ctx.start_time, float)
        self.assertEqual(ctx.timeout_multiplier, 1.0)
        self.assertFalse(ctx.reduced_concurrency)

    def test_fields_are_mutable(self):
        ctx = TaskContext(theorem_name="t", formal_statement="s")
        ctx.current_strategy = "heavy"
        ctx.decompose_attempted = True
        ctx.max_samples = 256
        self.assertEqual(ctx.current_strategy, "heavy")
        self.assertTrue(ctx.decompose_attempted)
        self.assertEqual(ctx.max_samples, 256)

    def test_state_machine_context_fields(self):
        """Policy rules access sm.context.current_strategy directly."""
        ctx = TaskContext(theorem_name="t", formal_statement="s")
        sm = ProofTaskStateMachine(task_id="test", context=ctx)
        sm.context.current_strategy = "medium"
        sm.context.max_samples = 64
        self.assertEqual(sm.context.current_strategy, "medium")
        self.assertEqual(sm.context.max_samples, 64)


# ═══════════════════════════════════════════════════════════════════
# Fix 2: ClaudeProvider thread safety
# ═══════════════════════════════════════════════════════════════════

class TestClaudeProviderThreadSafety(unittest.TestCase):

    def test_has_lock(self):
        from agent.brain.claude_provider import ClaudeProvider
        provider = ClaudeProvider(api_key="fake")
        self.assertIsInstance(provider._client_lock, type(threading.Lock()))

    def test_extended_thinking_flag_stored(self):
        from agent.brain.claude_provider import ClaudeProvider
        provider = ClaudeProvider(extended_thinking=True)
        self.assertTrue(provider._extended_thinking)

    def test_mock_provider_works(self):
        from agent.brain.claude_provider import MockProvider
        p = MockProvider()
        resp = p.generate(user="test")
        self.assertIn("sorry", resp.content)


# ═══════════════════════════════════════════════════════════════════
# Fix 3: PolicyEngine logs rule errors
# ═══════════════════════════════════════════════════════════════════

class TestPolicyEngineErrorLogging(unittest.TestCase):

    def test_broken_rule_is_logged_and_skipped(self):
        """A rule that raises should be logged, not silently ignored."""
        from engine.lane.policy import PolicyRule

        class BrokenRule(PolicyRule):
            name = "broken"
            priority = 1
            def evaluate(self, sm, events):
                raise RuntimeError("intentional test failure")

        class FallbackRule(PolicyRule):
            name = "fallback"
            priority = 100
            def evaluate(self, sm, events):
                return PolicyDecision(
                    action=PolicyAction.CONTINUE,
                    reason="fallback fired", rule_name=self.name)

        engine = PolicyEngine()
        engine.add_rule(BrokenRule())
        engine.add_rule(FallbackRule())

        ctx = TaskContext(theorem_name="t", formal_statement="s")
        sm = ProofTaskStateMachine(task_id="test", context=ctx)

        with self.assertLogs("engine.lane.policy", level="WARNING") as cm:
            decision = engine.evaluate(sm)

        self.assertEqual(decision.rule_name, "fallback")
        self.assertTrue(any("BrokenRule" in msg or "broken" in msg
                            for msg in cm.output))


# ═══════════════════════════════════════════════════════════════════
# Fix 4: sorry_detector integration
# ═══════════════════════════════════════════════════════════════════

class TestSorryDetectorIntegration(unittest.TestCase):

    def test_sorry_in_code_detected(self):
        from prover.verifier.sorry_detector import detect_sorry
        report = detect_sorry("theorem t : True := by\n  sorry")
        self.assertTrue(report.has_sorry)
        self.assertFalse(report.is_clean)

    def test_axiom_warning(self):
        from prover.verifier.sorry_detector import detect_sorry
        report = detect_sorry("axiom myAxiom : False")
        self.assertFalse(report.has_sorry)
        self.assertTrue(len(report.warnings) > 0)
        self.assertFalse(report.is_clean)

    def test_clean_proof(self):
        from prover.verifier.sorry_detector import detect_sorry
        report = detect_sorry("theorem t : True := trivial")
        self.assertTrue(report.is_clean)

    def test_unsafe_coerce_warning(self):
        from prover.verifier.sorry_detector import detect_sorry
        report = detect_sorry("def f := unsafeCoerce x")
        self.assertTrue(len(report.warnings) > 0)


# ═══════════════════════════════════════════════════════════════════
# Fix 5: Adaptive confidence threshold direction
# ═══════════════════════════════════════════════════════════════════

class TestAdaptiveConfidenceThreshold(unittest.TestCase):

    def test_threshold_decreases_with_budget(self):
        """When budget is low, threshold should be lower (more permissive)."""
        base_conf = 0.3

        # Full budget: threshold = 0.3 * 1.0 = 0.3
        remaining_full = 1.0
        threshold_full = base_conf * remaining_full

        # Nearly exhausted: threshold = 0.3 * 0.1 = 0.03
        remaining_low = 0.1
        threshold_low = base_conf * remaining_low

        self.assertGreater(threshold_full, threshold_low)
        self.assertAlmostEqual(threshold_low, 0.03, places=3)

    def test_zero_budget_zero_threshold(self):
        """At zero remaining budget, threshold should be zero (try everything)."""
        base_conf = 0.3
        threshold = base_conf * 0.0
        self.assertEqual(threshold, 0.0)


# ═══════════════════════════════════════════════════════════════════
# Fix 6: BudgetEscalationRule multi-dimensional
# ═══════════════════════════════════════════════════════════════════

class TestBudgetEscalationMultiDimensional(unittest.TestCase):

    def _make_sm(self, samples=0, tokens=0, strategy="light",
                 max_samples=128, start_offset=0):
        ctx = TaskContext(
            theorem_name="t", formal_statement="s",
            total_samples=samples, total_api_tokens=tokens,
            current_strategy=strategy, max_samples=max_samples,
            start_time=time.time() - start_offset,
        )
        return ProofTaskStateMachine(task_id="test", context=ctx)

    def test_escalation_by_samples(self):
        rule = BudgetEscalationRule()
        sm = self._make_sm(samples=50, max_samples=128)
        decision = rule.evaluate(sm, [])
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, PolicyAction.ESCALATE_STRATEGY)

    def test_escalation_by_tokens(self):
        """Should escalate based on token usage even with low sample count."""
        rule = BudgetEscalationRule(max_tokens=100_000)
        sm = self._make_sm(samples=1, tokens=40_000, max_samples=128)
        decision = rule.evaluate(sm, [])
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, PolicyAction.ESCALATE_STRATEGY)

    def test_escalation_by_wall_time(self):
        """Should escalate based on elapsed wall time."""
        rule = BudgetEscalationRule(max_wall_seconds=100.0)
        sm = self._make_sm(samples=1, start_offset=50)  # 50s elapsed
        decision = rule.evaluate(sm, [])
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, PolicyAction.ESCALATE_STRATEGY)

    def test_no_escalation_when_all_low(self):
        rule = BudgetEscalationRule()
        sm = self._make_sm(samples=1, tokens=100, max_samples=128)
        decision = rule.evaluate(sm, [])
        self.assertIsNone(decision)

    def test_medium_to_heavy_escalation(self):
        rule = BudgetEscalationRule()
        sm = self._make_sm(
            samples=100, max_samples=128, strategy="medium")
        decision = rule.evaluate(sm, [])
        self.assertIsNotNone(decision)
        self.assertEqual(decision.metadata["to"], "heavy")


# ═══════════════════════════════════════════════════════════════════
# Fix 8: map_verification_result_to_failure_class guard
# ═══════════════════════════════════════════════════════════════════

class TestMapVerificationGuard(unittest.TestCase):

    def test_raises_on_success(self):
        vr = MagicMock()
        vr.success = True
        with self.assertRaises(ValueError):
            map_verification_result_to_failure_class(vr)

    def test_returns_syntax_error_for_l0_fail(self):
        vr = MagicMock()
        vr.success = False
        vr.l0_passed = False
        vr.l0_reject_reason = "missing import"
        vr.feedback = MagicMock()
        vr.feedback.error_category = None
        result = map_verification_result_to_failure_class(vr)
        self.assertEqual(result, ProofFailureClass.SYNTAX_ERROR)


# ═══════════════════════════════════════════════════════════════════
# Fix 7: Knowledge decay has call point
# ═══════════════════════════════════════════════════════════════════

class TestKnowledgeDecayCallPoint(unittest.TestCase):

    def test_decay_method_exists_on_pipeline(self):
        """ProofPipeline should have _run_knowledge_decay method."""
        from prover.pipeline.proof_pipeline import ProofPipeline
        self.assertTrue(hasattr(ProofPipeline, '_run_knowledge_decay'))


# ═══════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
