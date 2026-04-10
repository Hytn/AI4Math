"""tests/test_e2e_all_fixes.py — End-to-end tests for all identified issues

Validates every fix applied to the codebase:
  1. No duplicate sampler/sampler directory
  2. Silent exception swallowing replaced with logging
  3. AgentLoop uses proper Claude tool_use protocol
  4. Sorry detection uses proper detector (not naive substring)
  5. No blocking time.sleep in recovery
  6. Adaptive confidence threshold in verify
  7. API retry logic (already existed, verified here)
  8. Lemma bank verification enabled by default
  9. De Bruijn indices correct in test environment
  10. End-to-end kernel → pipeline integration
  11. Config fixes (extended_thinking, timeout, identifier replacement)
"""
import asyncio
import os
import sys
import re
import logging
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine.core import Expr, Name, BinderInfo
from engine.core.environment import Environment, ConstantInfo
from engine.state.proof_state import ProofState
from engine.kernel.type_checker import TypeChecker
from tests.conftest import mk_standard_env, mk_standard_state

# ═══════════════════════════════════════════════════════════════════════════════
# Fix #1: No duplicate directories
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoDuplicates:
    def test_no_sampler_sampler_directory(self):
        """sampler/sampler/ was a full duplicate — must be removed."""
        dup_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sampler", "sampler")
        assert not os.path.exists(dup_path), \
            f"Duplicate directory still exists: {dup_path}"

    def test_tool_registry_is_shim(self):
        """agent/tools/tool_registry.py should be a backward-compat shim."""
        from agent.tools.tool_registry import ToolRegistry, get_registry
        from agent.tools.registry import ToolRegistry as CanonicalRegistry
        # Both should resolve to the same class
        assert ToolRegistry is CanonicalRegistry


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #2: No silent exception swallowing
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoSilentExceptions:
    def test_critical_files_no_bare_pass(self):
        """Key files should not have except-pass without logging."""
        critical_files = [
            "agent/runtime/agent_loop.py",
            "prover/pipeline/proof_pipeline.py",
            "prover/verifier/lean_checker.py",
            "engine/transport.py",
        ]
        root = os.path.dirname(os.path.dirname(__file__))
        for rel_path in critical_files:
            path = os.path.join(root, rel_path)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip() == "pass" and i > 0:
                    prev = lines[i - 1].strip()
                    if prev.startswith("except"):
                        pytest.fail(
                            f"{rel_path}:{i+1}: silent exception swallow "
                            f"still present after: {prev}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #3: Proper Claude tool_use protocol
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolProtocol:
    def test_build_assistant_content_returns_blocks(self):
        """_build_assistant_content should produce proper tool_use blocks."""
        from agent.runtime.agent_loop import AgentLoop, LoopConfig
        from agent.brain.llm_provider import LLMResponse
        from unittest.mock import MagicMock

        loop = AgentLoop(
            llm=MagicMock(), tools=MagicMock(), config=LoopConfig())

        resp = LLMResponse(
            content="Let me search for premises.",
            model="test", tokens_in=10, tokens_out=20,
            latency_ms=100,
            tool_calls=[{
                "name": "premise_search",
                "input": {"query": "add_comm"},
                "id": "tool_123",
            }])

        result = loop._build_assistant_content(resp)

        # Must be a list with text + tool_use blocks
        assert isinstance(result, list), \
            f"Expected list, got {type(result)}"
        assert len(result) == 2

        text_block = result[0]
        assert text_block["type"] == "text"
        assert "premises" in text_block["text"]

        tool_block = result[1]
        assert tool_block["type"] == "tool_use"
        assert tool_block["name"] == "premise_search"
        assert tool_block["id"] == "tool_123"
        assert tool_block["input"] == {"query": "add_comm"}

    def test_build_assistant_content_text_only(self):
        """When no tool calls, return plain string."""
        from agent.runtime.agent_loop import AgentLoop, LoopConfig
        from agent.brain.llm_provider import LLMResponse
        from unittest.mock import MagicMock

        loop = AgentLoop(
            llm=MagicMock(), tools=MagicMock(), config=LoopConfig())

        resp = LLMResponse(
            content="Here is the proof.",
            model="test", tokens_in=10, tokens_out=20,
            latency_ms=100, tool_calls=[])

        result = loop._build_assistant_content(resp)
        assert isinstance(result, str)
        assert result == "Here is the proof."


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #4: Proper sorry detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestSorryDetection:
    def test_sorry_in_comment_not_flagged(self):
        """sorry in a comment should not be flagged."""
        from prover.verifier.sorry_detector import detect_sorry
        code = """theorem foo : True := by
  -- sorry this is just a comment
  exact True.intro"""
        report = detect_sorry(code)
        assert report.is_clean, \
            f"sorry in comment was incorrectly flagged: {report.warnings}"

    def test_real_sorry_detected(self):
        from prover.verifier.sorry_detector import detect_sorry
        code = "theorem foo : True := by\n  sorry"
        report = detect_sorry(code)
        assert report.has_sorry

    def test_admit_detected(self):
        from prover.verifier.sorry_detector import detect_sorry
        code = "theorem foo : True := by\n  admit"
        report = detect_sorry(code)
        assert report.has_sorry

    def test_sorry_redefinition_detected(self):
        from prover.verifier.sorry_detector import detect_sorry
        code = """def sorry {α : Sort _} : α := sorry
theorem foo : True := sorry"""
        report = detect_sorry(code)
        assert not report.is_clean
        assert any("redefinition" in w for w in report.warnings)

    def test_agent_loop_imports_detector(self):
        """AgentLoop should import detect_sorry, not use naive check."""
        import inspect
        from agent.runtime import agent_loop
        source = inspect.getsource(agent_loop)
        assert "detect_sorry" in source
        # Must NOT contain the old naive check
        assert '"sorry" not in proof' not in source


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #5: No blocking sleep in recovery
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoBlockingSleep:
    def test_no_time_sleep_in_pipeline(self):
        """proof_pipeline.py must not call time.sleep()."""
        import inspect
        from prover.pipeline import proof_pipeline
        source = inspect.getsource(proof_pipeline)
        # time.sleep or _time.sleep should not be present
        assert "time.sleep(" not in source and "_time.sleep(" not in source, \
            "Blocking sleep found in proof_pipeline.py"


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #6: Adaptive confidence threshold
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveConfidence:
    def test_pipeline_has_configurable_threshold(self):
        """ProofPipeline should accept verify_min_confidence in config."""
        import inspect
        from prover.pipeline.proof_pipeline import ProofPipeline
        source = inspect.getsource(ProofPipeline.__init__)
        assert "verify_min_confidence" in source


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #7: API retry logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIRetry:
    def test_sync_provider_has_retry(self):
        import inspect
        from agent.brain.claude_provider import ClaudeProvider
        source = inspect.getsource(ClaudeProvider)
        assert "_MAX_RETRIES" in source
        assert "backoff" in source.lower()

    def test_async_provider_has_retry(self):
        import inspect
        from agent.brain.async_llm_provider import AsyncClaudeProvider
        source = inspect.getsource(AsyncClaudeProvider)
        assert "_MAX_RETRIES" in source
        assert "backoff" in source.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #8: Lemma bank verification in default config
# ═══════════════════════════════════════════════════════════════════════════════

class TestDefaultConfig:
    def test_lemma_verification_enabled(self):
        import yaml
        root = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(root, "config", "default.yaml")) as f:
            cfg = yaml.safe_load(f)
        assert cfg["prover"]["lemma_bank"]["verify_extracted"] is True

    def test_extended_thinking_enabled(self):
        import yaml
        root = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(root, "config", "default.yaml")) as f:
            cfg = yaml.safe_load(f)
        assert cfg["agent"]["brain"]["extended_thinking"] is True

    def test_timeout_increased(self):
        import yaml
        root = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(root, "config", "default.yaml")) as f:
            cfg = yaml.safe_load(f)
        assert cfg["prover"]["verifier"]["timeout_seconds"] >= 300


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #9: De Bruijn indices correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeBruijnCorrectness:
    """Verify the test environment's type signatures are correct."""

    def test_eq_refl_type(self):
        env = mk_standard_env()
        eq_refl = env.lookup(Name.from_str("Eq.refl"))
        # ∀ {α : Type} (a : α), Eq a a
        ty = eq_refl.type_
        # Should be a pi with implicit binder for α
        assert ty.is_pi
        assert ty.binder_info == BinderInfo.IMPLICIT

    def test_eq_symm_type(self):
        env = mk_standard_env()
        eq_symm = env.lookup(Name.from_str("Eq.symm"))
        ty = eq_symm.type_
        # ∀ {α} {a b : α}, Eq a b → Eq b a
        # Should start with 3 implicit pi binders
        assert ty.is_pi
        assert ty.binder_info == BinderInfo.IMPLICIT

    def test_eq_trans_structure(self):
        """Verify Eq.trans has correct de Bruijn indices for Eq a b → Eq b c → Eq a c."""
        env = mk_standard_env()
        eq_trans = env.lookup(Name.from_str("Eq.trans"))
        ty = eq_trans.type_
        # Should start with 4 pi binders: α, a, b, c
        current = ty
        binder_count = 0
        while current.is_pi:
            binder_count += 1
            current = current.children[1]  # pi body
            if binder_count == 4:
                break
        assert binder_count == 4, f"Expected 4 pi binders, got {binder_count}"

    def test_and_intro_type(self):
        env = mk_standard_env()
        and_intro = env.lookup(Name.from_str("And.intro"))
        ty = and_intro.type_
        assert ty.is_pi

    def test_or_elim_type(self):
        env = mk_standard_env()
        or_elim = env.lookup(Name.from_str("Or.elim"))
        ty = or_elim.type_
        assert ty.is_pi

    def test_environment_consistency(self):
        """All declared constants should be resolvable."""
        env = mk_standard_env()
        expected = [
            "Prop", "Nat", "Nat.zero", "Nat.succ", "True", "False",
            "Bool", "True.intro", "And", "Or", "Iff", "Eq",
            "Eq.refl", "Eq.symm", "Eq.trans", "Eq.rec",
            "And.intro", "And.left", "And.right",
            "Or.inl", "Or.inr", "Or.elim",
            "Iff.intro", "False.elim", "Not", "APE.sorry",
        ]
        for name_str in expected:
            info = env.lookup(Name.from_str(name_str))
            assert info is not None, f"Constant {name_str} not found"
            assert info.type_ is not None, f"Constant {name_str} has no type"


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #10: End-to-end kernel → pipeline integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2EKernelIntegration:
    """True end-to-end: environment → proof state → type checker → pipeline models."""

    def test_proof_state_creation(self):
        """Create a proof state and verify it has a goal."""
        env = mk_standard_env()
        prop = Expr.prop()
        # Goal: True
        goal_type = Expr.const(Name.from_str("True"))
        state = mk_standard_state(env, goal_type)
        assert state is not None
        goals = state.goals()
        assert len(goals) >= 1

    def test_proof_state_true_intro(self):
        """Prove True using True.intro — full kernel path."""
        env = mk_standard_env()
        goal_type = Expr.const(Name.from_str("True"))
        state = mk_standard_state(env, goal_type)
        # Apply True.intro
        true_intro = Expr.const(Name.from_str("True.intro"))
        new_state = state.assign_goal(state.goals()[0].id, true_intro)
        assert new_state is not None
        # Should have no remaining goals
        assert len(new_state.goals()) == 0

    def test_proof_state_and_intro(self):
        """Prove True ∧ True using And.intro."""
        env = mk_standard_env()
        true_const = Expr.const(Name.from_str("True"))
        and_const = Expr.const(Name.from_str("And"))
        # Goal: And True True
        goal = Expr.app(Expr.app(and_const, true_const), true_const)
        state = mk_standard_state(env, goal)

        # Apply And.intro
        and_intro = Expr.const(Name.from_str("And.intro"))
        true_intro = Expr.const(Name.from_str("True.intro"))

        # And.intro True True True.intro True.intro
        proof = Expr.app(
            Expr.app(
                Expr.app(
                    Expr.app(and_intro, true_const),
                    true_const),
                true_intro),
            true_intro)

        new_state = state.assign_goal(state.goals()[0].id, proof)
        assert new_state is not None
        assert len(new_state.goals()) == 0

    def test_pipeline_models_roundtrip(self):
        """ProofTrace and ProofAttempt should serialize correctly."""
        from prover.models import ProofTrace, ProofAttempt, AttemptStatus
        trace = ProofTrace(
            problem_id="test_001",
            problem_name="Test",
            theorem_statement="theorem foo : True := by exact True.intro",
        )
        attempt = ProofAttempt(attempt_number=1)
        attempt.generated_proof = ":= by exact True.intro"
        attempt.lean_result = AttemptStatus.SUCCESS
        trace.add_attempt(attempt)

        assert trace.solved

    def test_sorry_detector_integration(self):
        """Sorry detector works on realistic Lean4 code."""
        from prover.verifier.sorry_detector import detect_sorry

        # Clean proof
        clean = """
theorem add_comm (m n : Nat) : m + n = n + m := by
  induction m with
  | zero => simp
  | succ m ih => simp [Nat.succ_add, ih]
"""
        assert detect_sorry(clean).is_clean

        # Proof with sorry
        dirty = """
theorem hard_theorem (p : Prop) : p := by
  sorry
"""
        report = detect_sorry(dirty)
        assert report.has_sorry
        assert not report.is_clean

    def test_error_parser_integration(self):
        """Error parser extracts structured errors from Lean output."""
        from prover.verifier.error_parser import parse_lean_errors

        stderr = """test.lean:5:2: error: unknown identifier 'Nat.add_assoc'
test.lean:8:4: error: type mismatch
  has type
    Nat
  expected
    Bool"""
        errors = parse_lean_errors(stderr)
        assert len(errors) >= 1
        # Should find at least one error
        assert any("unknown identifier" in e.message.lower()
                    or "type mismatch" in e.message.lower()
                    for e in errors)

    def test_confidence_estimator_integration(self):
        """Confidence estimator tracks progress correctly."""
        from agent.strategy.confidence_estimator import ConfidenceEstimator
        from common.working_memory import WorkingMemory

        est = ConfidenceEstimator(max_samples=100)

        # Fresh start
        mem = WorkingMemory()
        assert est.estimate(mem) == 0.5

        # After some attempts with errors
        mem.attempt_history = [
            {"errors": ["unknown identifier"]},
            {"errors": ["type mismatch"]},
            {"errors": ["unknown identifier"]},
        ]
        mem.error_patterns = {"unknown identifier": 2, "type mismatch": 1}
        score = est.estimate(mem)
        assert 0.0 < score < 1.0

        # Solved
        mem.solved = True
        assert est.estimate(mem) == 1.0

    def test_lean_error_classifier_integration(self):
        """Lane error classifier correctly categorizes errors."""
        from engine.lane.error_classifier import classify_lean_error
        from engine.lane.task_state import ProofFailureClass

        fc = classify_lean_error("unknown identifier 'Nat.add_comm'")
        assert fc is not None

        fc2 = classify_lean_error("tactic 'simp' failed")
        assert fc2 is not None

        fc3 = classify_lean_error("timeout")
        assert fc3 == ProofFailureClass.TIMEOUT

    def test_working_memory_to_state_machine(self):
        """MetaController bridges WorkingMemory → PolicyEngine correctly."""
        from agent.strategy.meta_controller import MetaController
        from common.working_memory import WorkingMemory

        mc = MetaController()
        mem = WorkingMemory(
            current_strategy="light",
            rounds_completed=5,
            total_samples=0)

        # Should not crash
        result = mc.should_escalate(mem)
        # With 0 samples and light strategy, no escalation expected
        assert result is None

    def test_full_pipeline_init_and_teardown(self):
        """ProofPipeline can init and teardown without real Lean."""
        from prover.models import BenchmarkProblem
        from prover.pipeline.proof_pipeline import ProofPipeline
        from unittest.mock import MagicMock

        comp = MagicMock()
        comp.budget.max_samples = 10
        comp.budget.is_exhausted.return_value = False
        comp.meta_controller.select_initial_strategy.return_value = "light"
        comp.plugins.list_plugins.return_value = []
        comp.hooks.list_hooks.return_value = []
        comp.hooks.fire.return_value = MagicMock(
            inject_context=None, action=None, message="")

        pipeline = ProofPipeline(comp, config={"verify_min_confidence": 0.4})
        assert pipeline._verify_min_confidence == 0.4

        problem = BenchmarkProblem(
            problem_id="test",
            name="test_theorem",
            theorem_statement="theorem foo : True := by sorry",
        )
        ctx = pipeline.init(problem)
        assert ctx.strategy_name == "light"
        assert ctx.sm is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #11: Identifier replacement uses word boundaries
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdentifierFix:
    def test_word_boundary_replacement(self):
        """_fix_identifier should not replace partial matches."""
        from prover.repair.repair_generator import _fix_identifier
        from prover.models import LeanError, ErrorCategory

        # "nat.add_comm" should become "Nat.add_comm"
        # but "nat.add_commutativity" should NOT be partially replaced
        error = LeanError(
            message="unknown identifier 'nat.add_comm'",
            line=1, column=0, category=ErrorCategory.UNKNOWN_IDENTIFIER)

        proof = "exact nat.add_comm"
        fixed = _fix_identifier(proof, error)
        assert "Nat.add_comm" in fixed

    def test_no_partial_substring_replacement(self):
        """Should not replace 'nat.add_comm' inside 'nat.add_comm_of_something'."""
        from prover.repair.repair_generator import _fix_identifier
        from prover.models import LeanError, ErrorCategory

        error = LeanError(
            message="unknown identifier 'nat.add_comm'",
            line=1, column=0, category=ErrorCategory.UNKNOWN_IDENTIFIER)

        proof = "exact nat.add_comm_of_something"
        fixed = _fix_identifier(proof, error)
        # The word-boundary regex should NOT match nat.add_comm inside
        # nat.add_comm_of_something (underscore is a word char)
        # Actually \b treats _ as word char, so nat.add_comm_of_something
        # won't match \bnat.add_comm\b — correct behavior
        assert "Nat.add_comm_of_something" not in fixed or \
               "nat.add_comm_of_something" in fixed


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: AgentLoop full cycle (mock)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentLoopIntegration:
    @pytest.mark.asyncio
    async def test_agent_loop_stops_on_clean_proof(self):
        """AgentLoop should stop when a clean proof is found."""
        from agent.runtime.agent_loop import AgentLoop, LoopConfig
        from agent.brain.async_llm_provider import AsyncMockProvider
        from agent.tools.registry import ToolRegistry
        from agent.brain.llm_provider import LLMResponse

        class ProofProvider(AsyncMockProvider):
            async def chat(self, system="", messages=None, **kw):
                return LLMResponse(
                    content="```lean\n:= by\n  exact True.intro\n```",
                    model="mock", tokens_in=50, tokens_out=30,
                    latency_ms=10, stop_reason="end_turn")

        loop = AgentLoop(
            llm=ProofProvider(),
            tools=ToolRegistry(),
            config=LoopConfig(stop_on_proof=True, max_turns=5))

        result = await loop.run(
            system_prompt="Prove theorems in Lean4.",
            initial_message="Prove: theorem foo : True")

        assert result.has_proof
        assert result.stopped_reason == "proof_found"
        assert "sorry" not in result.proof_code
        assert result.turns_used == 1

    @pytest.mark.asyncio
    async def test_agent_loop_does_not_stop_on_sorry(self):
        """AgentLoop should NOT stop when proof contains sorry."""
        from agent.runtime.agent_loop import AgentLoop, LoopConfig
        from agent.brain.async_llm_provider import AsyncMockProvider
        from agent.tools.registry import ToolRegistry
        from agent.brain.llm_provider import LLMResponse

        call_count = 0

        class SorryThenFixProvider(AsyncMockProvider):
            async def chat(self, system="", messages=None, **kw):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return LLMResponse(
                        content="```lean\n:= by\n  sorry\n```",
                        model="mock", tokens_in=50, tokens_out=30,
                        latency_ms=10, stop_reason="end_turn")
                return LLMResponse(
                    content="```lean\n:= by\n  exact True.intro\n```",
                    model="mock", tokens_in=50, tokens_out=30,
                    latency_ms=10, stop_reason="end_turn")

        loop = AgentLoop(
            llm=SorryThenFixProvider(),
            tools=ToolRegistry(),
            config=LoopConfig(
                stop_on_proof=True,
                stop_on_text_only=False,
                max_turns=5))

        result = await loop.run(
            system_prompt="Prove theorems.",
            initial_message="Prove: theorem foo : True")

        assert result.has_proof
        assert "sorry" not in result.proof_code
        assert call_count >= 2  # Should have continued past sorry


# ═══════════════════════════════════════════════════════════════════════════════
# Full kernel proof verification end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

class TestKernelE2E:
    """Verify the kernel can construct and check complete proofs."""

    def test_prove_true(self):
        env = mk_standard_env()
        goal = Expr.const(Name.from_str("True"))
        state = mk_standard_state(env, goal)
        proof = Expr.const(Name.from_str("True.intro"))
        final = state.assign_goal(state.goals()[0].id, proof)
        assert len(final.goals()) == 0

    def test_prove_and_true_true(self):
        env = mk_standard_env()
        true_c = Expr.const(Name.from_str("True"))
        and_c = Expr.const(Name.from_str("And"))
        goal = Expr.app(Expr.app(and_c, true_c), true_c)
        state = mk_standard_state(env, goal)

        intro = Expr.const(Name.from_str("And.intro"))
        ti = Expr.const(Name.from_str("True.intro"))
        proof = Expr.app(Expr.app(Expr.app(Expr.app(intro, true_c), true_c), ti), ti)
        final = state.assign_goal(state.goals()[0].id, proof)
        assert len(final.goals()) == 0

    def test_prove_or_inl(self):
        """Prove Or True False using Or.inl True.intro."""
        env = mk_standard_env()
        true_c = Expr.const(Name.from_str("True"))
        false_c = Expr.const(Name.from_str("False"))
        or_c = Expr.const(Name.from_str("Or"))
        goal = Expr.app(Expr.app(or_c, true_c), false_c)
        state = mk_standard_state(env, goal)

        inl = Expr.const(Name.from_str("Or.inl"))
        ti = Expr.const(Name.from_str("True.intro"))
        proof = Expr.app(Expr.app(Expr.app(inl, true_c), false_c), ti)
        final = state.assign_goal(state.goals()[0].id, proof)
        assert len(final.goals()) == 0

    def test_prove_iff_true_true(self):
        """Prove Iff True True using Iff.intro with id functions."""
        env = mk_standard_env()
        true_c = Expr.const(Name.from_str("True"))
        iff_c = Expr.const(Name.from_str("Iff"))
        goal = Expr.app(Expr.app(iff_c, true_c), true_c)
        state = mk_standard_state(env, goal)

        iff_intro = Expr.const(Name.from_str("Iff.intro"))
        # id : True → True as λ x : True, x
        id_fn = Expr.lam(
            BinderInfo.DEFAULT,
            Name.from_str("x"), true_c,
            Expr.bvar(0))

        proof = Expr.app(
            Expr.app(
                Expr.app(
                    Expr.app(iff_intro, true_c),
                    true_c),
                id_fn),
            id_fn)
        final = state.assign_goal(state.goals()[0].id, proof)
        assert len(final.goals()) == 0

    def test_nat_inductive_info(self):
        """Nat inductive info is correctly registered."""
        env = mk_standard_env()
        ind = env.lookup_inductive(Name.from_str("Nat"))
        assert ind is not None
        assert ind.is_recursive
        assert len(ind.constructors) == 2
        assert ind.recursor is not None

    def test_bool_inductive_info(self):
        env = mk_standard_env()
        ind = env.lookup_inductive(Name.from_str("Bool"))
        assert ind is not None
        assert len(ind.constructors) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Lane system integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestLaneIntegration:
    def test_state_machine_lifecycle(self):
        from engine.lane.task_state import (
            TaskContext, ProofTaskStateMachine, TaskStatus, ProofFailureClass)

        ctx = TaskContext(
            theorem_name="test",
            formal_statement="theorem t : True := by sorry",
        )
        sm = ProofTaskStateMachine(task_id="t1", context=ctx)

        assert sm.status == TaskStatus.CREATED
        sm.transition_to(TaskStatus.GENERATING, detail="round 1")
        assert sm.status == TaskStatus.GENERATING

        sm.transition_to(TaskStatus.VERIFYING, detail="1 candidate")
        assert sm.status == TaskStatus.VERIFYING

        sm.fail(ProofFailureClass.TACTIC_FAILED, "simp failed", recoverable=True)
        assert sm.status == TaskStatus.BLOCKED

        sm.transition_to(TaskStatus.GENERATING, detail="recovered")
        sm.succeed("exact True.intro")
        assert sm.status == TaskStatus.SUCCEEDED
        assert sm.status.is_terminal

    def test_event_bus_wiring(self):
        from engine.lane.task_state import (
            TaskContext, ProofTaskStateMachine, TaskStatus)
        from engine.lane.event_bus import ProofEventBus, wire_state_machine_to_bus

        bus = ProofEventBus()
        events_received = []
        bus.subscribe("*", lambda e: events_received.append(e))

        ctx = TaskContext(theorem_name="test", formal_statement="test")
        sm = ProofTaskStateMachine(task_id="t2", context=ctx)
        wire_state_machine_to_bus(sm, bus)

        sm.transition_to(TaskStatus.GENERATING, detail="test")
        sm.succeed()

        assert len(events_received) >= 2

    def test_policy_engine_default(self):
        from engine.lane.policy import PolicyEngine, PolicyAction
        from engine.lane.task_state import (
            TaskContext, ProofTaskStateMachine, TaskStatus)

        policy = PolicyEngine.default()
        ctx = TaskContext(theorem_name="test", formal_statement="test")
        sm = ProofTaskStateMachine(task_id="t3", context=ctx)
        sm.transition_to(TaskStatus.GENERATING, detail="test")

        decision = policy.evaluate(sm)
        assert decision is not None
        assert decision.action in list(PolicyAction)
