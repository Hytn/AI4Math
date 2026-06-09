"""tests/test_fixes.py — Tests for all 12 fixes"""
import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDialogResumeSuccessRecovery:
    """Recover cached dialogs where final lean_verify succeeded at max_turns."""

    def _dialog(self, payload):
        import json
        return {
            "meta": {"problem_id": "p1", "problem_name": "t"},
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "lean_verify",
                            "arguments": json.dumps({
                                "code": "theorem t : True := by trivial"
                            }),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "lean_verify",
                    "content": json.dumps(payload),
                },
            ],
            "result": {
                "success": False,
                "termination": "max_turns",
                "successful_proof": "stale assistant text",
            },
        }

    def test_dialog_to_trace_recovers_verified_lean_tool_success(self):
        from run_eval import _dialog_to_trace_dict

        trace = _dialog_to_trace_dict(
            self._dialog({
                "verified": True,
                "sorry_free": True,
                "errors": [],
            }),
            fallback_problem_id="fallback",
        )

        assert trace["solved"] is True
        assert trace["correct_count"] == 1
        assert trace["successful_proof"] == "theorem t : True := by trivial"

    def test_dialog_to_trace_rejects_integrity_violation_success(self):
        from run_eval import _dialog_to_trace_dict

        trace = _dialog_to_trace_dict(
            self._dialog({
                "verified": True,
                "sorry_free": True,
                "errors": [],
                "integrity_violations": ["[critical] Uses native_decide"],
            }),
            fallback_problem_id="fallback",
        )

        assert trace["solved"] is False
        assert trace["correct_count"] == 0
        assert trace["successful_proof"] == ""

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════

class TestAutoVerifyResumeEvidence:
    def test_dialog_to_trace_accepts_auto_verified_final_proof(self):
        import json
        from run_eval import _dialog_to_trace_dict

        proof = "theorem target : True := by trivial"
        dialog = {
            "meta": {
                "problem_id": "p1",
                "problem_name": "target",
                "theorem_statement": "theorem target : True",
            },
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "lean_verify",
                            "arguments": json.dumps({"code": proof}),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "lean_verify",
                    "content": json.dumps({
                        "verified": False,
                        "sorry_free": True,
                        "errors": ["old failed attempt"],
                    }),
                },
            ],
            "result": {
                "success": True,
                "termination": "proof_found",
                "successful_proof": proof,
                "extra": {
                    "auto_verify": {
                        "source": "runner.auto_verify",
                        "backend": "lean4",
                        "verified": True,
                        "proves_target": True,
                        "sorry_free": True,
                    },
                },
            },
        }

        trace = _dialog_to_trace_dict(dialog, fallback_problem_id="fallback")

        assert trace["solved"] is True
        assert trace["correct_count"] == 1
        assert trace["successful_proof"] == proof

    def test_dialog_to_trace_rejects_auto_verify_wrong_target(self):
        from run_eval import _dialog_to_trace_dict

        dialog = {
            "meta": {
                "problem_id": "p1",
                "problem_name": "target",
                "theorem_statement": "theorem target : True",
            },
            "messages": [],
            "result": {
                "success": True,
                "termination": "proof_found",
                "successful_proof": "theorem helper : True := by trivial",
                "extra": {
                    "auto_verify": {
                        "verified": True,
                        "proves_target": True,
                        "sorry_free": True,
                    },
                },
            },
        }

        trace = _dialog_to_trace_dict(dialog, fallback_problem_id="fallback")

        assert trace["solved"] is False
        assert trace["successful_proof"] == ""


class TestNestedCommentStrip:
    """Fix #6: _strip_comments must handle nested /- -/ correctly."""

    def test_simple_block_comment(self):
        from prover.verifier.integrity_checker import _strip_comments
        code = "hello /- comment -/ world"
        assert "hello" in _strip_comments(code)
        assert "world" in _strip_comments(code)
        assert "comment" not in _strip_comments(code)

    def test_nested_block_comment(self):
        from prover.verifier.integrity_checker import _strip_comments
        code = "before /- outer /- inner -/ still_outer -/ after"
        result = _strip_comments(code)
        assert "before" in result
        assert "after" in result
        assert "inner" not in result
        assert "outer" not in result
        assert "still_outer" not in result

    def test_deeply_nested(self):
        from prover.verifier.integrity_checker import _strip_comments
        code = "ok /- a /- b /- c -/ d -/ e -/ end"
        result = _strip_comments(code)
        assert "ok" in result
        assert "end" in result
        assert "a" not in result
        assert "c" not in result

    def test_sorry_hidden_in_nested_comment(self):
        """Malicious proof hiding sorry inside nested comments."""
        from prover.verifier.integrity_checker import check_integrity
        # This code has sorry OUTSIDE any comment
        code_with_sorry = """
        theorem test : True := by
          /- /- nested -/ -/
          sorry
        """
        report = check_integrity(code_with_sorry)
        assert not report.passed, "sorry outside comments must be detected"

    def test_sorry_inside_nested_comment_is_safe(self):
        from prover.verifier.integrity_checker import _strip_comments
        code = "/- /- sorry -/ still comment -/"
        result = _strip_comments(code)
        assert "sorry" not in result

    def test_line_comment(self):
        from prover.verifier.integrity_checker import _strip_comments
        code = "hello -- this is a comment\nworld"
        result = _strip_comments(code)
        assert "hello" in result
        assert "world" in result
        assert "this is" not in result

    def test_string_literal_preserved(self):
        from prover.verifier.integrity_checker import _strip_comments
        code = 'let s := "hello -- not a comment"'
        result = _strip_comments(code)
        assert "not a comment" in result

    def test_mixed_comments(self):
        from prover.verifier.integrity_checker import _strip_comments
        code = "a /- block -/ b -- line\nc"
        result = _strip_comments(code)
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "block" not in result
        assert "line" not in result

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════

class TestAsyncCompileCache:
    """Fix #7: AsyncCompileCache should work in async context."""

    @pytest.mark.asyncio
    async def test_basic_put_get(self):
        from engine._core import AsyncCompileCache, FullVerifyResult
        cache = AsyncCompileCache(maxsize=10)
        result = FullVerifyResult(success=True, env_id=1)
        await cache.put("key1", result)
        got = await cache.get("key1")
        assert got is not None
        assert got.success is True

    @pytest.mark.asyncio
    async def test_miss(self):
        from engine._core import AsyncCompileCache
        cache = AsyncCompileCache(maxsize=10)
        got = await cache.get("missing")
        assert got is None
        assert cache.misses == 1

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        from engine._core import AsyncCompileCache, FullVerifyResult
        cache = AsyncCompileCache(maxsize=3)
        for i in range(5):
            await cache.put(f"k{i}", FullVerifyResult(success=True, env_id=i))
        # First 2 should be evicted
        assert await cache.get("k0") is None
        assert await cache.get("k1") is None
        assert await cache.get("k4") is not None

    @pytest.mark.asyncio
    async def test_stats(self):
        from engine._core import AsyncCompileCache, FullVerifyResult
        cache = AsyncCompileCache()
        await cache.put("a", FullVerifyResult(success=True))
        await cache.get("a")  # hit
        await cache.get("b")  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════

class TestWorldModel:
    """Fix #8: WorldModelPredictor interface and MockWorldModel."""

    def test_mock_sorry(self):
        from engine.world_model import MockWorldModel
        wm = MockWorldModel()
        pred = wm.predict("⊢ True", "sorry")
        assert pred.likely_success is True
        assert pred.confidence > 0.9

    def test_mock_intro_on_forall(self):
        from engine.world_model import MockWorldModel
        wm = MockWorldModel()
        pred = wm.predict("⊢ ∀ n, n + 0 = n", "intro n")
        assert pred.likely_success is True
        assert pred.confidence >= 0.5

    def test_mock_omega_on_nat(self):
        from engine.world_model import MockWorldModel
        wm = MockWorldModel()
        pred = wm.predict("⊢ Nat.add_comm n m", "omega")
        assert pred.likely_success is True

    def test_predict_batch_sorted(self):
        from engine.world_model import MockWorldModel
        wm = MockWorldModel()
        preds = wm.predict_batch(
            "⊢ ∀ n : Nat, n + 0 = n",
            ["sorry", "intro n", "ring", "unknown_tactic"])
        # sorry and intro should be near the top
        assert preds[0].tactic in ("sorry", "intro n")

    def test_filter_tactics(self):
        from engine.world_model import MockWorldModel
        wm = MockWorldModel()
        filtered = wm.filter_tactics(
            "⊢ True", ["trivial", "sorry", "garbage_tactic"])
        # All should pass (conservative filtering)
        assert len(filtered) >= 2

    def test_trained_model_fallback(self):
        from engine.world_model import TrainedWorldModel
        tm = TrainedWorldModel()  # no model, uses fallback
        pred = tm.predict("⊢ True", "trivial")
        assert pred is not None

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════

class TestPassKEarlyStop:
    """Fix #4 (
    expose early_stop / multi_role anymore — legacy non-profile path
    was deleted alongside the v3 multi-role chain.
    """
    pass

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════

class TestUnverifiedMarking:
    """Marking is now handled inside UnifiedProofRunner; this layer of
    test no longer applies after the v9 profile-only consolidation."""
    pass

# ═══════════════════════════════════════════════════════════════
# 整模块只有 ``FEW_SHOT_EXAMPLES`` 常量在主路径用, 已挪到
# common/few_shot.py。``build_prompt`` 只在测试里调过, 测试一并删除。
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════

class TestKnowledgeIntegration:
    """Fix #1: Knowledge reader/writer should be usable in prove_single."""

    def test_knowledge_store_creation(self):
        import tempfile
        from knowledge.store import UnifiedKnowledgeStore
        from knowledge.reader import KnowledgeReader
        from knowledge.writer import KnowledgeWriter

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = UnifiedKnowledgeStore(f.name)
            reader = KnowledgeReader(store)
            writer = KnowledgeWriter(store)
            assert reader is not None
            assert writer is not None

    @pytest.mark.asyncio
    async def test_knowledge_write_read_cycle(self):
        import tempfile
        from knowledge.store import UnifiedKnowledgeStore
        from knowledge.reader import KnowledgeReader
        from knowledge.writer import KnowledgeWriter
        from engine.proof_context_store import StepDetail

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = UnifiedKnowledgeStore(f.name)
            writer = KnowledgeWriter(store)
            reader = KnowledgeReader(store)

            # Write a step
            step = StepDetail(
                step_index=0,
                tactic="simp",
                env_id_before=0,
                env_id_after=1,
                goals_before=["⊢ n + 0 = n"],
                goals_after=[],
                error_message="",
                error_category="",
                elapsed_ms=5,
            )
            await writer.ingest_step(step, theorem="Nat.add_zero")

            # Read back
            text = await reader.render_for_prompt(
                goal="⊢ n + 0 = n", theorem="Nat.add_zero")
            # May be empty if not enough data, but shouldn't crash
            assert isinstance(text, str)



class TestStrictProofAcceptance:
    def test_extract_lean_code_rejects_plain_reasoning_text(self):
        from common.response_parser import extract_lean_code

        text = (
            "Let me think about the proof. We can derive a contradiction "
            "and then finish later."
        )

        assert extract_lean_code(text) == ""

    def test_dialog_to_trace_rejects_successful_natural_language_proof(self):
        from run_eval import _dialog_to_trace_dict

        dialog = {
            "meta": {
                "problem_id": "p1",
                "problem_name": "t",
                "theorem_statement": "theorem t : True",
            },
            "messages": [],
            "result": {
                "success": True,
                "termination": "proof_found",
                "successful_proof": "Let me think about the proof first.",
            },
        }

        trace = _dialog_to_trace_dict(dialog, fallback_problem_id="fallback")

        assert trace["solved"] is False
        assert trace["correct_count"] == 0


    def test_dialog_to_trace_requires_clean_verify_when_lean_verify_present(self):
        import json
        from run_eval import _dialog_to_trace_dict

        dialog = {
            "meta": {
                "problem_id": "p1",
                "problem_name": "target",
                "theorem_statement": "theorem target : True",
            },
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "lean_verify",
                            "arguments": json.dumps({
                                "code": "theorem target : True := by trivial"
                            }),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "lean_verify",
                    "content": json.dumps({
                        "verified": False,
                        "sorry_free": True,
                        "errors": ["transport failed"],
                    }),
                },
            ],
            "result": {
                "success": True,
                "termination": "proof_found",
                "successful_proof": "theorem target : True := by trivial",
            },
        }

        trace = _dialog_to_trace_dict(dialog, fallback_problem_id="fallback")

        assert trace["solved"] is False
        assert trace["correct_count"] == 0

    def test_dialog_to_trace_rejects_verified_wrong_target(self):
        import json
        from run_eval import _dialog_to_trace_dict

        dialog = {
            "meta": {
                "problem_id": "p1",
                "problem_name": "t",
                "theorem_statement": "theorem target : True",
            },
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "lean_verify",
                            "arguments": json.dumps({
                                "code": "example : True := by trivial"
                            }),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "lean_verify",
                    "content": json.dumps({
                        "verified": True,
                        "sorry_free": True,
                        "errors": [],
                    }),
                },
            ],
            "result": {"success": False, "termination": "max_turns"},
        }

        trace = _dialog_to_trace_dict(dialog, fallback_problem_id="fallback")

        assert trace["solved"] is False
        assert trace["correct_count"] == 0

    @pytest.mark.asyncio
    async def test_lean_verify_rejects_exploratory_example_for_target(self):
        import json
        from types import SimpleNamespace
        from agent.tools.base import ToolContext
        from agent.tools.builtin.lean_verify import LeanVerifyTool

        class Pool:
            def __init__(self):
                self.calls = 0

            async def verify_complete(self, theorem, proof, preamble=""):
                self.calls += 1
                return SimpleNamespace(
                    success=True, has_sorry=False, errors=[],
                    goals_remaining=[], stderr="", elapsed_ms=1)

        pool = Pool()
        tool = LeanVerifyTool(lean_pool=pool)
        result = await tool.execute(
            {"code": "example : True := by trivial"},
            ToolContext(theorem_statement="theorem target : True"),
        )
        payload = json.loads(result.content)

        assert payload["verified"] is False
        assert payload["proves_target"] is False
        assert "only for the target theorem" in payload["errors"][0]
        assert pool.calls == 0

    @pytest.mark.asyncio
    async def test_lean_verify_strips_imports_and_extracts_target_proof(self):
        import json
        from types import SimpleNamespace
        from agent.tools.base import ToolContext
        from agent.tools.builtin.lean_verify import LeanVerifyTool

        class Pool:
            def __init__(self):
                self.calls = []

            async def verify_complete(self, theorem, proof, preamble=""):
                self.calls.append((theorem, proof, preamble))
                return SimpleNamespace(
                    success=True, has_sorry=False, errors=[],
                    goals_remaining=[], stderr="", elapsed_ms=1)

        pool = Pool()
        tool = LeanVerifyTool(lean_pool=pool)
        code = (
            "import Mathlib\n"
            "open Nat\n\n"
            "theorem target : True := by\n"
            "  trivial"
        )
        result = await tool.execute(
            {"code": code},
            ToolContext(
                theorem_statement="theorem target : True :=",
                lean_preamble="import Mathlib\nopen Nat",
            ),
        )
        payload = json.loads(result.content)

        assert payload["verified"] is True
        assert payload["proves_target"] is True
        theorem, proof, preamble = pool.calls[-1]
        assert theorem == "theorem target : True :="
        assert proof.strip() == ":= by\n  trivial"
        assert preamble == "import Mathlib\nopen Nat"
        assert "normalization" in payload

    @pytest.mark.asyncio
    async def test_lean_verify_rejects_check_for_target(self):
        import json
        from agent.tools.base import ToolContext
        from agent.tools.builtin.lean_verify import LeanVerifyTool

        class Pool:
            async def verify_complete(self, theorem, proof, preamble=""):
                raise AssertionError("#check should not reach Lean")

        tool = LeanVerifyTool(lean_pool=Pool())
        result = await tool.execute(
            {"code": "#check Nat"},
            ToolContext(theorem_statement="theorem target : True :="),
        )
        payload = json.loads(result.content)

        assert payload["verified"] is False
        assert payload["proves_target"] is False
        assert "#check" in payload["errors"][0]

    @pytest.mark.asyncio
    async def test_goal_inspect_passes_problem_preamble(self):
        import json
        from types import SimpleNamespace
        from agent.tools.base import ToolContext
        from agent.tools.builtin.goal_inspect import GoalInspectTool

        class Pool:
            def __init__(self):
                self.calls = []

            async def verify_complete(self, theorem, proof, preamble=""):
                self.calls.append((theorem, proof, preamble))
                return SimpleNamespace(
                    success=False, has_sorry=True, errors=[],
                    goals_remaining=["⊢ True"])

        pool = Pool()
        tool = GoalInspectTool(lean_pool=pool)
        proof = ":= by\n  sorry"
        preamble = "import Mathlib\nopen scoped BigOperators"
        result = await tool.execute(
            {"proof_so_far": proof},
            ToolContext(
                theorem_statement="theorem target : True :=",
                lean_preamble=preamble,
            ),
        )
        payload = json.loads(result.content)

        assert payload["goal_count"] == 1
        assert pool.calls[-1] == (
            "theorem target : True :=",
            proof,
            preamble,
        )

    @pytest.mark.asyncio
    async def test_goal_inspect_strips_imports_and_extracts_target_proof(self):
        import json
        from types import SimpleNamespace
        from agent.tools.base import ToolContext
        from agent.tools.builtin.goal_inspect import GoalInspectTool

        class Pool:
            def __init__(self):
                self.calls = []

            async def verify_complete(self, theorem, proof, preamble=""):
                self.calls.append((theorem, proof, preamble))
                return SimpleNamespace(
                    success=False, has_sorry=True, errors=[],
                    goals_remaining=["⊢ True"])

        pool = Pool()
        tool = GoalInspectTool(lean_pool=pool)
        code = (
            "import Mathlib\n"
            "open Nat\n\n"
            "theorem target : True := by\n"
            "  sorry"
        )
        result = await tool.execute(
            {"proof_so_far": code},
            ToolContext(
                theorem_statement="theorem target : True :=",
                lean_preamble="import Mathlib\nopen Nat",
            ),
        )
        payload = json.loads(result.content)

        assert payload["goal_count"] == 1
        assert "normalization" in payload
        assert pool.calls[-1] == (
            "theorem target : True :=",
            ":= by\n  sorry",
            "import Mathlib\nopen Nat",
        )

    @pytest.mark.asyncio
    async def test_tactic_suggest_uses_heuristic_without_proof_state(self):
        import json
        from agent.tools.base import ToolContext
        from agent.tools.builtin.tactic_suggest import TacticSuggestTool

        class Pool:
            async def try_tactic(self, env_id, tactic):
                raise AssertionError("should not execute tactics without proof_state_id")

        tool = TacticSuggestTool(lean_pool=Pool())
        result = await tool.execute(
            {"goal_state": "⊢ n + 0 = n"},
            ToolContext(),
        )
        payload = json.loads(result.content)

        assert payload["mode"] == "heuristic"
        assert payload["suggestions"]

    @pytest.mark.asyncio
    async def test_lean_auto_does_not_use_base_env_without_proof_state(self):
        import json
        from agent.tools.base import ToolContext
        from agent.tools.builtin.lean_auto import LeanAutoTool

        class Pool:
            async def try_tactic(self, env_id, tactic):
                raise AssertionError("should not execute tactics without proof_state_id")

        tool = LeanAutoTool(lean_pool=Pool())
        result = await tool.execute(
            {"goal_context": "⊢ True"},
            ToolContext(),
        )
        payload = json.loads(result.content)

        assert payload["mode"] == "unavailable"
        assert payload["closing_tactics"] == []

    @pytest.mark.asyncio
    async def test_tactic_apply_rejects_sorry_before_pool_call(self):
        import json
        from agent.tools.base import ToolContext
        from agent.tools.builtin.tactic_apply import TacticApplyTool

        class Pool:
            async def try_tactic(self, env_id, tactic):
                raise AssertionError("sorry tactic should not reach Lean")

        tool = TacticApplyTool(lean_pool=Pool())
        result = await tool.execute(
            {"tactic": "have h : True := by sorry"},
            ToolContext(current_goals=["⊢ True"]),
        )
        payload = json.loads(result.content)

        assert payload["success"] is False
        assert payload["error_category"] == "integrity_violation"
        assert "sorry" in payload["error_message"]
