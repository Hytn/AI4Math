"""tests/test_fixes.py — Tests for all 12 fixes"""
import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# Fix #6: Nested comment stripping
# ═══════════════════════════════════════════════════════════════

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
# Fix #7: AsyncCompileCache
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
# Fix #8: WorldModelPredictor
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
# Fix #4: pass@k early stop behavior
# ═══════════════════════════════════════════════════════════════

class TestPassKEarlyStop:
    """Fix #4 (v9: removed). prove_single is profile-only and doesn't
    expose early_stop / multi_role anymore — legacy non-profile path
    was deleted alongside the v3 multi-role chain.
    """
    pass


# ═══════════════════════════════════════════════════════════════
# Fix #5: Unverified marking — legacy path deleted in v9.
# ═══════════════════════════════════════════════════════════════

class TestUnverifiedMarking:
    """Marking is now handled inside UnifiedProofRunner; this layer of
    test no longer applies after the v9 profile-only consolidation."""
    pass


# ═══════════════════════════════════════════════════════════════
# Fix #9 (build_prompt): 已在 v13 删除 — common.prompt_builder
# 整模块只有 ``FEW_SHOT_EXAMPLES`` 常量在主路径用, 已挪到
# common/few_shot.py。``build_prompt`` 只在测试里调过, 测试一并删除。
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# Fix #1: Knowledge system integration
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
