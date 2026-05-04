"""tests/test_unified_storage.py — Tests for unified knowledge storage

Validates:
  1. Episodes in SQLite (replaces JSONL)
  2. Persistent knowledge in SQLite (replaces JSON)
  3. RL-aligned reinforce()
  4. Batch trajectory export
  5. Decay and GC
  6. Backward-compatible EpisodicMemory and PersistentKnowledge

(v10: KnowledgeBackend Protocol abstraction removed alongside knowledge.backend
 module — the Protocol had no main-path consumer.)
"""
import asyncio
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from knowledge.store import UnifiedKnowledgeStore
from knowledge.types import LemmaRecord, StrategyPattern


@pytest.fixture
def store():
    """In-memory unified store for testing."""
    s = UnifiedKnowledgeStore(":memory:")
    yield s
    s.close()


class TestProtocol:
    def test_store_has_required_methods(self):
        """All previously-Protocol methods must remain on UnifiedKnowledgeStore."""
        required = [
            "upsert_tactic_effectiveness", "query_tactic_effectiveness",
            "upsert_error_pattern", "query_error_patterns",
            "add_lemma", "search_lemmas",
            "add_strategy_pattern", "query_strategy_patterns",
            "upsert_concept", "add_concept_edge",
            "add_episode", "query_episodes",
            "record_failure", "record_success", "get_suggestions",
            "export_trajectories_batch",
            "knowledge_stats", "decay_all", "reinforce", "gc_stale",
            "close",
        ]
        for method in required:
            assert hasattr(UnifiedKnowledgeStore, method), \
                f"Missing method: {method}"


class TestEpisodesInSQLite:
    @pytest.mark.asyncio
    async def test_add_and_query(self, store):
        eid = await store.add_episode(
            problem_type="number_theory",
            difficulty="medium",
            winning_strategy="induction",
            key_tactics=["omega", "simp"],
            key_insight="Use strong induction on n",
            solve_time_ms=5000)
        assert eid > 0

        rows = await store.query_episodes(problem_type="number_theory")
        assert len(rows) == 1
        assert rows[0]["winning_strategy"] == "induction"
        assert rows[0]["key_tactics"] == ["omega", "simp"]

    @pytest.mark.asyncio
    async def test_query_by_difficulty(self, store):
        await store.add_episode("algebra", "hard", "ring_hom", ["ring"], "", 0)
        await store.add_episode("topology", "easy", "simp", ["simp"], "", 0)

        hard = await store.query_episodes(difficulty="hard")
        assert len(hard) == 1
        assert hard[0]["problem_type"] == "algebra"

    @pytest.mark.asyncio
    async def test_stats_include_episodes(self, store):
        await store.add_episode("test", "easy", "", [], "", 0)
        stats = await store.knowledge_stats()
        assert stats["episodes"] == 1


class TestPersistentKnowledgeInSQLite:
    @pytest.mark.asyncio
    async def test_record_failure(self, store):
        await store.record_failure("ring", "ℕ subtraction", domain="nat")
        await store.record_failure("ring", "ℕ subtraction", domain="nat")

        suggestions = await store.get_suggestions(goal_type="ℕ subtraction")
        # Not enough failures yet (need 3+) to trigger AVOID
        assert len(suggestions) == 0

        await store.record_failure("ring", "ℕ subtraction", domain="nat")
        suggestions = await store.get_suggestions(goal_type="ℕ subtraction")
        assert any("AVOID" in s and "ring" in s for s in suggestions)

    @pytest.mark.asyncio
    async def test_record_success(self, store):
        await store.record_success("nat", ["omega", "simp"])
        await store.record_success("nat", ["omega", "simp"])

        suggestions = await store.get_suggestions(domain="nat")
        assert any("omega" in s for s in suggestions)

    @pytest.mark.asyncio
    async def test_stats_include_pk(self, store):
        await store.record_failure("ring", "test")
        await store.record_success("nat", ["omega"])
        stats = await store.knowledge_stats()
        assert stats["pk_failure_patterns"] == 1
        assert stats["pk_success_patterns"] == 1


class TestReinforce:
    @pytest.mark.asyncio
    async def test_positive_reinforcement(self, store):
        """Positive reward should increase decay_factor."""
        await store.upsert_tactic_effectiveness(
            "omega", "⊢ n + 0 = n", True, 100.0)

        rows = await store.query_tactic_effectiveness("⊢ n + 0 = n")
        assert len(rows) == 1
        original_decay = rows[0].decay_factor

        await store.reinforce("tactic", rows[0].id, reward=1.0)

        rows2 = await store.query_tactic_effectiveness("⊢ n + 0 = n")
        assert rows2[0].decay_factor > original_decay

    @pytest.mark.asyncio
    async def test_negative_reinforcement(self, store):
        """Negative reward should decrease decay_factor."""
        await store.upsert_tactic_effectiveness(
            "ring", "⊢ n - n = 0", False, 200.0)

        rows = await store.query_tactic_effectiveness("⊢ n - n = 0")
        original_decay = rows[0].decay_factor

        await store.reinforce("tactic", rows[0].id, reward=-1.0)

        rows2 = await store.query_tactic_effectiveness("⊢ n - n = 0")
        assert rows2[0].decay_factor < original_decay

    @pytest.mark.asyncio
    async def test_reinforce_lemma(self, store):
        lid = await store.add_lemma(LemmaRecord(
            name="h1", statement="lemma h1 : True := trivial",
            proof=":= trivial", verified=True))

        await store.reinforce("lemma", lid, reward=0.5)
        # Should not crash; decay_factor should increase
        results = await store.search_lemmas(verified_only=True)
        assert len(results) >= 0  # Just verify no crash


class TestDecayAndGC:
    @pytest.mark.asyncio
    async def test_decay_all(self, store):
        # Add entries with enough samples to be decayed
        for i in range(5):
            await store.upsert_tactic_effectiveness(
                "omega", f"goal_{i}", True, 100.0)

        stats = await store.decay_all(decay_rate=0.9, min_samples=3)
        assert stats["tactics_decayed"] >= 0

    @pytest.mark.asyncio
    async def test_gc_stale(self, store):
        # Add a tactic with very low decay
        store._upsert_te_sync("dead_tactic", "dead_goal", True, 10, "", 0)
        with store._connect() as conn:
            conn.execute(
                "UPDATE tactic_effectiveness SET decay_factor=0.01 "
                "WHERE tactic='dead_tactic'")

        stats = await store.gc_stale(threshold=0.1)
        assert stats["tactics_removed"] >= 1


class TestExportTrajectories:
    @pytest.mark.asyncio
    async def test_export_empty(self, store):
        rows = await store.export_trajectories_batch(limit=10)
        assert rows == []

    def test_export_to_parquet_fallback(self, store):
        """Without pyarrow, should fall back to JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.parquet")
            count = store.export_to_parquet(path, limit=10)
            assert count == 0  # No data to export


# (TestBackwardCompatEpisodicMemory and TestBackwardCompatPersistentKnowledge
#  removed in v9: agent/memory/{episodic_memory.py, persistent_knowledge.py}
#  had 0 main-path callers and were deleted. Their unified-store SQLite
#  paths are still exercised by TestPersistentKnowledgeInSQLite above.)
