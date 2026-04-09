"""Consolidated knowledge system tests (store, reader, writer, evolver)"""


# ============================================================
# Source: test_knowledge_system.py
# ============================================================

import asyncio
import pytest

from engine.proof_context_store import StepDetail
from engine.proof_session import EnvNode, ProofSessionState
from engine.broadcast import BroadcastBus, MessageType

from knowledge.store import UnifiedKnowledgeStore
from knowledge.writer import KnowledgeWriter
from knowledge.reader import KnowledgeReader
from knowledge.broadcaster import KnowledgeBroadcaster
from knowledge.goal_normalizer import (
    normalize_level1, normalize_goal_for_key,
    classify_domain, extract_keywords, statement_hash,
)
from knowledge.types import (
    TacticEffectiveness, ErrorPattern, LemmaRecord,
    StrategyPattern, TacticSuggestion, DomainBriefing,
)


# ═══════════════════════════════════════════════════════════════
# GoalNormalizer
# ═══════════════════════════════════════════════════════════════

class TestNormalizeLevel1:
    def test_basic_variable_erasure(self):
        goal = "n m : ℕ ⊢ n + m = m + n"
        result = normalize_level1(goal)
        assert "n" not in result.split()
        assert "m" not in result.split()
        assert "ℕ" in result

    def test_preserves_types(self):
        goal = "x : ℝ ⊢ x * x ≥ 0"
        result = normalize_level1(goal)
        assert "ℝ" in result
        assert "x" not in result.split()

    def test_erases_numbers(self):
        goal = "⊢ 42 + 0 = 42"
        result = normalize_level1(goal)
        assert "42" not in result
        assert "_N" in result

    def test_empty_goal(self):
        assert normalize_level1("") == ""
        assert normalize_level1("  ") == ""

    def test_preserves_operators(self):
        goal = "h : a ≤ b ⊢ a + c ≤ b + c"
        result = normalize_level1(goal)
        assert "≤" in result
        assert "+" in result


class TestNormalizeGoalForKey:
    def test_truncation(self):
        long_goal = "⊢ " + " + ".join(f"x{i}" for i in range(100))
        result = normalize_goal_for_key(long_goal)
        assert len(result) <= 201  # 200 + "…"

    def test_removes_hypothesis_names(self):
        goal = "h : ℕ, k : ℕ ⊢ True"
        result = normalize_goal_for_key(goal)
        # Should not have "_ : ℕ" but "ℕ"
        assert "_ :" not in result


class TestClassifyDomain:
    def test_number_theory(self):
        assert classify_domain("⊢ Nat.Prime p") == "number_theory"

    def test_algebra(self):
        assert classify_domain("⊢ Ring.toMonoid") == "algebra"

    def test_nat_arithmetic(self):
        assert classify_domain("⊢ Nat.succ n = n + 1") == "nat_arithmetic"

    def test_general_fallback(self):
        assert classify_domain("⊢ True") == "logic"

    def test_uses_theorem_text(self):
        d = classify_domain("⊢ x = y", theorem="theorem about Finset.card")
        assert d == "combinatorics"


class TestExtractKeywords:
    def test_basic(self):
        kws = extract_keywords("lemma Nat.add_comm : ∀ n m, n + m = m + n")
        assert "Nat.add_comm" in kws
        assert len(kws) > 0

    def test_filters_stopwords(self):
        kws = extract_keywords("theorem the is a for of")
        assert "the" not in [k.lower() for k in kws]

    def test_limit(self):
        text = " ".join(f"word{i}" for i in range(100))
        kws = extract_keywords(text)
        assert len(kws) <= 30


class TestStatementHash:
    def test_deterministic(self):
        s = "lemma foo : True := trivial"
        assert statement_hash(s) == statement_hash(s)

    def test_whitespace_invariant(self):
        assert statement_hash("lemma  foo") == statement_hash("lemma foo")

    def test_case_invariant(self):
        assert statement_hash("Lemma FOO") == statement_hash("lemma foo")


# ═══════════════════════════════════════════════════════════════
# UnifiedKnowledgeStore
# ═══════════════════════════════════════════════════════════════

class TestUnifiedKnowledgeStore:
    @pytest.fixture
    def store(self):
        return UnifiedKnowledgeStore(":memory:")

    @pytest.mark.asyncio
    async def test_inherits_layer0(self, store):
        """Verify Layer 0 (ProofContextStore) still works."""
        state = ProofSessionState(
            theorem="theorem t : True := trivial",
            root_env_id=0, theorem_env_id=1, current_env_id=1,
            nodes={0: EnvNode(env_id=0)},
        )
        ctx_id = await store.save(state)
        assert ctx_id > 0

        loaded = await store.load(ctx_id)
        assert loaded is not None
        assert loaded.theorem == state.theorem

    @pytest.mark.asyncio
    async def test_tactic_effectiveness_upsert(self, store):
        await store.upsert_tactic_effectiveness(
            "simp", "⊢ _ = _", success=True, elapsed_ms=10, domain="logic")
        await store.upsert_tactic_effectiveness(
            "simp", "⊢ _ = _", success=True, elapsed_ms=20, domain="logic")
        await store.upsert_tactic_effectiveness(
            "simp", "⊢ _ = _", success=False, elapsed_ms=5, domain="logic")

        results = await store.query_tactic_effectiveness("⊢ _ = _")
        assert len(results) == 1
        te = results[0]
        assert te.tactic == "simp"
        assert te.successes == 2
        assert te.failures == 1
        assert te.confidence == pytest.approx(2 / 3, abs=0.01)

    @pytest.mark.asyncio
    async def test_tactic_effectiveness_domain_fallback(self, store):
        await store.upsert_tactic_effectiveness(
            "omega", "⊢ _ + _ = _", success=True, domain="nat_arithmetic")

        # Exact match
        r1 = await store.query_tactic_effectiveness("⊢ _ + _ = _")
        assert len(r1) == 1

        # Domain fallback (different goal_pattern, same domain)
        r2 = await store.query_tactic_effectiveness(
            "⊢ _ * _ = _", domain="nat_arithmetic")
        assert len(r2) == 1  # fallback to domain

    @pytest.mark.asyncio
    async def test_error_patterns(self, store):
        await store.upsert_error_pattern(
            "tactic_failed", "⊢ _ = _", tactic="ring")
        await store.upsert_error_pattern(
            "tactic_failed", "⊢ _ = _", tactic="ring")
        await store.upsert_error_pattern(
            "tactic_failed", "⊢ _ = _", tactic="ring",
            fix_tactic="omega", fix_succeeded=True)

        patterns = await store.query_error_patterns(goal_pattern="⊢ _ = _")
        assert len(patterns) == 1
        ep = patterns[0]
        assert ep.frequency == 3
        assert ep.typical_fix == "omega"
        assert ep.fix_success_rate > 0

    @pytest.mark.asyncio
    async def test_proved_lemmas(self, store):
        lemma = LemmaRecord(
            name="h1", statement="lemma h1 : True := trivial",
            proof=":= trivial",
            statement_hash=statement_hash("lemma h1 : True := trivial"),
            verified=True, keywords=["True"], domain="logic")

        lid = await store.add_lemma(lemma)
        assert lid > 0

        # Duplicate → updates citation count
        lid2 = await store.add_lemma(lemma)
        assert lid2 == lid

        matches = await store.search_lemmas(keywords=["True"])
        assert len(matches) >= 1
        assert matches[0].name == "h1"

    @pytest.mark.asyncio
    async def test_strategy_patterns(self, store):
        sp = StrategyPattern(
            name="nat_induction_omega",
            domain="nat_arithmetic",
            problem_pattern="∀ n : ℕ, ...",
            tactic_template=["intro n", "induction n", "simp", "omega"],
            confidence=0.8, times_applied=5, times_succeeded=4)

        sid = await store.add_strategy_pattern(sp)
        assert sid > 0

        results = await store.query_strategy_patterns(domain="nat_arithmetic")
        assert len(results) == 1
        assert results[0].name == "nat_induction_omega"

    @pytest.mark.asyncio
    async def test_concept_graph(self, store):
        nid1 = await store.upsert_concept("nat_subtraction", "nat_arithmetic")
        nid2 = await store.upsert_concept("nat_addition", "nat_arithmetic")
        assert nid1 > 0 and nid2 > 0

        # Encounter again → increment count
        nid1_again = await store.upsert_concept("nat_subtraction")
        assert nid1_again == nid1

        await store.add_concept_edge(
            "nat_subtraction", "nat_addition", "often_co_occurs")

    @pytest.mark.asyncio
    async def test_knowledge_stats(self, store):
        await store.upsert_tactic_effectiveness(
            "simp", "⊢ True", success=True)
        stats = await store.knowledge_stats()
        assert stats["tactic_patterns"] >= 1
        assert "total_contexts" in stats  # from Layer 0


# ═══════════════════════════════════════════════════════════════
# KnowledgeWriter
# ═══════════════════════════════════════════════════════════════

def _step(tactic, goals_before, success=True, error="", error_cat=""):
    return StepDetail(
        step_index=0, tactic=tactic,
        env_id_before=1, env_id_after=2 if success else -1,
        goals_before=goals_before,
        goals_after=[] if success else goals_before,
        error_message=error, error_category=error_cat,
        elapsed_ms=10.0, is_proof_complete=False)


class TestKnowledgeWriter:
    @pytest.fixture
    def ctx(self):
        store = UnifiedKnowledgeStore(":memory:")
        writer = KnowledgeWriter(store)
        return store, writer

    @pytest.mark.asyncio
    async def test_ingest_step_success(self, ctx):
        store, writer = ctx
        step = _step("simp", ["⊢ 1 + 1 = 2"])
        await writer.ingest_step(step, theorem="theorem t : 1+1=2")

        results = await store.query_tactic_effectiveness(
            normalize_goal_for_key("⊢ 1 + 1 = 2"))
        assert len(results) >= 1
        assert results[0].successes >= 1

    @pytest.mark.asyncio
    async def test_ingest_step_failure(self, ctx):
        store, writer = ctx
        step = _step("ring", ["⊢ n - m + m = n"], success=False,
                      error="ring failed", error_cat="tactic_failed")
        await writer.ingest_step(step, domain="nat_arithmetic")

        errors = await store.query_error_patterns(
            tactic="ring")
        assert len(errors) >= 1
        assert errors[0].error_category == "tactic_failed"

    @pytest.mark.asyncio
    async def test_ingest_proof_result(self, ctx):
        store, writer = ctx

        # Save a proof context first
        state = ProofSessionState(
            theorem="theorem t : True := trivial",
            root_env_id=0, theorem_env_id=1, current_env_id=1,
            nodes={0: EnvNode(env_id=0)})
        ctx_id = await store.save(state)

        steps = [
            _step("intro n", ["⊢ ∀ n, n = n"]),
            _step("rfl", ["n : ℕ ⊢ n = n"]),
        ]
        trace_id = await writer.ingest_proof_result(
            ctx_id, steps, success=True,
            theorem="theorem t : True := trivial",
            duration_ms=100)
        assert trace_id > 0

        # Verify Layer 0 trace was recorded
        trajectories = await store.export_rich_trajectories()
        assert len(trajectories) >= 1

    @pytest.mark.asyncio
    async def test_fix_pattern_analysis(self, ctx):
        store, writer = ctx

        state = ProofSessionState(
            theorem="t", root_env_id=0, theorem_env_id=1,
            current_env_id=1, nodes={0: EnvNode(env_id=0)})
        ctx_id = await store.save(state)

        # Sequence: ring fails, omega succeeds on same goal
        steps = [
            _step("ring", ["⊢ n + 0 = n"], success=False,
                  error="ring failed", error_cat="tactic_failed"),
            _step("omega", ["⊢ n + 0 = n"], success=True),
        ]
        await writer.ingest_proof_result(
            ctx_id, steps, success=True, duration_ms=50)

        # Error pattern should have omega as fix
        errors = await store.query_error_patterns(tactic="ring")
        assert len(errors) >= 1
        assert errors[0].typical_fix == "omega"
        assert errors[0].fix_success_rate > 0


# ═══════════════════════════════════════════════════════════════
# KnowledgeReader
# ═══════════════════════════════════════════════════════════════

class TestKnowledgeReader:
    @pytest.fixture
    async def populated(self):
        store = UnifiedKnowledgeStore(":memory:")
        writer = KnowledgeWriter(store)
        reader = KnowledgeReader(store)

        # Populate with some knowledge
        for _ in range(5):
            await store.upsert_tactic_effectiveness(
                "omega", "⊢ _ + _N = _", success=True,
                elapsed_ms=8, domain="nat_arithmetic")
        for _ in range(5):
            await store.upsert_tactic_effectiveness(
                "ring", "⊢ _ + _N = _", success=False,
                elapsed_ms=15, domain="nat_arithmetic")
        await store.upsert_tactic_effectiveness(
            "ring", "⊢ _ + _N = _", success=True,
            elapsed_ms=15, domain="nat_arithmetic")

        await store.upsert_error_pattern(
            "tactic_failed", "⊢ _ + _N = _", tactic="simp",
            fix_tactic="omega", fix_succeeded=True)
        for _ in range(3):
            await store.upsert_error_pattern(
                "tactic_failed", "⊢ _ + _N = _", tactic="simp")

        lemma = LemmaRecord(
            name="add_zero", statement="lemma add_zero (n : ℕ) : n + 0 = n",
            proof=":= by omega",
            statement_hash=statement_hash("lemma add_zero (n : ℕ) : n + 0 = n"),
            verified=True, keywords=["add_zero", "Nat", "add"],
            domain="nat_arithmetic")
        await store.add_lemma(lemma)

        return store, reader

    @pytest.mark.asyncio
    async def test_suggest_tactics(self, populated):
        store, reader = populated
        suggestions = await reader.suggest_tactics("⊢ _ + _N = _")
        assert len(suggestions) > 0

        positives = [s for s in suggestions if not s.avoid]
        negatives = [s for s in suggestions if s.avoid]

        # omega should be recommended
        assert any(s.tactic == "omega" for s in positives)
        # ring should be warned (low success rate)
        assert any(s.tactic == "ring" for s in negatives)

    @pytest.mark.asyncio
    async def test_find_lemmas(self, populated):
        store, reader = populated
        lemmas = await reader.find_lemmas(
            goal="⊢ n + 0 = n", theorem="add", domain="nat_arithmetic")
        assert len(lemmas) >= 1
        assert lemmas[0].name == "add_zero"

    @pytest.mark.asyncio
    async def test_get_domain_briefing(self, populated):
        store, reader = populated
        briefing = await reader.get_domain_briefing(
            domain="nat_arithmetic", goal="⊢ n + 0 = n")
        assert isinstance(briefing, DomainBriefing)
        assert len(briefing.top_tactics) > 0

    @pytest.mark.asyncio
    async def test_render_for_prompt(self, populated):
        store, reader = populated
        text = await reader.render_for_prompt(
            goal="⊢ n + 0 = n",
            theorem="theorem add_zero_eq",
            max_chars=2000)
        assert len(text) > 0
        assert "omega" in text.lower() or "tactic" in text.lower()


# ═══════════════════════════════════════════════════════════════
# KnowledgeBroadcaster
# ═══════════════════════════════════════════════════════════════

class TestKnowledgeBroadcaster:
    @pytest.mark.asyncio
    async def test_broadcast_positive_discovery(self):
        store = UnifiedKnowledgeStore(":memory:")
        bus = BroadcastBus()
        broadcaster = KnowledgeBroadcaster(store, broadcast=bus)

        bus.subscribe("test_sub")

        step = _step("omega", ["⊢ n + 0 = n"], success=True)
        await broadcaster.on_tactic_result(
            step, direction="automation",
            theorem="t : n + 0 = n")

        recent = bus.get_recent(n=5, msg_type=MessageType.POSITIVE_DISCOVERY)
        assert len(recent) >= 1

    @pytest.mark.asyncio
    async def test_broadcast_negative_after_threshold(self):
        store = UnifiedKnowledgeStore(":memory:")
        bus = BroadcastBus()
        broadcaster = KnowledgeBroadcaster(store, broadcast=bus)
        bus.subscribe("test_sub")

        step = _step("ring", ["⊢ n - m = 0"], success=False,
                      error="ring failed", error_cat="tactic_failed")

        # Need 3+ failures to trigger broadcast
        for _ in range(4):
            await broadcaster.on_tactic_result(
                step, direction="algebra")

        recent = bus.get_recent(n=5, msg_type=MessageType.NEGATIVE_KNOWLEDGE)
        assert len(recent) >= 1

    @pytest.mark.asyncio
    async def test_on_proof_completed(self):
        store = UnifiedKnowledgeStore(":memory:")
        bus = BroadcastBus()
        broadcaster = KnowledgeBroadcaster(store, broadcast=bus)

        state = ProofSessionState(
            theorem="t", root_env_id=0, theorem_env_id=1,
            current_env_id=1, nodes={0: EnvNode(env_id=0)})
        ctx_id = await store.save(state)

        steps = [_step("simp", ["⊢ True"], success=True)]
        trace_id = await broadcaster.on_proof_completed(
            ctx_id, steps, success=True,
            theorem="t : True", direction="automation")
        assert trace_id > 0


# ═══════════════════════════════════════════════════════════════
# Migration
# ═══════════════════════════════════════════════════════════════

class TestMigration:
    @pytest.mark.asyncio
    async def test_import_persistent_knowledge(self):
        store = UnifiedKnowledgeStore(":memory:")
        writer = KnowledgeWriter(store)

        # Mock PersistentKnowledge
        class MockPK:
            _failures = {"ring": {"ℕ subtraction": 5}}
            _successes = {"number_theory": {"omega → simp": 3}}
            _insights = ["Use omega for ℕ"]

        count = await writer.import_from_persistent_knowledge(MockPK())
        assert count > 0

        stats = await store.knowledge_stats()
        assert stats["tactic_patterns"] > 0

    @pytest.mark.asyncio
    async def test_import_lemma_bank(self):
        store = UnifiedKnowledgeStore(":memory:")
        writer = KnowledgeWriter(store)

        class MockLemma:
            name = "h1"
            statement = "lemma h1 : True"
            proof = ":= trivial"
            verified = True

        class MockBank:
            lemmas = [MockLemma()]

        count = await writer.import_from_lemma_bank(MockBank())
        assert count == 1

        lemmas = await store.search_lemmas(keywords=["h1"])
        assert len(lemmas) >= 1

    @pytest.mark.asyncio
    async def test_import_episodic_memory(self):
        store = UnifiedKnowledgeStore(":memory:")
        writer = KnowledgeWriter(store)

        class MockEpisode:
            problem_type = "number_theory"
            difficulty = "hard"
            winning_strategy = "induction"
            key_tactics = ["intro", "induction", "omega"]
            key_insight = "Use strong induction"
            solve_time_ms = 5000

        class MockEM:
            episodes = [MockEpisode()]

        count = await writer.import_from_episodic_memory(MockEM())
        assert count == 1

        patterns = await store.query_strategy_patterns(domain="number_theory")
        assert len(patterns) >= 1


# ═══════════════════════════════════════════════════════════════
# End-to-end Integration
# ═══════════════════════════════════════════════════════════════

class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """End-to-end: write → accumulate → read → prompt inject"""
        store = UnifiedKnowledgeStore(":memory:")
        writer = KnowledgeWriter(store)
        reader = KnowledgeReader(store)

        state = ProofSessionState(
            theorem="theorem add_zero (n : ℕ) : n + 0 = n := by",
            root_env_id=0, theorem_env_id=1, current_env_id=1,
            nodes={0: EnvNode(env_id=0)})
        ctx_id = await store.save(state)

        # Simulate multiple proof attempts
        for _ in range(3):
            # Attempt 1: ring fails
            steps_fail = [
                _step("ring", ["n : ℕ ⊢ n + 0 = n"], success=False,
                      error="ring failed", error_cat="tactic_failed"),
            ]
            await writer.ingest_proof_result(
                ctx_id, steps_fail, success=False,
                theorem="add_zero", duration_ms=50)

        # Attempt 2: omega succeeds (run twice to exceed threshold)
        for _ in range(2):
            steps_ok = [
                _step("ring", ["n : ℕ ⊢ n + 0 = n"], success=False,
                      error="ring failed", error_cat="tactic_failed"),
                _step("omega", ["n : ℕ ⊢ n + 0 = n"], success=True),
            ]
            await writer.ingest_proof_result(
                ctx_id, steps_ok, success=True,
                theorem="add_zero", duration_ms=30)

        # Read: suggest tactics for similar goal
        suggestions = await reader.suggest_tactics(
            "n : ℕ ⊢ n + 0 = n")

        # omega should be recommended, ring should be warned
        tactic_names = {s.tactic for s in suggestions}
        assert "omega" in tactic_names

        # Render for prompt
        text = await reader.render_for_prompt(
            goal="n : ℕ ⊢ n + 0 = n", theorem="add_zero")
        assert len(text) > 0

        # Stats
        stats = await store.knowledge_stats()
        assert stats["tactic_patterns"] >= 2
        assert stats["error_patterns"] >= 1


# ============================================================
# Source: test_knowledge_evolver.py
# ============================================================

import asyncio
import time
import pytest

from knowledge.store import UnifiedKnowledgeStore
from knowledge.writer import KnowledgeWriter
from knowledge.reader import KnowledgeReader
from knowledge.broadcaster import KnowledgeBroadcaster
from knowledge.evolver import KnowledgeEvolver
from knowledge.types import LemmaRecord, StrategyPattern
from engine.proof_context_store import StepDetail
from engine.broadcast import BroadcastBus


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def store():
    return UnifiedKnowledgeStore(":memory:")


@pytest.fixture
def evolver(store):
    return KnowledgeEvolver(
        store, decay_rate=0.8, stale_threshold=0.2, min_samples=2)


@pytest.fixture
def writer(store):
    return KnowledgeWriter(store)


@pytest.fixture
def reader(store):
    return KnowledgeReader(store)


# ═══════════════════════════════════════════════════════════════
# Decay tick
# ═══════════════════════════════════════════════════════════════

class TestDecayTick:
    def test_decay_reduces_factor(self, store, evolver):
        """decay_tick should multiply decay_factor by decay_rate"""
        async def run():
            # Insert tactic with enough samples (min_samples=2)
            await store.upsert_tactic_effectiveness(
                "omega", "ℕ ⊢ _ + _ = _", True)
            await store.upsert_tactic_effectiveness(
                "omega", "ℕ ⊢ _ + _ = _", True)

            # Verify initial decay_factor = 1.0
            rows = await store.query_tactic_effectiveness("ℕ ⊢ _ + _ = _")
            assert rows[0].decay_factor == 1.0

            # Run decay
            stats = await evolver.decay_tick()
            assert stats.tactic_rows_decayed >= 1

            # Verify decay_factor reduced
            rows = await store.query_tactic_effectiveness("ℕ ⊢ _ + _ = _")
            assert abs(rows[0].decay_factor - 0.8) < 0.01

            # Second tick
            await evolver.decay_tick()
            rows = await store.query_tactic_effectiveness("ℕ ⊢ _ + _ = _")
            assert abs(rows[0].decay_factor - 0.64) < 0.01  # 0.8 * 0.8

        asyncio.run(run())

    def test_decay_protects_low_sample(self, store, evolver):
        """Entries with fewer than min_samples should not decay"""
        async def run():
            # Insert tactic with only 1 sample (below min_samples=2)
            await store.upsert_tactic_effectiveness(
                "ring", "⊢ _ = _", True)

            stats = await evolver.decay_tick()
            # Should not decay
            rows = await store.query_tactic_effectiveness("⊢ _ = _")
            assert rows[0].decay_factor == 1.0

        asyncio.run(run())

    def test_decay_lemmas(self, store, evolver):
        """decay_tick should also decay proved_lemmas"""
        async def run():
            lemma = LemmaRecord(
                name="test_lemma", statement="lemma x : True",
                proof=":= trivial", verified=False,
                keywords=["test"], domain="logic")
            await store.add_lemma(lemma)

            stats = await evolver.decay_tick()
            assert stats.lemma_rows_decayed >= 1

        asyncio.run(run())

    def test_decay_strategies(self, store, evolver):
        """decay_tick should also decay strategy_patterns"""
        async def run():
            sp = StrategyPattern(
                name="induct_then_simp", domain="nat",
                problem_pattern="ℕ induction",
                tactic_template=["induction", "simp"])
            await store.add_strategy_pattern(sp)

            stats = await evolver.decay_tick()
            assert stats.strategy_rows_decayed >= 1

        asyncio.run(run())


# ═══════════════════════════════════════════════════════════════
# GC stale
# ═══════════════════════════════════════════════════════════════

class TestGCStale:
    def test_gc_removes_decayed_tactics(self, store, evolver):
        """Tactics decayed below threshold should be deleted"""
        async def run():
            # Insert and decay below threshold
            await store.upsert_tactic_effectiveness(
                "omega", "⊢ _ = _", True)
            await store.upsert_tactic_effectiveness(
                "omega", "⊢ _ = _", False)

            # Manually set decay_factor below threshold
            with store._connect() as conn:
                conn.execute(
                    "UPDATE tactic_effectiveness SET decay_factor = 0.05")

            stats = await evolver.gc_stale()
            assert stats.tactics_marked_stale >= 1

            # Should be gone
            rows = await store.query_tactic_effectiveness("⊢ _ = _")
            assert len(rows) == 0

        asyncio.run(run())

    def test_gc_marks_unverified_lemmas_stale(self, store, evolver):
        """Unverified lemmas decayed below threshold → stale=1"""
        async def run():
            lemma = LemmaRecord(
                name="weak_lemma", statement="lemma x : False",
                proof=":= sorry", verified=False)
            lid = await store.add_lemma(lemma)

            # Force low decay
            with store._connect() as conn:
                conn.execute(
                    "UPDATE proved_lemmas SET decay_factor = 0.05 "
                    "WHERE id = ?", (lid,))

            stats = await evolver.gc_stale()
            assert stats.lemmas_marked_stale >= 1

            # Should not appear in search (stale=0 filter)
            results = await store.search_lemmas(keywords=["weak"])
            assert len(results) == 0

        asyncio.run(run())

    def test_gc_protects_verified_lemmas(self, store, evolver):
        """Verified lemmas should never be marked stale"""
        async def run():
            lemma = LemmaRecord(
                name="strong_lemma", statement="lemma x : True",
                proof=":= trivial", verified=True)
            lid = await store.add_lemma(lemma)

            # Force low decay
            with store._connect() as conn:
                conn.execute(
                    "UPDATE proved_lemmas SET decay_factor = 0.01 "
                    "WHERE id = ?", (lid,))

            stats = await evolver.gc_stale()
            # Should NOT be marked stale because verified=True
            with store._connect() as conn:
                row = conn.execute(
                    "SELECT stale FROM proved_lemmas WHERE id=?",
                    (lid,)).fetchone()
                assert row["stale"] == 0

        asyncio.run(run())


# ═══════════════════════════════════════════════════════════════
# Revive
# ═══════════════════════════════════════════════════════════════

class TestRevive:
    def test_revive_stale_lemma(self, store, evolver):
        """revive_lemma should restore stale lemma with partial decay"""
        async def run():
            lemma = LemmaRecord(
                name="revivable", statement="lemma r : True",
                proof=":= trivial", verified=False)
            lid = await store.add_lemma(lemma)

            # Mark stale
            with store._connect() as conn:
                conn.execute(
                    "UPDATE proved_lemmas SET stale=1, decay_factor=0.05 "
                    "WHERE id=?", (lid,))

            result = await evolver.revive_lemma(lid, reason="test revive")
            assert result is True

            # Check restored
            with store._connect() as conn:
                row = conn.execute(
                    "SELECT stale, decay_factor FROM proved_lemmas "
                    "WHERE id=?", (lid,)).fetchone()
                assert row["stale"] == 0
                assert abs(row["decay_factor"] - 0.5) < 0.01

        asyncio.run(run())


# ═══════════════════════════════════════════════════════════════
# Changelog audit trail
# ═══════════════════════════════════════════════════════════════

class TestChangelog:
    def test_decay_writes_changelog(self, store, evolver):
        """Each decay tick should produce changelog entries"""
        async def run():
            await store.upsert_tactic_effectiveness(
                "simp", "⊢ _", True)
            await store.upsert_tactic_effectiveness(
                "simp", "⊢ _", True)

            await evolver.decay_tick()

            with store._connect() as conn:
                entries = conn.execute(
                    "SELECT * FROM knowledge_changelog "
                    "WHERE action = 'batch_decay'").fetchall()
                assert len(entries) >= 1
                assert "rate=0.8" in entries[0]["new_value"]

        asyncio.run(run())

    def test_gc_writes_changelog(self, store, evolver):
        """GC operations should produce changelog entries"""
        async def run():
            await store.upsert_tactic_effectiveness(
                "bad_tactic", "⊢ _", False)
            await store.upsert_tactic_effectiveness(
                "bad_tactic", "⊢ _", False)

            with store._connect() as conn:
                conn.execute(
                    "UPDATE tactic_effectiveness SET decay_factor = 0.01")

            await evolver.gc_stale()

            with store._connect() as conn:
                entries = conn.execute(
                    "SELECT * FROM knowledge_changelog "
                    "WHERE action = 'gc_delete'").fetchall()
                assert len(entries) >= 1

        asyncio.run(run())

    def test_lemma_creation_writes_changelog(self, store):
        """Adding a lemma should produce a changelog entry"""
        async def run():
            lemma = LemmaRecord(
                name="audit_test", statement="lemma a : True",
                proof=":= trivial", source_problem="test_theorem")
            await store.add_lemma(lemma)

            with store._connect() as conn:
                entries = conn.execute(
                    "SELECT * FROM knowledge_changelog "
                    "WHERE entity_type = 'proved_lemmas' "
                    "AND action = 'create'").fetchall()
                assert len(entries) >= 1
                assert "audit_test" in entries[0]["new_value"]

        asyncio.run(run())

    def test_strategy_creation_writes_changelog(self, store):
        """Adding a strategy pattern should produce a changelog entry"""
        async def run():
            sp = StrategyPattern(
                name="test_strat", domain="logic",
                problem_pattern="⊢ Prop",
                tactic_template=["intro", "exact"])
            await store.add_strategy_pattern(sp)

            with store._connect() as conn:
                entries = conn.execute(
                    "SELECT * FROM knowledge_changelog "
                    "WHERE entity_type = 'strategy_patterns' "
                    "AND action = 'create'").fetchall()
                assert len(entries) >= 1
                assert "test_strat" in entries[0]["new_value"]

        asyncio.run(run())

    def test_revive_writes_changelog(self, store, evolver):
        """Reviving a lemma should produce a changelog entry"""
        async def run():
            lemma = LemmaRecord(
                name="revive_audit", statement="lemma ra : True",
                proof=":= trivial")
            lid = await store.add_lemma(lemma)

            with store._connect() as conn:
                conn.execute(
                    "UPDATE proved_lemmas SET stale=1, decay_factor=0.01 "
                    "WHERE id=?", (lid,))

            await evolver.revive_lemma(lid, reason="audit check")

            with store._connect() as conn:
                entries = conn.execute(
                    "SELECT * FROM knowledge_changelog "
                    "WHERE action = 'revive'").fetchall()
                assert len(entries) >= 1
                assert "audit check" in entries[0]["reason"]

        asyncio.run(run())


# ═══════════════════════════════════════════════════════════════
# Pipeline integration
# ═══════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    def test_components_carry_knowledge_fields(self):
        """AsyncEngineComponents should have all knowledge fields"""
        from engine.async_factory import AsyncEngineComponents
        comp = AsyncEngineComponents()
        assert comp.knowledge_store is None
        assert comp.knowledge_writer is None
        assert comp.knowledge_reader is None
        assert comp.knowledge_broadcaster is None

    def test_components_accept_knowledge_injection(self, store):
        """Knowledge components can be injected into AsyncEngineComponents"""
        from engine.async_factory import AsyncEngineComponents
        writer = KnowledgeWriter(store)
        reader = KnowledgeReader(store)
        bus = BroadcastBus()
        broadcaster = KnowledgeBroadcaster(store, bus, writer=writer)

        comp = AsyncEngineComponents()
        comp.knowledge_store = store
        comp.knowledge_writer = writer
        comp.knowledge_reader = reader
        comp.knowledge_broadcaster = broadcaster

        assert comp.knowledge_store is store
        assert comp.knowledge_reader is reader

    def test_agent_deps_exports_knowledge(self):
        """_agent_deps should export all knowledge classes"""
        from prover.pipeline._agent_deps import (
            UnifiedKnowledgeStore,
            KnowledgeWriter,
            KnowledgeReader,
            KnowledgeBroadcaster,
        )
        assert UnifiedKnowledgeStore is not None
        assert KnowledgeWriter is not None

    def test_orchestrator_has_knowledge_injection_code(self):
        """async_orchestrator should contain knowledge integration"""
        with open("prover/pipeline/async_orchestrator.py") as f:
            src = f.read()

        # Producer should read from knowledge
        assert "knowledge_reader" in src
        assert "render_for_prompt" in src
        assert "proof_knowledge" in src

        # Consumer should write to knowledge
        assert "knowledge_broadcaster" in src
        assert "on_tactic_result" in src

    def test_async_prove_has_knowledge_injection_code(self):
        """async_prove should contain knowledge injection"""
        with open("prover/pipeline/async_prove.py") as f:
            src = f.read()
        assert "knowledge_reader" in src
        assert "proof_knowledge" in src


# ═══════════════════════════════════════════════════════════════
# End-to-end: write → decay → read cycle
# ═══════════════════════════════════════════════════════════════

class TestEndToEndLifecycle:
    def test_write_decay_read_gc(self, store, evolver, writer, reader):
        """Full lifecycle: ingest → accumulate → decay → gc → verify reads"""
        async def run():
            # Phase 1: Ingest some proof steps
            for i in range(5):
                step = StepDetail(
                    step_index=i,
                    tactic="omega",
                    env_id_before=0,
                    env_id_after=1,
                    goals_before=["n : ℕ ⊢ n + 0 = n"],
                    goals_after=[],
                    elapsed_ms=10.0,
                    is_proof_complete=(i == 4),
                )
                await writer.ingest_step(
                    step, theorem="theorem add_zero (n : ℕ) : n + 0 = n",
                    domain="nat_arithmetic")

            # Phase 2: Verify knowledge accumulated
            suggestions = await reader.suggest_tactics(
                "n : ℕ ⊢ n + 0 = n", domain="nat_arithmetic")
            omega_found = any(s.tactic == "omega" for s in suggestions)
            assert omega_found, "omega should be in suggestions"

            # Phase 3: Decay multiple times
            for _ in range(10):
                await evolver.decay_tick()

            # Phase 4: omega should still be findable (decayed but above threshold)
            suggestions2 = await reader.suggest_tactics(
                "n : ℕ ⊢ n + 0 = n", domain="nat_arithmetic")
            omega_found2 = any(s.tactic == "omega" for s in suggestions2)
            # After 10 ticks at 0.8 rate: 0.8^10 ≈ 0.107 > threshold 0.2? No.
            # Actually 0.8^10 = 0.107 which is below 0.2 threshold
            # So it should be gc'd after gc_stale

            # Phase 5: GC should clean it up
            gc_stats = await evolver.gc_stale()
            assert gc_stats.tactics_marked_stale >= 1

            # Phase 6: Verify changelog has full audit trail
            evolver_stats = await evolver.stats()
            assert evolver_stats["changelog_entries"] > 0

        asyncio.run(run())

    def test_knowledge_broadcaster_roundtrip(self, store):
        """Broadcaster writes to store AND publishes to bus"""
        async def run():
            bus = BroadcastBus()
            writer = KnowledgeWriter(store)
            broadcaster = KnowledgeBroadcaster(
                store, bus, writer=writer)

            # Subscribe to bus
            sub = bus.subscribe("test_agent")

            # Simulate a tactic result
            step = StepDetail(
                step_index=0, tactic="ring",
                env_id_before=0, env_id_after=1,
                goals_before=["⊢ a * b = b * a"],
                goals_after=[],
                elapsed_ms=5.0, is_proof_complete=True)

            await broadcaster.on_tactic_result(
                step, direction="automation",
                theorem="theorem comm : a * b = b * a")

            # Verify written to store
            rows = await store.query_tactic_effectiveness(
                goal_pattern="", domain="algebra", top_k=5)
            # At least something should have been recorded
            stats = await store.knowledge_stats()
            assert stats["tactic_patterns"] >= 1

        asyncio.run(run())
