"""Consolidated knowledge system tests (store, reader, writer, goal_normalizer)


modules. GoalNormalizer is alive (used by store/reader/writer at runtime)
so those tests are kept.
"""

# ============================================================
# Source: test_knowledge_system.py
# ============================================================

import asyncio
import pytest

from engine.proof_context_store import StepDetail, EnvNode, ProofSessionState
from engine.broadcast import BroadcastBus, MessageType

from knowledge.store import UnifiedKnowledgeStore
from knowledge.writer import KnowledgeWriter
from knowledge.reader import KnowledgeReader
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
# Migration
# ═══════════════════════════════════════════════════════════════

# (test_import_lemma_bank deleted in 
#  one-shot helper for the now-deleted prover/lemma_bank/ directory.)
# (test_import_persistent_knowledge / test_import_episodic_memory deleted in 

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

# ─────────────────────────────────────────────────────────────────

# alongside their deleted modules. Lifecycle features they covered
# (decay_tick / gc_stale / revive / changelog) had no main-path callers
# in v9 and were unwired infrastructure.
# ─────────────────────────────────────────────────────────────────
