"""tests/test_prover/test_premise.py — 前提检索模块测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pytest
from prover.premise.bm25_retriever import BM25Retriever, tokenize
from prover.premise.embedding_retriever import EmbeddingRetriever
from prover.premise.reranker import PremiseReranker
from prover.premise.selector import PremiseSelector
from prover.premise.tactic_suggester import (
    suggest_tactics, classify_goal, _type_matches)


# ── BM25 Retriever ──

class TestTokenize:
    def test_basic(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_camel_case(self):
        tokens = tokenize("natAddComm")
        assert "nat" in tokens
        assert "add" in tokens
        assert "comm" in tokens

    def test_lean_statement(self):
        tokens = tokenize("theorem Nat.add_comm (n m : Nat) : n + m = m + n")
        assert "nat" in tokens
        assert "add" in tokens
        assert "comm" in tokens
        assert "theorem" in tokens

    def test_single_char_kept(self):
        tokens = tokenize("a + b = c")
        # single-char math variables are now kept for better retrieval
        assert "a" in tokens
        assert "b" in tokens
        assert "c" in tokens


class TestBM25Retriever:
    def setup_method(self):
        self.bm25 = BM25Retriever()
        self.bm25.add_document("Nat.add_comm", "theorem Nat.add_comm (n m : Nat) : n + m = m + n")
        self.bm25.add_document("Nat.mul_comm", "theorem Nat.mul_comm (n m : Nat) : n * m = m * n")
        self.bm25.add_document("And.intro", "theorem And.intro {a b : Prop} (ha : a) (hb : b) : a ∧ b")
        self.bm25.add_document("Or.inl", "theorem Or.inl {a b : Prop} (h : a) : a ∨ b")
        self.bm25.build()

    def test_basic_retrieval(self):
        results = self.bm25.retrieve("Nat add commutative", top_k=3)
        assert len(results) > 0
        assert results[0]["name"] == "Nat.add_comm"

    def test_empty_query(self):
        results = self.bm25.retrieve("")
        assert results == []

    def test_no_match(self):
        results = self.bm25.retrieve("ZZZZnonexistent")
        assert results == []

    def test_top_k_limit(self):
        results = self.bm25.retrieve("theorem", top_k=2)
        assert len(results) <= 2

    def test_scores_ordered(self):
        results = self.bm25.retrieve("Nat commutative", top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_size(self):
        assert self.bm25.size == 4

    def test_add_documents_batch(self):
        bm25 = BM25Retriever()
        bm25.add_documents([
            {"name": "a", "statement": "lemma a : True"},
            {"name": "b", "statement": "lemma b : False → True"},
        ])
        assert bm25.size == 2


# ── Embedding Retriever ──

class TestEmbeddingRetriever:
    def setup_method(self):
        self.embed = EmbeddingRetriever()
        self.embed.add_document("Nat.add_comm", "theorem Nat.add_comm (n m : Nat) : n + m = m + n")
        self.embed.add_document("Nat.mul_comm", "theorem Nat.mul_comm (n m : Nat) : n * m = m * n")
        self.embed.add_document("And.intro", "theorem And.intro {a b : Prop} : a ∧ b")
        self.embed.build()

    def test_basic_retrieval(self):
        results = self.embed.retrieve("Nat add", top_k=3)
        assert len(results) > 0

    def test_scores_positive(self):
        results = self.embed.retrieve("commutative", top_k=3)
        for r in results:
            assert r["score"] >= 0

    def test_empty_index(self):
        empty = EmbeddingRetriever()
        assert empty.retrieve("anything") == []


# ── Reranker ──

class TestPremiseReranker:
    def test_basic_rerank(self):
        reranker = PremiseReranker()
        candidates = [
            {"name": "a", "statement": "lemma a : Nat → Nat", "score": 1.0},
            {"name": "b", "statement": "lemma b : Prop", "score": 0.5},
        ]
        results = reranker.rerank(candidates, "Nat → Nat", goal_type="Nat → Nat")
        assert len(results) == 2
        assert all("rrf_score" in r for r in results)

    def test_empty_candidates(self):
        reranker = PremiseReranker()
        assert reranker.rerank([], "query") == []

    def test_tactic_relevance(self):
        reranker = PremiseReranker()
        candidates = [
            {"name": "eq_lemma", "statement": "lemma eq_lemma : a = b ↔ b = a", "score": 0.5},
            {"name": "other", "statement": "lemma other : True", "score": 0.5},
        ]
        results = reranker.rerank(candidates, "query", tactic_hint="simp")
        assert len(results) == 2


# ── Unified Selector ──

class TestPremiseSelector:
    def test_hybrid_mode(self):
        selector = PremiseSelector({"mode": "hybrid"})
        results = selector.retrieve("Nat add commutative", top_k=5)
        assert len(results) > 0
        assert any("Nat.add_comm" in r.get("name", "") for r in results)

    def test_bm25_mode(self):
        selector = PremiseSelector({"mode": "bm25"})
        results = selector.retrieve("And intro", top_k=3)
        assert len(results) > 0

    def test_none_mode(self):
        selector = PremiseSelector({"mode": "none"})
        results = selector.retrieve("anything")
        assert results == []

    def test_add_custom_premises(self):
        selector = PremiseSelector({"mode": "bm25"})
        selector.add_premises([
            {"name": "custom", "statement": "lemma custom : zzz_unique"}
        ])
        results = selector.retrieve("zzz_unique", top_k=3)
        assert any(r["name"] == "custom" for r in results)


# ── Tactic Suggester ──

class TestTacticSuggester:
    def test_classify_forall(self):
        assert classify_goal("∀ (n : Nat), P n") == "forall"

    def test_classify_implication(self):
        assert classify_goal("P → Q") == "implication"

    def test_classify_conjunction(self):
        assert classify_goal("P ∧ Q") == "conjunction"

    def test_classify_equality(self):
        assert classify_goal("a + b = b + a") == "equality"

    def test_classify_inequality(self):
        assert classify_goal("a ≤ b") == "le"

    def test_suggest_for_implication(self):
        tactics = suggest_tactics("P → Q")
        assert "intro" in tactics

    def test_suggest_for_equality(self):
        tactics = suggest_tactics("n + m = m + n")
        assert any(t in tactics for t in ["rfl", "ring", "omega", "simp"])

    def test_suggest_includes_fallbacks(self):
        tactics = suggest_tactics("SomeRandomGoal")
        assert len(tactics) > 0

    def test_max_suggestions(self):
        tactics = suggest_tactics("P → Q → R", max_suggestions=3)
        assert len(tactics) <= 3

    def test_type_matches(self):
        assert _type_matches("h : P", "P")
        assert not _type_matches("h : P", "Q")
