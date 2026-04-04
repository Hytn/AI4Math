"""tests/test_benchmarks/test_metrics.py — 评测指标模块测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pytest
from benchmarks.metrics import pass_at_k, compute_metrics, MetricsSummary
from knowledge.retriever import KnowledgeRetriever


# ── pass@k ──

class TestPassAtK:
    def test_all_correct(self):
        assert pass_at_k(10, 10, 1) == 1.0
        assert pass_at_k(10, 10, 5) == 1.0

    def test_none_correct(self):
        assert pass_at_k(10, 0, 1) == 0.0
        assert pass_at_k(10, 0, 5) == 0.0

    def test_one_correct_pass1(self):
        result = pass_at_k(10, 1, 1)
        assert 0 < result < 1

    def test_half_correct_pass1(self):
        result = pass_at_k(10, 5, 1)
        assert abs(result - 0.5) < 0.01

    def test_pass_at_k_increases(self):
        r1 = pass_at_k(10, 3, 1)
        r5 = pass_at_k(10, 3, 5)
        r10 = pass_at_k(10, 3, 10)
        assert r1 <= r5 <= r10

    def test_n_less_than_k(self):
        assert pass_at_k(3, 1, 5) == 1.0  # can't sample 5 from 3


# ── compute_metrics ──

class TestComputeMetrics:
    def test_basic_metrics(self):
        traces = [
            {"solved": True, "total_attempts": 3, "total_tokens": 1000},
            {"solved": False, "total_attempts": 5, "total_tokens": 2000},
            {"solved": True, "total_attempts": 1, "total_tokens": 500},
        ]
        m = compute_metrics(traces)
        assert m["total"] == 3
        assert m["solved"] == 2
        assert abs(m["solve_rate"] - 2/3) < 0.01
        assert m["total_tokens"] == 3500
        assert "pass@1" in m

    def test_empty_traces(self):
        m = compute_metrics([])
        assert m["total"] == 0

    def test_all_solved(self):
        traces = [{"solved": True, "total_attempts": 1, "total_tokens": 100}] * 5
        m = compute_metrics(traces)
        assert m["solve_rate"] == 1.0
        assert m["pass@1"] == 1.0

    def test_k_values(self):
        traces = [{"solved": True, "total_attempts": 10, "total_tokens": 100}]
        m = compute_metrics(traces, k_values=[1, 5, 10, 20])
        assert "pass@1" in m
        assert "pass@20" in m

    def test_error_distribution(self):
        traces = [{
            "solved": False, "total_attempts": 1, "total_tokens": 100,
            "attempts": [{"lean_errors": [
                {"category": "type_mismatch"},
                {"category": "type_mismatch"},
                {"category": "tactic_failed"},
            ]}]
        }]
        m = compute_metrics(traces)
        assert m["error_distribution"].get("type_mismatch", 0) == 2

    def test_metrics_summary_table(self):
        m = compute_metrics([
            {"solved": True, "total_attempts": 2, "total_tokens": 500},
        ])
        summary = MetricsSummary("test_bench", m)
        table = summary.to_table()
        assert "test_bench" in table
        assert "1/1" in table


# ── Knowledge Retriever ──

class TestKnowledgeRetriever:
    def test_basic_retrieve(self):
        kr = KnowledgeRetriever()
        results = kr.retrieve("Nat add commutative", top_k=5)
        assert len(results) > 0
        assert isinstance(results[0], str)

    def test_retrieve_full(self):
        kr = KnowledgeRetriever()
        bundle = kr.retrieve_full("theorem t (n m : Nat) : n + m = m + n",
                                   goal_target="n + m = m + n")
        assert "premises" in bundle
        assert "templates" in bundle
        assert "tactics" in bundle
        assert "goal_shape" in bundle
        assert bundle["goal_shape"] == "equality"

    def test_add_custom(self):
        kr = KnowledgeRetriever()
        kr.add_premises([{"name": "custom_zzz", "statement": "lemma zzz : True"}])
        results = kr.retrieve("zzz", top_k=3)
        assert any("zzz" in r for r in results)
