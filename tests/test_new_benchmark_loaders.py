"""tests/test_new_benchmark_loaders.py — leaneval / matharena loader 测试

全部基于 tmp_path fixture 构造最小仓库结构, 不依赖网络与真实数据。
"""
from __future__ import annotations

import json
import sys
import textwrap

import pytest


# ─── lean-eval ──────────────────────────────────────────────────────

def _make_leaneval_repo(tmp_path):
    repo = tmp_path / "LeanEval"
    (repo / "manifests").mkdir(parents=True)
    (repo / "LeanEval").mkdir()
    (repo / "manifests" / "problems.toml").write_text(textwrap.dedent("""\
        [[problems]]
        id = "p001"
        declaration = "leaneval_p001"
        difficulty = "easy"
        description = "one plus one"

        [[problems]]
        id = "p002"
        declaration = "leaneval_p002"
    """), encoding="utf-8")
    (repo / "LeanEval" / "Basic.lean").write_text(textwrap.dedent("""\
        import Mathlib

        @[eval_problem]
        theorem leaneval_p001 : 1 + 1 = 2 := by sorry

        @[eval_problem (priority := high)]
        theorem leaneval_p002 (n : ℕ) :
            n + 0 = n := by sorry
    """), encoding="utf-8")
    return repo


class TestLeanEvalLoader:
    def test_loads_manifest_and_statements(self, tmp_path):
        from benchmarks.datasets.leaneval.loader import load
        repo = _make_leaneval_repo(tmp_path)
        ps = load(str(repo))
        assert len(ps) == 2
        by_name = {p.name: p for p in ps}
        assert "1 + 1 = 2" in by_name["p001"].theorem_statement
        assert by_name["p001"].theorem_statement.rstrip().endswith(
            ":= by sorry")
        # 多行声明也能抽到
        assert "n + 0 = n" in by_name["p002"].theorem_statement
        # comparator 范式标记必须在 tags 里
        assert "comparator_scored" in by_name["p001"].tags
        assert by_name["p001"].difficulty == "easy"

    def test_missing_repo_returns_empty(self, tmp_path):
        from benchmarks.datasets.leaneval.loader import load
        assert load(str(tmp_path / "nope")) == []

    def test_limit(self, tmp_path):
        from benchmarks.datasets.leaneval.loader import load
        repo = _make_leaneval_repo(tmp_path)
        assert len(load(str(repo), limit=1)) == 1

    def test_workspace_fallback_when_no_attr(self, tmp_path):
        """manifest 有题但 LeanEval/ 没声明时, 从 generated/<id>/ 兜底。"""
        from benchmarks.datasets.leaneval.loader import load
        repo = _make_leaneval_repo(tmp_path)
        (repo / "manifests" / "problems.toml").write_text(textwrap.dedent("""\
            [[problems]]
            id = "p003"
            declaration = "leaneval_p003"
        """), encoding="utf-8")
        ws = repo / "generated" / "p003"
        ws.mkdir(parents=True)
        (ws / "Solution.lean").write_text(
            "theorem leaneval_p003 : True := by sorry", encoding="utf-8")
        ps = load(str(repo))
        assert len(ps) == 1 and "True" in ps[0].theorem_statement


# ─── MathArena (formalized) ─────────────────────────────────────────

def _make_matharena_root(tmp_path):
    comp = tmp_path / "MathArena" / "aime_2026"
    comp.mkdir(parents=True)
    rows = [
        {"problem_idx": 1, "problem": "Compute 1+1.", "answer": "2",
         "statement": "theorem matharena_q1 : 1 + 1 = 2 := by sorry",
         "flagged": False, "formalizer_model": "test-model"},
        {"problem_idx": 2, "problem": "Bad one.", "answer": "0",
         "statement": "theorem matharena_q2 : False := by sorry",
         "flagged": True, "formalizer_model": "test-model"},
    ]
    with open(comp / "formalized.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return tmp_path / "MathArena"


class TestMathArenaLoader:
    def test_loads_and_skips_flagged(self, tmp_path):
        from benchmarks.datasets.matharena.loader import load
        root = _make_matharena_root(tmp_path)
        ps = load(str(root), split="aime_2026")
        assert len(ps) == 1
        p = ps[0]
        assert p.problem_id == "matharena_aime_2026_1"
        assert "1 + 1 = 2" in p.theorem_statement
        # formalizer 溯源与"未人工核对"标记必须保留到评测产物
        assert "formalizer:test-model" in p.tags
        assert "autoformalized_unverified" in p.tags
        assert p.natural_language == "Compute 1+1."

    def test_split_all_scans_every_comp(self, tmp_path):
        from benchmarks.datasets.matharena.loader import load
        root = _make_matharena_root(tmp_path)
        ps = load(str(root), split="test")
        assert len(ps) == 1

    def test_missing_root_returns_empty(self, tmp_path):
        from benchmarks.datasets.matharena.loader import load
        assert load(str(tmp_path / "nope")) == []


# ─── run_leaneval.py 的提交提取 ────────────────────────────────────

class TestRunLeanEvalHelpers:
    def test_proof_from_dialog(self, tmp_path):
        sys.path.insert(0, "scripts/eval")
        try:
            from run_leaneval import proof_from_dialog
        finally:
            sys.path.pop(0)
        d = tmp_path / "dialog.json"
        d.write_text(json.dumps({
            "result": {"successful_proof": ""},
            "messages": [
                {"role": "assistant",
                 "content": "Try this:\n```lean\ntheorem t : True := by trivial\n```"},
            ],
        }), encoding="utf-8")
        proof = proof_from_dialog(d)
        assert "trivial" in proof

    def test_proof_prefers_successful_proof(self, tmp_path):
        sys.path.insert(0, "scripts/eval")
        try:
            from run_leaneval import proof_from_dialog
        finally:
            sys.path.pop(0)
        d = tmp_path / "dialog.json"
        d.write_text(json.dumps({
            "result": {"successful_proof": "theorem good : True := by trivial"},
            "messages": [{"role": "assistant", "content": "```lean\nbad\n```"}],
        }), encoding="utf-8")
        assert "good" in proof_from_dialog(d)
