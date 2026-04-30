"""tests/test_backends/test_numinamath_loader.py — NuminaMath-LEAN loader tests."""
import json
import pytest
from pathlib import Path

from benchmarks.datasets.numinamath_lean.loader import (
    load, hf_download_hint, _record_to_problem, _difficulty_from_record,
)


def test_hf_download_hint_format():
    h = hf_download_hint()
    assert "huggingface-cli" in h
    assert "AI-MO/NuminaMath-LEAN" in h


def test_difficulty_from_record_olympiad():
    assert _difficulty_from_record({"source": "imo_2020"}) == "competition"
    assert _difficulty_from_record({"source": "olympiad-foo"}) == "competition"
    assert _difficulty_from_record({"source": "amc_aime"}) == "hard"
    assert _difficulty_from_record({"source": "amc_2023"}) == "medium"
    assert _difficulty_from_record({"source": "school"}) == "easy"
    assert _difficulty_from_record(
        {"source": "x", "problem_type": "Number Theory"}) == "medium"


def test_record_to_problem_returns_none_without_formal_statement():
    rec = {"problem": "Find n.", "formal_statement": ""}
    assert _record_to_problem(rec, "test", 0) is None


def test_record_to_problem_normal_record():
    rec = {
        "problem": "Find the smallest n such that n^2 > 100.",
        "answer": "11",
        "source": "amc_2023",
        "problem_type": "Algebra",
        "formal_statement": "theorem p : ∃ n : ℕ, n^2 > 100",
        "formal_proof": "sorry",
    }
    bp = _record_to_problem(rec, "test", 5)
    assert bp is not None
    assert bp.problem_id == "numinamath_test_000005"
    assert bp.theorem_statement.startswith("theorem p")
    assert bp.source == "NuminaMath-LEAN"


def test_load_missing_directory_returns_empty(tmp_path):
    out = load(str(tmp_path / "nonexistent"), split="test")
    assert out == []


def test_load_jsonl_split(tmp_path):
    """End-to-end load from a synthetic JSONL file."""
    path = tmp_path / "ds"
    path.mkdir()
    rows = [
        {"problem": "Q1", "formal_statement": "theorem q1 : True",
         "source": "imo"},
        {"problem": "Q2", "formal_statement": "theorem q2 : 1 = 1",
         "source": "amc"},
        # Malformed: no formal_statement → skipped
        {"problem": "Q3", "formal_statement": ""},
    ]
    (path / "test.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    problems = load(str(path), split="test")
    assert len(problems) == 2
    assert problems[0].problem_id == "numinamath_test_000000"
    assert problems[1].theorem_statement.startswith("theorem q2")


def test_load_skips_malformed_jsonl_lines(tmp_path):
    path = tmp_path / "ds"
    path.mkdir()
    content = (
        '{"formal_statement": "theorem a : True", "problem": "ok"}\n'
        'not-json-at-all\n'
        '{"formal_statement": "theorem b : True", "problem": "ok2"}\n'
        '\n'  # blank line
    )
    (path / "test.jsonl").write_text(content, encoding="utf-8")
    problems = load(str(path), split="test")
    assert len(problems) == 2


def test_load_missing_split_file_returns_empty(tmp_path):
    """If the directory exists but has no matching split file, return []."""
    path = tmp_path / "ds"
    path.mkdir()
    # Intentionally write a wrong-named file
    (path / "valid.jsonl").write_text(
        '{"formal_statement": "theorem x : True"}', encoding="utf-8")
    problems = load(str(path), split="test")
    assert problems == []


def test_load_via_top_level_dispatcher(tmp_path):
    """Confirm registration in benchmarks.loader works."""
    from benchmarks.loader import load_benchmark, list_benchmarks
    assert "numinamath_lean" in list_benchmarks()

    # Set up a minimal dataset on disk
    path = tmp_path / "ds"
    path.mkdir()
    (path / "test.jsonl").write_text(
        json.dumps({"formal_statement": "theorem x : True"}),
        encoding="utf-8")

    # Try each name alias
    for alias in ("numinamath_lean", "numinamath", "numina"):
        problems = load_benchmark(alias, split="test", path=str(path))
        assert len(problems) == 1, f"alias {alias!r} failed"
