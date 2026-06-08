import json

from benchmarks.datasets.proofnet.loader import load


def test_proofnet_loader_reads_deepseek_jsonl_and_filters_split(tmp_path):
    repo = tmp_path / "ProofNet"
    repo.mkdir()
    rows = [
        {
            "name": "valid_one",
            "split": "valid",
            "informal_prefix": "/-- valid stmt -/\n",
            "formal_statement": "theorem valid_one : True :=",
            "header": "import Mathlib\n",
        },
        {
            "name": "test_one",
            "split": "test",
            "informal_prefix": "/-- test stmt -/\n",
            "formal_statement": "theorem test_one : True :=",
            "header": "import Mathlib\nopen Nat\n",
        },
    ]
    (repo / "proofnet.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    formal = repo / "formal"
    formal.mkdir()
    (formal / "Noise.lean").write_text(
        "theorem should_not_be_loaded : True := by trivial\n",
        encoding="utf-8",
    )

    test_problems = load(str(repo), split="test")
    valid_problems = load(str(repo), split="valid")
    all_problems = load(str(repo), split="all")

    assert [p.problem_id for p in test_problems] == ["proofnet_000002_test_one"]
    assert [p.problem_id for p in valid_problems] == ["proofnet_000001_valid_one"]
    assert {p.problem_id for p in all_problems} == {
        "proofnet_000001_valid_one",
        "proofnet_000002_test_one",
    }
    assert test_problems[0].theorem_statement == "theorem test_one : True :="
    assert test_problems[0].lean_preamble == "import Mathlib\nopen Nat"
    assert test_problems[0].natural_language == "test stmt"
    assert test_problems[0].tags == ["test"]


def test_proofnet_jsonl_keeps_duplicate_names_as_distinct_entries(tmp_path):
    repo = tmp_path / "ProofNet"
    repo.mkdir()
    (repo / "proofnet.jsonl").write_text(
        '{"name":"dup","split":"test","formal_statement":"theorem dup : True :="}\n'
        '{"name":"dup","split":"test","formal_statement":"theorem dup : True :="}\n',
        encoding="utf-8",
    )

    problems = load(str(repo), split="test")

    assert len(problems) == 2
    assert [p.name for p in problems] == ["dup", "dup"]
    assert [p.problem_id for p in problems] == [
        "proofnet_000001_dup",
        "proofnet_000002_dup",
    ]


def test_proofnet_loader_requires_deepseek_jsonl(tmp_path):
    repo = tmp_path / "ProofNet"
    repo.mkdir()
    (repo / "formal").mkdir()

    assert load(str(repo), split="test") == []
