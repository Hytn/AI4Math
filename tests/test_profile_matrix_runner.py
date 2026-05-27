from __future__ import annotations

from pathlib import Path

import importlib.util
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "scripts" / "eval" / "run_profile_matrix.py"


spec = importlib.util.spec_from_file_location("run_profile_matrix", MODULE_PATH)
matrix = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = matrix
spec.loader.exec_module(matrix)


def test_parse_matrix_accepts_string_and_mapping_entries():
    cfg = {
        "benchmarks": [
            "minif2f",
            {"name": "proofnet", "split": "valid", "project_dir": "data/ProofNet"},
        ],
        "profiles": ["whole_proof", {"name": "dsp", "max_turns": 3}],
    }

    benchmarks, profiles = matrix.parse_matrix(cfg)

    assert [b.name for b in benchmarks] == ["minif2f", "proofnet"]
    assert benchmarks[1].split == "valid"
    assert benchmarks[1].project_dir == "data/ProofNet"
    assert [p.name for p in profiles] == ["whole_proof", "dsp"]
    assert profiles[1].options["max_turns"] == 3


def test_build_runs_expands_cross_product_and_constructs_run_eval_commands(tmp_path):
    cfg = {
        "provider": "mock",
        "lean_mode": "skip",
        "max_samples": 1,
        "limit": 2,
        "resume": True,
        "no_knowledge": True,
        "benchmarks": [
            {"name": "minif2f", "project_dir": "data/miniF2F"},
            {"name": "proofnet", "project_dir": "data/ProofNet", "limit": 1},
        ],
        "profiles": [
            "whole_proof",
            {"name": "dsp", "temperature": 1.0},
        ],
    }

    runs = matrix.build_runs(cfg, output_root=tmp_path, python_exe="python3")

    assert len(runs) == 4
    first = runs[0]
    assert first.output_dir == tmp_path / "minif2f" / "whole_proof"
    assert first.log_file == tmp_path / "logs" / "minif2f" / "whole_proof.log"
    assert first.command[:3] == ["python3", str(REPO_ROOT / "run_eval.py"), "--benchmark"]
    assert "--profile" in first.command
    assert first.command[first.command.index("--profile") + 1] == "whole_proof"
    assert "--resume" in first.command
    assert "--no-knowledge" in first.command
    assert first.command[first.command.index("--project-dir") + 1] == "data/miniF2F"

    proofnet = runs[2]
    assert proofnet.command[proofnet.command.index("--limit") + 1] == "1"
    dsp = runs[1]
    assert dsp.command[dsp.command.index("--temperature") + 1] == "1.0"


def test_filters_reduce_matrix(tmp_path):
    cfg = {
        "benchmarks": ["minif2f", "proofnet"],
        "profiles": ["whole_proof", "dsp"],
    }

    runs = matrix.build_runs(
        cfg,
        output_root=tmp_path,
        python_exe="python3",
        benchmark_filter={"proofnet"},
        profile_filter={"dsp"},
    )

    assert len(runs) == 1
    assert runs[0].benchmark.name == "proofnet"
    assert runs[0].profile.name == "dsp"
