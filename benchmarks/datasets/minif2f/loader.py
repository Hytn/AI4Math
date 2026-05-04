"""benchmarks/datasets/minif2f/loader.py — miniF2F (Lean 4) 加载器

支持 yangky11/miniF2F-lean4 仓库结构:
  MiniF2F/
    Test.lean       ← 244 道 test
    Valid.lean       ← 244 道 valid

v11: 共享解析逻辑下沉到 ``benchmarks.datasets._base.parse_lean_files``,
本文件只保留路径探测 + 难度启发式。
"""
from __future__ import annotations

import logging
from pathlib import Path

from benchmarks.datasets._base import parse_lean_files
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)


def _difficulty(name: str) -> str:
    nl = name.lower()
    if "imo" in nl: return "competition"
    if "aime" in nl: return "hard"
    if "amc" in nl: return "medium"
    if "mathd" in nl: return "easy"
    return "medium"


def _candidate_files(repo: Path, split: str) -> list[Path]:
    """Resolve which Lean files to parse for the given split.

    Tries the canonical yangky11 layout first (MiniF2F/Test.lean), then
    a per-file directory layout, then a full recursive scan.
    """
    file_candidates = {
        "test":  ["MiniF2F/Test.lean", "test.lean", "Test.lean"],
        "valid": ["MiniF2F/Valid.lean", "valid.lean", "Valid.lean"],
    }
    for cand in file_candidates.get(split, file_candidates["test"]):
        p = repo / cand
        if p.is_file():
            return [p]

    dir_candidates = {
        "test":  ["MiniF2F/Test", "Test", "test"],
        "valid": ["MiniF2F/Valid", "Valid", "valid"],
    }
    for cand in dir_candidates.get(split, dir_candidates["test"]):
        p = repo / cand
        if p.is_dir():
            return sorted(p.rglob("*.lean"))

    return sorted(repo.rglob("*.lean"))


def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    """加载 miniF2F 数据集。

    yangky11/miniF2F-lean4 的 ``MiniF2F/Test.lean`` 是一个**索引文件**, 里面
    只有 244 行 ``import``, 没有 theorem 块。所以即便文件存在, 我们也得在
    解析返回 0 题时回退到目录扫描——这是 v10 的隐式契约 (它通过 if 判断
    解析结果非空来达到), v11 显式表达。
    """
    repo = Path(repo_path)
    if not repo.exists():
        logger.warning(f"miniF2F 路径不存在: {repo_path}")
        return []

    files = _candidate_files(repo, split)
    problems = parse_lean_files(
        files,
        problem_id_prefix=f"minif2f_{split}_",
        source="miniF2F",
        difficulty_fn=_difficulty,
        skip_sorry_in_statement=False,
    )

    # If the candidate file was an index-only file (just `import` lines),
    # fall through to a per-file directory scan.
    if not problems:
        dir_candidates = {
            "test":  ["MiniF2F/Test", "Test", "test"],
            "valid": ["MiniF2F/Valid", "Valid", "valid"],
        }
        for cand in dir_candidates.get(split, dir_candidates["test"]):
            p = repo / cand
            if p.is_dir():
                problems = parse_lean_files(
                    sorted(p.rglob("*.lean")),
                    problem_id_prefix=f"minif2f_{split}_",
                    source="miniF2F",
                    difficulty_fn=_difficulty,
                    skip_sorry_in_statement=False,
                )
                if problems:
                    break

    # Last-resort: full recursive scan
    if not problems:
        problems = parse_lean_files(
            sorted(repo.rglob("*.lean")),
            problem_id_prefix=f"minif2f_{split}_",
            source="miniF2F",
            difficulty_fn=_difficulty,
            skip_sorry_in_statement=False,
        )

    logger.info(f"miniF2F: 加载了 {len(problems)} 道题")
    return problems
