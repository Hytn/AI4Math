"""benchmarks/datasets/putnambench/loader.py — PutnamBench 加载器

支持 trishullab/PutnamBench 仓库结构:
  lean4/src/putnam_*.lean    ← Lean 4 题目

v11: 共享解析逻辑下沉到 ``_base.parse_lean_files``。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from benchmarks.datasets._base import parse_lean_files
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)


def _difficulty(name: str) -> str:
    nl = name.lower()
    if not re.search(r'(\d{4})', name):
        return "competition"
    if "_a1" in nl or "_a2" in nl:
        return "medium"
    if "_b5" in nl or "_b6" in nl:
        return "hard"
    return "competition"


def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    repo = Path(repo_path)
    if not repo.exists():
        logger.warning(f"PutnamBench 路径不存在: {repo_path}")
        return []

    src = repo / "lean4" / "src"
    files = sorted((src if src.exists() else repo).rglob("*.lean"))
    problems = parse_lean_files(
        files,
        problem_id_prefix="putnam_",
        source="PutnamBench",
        difficulty_fn=_difficulty,
        skip_sorry_in_statement=True,  # PutnamBench: drop malformed entries
    )
    logger.info(f"PutnamBench: 加载了 {len(problems)} 道题")
    return problems
