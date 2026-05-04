"""benchmarks/datasets/proofnet/loader.py — ProofNet (Lean 4) 加载器

支持 rahul3613/ProofNet-lean4 仓库结构:
  ProofNetLean4/*.lean

v11: 共享解析逻辑下沉到 ``_base.parse_lean_files``。
"""
from __future__ import annotations

import logging
from pathlib import Path

from benchmarks.datasets._base import parse_lean_files
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)


def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    repo = Path(repo_path)
    if not repo.exists():
        logger.warning(f"ProofNet 路径不存在: {repo_path}")
        return []

    src = repo / "ProofNetLean4"
    files = sorted((src if src.exists() else repo).rglob("*.lean"))
    problems = parse_lean_files(
        files,
        problem_id_prefix="proofnet_",
        source="ProofNet",
        difficulty_fn=lambda _name: "undergraduate",
        skip_sorry_in_statement=False,
    )
    logger.info(f"ProofNet: 加载了 {len(problems)} 道题")
    return problems
