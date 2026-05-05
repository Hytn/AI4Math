"""benchmarks/datasets/fate/loader.py — FATE-M/H/X 加载器

FATE (Formal Algebra Theorem Evaluation) 系列基准:
  FATE-M: 150 题, 本科抽象代数 (medium)
  FATE-H: 100 题, 荣誉课程/研究生级 (hard)
  FATE-X: 100 题, 博士资格考试级 (extreme)

数据来源: https://github.com/frenzymath
论文: "FATE: A Formal Benchmark Series for Frontier Algebra of Multiple Difficulty Levels"


本地 (FATE 的 JSON 同时携带 informal_statement, 不能简单复用 base)。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from benchmarks.datasets._base import parse_lean_files
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)

_DIFFICULTY_MAP = {
    "fate-m": "medium", "fatem": "medium",
    "fate-h": "hard",   "fateh": "hard",
    "fate-x": "extreme", "fatex": "extreme",
}

def _try_json(path: Path, prefix: str, difficulty: str
              ) -> list[BenchmarkProblem]:
    """Load FATE from its JSON manifest if present. Returns [] otherwise.

    The JSON path supplies natural-language informal_statement, which
    the bare-Lean path can't recover; that's why we keep this logic
    instead of folding it into ``parse_lean_files``.
    """
    for json_name in [f"{path.name}.json", "manifest.json", "problems.json"]:
        json_file = path / json_name
        if not json_file.exists():
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"FATE JSON 解析错误 ({json_file}): {e}")
            continue

        problems: list[BenchmarkProblem] = []
        for item in data:
            pid = item.get("id", len(problems) + 1)
            formal = item.get("formal_statement", "")
            informal = item.get("informal_statement", "")

            name_match = re.search(r'(?:theorem|lemma)\s+(\S+)', formal)
            name = name_match.group(1) if name_match else f"{prefix}_{pid}"

            stmt_lines = [l for l in formal.strip().split('\n')
                          if not l.strip().startswith('import ')]
            stmt = '\n'.join(stmt_lines).strip()
            parts_by = re.split(r'\s*:=\s*by\b', stmt, maxsplit=1)
            parts_eq = re.split(r'\s*:=\s*sorry\s*$', stmt, maxsplit=1)
            if len(parts_by) > 1:
                stmt = parts_by[0].strip()
            elif len(parts_eq) > 1:
                stmt = parts_eq[0].strip()

            problems.append(BenchmarkProblem(
                problem_id=f"{prefix}_{pid}",
                name=name,
                theorem_statement=stmt,
                difficulty=difficulty,
                source=f"FATE-{difficulty[0].upper()}",
                natural_language=informal,
            ))
        logger.info(
            f"FATE: 从 {json_file} 加载了 {len(problems)} 道题 "
            f"(difficulty={difficulty})")
        return problems
    return []

def load(repo_path: str, split: str = "test", variant: str = "") -> list[BenchmarkProblem]:
    path = Path(repo_path)
    if not path.exists():
        logger.warning(f"FATE 路径不存在: {repo_path}")
        return []

    dir_name = path.name.lower().replace("_", "-")
    difficulty = _DIFFICULTY_MAP.get(dir_name, variant or "unknown")
    prefix = dir_name.replace("-", "")  # fatem / fateh / fatex

    # 1) JSON manifest path (preferred — carries informal statements)
    json_problems = _try_json(path, prefix, difficulty)
    if json_problems:
        return json_problems

    # 2) Bare Lean files fallback
    files: list[Path] = []
    for cand in [path / "FATEM", path / "FATEH", path / "FATEX",
                 path / "src", path]:
        if cand.is_dir():
            files.extend(sorted(cand.rglob("*.lean")))
    problems = parse_lean_files(
        files,
        problem_id_prefix=f"{prefix}_",
        source=f"FATE-{difficulty[0].upper()}",
        difficulty_fn=lambda _name: difficulty,
        skip_sorry_in_statement=False,
    )
    logger.info(
        f"FATE: 从 {repo_path} 加载了 {len(problems)} 道题 "
        f"(difficulty={difficulty})")
    return problems
