"""benchmarks/datasets/formalmath/loader.py — FormalMATH 加载器

支持 Sphere-AI-Lab/FormalMATH-Bench 仓库结构:
  data/*.jsonl 或 lean4/*.lean

v11: Lean 文件回退路径下沉到 ``_base.parse_lean_files``。JSONL/JSON 路径
保留本地 (它们携带 informal_statement / difficulty)。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from benchmarks.datasets._base import parse_lean_files
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)


def _from_record(item: dict, line_num: int) -> BenchmarkProblem | None:
    stmt = item.get("formal_statement",
                     item.get("theorem_statement", ""))
    if not stmt:
        return None
    name = item.get("name", item.get("id", f"fm_{line_num}"))
    return BenchmarkProblem(
        problem_id=f"formalmath_{name}",
        name=str(name),
        theorem_statement=stmt.strip(),
        difficulty=item.get("difficulty", item.get("level", "unknown")),
        source="FormalMATH",
        natural_language=item.get("informal_statement", ""),
    )


def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    path = Path(repo_path)
    if not path.exists():
        logger.warning(f"FormalMATH 路径不存在: {repo_path}")
        return []

    problems: list[BenchmarkProblem] = []

    # 1) JSONL (preferred — most common in this dataset)
    for jsonl_file in sorted(path.rglob("*.jsonl")):
        try:
            text = jsonl_file.read_text(encoding="utf-8").strip()
        except OSError as e:
            logger.warning(f"FormalMATH 读取失败 ({jsonl_file}): {e}")
            continue
        for line_num, line in enumerate(text.split("\n")):
            if not line.strip():
                continue
            try:
                problem = _from_record(json.loads(line), line_num)
                if problem is not None:
                    problems.append(problem)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"FormalMATH JSONL 解析错误 ({jsonl_file}:{line_num}): {e}")

    # 2) JSON arrays fallback
    if not problems:
        for json_file in sorted(path.rglob("*.json")):
            if "lake" in json_file.name.lower():
                continue
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"FormalMATH JSON 解析错误 ({json_file}): {e}")
                continue
            if isinstance(data, list):
                for i, item in enumerate(data):
                    p = _from_record(item, i)
                    if p is not None:
                        problems.append(p)

    # 3) Bare Lean files
    if not problems:
        files = sorted(path.rglob("*.lean"))
        problems = parse_lean_files(
            files,
            problem_id_prefix="formalmath_",
            source="FormalMATH",
            difficulty_fn=lambda _name: "unknown",
            skip_sorry_in_statement=False,
        )

    logger.info(f"FormalMATH: 加载了 {len(problems)} 道题")
    return problems
