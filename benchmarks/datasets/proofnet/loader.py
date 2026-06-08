"""benchmarks/datasets/proofnet/loader.py — DeepSeek-Prover-V1.5 ProofNet loader.

This loader intentionally uses only ``proofnet.jsonl`` from
``deepseek-ai/DeepSeek-Prover-V1.5/datasets/proofnet.jsonl``.
That benchmark has 371 rows total, split into valid=185 and test=186.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)


def _normalise_split(split: str) -> str:
    value = (split or "test").strip().lower()
    if value in {"", "default"}:
        return "test"
    if value in {"all", "full", "both"}:
        return "all"
    return value


def _record_statement(item: dict) -> str:
    stmt = str(item.get("formal_statement") or item.get("theorem_statement") or "")
    return stmt.strip()


def _record_informal(item: dict) -> str:
    raw = item.get("informal_statement") or item.get("informal_stmt") or ""
    if raw:
        return str(raw).strip()
    prefix = str(item.get("informal_prefix") or "").strip()
    if prefix.startswith("/--") and prefix.endswith("-/"):
        return prefix[3:-2].strip()
    return prefix


def _load_jsonl(jsonl_file: Path, split: str) -> list[BenchmarkProblem]:
    wanted_split = _normalise_split(split)
    problems: list[BenchmarkProblem] = []

    try:
        lines: Iterable[str] = jsonl_file.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        logger.warning(f"ProofNet JSONL 读取失败 ({jsonl_file}): {e}")
        return []

    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(f"ProofNet JSONL 解析错误 ({jsonl_file}:{line_num}): {e}")
            continue
        item_split = str(item.get("split") or "test").lower()
        if wanted_split != "all" and item_split != wanted_split:
            continue

        name = str(item.get("name") or f"proofnet_{line_num}")
        problem_id = f"proofnet_{line_num:06d}_{name}"
        stmt = _record_statement(item)
        if not stmt:
            logger.warning(f"ProofNet JSONL 空 formal_statement: {jsonl_file}:{line_num}")
            continue

        problems.append(BenchmarkProblem(
            problem_id=problem_id,
            name=name,
            theorem_statement=stmt,
            difficulty="undergraduate",
            source="ProofNet",
            natural_language=_record_informal(item),
            lean_preamble=str(item.get("header") or "").strip(),
            tags=[item_split] if item_split else [],
        ))

    logger.info(
        f"ProofNet: 从 {jsonl_file} 加载了 {len(problems)} 道题 "
        f"(split={wanted_split})")
    return problems


def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    repo = Path(repo_path)
    if not repo.exists():
        logger.warning(f"ProofNet 路径不存在: {repo_path}")
        return []

    jsonl_file = repo / "proofnet.jsonl"
    if not jsonl_file.is_file():
        logger.warning(
            f"ProofNet: 未找到 DeepSeek-Prover-V1.5 proofnet.jsonl: {jsonl_file}")
        return []
    return _load_jsonl(jsonl_file, split)
