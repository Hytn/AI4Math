"""benchmarks/datasets/proofnet/loader.py — PAug/ProofNetSharp loader.

The preferred local layout mirrors the Hugging Face dataset exactly::

    data/ProofNet/data/valid-00000-of-00001.parquet
    data/ProofNet/data/test-00000-of-00001.parquet

Those parquet files keep the original ProofNetSharp columns unchanged:
``id``, ``nl_statement``, ``nl_proof``, ``lean4_src_header``, and
``lean4_formalization``.  This loader maps them to ``BenchmarkProblem`` only
in memory so the dataset files themselves stay byte-identical to the source.
A legacy ``proofnet.jsonl`` fallback is kept for older local checkouts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)


_SPLIT_FILES = {
    "valid": "data/valid-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
_SPLIT_OFFSETS = {"valid": 0, "test": 185}


def _normalise_split(split: str) -> str:
    value = (split or "test").strip().lower()
    if value in {"", "default"}:
        return "test"
    if value in {"all", "full", "both"}:
        return "all"
    return value


def _name_from_id(raw_id: str, fallback: str) -> str:
    raw = str(raw_id or "").strip()
    if "|" in raw:
        tail = raw.rsplit("|", 1)[-1].strip()
        if tail:
            return tail
    return raw or fallback


def _record_statement(item: dict) -> str:
    stmt = str(
        item.get("lean4_formalization")
        or item.get("formal_statement")
        or item.get("theorem_statement")
        or ""
    )
    return stmt.strip()


def _record_informal(item: dict) -> str:
    raw = item.get("nl_statement") or item.get("informal_statement") or item.get("informal_stmt") or ""
    if raw:
        return str(raw).strip()
    prefix = str(item.get("informal_prefix") or "").strip()
    if prefix.startswith("/--") and prefix.endswith("-/"):
        return prefix[3:-2].strip()
    return prefix


def _record_preamble(item: dict) -> str:
    return str(item.get("lean4_src_header") or item.get("header") or "").strip()


def _to_problem(item: dict, *, split: str, global_index: int) -> BenchmarkProblem | None:
    raw_id = str(item.get("id") or item.get("proofnetsharp_id") or "").strip()
    name = _name_from_id(raw_id, f"proofnet_{global_index}")
    stmt = _record_statement(item)
    if not stmt:
        logger.warning("ProofNetSharp 空 lean4_formalization: split=%s index=%s", split, global_index)
        return None
    tags = [split]
    if raw_id:
        tags.append(raw_id)
    return BenchmarkProblem(
        problem_id=f"proofnet_{global_index:06d}_{name}",
        name=name,
        theorem_statement=stmt,
        difficulty="undergraduate",
        source="ProofNetSharp",
        natural_language=_record_informal(item),
        lean_preamble=_record_preamble(item),
        tags=tags,
    )


def _load_parquet_file(parquet_file: Path, split: str, offset: int) -> list[BenchmarkProblem]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "ProofNetSharp parquet loading requires pyarrow. Install with: "
            "python3 -m pip install pyarrow"
        ) from e

    try:
        rows = pq.read_table(parquet_file).to_pylist()
    except OSError as e:
        logger.warning("ProofNetSharp parquet 读取失败 (%s): %s", parquet_file, e)
        return []

    problems: list[BenchmarkProblem] = []
    for local_idx, item in enumerate(rows, 1):
        problem = _to_problem(item, split=split, global_index=offset + local_idx)
        if problem is not None:
            problems.append(problem)
    logger.info("ProofNetSharp: 从 %s 加载了 %d 道题", parquet_file, len(problems))
    return problems


def _load_parquet(repo: Path, split: str) -> list[BenchmarkProblem]:
    wanted = _normalise_split(split)
    splits = ["valid", "test"] if wanted == "all" else [wanted]
    problems: list[BenchmarkProblem] = []
    for sp in splits:
        rel = _SPLIT_FILES.get(sp)
        if rel is None:
            logger.warning("ProofNetSharp: 不支持 split=%s", sp)
            continue
        parquet_file = repo / rel
        if not parquet_file.is_file():
            logger.warning("ProofNetSharp: 未找到 parquet 文件: %s", parquet_file)
            continue
        problems.extend(_load_parquet_file(parquet_file, sp, _SPLIT_OFFSETS[sp]))
    logger.info("ProofNetSharp: 加载 %d 道题 (split=%s)", len(problems), wanted)
    return problems


def _load_jsonl(jsonl_file: Path, split: str) -> list[BenchmarkProblem]:
    wanted_split = _normalise_split(split)
    problems: list[BenchmarkProblem] = []

    try:
        lines: Iterable[str] = jsonl_file.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        logger.warning("ProofNet JSONL 读取失败 (%s): %s", jsonl_file, e)
        return []

    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("ProofNet JSONL 解析错误 (%s:%s): %s", jsonl_file, line_num, e)
            continue
        item_split = str(item.get("split") or "test").lower()
        if wanted_split != "all" and item_split != wanted_split:
            continue
        problem = _to_problem(item, split=item_split, global_index=line_num)
        if problem is not None:
            problems.append(problem)

    logger.info("ProofNet: 从 %s 加载了 %d 道题 (split=%s)", jsonl_file, len(problems), wanted_split)
    return problems


def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    repo = Path(repo_path)
    if not repo.exists():
        logger.warning("ProofNetSharp 路径不存在: %s", repo_path)
        return []

    has_parquet = any((repo / rel).is_file() for rel in _SPLIT_FILES.values())
    if has_parquet:
        return _load_parquet(repo, split)

    jsonl_file = repo / "proofnet.jsonl"
    if jsonl_file.is_file():
        return _load_jsonl(jsonl_file, split)

    logger.warning(
        "ProofNetSharp: 未找到 Hugging Face parquet 文件或 proofnet.jsonl: %s",
        repo,
    )
    return []
