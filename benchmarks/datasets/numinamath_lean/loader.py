"""benchmarks/datasets/numinamath_lean/loader.py — NuminaMath-LEAN loader.

Loads the AI-MO/NuminaMath-LEAN dataset
(https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN). This is the
training/eval dataset behind Kimina-Prover 72B and the largest open
collection of human-annotated formal statements paired with NL
problems and proof attempts.

Dataset shape
-------------

The Hugging Face release ships as Parquet files (or JSONL after
conversion). Each record is a dict::

    {
        "problem":        "<NL problem statement>",
        "question_type":  "proof" | "value" | "...",
        "answer":         "<NL answer, may be empty>",
        "source":         "olympiads-ref" | "amc_aime" | ...,
        "problem_type":   "Algebra" | "Number Theory" | ...,
        "author":         "human" | "autoformalizer",
        "formal_statement": "theorem ... : ...",
        "formal_proof":     "by ..." | "",
        "rl_data": {"n_proofs": int, "n_correct_proofs": int}
    }

We expose this as ``BenchmarkProblem`` records, preserving the raw
NL problem (in the ``natural_language`` extra field) so the
``nfl_hybrid`` profile can use it.

How to obtain the data
----------------------

Two supported layouts:

1. **JSONL files** (preferred for offline runs)::

       data/NuminaMath-LEAN/
         train.jsonl     # used as 'train' split
         test.jsonl      # used as 'test' split

   Convert from the HF parquet with::

       python -c "import pandas as pd; \\
                  pd.read_parquet('train-00000-of-00001.parquet') \\
                    .to_json('train.jsonl', orient='records', lines=True)"

2. **Direct Parquet** (auto-detected if pandas is installed)::

       data/NuminaMath-LEAN/
         train-00000-of-00001.parquet
         test-00000-of-00001.parquet

If the directory is empty, ``load()`` returns ``[]`` and the
top-level ``benchmarks/loader.py`` prints a helpful download hint.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)

def _difficulty_from_record(rec: dict) -> str:
    src = (rec.get("source", "") or "").lower()
    if "imo" in src or "olympiad" in src:
        return "competition"
    if "aime" in src or "putnam" in src:
        return "hard"
    if "amc" in src:
        return "medium"
    if rec.get("problem_type") in ("Number Theory", "Algebra"):
        return "medium"
    return "easy"

def _record_to_problem(rec: dict, split: str,
                        idx: int) -> Optional[BenchmarkProblem]:
    formal = (rec.get("formal_statement") or "").strip()
    if not formal:
        return None
    name = rec.get("name") or f"numinamath_{split}_{idx}"
    pid = f"numinamath_{split}_{idx:06d}"

    # The NL problem text is what makes NuminaMath-LEAN useful for
    # NFL-HR / nfl_hybrid: the LLM sees both the formal target AND
    # the natural-language QA. We thread it through the top-level
    # ``natural_language`` field of BenchmarkProblem (which exists
    # on every model version) so downstream tools don't have to
    # know about a metadata blob.
    nl_problem = rec.get("problem", "") or ""

    extras = {
        "natural_language": nl_problem,
        "informal_answer": rec.get("answer", "") or "",
        "formal_proof_reference": rec.get("formal_proof", "") or "",
        "source": rec.get("source", ""),
        "author": rec.get("author", ""),
        "question_type": rec.get("question_type", ""),
        "problem_type": rec.get("problem_type", ""),
        "rl_data": rec.get("rl_data", {}) or {},
    }

    try:
        return BenchmarkProblem(
            problem_id=pid,
            name=name,
            theorem_statement=formal,
            difficulty=_difficulty_from_record(rec),
            source="NuminaMath-LEAN",
            natural_language=nl_problem,
            metadata=extras,
        )
    except TypeError:
        # BenchmarkProblem may not have `metadata` field on older
        # versions; fall back to setting only top-level fields. The
        # NL problem still survives via ``natural_language``.
        return BenchmarkProblem(
            problem_id=pid,
            name=name,
            theorem_statement=formal,
            difficulty=_difficulty_from_record(rec),
            source="NuminaMath-LEAN",
            natural_language=nl_problem,
        )

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.debug(f"NuminaMath-LEAN: skipping malformed line: {e}")

def _iter_parquet(path: Path):
    try:
        import pandas as pd  # noqa: I001
    except ImportError:
        logger.warning(
            "NuminaMath-LEAN: pandas not installed, cannot read parquet. "
            "Install pandas + pyarrow, or convert to JSONL.")
        return
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        yield row.to_dict()

def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    """Load NuminaMath-LEAN problems from disk.

    Returns an empty list (not an exception) when the data isn't
    present, so callers can fall back to other benchmarks.
    """
    path = Path(repo_path)
    if not path.exists():
        logger.warning(f"NuminaMath-LEAN path missing: {repo_path}")
        return []

    # Look for JSONL first.
    jsonl_candidates = [
        path / f"{split}.jsonl",
        path / f"{split}.json",
        path / f"numinamath_lean_{split}.jsonl",
    ]
    parquet_candidates = sorted(path.glob(f"{split}*.parquet"))

    problems: list[BenchmarkProblem] = []

    chosen = None
    for c in jsonl_candidates:
        if c.exists():
            chosen = ("jsonl", c)
            break
    if chosen is None and parquet_candidates:
        chosen = ("parquet", parquet_candidates[0])

    if chosen is None:
        logger.warning(
            f"NuminaMath-LEAN: no {split} split file found in {repo_path}. "
            "Expected one of: {jsonl_candidates} or "
            "{split}*.parquet")
        return []

    kind, src_path = chosen
    iterator = _iter_jsonl(src_path) if kind == "jsonl" else _iter_parquet(src_path)
    for idx, rec in enumerate(iterator):
        bp = _record_to_problem(rec, split, idx)
        if bp is not None:
            problems.append(bp)

    logger.info(
        f"NuminaMath-LEAN: loaded {len(problems)} problems "
        f"from {src_path.name} (split={split})")
    return problems

def hf_download_hint() -> str:
    """Return the recommended one-liner to fetch the dataset."""
    return (
        "huggingface-cli download AI-MO/NuminaMath-LEAN "
        "--repo-type dataset --local-dir data/NuminaMath-LEAN")
