"""benchmarks/datasets/_base.py — Shared Lean theorem-file loader.

Six dataset loaders (minif2f / putnambench / proofnet / fate /
formalmath / numinamath_lean) all do the same three things: walk a
directory of ``.lean`` files, regex out ``theorem`` / ``lemma`` blocks,
split off ``:= by`` to keep the statement.  They've drifted in trivial
ways (a sorry filter here, a missing one there, slightly different
``_THEOREM_RE``) — drift that produces inconsistent BenchmarkProblems
when two loaders see the same file.

This module factors the shared core into ``parse_lean_files``.
Per-dataset loaders become ~20 lines: a directory probe + one call.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Iterable, Optional

from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)


# Boundary keywords that delimit a single theorem block. Anything in
# this set, when seen at the start of a line, ends the previous block.
# Matches the most permissive form across all six original loaders.
_BOUNDARY_KEYWORDS = (
    "theorem", "lemma", "end", "noncomputable",
    "open", "section", "namespace",
)
_BOUNDARY_PATTERN = "|".join(_BOUNDARY_KEYWORDS) + r"|--|/-"

_THEOREM_RE = re.compile(
    rf'^(theorem|lemma)\s+(\S+)\s*([\s\S]*?)'
    rf'(?=\n(?:{_BOUNDARY_PATTERN})\s|\Z)',
    re.MULTILINE,
)


def _split_statement(full_text: str) -> str:
    """Strip ``:= by ...`` (or bare ``:= ...``) to keep just the statement."""
    parts_by = re.split(r'\s*:=\s*by\b', full_text, maxsplit=1)
    if len(parts_by) > 1:
        return parts_by[0].strip()
    parts_eq = re.split(r'\s*:=\s*', full_text, maxsplit=1)
    if len(parts_eq) > 1:
        return parts_eq[0].strip()
    return full_text.strip()


def parse_lean_files(
    files: Iterable[Path],
    *,
    problem_id_prefix: str,
    source: str,
    difficulty_fn: Optional[Callable[[str], str]] = None,
    skip_sorry_in_statement: bool = False,
    extra_fields: Optional[Callable[[str, str], dict]] = None,
) -> list[BenchmarkProblem]:
    """Parse a list of Lean source files into BenchmarkProblems.

    Args:
        files: iterable of Path to .lean files (lakefile-like names skipped).
        problem_id_prefix: prefix joined with theorem name to form
            ``problem_id``. Pass an empty string to use the bare name.
        source: filled into BenchmarkProblem.source ("miniF2F", etc).
        difficulty_fn: optional ``name -> str`` heuristic; defaults to "medium".
        skip_sorry_in_statement: if True, drop entries whose extracted
            statement still contains ``sorry`` (PutnamBench-style guard).
        extra_fields: optional callable ``(name, full_text) -> dict`` whose
            return value is merged into BenchmarkProblem kwargs (e.g. for
            additional metadata loaders may want to attach).
    """
    problems: list[BenchmarkProblem] = []
    seen_files = 0
    for lean_file in files:
        if "lakefile" in lean_file.name.lower():
            continue
        try:
            content = lean_file.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            logger.debug(f"skipping unreadable {lean_file}: {e}")
            continue
        seen_files += 1
        for m in _THEOREM_RE.finditer(content):
            name = m.group(2)
            stmt = _split_statement(m.group(0).strip())
            if skip_sorry_in_statement and "sorry" in stmt:
                continue
            kwargs = dict(
                problem_id=f"{problem_id_prefix}{name}" if problem_id_prefix else name,
                name=name,
                theorem_statement=stmt,
                difficulty=(difficulty_fn(name) if difficulty_fn else "medium"),
                source=source,
            )
            if extra_fields is not None:
                try:
                    kwargs.update(extra_fields(name, m.group(0)))
                except Exception as e:
                    logger.debug(f"extra_fields for {name}: {e}")
            problems.append(BenchmarkProblem(**kwargs))
    logger.debug(
        f"{source}: walked {seen_files} files, extracted {len(problems)} problems")
    return problems


def walk_lean_files(root: Path,
                    candidate_paths: list[str] = None) -> list[Path]:
    """Find all relevant ``.lean`` files under root.

    If ``candidate_paths`` is given, the first one that resolves to a
    file or directory is used; otherwise we recurse from ``root``.
    """
    if not root.exists():
        return []
    if candidate_paths:
        for cand in candidate_paths:
            p = root / cand
            if p.is_file():
                return [p]
            if p.is_dir():
                return sorted(p.rglob("*.lean"))
    return sorted(root.rglob("*.lean"))
