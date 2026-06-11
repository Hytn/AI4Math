"""benchmarks/datasets/matharena/loader.py — MathArena (formalized) 加载器

加载 ``scripts/eval/matharena_autoformalize.py`` 产出的
``data/MathArena/<comp>/formalized.jsonl``。

约定:
  - ``split`` 参数 = 竞赛名 (aime_2026 / hmmt_feb_2026 / ...);
    split 为空或 "test" 时加载 data_root 下**所有**竞赛;
  - ``flagged=true`` 的条目跳过 (形式化未过静态红线, 待人工修订);
  - License 提醒: MathArena 数据 CC BY-NC-SA 4.0, 本仓库不分发数据,
    formalized.jsonl 同样不应提交进 git。

⚠️ 评测报告须注明 formalizer 模型 (条目里的 formalizer_model 字段) —
   自动形式化未经人工 faithfulness 核对前, 结果不可与 miniF2F 等
   人工形式化基准直接比较。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)


def _load_comp(comp_dir: Path) -> list[BenchmarkProblem]:
    f = comp_dir / "formalized.jsonl"
    if not f.exists():
        return []
    comp = comp_dir.name
    out: list[BenchmarkProblem] = []
    n_flagged = 0
    for ln, line in enumerate(
            f.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("matharena %s line %d: bad json", comp, ln)
            continue
        if e.get("flagged"):
            n_flagged += 1
            continue
        stmt = (e.get("statement") or "").strip()
        if not stmt:
            continue
        idx = e.get("problem_idx", ln)
        out.append(BenchmarkProblem(
            problem_id=f"matharena_{comp}_{idx}",
            name=f"{comp}_q{idx}",
            theorem_statement=stmt,
            difficulty="competition",
            source=f"matharena/{comp}",
            natural_language=e.get("problem", ""),
            tags=["matharena", comp,
                  f"formalizer:{e.get('formalizer_model', 'unknown')}",
                  "autoformalized_unverified"],
            lean_preamble="import Mathlib\n",
        ))
    if n_flagged:
        logger.warning(
            "matharena %s: skipped %d flagged formalizations "
            "(human revision pending)", comp, n_flagged)
    return out


def load(path: str = "data/MathArena", split: str = "test",
         limit: int = 0) -> list[BenchmarkProblem]:
    root = Path(path)
    if not root.is_dir():
        return []
    comps: list[Path]
    if split and split not in ("test", "all", ""):
        comps = [root / split]
    else:
        comps = sorted(d for d in root.iterdir() if d.is_dir())
    problems: list[BenchmarkProblem] = []
    for c in comps:
        problems.extend(_load_comp(c))
    if limit > 0:
        problems = problems[:limit]
    logger.info("matharena: loaded %d formalized problems from %s "
                "(split=%s)", len(problems), path, split or "all")
    return problems
