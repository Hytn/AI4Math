"""benchmarks/datasets/putnambench/loader.py — PutnamBench 加载器

支持 trishullab/PutnamBench 仓库结构:
  lean4/src/putnam_*.lean    ← Lean 4 题目
"""
from __future__ import annotations
import re, logging
from pathlib import Path
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)

_THEOREM_RE = re.compile(
    r'^(theorem|lemma)\s+(\S+)\s*([\s\S]*?)(?=\n(?:theorem|lemma|end|--|/-|noncomputable|open|section|namespace)\s|\Z)',
    re.MULTILINE)

def _difficulty_from_year(name: str) -> str:
    # putnam_1988_b1 → extract year
    m = re.search(r'(\d{4})', name)
    if not m: return "competition"
    year = int(m.group(1))
    # A problems tend to be easier, B problems harder
    if "_a1" in name.lower() or "_a2" in name.lower(): return "medium"
    if "_b5" in name.lower() or "_b6" in name.lower(): return "hard"
    return "competition"

def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    path = Path(repo_path)
    if not path.exists():
        logger.warning(f"PutnamBench 路径不存在: {repo_path}")
        return []

    # trishullab/PutnamBench 结构: lean4/src/putnam_*.lean
    lean4_dir = path / "lean4" / "src"
    if not lean4_dir.exists():
        # 也尝试根目录
        lean4_dir = path

    problems = []
    for lean_file in sorted(lean4_dir.rglob("*.lean")):
        if "lakefile" in lean_file.name.lower():
            continue
        content = lean_file.read_text(encoding="utf-8", errors="ignore")
        for m in _THEOREM_RE.finditer(content):
            name = m.group(2)
            full_text = m.group(0).strip()
            stmt = re.split(r'\s*:=\s*by\b', full_text, maxsplit=1)[0].strip()
            if not stmt:
                stmt = re.split(r'\s*:=\s*', full_text, maxsplit=1)[0].strip()
            if "sorry" in stmt:
                continue  # skip malformed
            problems.append(BenchmarkProblem(
                problem_id=f"putnam_{name}",
                name=name,
                theorem_statement=stmt,
                difficulty=_difficulty_from_year(name),
                source="PutnamBench",
            ))

    logger.info(f"PutnamBench: 加载了 {len(problems)} 道题")
    return problems
