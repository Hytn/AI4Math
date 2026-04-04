"""benchmarks/datasets/proofnet/loader.py — ProofNet (Lean 4) 加载器

支持 rahul3613/ProofNet-lean4 仓库结构:
  ProofNetLean4/*.lean
"""
from __future__ import annotations
import re, logging
from pathlib import Path
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)

_THEOREM_RE = re.compile(
    r'^(theorem|lemma)\s+(\S+)\s*([\s\S]*?)(?=\n(?:theorem|lemma|end|--|/-|noncomputable|open|section|namespace)\s|\Z)',
    re.MULTILINE)

def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    path = Path(repo_path)
    if not path.exists():
        logger.warning(f"ProofNet 路径不存在: {repo_path}")
        return []

    # rahul3613/ProofNet-lean4 结构: ProofNetLean4/*.lean
    src_dir = path / "ProofNetLean4"
    if not src_dir.exists():
        src_dir = path

    problems = []
    for lean_file in sorted(src_dir.rglob("*.lean")):
        if "lakefile" in lean_file.name.lower():
            continue
        content = lean_file.read_text(encoding="utf-8", errors="ignore")
        for m in _THEOREM_RE.finditer(content):
            name = m.group(2)
            full_text = m.group(0).strip()
            stmt = re.split(r'\s*:=\s*by\b', full_text, maxsplit=1)[0].strip()
            if not stmt:
                stmt = re.split(r'\s*:=\s*', full_text, maxsplit=1)[0].strip()
            problems.append(BenchmarkProblem(
                problem_id=f"proofnet_{name}",
                name=name,
                theorem_statement=stmt,
                difficulty="undergraduate",
                source="ProofNet",
            ))

    logger.info(f"ProofNet: 加载了 {len(problems)} 道题")
    return problems
