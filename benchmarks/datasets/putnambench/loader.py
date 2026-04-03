"""benchmarks/datasets/putnambench/loader.py — putnambench dataset loader"""
from __future__ import annotations
import json, re, logging
from pathlib import Path
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)

def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    """Load putnambench problems from local clone."""
    path = Path(repo_path)
    if not path.exists():
        logger.warning(f"putnambench path not found: {repo_path}")
        return []

    # Try JSON manifest first
    manifest = path / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            data = json.load(f)
        return [BenchmarkProblem(
            problem_id=item.get("problem_id", f"putnambench_{i}"),
            name=item.get("name", f"problem_{i}"),
            theorem_statement=item["theorem_statement"],
            difficulty=item.get("difficulty", "unknown"),
            source="putnambench",
            natural_language=item.get("natural_language", ""),
        ) for i, item in enumerate(data) if item.get("split", "test") == split]

    # Try parsing .lean files
    problems = []
    for lean_file in sorted(path.rglob("*.lean")):
        content = lean_file.read_text(encoding="utf-8")
        for m in re.finditer(r"^(theorem\s+(\S+).*?)(?=\n(?:theorem|lemma|def|end)\s|\Z)", content, re.MULTILINE | re.DOTALL):
            name = m.group(2)
            stmt = m.group(1).split(":=")[0].strip() if ":=" in m.group(1) else m.group(1).strip()
            problems.append(BenchmarkProblem(
                problem_id=f"putnambench_{split}_{name}", name=name,
                theorem_statement=stmt, source="putnambench"))
    logger.info(f"Loaded {len(problems)} problems from putnambench")
    return problems
