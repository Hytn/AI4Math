"""benchmarks/datasets/formalmath/loader.py — FormalMATH 加载器

支持 Sphere-AI-Lab/FormalMATH-Bench 仓库结构:
  data/*.jsonl 或 lean4/*.lean
"""
from __future__ import annotations
import re, json, logging
from pathlib import Path
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)

_THEOREM_RE = re.compile(
    r'^(theorem|lemma)\s+(\S+)\s*([\s\S]*?)(?=\n(?:theorem|lemma|end|--|/-|noncomputable|open|section|namespace)\s|\Z)',
    re.MULTILINE)

def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    path = Path(repo_path)
    if not path.exists():
        logger.warning(f"FormalMATH 路径不存在: {repo_path}")
        return []

    problems = []

    # 尝试 JSONL 文件 (优先)
    for jsonl_file in sorted(path.rglob("*.jsonl")):
        try:
            for line_num, line in enumerate(jsonl_file.read_text(encoding="utf-8").strip().split("\n")):
                if not line.strip(): continue
                item = json.loads(line)
                stmt = item.get("formal_statement", item.get("theorem_statement", ""))
                name = item.get("name", item.get("id", f"fm_{line_num}"))
                if not stmt: continue
                problems.append(BenchmarkProblem(
                    problem_id=f"formalmath_{name}",
                    name=str(name),
                    theorem_statement=stmt.strip(),
                    difficulty=item.get("difficulty", item.get("level", "unknown")),
                    source="FormalMATH",
                    natural_language=item.get("informal_statement", ""),
                ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"FormalMATH JSONL 解析错误 ({jsonl_file}): {e}")

    # 尝试 JSON 文件
    if not problems:
        for json_file in sorted(path.rglob("*.json")):
            if "lake" in json_file.name.lower(): continue
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        stmt = item.get("formal_statement", item.get("theorem_statement", ""))
                        name = item.get("name", item.get("id", f"fm_{i}"))
                        if not stmt: continue
                        problems.append(BenchmarkProblem(
                            problem_id=f"formalmath_{name}",
                            name=str(name),
                            theorem_statement=stmt.strip(),
                            difficulty=item.get("difficulty", "unknown"),
                            source="FormalMATH",
                            natural_language=item.get("informal_statement", ""),
                        ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"FormalMATH JSON 解析错误 ({json_file}): {e}")

    # Fallback: 解析 .lean 文件
    if not problems:
        for lean_file in sorted(path.rglob("*.lean")):
            if "lakefile" in lean_file.name.lower(): continue
            content = lean_file.read_text(encoding="utf-8", errors="ignore")
            for m in _THEOREM_RE.finditer(content):
                name = m.group(2)
                full_text = m.group(0).strip()
                stmt = re.split(r'\s*:=\s*by\b', full_text, maxsplit=1)
                stmt2 = re.split(r'\s*:=\s*', full_text, maxsplit=1)
                if len(stmt) > 1:
                    stmt = stmt[0].strip()
                elif len(stmt2) > 1:
                    stmt = stmt2[0].strip()
                else:
                    stmt = full_text.strip()
                problems.append(BenchmarkProblem(
                    problem_id=f"formalmath_{name}",
                    name=name,
                    theorem_statement=stmt,
                    difficulty="unknown",
                    source="FormalMATH",
                ))

    logger.info(f"FormalMATH: 加载了 {len(problems)} 道题")
    return problems
