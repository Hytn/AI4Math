"""benchmarks/datasets/fate/loader.py — FATE-M/H/X 加载器

FATE (Formal Algebra Theorem Evaluation) 系列基准:
  FATE-M: 150 题, 本科抽象代数 (medium)
  FATE-H: 100 题, 荣誉课程/研究生级 (hard)
  FATE-X: 100 题, 博士资格考试级 (extreme)

数据来源: https://github.com/frenzymath
论文: "FATE: A Formal Benchmark Series for Frontier Algebra of Multiple Difficulty Levels"
"""
from __future__ import annotations
import json, re, logging
from pathlib import Path
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)

# FATE 难度等级到 variant 的映射
_DIFFICULTY_MAP = {
    "fate-m": "medium",
    "fatem": "medium",
    "fate-h": "hard",
    "fateh": "hard",
    "fate-x": "extreme",
    "fatex": "extreme",
}

_THEOREM_RE = re.compile(
    r'^(theorem|lemma)\s+(\S+)\s*([\s\S]*?)(?=\n(?:theorem|lemma|end|--|/-|noncomputable|open|section|namespace)\s|\Z)',
    re.MULTILINE)


def load(repo_path: str, split: str = "test", variant: str = "") -> list[BenchmarkProblem]:
    """加载 FATE 数据集。

    Args:
        repo_path: FATE-M / FATE-H / FATE-X 仓库路径
        split: 数据集切分 (FATE 不分 train/test, 全部用于评测)
        variant: 难度变体, 从路径名自动推断
    """
    path = Path(repo_path)
    if not path.exists():
        logger.warning(f"FATE 路径不存在: {repo_path}")
        return []

    # 推断 variant
    dir_name = path.name.lower().replace("_", "-")
    difficulty = _DIFFICULTY_MAP.get(dir_name, variant or "unknown")
    prefix = dir_name.replace("-", "")  # fatem, fateh, fatex

    problems = []

    # 优先从 JSON 文件加载 (结构化数据, 更可靠)
    for json_name in [f"{path.name}.json", "manifest.json", "problems.json"]:
        json_file = path / json_name
        if json_file.exists():
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    pid = item.get("id", len(problems) + 1)
                    formal = item.get("formal_statement", "")
                    informal = item.get("informal_statement", "")
                    source = item.get("source", "FATE")

                    # 从 formal_statement 中提取 theorem 名
                    name_match = re.search(r'(?:theorem|lemma)\s+(\S+)', formal)
                    name = name_match.group(1) if name_match else f"{prefix}_{pid}"

                    # 提取 statement (去掉 import 和 := sorry 部分)
                    stmt = formal.strip()
                    # 去掉 import 行
                    stmt_lines = [l for l in stmt.split('\n') if not l.strip().startswith('import ')]
                    stmt = '\n'.join(stmt_lines).strip()
                    # 去掉 := by sorry / := sorry
                    parts_by = re.split(r'\s*:=\s*by\b', stmt, maxsplit=1)
                    parts_eq = re.split(r'\s*:=\s*sorry\s*$', stmt, maxsplit=1)
                    if len(parts_by) > 1:
                        stmt = parts_by[0].strip()
                    elif len(parts_eq) > 1:
                        stmt = parts_eq[0].strip()

                    problems.append(BenchmarkProblem(
                        problem_id=f"{prefix}_{pid}",
                        name=name,
                        theorem_statement=stmt,
                        difficulty=difficulty,
                        source=f"FATE-{difficulty[0].upper()}",
                        natural_language=informal,
                    ))
                logger.info(f"FATE: 从 {json_file} 加载了 {len(problems)} 道题 (difficulty={difficulty})")
                return problems
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"FATE JSON 解析错误 ({json_file}): {e}")

    # 回退: 解析 .lean 文件
    lean_dirs = [
        path / "FATEM", path / "FATEH", path / "FATEX",
        path / "src", path,
    ]
    for lean_dir in lean_dirs:
        if not lean_dir.is_dir():
            continue
        for lean_file in sorted(lean_dir.rglob("*.lean")):
            if "lakefile" in lean_file.name.lower():
                continue
            content = lean_file.read_text(encoding="utf-8", errors="ignore")
            for m in _THEOREM_RE.finditer(content):
                name = m.group(2)
                full_text = m.group(0).strip()
                parts_by = re.split(r'\s*:=\s*by\b', full_text, maxsplit=1)
                parts_eq = re.split(r'\s*:=\s*', full_text, maxsplit=1)
                if len(parts_by) > 1:
                    stmt = parts_by[0].strip()
                elif len(parts_eq) > 1:
                    stmt = parts_eq[0].strip()
                else:
                    stmt = full_text.strip()
                problems.append(BenchmarkProblem(
                    problem_id=f"{prefix}_{name}",
                    name=name,
                    theorem_statement=stmt,
                    difficulty=difficulty,
                    source=f"FATE-{difficulty[0].upper()}",
                ))

    logger.info(f"FATE: 从 {repo_path} 加载了 {len(problems)} 道题 (difficulty={difficulty})")
    return problems
