"""benchmarks/datasets/minif2f/loader.py — miniF2F (Lean 4) 加载器

支持 yangky11/miniF2F-lean4 仓库结构:
  MiniF2F/
    Test.lean       ← 244 道 test
    Valid.lean       ← 244 道 valid
"""
from __future__ import annotations
import re, json, logging
from pathlib import Path
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)

# miniF2F 的 Lean 4 文件中定理格式:
#   theorem aime_1983_p1 ... : ... := by sorry
_THEOREM_RE = re.compile(
    r'^(theorem|lemma)\s+(\S+)\s*([\s\S]*?)(?=\n(?:theorem|lemma|end|--|/-)\s|\Z)',
    re.MULTILINE)

def _difficulty_from_name(name: str) -> str:
    name_l = name.lower()
    if "imo" in name_l: return "competition"
    if "aime" in name_l: return "hard"
    if "amc" in name_l: return "medium"
    if "mathd" in name_l: return "easy"
    return "medium"

def _parse_lean_file(path: Path, split: str) -> list[BenchmarkProblem]:
    content = path.read_text(encoding="utf-8")
    problems = []
    for m in _THEOREM_RE.finditer(content):
        kind = m.group(1)
        name = m.group(2)
        full_text = m.group(0).strip()
        # 提取 statement (去掉 := by sorry 部分)
        stmt = re.split(r'\s*:=\s*by\b', full_text, maxsplit=1)[0].strip()
        if not stmt:
            stmt = re.split(r'\s*:=\s*', full_text, maxsplit=1)[0].strip()
        problems.append(BenchmarkProblem(
            problem_id=f"minif2f_{split}_{name}",
            name=name,
            theorem_statement=stmt,
            difficulty=_difficulty_from_name(name),
            source="miniF2F",
        ))
    return problems

def load(repo_path: str, split: str = "test") -> list[BenchmarkProblem]:
    """加载 miniF2F 数据集。"""
    path = Path(repo_path)
    if not path.exists():
        logger.warning(f"miniF2F 路径不存在: {repo_path}")
        return []

    # yangky11/miniF2F-lean4 结构: MiniF2F/Test.lean, MiniF2F/Valid.lean
    split_file_map = {
        "test": ["MiniF2F/Test.lean", "test.lean", "Test.lean"],
        "valid": ["MiniF2F/Valid.lean", "valid.lean", "Valid.lean"],
    }
    candidates = split_file_map.get(split, split_file_map["test"])

    for candidate in candidates:
        lean_file = path / candidate
        if lean_file.exists():
            problems = _parse_lean_file(lean_file, split)
            if problems:
                logger.info(f"miniF2F: 从 {lean_file} 加载了 {len(problems)} 道题")
                return problems

    # Per-file structure: MiniF2F/Test/*.lean or MiniF2F/Valid/*.lean
    split_dir_map = {
        "test": ["MiniF2F/Test", "Test", "test"],
        "valid": ["MiniF2F/Valid", "Valid", "valid"],
    }
    dir_candidates = split_dir_map.get(split, split_dir_map["test"])
    for dir_name in dir_candidates:
        split_dir = path / dir_name
        if split_dir.is_dir():
            all_problems = []
            for lean_file in sorted(split_dir.rglob("*.lean")):
                all_problems.extend(_parse_lean_file(lean_file, split))
            if all_problems:
                logger.info(f"miniF2F: 从 {split_dir} 加载了 {len(all_problems)} 道题")
                return all_problems

    # Fallback: 扫描所有 .lean 文件
    all_problems = []
    for lean_file in sorted(path.rglob("*.lean")):
        if lean_file.name.startswith("lake") or "lakefile" in lean_file.name.lower():
            continue
        all_problems.extend(_parse_lean_file(lean_file, split))

    logger.info(f"miniF2F: 从 {repo_path} 加载了 {len(all_problems)} 道题 (全部)")
    return all_problems
