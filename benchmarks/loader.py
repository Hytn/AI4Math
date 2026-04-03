"""
benchmarks/loader.py — 基准数据集加载器

支持从本地克隆的 miniF2F 仓库加载题目。
后续可扩展 PutnamBench、FATE-M 等。

miniF2F 目录结构 (Lean 4 版):
  miniF2F/
    lean/
      miniF2F/
        Test.lean      (或按题目分文件)
        Valid.lean

也支持从 JSON manifest 加载（更灵活）。
"""

from __future__ import annotations

import re
import json
import logging
from pathlib import Path
from typing import Optional

from core.models import BenchmarkProblem

logger = logging.getLogger(__name__)


# ── miniF2F 加载 ───────────────────────────────────────────────

def load_minif2f(
    repo_path: str,
    split: str = "test",
) -> list[BenchmarkProblem]:
    """
    从本地 miniF2F 仓库加载题目。

    Args:
        repo_path: miniF2F 仓库的本地路径
        split:     "test" 或 "valid"

    Returns:
        题目列表
    """
    repo = Path(repo_path)

    # 尝试多种目录结构
    candidates = [
        repo / "lean" / "miniF2F" / f"{split.capitalize()}.lean",
        repo / "lean4" / "miniF2F" / f"{split.capitalize()}.lean",
        repo / f"{split.capitalize()}.lean",
        repo / "Mathlib" / "miniF2F" / f"{split.capitalize()}.lean",
    ]

    # 也支持按题目分文件的结构
    split_dir_candidates = [
        repo / "lean" / "miniF2F" / split,
        repo / "lean4" / "miniF2F" / split,
        repo / split,
    ]

    # 尝试单文件加载
    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Loading miniF2F {split} from {candidate}")
            return _parse_lean_file(candidate, source="miniF2F", split=split)

    # 尝试目录加载
    for dir_candidate in split_dir_candidates:
        if dir_candidate.is_dir():
            logger.info(f"Loading miniF2F {split} from directory {dir_candidate}")
            problems = []
            for lean_file in sorted(dir_candidate.glob("*.lean")):
                problems.extend(_parse_lean_file(lean_file, source="miniF2F", split=split))
            return problems

    logger.error(f"Could not find miniF2F {split} data in {repo_path}")
    return []


def _parse_lean_file(
    path: Path,
    source: str = "miniF2F",
    split: str = "test",
) -> list[BenchmarkProblem]:
    """
    从单个 .lean 文件中提取所有 theorem 声明。

    解析策略：用正则匹配 `theorem` 关键字，提取到下一个 `:=` 或文件结尾。
    """
    content = path.read_text(encoding="utf-8")
    problems = []

    # 匹配 theorem 声明: theorem name ... :=
    # 这个正则比较保守，可能需要根据实际格式调整
    theorem_pattern = re.compile(
        r"^(theorem\s+\S+.*?)(?=\n(?:theorem|lemma|def|example|#|end|section|namespace)\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    for match in theorem_pattern.finditer(content):
        full_text = match.group(1).strip()

        # 提取定理名
        name_match = re.match(r"theorem\s+(\S+)", full_text)
        if not name_match:
            continue
        name = name_match.group(1)

        # 分离 statement 和 proof
        # 找到 `:= by` 或 `:=` 的位置
        assign_match = re.search(r"\s*:=\s*", full_text)
        if assign_match:
            theorem_statement = full_text[:assign_match.start()].strip()
        else:
            theorem_statement = full_text.strip()

        problems.append(BenchmarkProblem(
            problem_id=f"{source}_{split}_{name}",
            name=name,
            theorem_statement=theorem_statement,
            source=source,
        ))

    logger.info(f"Parsed {len(problems)} theorems from {path.name}")
    return problems


# ── JSON Manifest 加载 ─────────────────────────────────────────

def load_from_json(path: str) -> list[BenchmarkProblem]:
    """
    从 JSON 文件加载题目。适用于预处理好的题目集。

    JSON 格式:
    [
      {
        "problem_id": "...",
        "name": "...",
        "theorem_statement": "theorem xxx : ...",
        "difficulty": "easy",
        "source": "miniF2F",
        "natural_language": "Prove that ...",
        "tags": ["algebra", "inequality"]
      },
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    problems = []
    for item in data:
        problems.append(BenchmarkProblem(
            problem_id=item["problem_id"],
            name=item["name"],
            theorem_statement=item["theorem_statement"],
            difficulty=item.get("difficulty", "unknown"),
            source=item.get("source", ""),
            natural_language=item.get("natural_language", ""),
            tags=item.get("tags", []),
        ))

    logger.info(f"Loaded {len(problems)} problems from {path}")
    return problems


# ── 内置示例题目 (用于无外部数据时的冒烟测试) ──────────────────

BUILTIN_EXAMPLES = [
    BenchmarkProblem(
        problem_id="example_nat_add_comm",
        name="nat_add_comm",
        theorem_statement="theorem nat_add_comm (a b : Nat) : a + b = b + a",
        difficulty="easy",
        source="builtin",
        natural_language="Prove that addition of natural numbers is commutative.",
    ),
    BenchmarkProblem(
        problem_id="example_int_mul_comm",
        name="int_mul_comm",
        theorem_statement="theorem int_mul_comm (a b : Int) : a * b = b * a",
        difficulty="easy",
        source="builtin",
        natural_language="Prove that multiplication of integers is commutative.",
    ),
    BenchmarkProblem(
        problem_id="example_abs_nonneg",
        name="abs_nonneg",
        theorem_statement="theorem abs_nonneg_example (a : Int) : 0 ≤ |a|",
        difficulty="easy",
        source="builtin",
        natural_language="Prove that the absolute value of any integer is non-negative.",
    ),
    BenchmarkProblem(
        problem_id="example_sum_first_n",
        name="sum_first_n",
        theorem_statement=(
            "theorem sum_first_n (n : Nat) : "
            "2 * (Finset.range (n + 1)).sum id = n * (n + 1)"
        ),
        difficulty="medium",
        source="builtin",
        natural_language="Prove that 2 * (0 + 1 + ... + n) = n * (n + 1).",
    ),
    BenchmarkProblem(
        problem_id="example_amgm_two",
        name="amgm_two_vars",
        theorem_statement=(
            "theorem amgm_two_vars (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : "
            "a * b ≤ (a + b) ^ 2 / 4"
        ),
        difficulty="medium",
        source="builtin",
        natural_language="Prove the AM-GM inequality for two non-negative reals: ab ≤ ((a+b)/2)².",
    ),
]


def load_builtin_examples() -> list[BenchmarkProblem]:
    """加载内置示例题目"""
    return list(BUILTIN_EXAMPLES)


# ── 统一入口 ───────────────────────────────────────────────────

def load_benchmark(
    benchmark: str,
    split: str = "test",
    path: str = "",
) -> list[BenchmarkProblem]:
    """
    统一的基准加载入口。

    Args:
        benchmark: "miniF2F" / "builtin" / "json"
        split:     "test" / "valid"
        path:      仓库路径或 JSON 文件路径

    Returns:
        题目列表
    """
    if benchmark.lower() == "builtin":
        return load_builtin_examples()
    elif benchmark.lower() == "minif2f":
        if not path:
            logger.warning("miniF2F path not specified; using builtin examples")
            return load_builtin_examples()
        return load_minif2f(path, split)
    elif benchmark.lower() == "json":
        return load_from_json(path)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
