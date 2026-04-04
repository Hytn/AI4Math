"""benchmarks/loader.py — 统一数据集加载入口

真实基准:
  builtin      — 内置冒烟测试 (5 题)
  minif2f      — miniF2F (488 题, yangky11/miniF2F-lean4)
  putnambench  — PutnamBench (672 题, trishullab/PutnamBench)
  proofnet     — ProofNet (371 题, rahul3613/ProofNet-lean4)
  formalmath   — FormalMATH (5560 题, Sphere-AI-Lab/FormalMATH-Bench)
"""
from __future__ import annotations
import logging
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)

# 默认数据路径
_DEFAULT_PATHS = {
    "builtin":     "",
    "minif2f":     "data/miniF2F",
    "putnambench": "data/PutnamBench",
    "proofnet":    "data/ProofNet",
    "formalmath":  "data/FormalMATH",
}


def load_benchmark(benchmark: str, split: str = "test",
                   path: str = "", limit: int = 0) -> list[BenchmarkProblem]:
    """加载指定基准数据集。

    Args:
        benchmark: 数据集名称
        split: 数据集切分 (test / valid)
        path: 数据集路径 (空则用默认路径)
        limit: 最多加载几题 (0=全部)
    """
    b = benchmark.lower().replace("-", "").replace("_", "")

    if b == "builtin":
        from benchmarks.datasets.builtin.problems import BUILTIN_PROBLEMS
        problems = list(BUILTIN_PROBLEMS)
    elif b == "minif2f":
        from benchmarks.datasets.minif2f.loader import load
        data_path = path or _DEFAULT_PATHS["minif2f"]
        problems = load(data_path, split)
        if not problems:
            logger.error(
                f"miniF2F: 未找到数据。请先下载:\n"
                f"  git clone https://github.com/yangky11/miniF2F-lean4 {data_path}\n"
                f"或指定 --path 参数。")
    elif b == "putnambench" or b == "putnam":
        from benchmarks.datasets.putnambench.loader import load
        data_path = path or _DEFAULT_PATHS["putnambench"]
        problems = load(data_path, split)
        if not problems:
            logger.error(
                f"PutnamBench: 未找到数据。请先下载:\n"
                f"  git clone https://github.com/trishullab/PutnamBench {data_path}\n"
                f"或指定 --path 参数。")
    elif b == "proofnet":
        from benchmarks.datasets.proofnet.loader import load
        data_path = path or _DEFAULT_PATHS["proofnet"]
        problems = load(data_path, split)
        if not problems:
            logger.error(
                f"ProofNet: 未找到数据。请先下载:\n"
                f"  git clone https://github.com/rahul3613/ProofNet-lean4 {data_path}\n"
                f"或指定 --path 参数。")
    elif b == "formalmath":
        from benchmarks.datasets.formalmath.loader import load
        data_path = path or _DEFAULT_PATHS["formalmath"]
        problems = load(data_path, split)
        if not problems:
            logger.error(
                f"FormalMATH: 未找到数据。请先下载:\n"
                f"  git clone https://github.com/Sphere-AI-Lab/FormalMATH-Bench {data_path}\n"
                f"或指定 --path 参数。")
    else:
        available = list(_DEFAULT_PATHS.keys())
        raise ValueError(f"未知数据集: {benchmark}。可用: {available}")

    if limit > 0:
        problems = problems[:limit]

    logger.info(f"[{benchmark}] 加载 {len(problems)} 道题 (split={split})")
    return problems


def list_benchmarks() -> dict:
    """列出所有可用数据集。"""
    return dict(_DEFAULT_PATHS)
