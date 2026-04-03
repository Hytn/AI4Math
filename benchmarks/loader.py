"""benchmarks/loader.py — 统一数据集加载入口"""
from __future__ import annotations
import logging
from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)

def load_benchmark(benchmark: str, split: str = "test", path: str = "") -> list[BenchmarkProblem]:
    b = benchmark.lower().replace("-", "").replace("_", "")
    if b == "builtin":
        from benchmarks.datasets.builtin.problems import BUILTIN_PROBLEMS
        return list(BUILTIN_PROBLEMS)
    elif b == "minif2f":
        from benchmarks.datasets.minif2f.loader import load
        return load(path or "data/miniF2F", split)
    elif b == "putnambench":
        from benchmarks.datasets.putnambench.loader import load
        return load(path or "data/PutnamBench", split)
    elif b in ("fatem", "fateh", "fatex", "fate"):
        from benchmarks.datasets.fate.loader import load
        suffix = b.replace("fate", "") or "m"
        return load(path or f"data/FATE-{suffix.upper()}", split)
    elif b == "proofnet":
        from benchmarks.datasets.proofnet.loader import load
        return load(path or "data/ProofNet", split)
    elif b == "formalmath":
        from benchmarks.datasets.formalmath.loader import load
        return load(path or "data/FormalMATH", split)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: builtin, miniF2F, PutnamBench, FATE-M/H/X, ProofNet, FormalMATH")
