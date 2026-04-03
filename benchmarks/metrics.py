"""benchmarks/metrics.py — 评测指标计算"""
from __future__ import annotations

def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k: return 1.0
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result

def compute_metrics(traces: list[dict]) -> dict:
    n = len(traces); solved = sum(1 for t in traces if t.get("solved"))
    return {"total": n, "solved": solved, "solve_rate": solved/n if n else 0,
            "pass@1": solved/n if n else 0}
