"""benchmarks/metrics.py — 评测指标计算

支持: pass@k, solve_rate, token_efficiency, time_efficiency, error_distribution.
"""
from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass, field


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric (unbiased estimator).

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value for pass@k

    Returns:
        Probability that at least one of k samples is correct.
    """
    if n < k:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def compute_metrics(traces: list[dict],
                    k_values: list[int] = None) -> dict:
    """Compute comprehensive evaluation metrics from proof traces.

    Args:
        traces: List of trace dicts with 'solved', 'total_attempts', 'total_tokens', etc.
        k_values: Values of k for pass@k computation.

    Returns:
        Dict with all computed metrics.
    """
    k_values = k_values or [1, 5, 10]
    n = len(traces)
    if n == 0:
        return {"total": 0}

    solved = sum(1 for t in traces if t.get("solved"))
    solve_rate = solved / n

    # pass@k
    pass_at = {}
    for k in k_values:
        scores = []
        for t in traces:
            total_samples = t.get("total_attempts", 1)
            correct = 1 if t.get("solved") else 0
            scores.append(pass_at_k(total_samples, correct, min(k, total_samples)))
        pass_at[f"pass@{k}"] = sum(scores) / len(scores) if scores else 0

    # Token efficiency
    total_tokens = sum(t.get("total_tokens", 0) for t in traces)
    solved_tokens = sum(t.get("total_tokens", 0) for t in traces if t.get("solved"))
    avg_tokens_per_problem = total_tokens / n if n else 0
    avg_tokens_per_solved = solved_tokens / solved if solved else 0

    # Attempt efficiency
    total_attempts = sum(t.get("total_attempts", 0) for t in traces)
    avg_attempts = total_attempts / n if n else 0

    # Error distribution
    error_cats = Counter()
    for t in traces:
        for a in t.get("attempts", []):
            for e in a.get("lean_errors", []):
                cat = e.get("category", "other")
                if isinstance(cat, str):
                    error_cats[cat] += 1

    # Difficulty breakdown
    difficulty_stats = {}
    for t in traces:
        diff = t.get("difficulty", "unknown")
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "solved": 0}
        difficulty_stats[diff]["total"] += 1
        if t.get("solved"):
            difficulty_stats[diff]["solved"] += 1

    return {
        "total": n,
        "solved": solved,
        "solve_rate": round(solve_rate, 4),
        **pass_at,
        "total_tokens": total_tokens,
        "avg_tokens_per_problem": round(avg_tokens_per_problem),
        "avg_tokens_per_solved": round(avg_tokens_per_solved),
        "total_attempts": total_attempts,
        "avg_attempts": round(avg_attempts, 2),
        "error_distribution": dict(error_cats.most_common(10)),
        "difficulty_breakdown": difficulty_stats,
    }


@dataclass
class MetricsSummary:
    """Human-readable metrics summary."""
    benchmark: str
    metrics: dict = field(default_factory=dict)

    def to_table(self) -> str:
        m = self.metrics
        lines = [
            f"{'='*50}",
            f"  Benchmark: {self.benchmark}",
            f"{'='*50}",
            f"  Solved:     {m.get('solved', 0)}/{m.get('total', 0)} ({m.get('solve_rate', 0):.1%})",
        ]
        for k in [1, 5, 10]:
            key = f"pass@{k}"
            if key in m:
                lines.append(f"  pass@{k}:    {m[key]:.3f}")
        lines.extend([
            f"  Avg attempts: {m.get('avg_attempts', 0):.1f}",
            f"  Total tokens: {m.get('total_tokens', 0):,}",
            f"{'='*50}",
        ])
        return "\n".join(lines)
