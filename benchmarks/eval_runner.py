"""benchmarks/eval_runner.py — 批量评测执行器"""
from __future__ import annotations
import time
import logging
from collections import defaultdict
from pathlib import Path
from prover.models import BenchmarkProblem, ProofTrace, EvalResult

logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(self, orchestrator, output_dir="results"):
        self.orchestrator = orchestrator
        self.output_dir = Path(output_dir)

    def run(self, problems: list[BenchmarkProblem], benchmark_name: str = "unknown",
            split: str = "test") -> EvalResult:
        traces = []
        solved_count = 0
        total_tokens = 0
        total_attempts = 0
        total_repair_rounds = 0
        error_dist: dict[str, int] = defaultdict(int)
        per_difficulty: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "solved": 0, "tokens": 0})
        solved_token_counts = []

        eval_start = time.time()

        for i, p in enumerate(problems, 1):
            logger.info(f"\n[{i}/{len(problems)}] {p.name} (difficulty={p.difficulty})")
            trace = self.orchestrator.prove(p)
            traces.append(trace)
            trace.save(self.output_dir / "traces" / f"{p.problem_id}.json")

            if trace.solved:
                solved_count += 1
                solved_token_counts.append(trace.total_tokens)

            total_tokens += trace.total_tokens
            total_attempts += trace.total_attempts

            # Aggregate error distribution
            for cat, cnt in trace.error_distribution.items():
                error_dist[cat] += cnt

            # Per-difficulty tracking
            diff = p.difficulty or "unknown"
            per_difficulty[diff]["total"] += 1
            if trace.solved:
                per_difficulty[diff]["solved"] += 1
            per_difficulty[diff]["tokens"] += trace.total_tokens

            # Track repair rounds
            for attempt in trace.attempts:
                if hasattr(attempt, 'repair_rounds'):
                    total_repair_rounds += attempt.repair_rounds

        total_duration_ms = int((time.time() - eval_start) * 1000)
        n = len(problems)

        # Compute per-difficulty solve rates
        per_diff_final = {}
        for diff, stats in per_difficulty.items():
            per_diff_final[diff] = {
                "total": stats["total"],
                "solved": stats["solved"],
                "solve_rate": stats["solved"] / stats["total"] if stats["total"] else 0,
                "avg_tokens": stats["tokens"] / stats["total"] if stats["total"] else 0,
            }

        # Median solve tokens
        median_tokens = 0
        if solved_token_counts:
            sorted_tokens = sorted(solved_token_counts)
            mid = len(sorted_tokens) // 2
            median_tokens = sorted_tokens[mid]

        result = EvalResult(
            benchmark=benchmark_name,
            split=split,
            total_problems=n,
            solved=solved_count,
            solve_rate=solved_count / n if n else 0,
            total_tokens=total_tokens,
            total_duration_ms=total_duration_ms,
            avg_attempts=total_attempts / n if n else 0,
            per_problem=[
                {"id": t.problem_id, "solved": t.solved,
                 "attempts": t.total_attempts, "tokens": t.total_tokens}
                for t in traces
            ],
            error_distribution=dict(error_dist),
            per_difficulty=per_diff_final,
            avg_repair_rounds=total_repair_rounds / max(1, total_attempts),
            median_solve_tokens=median_tokens,
        )

        result.save(self.output_dir / "evals" / f"eval_{benchmark_name}_{split}.json")

        # Log comprehensive summary
        logger.info(f"\n{'='*60}")
        logger.info(result.summary())
        if per_diff_final:
            logger.info("Per-difficulty breakdown:")
            for diff, stats in sorted(per_diff_final.items()):
                logger.info(
                    f"  {diff}: {stats['solved']}/{stats['total']} "
                    f"({stats['solve_rate']:.1%})")
        if error_dist:
            logger.info("Error distribution:")
            for cat, cnt in sorted(error_dist.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"  {cat}: {cnt}")
        logger.info(f"{'='*60}")

        return result
