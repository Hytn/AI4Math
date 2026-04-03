"""benchmarks/eval_runner.py — 批量评测执行器"""
from __future__ import annotations
import logging
from pathlib import Path
from prover.models import BenchmarkProblem, ProofTrace, EvalResult

logger = logging.getLogger(__name__)

class EvalRunner:
    def __init__(self, orchestrator, output_dir="results"):
        self.orchestrator = orchestrator; self.output_dir = Path(output_dir)

    def run(self, problems: list[BenchmarkProblem], benchmark_name: str = "unknown",
            split: str = "test") -> EvalResult:
        traces, solved_count, total_tokens, total_attempts = [], 0, 0, 0
        for i, p in enumerate(problems, 1):
            logger.info(f"\n[{i}/{len(problems)}] {p.name}")
            trace = self.orchestrator.prove(p)
            traces.append(trace)
            trace.save(self.output_dir / "traces" / f"{p.problem_id}.json")
            if trace.solved: solved_count += 1
            total_tokens += trace.total_tokens; total_attempts += trace.total_attempts

        n = len(problems)
        result = EvalResult(benchmark=benchmark_name, split=split, total_problems=n,
                            solved=solved_count, solve_rate=solved_count/n if n else 0,
                            total_tokens=total_tokens, total_duration_ms=0,
                            avg_attempts=total_attempts/n if n else 0,
                            per_problem=[{"id": t.problem_id, "solved": t.solved, "attempts": t.total_attempts}
                                         for t in traces])
        result.save(self.output_dir / "evals" / f"eval_{benchmark_name}_{split}.json")
        logger.info(f"\n{'='*60}\n{result.summary()}\n{'='*60}")
        return result
