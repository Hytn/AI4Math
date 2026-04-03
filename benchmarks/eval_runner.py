"""
benchmarks/eval_runner.py — 批量评测执行器

职责：对一组题目批量运行 orchestrator，收集结果，生成报告。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Callable

from core.models import BenchmarkProblem, ProofTrace, EvalResult
from core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class EvalRunner:
    """批量评测执行器"""

    def __init__(
        self,
        orchestrator: Orchestrator,
        output_dir: str = "results",
        on_problem_done: Optional[Callable[[BenchmarkProblem, ProofTrace], None]] = None,
    ):
        self.orchestrator = orchestrator
        self.output_dir = Path(output_dir)
        self.on_problem_done = on_problem_done

    def run(
        self,
        problems: list[BenchmarkProblem],
        benchmark_name: str = "unknown",
        split: str = "test",
    ) -> EvalResult:
        """
        对一组题目运行评测。

        Args:
            problems:       题目列表
            benchmark_name: 基准名称 (用于报告)
            split:          数据集分割

        Returns:
            EvalResult 汇总
        """
        traces: list[ProofTrace] = []
        solved_count = 0
        total_tokens = 0
        total_duration = 0
        total_attempts = 0

        for i, problem in enumerate(problems, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{len(problems)}] {problem.name}")
            logger.info(f"{'='*60}")

            trace = self.orchestrator.prove(problem)
            traces.append(trace)

            # 保存单题轨迹
            trace_path = self.output_dir / "traces" / f"{problem.problem_id}.json"
            trace.save(trace_path)

            if trace.solved:
                solved_count += 1
            total_tokens += trace.total_tokens
            total_duration += trace.total_duration_ms
            total_attempts += trace.total_attempts

            if self.on_problem_done:
                self.on_problem_done(problem, trace)

        # 汇总
        n = len(problems)
        result = EvalResult(
            benchmark=benchmark_name,
            split=split,
            total_problems=n,
            solved=solved_count,
            solve_rate=solved_count / n if n > 0 else 0.0,
            total_tokens=total_tokens,
            total_duration_ms=total_duration,
            avg_attempts_per_problem=total_attempts / n if n > 0 else 0.0,
            config_snapshot=traces[0].config_snapshot if traces else {},
            per_problem=[
                {
                    "problem_id": t.problem_id,
                    "name": t.problem_name,
                    "solved": t.solved,
                    "attempts": t.total_attempts,
                    "tokens": t.total_tokens,
                    "duration_ms": t.total_duration_ms,
                }
                for t in traces
            ],
        )

        # 保存汇总报告
        result.save(self.output_dir / f"eval_{benchmark_name}_{split}.json")

        logger.info(f"\n{'='*60}")
        logger.info(f"EVAL COMPLETE: {result.summary()}")
        logger.info(f"{'='*60}")

        return result
