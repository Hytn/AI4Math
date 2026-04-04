#!/usr/bin/env python3
"""run_eval.py — 真实基准评测入口

Usage:
    python run_eval.py --benchmark minif2f --provider mock
    python run_eval.py --benchmark putnambench --provider anthropic --limit 10
    python run_eval.py --benchmark all --provider mock
"""
import argparse, json, logging, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.loader import load_benchmark, list_benchmarks
from benchmarks.metrics import compute_metrics, MetricsSummary
from prover.models import BenchmarkProblem, ProofTrace, ProofAttempt, AttemptStatus
from agent.brain.claude_provider import create_provider
from agent.brain.prompt_builder import build_prompt
from agent.brain.response_parser import extract_lean_code
from agent.brain.roles import AgentRole, ROLE_PROMPTS
from prover.verifier.sorry_detector import detect_sorry
from prover.codegen.code_formatter import format_lean_code
from prover.premise.selector import PremiseSelector
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def prove_single(problem: BenchmarkProblem, llm, premise_selector,
                 max_samples: int = 8, lean_env=None,
                 lean_mode: str = "skip") -> ProofTrace:
    """对单道题进行证明尝试。"""
    trace = ProofTrace(
        problem_id=problem.problem_id,
        problem_name=problem.name,
        theorem_statement=problem.theorem_statement,
        natural_language=problem.natural_language,
    )

    # 检索相关前提
    premises = premise_selector.retrieve(problem.theorem_statement, top_k=10)
    premise_strs = [f"{r['name']}: {r['statement']}" for r in premises]

    for attempt_idx in range(max_samples):
        attempt = ProofAttempt(attempt_number=attempt_idx + 1)
        t0 = time.time()

        try:
            # 构建 prompt
            prompt = build_prompt(
                theorem_statement=problem.theorem_statement,
                premises=premise_strs[:10],
            )

            # 调用 LLM 生成证明
            temp = 0.3 + (attempt_idx * 0.1)  # 逐步提高 temperature
            resp = llm.generate(
                system=ROLE_PROMPTS[AgentRole.PROOF_GENERATOR],
                user=prompt,
                temperature=min(temp, 1.0),
            )
            proof = extract_lean_code(resp.content)
            proof = format_lean_code(proof) if proof.strip() else ""

            attempt.generated_proof = proof
            attempt.llm_model = resp.model
            attempt.llm_tokens_in = resp.tokens_in
            attempt.llm_tokens_out = resp.tokens_out
            attempt.llm_latency_ms = resp.latency_ms

        except Exception as e:
            attempt.lean_result = AttemptStatus.LLM_ERROR
            attempt.lean_stderr = str(e)
            trace.add_attempt(attempt)
            continue

        if not proof.strip():
            attempt.lean_result = AttemptStatus.LLM_ERROR
            attempt.lean_stderr = "Empty proof"
            trace.add_attempt(attempt)
            continue

        # Sorry 检测
        sorry_report = detect_sorry(proof)
        if sorry_report.has_sorry:
            attempt.lean_result = AttemptStatus.LEAN_ERROR
            attempt.lean_stderr = f"Proof contains sorry ({len(sorry_report.locations)} locations)"
            attempt.lean_check_ms = int((time.time() - t0) * 1000)
            trace.add_attempt(attempt)
            continue

        # Lean4 验证
        if lean_mode == "real" and lean_env:
            try:
                from prover.verifier.lean_checker import LeanChecker
                checker = LeanChecker(lean_env)
                status, errors, stderr, check_ms = checker.check(
                    problem.theorem_statement, proof)
                attempt.lean_result = status
                attempt.lean_errors = errors
                attempt.lean_stderr = stderr
                attempt.lean_check_ms = check_ms
            except Exception as e:
                attempt.lean_result = AttemptStatus.LEAN_ERROR
                attempt.lean_stderr = str(e)
        else:
            # Skip 模式: 标记为需要后续验证
            attempt.lean_result = AttemptStatus.LEAN_ERROR
            attempt.lean_stderr = "Lean verification skipped (--lean-mode=skip)"
            attempt.lean_check_ms = 0

        trace.add_attempt(attempt)

        # 如果验证通过, 提前退出
        if attempt.lean_result == AttemptStatus.SUCCESS:
            break

    return trace


def main():
    parser = argparse.ArgumentParser(description="AI4Math 真实基准评测")
    parser.add_argument("--benchmark", default="builtin",
                        help="数据集名: builtin/minif2f/putnambench/proofnet/formalmath/all")
    parser.add_argument("--split", default="test", help="数据集切分: test/valid")
    parser.add_argument("--provider", default="mock", help="LLM: mock/anthropic")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--max-samples", type=int, default=8, help="每题最大尝试次数")
    parser.add_argument("--limit", type=int, default=0, help="每个 benchmark 最多评几题")
    parser.add_argument("--lean-mode", default="skip",
                        help="Lean 验证: real (需 lean4) / skip")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    # 初始化
    llm = create_provider({"provider": args.provider, "model": args.model,
                            "api_key": os.environ.get("ANTHROPIC_API_KEY", "")})
    premise_selector = PremiseSelector({"mode": "hybrid"})
    lean_env = None
    if args.lean_mode == "real":
        try:
            from agent.executor.lean_env import LeanEnvironment
            lean_env = LeanEnvironment(mode="local")
        except Exception as e:
            logger.warning(f"无法初始化 Lean 环境: {e}, 回退到 skip 模式")
            args.lean_mode = "skip"

    # 确定要跑的 benchmarks
    if args.benchmark == "all":
        bench_names = ["builtin", "minif2f", "putnambench", "proofnet", "formalmath"]
    else:
        bench_names = [args.benchmark]

    all_results = {}
    for bench_name in bench_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"  评测: {bench_name} (split={args.split})")
        logger.info(f"{'='*60}\n")

        try:
            problems = load_benchmark(bench_name, args.split, limit=args.limit)
        except Exception as e:
            logger.error(f"  加载 {bench_name} 失败: {e}")
            continue

        if not problems:
            logger.warning(f"  {bench_name}: 未找到题目 (路径是否正确?)")
            continue

        logger.info(f"  共 {len(problems)} 道题\n")

        traces = []
        for i, problem in enumerate(problems, 1):
            logger.info(f"  [{i}/{len(problems)}] {problem.name}")
            trace = prove_single(problem, llm, premise_selector,
                                 max_samples=args.max_samples,
                                 lean_env=lean_env, lean_mode=args.lean_mode)
            traces.append(trace)

            status = "✓ 通过" if trace.solved else f"✗ ({trace.total_attempts} 次)"
            logger.info(f"           {status}")

            # 保存每道题的 trace
            out_dir = Path(args.output_dir) / "traces" / bench_name
            trace.save(out_dir / f"{problem.problem_id}.json")

        # 计算指标
        trace_dicts = [t.to_dict() for t in traces]
        metrics = compute_metrics(trace_dicts, k_values=[1, 5, 10])
        summary = MetricsSummary(bench_name, metrics)

        logger.info(f"\n{summary.to_table()}")

        # 保存汇总结果
        eval_dir = Path(args.output_dir) / "evals"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_file = eval_dir / f"eval_{bench_name}_{args.split}.json"
        with open(eval_file, "w") as f:
            json.dump({
                "benchmark": bench_name,
                "split": args.split,
                "provider": args.provider,
                "model": llm.model_name,
                "max_samples": args.max_samples,
                "lean_mode": args.lean_mode,
                "metrics": metrics,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"  结果 → {eval_file}")

        all_results[bench_name] = metrics

    # 全局汇总
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info(f"  全局汇总")
        logger.info(f"{'='*60}\n")
        logger.info(f"  {'基准':<15} {'题数':>6} {'通过':>6} {'通过率':>8} {'pass@1':>8}")
        logger.info(f"  {'─'*50}")
        for name, m in all_results.items():
            logger.info(f"  {name:<15} {m['total']:>6} {m['solved']:>6} "
                        f"{m['solve_rate']:>7.1%} {m.get('pass@1', 0):>7.3f}")


if __name__ == "__main__":
    main()
