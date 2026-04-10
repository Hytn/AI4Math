#!/usr/bin/env python3
"""run_eval.py — 真实基准评测入口 (v3)

v3 改进 (在 v2 基础上):
  6. 知识闭环: 每次尝试结果写入知识库, 下轮 prompt 注入积累知识
  7. 修复链路: 失败后的下一轮 prompt 包含错误诊断和修复建议
  8. 多角色模式: --multi-role 启用 Generator → Repair → Decomposer 链
  9. pass@k 修复: 默认跑满全部 samples, --early-stop 才提前退出
  10. 验证标注: lean_mode=skip 时结果明确标注 [unverified]

v2 改进:
  1. 增量保存: 每道题完成后立即写入磁盘, 中途崩溃不丢失
  2. 断点续跑: 自动检测已完成的 trace, 跳过已评测的题目
  3. pass@k 正确统计: 不在首次成功时中断, 记录 correct_count
  4. 支持 --resume 参数显式启用断点续跑
  5. 总结报告包含 pass@1/5/10/32

Usage:
    python run_eval.py --benchmark minif2f --provider mock
    python run_eval.py --benchmark minif2f --provider anthropic --resume
    python run_eval.py --benchmark all --provider anthropic --limit 10
    python run_eval.py --benchmark minif2f --provider anthropic --multi-role
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.loader import load_benchmark, list_benchmarks
from benchmarks.metrics import compute_metrics, MetricsSummary
from prover.models import (
    BenchmarkProblem, ProofTrace, ProofAttempt, AttemptStatus,
)
from agent.brain.claude_provider import create_provider
from common.prompt_builder import build_prompt
from common.response_parser import extract_lean_code
from common.roles import AgentRole, ROLE_PROMPTS
from prover.verifier.sorry_detector import detect_sorry
from prover.codegen.code_formatter import format_lean_code
from prover.premise.selector import PremiseSelector
from prover.repair.repair_generator import RepairGenerator
from knowledge.store import UnifiedKnowledgeStore
from knowledge.reader import KnowledgeReader
from knowledge.writer import KnowledgeWriter
from engine.proof_context_store import StepDetail
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def prove_single(problem: BenchmarkProblem, llm, premise_selector,
                 max_samples: int = 8, lean_env=None,
                 lean_mode: str = "skip",
                 temperature: float = 0.8,
                 temp_mode: str = "fixed",
                 knowledge_reader: KnowledgeReader = None,
                 knowledge_writer: KnowledgeWriter = None,
                 multi_role: bool = False,
                 early_stop: bool = False) -> ProofTrace:
    """对单道题进行证明尝试。

    v3 改进:
      - 知识闭环: 每次尝试后写入知识库, 下轮注入积累知识 (Fix #1)
      - 错误反馈: 失败后将错误诊断注入下轮 prompt (Fix #9)
      - 多角色: Generator → Repair → Generator 交替 (Fix #3)
      - pass@k: 默认跑满全部 samples, 仅 early_stop=True 时提前退出 (Fix #4)
      - 验证标注: lean_mode=skip 时 stderr 明确标注 [unverified] (Fix #5)

    温度调度模式:
      - "fixed": 所有样本使用相同温度 (默认 0.8), 满足 pass@k i.i.d. 假设
      - "escalating": 逐步提高温度 (0.3 → 1.0) 增加多样性
    """
    trace = ProofTrace(
        problem_id=problem.problem_id,
        problem_name=problem.name,
        theorem_statement=problem.theorem_statement,
        natural_language=problem.natural_language,
    )

    # 检索相关前提
    premises = premise_selector.retrieve(problem.theorem_statement, top_k=10)
    premise_strs = [f"{r['name']}: {r['statement']}" for r in premises]

    # Fix #1: 从知识库注入积累知识
    knowledge_context = ""
    if knowledge_reader:
        try:
            knowledge_context = asyncio.get_event_loop().run_until_complete(
                knowledge_reader.render_for_prompt(
                    goal=problem.theorem_statement,
                    theorem=problem.theorem_statement,
                    max_chars=1200))
        except RuntimeError:
            # No event loop running, create one
            knowledge_context = asyncio.run(
                knowledge_reader.render_for_prompt(
                    goal=problem.theorem_statement,
                    theorem=problem.theorem_statement,
                    max_chars=1200))
        except Exception as e:
            logger.debug(f"  知识注入跳过: {e}")

    # Fix #9: 跟踪上一次失败的错误信息, 用于修复链路
    last_failed_proof = ""
    last_error_analysis = ""
    error_history_parts = []  # 积累所有历史错误摘要

    # Fix #3: 多角色修复器
    repair_gen = RepairGenerator(llm) if multi_role else None

    for attempt_idx in range(max_samples):
        attempt = ProofAttempt(attempt_number=attempt_idx + 1)
        t0 = time.time()

        try:
            # Fix #3: 多角色 — 偶数轮用 Generator, 失败后奇数轮用 Repair
            use_repair = (multi_role and last_failed_proof
                          and attempt_idx % 2 == 1)

            if use_repair:
                # Repair Agent 接力
                repairs = repair_gen.generate_repair(
                    theorem=problem.theorem_statement,
                    failed_proof=last_failed_proof,
                    error_analysis=last_error_analysis,
                    max_repairs=1,
                    temperature=0.4)
                proof = repairs[0] if repairs else ""
                proof = format_lean_code(proof) if proof.strip() else ""
                attempt.llm_model = llm.model_name
                attempt.repair_rounds = 1
                attempt.prompt_summary = "repair_agent"
            else:
                # Fix #9: 如有上轮错误, 构建包含错误诊断的 retry prompt
                error_history_str = ""
                if error_history_parts:
                    # 只保留最近 3 条历史
                    recent = error_history_parts[-3:]
                    error_history_str = "\n".join(recent)

                # Fix #1: 将知识库知识附加到 premises 中
                effective_premises = list(premise_strs[:10])
                if knowledge_context:
                    effective_premises.append(
                        f"[Knowledge from past proofs]\n{knowledge_context}")

                prompt = build_prompt(
                    theorem_statement=problem.theorem_statement,
                    premises=effective_premises,
                    error_analysis=last_error_analysis if last_failed_proof else "",
                    error_history=error_history_str,
                    failed_proof=last_failed_proof,
                    attempt_number=attempt_idx + 1,
                )

                # 温度调度
                if temp_mode == "escalating":
                    temp = 0.3 + (attempt_idx * 0.1)
                    temp = min(temp, 1.0)
                else:
                    temp = temperature

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

        # Sorry 检测 (快速预过滤, 无需调用 Lean)
        sorry_report = detect_sorry(proof)
        if sorry_report.has_sorry:
            attempt.lean_result = AttemptStatus.LEAN_ERROR
            attempt.lean_stderr = (
                f"Proof contains sorry ({len(sorry_report.locations)} locations)")
            attempt.lean_check_ms = int((time.time() - t0) * 1000)
            # Fix #9: 记录错误用于下轮修复
            last_failed_proof = proof
            last_error_analysis = (
                "Proof contains `sorry` — it is incomplete. "
                "Replace all sorry placeholders with actual proof terms.")
            error_history_parts.append(
                f"Attempt #{attempt_idx+1}: sorry detected")
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
            # Fix #5: 明确标注未验证状态
            attempt.lean_result = AttemptStatus.LEAN_ERROR
            attempt.lean_stderr = (
                "[unverified] Lean verification skipped (--lean-mode=skip). "
                "This result has NOT been verified by Lean4.")
            attempt.lean_check_ms = 0

        trace.add_attempt(attempt)

        # Fix #9: 记录错误反馈用于下一轮
        if attempt.lean_result != AttemptStatus.SUCCESS:
            last_failed_proof = proof
            # 构建错误分析文本
            error_parts = []
            if attempt.lean_stderr and "[unverified]" not in attempt.lean_stderr:
                error_parts.append(f"Lean error: {attempt.lean_stderr[:300]}")
            for e in attempt.lean_errors[:3]:
                error_parts.append(
                    f"- [{e.category.value}] {e.message[:150]}")
                if e.expected_type:
                    error_parts.append(f"  expected: {e.expected_type}")
                if e.actual_type:
                    error_parts.append(f"  actual: {e.actual_type}")
            last_error_analysis = "\n".join(error_parts) if error_parts else ""
            if last_error_analysis:
                error_history_parts.append(
                    f"Attempt #{attempt_idx+1}: {attempt.lean_errors[0].category.value if attempt.lean_errors else 'error'}")
        else:
            # 成功时清空错误状态
            last_failed_proof = ""
            last_error_analysis = ""

        # Fix #1: 将结果写入知识库 (无论成功或失败)
        if knowledge_writer:
            try:
                step = StepDetail(
                    step_index=attempt_idx,
                    tactic=proof[:200],
                    env_id_before=0,
                    env_id_after=0 if attempt.lean_result == AttemptStatus.SUCCESS else -1,
                    goals_before=[problem.theorem_statement],
                    goals_after=[] if attempt.lean_result == AttemptStatus.SUCCESS else [problem.theorem_statement],
                    error_message=attempt.lean_stderr if attempt.lean_result != AttemptStatus.SUCCESS else "",
                    error_category=attempt.lean_errors[0].category.value if attempt.lean_errors else "",
                    elapsed_ms=attempt.lean_check_ms,
                )
                asyncio.get_event_loop().run_until_complete(
                    knowledge_writer.ingest_step(
                        step, theorem=problem.theorem_statement))
            except Exception as e:
                logger.debug(f"  知识写入跳过: {e}")

        # Fix #4: 默认跑满全部 samples, 仅 early_stop=True 时提前退出
        # 这保证 pass@k 的无偏估计 (公式需要精确的 n 和 c)
        if early_stop and trace.correct_count >= 3:
            break

    return trace


def load_existing_traces(trace_dir: Path) -> dict[str, dict]:
    """Load all existing trace files from a directory.

    Returns: {problem_id: trace_dict}
    """
    existing = {}
    if not trace_dir.exists():
        return existing

    for trace_file in trace_dir.glob("*.json"):
        try:
            with open(trace_file) as f:
                data = json.load(f)
            pid = data.get("problem_id", trace_file.stem)
            existing[pid] = data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"  跳过损坏的 trace: {trace_file}: {e}")
    return existing


def main():
    parser = argparse.ArgumentParser(description="AI4Math 真实基准评测 (v3)")
    parser.add_argument("--benchmark", default="builtin",
                        help="数据集名: builtin/minif2f/putnambench/proofnet/all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--provider", default="mock", help="LLM: mock/anthropic")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lean-mode", default="skip",
                        help="Lean 验证: real / skip")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑: 跳过已有 trace 的题目")
    # Fix #4: 反转默认行为 — 默认跑满, 需显式 --early-stop 才提前退出
    parser.add_argument("--early-stop", action="store_true",
                        help="在 3 次成功后提前退出 (节省 API 调用, 但 pass@k 估计有偏)")
    # Fix #3: 多角色模式
    parser.add_argument("--multi-role", action="store_true",
                        help="启用多角色: Generator + Repair Agent 交替")
    # Fix #1: 知识系统开关
    parser.add_argument("--no-knowledge", action="store_true",
                        help="禁用知识系统 (不沉淀/不注入)")
    # 向后兼容旧参数
    parser.add_argument("--no-early-stop", action="store_true",
                        help="(已废弃, 现在默认不提前退出)")
    args = parser.parse_args()

    if args.no_early_stop:
        logger.info("  注意: --no-early-stop 已废弃, v3 默认跑满全部 samples")

    # 初始化 LLM
    llm = create_provider({
        "provider": args.provider,
        "model": args.model,
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    })
    premise_selector = PremiseSelector({"mode": "hybrid"})

    # Fix #1: 初始化知识系统
    knowledge_reader = None
    knowledge_writer = None
    if not args.no_knowledge:
        try:
            knowledge_dir = Path(args.output_dir) / "knowledge"
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(knowledge_dir / "knowledge.db")
            knowledge_store = UnifiedKnowledgeStore(db_path)
            knowledge_reader = KnowledgeReader(knowledge_store)
            knowledge_writer = KnowledgeWriter(knowledge_store)
            logger.info(f"  知识系统已启用: {db_path}")
        except Exception as e:
            logger.warning(f"  知识系统初始化失败: {e}, 继续不带知识系统")

    # 初始化 Lean 环境
    lean_env = None
    if args.lean_mode == "real":
        try:
            from agent.executor.lean_env import LeanEnvironment
            lean_env = LeanEnvironment(mode="local")
        except Exception as e:
            logger.warning(f"无法初始化 Lean 环境: {e}, 回退到 skip 模式")
            args.lean_mode = "skip"

    # Fix #5: 未验证模式警告
    if args.lean_mode == "skip":
        logger.info("  ⚠ Lean 验证已跳过 (--lean-mode=skip): "
                     "所有 pass@k 指标标记为 [unverified]")

    # 确定 benchmarks
    if args.benchmark == "all":
        bench_names = ["builtin", "minif2f", "putnambench", "proofnet"]
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
            logger.warning(f"  {bench_name}: 未找到题目")
            continue

        # 断点续跑: 加载已有 traces
        trace_dir = Path(args.output_dir) / "traces" / bench_name
        existing_traces = {}
        if args.resume:
            existing_traces = load_existing_traces(trace_dir)
            if existing_traces:
                logger.info(f"  断点续跑: 已有 {len(existing_traces)} 道题的结果")

        logger.info(f"  共 {len(problems)} 道题, "
                     f"待评测 {len(problems) - len(existing_traces)} 道\n")

        trace_dicts = []
        skipped = 0
        t_start = time.time()

        for i, problem in enumerate(problems, 1):
            # 断点续跑: 跳过已完成的题目
            if problem.problem_id in existing_traces:
                trace_dicts.append(existing_traces[problem.problem_id])
                skipped += 1
                continue

            logger.info(f"  [{i}/{len(problems)}] {problem.name}")

            trace = prove_single(
                problem, llm, premise_selector,
                max_samples=args.max_samples,
                lean_env=lean_env, lean_mode=args.lean_mode,
                knowledge_reader=knowledge_reader,
                knowledge_writer=knowledge_writer,
                multi_role=args.multi_role,
                early_stop=args.early_stop,
            )

            status = ("✓ 通过" if trace.solved
                      else f"✗ ({trace.total_attempts} 次)")
            if trace.correct_count > 1:
                status += f" [correct={trace.correct_count}/{trace.total_attempts}]"
            logger.info(f"           {status}")

            # 增量保存: 立即写入磁盘
            trace.save(trace_dir / f"{problem.problem_id}.json")
            trace_dicts.append(trace.to_dict())

        elapsed = time.time() - t_start

        # 计算指标
        k_values = [1, 5, 10]
        if args.max_samples >= 32:
            k_values.append(32)
        metrics = compute_metrics(trace_dicts, k_values=k_values)

        # Fix #5: 标注验证状态
        if args.lean_mode == "skip":
            metrics["verification"] = "unverified"
        else:
            metrics["verification"] = "lean4"

        summary = MetricsSummary(bench_name, metrics)

        # Fix #5: 在报告中标注验证模式
        unverified_tag = " [unverified]" if args.lean_mode == "skip" else ""
        logger.info(f"\n{summary.to_table()}")
        if unverified_tag:
            logger.info(f"  ⚠ 以上指标未经 Lean4 验证, 仅表示 LLM 生成了非空无 sorry 的代码")
        if skipped:
            logger.info(f"  (其中 {skipped} 道题使用了断点续跑缓存)")
        logger.info(f"  本轮耗时: {elapsed:.1f}s")

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
                "verification": "lean4" if args.lean_mode == "real" else "unverified",
                "multi_role": args.multi_role,
                "knowledge_enabled": not args.no_knowledge,
                "resumed_count": skipped,
                "elapsed_s": round(elapsed, 1),
                "metrics": metrics,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"  结果 → {eval_file}")

        all_results[bench_name] = metrics

    # 全局汇总
    if len(all_results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info(f"  全局汇总")
        logger.info(f"{'='*60}\n")
        unv = " [unverified]" if args.lean_mode == "skip" else ""
        header = f"  {'基准':<15} {'题数':>6} {'通过':>6} {'通过率':>8}"
        for k in [1, 5, 10]:
            header += f" {'pass@'+str(k):>8}"
        logger.info(header + unv)
        logger.info(f"  {'─'*65}")
        for name, m in all_results.items():
            line = (f"  {name:<15} {m['total']:>6} {m['solved']:>6} "
                    f"{m['solve_rate']:>7.1%}")
            for k in [1, 5, 10]:
                line += f" {m.get(f'pass@{k}', 0):>7.3f}"
            logger.info(line)


if __name__ == "__main__":
    main()
