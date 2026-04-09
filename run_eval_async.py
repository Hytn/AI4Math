#!/usr/bin/env python3
"""run_eval_async.py — 异步流水线评测入口

集成 Claw Code 6 大模式:
  模式1: ProofTaskStateMachine — 每道题有显式状态机
  模式2: ProofEventBus — 类型化事件替代日志
  模式3: RecoveryRegistry — 失败自动恢复
  模式4: ProofTaskPacket — 结构化任务规格
  模式5: ProofDashboard — 机器可读状态看板
  模式6: PolicyEngine — 可执行策略规则

Usage:
    python run_eval_async.py --benchmark minif2f --provider anthropic
    python run_eval_async.py --benchmark minif2f --provider mock --workers 4
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.loader import load_benchmark
from benchmarks.metrics import compute_metrics, MetricsSummary
from prover.models import (
    BenchmarkProblem, ProofTrace, ProofAttempt, AttemptStatus,
)
from agent.brain.async_llm_provider import create_async_provider
from common.prompt_builder import build_prompt
from common.response_parser import extract_lean_code
from common.roles import AgentRole, ROLE_PROMPTS
from prover.verifier.sorry_detector import detect_sorry
from prover.codegen.code_formatter import format_lean_code
from prover.premise.selector import PremiseSelector
from knowledge.store import UnifiedKnowledgeStore
from knowledge.reader import KnowledgeReader
from knowledge.writer import KnowledgeWriter
from engine.proof_context_store import StepDetail

# ── 模式 1-6: Lane Runtime ──
from engine.lane.task_state import (
    TaskStatus, ProofFailureClass, TaskContext, ProofTaskStateMachine,
)
from engine.lane.event_bus import ProofEventBus, wire_state_machine_to_bus
from engine.lane.recovery import RecoveryRegistry
from engine.lane.task_packet import ProofTaskPacket, validate_packet
from engine.lane.policy import PolicyEngine, PolicyAction
from engine.lane.dashboard import ProofDashboard
from engine.observability import metrics as obs_metrics, MetricsExporter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


async def prove_single_async(
        problem: BenchmarkProblem,
        llm,
        premise_selector: PremiseSelector,
        max_samples: int = 8,
        temperature: float = 0.8,
        knowledge_reader: KnowledgeReader = None,
        knowledge_writer: KnowledgeWriter = None,
        concurrency: int = 4,
        event_bus: ProofEventBus = None,
        dashboard: ProofDashboard = None,
        policy: PolicyEngine = None) -> ProofTrace:
    """异步证明单道题 — 集成 Claw Code 6 大模式。

    模式1: ProofTaskStateMachine 跟踪每道题的完整生命周期
    模式2: ProofEventBus 发出类型化事件 (不再依赖 logger.info)
    模式3: RecoveryRegistry 处理 API 错误/超时的自动恢复
    模式4: ProofTaskPacket 规范化输入 (验证级别/预算/策略)
    模式6: PolicyEngine 在每轮结束时评估是否升级策略
    """
    # ── 模式 4: 构建结构化任务包 ──
    packet = ProofTaskPacket(
        theorem_name=problem.name,
        formal_statement=problem.theorem_statement,
        difficulty=problem.difficulty,
        natural_language=problem.natural_language,
        max_samples=max_samples,
        temperature=temperature,
    )

    # ── 模式 1: 创建状态机 ──
    ctx = TaskContext(
        theorem_name=problem.name,
        formal_statement=problem.theorem_statement,
        difficulty=problem.difficulty,
    )
    sm = ProofTaskStateMachine(
        task_id=f"lane_{problem.problem_id}", context=ctx)

    # ── 模式 2: 接入事件总线 ──
    if event_bus:
        wire_state_machine_to_bus(sm, event_bus)

    # ── 模式 5: 注册到 Dashboard ──
    if dashboard:
        dashboard.register_task(sm)

    # ── 模式 3/6: 恢复注册表 + 策略引擎 ──
    recovery = RecoveryRegistry()
    policy = policy or PolicyEngine.default()

    trace = ProofTrace(
        problem_id=problem.problem_id,
        problem_name=problem.name,
        theorem_statement=problem.theorem_statement,
        natural_language=problem.natural_language,
    )

    # ── 模式 1: 知识加载阶段 ──
    premises = premise_selector.retrieve(problem.theorem_statement, top_k=10)
    premise_strs = [f"{r['name']}: {r['statement']}" for r in premises]

    knowledge_context = ""
    if knowledge_reader:
        sm.transition_to(TaskStatus.KNOWLEDGE_LOADING)
        try:
            knowledge_context = await knowledge_reader.render_for_prompt(
                goal=problem.theorem_statement,
                theorem=problem.theorem_statement,
                max_chars=1200)
            ctx.knowledge_injected = True
        except Exception as e:
            sm.fail(ProofFailureClass.KNOWLEDGE_ERROR, str(e))
            # 模式 6: 策略决策 — 知识加载失败不致命
            decision = policy.evaluate(sm)
            if decision.action != PolicyAction.GIVE_UP:
                sm.transition_to(TaskStatus.GENERATING)
            else:
                return trace

    effective_premises = list(premise_strs[:10])
    if knowledge_context:
        effective_premises.append(
            f"[Knowledge from past proofs]\n{knowledge_context}")

    # ── 模式 1: 进入生成阶段 ──
    if sm.status != TaskStatus.GENERATING:
        sm.transition_to(TaskStatus.GENERATING,
                         detail=f"samples={max_samples}")

    sem = asyncio.Semaphore(concurrency)
    results = []

    async def attempt_one(attempt_idx: int):
        async with sem:
            attempt = ProofAttempt(attempt_number=attempt_idx + 1)
            t0 = time.time()

            try:
                with obs_metrics.timer("llm_generate"):
                    prompt = build_prompt(
                        theorem_statement=problem.theorem_statement,
                        premises=effective_premises,
                    )
                    resp = await llm.generate(
                        system=ROLE_PROMPTS[AgentRole.PROOF_GENERATOR],
                        user=prompt,
                        temperature=temperature,
                    )
                proof = extract_lean_code(resp.content)
                proof = format_lean_code(proof) if proof.strip() else ""

                attempt.generated_proof = proof
                attempt.llm_model = resp.model
                attempt.llm_tokens_in = resp.tokens_in
                attempt.llm_tokens_out = resp.tokens_out
                attempt.llm_latency_ms = resp.latency_ms
                ctx.total_api_tokens += resp.tokens_in + resp.tokens_out

            except Exception as e:
                attempt.lean_result = AttemptStatus.LLM_ERROR
                attempt.lean_stderr = str(e)
                # ── 模式 3: API 错误自动恢复 ──
                sm.fail(ProofFailureClass.API_ERROR, str(e), recoverable=True)
                if recovery.should_recover(
                        ProofFailureClass.API_ERROR, sm.recovery_attempts):
                    await asyncio.sleep(recovery.get(
                        ProofFailureClass.API_ERROR).backoff_seconds)
                    sm.transition_to(TaskStatus.GENERATING)
                return attempt

            if not proof.strip():
                attempt.lean_result = AttemptStatus.LLM_ERROR
                attempt.lean_stderr = "Empty proof"
                return attempt

            sorry_report = detect_sorry(proof)
            if sorry_report.has_sorry:
                attempt.lean_result = AttemptStatus.LEAN_ERROR
                attempt.lean_stderr = (
                    f"Proof contains sorry "
                    f"({len(sorry_report.locations)} locations)")
                attempt.lean_check_ms = int((time.time() - t0) * 1000)
                sm.fail(ProofFailureClass.SORRY_DETECTED,
                        "sorry detected", recoverable=True)
                return attempt

            attempt.lean_result = AttemptStatus.LEAN_ERROR
            attempt.lean_stderr = (
                "[unverified] Lean verification skipped (async mode). "
                "This result has NOT been verified by Lean4.")
            attempt.lean_check_ms = 0

            if knowledge_writer:
                try:
                    step = StepDetail(
                        tactic=proof[:200],
                        goals_before=[problem.theorem_statement],
                        env_id_after=-1,
                        error_message="",
                        error_category="",
                        elapsed_ms=0,
                    )
                    await knowledge_writer.ingest_step(
                        step, theorem=problem.theorem_statement)
                except Exception:
                    pass

            return attempt

    # ── 模式 1: 验证阶段 ──
    sm.transition_to(TaskStatus.VERIFYING, detail=f"{max_samples} candidates")
    obs_metrics.increment("proof_tasks_started")

    tasks = [attempt_one(i) for i in range(max_samples)]
    attempts = await asyncio.gather(*tasks)
    ctx.total_samples = len(attempts)

    for attempt in attempts:
        trace.add_attempt(attempt)

    # ── 模式 1: 终态转换 ──
    if trace.solved:
        sm.succeed(trace.best_proof or "")
        obs_metrics.increment("proof_tasks_succeeded")
    else:
        # ── 模式 6: 策略评估 ──
        ctx.rounds_completed = 1
        decision = policy.evaluate(sm)
        if decision.action == PolicyAction.GIVE_UP:
            sm.give_up(decision.reason)
        else:
            sm.give_up(f"all {max_samples} attempts exhausted")
        obs_metrics.increment("proof_tasks_failed")

    return trace


async def prove_batch_async(
        problems: list[BenchmarkProblem],
        llm,
        premise_selector: PremiseSelector,
        max_samples: int = 8,
        concurrency: int = 4,
        knowledge_reader: KnowledgeReader = None,
        knowledge_writer: KnowledgeWriter = None,
        trace_dir: Path = None,
        existing_traces: dict = None,
        event_bus: ProofEventBus = None,
        dashboard: ProofDashboard = None,
        policy: PolicyEngine = None) -> list[dict]:
    """异步批量评测 — 集成 Dashboard (模式5) 和 EventBus (模式2)。"""
    existing_traces = existing_traces or {}
    trace_dicts = []
    problem_sem = asyncio.Semaphore(concurrency)

    async def process_one(i: int, problem: BenchmarkProblem):
        if problem.problem_id in existing_traces:
            return existing_traces[problem.problem_id]

        async with problem_sem:
            logger.info(f"  [{i+1}/{len(problems)}] {problem.name}")
            trace = await prove_single_async(
                problem, llm, premise_selector,
                max_samples=max_samples,
                knowledge_reader=knowledge_reader,
                knowledge_writer=knowledge_writer,
                concurrency=2,
                event_bus=event_bus,
                dashboard=dashboard,
                policy=policy)

            status = ("✓" if trace.solved
                      else f"✗ ({trace.total_attempts})")
            # ── 模式 5: Dashboard 状态行 ──
            dash_line = f" | {dashboard.summary_line()}" if dashboard else ""
            logger.info(f"           {status}{dash_line}")

            if trace_dir:
                trace.save(trace_dir / f"{problem.problem_id}.json")

            return trace.to_dict()

    tasks = [process_one(i, p) for i, p in enumerate(problems)]
    trace_dicts = await asyncio.gather(*tasks)
    return list(trace_dicts)


async def async_main():
    parser = argparse.ArgumentParser(
        description="AI4Math 异步流水线评测 (Lane Runtime)")
    parser.add_argument("--benchmark", default="builtin")
    parser.add_argument("--split", default="test")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--workers", type=int, default=4,
                        help="并发 LLM 调用数")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-knowledge", action="store_true")
    parser.add_argument("--metrics-port", type=int, default=0,
                        help="Prometheus metrics HTTP port (0=disabled)")
    args = parser.parse_args()

    # ── 模式 2: 全局事件总线 ──
    event_bus = ProofEventBus()

    # ── 模式 5: 全局 Dashboard ──
    dashboard = ProofDashboard()

    # ── 模式 6: 全局策略引擎 ──
    policy = PolicyEngine.default()

    # ── 模式 7 (bonus): 可观测性导出 ──
    exporter = MetricsExporter(obs_metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exporter.start_periodic_export(
        str(output_dir / "metrics.json"), interval_seconds=30)
    if args.metrics_port > 0:
        exporter.start_http_server(port=args.metrics_port)

    # 订阅事件用于日志
    def _log_event(event):
        if event.failure:
            obs_metrics.increment("lane_failures",
                                   failure_class=event.failure.failure_class.value)
    event_bus.subscribe("task.failure.*", _log_event)
    event_bus.subscribe("task.succeeded", lambda e: obs_metrics.increment("lane_successes"))

    # 初始化异步 LLM
    llm = create_async_provider({
        "provider": args.provider,
        "model": args.model,
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    })
    premise_selector = PremiseSelector({"mode": "hybrid"})

    # 知识系统
    knowledge_reader = None
    knowledge_writer = None
    if not args.no_knowledge:
        try:
            knowledge_dir = Path(args.output_dir) / "knowledge"
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            store = UnifiedKnowledgeStore(
                str(knowledge_dir / "knowledge.db"))
            knowledge_reader = KnowledgeReader(store)
            knowledge_writer = KnowledgeWriter(store)
            logger.info(f"  知识系统已启用")
        except Exception as e:
            logger.warning(f"  知识系统初始化失败: {e}")

    bench_names = (["builtin", "minif2f", "putnambench", "proofnet"]
                   if args.benchmark == "all" else [args.benchmark])

    for bench_name in bench_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"  异步评测: {bench_name}")
        logger.info(f"{'='*60}\n")

        try:
            problems = load_benchmark(
                bench_name, args.split, limit=args.limit)
        except Exception as e:
            logger.error(f"  加载失败: {e}")
            continue

        if not problems:
            continue

        trace_dir = Path(args.output_dir) / "traces" / bench_name
        existing = {}
        if args.resume:
            from run_eval import load_existing_traces
            existing = load_existing_traces(trace_dir)

        t0 = time.time()
        trace_dicts = await prove_batch_async(
            problems, llm, premise_selector,
            max_samples=args.max_samples,
            concurrency=args.workers,
            knowledge_reader=knowledge_reader,
            knowledge_writer=knowledge_writer,
            trace_dir=trace_dir,
            existing_traces=existing,
            event_bus=event_bus,
            dashboard=dashboard,
            policy=policy)
        elapsed = time.time() - t0

        k_values = [1, 5, 10]
        if args.max_samples >= 32:
            k_values.append(32)
        metrics = compute_metrics(trace_dicts, k_values=k_values)
        metrics["verification"] = "unverified"
        summary = MetricsSummary(bench_name, metrics)

        logger.info(f"\n{summary.to_table()}")
        logger.info(f"  ⚠ 异步模式: 所有指标 [unverified]")
        logger.info(f"  耗时: {elapsed:.1f}s (并发={args.workers})")

        eval_dir = Path(args.output_dir) / "evals"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / f"eval_async_{bench_name}.json", "w") as f:
            json.dump({
                "benchmark": bench_name, "async": True,
                "workers": args.workers,
                "metrics": metrics,
                "elapsed_s": round(elapsed, 1),
            }, f, indent=2, ensure_ascii=False)

    # 清理
    if hasattr(llm, 'close'):
        await llm.close()

    # ── 模式 5: 最终 Dashboard 快照 ──
    final_snapshot = dashboard.snapshot()
    snapshot_path = output_dir / "dashboard_final.json"
    with open(snapshot_path, "w") as f:
        json.dump(final_snapshot, f, indent=2)
    logger.info(f"\n  Dashboard final: {dashboard.summary_line()}")
    logger.info(f"  Dashboard saved: {snapshot_path}")

    # ── 导出最终指标 ──
    exporter.export_json(str(output_dir / "metrics_final.json"))
    exporter.stop()


if __name__ == "__main__":
    asyncio.run(async_main())
