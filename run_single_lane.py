#!/usr/bin/env python3
"""run_single_lane.py — 单题 Lane 调试: 逐步遍历全管线

本脚本让你跟着一道题目走完 AI4Math 的完整数据管线:

  读题 → 知识注入 → 方向规划 → 多智能体并行生成 → 三级验证
  → 策略决策 → 知识沉淀 → 状态压缩 → Green Contract 检查

每一步都有详细的中间状态输出, 适合单步调试和理解系统脊柱。

Usage::

    # Mock 模式 (无需 API Key, 30 秒):
    python run_single_lane.py

    # 指定内置题目:
    python run_single_lane.py --builtin nat_add_comm

    # 自定义定理:
    python run_single_lane.py --theorem "theorem t (n : Nat) : n + 0 = n"

    # 真实 Claude API:
    python run_single_lane.py --provider anthropic --builtin nat_add_comm

    # 详细输出 (含 LLM prompt / response 全文):
    python run_single_lane.py --verbose
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Color helpers ───────────────────────────────────────────────────────────
R = '\033[91m'; G = '\033[92m'; Y = '\033[93m'
C = '\033[96m'; W = '\033[97m'; DIM = '\033[2m'; N = '\033[0m'; B = '\033[1m'

def header(title):
    print(f"\n{W}{'═'*60}{N}")
    print(f"{W}  {title}{N}")
    print(f"{W}{'═'*60}{N}")

def step(n, title):
    print(f"\n{C}── Step {n}: {title} ──{N}")

def ok(msg): print(f"  {G}✓{N} {msg}")
def info(msg): print(f"  {C}ℹ{N} {msg}")
def warn(msg): print(f"  {Y}⚠{N} {msg}")
def fail(msg): print(f"  {R}✗{N} {msg}")
def dim(msg): print(f"  {DIM}{msg}{N}")
def kv(k, v): print(f"  {B}{k}:{N} {v}")


def main():
    parser = argparse.ArgumentParser(
        description="AI4Math — Single Problem Lane Debug")
    parser.add_argument("--builtin", type=str, default=None,
                        help="Builtin problem name (e.g. nat_add_comm)")
    parser.add_argument("--theorem", type=str, default=None,
                        help="Custom Lean4 theorem statement")
    parser.add_argument("--provider", default="mock",
                        choices=["mock", "anthropic"],
                        help="LLM provider (default: mock)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from common.logging_config import setup_logging
    if args.verbose:
        setup_logging(level="DEBUG")
    else:
        setup_logging(level="WARNING")

    asyncio.run(run_debug(args))


async def run_debug(args):
    header("AI4Math — Single Problem Lane Debug")
    print(f"  Provider: {args.provider} | Model: {args.model} | "
          f"Samples: {args.max_samples}")

    # ══════════════════════════════════════════════════════════════════
    # Step 1: 读题
    # ══════════════════════════════════════════════════════════════════
    step(1, "读题 & 问题加载")

    from prover.models import BenchmarkProblem
    from benchmarks.datasets.builtin.problems import BUILTIN_PROBLEMS

    if args.theorem:
        problem = BenchmarkProblem("custom", "custom", args.theorem)
        ok(f"自定义定理")
    elif args.builtin:
        matching = [p for p in BUILTIN_PROBLEMS if args.builtin in p.name]
        problem = matching[0] if matching else BUILTIN_PROBLEMS[0]
        ok(f"内置题: {problem.name}")
    else:
        problem = BUILTIN_PROBLEMS[0]
        ok(f"默认题: {problem.name}")

    kv("Name", problem.name)
    kv("Statement", problem.theorem_statement)
    kv("Difficulty", problem.difficulty)

    # ══════════════════════════════════════════════════════════════════
    # Step 2: 构建 Lane 组件
    # ══════════════════════════════════════════════════════════════════
    step(2, "组装 Lane 运行时组件")

    from engine.lane.task_packet import ProofTaskPacket, validate_packet
    from engine.lane.event_bus import ProofEventBus
    from engine.lane.dashboard import ProofDashboard
    from engine.lane.policy import PolicyEngine
    from engine.lane.integration import LaneProofRunner
    from engine.lane.green_contract import GreenLevel, ProofGreenContract
    from engine.lane.summary_compression import compress_proof_status

    # Event bus — capture all events for display
    bus = ProofEventBus()
    event_log = []
    bus.subscribe("*", lambda e: event_log.append(e))
    ok("EventBus: 已创建, 订阅 * 模式")

    # Dashboard
    dashboard = ProofDashboard()
    ok("Dashboard: 已创建")

    # Policy engine
    policy = PolicyEngine.default()
    ok(f"PolicyEngine: {len(policy._rules)} 条规则已加载")
    for rule in policy._rules:
        dim(f"  Rule: {rule.name} (priority={rule.priority})")

    # Knowledge system
    from knowledge.store import UnifiedKnowledgeStore
    from knowledge.writer import KnowledgeWriter
    from knowledge.reader import KnowledgeReader
    store = UnifiedKnowledgeStore(":memory:")
    writer = KnowledgeWriter(store)
    reader = KnowledgeReader(store)
    ok("Knowledge: 内存数据库已创建 (store + writer + reader)")

    # LLM provider
    from agent.brain.async_llm_provider import (
        AsyncMockProvider, AsyncClaudeProvider)

    if args.provider == "mock":
        async_llm = AsyncMockProvider()
        ok(f"LLM: mock (async)")
    else:
        async_llm = AsyncClaudeProvider(model=args.model)
        ok(f"LLM: Claude ({args.model})")

    # Agent pool
    from agent.runtime.async_agent_pool import AsyncAgentPool
    pool = AsyncAgentPool(llm=async_llm, max_workers=4)
    ok("AsyncAgentPool: 4 workers")

    # Direction planner
    from agent.strategy.direction_planner import DirectionPlanner
    planner = DirectionPlanner()
    ok("DirectionPlanner: 默认规划器")

    # Broadcast bus
    from engine.broadcast import BroadcastBus
    broadcast = BroadcastBus()
    ok("BroadcastBus: 已创建")

    # ══════════════════════════════════════════════════════════════════
    # Step 3: 知识注入预览
    # ══════════════════════════════════════════════════════════════════
    step(3, "知识注入 (Knowledge Loading)")

    try:
        knowledge_text = await reader.render_for_prompt(
            goal=problem.theorem_statement,
            theorem=problem.theorem_statement,
            max_chars=1500)
        if knowledge_text:
            ok(f"知识注入: {len(knowledge_text)} chars")
            dim(knowledge_text[:200])
        else:
            info("知识库为空 (首次运行, 无积累知识)")
    except Exception as e:
        warn(f"知识加载跳过: {e}")

    # ══════════════════════════════════════════════════════════════════
    # Step 4: 方向规划
    # ══════════════════════════════════════════════════════════════════
    step(4, "方向规划 (Direction Planning)")

    directions = planner.plan(problem)
    ok(f"规划了 {len(directions)} 个探索方向:")
    for d in directions:
        kv(f"  {d.name}", f"role={d.role} temp={d.temperature}")
        if d.strategic_hint:
            dim(f"    hint: {d.strategic_hint[:80]}...")

    # ══════════════════════════════════════════════════════════════════
    # Step 5: 构建 TaskPacket 并运行 LaneProofRunner
    # ══════════════════════════════════════════════════════════════════
    step(5, "构建 TaskPacket & 运行证明循环")

    packet = validate_packet(ProofTaskPacket(
        theorem_name=problem.name,
        formal_statement=problem.theorem_statement,
        domain=getattr(problem, 'domain', ''),
        difficulty=problem.difficulty,
        max_samples=args.max_samples,
        max_wall_seconds=300,
        initial_strategy="light",
        inject_knowledge=True,
        deposit_knowledge=True,
    ))
    kv("Packet", f"samples={packet.max_samples} strategy={packet.initial_strategy}")

    runner = LaneProofRunner(
        agent_pool=pool,
        scheduler=None,  # No real Lean4 REPL in debug mode
        direction_planner=planner,
        knowledge_reader=reader,
        knowledge_writer=writer,
        broadcast=broadcast,
        event_bus=bus,
        dashboard=dashboard,
        policy=policy,
    )

    t0 = time.time()
    info("开始证明循环...")
    sm = await runner.run(packet)
    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════════════
    # Step 6: 状态机结果
    # ══════════════════════════════════════════════════════════════════
    step(6, "状态机结果 (TaskStateMachine)")

    snap = sm.snapshot()
    status_color = G if sm.status == TaskStatus.SUCCEEDED else (
        Y if sm.status == TaskStatus.GIVEN_UP else R)
    print(f"\n  {status_color}{B}状态: {sm.status.value.upper()}{N}")
    kv("Rounds", sm.context.rounds_completed)
    kv("Samples", sm.context.total_samples)
    kv("Recoveries", sm.recovery_attempts)
    kv("Elapsed", f"{elapsed:.1f}s")
    if sm.context.best_attempt_code:
        kv("Best code", sm.context.best_attempt_code[:120])

    # ══════════════════════════════════════════════════════════════════
    # Step 7: 事件流
    # ══════════════════════════════════════════════════════════════════
    step(7, f"事件流 ({len(event_log)} events)")

    from engine.lane.task_state import TaskStatus as TS
    for e in event_log[-20:]:  # last 20
        prefix = G if "succeeded" in e.event_name else (
            R if "failure" in e.event_name else C)
        detail = f" — {e.detail}" if e.detail else ""
        failure = ""
        if e.failure:
            failure = f" [{e.failure.failure_class.value}: {e.failure.message[:50]}]"
        print(f"  {prefix}#{e.seq:02d}{N} {e.event_name}{detail}{failure}")

    # ══════════════════════════════════════════════════════════════════
    # Step 8: Green Contract 检查
    # ══════════════════════════════════════════════════════════════════
    step(8, "Green Contract 检查")

    if sm.status == TaskStatus.SUCCEEDED:
        level = GreenLevel.GOALS_CLOSED  # No real REPL → assume L1
    else:
        level = GreenLevel.NONE

    for name, contract in [
        ("Quick Filter", ProofGreenContract.for_quick_filter()),
        ("Candidate", ProofGreenContract.for_candidate()),
        ("Deposit", ProofGreenContract.for_deposit()),
        ("Certification", ProofGreenContract.for_certification()),
    ]:
        outcome = contract.evaluate(level)
        icon = G + "✅" if outcome.satisfied else R + "❌"
        print(f"  {icon}{N} {name}: {outcome.summary}")

    # ══════════════════════════════════════════════════════════════════
    # Step 9: 压缩状态摘要
    # ══════════════════════════════════════════════════════════════════
    step(9, "压缩状态摘要 (Summary Compression)")

    summary = compress_proof_status(sm, policy=policy)
    kv("One-liner", summary.one_liner)
    print()
    info("Prompt 注入格式:")
    print(f"{DIM}{summary.for_prompt(max_chars=400)}{N}")

    # ══════════════════════════════════════════════════════════════════
    # Step 10: Dashboard 全局视图
    # ══════════════════════════════════════════════════════════════════
    step(10, "Dashboard 全局视图")

    kv("Summary", dashboard.summary_line())
    from engine.lane.summary_compression import compress_dashboard
    compressed = compress_dashboard(dashboard, policy)
    kv("Compressed", compressed["one_liner"])

    # ══════════════════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════════════════
    header("调试完成")
    if sm.status == TaskStatus.SUCCEEDED:
        print(f"  {G}✓ 证明找到!{N}")
    else:
        print(f"  {Y}证明未找到 (状态: {sm.status.value}){N}")
        if args.provider == "mock":
            print(f"  {DIM}提示: Mock 模式下无真实 LLM, 请用 --provider anthropic{N}")

    print(f"\n  {DIM}数据管线遍历完成。每一步的输出对应 README '项目脊柱' 中的数据流:{N}")
    print(f"  {DIM}  读题 → 知识注入 → 方向规划 → 并行生成 → 验证{N}")
    print(f"  {DIM}  → 策略决策 → 知识沉淀 → 状态压缩 → Green Contract{N}")
    print()


from engine.lane.task_state import TaskStatus  # noqa: E402

if __name__ == "__main__":
    main()
