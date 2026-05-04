#!/usr/bin/env python3
"""run_eval.py — 基准评测入口 (v9, profile-only)

每次 sample = 一次 ``UnifiedProofRunner.run(profile)``。
profile 选择算法 (whole_proof_repair / repair / conjecture_driven / ...)。
增量保存到 ``results/traces/<id>/dialog.json``, 支持断点续跑。

Usage::

    python run_eval.py --benchmark minif2f --profile whole_proof_repair --provider mock
    python run_eval.py --benchmark minif2f --profile repair --provider anthropic --resume
    python run_eval.py --benchmark all --profile heterogeneous --limit 10

历史路径 (v8 之前的 multi-role 老链路) 已在 v9 删除——所有路径统一通过
``prover.unified.UnifiedProofRunner``。
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.loader import load_benchmark
from benchmarks.metrics import compute_metrics, MetricsSummary
from prover.models import (
    BenchmarkProblem, ProofTrace,
)
from prover.premise.selector import PremiseSelector
from agent.brain.async_llm_provider import create_async_provider
from common.constants import DEFAULT_CLAUDE_MODEL
from knowledge.store import UnifiedKnowledgeStore
from knowledge.reader import KnowledgeReader
from knowledge.writer import KnowledgeWriter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)



def prove_single(problem: BenchmarkProblem, llm, premise_selector,
                 *,
                 profile: str,
                 max_samples: int = 8,
                 lean_env=None,
                 lean_mode: str = "skip",
                 knowledge_reader: KnowledgeReader = None,
                 knowledge_writer: KnowledgeWriter = None,
                 kimina_backend=None,
                 pantograph_backend=None,
                 lookeng_backend=None,
                 world_model=None,
                 dialog_index=None,
                 policy_engine=None,
                 plugin_loader=None,
                 persistent_lemma_bank=None) -> ProofTrace:
    """对单道题运行 ``UnifiedProofRunner.run(profile)``, 累积 pass@k。

    v9: profile-only。每次 sample = 一次独立 UnifiedProofRunner.run。
    v10: 新增 world_model / dialog_index 透传, 把基础设施真正接通到 CLI。
    v15: 新增 policy_engine / plugin_loader / persistent_lemma_bank 透传 ——
         v14 reservoir 之前只能从代码注入, 现在三处都从 CLI 走通。

    Args:
        profile: 必填。从 ``prover.unified.PRESETS`` 选择算法。
        kimina_backend / pantograph_backend / lookeng_backend:
            可选社区基建; None 时对应 ToolKit 进入 fallback 模式。
        world_model: 可选 sklearn world-model. 见 --world-model.
        dialog_index: 可选 DialogIndex. 见 --dialog-index.
        policy_engine: 可选 PolicyEngine. 见 --policy-engine.
        plugin_loader: 可选 PluginLoader. 见 --plugins-dir.
        persistent_lemma_bank: 可选 PersistentLemmaBank. 见 --lemma-bank-db.
    """
    if not profile:
        raise SystemExit(
            "--profile is required as of v9; non-profile path removed. "
            "Use one of: whole_proof_repair, repair, conjecture_driven, ..."
        )
    return _prove_single_unified(
        problem, llm, premise_selector,
        max_samples=max_samples, lean_env=lean_env,
        lean_mode=lean_mode,
        knowledge_reader=knowledge_reader,
        knowledge_writer=knowledge_writer,
        profile_name=profile,
        kimina_backend=kimina_backend,
        pantograph_backend=pantograph_backend,
        lookeng_backend=lookeng_backend,
        world_model=world_model,
        dialog_index=dialog_index,
        policy_engine=policy_engine,
        plugin_loader=plugin_loader,
        persistent_lemma_bank=persistent_lemma_bank,
    )


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


def _prove_single_unified(
    problem: BenchmarkProblem,
    llm,
    premise_selector,
    *,
    max_samples: int,
    lean_env,
    lean_mode: str,
    knowledge_reader,
    knowledge_writer,
    profile_name: str,
    kimina_backend=None,
    pantograph_backend=None,
    lookeng_backend=None,
    world_model=None,
    dialog_index=None,
    policy_engine=None,
    plugin_loader=None,
    persistent_lemma_bank=None,
) -> ProofTrace:
    """v3 unified 主管线路径: 每个 sample = 一次 UnifiedProofRunner.run。

    每次 run 根据 profile 决定方法学 (whole_proof / repair / reprover / ...);
    sample 之间是独立的 i.i.d. 调用 (满足 pass@k 假设)。

    Infrastructure-merge: 三个可选 backend (kimina/pantograph/lookeng) 由调
    用方在外层构建好后注入; 这里仅透传给 UnifiedProofRunner. None 时对应工具
    自动降级为 fallback.
    """
    from prover.unified import (
        UnifiedProofRunner, get_profile, unified_to_attempt,
    )

    trace = ProofTrace(
        problem_id=problem.problem_id,
        problem_name=problem.name,
        theorem_statement=problem.theorem_statement,
        natural_language=problem.natural_language,
        config_snapshot={"profile": profile_name, "via": "unified"},
    )

    # Resolve profile (or fall back gracefully)
    try:
        profile = get_profile(profile_name)
    except ValueError as e:
        logger.error(f"  unknown profile {profile_name!r}: {e}")
        return trace

    # v9: llm is always async (from create_async_provider).
    async_llm = llm

    # v11: extract a real AsyncLeanPool. Previously the code did
    # ``getattr(lean_env, "pool", None) or lean_env`` and ended up handing
    # ``LeanEnvironment`` (a bare subprocess wrapper, no ``verify_complete``)
    # to the runner, so ``--lean-mode real`` silently failed every verify.
    # We now require ``lean_env.pool`` to be a real pool; if absent, the
    # caller passed something that can't verify and we leave lean_pool=None
    # so the prefilter-only path is used (same as ``--lean-mode skip``).
    lean_pool = None
    if lean_env is not None:
        candidate = getattr(lean_env, "pool", None)
        if candidate is not None and hasattr(candidate, "verify_complete"):
            lean_pool = candidate
        elif hasattr(lean_env, "verify_complete"):
            # Caller passed an AsyncLeanPool directly (or a wrapper that
            # already exposes verify_complete) — accept it as the pool.
            lean_pool = lean_env
        else:
            logger.warning(
                f"lean_env of type {type(lean_env).__name__} has no "
                f"verify_complete; running with lean_pool=None (prefilter only). "
                f"This typically means you passed agent.executor.LeanEnvironment "
                f"instead of an AsyncLeanPool.")

    # Knowledge store (if writer/reader given, share their store)
    knowledge_store = None
    if knowledge_reader is not None:
        knowledge_store = getattr(knowledge_reader, "store", None)

    runner = UnifiedProofRunner(
        llm=async_llm,
        lean_pool=lean_pool,
        knowledge_store=knowledge_store,
        knowledge_writer=knowledge_writer,
        retriever=premise_selector,
        broadcast_bus=None,
        kimina_backend=kimina_backend,
        pantograph_backend=pantograph_backend,
        lookeng_backend=lookeng_backend,
        world_model=world_model,
        dialog_index=dialog_index,
        # v15 reservoirs
        policy_engine=policy_engine,
        plugin_loader=plugin_loader,
        persistent_lemma_bank=persistent_lemma_bank,
    )

    # 跑 max_samples 次 (pass@k 兼容)
    for sample_idx in range(max_samples):
        try:
            ur = asyncio.run(runner.run(problem, profile=profile))
        except RuntimeError as e:
            # 已在事件循环里 → 用 ensure_future
            try:
                loop = asyncio.get_event_loop()
                ur = loop.run_until_complete(runner.run(problem, profile=profile))
            except Exception as e2:
                logger.error(f"    sample {sample_idx+1} failed: {e2}")
                continue
        except Exception as e:
            logger.error(f"    sample {sample_idx+1} failed: {e}")
            continue

        attempt = unified_to_attempt(ur, attempt_number=sample_idx + 1)
        trace.add_attempt(attempt)

        if ur.success:
            trace.solved = True
            trace.successful_proof = ur.proof_code
            trace.correct_count += 1
            # 写知识库
            if knowledge_writer is not None:
                try:
                    asyncio.run(knowledge_writer.observe_solved_attempt(
                        problem=problem.theorem_statement,
                        proof=ur.proof_code,
                        domain=getattr(problem, "domain", ""),
                    ))
                except Exception as e:
                    logger.debug(f"    knowledge write failed: {e}")

    return trace


def main():
    parser = argparse.ArgumentParser(description="AI4Math 真实基准评测 (v3)")
    parser.add_argument("--benchmark", default="builtin",
                        help="数据集名: builtin/minif2f/putnambench/proofnet/all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--provider", default="mock", help="LLM: mock/anthropic")
    parser.add_argument("--model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lean-mode", default="skip",
                        help="Lean 验证: real / skip")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑: 跳过已有 trace 的题目")
    parser.add_argument("--no-knowledge", action="store_true",
                        help="禁用知识系统 (不沉淀/不注入)")
    parser.add_argument(
        "--profile", required=True,
        help=(
            "(REQUIRED in v9) prover.unified profile 名. "
            "可选: whole_proof, whole_proof_repair, dsp, reprover, "
            "leandojo, heterogeneous, kimina_batch, pantograph_dsp, "
            "lookeng_lemma, nfl_hybrid, conjecture_driven, ..."
        ))
    parser.add_argument(
        "--backend", default="auto",
        choices=["auto", "local", "socket", "http", "kimina",
                  "pantograph", "lookeng", "mock", "fallback"],
        help=(
            "Lean 4 verification backend. 默认 auto. "
            "kimina_batch profile 自动选 kimina; pantograph_dsp 自动选 "
            "pantograph; lookeng_lemma 自动选 lookeng."))
    parser.add_argument("--backend-url", default=None,
                          help="HTTP/Kimina backend URL.")
    parser.add_argument("--backend-api-key", default=None,
                          help="HTTP/Kimina backend API key.")

    # v10: previously-locked features now reachable from the CLI.
    parser.add_argument(
        "--world-model", default=None, metavar="PATH",
        help=("Path to a trained sklearn world-model pickle. When set, "
              "step-level profiles short-circuit high-confidence-failure "
              "tactics. Off by default."))
    parser.add_argument(
        "--dialog-index", default=None, metavar="DB_PATH",
        help=("SQLite DialogIndex of past solved dialogs. When set and "
              "the active profile has inject_similar_dialogs=True, "
              "similar past dialogs are injected as in-context demos."))

    # v12: opt-in LLM response caching.
    parser.add_argument(
        "--cache", action="store_true",
        help="Wrap LLM in AsyncCachedProvider (in-process LRU). "
             "Useful for pass@k sweeps and resumed runs where the "
             "system prompt + few-shot prefix repeats.")
    parser.add_argument("--cache-all", action="store_true",
                          help="With --cache, cache responses at any temperature.")

    # ── v15: factory-loaded v14 reservoirs (parity with run_unified.py) ──
    parser.add_argument(
        "--policy-engine", action="store_true",
        help=("Enable engine.policy.PolicyEngine with the 5 default rules. "
              "Off by default to preserve v13 hardcoded behaviour."))
    parser.add_argument(
        "--plugins-dir", default=None, metavar="DIR",
        help=("Domain-plugin root(s), comma-separated. Each subdirectory "
              "with a ``plugin.yaml`` is loaded; runner injects the "
              "best-matching plugin's few-shot/premises/hint per problem. "
              "Try ``--plugins-dir plugins/strategies``."))
    parser.add_argument(
        "--lemma-bank-db", default=None, metavar="DB_PATH",
        help=("SQLite path for the cross-problem lemma bank (BM25). "
              "Use a stable path across eval runs to accumulate."))
    parser.add_argument(
        "--lean-version", default=None, metavar="TAG",
        help="Lean toolchain tag stamped on lemmas written to --lemma-bank-db.")
    parser.add_argument(
        "--mathlib-rev", default=None, metavar="HASH",
        help="Mathlib commit stamped on lemmas written to --lemma-bank-db.")

    # ── v15: OpenAI-compatible providers ──
    parser.add_argument(
        "--api-base", default=None, metavar="URL",
        help=("OpenAI-compatible API base URL. Used by --provider="
              "openai/deepseek/vllm/sglang/ollama/openai_compat."))

    args = parser.parse_args()

    # 初始化 LLM (async, v15: 支持 anthropic / mock / openai-compatible)
    llm_cfg = {
        "provider": args.provider,
        "model": args.model,
        "api_key": "",
        "api_base": getattr(args, "api_base", None) or "",
    }
    # Per-provider api_key env-var lookup. Anthropic stays
    # backward-compatible (auto-pick from env); OpenAI-family providers
    # fall back to env vars inside create_async_provider itself.
    if args.provider == "anthropic":
        llm_cfg["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    elif args.provider == "deepseek":
        llm_cfg["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
    elif args.provider == "openai":
        llm_cfg["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    try:
        llm = create_async_provider(llm_cfg)
    except ValueError as e:
        raise SystemExit(str(e))
    if args.cache:
        from agent.brain.async_llm_provider import AsyncCachedProvider
        llm = AsyncCachedProvider(llm, cache_all=args.cache_all)
        logger.info(
            f"  LLM cache enabled (cache_all={args.cache_all})")
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

    # 初始化 Lean 环境 (v11: 用真正的 AsyncLeanPool, 不再用
    # agent.executor.LeanEnvironment —— 后者不暴露 verify_complete,
    # 等于让 UnifiedProofRunner 在 real 模式下静默退化到 prefilter.)
    lean_env = None
    if args.lean_mode == "real":
        try:
            from engine.async_lean_pool import AsyncLeanPool
            pool_size = 4
            lean_env = AsyncLeanPool(pool_size=pool_size)
            asyncio.run(lean_env.start())
            logger.info(f"  Lean 4 池已启动 (pool_size={pool_size})")
        except Exception as e:
            logger.warning(f"无法启动 AsyncLeanPool: {e}, 回退到 skip 模式")
            args.lean_mode = "skip"
            lean_env = None

    # Infrastructure-merge: 构建可选 backend (共享 factory).
    from prover.unified.factory import (
        resolve_backend_kind, build_infra_backends,
        load_world_model, load_dialog_index,
    )
    chosen_backend = resolve_backend_kind(args.backend, args.profile)
    kimina_backend, pantograph_backend, lookeng_backend = asyncio.run(
        build_infra_backends(
            chosen_backend, url=args.backend_url, api_key=args.backend_api_key))

    if args.lean_mode == "skip":
        logger.info("  ⚠ Lean 验证已跳过 (--lean-mode=skip): "
                     "所有 pass@k 指标标记为 [unverified]")

    # v10/v11: optional features (shared factory loaders)
    # v15: same treatment for v14 reservoirs (policy / plugins / lemma bank).
    from prover.unified.factory import (
        load_policy_engine, load_plugin_loader, load_persistent_lemma_bank,
    )
    world_model = load_world_model(args.world_model)
    dialog_index = load_dialog_index(args.dialog_index)
    policy_engine = load_policy_engine(getattr(args, "policy_engine", False))
    plugin_loader = load_plugin_loader(getattr(args, "plugins_dir", None))
    persistent_lemma_bank = load_persistent_lemma_bank(
        getattr(args, "lemma_bank_db", None),
        lean_version=getattr(args, "lean_version", None),
        mathlib_rev=getattr(args, "mathlib_rev", None),
    )

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
                profile=args.profile,
                max_samples=args.max_samples,
                lean_env=lean_env, lean_mode=args.lean_mode,
                knowledge_reader=knowledge_reader,
                knowledge_writer=knowledge_writer,
                kimina_backend=kimina_backend,
                pantograph_backend=pantograph_backend,
                lookeng_backend=lookeng_backend,
                world_model=world_model,
                dialog_index=dialog_index,
                # v15 reservoirs
                policy_engine=policy_engine,
                plugin_loader=plugin_loader,
                persistent_lemma_bank=persistent_lemma_bank,
            )

            status = ("✓ 通过" if trace.solved
                      else f"✗ ({trace.total_attempts} 次)")
            if trace.correct_count > 1:
                status += f" [correct={trace.correct_count}/{trace.total_attempts}]"
            logger.info(f"           {status}")

            # 增量保存: 单文件 dialog.json (v3.0 schema, agent.persistence.unified_storage).
            # 没有 result.json / meta_config.json / trace.json 这些副产物 ——
            # 所有 meta + result 全部内联进 dialog.json 的 wrapped object.
            trace.save_unified(
                trace_dir / problem.problem_id,
                model=getattr(llm, "model_name", ""),
            )
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

    # v12: cache hit-rate at end of run.
    cache_stats = getattr(llm, "cache_stats", None)
    if callable(cache_stats):
        st = cache_stats()
        logger.info(
            f"\nLLM cache: hits={st['hits']} misses={st['misses']} "
            f"hit_rate={st['hit_rate']:.1%} size={st['size']}")


if __name__ == "__main__":
    main()
