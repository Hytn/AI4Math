"""run_unified.py — CLI for the unified proof pipeline.

Every theorem-proving algorithm is a `--profile` away. Examples::

    # DeepSeek-Prover style: single shot, no tools.
    python run_unified.py --builtin nat_add_comm --profile whole_proof

    # Whole-proof + repair (current AI4Math main path).
    python run_unified.py --builtin nat_add_comm --profile whole_proof_repair

    # ReProver style: retrieval + step-level tactic application.
    python run_unified.py --builtin nat_add_comm --profile reprover

    # LeanDojo-style pure step-level (no retrieval).
    python run_unified.py --builtin nat_add_comm --profile leandojo

    # MCTS with UCB1 outer search.
    python run_unified.py --builtin nat_add_comm --profile mcts

    # Best-first search.
    python run_unified.py --builtin nat_add_comm --profile best_first

    # Heterogeneous parallel (your project's existing flagship).
    python run_unified.py --builtin nat_add_comm --profile heterogeneous

    # User-defined profile from YAML.
    python run_unified.py --builtin nat_add_comm \\
        --profile-yaml config/profiles/mcts_with_retrieval.yaml \\
        --profile mcts_with_retrieval

The output is always a single self-contained ``dialog.json`` —— same format
across all profiles. The schema_version, meta.tools, meta.system_prompt,
messages, result fields tell you exactly which algorithm ran.
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Repo root on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.constants import DEFAULT_CLAUDE_MODEL  # noqa: E402

logger = logging.getLogger("run_unified")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--builtin", type=str, help="Builtin problem name")
    src.add_argument("--theorem", type=str,
                     help="Lean 4 theorem statement (raw)")
    src.add_argument("--benchmark", type=str,
                     help="Benchmark name (minif2f, putnambench, ...)")

    p.add_argument("--profile", type=str, default="whole_proof_repair",
                    help="Built-in or registered profile name")
    p.add_argument("--profile-yaml", type=str, default=None,
                    help="Load extra profile from YAML before resolving --profile")

    p.add_argument("--provider", type=str, default="anthropic")
    p.add_argument("--model", type=str, default=None,
                    help="Override model")
    p.add_argument("--lean", action="store_true",
                    help="Connect to a real Lean 4 REPL pool (else mock)")

    # ── Backend selection (infrastructure-merge feature) ────────────────
    p.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "local", "socket", "http", "kimina",
                  "pantograph", "lookeng", "mock", "fallback"],
        help=("Which Lean 4 verification backend to use. 'auto' probes "
              "local→socket→http and falls through to fallback. "
              "'kimina'/'http' uses the Kimina Lean Server REST API; "
              "'pantograph' enables mvar focus + DSP drafting; "
              "'lookeng' enables stateless lemma-by-lemma proving. "
              "Some profiles imply a backend (kimina_batch → kimina, "
              "pantograph_dsp → pantograph, lookeng_lemma → lookeng) "
              "and will auto-select if you pass 'auto'."))
    p.add_argument(
        "--backend-url", type=str, default=None,
        help=("URL for HTTP/Kimina backend "
              "(default: $KIMINA_SERVER_URL or http://localhost:8000)."))
    p.add_argument(
        "--backend-api-key", type=str, default=None,
        help="API key for HTTP/Kimina backend (default: $KIMINA_API_KEY).")

    p.add_argument("--config", type=str, default="config/default.yaml",
                    help="Path to YAML config file (default: config/default.yaml). "
                         "CLI flags and APE_*=... env vars override it.")
    p.add_argument("--out", type=str, default="results/unified",
                    help="Output dir for dialog.json")

    # v10: previously-locked features now reachable from the CLI.
    # Both default to None (off) — the runner already accepts these
    # as constructor args and gracefully no-ops when None.
    p.add_argument(
        "--world-model", type=str, default=None,
        metavar="PATH",
        help=("Path to a trained sklearn world-model pickle "
              "(produced by scripts/train_world_model.py). When set, "
              "tactic_apply / step-level profiles short-circuit "
              "high-confidence-failure tactics instead of sending them "
              "to Lean. Off by default."))
    p.add_argument(
        "--dialog-index", type=str, default=None,
        metavar="DB_PATH",
        help=("Path to a SQLite DialogIndex of past solved dialogs "
              "(see knowledge.dialog_index). When set and the active "
              "profile has observation.inject_similar_dialogs=True, "
              "similar past dialogs are prepended to the initial user "
              "message as in-context demos. Off by default."))
    p.add_argument(
        "--knowledge-db", type=str, default=None,
        metavar="DB_PATH",
        help=("Path to a SQLite knowledge store. When set, the runner "
              "deposits successful proofs and reads briefings. Off by "
              "default (no knowledge persistence in single-run mode)."))
    # v12: opt-in LLM response caching. Off by default to preserve
    # legacy behaviour. With ``--cache`` the runner wraps the provider
    # in AsyncCachedProvider; same prompt → cached LLMResponse on hit.
    # Particularly valuable for pass@k sweeps and resumed eval runs.
    p.add_argument(
        "--cache", action="store_true",
        help="Wrap the LLM in AsyncCachedProvider. Caches identical "
             "(system, messages, tools, T, max_tokens) calls in-process. "
             "By default only T=0 calls are cached; --cache-all forces "
             "caching at any temperature.")
    p.add_argument("--cache-all", action="store_true",
                    help="With --cache, cache responses at any temperature.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_problem(args):
    """Resolve --builtin / --theorem / --benchmark to a BenchmarkProblem."""
    from prover.models import BenchmarkProblem
    if args.theorem:
        return BenchmarkProblem(
            problem_id="custom",
            name="custom",
            theorem_statement=args.theorem)
    if args.builtin:
        from benchmarks.datasets.builtin.problems import BUILTIN_PROBLEMS
        for prob in BUILTIN_PROBLEMS:
            if prob.name == args.builtin or prob.problem_id == args.builtin:
                return prob
        raise SystemExit(f"unknown builtin: {args.builtin}")
    if args.benchmark:
        from benchmarks.loader import load_benchmark
        problems = load_benchmark(args.benchmark)
        if not problems:
            raise SystemExit(f"no problems in benchmark: {args.benchmark}")
        return problems[0]
    raise SystemExit("specify one of --theorem / --builtin / --benchmark")


def build_llm(args):
    """Provider-aware LLM construction.

    v12: when ``--cache`` is set, wrap the constructed provider in
    AsyncCachedProvider. The cache lives for the lifetime of the
    process — across multiple problems in --benchmark mode this can
    cut LLM cost meaningfully (system prompt + few-shot prefix is
    shared across problems).
    """
    if args.provider == "anthropic":
        from agent.brain.async_llm_provider import AsyncClaudeProvider
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY not set")
        provider = AsyncClaudeProvider(
            model=args.model or DEFAULT_CLAUDE_MODEL,
            api_key=api_key,
        )
    elif args.provider == "mock":
        from agent.brain.async_llm_provider import AsyncMockProvider
        provider = AsyncMockProvider()
    else:
        raise SystemExit(f"provider not yet wired: {args.provider}")

    if getattr(args, "cache", False):
        from agent.brain.async_llm_provider import AsyncCachedProvider
        provider = AsyncCachedProvider(
            provider, cache_all=getattr(args, "cache_all", False))
        logger.info(
            "LLM cache enabled (cache_all=%s)",
            getattr(args, "cache_all", False))
    return provider


def build_lean_pool(args, cfg=None):
    if not args.lean:
        return None
    try:
        from engine.async_lean_pool import AsyncLeanPool
        pool_size = (cfg or {}).get("lean_pool_size", 4)
        return AsyncLeanPool(pool_size=pool_size)
    except Exception as e:
        logger.warning(f"could not start AsyncLeanPool: {e}")
        return None


async def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # v9: load config (YAML + env overrides). CLI flags still win.
    from config.schema import load_config
    cfg = load_config(args.config)
    logger.info(
        f"loaded config from {args.config} "
        f"(lean_pool_size={cfg.get('lean_pool_size', 4)}, "
        f"premise.mode={cfg.get('prover', {}).get('premise', {}).get('mode', 'hybrid')})")

    # Load YAML-defined profile if provided
    if args.profile_yaml:
        from prover.unified import load_profile_from_yaml, register_profile
        register_profile(load_profile_from_yaml(args.profile_yaml))
        logger.info(f"registered profile from {args.profile_yaml}")

    from prover.unified import UnifiedProofRunner, get_profile, PRESETS
    if args.profile not in PRESETS:
        raise SystemExit(
            f"unknown profile '{args.profile}'. "
            f"Available: {sorted(PRESETS)}")

    problem = load_problem(args)
    llm = build_llm(args)
    lean_pool = build_lean_pool(args, cfg)

    # ── Build infrastructure backends per --backend / per profile ───────
    from prover.unified.factory import (
        resolve_backend_kind, build_infra_backends,
    )
    chosen = resolve_backend_kind(args.backend, args.profile)
    kimina_backend, pantograph_backend, lookeng_backend = (
        await build_infra_backends(
            chosen, url=args.backend_url, api_key=args.backend_api_key)
    )

    # ── v10/v11: optional features previously locked behind code-only access ──
    # v11: factory loaders make these one-liners + uniform with run_eval.py.
    from prover.unified.factory import (
        load_world_model, load_dialog_index, load_knowledge,
    )
    world_model = load_world_model(args.world_model)
    dialog_index = load_dialog_index(args.dialog_index)
    knowledge_store, knowledge_writer, _knowledge_reader = load_knowledge(
        args.knowledge_db)

    runner = UnifiedProofRunner(
        llm=llm,
        lean_pool=lean_pool,
        knowledge_store=knowledge_store,
        knowledge_writer=knowledge_writer,
        retriever=None,
        kimina_backend=kimina_backend,
        pantograph_backend=pantograph_backend,
        lookeng_backend=lookeng_backend,
        world_model=world_model,
        dialog_index=dialog_index,
    )
    result = await runner.run(problem, profile_name=args.profile)

    # Print summary
    profile = get_profile(args.profile)
    print(f"\n{'═' * 64}")
    print(f"Profile : {profile.name}  —  {profile.description}")
    print(f"Tools   : {[t.value for t in profile.tools]}")
    print(f"Search  : {profile.search.kind}, max_turns={profile.max_turns}")
    print(f"{'─' * 64}")
    print(f"Success : {result.success}")
    print(f"Duration: {result.total_duration_ms} ms")
    if result.proof_code:
        print(f"Proof   :\n{result.proof_code}")
    if result.search_summary:
        print(f"Search  : {result.search_summary}")
    print(f"{'═' * 64}\n")

    # Save dialog.json
    out_dir = Path(args.out) / problem.problem_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 取 profile 的真实 system_prompt + tool 描述, 写入 dialog.json
    from prover.unified.system_prompts import render_system_prompt
    from prover.unified.tool_kits import build_tool_registry
    real_system_prompt = render_system_prompt(profile.framing)
    registry = build_tool_registry(
        profile, lean_pool=lean_pool,
        knowledge_store=None, retriever=None,
        broadcast_bus=None, search_state=None,
    )
    tools_meta = []
    for kit in profile.tools:
        tool = registry.get(kit.value) if hasattr(registry, "get") else None
        tools_meta.append({
            "name": kit.value,
            "description": getattr(tool, "description", "") if tool else "",
            "parameters": getattr(tool, "input_schema", {}) if tool else {},
            "server_id": "builtin",
        })

    saved = result.save_unified(
        str(out_dir),
        problem_id=problem.problem_id,
        model=args.model or DEFAULT_CLAUDE_MODEL,
        provider=args.provider,
        system_prompt=real_system_prompt,
        tools=tools_meta,
        initial_task=problem.theorem_statement,
    )
    if saved:
        print(f"dialog.json saved: {out_dir}/dialog.json")

    # v12: surface cache stats so users can see whether --cache helped.
    cache_stats = getattr(llm, "cache_stats", None)
    if callable(cache_stats):
        st = cache_stats()
        print(f"LLM cache: hits={st['hits']} misses={st['misses']} "
              f"hit_rate={st['hit_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
