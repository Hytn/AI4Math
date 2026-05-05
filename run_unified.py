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

    # ── 
    # through code-only ``UnifiedProofRunner(...)`` kwargs, never via CLI) ──
    p.add_argument(
        "--policy-engine", action="store_true",
        help=("Enable engine.policy.PolicyEngine with the 5 default rules "
              "(InfraRecovery / ConsecutiveSameError / BudgetEscalation / "
              "BankedLemmaDecompose / Reflection). When enabled, multi-turn "
              "loops can early-terminate or switch strategy on declarative "
              "rules instead of running until ``max_turns``. Off by default "
              "to preserve v13 hardcoded behaviour."))
    p.add_argument(
        "--plugins-dir", type=str, default=None,
        metavar="DIR",
        help=("Domain-plugin root(s), comma-separated. Each subdirectory "
              "with a ``plugin.yaml`` is loaded; the runner injects the "
              "best-matching plugin's few-shot / extra premises / strategic "
              "hint into the initial user message per problem. Try "
              "``--plugins-dir plugins/strategies`` for the bundled "
              "algebra/analysis/number-theory pack. Off by default."))
    p.add_argument(
        "--lemma-bank-db", type=str, default=None,
        metavar="DB_PATH",
        help=("SQLite path for the cross-problem lemma bank (BM25 "
              "retrieval, fed by ConjectureProposeTool). When set, "
              "successful auxiliary lemmas from one problem are visible "
              "to LemmaBankTool fallback in later problems. Off by "
              "default; use a stable path across runs to accumulate."))
    p.add_argument(
        "--lean-version", type=str, default=None,
        metavar="TAG",
        help=("Lean toolchain tag stamped on lemmas written to "
              "--lemma-bank-db. "
              "Lets a future ``recheck_after_upgrade`` step flag "
              "lemmas extracted under a different toolchain."))
    p.add_argument(
        "--mathlib-rev", type=str, default=None,
        metavar="HASH",
        help="Mathlib commit stamped on lemmas written to --lemma-bank-db.")

    # ── 
    p.add_argument(
        "--api-base", type=str, default=None,
        metavar="URL",
        help=("OpenAI-compatible API base URL. Used by --provider=openai/"
              "deepseek/vllm/sglang/ollama/openai_compat. If unset, each "
              "alias picks a sensible default (vllm→localhost:8000, "
              "deepseek→api.deepseek.com, ...)."))

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

    # ── Sampling overrides (v17) ─────────────────────────────────────
    # These let you override profile defaults from the CLI without
    # editing profiles.py. Critical for prover models: DeepSeek-Prover-V2
    # paper uses temperature=1.0 for pass@k diversity, but the bundled
    # profiles default to 0.7 (better for general LLMs in repair loops).
    p.add_argument(
        "--temperature", type=float, default=None, metavar="T",
        help=("Override profile temperature. Recommended T=1.0 for "
              "specialised prover models (DeepSeek-Prover-V2, Kimina); "
              "T=0.6-0.8 for general LLMs. None = use profile default."))
    p.add_argument(
        "--max-turns", type=int, default=None, metavar="N",
        help=("Override profile max_turns. Useful for budget-constrained "
              "experiments or for stretching short profiles. "
              "None = use profile default."))
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

    
    AsyncCachedProvider. The cache lives for the lifetime of the
    process — across multiple problems in --benchmark mode this can
    cut LLM cost meaningfully (system prompt + few-shot prefix is
    shared across problems).

    
    OpenAI-compatible aliases (deepseek / vllm / sglang / ollama /
    openai / openai_compat) all work without a per-alias if-branch
    here. ``--api-base`` overrides the alias default base URL.
    """
    from agent.brain.async_llm_provider import create_async_provider
    cfg = {
        "provider": args.provider,
        "model": args.model or "",
        "api_key": "",
        "api_base": getattr(args, "api_base", None) or "",
    }
    # Per-provider api_key env-var lookup. Anthropic is required-or-die
    # for backward compatibility; OpenAI-family providers fall back to
    # env vars inside create_async_provider itself.
    if args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY not set")
        cfg["api_key"] = api_key
        cfg["model"] = cfg["model"] or DEFAULT_CLAUDE_MODEL
    elif args.provider == "deepseek":
        cfg["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
    elif args.provider == "openai":
        cfg["api_key"] = os.environ.get("OPENAI_API_KEY", "")

    try:
        provider = create_async_provider(cfg)
    except ValueError as e:
        raise SystemExit(str(e))

    if getattr(args, "cache", False):
        from agent.brain.async_llm_provider import AsyncCachedProvider
        provider = AsyncCachedProvider(
            provider, cache_all=getattr(args, "cache_all", False))
        logger.info(
            "LLM cache enabled (cache_all=%s)",
            getattr(args, "cache_all", False))
    return provider

def build_lean_pool(args, cfg=None):
    """构造 Lean 4 REPL 池。

    三种情形:
      * ``--backend mock`` —— 注入 ``MockTransport`` factory,所有 verify
        默认走"成功"响应。这是冒烟评测无 Lean 时唯一能产出
        ``success: true`` 的 dialog 路径。``meta.backends.lean_pool.is_mock``
        会标记为 True,评测脚本应据此排除非真实证明。
      * ``--lean`` —— 真实 Lean(``LocalTransport``)。
      * 其他 —— 不构造池,runner 端跳过 verify。
    """
    if getattr(args, "backend", None) == "mock":
        try:
            from engine.async_lean_pool import AsyncLeanPool
            from engine.transport import MockTransport
            pool_size = (cfg or {}).get("lean_pool_size", 2)
            return AsyncLeanPool(
                pool_size=pool_size,
                transport_factory=lambda _sid: MockTransport(),
            )
        except Exception as e:
            logger.warning(f"could not start mock AsyncLeanPool: {e}")
            return None
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
    if lean_pool is not None:
        # 池构造时未启动;在这里 start (启动是 async 的,只能在事件循环里)。
        await lean_pool.start()

    # ── Build infrastructure backends per --backend / per profile ───────
    from prover.unified.factory import (
        resolve_backend_kind, build_infra_backends,
    )
    chosen = resolve_backend_kind(args.backend, args.profile)
    kimina_backend, pantograph_backend, lookeng_backend = (
        await build_infra_backends(
            chosen, url=args.backend_url, api_key=args.backend_api_key)
    )

    # ── v10/optional features previously locked behind code-only access ──

    from prover.unified.factory import (
        load_world_model, load_dialog_index, load_knowledge,
        load_policy_engine, load_plugin_loader, load_persistent_lemma_bank,
    )
    world_model = load_world_model(args.world_model)
    dialog_index = load_dialog_index(args.dialog_index)
    knowledge_store, knowledge_writer, _knowledge_reader = load_knowledge(
        args.knowledge_db)
    policy_engine = load_policy_engine(getattr(args, "policy_engine", False))
    plugin_loader = load_plugin_loader(getattr(args, "plugins_dir", None))
    persistent_lemma_bank = load_persistent_lemma_bank(
        getattr(args, "lemma_bank_db", None),
        lean_version=getattr(args, "lean_version", None),
        mathlib_rev=getattr(args, "mathlib_rev", None),
    )

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
        policy_engine=policy_engine,
        plugin_loader=plugin_loader,
        persistent_lemma_bank=persistent_lemma_bank,
    )

    # Apply CLI overrides (--temperature / --max-turns) BEFORE running
    profile = get_profile(args.profile)
    overrides = {}
    if getattr(args, "temperature", None) is not None:
        overrides["temperature"] = args.temperature
    if getattr(args, "max_turns", None) is not None:
        overrides["max_turns"] = args.max_turns
    if overrides:
        from dataclasses import replace as _dc_replace
        from prover.unified import register_profile
        profile = _dc_replace(profile, **overrides)
        register_profile(profile)   # so runner.get_profile sees it
        logger.info(f"  profile overrides applied: {overrides}")

    result = await runner.run(problem, profile_name=args.profile)

    # Print summary
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

    cache_stats = getattr(llm, "cache_stats", None)
    if callable(cache_stats):
        st = cache_stats()
        print(f"LLM cache: hits={st['hits']} misses={st['misses']} "
              f"hit_rate={st['hit_rate']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())
