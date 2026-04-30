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

    p.add_argument("--out", type=str, default="results/unified",
                    help="Output dir for dialog.json")
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
    """Provider-aware LLM construction."""
    if args.provider == "anthropic":
        from agent.brain.async_llm_provider import AsyncClaudeProvider
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY not set")
        return AsyncClaudeProvider(
            model=args.model or "claude-sonnet-4-20250514",
            api_key=api_key,
        )
    if args.provider == "mock":
        from agent.brain.async_llm_provider import AsyncMockProvider
        return AsyncMockProvider()
    raise SystemExit(f"provider not yet wired: {args.provider}")


def build_lean_pool(args):
    if not args.lean:
        return None
    try:
        from engine.lean_pool import LeanPool
        return LeanPool()
    except Exception as e:
        logger.warning(f"could not start LeanPool: {e}")
        return None


async def _build_infra_backends(kind: str,
                                 url: str = None,
                                 api_key: str = None):
    """Construct the optional backend trio (kimina, pantograph, lookeng).

    Returns ``(kimina, pantograph, lookeng)`` — any can be None. Each is
    only built when ``kind`` requests it, so a default ``--backend auto``
    run pays no cost for backends it doesn't use.

    The runner's tools tolerate any of these being None or in fallback —
    they register in fallback mode and return a structured "unavailable"
    error if the LLM tries to call them, rather than crashing the loop.
    """
    kimina = pantograph = lookeng = None

    if kind in ("kimina", "http"):
        from engine.backends.kimina_server import KiminaServerBackend
        kimina = KiminaServerBackend(base_url=url, api_key=api_key)
        await kimina.start()
        logger.info(f"Kimina backend started "
                    f"(fallback={kimina.is_fallback})")

    elif kind == "pantograph":
        from engine.backends.pantograph import PantographBackend
        pantograph = PantographBackend()
        await pantograph.start()
        logger.info(f"Pantograph backend started "
                    f"(mode={pantograph.mode})")

    elif kind == "lookeng":
        from engine.backends.lookeng import LooKengBackend
        lookeng = LooKengBackend()
        await lookeng.start()
        logger.info("LooKeng backend started")

    # 'local', 'socket', 'mock', 'fallback', 'auto' don't need
    # any of the infrastructure backends; they're handled by the
    # standard lean_pool path.

    return kimina, pantograph, lookeng


async def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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
    lean_pool = build_lean_pool(args)

    # ── Build infrastructure backends per --backend / per profile ───────
    #
    # Some profiles imply a specific backend (e.g. kimina_batch wants
    # the Kimina server). When --backend=auto we honour that implicit
    # preference; an explicit --backend overrides it.
    profile_backend_hint = {
        "kimina_batch":   "kimina",
        "pantograph_dsp": "pantograph",
        "lookeng_lemma":  "lookeng",
    }
    chosen = args.backend
    if chosen == "auto" and args.profile in profile_backend_hint:
        chosen = profile_backend_hint[args.profile]
        logger.info(
            f"profile '{args.profile}' implies backend '{chosen}', using it")

    kimina_backend, pantograph_backend, lookeng_backend = await _build_infra_backends(
        chosen, args.backend_url, args.backend_api_key)

    runner = UnifiedProofRunner(
        llm=llm,
        lean_pool=lean_pool,
        knowledge_store=None,
        retriever=None,
        kimina_backend=kimina_backend,
        pantograph_backend=pantograph_backend,
        lookeng_backend=lookeng_backend,
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
        model=args.model or "claude-sonnet-4-20250514",
        provider=args.provider,
        system_prompt=real_system_prompt,
        tools=tools_meta,
        initial_task=problem.theorem_statement,
    )
    if saved:
        print(f"dialog.json saved: {out_dir}/dialog.json")


if __name__ == "__main__":
    asyncio.run(main())
