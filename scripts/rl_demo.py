#!/usr/bin/env python3
"""scripts/rl_demo.py — End-to-end RL pipeline smoke demo (v7.1)

Proves that the V7+ unification is *runnable today* without verl,
slime, vLLM, or even Lean installed. Uses ``MockPolicy`` over the
``backend="mock"`` ProofEnv to exercise every wire of the pipeline:

    problems → TreeRolloutSampler.collect_rollouts(problems, MockPolicy)
             → list[Trajectory]
             → to_grpo_batch  (verl-DataProto-shape)
             → save_batch_jsonl  (one line per traj on disk)
             → to_sft_jsonl    (successful-only, chat template)
             → t.save_unified  (per-trajectory dialog.json)

Run::

    python scripts/rl_demo.py --num-problems 4 --branching-factor 3 \\
        --output-dir /tmp/rl_demo

You'll see a printed summary like::

    [demo] Generated 12 trajectories across 4 problems
    [demo] success_rate=0.75 avg_turns=2.1 avg_reward=0.62
    [demo] GRPO batch  → /tmp/rl_demo/grpo_batch.jsonl  (12 rows)
    [demo] SFT JSONL   → /tmp/rl_demo/sft.jsonl         (9 records)
    [demo] dialog.json → /tmp/rl_demo/traces/<id>/dialog.json

Useful flags::

    --backend kimina --backend-url http://localhost:8000
        # uses Kimina Lean Server (requires running container)
    --policy openai --policy-url http://localhost:8001/v1 --policy-model <m>
        # uses a vLLM/SGLang-served model as policy
    --search-kind ucb
        # MCTS-UCB1 instead of best-first
    --grpo-normalize
        # group-normalised advantages

This is the script you run before claiming "the v7 unification works
end-to-end". If this exits 0 with non-zero ``success_rate``, every
join in the pipeline is wired correctly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Imports from the v7 unification ───────────────────────────────────
from sampler import (
    ProofEnvConfig, TreeRolloutSampler, TreeRolloutConfig,
    MockPolicy, OpenAIPolicy, to_grpo_batch, to_sft_jsonl, save_batch_jsonl,
    VERL_AVAILABLE, SLIME_AVAILABLE,
)


logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("rl_demo")


# ═══════════════════════════════════════════════════════════════════════
# Demo problems (no Lean needed — they pair with mock backend)
# ═══════════════════════════════════════════════════════════════════════

DEMO_PROBLEMS = [
    {
        "problem_id": "demo_intro_h",
        "theorem_statement": "theorem demo1 (P : Prop) (h : P) : P := by",
        "header": "",
    },
    {
        "problem_id": "demo_simp",
        "theorem_statement": "theorem demo2 (n : Nat) : n + 0 = n := by",
        "header": "",
    },
    {
        "problem_id": "demo_ring",
        "theorem_statement": "theorem demo3 (a b : Int) : a + b = b + a := by",
        "header": "",
    },
    {
        "problem_id": "demo_exact",
        "theorem_statement": "theorem demo4 (h : True) : True := by",
        "header": "",
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Mock-friendly env override
# ═══════════════════════════════════════════════════════════════════════

def _make_mock_env_factory(env_config: ProofEnvConfig):
    """Build a ProofEnv whose step() is a deterministic mock Lean.

    For ``backend="mock"`` we don't need a real ProofEnv — we wire a
    fake reset/step pair that treats certain tactics as winning.
    Mirrors the helper in ``tests/test_rl_unified.py``.
    """
    from sampler.proof_env import ProofEnv
    from sampler.trajectory import RewardInfo, Trajectory, Turn
    from unittest.mock import MagicMock

    def _build():
        env = ProofEnv(env_config)
        env._pool = MagicMock()
        env._verifier = MagicMock()

        async def fake_reset(problem):
            env._problem = problem
            env._turn_idx = 0
            env._done = False
            env._goals_remaining = 1
            env._accumulated_feedback = []
            env._episode_start = time.time()
            env._trajectory = Trajectory(
                problem_id=problem["problem_id"],
                theorem_statement=problem["theorem_statement"],
            )
            return f"prove: {problem['theorem_statement']}"

        async def fake_step(action: str):
            a = action.strip()
            if a in ("exact h", "rfl", "trivial"):
                r = RewardInfo(scalar=1.0, is_terminal=True,
                                verification_level="L2", goals_remaining=0)
                obs = "[TERMINATED: success]"
                done = True
            elif a in ("intro h", "intro", "simp", "ring"):
                r = RewardInfo(scalar=0.05, is_terminal=False,
                                verification_level="L1", goals_remaining=1)
                obs = f"goal after `{a}`: <new state>"
                done = False
            else:
                r = RewardInfo(scalar=0.0, is_terminal=False,
                                verification_level="L1",
                                error_class="tactic_failed")
                obs = f"error on '{a}'"
                done = False
            env._trajectory.add_turn(Turn(
                turn_idx=env._turn_idx, observation=obs,
                action=action, reward=r))
            env._turn_idx += 1
            return obs, r, done, {}

        env.reset = fake_reset
        env.step = fake_step
        return env

    return _build


# ═══════════════════════════════════════════════════════════════════════
# Demo orchestration
# ═══════════════════════════════════════════════════════════════════════

async def run_demo(args) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)

    # ── 1. Build policy ───────────────────────────────────────────────
    if args.policy == "mock":
        policy = MockPolicy(
            tactics=[
                # pairs with the mock env: 'exact h' wins; the rest
                # make progress without solving.
                "exact h", "intro h", "simp", "ring",
                "trivial", "rfl",
            ],
            shuffle=True,
            seed=args.seed,
            token_ids_for={
                "exact h": [10, 11, 12],
                "intro h": [20, 21],
                "simp": [30],
                "ring": [40],
                "trivial": [50],
                "rfl": [60],
            },
        )
    elif args.policy == "openai":
        policy = OpenAIPolicy(
            base_url=args.policy_url,
            model=args.policy_model,
            api_key=args.policy_api_key,
            temperature=args.policy_temperature,
        )
    else:
        raise ValueError(f"unknown --policy {args.policy}")

    # ── 2. Build sampler with the requested backend ───────────────────
    env_cfg = ProofEnvConfig(
        backend=args.backend,
        backend_url=args.backend_url,
        backend_api_key=args.backend_api_key,
        pool_size=args.pool_size,
        max_turns=args.max_turns,
        lean_timeout_s=args.lean_timeout_s,
    )
    cfg = TreeRolloutConfig(
        env_config=env_cfg,
        num_envs=args.pool_size,
        max_concurrent_problems=args.pool_size,
        search_kind=args.search_kind,
        branching_factor=args.branching_factor,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        max_paths_per_problem=args.paths_per_problem,
        group_normalize_rewards=args.grpo_normalize,
    )
    sampler = TreeRolloutSampler(cfg, policy_fn=policy)

    # ── 3. Pool setup — for backend=mock we skip real Lean and
    #       inject the fake env factory above ─────────────────────────
    if args.backend == "mock_offline":
        # Special offline mode: completely bypass AsyncLeanPool by
        # injecting fake envs. Produces the same trajectory shape.
        builder = _make_mock_env_factory(env_cfg)
        sampler._env_pool = [builder() for _ in range(cfg.num_envs)]
        sampler._env_queue = asyncio.Queue()
        for env in sampler._env_pool:
            sampler._env_queue.put_nowait(env)
        sampler._env_semaphore = asyncio.Semaphore(cfg.num_envs)
        sampler._setup_done = True
    else:
        await sampler.setup()

    # ── 4. Roll out ───────────────────────────────────────────────────
    problems = DEMO_PROBLEMS[:args.num_problems]
    log.info("Starting roll-out: %d problems, branching=%d, search=%s, "
                "backend=%s, policy=%s",
                len(problems), cfg.branching_factor, cfg.search_kind,
                env_cfg.backend, args.policy)
    t0 = time.time()
    trajectories = await sampler.collect_rollouts(problems)
    elapsed = time.time() - t0

    if not trajectories:
        log.error("No trajectories produced — pipeline broken")
        return 1

    # ── 5. Stats ──────────────────────────────────────────────────────
    stats = TreeRolloutSampler.batch_stats(trajectories)
    print(f"\n[demo] generated {len(trajectories)} trajectories "
            f"across {len(problems)} problems in {elapsed:.2f}s")
    print(f"[demo] success_rate={stats['success_rate']:.2f}  "
            f"avg_turns={stats['avg_turns']:.2f}  "
            f"avg_reward={stats['avg_reward']:.3f}")
    print(f"[demo] termination_dist={stats['termination_dist']}")

    # ── 6. GRPO batch dump ────────────────────────────────────────────
    grpo = to_grpo_batch(trajectories,
                            advantage_kind=("centered_normalized"
                                              if args.grpo_normalize
                                              else "raw"))
    grpo_path = output_dir / "grpo_batch.jsonl"
    n = save_batch_jsonl(grpo, grpo_path)
    print(f"[demo] GRPO batch  → {grpo_path}  ({n} rows)")
    if n >= 2:
        adv = grpo["advantages"]
        print(f"[demo]   advantages min={min(adv):+.3f} "
                f"max={max(adv):+.3f}  mean={sum(adv)/len(adv):+.3f}")

    # ── 7. SFT JSONL (successful only) ────────────────────────────────
    sft_path = output_dir / "sft.jsonl"
    n_sft = to_sft_jsonl(trajectories, sft_path,
                            successful_only=True, preset=args.sft_preset)
    print(f"[demo] SFT JSONL   → {sft_path}  ({n_sft} records)")

    # ── 8. Per-trajectory dialog.json ──────────────────────────────────
    n_dialogs = 0
    for t in trajectories[: args.dump_dialogs]:
        try:
            task_dir = traces_dir / f"{t.problem_id}_leaf{t.metadata.get('leaf_node_id', 0)}"
            t.save_unified(
                task_dir,
                model=args.policy_model if args.policy == "openai" else "mock",
                provider=args.policy,
                system_prompt="(see meta)",
            )
            n_dialogs += 1
        except Exception as e:
            log.debug("save_unified failed for %s: %r", t.problem_id, e)
    print(f"[demo] dialog.json → {traces_dir}/  "
            f"({n_dialogs}/{min(args.dump_dialogs, len(trajectories))} written)")

    # ── 9. Framework integration status ───────────────────────────────
    print(f"\n[demo] verl integration:  {'REAL' if VERL_AVAILABLE else 'STUB (install verl for real registration)'}")
    print(f"[demo] slime integration: {'REAL' if SLIME_AVAILABLE else 'STUB (install slime for real registration)'}")

    if args.dump_first_traj:
        print("\n[demo] First trajectory dump:")
        first = trajectories[0]
        print(json.dumps({
            "problem_id": first.problem_id,
            "success": first.success,
            "num_turns": first.num_turns,
            "total_reward": first.total_reward,
            "termination": first.termination.value,
            "turns": [
                {"turn_idx": t.turn_idx, "action": t.action,
                  "reward": t.reward.scalar,
                  "is_terminal": t.reward.is_terminal}
                for t in first.turns
            ],
            "metadata": first.metadata,
        }, indent=2, default=str))

    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    # Sampling shape
    ap.add_argument("--num-problems", type=int, default=4,
                      help="how many demo problems to roll out")
    ap.add_argument("--branching-factor", type=int, default=3,
                      help="k candidates per tree node expansion")
    ap.add_argument("--max-nodes", type=int, default=24)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--paths-per-problem", type=int, default=4,
                      help="max trajectories emitted per problem (= GRPO group size)")
    ap.add_argument("--search-kind", choices=("best_first", "ucb", "beam"),
                      default="best_first")
    ap.add_argument("--pool-size", type=int, default=2)
    ap.add_argument("--lean-timeout-s", type=int, default=10)
    ap.add_argument("--grpo-normalize", action="store_true",
                      help="apply per-group advantage normalisation")

    # Backend selection
    ap.add_argument("--backend",
                      choices=("local", "kimina", "http", "socket",
                                "pantograph", "lookeng", "mock",
                                "fallback", "mock_offline"),
                      default="mock_offline",
                      help="verification backend "
                              "(mock_offline = no Lean dependency at all, "
                              "the demo's default)")
    ap.add_argument("--backend-url", default=None,
                      help="for kimina/http: the Lean server URL")
    ap.add_argument("--backend-api-key", default=None)

    # Policy
    ap.add_argument("--policy", choices=("mock", "openai"),
                      default="mock")
    ap.add_argument("--policy-url", default=None,
                      help="OpenAI-compatible endpoint root, e.g. http://localhost:8001/v1")
    ap.add_argument("--policy-model", default="gpt-4o-mini")
    ap.add_argument("--policy-api-key", default=None)
    ap.add_argument("--policy-temperature", type=float, default=0.9)

    # Output
    ap.add_argument("--output-dir", default="/tmp/rl_demo")
    ap.add_argument("--sft-preset", default="qwen3")
    ap.add_argument("--dump-dialogs", type=int, default=2,
                      help="how many per-traj dialog.json to write")
    ap.add_argument("--dump-first-traj", action="store_true",
                      help="print the first trajectory inline")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.policy == "openai" and not args.policy_url:
        ap.error("--policy openai requires --policy-url")

    sys.exit(asyncio.run(run_demo(args)))


if __name__ == "__main__":
    main()
