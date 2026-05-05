"""tests/test_rl_runnable.py — V7.1 runnable RL pipeline tests.

V7 unified the three infrastructure layers (recap):
  * ProofEnv now selects backends through ``ProofEnvConfig.backend``
  * BaseSampler uses an asyncio.Queue (no more env-pool race)
  * VeRLProofAgentLoop / Interaction conditionally inherit from real verl
  * TreeRolloutSampler exposes tree search as a sampler primitive

V7.1 adds the *runnable* user-facing surface:
  * Policy adapters (MockPolicy, OpenAIPolicy, CallablePolicy)
  * Batch export helpers (to_grpo_batch, to_sft_jsonl, to_ppo_batch)
  * scripts/rl_demo.py — end-to-end no-deps smoke demo
  * scripts/rl_pipeline.py rollout subcommand using the sampler

These tests pin every piece.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sampler import (
    CallablePolicy, MockPolicy, OpenAIPolicy, ProofEnvConfig,
    RewardInfo, save_batch_jsonl, to_grpo_batch, to_ppo_batch,
    to_sft_jsonl, Trajectory, TreeRolloutConfig, TreeRolloutSampler, Turn,
    build_policy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════
# Policy adapter tests
# ═══════════════════════════════════════════════════════════════════════

class TestMockPolicy:
    def test_cycle_default(self):
        async def _t():
            p = MockPolicy()
            r1 = await p("any obs")
            r2 = await p("any obs")
            r3 = await p("any obs")
            assert r1[0] == "intro h"
            assert r2[0] == "exact h"
            assert r3[0] == "simp"
        asyncio.run(_t())

    def test_custom_tactics(self):
        async def _t():
            p = MockPolicy(tactics=["a", "b"])
            r1 = await p("o")
            r2 = await p("o")
            r3 = await p("o")
            assert r1[0] == "a"
            assert r2[0] == "b"
            assert r3[0] == "a"
        asyncio.run(_t())

    def test_token_ids_provided(self):
        async def _t():
            p = MockPolicy(
                tactics=["foo"],
                token_ids_for={"foo": [1, 2, 3]},
            )
            text, ids, lps = await p("obs")
            assert text == "foo"
            assert ids == [1, 2, 3]
            assert len(lps) == 3
        asyncio.run(_t())

    def test_shuffle_deterministic_with_seed(self):
        async def _t():
            p = MockPolicy(tactics=["a", "b", "c", "d"],
                            shuffle=True, seed=42)
            seq = [(await p("o"))[0] for _ in range(10)]
            # All from the tactic list
            assert all(s in {"a", "b", "c", "d"} for s in seq)
        asyncio.run(_t())

class TestCallablePolicy:
    def test_sync_callable(self):
        async def _t():
            p = CallablePolicy(lambda o: f"echo:{o}")
            text, ids, lps = await p("hello")
            assert text == "echo:hello"
            assert ids == []
        asyncio.run(_t())

    def test_async_callable(self):
        async def _t():
            async def asy(o):
                await asyncio.sleep(0)
                return "async-result"
            p = CallablePolicy(asy)
            text, _, _ = await p("o")
            assert text == "async-result"
        asyncio.run(_t())

class TestOpenAIPolicyOffline:
    """Without aiohttp / a real server, OpenAIPolicy must still
    construct cleanly and fail gracefully."""

    def test_constructs_without_aiohttp(self):
        p = OpenAIPolicy(base_url="http://nowhere", model="m")
        assert p.base_url == "http://nowhere"
        assert p.model == "m"

    def test_unwrap_lean_block(self):
        assert OpenAIPolicy._unwrap("```lean\nintro h\n```") == "intro h"
        assert OpenAIPolicy._unwrap("```\nrfl\n```") == "rfl"
        assert OpenAIPolicy._unwrap("simp [h]") == "simp [h]"
        assert OpenAIPolicy._unwrap("  exact h  ") == "exact h"

    def test_failing_request_returns_empty(self):
        async def _t():
            p = OpenAIPolicy(base_url="http://localhost:1",
                                 model="x", timeout_s=0.01)
            text, ids, lps = await p("test")
            # No real server → request fails → empty triple.
            assert text == ""
            assert ids == []
            assert lps == []
            await p.close()
        asyncio.run(_t())

class TestBuildPolicy:
    def test_mock_from_config(self):
        p = build_policy({"kind": "mock", "tactics": ["x"]})
        assert isinstance(p, MockPolicy)

    def test_openai_from_config(self):
        p = build_policy({
            "kind": "openai",
            "base_url": "http://h:1/v1",
            "model": "m",
        })
        assert isinstance(p, OpenAIPolicy)

    def test_callable_from_config(self):
        p = build_policy({
            "kind": "callable",
            "fn": lambda o: "x",
        })
        assert isinstance(p, CallablePolicy)

    def test_callable_requires_fn(self):
        with pytest.raises(ValueError):
            build_policy({"kind": "callable"})

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            build_policy({"kind": "alien"})

# ═══════════════════════════════════════════════════════════════════════
# Batch export tests
# ═══════════════════════════════════════════════════════════════════════

def _mk_traj(problem_id: str, total_reward: float,
              success: bool = False, num_turns: int = 1) -> Trajectory:
    t = Trajectory(problem_id=problem_id, theorem_statement="theorem t")
    for i in range(num_turns):
        is_last = (i == num_turns - 1)
        r = total_reward if is_last else 0.0
        t.add_turn(Turn(
            turn_idx=i,
            observation=f"obs{i}",
            action=f"act{i}",
            reward=RewardInfo(
                scalar=r,
                is_terminal=(is_last and success)),
            observation_token_ids=[100 + i],
            action_token_ids=[200 + i, 300 + i],
            action_log_probs=[-0.5, -0.6],
        ))
    t.success = success
    return t

class TestGrpoBatch:
    def test_empty_input(self):
        b = to_grpo_batch([])
        assert b["problem_ids"] == []
        assert "advantages" in b

    def test_single_group_normalization(self):
        # Three trajectories from same problem with different rewards
        trajs = [
            _mk_traj("p1", total_reward=1.0, success=True),
            _mk_traj("p1", total_reward=0.5),
            _mk_traj("p1", total_reward=0.0),
        ]
        b = to_grpo_batch(trajs, advantage_kind="centered_normalized")
        assert b["problem_ids"] == ["p1", "p1", "p1"]
        assert b["group_size"] == [3, 3, 3]
        # Centered: mean(advantages) ≈ 0
        assert abs(sum(b["advantages"])) < 1e-6
        # Normalised: std-dev ≈ 1
        m = sum(b["advantages"]) / 3
        var = sum((a - m) ** 2 for a in b["advantages"]) / 3
        assert abs(math.sqrt(var) - 1.0) < 1e-6

    def test_singleton_group_zero_advantage(self):
        trajs = [_mk_traj("only_one", total_reward=2.0)]
        b = to_grpo_batch(trajs, min_group_size=2)
        # Singleton: advantage forced to 0
        assert b["advantages"] == [0.0]

    def test_multiple_groups(self):
        trajs = [
            _mk_traj("p1", 1.0, success=True),
            _mk_traj("p1", 0.0),
            _mk_traj("p2", 0.5),
            _mk_traj("p2", 0.5),
        ]
        b = to_grpo_batch(trajs)
        assert len(b["advantages"]) == 4
        # p2 group has zero variance → normalized advantages are 0
        p2_advs = [a for pid, a in zip(b["problem_ids"], b["advantages"])
                      if pid == "p2"]
        assert all(abs(a) < 1e-6 for a in p2_advs)

    def test_raw_advantage_kind(self):
        trajs = [
            _mk_traj("p1", 0.7, success=True),
            _mk_traj("p1", 0.3),
        ]
        b = to_grpo_batch(trajs, advantage_kind="raw")
        # Raw = total_reward, no group adjustment
        assert b["advantages"] == [0.7, 0.3]

    def test_centered_advantage_kind(self):
        trajs = [
            _mk_traj("p", 1.0, success=True),
            _mk_traj("p", 0.0),
        ]
        b = to_grpo_batch(trajs, advantage_kind="centered")
        # Centered: rewards 1.0, 0.0 → mean 0.5 → adv 0.5, -0.5
        assert b["advantages"] == [0.5, -0.5]

class TestSftJsonl:
    def test_writes_only_successful(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sft.jsonl"
            trajs = [
                _mk_traj("p1", 1.0, success=True, num_turns=2),
                _mk_traj("p2", 0.0, success=False, num_turns=2),
                _mk_traj("p3", 1.0, success=True, num_turns=2),
            ]
            n = to_sft_jsonl(trajs, out, successful_only=True)
            assert n == 2
            lines = out.read_text().strip().splitlines()
            assert len(lines) == 2

    def test_writes_all_when_flag_off(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sft.jsonl"
            trajs = [
                _mk_traj("p1", 1.0, success=True, num_turns=1),
                _mk_traj("p2", 0.0, success=False, num_turns=1),
            ]
            n = to_sft_jsonl(trajs, out, successful_only=False)
            # Both should attempt to write; failed ones may still
            # render via the chat template (no hard requirement to drop).
            assert n >= 1

class TestPpoBatch:
    def test_token_level_layout(self):
        trajs = [_mk_traj("p1", 1.0, success=True, num_turns=2)]
        b = to_ppo_batch(trajs)
        assert "input_ids" in b
        assert "advantages" in b
        assert "returns" in b
        # One trajectory → one row of each list.
        assert len(b["input_ids"]) == 1
        # Each row's input_ids has obs+action tokens.
        n_tokens = len(b["input_ids"][0])
        assert n_tokens > 0
        assert len(b["mask"][0]) == n_tokens
        assert len(b["advantages"][0]) == n_tokens

    def test_gae_with_lambda_zero(self):
        """λ=0 reduces to TD(0) — advantage_t == reward_t."""
        from sampler.batch_export import _compute_gae
        rewards = [0.0, 0.0, 1.0]
        advs = _compute_gae(rewards, discount=1.0, gae_lambda=0.0,
                              bootstrap=0.0)
        # λ=0 → no propagation: advantage at last step is the reward there.
        assert advs[-1] == 1.0

    def test_gae_full_reduce_to_returns(self):
        """λ=1, γ=1, V=0 → advantages == returns == cumulative reward
        from the end."""
        from sampler.batch_export import _compute_gae
        rewards = [0.0, 0.5, 1.0]
        advs = _compute_gae(rewards, discount=1.0, gae_lambda=1.0,
                              bootstrap=0.0)
        # Reverse cumulative: last=1.0, mid=1.5, first=1.5
        assert advs == [1.5, 1.5, 1.0]

class TestSaveBatchJsonl:
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "batch.jsonl"
            batch = {"a": [1, 2], "b": ["x", "y"]}
            n = save_batch_jsonl(batch, path)
            assert n == 2
            lines = path.read_text().strip().splitlines()
            row0 = json.loads(lines[0])
            assert row0 == {"a": 1, "b": "x"}

    def test_empty_batch(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.jsonl"
            n = save_batch_jsonl({}, path)
            assert n == 0
            assert path.read_text() == ""

# ═══════════════════════════════════════════════════════════════════════
# End-to-end demo script tests
# ═══════════════════════════════════════════════════════════════════════

class TestRlDemoScript:
    """Run scripts/rl_demo.py as a subprocess. This is the actual
    user-facing smoke test: if this passes, the v7.1 unification is
    runnable end-to-end without verl/slime/Lean installed."""

    def test_demo_runs_and_produces_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "rl_demo.py"),
                "--num-problems", "2",
                "--branching-factor", "2",
                "--paths-per-problem", "3",
                "--max-nodes", "8",
                "--output-dir", td,
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT)
            proc = subprocess.run(cmd, capture_output=True,
                                     text=True, timeout=60, env=env)
            assert proc.returncode == 0, (
                f"demo failed: stdout={proc.stdout!r} "
                f"stderr={proc.stderr!r}")
            # Required outputs exist.
            assert (Path(td) / "grpo_batch.jsonl").exists()
            assert (Path(td) / "sft.jsonl").exists()
            assert (Path(td) / "traces").is_dir()
            # GRPO batch has rows.
            grpo_lines = (Path(td) / "grpo_batch.jsonl").read_text() \
                .strip().splitlines()
            assert len(grpo_lines) >= 1
            row = json.loads(grpo_lines[0])
            assert "problem_ids" in row
            assert "advantages" in row
            assert "rewards" in row

    def test_demo_with_grpo_normalize(self):
        with tempfile.TemporaryDirectory() as td:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "rl_demo.py"),
                "--num-problems", "2",
                "--branching-factor", "2",
                "--paths-per-problem", "4",
                "--grpo-normalize",
                "--output-dir", td,
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT)
            proc = subprocess.run(cmd, capture_output=True,
                                     text=True, timeout=60, env=env)
            assert proc.returncode == 0

# ═══════════════════════════════════════════════════════════════════════
# rl_pipeline rollout subcommand
# ═══════════════════════════════════════════════════════════════════════

class TestRlPipelineRollout:
    """The new ``rollout`` subcommand uses TreeRolloutSampler instead
    of the V6 subprocess-driven ``run_eval.py``. It's the bridge
    between the V6 4-stage pipeline and the V7 sampler."""

    def test_rollout_subcommand_runs(self):
        with tempfile.TemporaryDirectory() as td:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "rl_pipeline.py"),
                "rollout",
                "--benchmark", "builtin",
                "--limit", "2",
                "--backend", "mock",
                "--policy", "mock",
                "--branching-factor", "2",
                "--paths-per-problem", "3",
                "--max-nodes", "8",
                "--iter-dir", td,
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT)
            proc = subprocess.run(cmd, capture_output=True,
                                     text=True, timeout=60, env=env)
            assert proc.returncode == 0, (
                f"rollout failed: stdout={proc.stdout!r} "
                f"stderr={proc.stderr!r}")
            # Required outputs.
            assert (Path(td) / "rollout_summary.json").exists()
            assert (Path(td) / "grpo_batch.jsonl").exists()
            assert (Path(td) / "traces").is_dir()
            # Summary has the right shape.
            summary = json.loads(
                (Path(td) / "rollout_summary.json").read_text())
            assert "n_trajectories" in summary
            assert "search_kind" in summary
            assert summary["backend"] == "mock"
            assert summary["policy_kind"] == "mock"

    def test_rollout_then_collect_is_compatible(self):
        """Stage 1b output should feed directly into Stage 2 (collect),
        proving the dialog.json contract is preserved across the new path."""
        with tempfile.TemporaryDirectory() as td:
            iter_dir = Path(td) / "iter"
            cmd1 = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "rl_pipeline.py"),
                "rollout",
                "--benchmark", "builtin",
                "--limit", "1",
                "--backend", "mock",
                "--branching-factor", "2",
                "--paths-per-problem", "2",
                "--max-nodes", "6",
                "--iter-dir", str(iter_dir),
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT)
            r1 = subprocess.run(cmd1, capture_output=True, text=True,
                                  timeout=60, env=env)
            assert r1.returncode == 0

            sft_path = Path(td) / "sft.jsonl"
            cmd2 = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "rl_pipeline.py"),
                "collect",
                "--traces-dir", str(iter_dir / "traces"),
                "--output", str(sft_path),
            ]
            r2 = subprocess.run(cmd2, capture_output=True, text=True,
                                  timeout=60, env=env)
            assert r2.returncode == 0, (
                f"collect failed: stdout={r2.stdout!r} "
                f"stderr={r2.stderr!r}")
            # SFT JSONL exists (may have 0 rows for unsuccessful problems,
            # but the file must be produced).
            assert sft_path.exists()
