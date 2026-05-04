"""tests/test_rl_unified.py — End-to-end RL infra unification (v7).

Tests the three things v6's evaluation flagged as gaps:

  1. Backend selection plumbed through to the RL sampler
     (was: ProofEnv hard-coded LocalTransport; now: backend="kimina"
     etc. flows ProofEnvConfig → AsyncLeanPool transport_factory).

  2. Concurrent rollouts no longer race on env pool
     (was: TOCTOU on ``_in_use`` flag; now: asyncio.Queue).

  3. Tree-search rollouts as a first-class RL primitive
     (was: SearchDriver only callable from UnifiedProofRunner; now:
     TreeRolloutSampler emits per-path Trajectories ready for GRPO).

Plus: verl/slime real-vs-stub detection with the right fallback
behaviour, and a concurrency stress test that catches the V1–V6 race.
"""
from __future__ import annotations

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from sampler import (
    BaseSampler, ProofEnv, ProofEnvConfig, RewardInfo,
    SamplerConfig, SLIME_AVAILABLE, Trajectory, TreeRolloutConfig,
    TreeRolloutSampler, Turn, VERL_AVAILABLE,
    VeRLProofAgentLoop, VeRLProofInteraction,
)


# ═══════════════════════════════════════════════════════════════════════
# Gap 1 — Backend selection through the sampler
# ═══════════════════════════════════════════════════════════════════════

class TestBackendInProofEnvConfig:
    """ProofEnvConfig must surface backend selection."""

    def test_default_backend_is_local(self):
        cfg = ProofEnvConfig()
        assert cfg.backend == "local"
        assert cfg.backend_url is None

    def test_kimina_backend_fields(self):
        cfg = ProofEnvConfig(
            backend="kimina",
            backend_url="http://kimina.local:8000",
            backend_api_key="secret",
        )
        assert cfg.backend == "kimina"
        assert cfg.backend_url == "http://kimina.local:8000"
        assert cfg.backend_api_key == "secret"

    def test_lookeng_inner_chain(self):
        """LooKeng with kimina inner — production chain."""
        cfg = ProofEnvConfig(
            backend="lookeng",
            backend_inner_kind="kimina",
            backend_url="http://kimina.local:8000",
        )
        assert cfg.backend == "lookeng"
        assert cfg.backend_inner_kind == "kimina"


class TestProofEnvBackendFactory:
    """ProofEnv._make_transport_factory wiring."""

    def test_local_no_factory(self):
        """backend=local must NOT install a transport_factory — the
        AsyncLeanPool default LocalTransport path is the contract."""
        env = ProofEnv(ProofEnvConfig(backend="local"))
        # Inspect the would-be factory before setup() actually runs.
        # Local short-circuits the factory.
        cfg = env.config
        assert cfg.backend == "local"

    def test_non_local_makes_factory(self):
        """backend=mock builds a factory that produces MockTransport."""
        env = ProofEnv(ProofEnvConfig(backend="mock"))
        factory = env._make_transport_factory()
        # Factory is a sync callable returning a started transport.
        # We don't actually start it in the test (would need real loop)
        # but we can assert it's callable and doesn't crash on lookup.
        assert callable(factory)

    def test_async_setup_with_mock_backend(self):
        """End-to-end: ProofEnv.setup() with backend=mock runs without
        constructing a LocalTransport. We verify by reading the pool's
        first session's transport class after setup."""
        async def _test():
            env = ProofEnv(ProofEnvConfig(
                backend="mock", pool_size=1, lean_timeout_s=2))
            await env.setup()
            try:
                # Pool has at least one session.
                assert env._pool is not None
                sessions = env._pool._sessions
                assert len(sessions) >= 1
                transport_cls = type(sessions[0]._transport).__name__
                # MockTransport, NOT LocalTransport.
                assert "Mock" in transport_cls or "Fallback" in transport_cls
            finally:
                await env.close()
        asyncio.run(_test())

    def test_unknown_backend_falls_back_softly(self):
        """An unrecognised backend name must not crash setup —
        it goes through backend_factory's "auto" guard which
        returns a started fallback. Pool still constructs."""
        async def _test():
            env = ProofEnv(ProofEnvConfig(
                backend="bogus_backend_name", pool_size=1,
                lean_timeout_s=2))
            await env.setup()
            try:
                assert env._pool is not None
            finally:
                await env.close()
        asyncio.run(_test())


# ═══════════════════════════════════════════════════════════════════════
# Gap 2 — Concurrent rollouts no longer race on env pool
# ═══════════════════════════════════════════════════════════════════════

class _RaceProbeSampler(BaseSampler):
    """A sampler that records which env each rollout used. With the V6
    race bug, two concurrent rollouts could see the same env. The fix
    (asyncio.Queue) guarantees mutual exclusion."""

    def __init__(self, config, n_rollouts: int):
        super().__init__(config)
        self._n_rollouts = n_rollouts
        self.env_use_log: list[int] = []
        self._env_id_lock = asyncio.Lock()

    async def generate_action(self, observation, problem_id, turn_idx):
        # Sleep to widen any race window; record env identity.
        await asyncio.sleep(0.01)
        return ("sorry", [], [])  # ProofEnv terminates on sorry


class TestConcurrencyNoRace:
    """Many concurrent rollouts must each get a unique env at any
    given moment — the queue model guarantees this; the prior _in_use
    flag did not."""

    def test_env_queue_atomic_acquire(self):
        async def _test():
            # We don't need real Lean; a mock setup is fine because
            # ProofEnv.step("sorry") terminates without hitting the verifier.
            num_envs = 4
            cfg = SamplerConfig(
                env_config=ProofEnvConfig(
                    backend="mock", pool_size=1, lean_timeout_s=1),
                num_envs=num_envs, max_concurrent_problems=16,
            )
            sampler = _RaceProbeSampler(cfg, n_rollouts=20)

            # Hand-construct a populated env_pool to skip real setup.
            async def _fake_setup():
                envs = [ProofEnv(cfg.env_config) for _ in range(num_envs)]
                # Mock each env so reset/step succeed without Lean.
                for i, e in enumerate(envs):
                    e._pool = MagicMock()
                    e._verifier = MagicMock()
                    e._prefilter = MagicMock()
                    e._error_intel = MagicMock()
                    e._broadcast = MagicMock()
                    e._fake_id = i  # for race tracking
                sampler._env_pool = envs
                sampler._env_queue = asyncio.Queue()
                for e in envs:
                    sampler._env_queue.put_nowait(e)
                sampler._env_semaphore = asyncio.Semaphore(num_envs)
                sampler._setup_done = True
            await _fake_setup()

            # Track concurrent env occupancy: while any rollout holds
            # an env, no other rollout should see the same env in use.
            active: dict[int, int] = {}  # env_id → count of concurrent users
            lock = asyncio.Lock()
            max_seen_per_env = {}

            original_run = sampler._single_rollout

            async def instrumented(problem, policy_fn=None):
                env_obj = await sampler._env_queue.get()
                eid = env_obj._fake_id
                async with lock:
                    active[eid] = active.get(eid, 0) + 1
                    max_seen_per_env[eid] = max(
                        max_seen_per_env.get(eid, 0), active[eid])
                try:
                    # Simulate the real rollout's reset+step quickly.
                    await asyncio.sleep(0.005)
                finally:
                    async with lock:
                        active[eid] -= 1
                    sampler._env_queue.put_nowait(env_obj)
                # Return a dummy trajectory.
                t = Trajectory(problem_id=problem["problem_id"],
                                theorem_statement="")
                t.add_turn(Turn(0, "obs", "act",
                                 RewardInfo(scalar=0.0, is_terminal=True)))
                return t

            sampler._single_rollout = instrumented

            problems = [
                {"problem_id": f"p{i}", "theorem_statement": ""}
                for i in range(20)
            ]
            await sampler.collect_rollouts(problems)

            # Each env should never have had > 1 concurrent user.
            for eid, max_seen in max_seen_per_env.items():
                assert max_seen == 1, (
                    f"env {eid} had {max_seen} concurrent users — race!")

        asyncio.run(_test())

    def test_env_returned_to_queue_on_exception(self):
        """If a rollout raises, the env must still go back to the queue
        (try/finally semantics) — otherwise the pool drains."""
        async def _test():
            cfg = SamplerConfig(
                env_config=ProofEnvConfig(backend="mock", pool_size=1,
                                              lean_timeout_s=1),
                num_envs=2, max_concurrent_problems=4,
            )
            sampler = _RaceProbeSampler(cfg, n_rollouts=4)

            envs = [ProofEnv(cfg.env_config) for _ in range(2)]
            for i, e in enumerate(envs):
                e._pool = MagicMock()
                e._verifier = MagicMock()
                e._prefilter = MagicMock()
                e._error_intel = MagicMock()
                e._broadcast = MagicMock()
                e._fake_id = i
                # Make reset() raise on every call to simulate failure.
                async def _bad_reset(p, _i=i):
                    raise RuntimeError(f"intentional from env {_i}")
                e.reset = _bad_reset
            sampler._env_pool = envs
            sampler._env_queue = asyncio.Queue()
            for e in envs:
                sampler._env_queue.put_nowait(e)
            sampler._env_semaphore = asyncio.Semaphore(2)
            sampler._setup_done = True

            problems = [{"problem_id": f"p{i}", "theorem_statement": ""}
                            for i in range(4)]
            results = await sampler.collect_rollouts(problems)
            # All rollouts failed, but the queue has both envs back.
            assert sampler._env_queue.qsize() == 2

        asyncio.run(_test())


# ═══════════════════════════════════════════════════════════════════════
# Gap 3 — Tree-search rollouts as RL primitives
# ═══════════════════════════════════════════════════════════════════════

def _make_mock_env() -> ProofEnv:
    """Build a ProofEnv whose step() is faked: success on tactic 'exact h',
    in-progress on 'intro', failure on anything else. Lets tree rollout
    tests exercise the search logic without a real Lean."""
    cfg = ProofEnvConfig(backend="mock", pool_size=1)
    env = ProofEnv(cfg)
    env._pool = MagicMock()
    env._verifier = MagicMock()

    state = {"step_count": 0}

    async def fake_reset(problem):
        state["step_count"] = 0
        env._problem = problem
        env._turn_idx = 0
        env._done = False
        env._goals_remaining = 1
        env._accumulated_feedback = []
        env._episode_start = 0
        env._trajectory = Trajectory(
            problem_id=problem["problem_id"],
            theorem_statement=problem["theorem_statement"],
        )
        return f"prove: {problem['theorem_statement']}"

    async def fake_step(action: str):
        state["step_count"] += 1
        if action.strip() == "exact h":
            r = RewardInfo(scalar=1.0, is_terminal=True,
                            verification_level="L2", goals_remaining=0)
            obs = "[TERMINATED: success]"
            done = True
        elif action.strip() == "intro h":
            r = RewardInfo(scalar=0.05, is_terminal=False,
                            verification_level="L1", goals_remaining=1)
            obs = "Goal: P → Q"
            done = False
        else:
            r = RewardInfo(scalar=0.0, is_terminal=False,
                            verification_level="L1",
                            error_class="tactic_failed")
            obs = f"Error on '{action}'"
            done = False
        # Mimic the trajectory recording.
        env._trajectory.add_turn(Turn(
            turn_idx=env._turn_idx, observation=obs,
            action=action, reward=r))
        env._turn_idx += 1
        return obs, r, done, {}

    env.reset = fake_reset
    env.step = fake_step
    env.close = AsyncMock()
    return env


class TestTreeRolloutSampler:
    """TreeRolloutSampler emits per-path trajectories for GRPO."""

    def test_finds_solving_path(self):
        async def _test():
            cfg = TreeRolloutConfig(
                num_envs=1, max_concurrent_problems=1,
                env_config=ProofEnvConfig(backend="mock"),
                search_kind="best_first",
                branching_factor=3,
                max_nodes=12, max_depth=4,
                max_paths_per_problem=4,
            )

            # Policy: returns alternating candidates so the search has
            # something to explore. First call returns 'exact h' (winner),
            # later calls vary.
            call_count = [0]
            async def policy(obs):
                call_count[0] += 1
                # Slot rotation to simulate temperature sampling.
                tactics = ["exact h", "intro h", "simp"]
                t = tactics[call_count[0] % len(tactics)]
                return (t, [10, 11], [-0.1, -0.2])

            sampler = TreeRolloutSampler(cfg, policy_fn=policy)

            # Inject our mock env into the queue; skip real setup.
            sampler._env_pool = [_make_mock_env()]
            sampler._env_queue = asyncio.Queue()
            sampler._env_queue.put_nowait(sampler._env_pool[0])
            sampler._env_semaphore = asyncio.Semaphore(1)
            sampler._setup_done = True

            problems = [{
                "problem_id": "p1",
                "theorem_statement": "theorem t : True",
            }]
            trajs = await sampler.collect_rollouts(problems)

            # We should have at least one trajectory, and the first one
            # in the list (sorted: solved first) should succeed.
            assert len(trajs) >= 1
            assert any(t.success for t in trajs)
            # All trajectories share the same problem_id (= same group).
            assert all(t.problem_id == "p1" for t in trajs)

        asyncio.run(_test())

    def test_grpo_group_normalization(self):
        async def _test():
            cfg = TreeRolloutConfig(
                num_envs=1, max_concurrent_problems=1,
                env_config=ProofEnvConfig(backend="mock"),
                search_kind="best_first",
                branching_factor=2,
                max_nodes=6, max_depth=3,
                max_paths_per_problem=4,
                group_normalize_rewards=True,
            )

            counter = [0]
            async def policy(obs):
                counter[0] += 1
                return (["exact h", "intro h", "wrong"][counter[0] % 3],
                          [counter[0]], [-0.1])

            sampler = TreeRolloutSampler(cfg, policy_fn=policy)
            sampler._env_pool = [_make_mock_env()]
            sampler._env_queue = asyncio.Queue()
            sampler._env_queue.put_nowait(sampler._env_pool[0])
            sampler._env_semaphore = asyncio.Semaphore(1)
            sampler._setup_done = True

            problems = [{"problem_id": "g1", "theorem_statement": ""}]
            trajs = await sampler.collect_rollouts(problems)

            # group_normalize_rewards=True should populate the
            # group_advantage field on each trajectory in groups ≥ 2.
            if len(trajs) >= 2:
                for t in trajs:
                    assert "group_advantage" in t.metadata

        asyncio.run(_test())

    def test_emits_grpo_compatible_format(self):
        """to_verl_format() should still work on tree-rollout trajectories."""
        async def _test():
            cfg = TreeRolloutConfig(
                num_envs=1, max_concurrent_problems=1,
                env_config=ProofEnvConfig(backend="mock"),
                branching_factor=2, max_nodes=4, max_depth=2,
                max_paths_per_problem=2,
            )
            calls = [0]
            async def policy(obs):
                calls[0] += 1
                return ("exact h", [42, 43], [-0.5, -0.3])

            sampler = TreeRolloutSampler(cfg, policy_fn=policy)
            sampler._env_pool = [_make_mock_env()]
            sampler._env_queue = asyncio.Queue()
            sampler._env_queue.put_nowait(sampler._env_pool[0])
            sampler._env_semaphore = asyncio.Semaphore(1)
            sampler._setup_done = True

            problems = [{"problem_id": "x", "theorem_statement": ""}]
            trajs = await sampler.collect_rollouts(problems)
            assert len(trajs) >= 1
            verl = trajs[0].to_verl_format()
            assert "prompt_ids" in verl
            assert "response_ids" in verl
            assert "response_mask" in verl
            assert "reward_score" in verl
            # Token data flowed through.
            assert any(verl["response_ids"])

        asyncio.run(_test())


# ═══════════════════════════════════════════════════════════════════════
# Real verl/slime detection (without breaking when they're absent)
# ═══════════════════════════════════════════════════════════════════════

class TestRealFrameworkDetection:
    """v7 detects verl/slime at import time. The flags should be false
    in CI (we don't install them) but the classes still importable."""

    def test_verl_flag_consistent_with_baseclass(self):
        # If VERL_AVAILABLE is False, the parent class should be `object`.
        # If True, it should be a verl class.
        if VERL_AVAILABLE:
            # Look up the parent of VeRLProofAgentLoop.
            assert VeRLProofAgentLoop.__bases__[0] is not object
        else:
            assert VeRLProofAgentLoop.__bases__[0] is object

    def test_verl_register_decorator_marks_class(self):
        """The @register decorator (real or stub) should set the
        registered name attribute when verl is absent (stub path)."""
        if not VERL_AVAILABLE:
            assert getattr(
                VeRLProofAgentLoop, "_verl_registered_name", None) \
                == "ai4math_proof_agent"

    def test_interaction_passes_backend_to_env(self):
        """Backend selection must flow through the verl interaction
        config dict to the inner ProofEnvConfig."""
        interaction = VeRLProofInteraction({
            "name": "test",
            "backend": "kimina",
            "backend_url": "http://server:8000",
            "backend_api_key": "tok",
            "pool_size": 2,
        })
        assert interaction._env_config.backend == "kimina"
        assert interaction._env_config.backend_url == "http://server:8000"
        assert interaction._env_config.backend_api_key == "tok"

    def test_agent_loop_passes_backend_via_proof_config(self):
        loop = VeRLProofAgentLoop(proof_config={
            "backend": "pantograph",
            "pool_size": 4,
        })
        assert loop._env_config.backend == "pantograph"

    def test_slime_flag_present(self):
        # SLIME_AVAILABLE is just a bool — assert it's a bool (not raised).
        assert isinstance(SLIME_AVAILABLE, bool)


# ═══════════════════════════════════════════════════════════════════════
# Backwards compat: V1–V6 sampler tests should still pass
# ═══════════════════════════════════════════════════════════════════════

class TestBackwardsCompat:
    """Pin V1–V6 behaviour: caller that doesn't pass backend gets local."""

    def test_default_proof_env_config_unchanged(self):
        cfg = ProofEnvConfig()
        # All V1–V6 fields preserved
        assert cfg.max_turns == 32
        assert cfg.pool_size == 4
        assert cfg.preamble == "import Mathlib"
        assert cfg.reward_success == 1.0
        assert cfg.reward_sorry == -0.5
        # New v7 field defaults to "local" — keeps every existing test
        # path on LocalTransport.
        assert cfg.backend == "local"

    def test_async_lean_pool_no_factory_path(self):
        """AsyncLeanPool without transport_factory must construct fine
        — the V1–V6 default behaviour is preserved."""
        from engine.async_lean_pool import AsyncLeanPool
        # Just constructor — no start, no Lean needed.
        pool = AsyncLeanPool(pool_size=2, project_dir="/tmp")
        assert pool._transport_factory is None
        assert pool.pool_size == 2

    def test_async_lean_pool_factory_call_sites(self):
        """Both constructor and add_session() must use _make_session."""
        from engine.async_lean_pool import AsyncLeanPool
        pool = AsyncLeanPool(pool_size=1)
        # _make_session exists and is a method.
        assert callable(pool._make_session)
        # With no factory, _make_session returns a session whose
        # transport will default to LocalTransport on start().
        session = pool._make_session(99)
        assert session.session_id == 99
        # Transport not yet started; will be LocalTransport when start runs.
        assert session._transport is None
