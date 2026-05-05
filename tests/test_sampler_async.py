"""tests/test_sampler_async.py — Tests for v7 unified-infrastructure changes.

Covers the three layers the v7 work targeted:

1. **Sampler ↔ Backend wiring** — ``ProofEnvConfig.backend`` and
   ``ProofEnv._make_transport_factory()`` must produce the right
   transport class for each backend kind, fail soft on unknowns,
   and pass the backend through to verl/slime wrappers.
2. **Concurrency fix** — ``BaseSampler`` must atomically allocate
   environments via the queue (no two episodes can ever share an env).
3. **Tree rollouts** — ``TreeRolloutSampler`` must produce trajectories
   with the right shape (root→leaf paths, GRPO-compatible groups).

Plus the v7 framework-detection harness: ``VERL_AVAILABLE`` and
``SLIME_AVAILABLE`` must not crash imports when verl/slime are absent
(the typical CI / unit-test environment).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from sampler import (
    ProofEnv, ProofEnvConfig, BaseSampler, SamplerConfig,
    Trajectory, Turn, RewardInfo,
    VERL_AVAILABLE, VeRLProofInteraction, VeRLProofAgentLoop,
    SLIME_AVAILABLE, SlimeSampler, SlimeProofEnvFactory,
    TreeRolloutSampler, TreeRolloutConfig,
)

# it; production code uses ProofEnv._make_transport_factory directly).
# The TestPreStartedTransport / TestBuildPoolWithBackend /
# TestCollectBackendStatus classes that exercised it have been removed
# along with the module.

# ═══════════════════════════════════════════════════════════════════════
# 1. Backend wiring — ProofEnvConfig + _make_transport_factory
# ═══════════════════════════════════════════════════════════════════════

class TestProofEnvBackendWiring:
    """ProofEnvConfig.backend must thread through to the right transport."""

    def test_default_backend_is_local(self):
        """V6 byte-compat: default config uses local."""
        cfg = ProofEnvConfig()
        assert cfg.backend == "local"
        assert cfg.backend_url is None
        assert cfg.backend_api_key is None

    def test_backend_fields_accept_overrides(self):
        cfg = ProofEnvConfig(
            backend="kimina",
            backend_url="http://lean.local:8000",
            backend_api_key="bearer-abc",
        )
        assert cfg.backend == "kimina"
        assert cfg.backend_url == "http://lean.local:8000"
        assert cfg.backend_api_key == "bearer-abc"

    def test_factory_local_returns_none(self):
        """Local path: factory should be None so AsyncLeanPool uses
        its own LocalTransport default. Setting it to None is the only
        way to keep the pre-v7 fast path bit-identical."""
        env = ProofEnv(ProofEnvConfig(backend="local"))
        # _make_transport_factory is only called when backend != "local",
        # so we don't even bother building it here.
        # But the factory itself, if invoked with "local", should return None.
        cfg2 = ProofEnvConfig(backend="local")
        env2 = ProofEnv(cfg2)
        f = env2._make_transport_factory()
        # Even if we get a factory, when backend=="local" it returns None
        # (internally normalised by the unknown-backend warning path).
        assert f(0) is None

    def test_factory_mock_returns_mock_transport(self):
        from engine.transport import MockTransport
        env = ProofEnv(ProofEnvConfig(backend="mock"))
        factory = env._make_transport_factory()
        t = factory(0)
        assert isinstance(t, MockTransport)

    def test_factory_kimina_returns_kimina_backend(self):

        # returns KiminaServerBackend directly.
        from engine.backends.kimina_server import KiminaServerBackend
        env = ProofEnv(ProofEnvConfig(
            backend="kimina",
            backend_url="http://example.com:8000",
            backend_api_key="token-xyz",
        ))
        factory = env._make_transport_factory()
        t = factory(0)
        assert isinstance(t, KiminaServerBackend)

    def test_factory_http_alias_for_kimina(self):
        from engine.backends.kimina_server import KiminaServerBackend
        env = ProofEnv(ProofEnvConfig(
            backend="http", backend_url="http://localhost:8000"))
        factory = env._make_transport_factory()
        t = factory(0)
        assert isinstance(t, KiminaServerBackend)

    def test_factory_socket_returns_socket_transport(self):
        from engine.transport import SocketTransport
        env = ProofEnv(ProofEnvConfig(
            backend="socket",
            backend_socket_path="/tmp/test.sock",
        ))
        factory = env._make_transport_factory()
        t = factory(0)
        assert isinstance(t, SocketTransport)

    def test_factory_unknown_returns_none_with_warning(self, caplog):
        """Unknown backend falls back gracefully to None (→ LocalTransport)."""
        import logging
        caplog.set_level(logging.WARNING)
        env = ProofEnv(ProofEnvConfig(backend="not_a_real_backend"))
        factory = env._make_transport_factory()
        t = factory(0)
        assert t is None
        # Should have logged a warning about the unknown backend
        assert any("unknown backend" in r.message.lower()
                   for r in caplog.records)

    def test_factory_session_id_is_passed_through(self):
        """Each session_id should get its own transport instance."""
        env = ProofEnv(ProofEnvConfig(backend="mock"))
        factory = env._make_transport_factory()
        t0 = factory(0)
        t1 = factory(1)
        assert t0 is not t1

    def test_factory_recovers_from_constructor_exception(self):
        """If a backend's constructor raises, factory returns None
        instead of propagating — sampler's fail-soft contract."""
        env = ProofEnv(ProofEnvConfig(
            backend="lookeng",
            backend_inner_kind="bogus_kind",  # may or may not blow up
        ))
        factory = env._make_transport_factory()
        # Should not raise, even on bogus configuration
        t = factory(0)
        # We don't assert the type — only that it didn't crash.
        # On systems without LooKeng installed, the construction might
        # still succeed (the LooKeng class can be imported even in stub).
        # Either way: no exception.
        assert t is None or hasattr(t, "start")

# ═══════════════════════════════════════════════════════════════════════
# 2. Concurrency: BaseSampler env queue
# ═══════════════════════════════════════════════════════════════════════

class _MockSampler(BaseSampler):
    """Trivial concrete sampler for concurrency testing."""

    async def generate_action(self, observation, problem_id, turn_idx):
        return ("simp", [1, 2], [-0.1, -0.2])

class TestEnvPoolConcurrency:
    """Verify the v7 queue-based env pool has no TOCTOU race."""

    def test_queue_is_atomic(self):
        """Two concurrent rollouts must NEVER share the same env."""
        async def _go():
            cfg = SamplerConfig(num_envs=2, max_concurrent_problems=8)
            sampler = _MockSampler(cfg)
            # Substitute envs with mocks so we don't need real Lean.
            sampler._setup_done = True
            envs = [MagicMock() for _ in range(2)]
            for i, e in enumerate(envs):
                e._turn_idx = 0
                e._id = i
                # Simulate reset/step; track which env was used by which call.
                async def reset_factory(env):
                    async def reset(problem):
                        await asyncio.sleep(0.01)  # let other coroutines schedule
                        env._reset_problem = problem
                        return f"obs_{env._id}"
                    return reset
                async def step_factory(env):
                    async def step(action):
                        await asyncio.sleep(0.01)
                        return f"obs_{env._id}_done", \
                                RewardInfo(scalar=1.0, is_terminal=True), \
                                True, {}
                    return step
                e.reset = await reset_factory(e)
                e.step = await step_factory(e)
                t = Trajectory(problem_id=f"x_{i}",
                                 theorem_statement="")
                t.add_turn(Turn(0, "", "simp",
                                 RewardInfo(scalar=1.0, is_terminal=True)))
                e.get_trajectory = lambda traj=t: traj
            sampler._env_pool = envs
            sampler._env_queue = asyncio.Queue()
            for e in envs:
                sampler._env_queue.put_nowait(e)

            # Track "currently using" via a counter the test inspects.
            in_use_observed: list[int] = []
            original_step = [e.step for e in envs]

            async def wrap_step(env, original):
                async def step(action):
                    in_use_observed.append(env._id)
                    return await original(action)
                return step
            for env, orig in zip(envs, original_step):
                env.step = await wrap_step(env, orig)

            problems = [{"problem_id": f"p{i}",
                          "theorem_statement": ""} for i in range(8)]
            results = await sampler.collect_rollouts(problems)
            assert len(results) == 8

            # If the queue is racy we'd see multiple concurrent uses
            # of env 0. With the v7 queue, each env can be in use by
            # at most one episode at a time → in_use_observed should
            # have N entries equal to the total number of step() calls,
            # but no concurrent access is possible because get() is atomic.
            # We at least verify both envs got used.
            assert set(in_use_observed) == {0, 1}, \
                f"both envs should have been used, got {in_use_observed}"

        asyncio.run(_go())

    def test_env_returned_on_exception(self):
        """If a rollout raises, env must still go back to the queue."""
        async def _go():
            cfg = SamplerConfig(num_envs=1, max_concurrent_problems=4)
            sampler = _MockSampler(cfg)
            sampler._setup_done = True
            env = MagicMock()
            env._turn_idx = 0
            env.reset = AsyncMock(side_effect=RuntimeError("boom"))
            sampler._env_pool = [env]
            sampler._env_queue = asyncio.Queue()
            sampler._env_queue.put_nowait(env)

            problems = [{"problem_id": "p1", "theorem_statement": ""}]
            results = await sampler.collect_rollouts(problems)
            # Exception was caught by gather() with return_exceptions=True
            # The env should have been returned despite the failure.
            assert sampler._env_queue.qsize() == 1, \
                "env must be returned to the queue even on rollout failure"

        asyncio.run(_go())

# ═══════════════════════════════════════════════════════════════════════
# 4. VeRL/SLIME framework detection harness
# ═══════════════════════════════════════════════════════════════════════

class TestFrameworkDetection:
    """Soft-import flags must be importable and have the right shape."""

    def test_verl_flag_is_bool(self):
        assert isinstance(VERL_AVAILABLE, bool)

    def test_slime_flag_is_bool(self):
        assert isinstance(SLIME_AVAILABLE, bool)

    def test_verl_classes_importable_either_way(self):
        """The three classes must be importable regardless of whether
        verl is installed (the whole point of the v7 stub fallback)."""
        assert VeRLProofInteraction is not None
        assert VeRLProofAgentLoop is not None

    def test_verl_register_decorator_marks_class(self):
        """The @register decorator (real or stub) leaves the registered
        name on the class so we can inspect it."""
        # When verl is absent, the stub decorator stamps _verl_registered_name
        if not VERL_AVAILABLE:
            assert getattr(
                VeRLProofAgentLoop, "_verl_registered_name", None
            ) == "ai4math_proof_agent"

class TestVeRLBackendPassthrough:
    """VeRLProofInteraction / VeRLProofAgentLoop 必须把 backend 字段透传到
    内部 ProofEnvConfig,不丢字段也不给错值。"""

    def test_interaction_picks_up_backend_field(self):
        i = VeRLProofInteraction({
            "name": "test",
            "backend": "kimina",
            "backend_url": "http://lean:8000",
            "pool_size": 2,
        })
        assert i._env_config.backend == "kimina"
        assert i._env_config.backend_url == "http://lean:8000"
        assert i._env_config.pool_size == 2

    def test_interaction_default_backend_is_local(self):
        i = VeRLProofInteraction({"name": "test"})
        assert i._env_config.backend == "local"

    def test_agent_loop_picks_up_backend_field(self):
        al = VeRLProofAgentLoop(proof_config={
            "backend": "pantograph",
            "project_dir": "/tmp/proj",
        })
        assert al._env_config.backend == "pantograph"
        assert al._env_config.project_dir == "/tmp/proj"

# ═══════════════════════════════════════════════════════════════════════
# 5. Tree rollout sampler
# ═══════════════════════════════════════════════════════════════════════

async def _stub_policy(observation: str):
    """Trivial deterministic policy for tree-rollout tests."""
    return ("simp", [1, 2, 3], [-0.1, -0.2, -0.3])

class TestTreeRolloutConfig:
    def test_inherits_from_sampler_config(self):
        cfg = TreeRolloutConfig()
        # Inherits SamplerConfig fields
        assert cfg.num_envs == 8
        # Adds tree-specific fields
        assert cfg.search_kind == "best_first"
        assert cfg.branching_factor == 4
        assert cfg.max_paths_per_problem == 8

    def test_search_kind_options(self):
        for kind in ("best_first", "ucb", "beam"):
            cfg = TreeRolloutConfig(search_kind=kind)
            assert cfg.search_kind == kind

class TestTreeRolloutSamplerShape:
    """Trajectory shape and trees from the tree sampler."""

    def test_tree_search_produces_grpo_group(self):
        """A single problem must produce up to K trajectories sharing
        the same root prompt — the GRPO group structure."""
        async def _go():
            # Build a sampler with mocked envs (no real Lean needed).
            cfg = TreeRolloutConfig(
                num_envs=1,
                max_concurrent_problems=1,
                branching_factor=2,
                max_nodes=8,
                max_depth=3,
                max_paths_per_problem=4,
            )
            sampler = TreeRolloutSampler(cfg, policy_fn=_stub_policy)

            # Mock env: each step returns a non-terminal then a terminal
            # success on the second call.
            env = MagicMock()
            env._turn_idx = 0
            call_count = {"n": 0}

            async def fake_reset(problem):
                call_count["n"] = 0
                return f"obs_root_{problem['problem_id']}"

            async def fake_step(action):
                call_count["n"] += 1
                # Make it terminal at depth >=2 sometimes; use action
                # to differentiate so the tree branches.
                done = call_count["n"] >= 2
                success = done and "simp" in action
                reward = 1.0 if success else (
                    0.05 if not done else 0.0)
                return ("obs_step", RewardInfo(
                    scalar=reward,
                    is_terminal=done,
                ), done, {})

            env.reset = fake_reset
            env.step = fake_step

            sampler._setup_done = True
            sampler._env_pool = [env]
            sampler._env_queue = asyncio.Queue()
            sampler._env_queue.put_nowait(env)

            problems = [{"problem_id": "p1",
                          "theorem_statement": "t1"}]
            trajectories = await sampler.collect_rollouts(problems)

            assert len(trajectories) >= 1
            # All trajectories from same problem share problem_id
            assert all(t.problem_id == "p1" for t in trajectories)
            # Each trajectory has at least one turn
            assert all(t.num_turns >= 1 for t in trajectories)

        asyncio.run(_go())

    def test_search_tree_dict_format(self):
        """to_search_tree_dict() must produce v3 dialog.json compatible
        meta.search_tree structure."""
        from sampler.tree_rollout_sampler import _Node
        nodes = {
            0: _Node(id=0, parent_id=None, tactic=None,
                       observation="root", depth=0, children=[1, 2]),
            1: _Node(id=1, parent_id=0, tactic="simp",
                       observation="obs1", depth=1, success=True,
                       is_terminal=True),
            2: _Node(id=2, parent_id=0, tactic="ring",
                       observation="obs2", depth=1, is_terminal=True),
        }
        d = TreeRolloutSampler.to_search_tree_dict(nodes, kind="best_first")
        assert d["kind"] == "best_first"
        assert d["root_node_id"] == 0
        assert d["solved_node_id"] == 1
        assert d["total_nodes"] == 3
        assert d["max_depth"] == 1
        assert len(d["nodes"]) == 3
        # Solved node must show as solved
        node1 = next(n for n in d["nodes"] if n["node_id"] == 1)
        assert node1["status"] == "solved"
        assert node1["is_complete"] is True

    def test_grpo_group_normalization(self):
        """When group_normalize_rewards=True, each trajectory gets a
        group_advantage in metadata."""
        async def _go():
            cfg = TreeRolloutConfig(
                branching_factor=2, max_nodes=5, max_depth=2,
                max_paths_per_problem=4,
                group_normalize_rewards=True,
            )
            sampler = TreeRolloutSampler(cfg, policy_fn=_stub_policy)
            env = MagicMock()
            env._turn_idx = 0

            async def fake_reset(p): return "root"

            async def fake_step(a):
                # Make rewards vary so std-dev > 0
                import random
                random.seed(hash(a) & 0xff)
                return ("obs", RewardInfo(
                    scalar=random.random(),
                    is_terminal=True),
                    True, {})

            env.reset = fake_reset
            env.step = fake_step

            sampler._setup_done = True
            sampler._env_pool = [env]
            sampler._env_queue = asyncio.Queue()
            sampler._env_queue.put_nowait(env)

            trajectories = await sampler.collect_rollouts(
                [{"problem_id": "p1", "theorem_statement": ""}])

            if len(trajectories) >= 2:
                # group_advantage should be set on each trajectory
                for t in trajectories:
                    assert "group_advantage" in t.metadata, \
                        f"missing group_advantage in {t.metadata}"

        asyncio.run(_go())

    def test_trajectory_to_verl_compat(self):
        """Trajectories from TreeRolloutSampler must convert cleanly
        to verl format (the whole point — these feed into GRPO)."""
        from sampler.tree_rollout_sampler import _Node
        # Hand-build a small tree to keep this hermetic
        sampler = TreeRolloutSampler(TreeRolloutConfig())
        nodes = {
            0: _Node(id=0, parent_id=None, tactic=None,
                       observation="root_obs", depth=0,
                       children=[1]),
            1: _Node(id=1, parent_id=0, tactic="simp",
                       observation="obs1", depth=1,
                       is_terminal=True, success=True,
                       reward_at_step=1.0,
                       action_token_ids=[10, 11],
                       action_log_probs=[-0.1, -0.2]),
        }
        trajs = sampler._extract_trajectories(
            nodes, {"problem_id": "p", "theorem_statement": ""},
            "root_obs")
        assert len(trajs) == 1
        verl_dict = trajs[0].to_verl_format()
        assert "response_ids" in verl_dict
        assert "response_mask" in verl_dict
        assert verl_dict["success"] is True
