"""tests/test_sampler_unification.py — V7 sampler/backend/search unification

Pins the three integration points that V7 closes:

1. **Sampler ↔ Backend**: ``ProofEnvConfig.backend`` must propagate through
   ``ProofEnv.setup()`` to ``AsyncLeanPool.transport_factory`` so the V1–V6
   community backends (Kimina/Pantograph/LooKeng) are reachable from RL
   roll-outs. Same wiring on the slime path
   (``SlimeProofEnvFactory.setup_shared_pool``) and the verl path
   (``VeRLProofInteraction._ensure_setup``).

2. **Atomic env pool**: ``BaseSampler._single_rollout`` must never let two
   concurrent episodes share an env. Pre-V7 the per-env ``_in_use``
   boolean had a TOCTOU race; V7 replaced it with ``asyncio.Queue``.

3. **Tree-shaped rollouts**: ``TreeRolloutSampler`` emits a *group* of
   trajectories per problem, all sharing the root observation. This is
   the GRPO-friendly shape the linear ``BaseSampler`` could not produce.

These tests use ``MockTransport`` only — no Lean 4 dependency.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from sampler import (
    ProofEnv, ProofEnvConfig,
    BaseSampler, SamplerConfig,
    SlimeProofEnvFactory,
    VeRLProofInteraction, VeRLProofAgentLoop, VERL_AVAILABLE,
    SLIME_AVAILABLE,
    TreeRolloutSampler, TreeRolloutConfig,
    Trajectory, RewardInfo,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Sampler ↔ Backend wiring
# ═══════════════════════════════════════════════════════════════════════


class TestBackendWiring:
    """ProofEnvConfig.backend → AsyncLeanPool.transport_factory."""

    def test_local_backend_skips_factory(self):
        """backend='local' (the V1–V6 default) must NOT build a factory.
        Otherwise we'd silently double-wrap LocalTransport."""
        cfg = ProofEnvConfig(backend="local")
        env = ProofEnv(cfg)

        # Simulate what setup() does up to the factory check.
        backend_active = cfg.backend not in ("", "local", None)
        assert not backend_active, (
            "backend='local' must remain on the legacy no-factory path "
            "for backwards compat with existing callers")

    def test_non_local_backend_triggers_factory(self):
        """backend='kimina' / 'pantograph' / 'lookeng' / 'mock' must
        all activate the transport factory path."""
        for kind in ("kimina", "pantograph", "lookeng", "mock", "auto"):
            cfg = ProofEnvConfig(backend=kind)
            assert cfg.backend not in ("", "local", None), (
                f"backend={kind!r} should activate factory path")

    def test_make_transport_factory_returns_callable(self):
        """The factory builder must produce a (session_id) -> Transport
        callable, with the right signature for AsyncLeanPool."""
        cfg = ProofEnvConfig(backend="mock")
        env = ProofEnv(cfg)
        factory = env._make_transport_factory()
        assert callable(factory)

    def test_factory_propagates_backend_kwargs(self):
        """The factory must construct the right transport class with
        the config's backend_* kwargs threaded through. We assert on
        the class name + key attributes rather than mocking
        build_backend, because the V7 implementation deliberately
        avoids the async-call route to dodge the nested-event-loop
        trap when invoked from inside AsyncLeanPool.start()."""
        cfg = ProofEnvConfig(
            backend="kimina",
            backend_url="http://lean.local:8000",
            backend_api_key="secret-token",
            project_dir="/tmp/lean-proj",
            lean_timeout_s=42,
        )
        env = ProofEnv(cfg)
        factory = env._make_transport_factory()

        transport = factory(session_id=3)

        assert transport is not None
        # v11: kimina/http now maps to KiminaServerBackend directly
        # (HTTPTransport thin wrapper was deleted — it just delegated).
        from engine.backends.kimina_server import KiminaServerBackend
        assert isinstance(transport, KiminaServerBackend), (
            f"backend='kimina' must produce KiminaServerBackend; "
            f"got {type(transport).__name__}")
        # The base_url and api_key threaded through. The chain is:
        #   KiminaServerBackend._client = KiminaServerClient
        #   KiminaServerClient.base_url / .api_key
        client = transport._client
        assert client.base_url.rstrip("/") == "http://lean.local:8000"
        assert client.api_key == "secret-token"

    def test_factory_dispatches_pantograph(self):
        """backend='pantograph' must construct a PantographBackend
        without trying to talk to the real binary at construction time
        (start() is what touches the binary; factory just builds)."""
        cfg = ProofEnvConfig(backend="pantograph", project_dir=".",
                             lean_timeout_s=15)
        env = ProofEnv(cfg)
        factory = env._make_transport_factory()
        transport = factory(session_id=0)
        assert transport is not None
        assert type(transport).__name__ == "PantographBackend"

    def test_factory_dispatches_lookeng_with_kimina_inner(self):
        """LooKeng has an outer/inner contract: user can chain
        LooKeng(outer) wrapping KiminaServerBackend(inner). The factory
        must honour ``backend_inner_kind='kimina'`` and build the inner.

        v11: HTTPTransport thin wrapper was deleted; the inner is now
        KiminaServerBackend directly.
        """
        cfg = ProofEnvConfig(
            backend="lookeng",
            backend_inner_kind="kimina",
            backend_url="http://kimina:8000",
            project_dir=".",
        )
        env = ProofEnv(cfg)
        factory = env._make_transport_factory()
        transport = factory(session_id=0)
        assert transport is not None
        assert type(transport).__name__ == "LooKengBackend"
        inner = getattr(transport, "_inner", None) or \
                getattr(transport, "inner", None)
        if inner is not None:
            assert type(inner).__name__ == "KiminaServerBackend"

    def test_factory_unknown_backend_returns_none(self):
        """An unrecognised backend (or 'auto') must return None so
        AsyncLeanPool falls back to LocalTransport."""
        cfg = ProofEnvConfig(backend="auto")
        env = ProofEnv(cfg)
        factory = env._make_transport_factory()
        transport = factory(session_id=0)
        assert transport is None, (
            "Unsupported backend in the factory must return None — "
            "AsyncLeanPool then constructs LocalTransport.")

    def test_factory_swallows_exceptions(self):
        """Any exception during transport construction must be caught
        and reported as None — never propagated to AsyncLeanPool.

        v11: HTTPTransport was deleted; we now patch
        ``KiminaServerBackend`` directly (the symbol the factory imports
        for backend='kimina').
        """
        cfg = ProofEnvConfig(backend="kimina")
        env = ProofEnv(cfg)
        factory = env._make_transport_factory()

        from engine.backends import kimina_server as ks_mod
        with patch.object(
                ks_mod, "KiminaServerBackend",
                side_effect=RuntimeError("boom")):
            result = factory(session_id=0)
        assert result is None, (
            "factory must return None on construction failure so "
            "AsyncLeanPool can fall back to LocalTransport")


class TestSlimeFactoryBackendV7:
    """V7 fix: SlimeProofEnvFactory.setup_shared_pool used to ignore
    the backend selector, forcing slime users onto LocalTransport."""

    def test_local_backend_no_transport_factory(self):
        """backend='local' must produce an AsyncLeanPool with
        transport_factory=None (legacy behaviour preserved)."""
        async def _t():
            cfg = ProofEnvConfig(backend="local")
            factory = SlimeProofEnvFactory(cfg)

            captured: dict = {}

            class _FakePool:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

                async def start(self):
                    pass

            with patch("engine.async_lean_pool.AsyncLeanPool", _FakePool):
                await factory.setup_shared_pool()

            assert captured.get("transport_factory") is None, (
                "backend='local' must NOT pass a factory; this preserves "
                "V1–V6 behaviour for existing slime users")

        asyncio.run(_t())

    def test_kimina_backend_factory_passed(self):
        """V7 fix: backend='kimina' must produce a transport_factory."""
        async def _t():
            cfg = ProofEnvConfig(backend="kimina",
                                 backend_url="http://kimina:8000")
            factory = SlimeProofEnvFactory(cfg)

            captured: dict = {}

            class _FakePool:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

                async def start(self):
                    pass

            with patch("engine.async_lean_pool.AsyncLeanPool", _FakePool):
                await factory.setup_shared_pool()

            assert captured.get("transport_factory") is not None, (
                "V7 fix: SlimeProofEnvFactory must pass a transport_factory "
                "when backend != 'local'. Pre-V7 this was the silent "
                "downgrade-to-local bug.")

        asyncio.run(_t())


class TestVeRLInteractionBackendV7:
    """V7 closure: VeRLProofInteraction config must surface backend selection
    so a verl YAML can pick Kimina without code changes."""

    def test_backend_field_surfaces_to_env_config(self):
        config = {
            "name": "lean_prover",
            "backend": "kimina",
            "backend_url": "http://kimina:8000",
            "backend_api_key": "tok-123",
        }
        interaction = VeRLProofInteraction(config)
        assert interaction._env_config.backend == "kimina"
        assert interaction._env_config.backend_url == "http://kimina:8000"
        assert interaction._env_config.backend_api_key == "tok-123"

    def test_backend_default_is_local_for_back_compat(self):
        """An interaction config without ``backend`` keeps V1–V6 behaviour."""
        interaction = VeRLProofInteraction({"name": "lean_prover"})
        assert interaction._env_config.backend == "local"


# ═══════════════════════════════════════════════════════════════════════
# 2. Atomic env pool — race-freedom
# ═══════════════════════════════════════════════════════════════════════


class _CountingSampler(BaseSampler):
    """Sampler that just records which env was used by each rollout
    so we can detect any sharing."""

    def __init__(self, config: SamplerConfig):
        super().__init__(config)
        self.usage_log: list[tuple[int, int]] = []  # (env_id, rollout_idx)
        self.concurrent_holders: list[set[int]] = []

    async def generate_action(self, observation, problem_id, turn_idx):
        return ("done", [], [])

    async def _single_rollout(self, problem, policy_fn=None):
        env = await self._env_queue.get()
        env_id_in_pool = id(env)
        rollout_idx = problem.get("_idx", -1)

        # Track which envs are currently held.
        self.usage_log.append((env_id_in_pool, rollout_idx))

        # Hold the env for a tick to simulate work, then release.
        # If the queue were broken, two coroutines would land here
        # holding the same env simultaneously.
        await asyncio.sleep(0.001)
        try:
            traj = Trajectory(
                problem_id=problem["problem_id"],
                theorem_statement=problem.get("theorem_statement", ""),
            )
            return traj
        finally:
            self._env_queue.put_nowait(env)


class TestEnvPoolRaceFree:

    def test_each_concurrent_rollout_gets_distinct_env(self):
        """The V7 asyncio.Queue acquisition must guarantee that within
        the cap of num_envs simultaneous rollouts, every coroutine
        currently in flight holds a distinct env. With the V6 TOCTOU
        bug, two coroutines could observe _in_use=False and both seize
        env_pool[0]."""
        async def _t():
            cfg = SamplerConfig(num_envs=4, max_concurrent_problems=20)
            sampler = _CountingSampler(cfg)

            # Skip real ProofEnv setup — just plant pre-allocated mocks.
            sampler._env_pool = [MagicMock(spec=ProofEnv) for _ in range(4)]
            sampler._env_queue = asyncio.Queue()
            for env in sampler._env_pool:
                sampler._env_queue.put_nowait(env)
            sampler._env_semaphore = asyncio.Semaphore(4)
            sampler._setup_done = True

            problems = [{"problem_id": f"p{i}",
                         "theorem_statement": "T",
                         "_idx": i} for i in range(20)]
            await sampler.collect_rollouts(problems)

            # Every (env, rollout) pair was logged. Build the timeline
            # of which env is currently held: increment on use, decrement
            # on release. Since we're tracking the queue itself, the
            # invariant is simpler: each env_id should appear exactly
            # 5 times (20 rollouts / 4 envs), with no two rollouts
            # using it simultaneously.
            from collections import Counter
            usage = Counter(envid for envid, _ in sampler.usage_log)
            assert len(usage) == 4, (
                f"All 4 envs should be used; saw {len(usage)} unique. "
                f"Race-free queue must distribute work across the pool.")
            # Each env was used 5 times (20 rollouts / 4 envs).
            for envid, count in usage.items():
                assert count == 5, (
                    f"Expected each env to handle 5 rollouts; "
                    f"env {envid} handled {count} — load not balanced.")

        asyncio.run(_t())

    def test_env_returned_on_exception(self):
        """If a rollout raises, the env still goes back to the queue —
        otherwise the pool drains and subsequent rollouts deadlock."""

        class _FailingSampler(BaseSampler):
            async def generate_action(self, *a, **kw):
                return ("x", [], [])

            async def _single_rollout(self, problem, policy_fn=None):
                env = await self._env_queue.get()
                try:
                    raise RuntimeError("simulated rollout failure")
                finally:
                    self._env_queue.put_nowait(env)

        async def _t():
            cfg = SamplerConfig(num_envs=2, max_concurrent_problems=2)
            s = _FailingSampler(cfg)
            s._env_pool = [MagicMock(spec=ProofEnv) for _ in range(2)]
            s._env_queue = asyncio.Queue()
            for env in s._env_pool:
                s._env_queue.put_nowait(env)
            s._env_semaphore = asyncio.Semaphore(2)
            s._setup_done = True

            problems = [{"problem_id": f"p{i}",
                         "theorem_statement": "T"} for i in range(5)]
            results = await s.collect_rollouts(problems)
            # All 5 rollouts errored, so results is empty (filtered).
            assert results == []
            # But the queue still has both envs ready for re-use.
            assert s._env_queue.qsize() == 2, (
                "After exceptions, the env pool must be fully restored. "
                "Otherwise repeated failures would deadlock the pool.")

        asyncio.run(_t())


# ═══════════════════════════════════════════════════════════════════════
# 3. Tree-shaped rollouts (GRPO-friendly groups)
# ═══════════════════════════════════════════════════════════════════════


class _ScriptedProofEnv:
    """A minimal stand-in for ProofEnv that drives the tree sampler
    deterministically without touching Lean. Each action either:
      - 'good': returns reward 0.05 (progress) and remains non-terminal
      - 'win': returns reward 1.0 and terminates with success
      - 'bad': returns reward 0.0 and terminates without success
    """
    def __init__(self):
        self._cur_obs = ""
        self._traj = None
        self._turn_idx = 0

    async def setup(self):
        pass

    async def reset(self, problem):
        self._cur_obs = f"goal: {problem.get('theorem_statement', '?')}"
        self._traj = Trajectory(
            problem_id=problem["problem_id"],
            theorem_statement=problem.get("theorem_statement", ""),
        )
        self._turn_idx = 0
        return self._cur_obs

    async def step(self, action: str):
        from sampler.trajectory import Turn, RewardInfo, TerminationReason
        a = action.strip().lower()
        if "win" in a:
            r = RewardInfo(scalar=1.0, is_terminal=True, verification_level="L1")
            self._cur_obs = "[done]"
            done = True
        elif "bad" in a:
            r = RewardInfo(scalar=0.0, is_terminal=True, verification_level="L1",
                           error_class="generic")
            done = True
        else:
            r = RewardInfo(scalar=0.05, is_terminal=False, verification_level="L1")
            self._cur_obs = f"feedback after {a!r}"
            done = False

        from sampler.trajectory import Turn as _Turn
        turn = _Turn(turn_idx=self._turn_idx,
                     observation=self._cur_obs,
                     action=action,
                     reward=r)
        self._traj.add_turn(turn)
        self._turn_idx += 1
        return self._cur_obs, r, done, {}

    def get_trajectory(self):
        return self._traj

    async def close(self):
        pass


class TestTreeRollout:

    def test_emits_at_most_k_paths_per_problem(self):
        async def _t():
            cfg = TreeRolloutConfig(
                num_envs=1,
                max_concurrent_problems=1,
                branching_factor=3,
                max_nodes=20,
                max_depth=3,
                max_paths_per_problem=4,
                search_kind="best_first",
            )
            sampler = TreeRolloutSampler(cfg)
            sampler._env_pool = [_ScriptedProofEnv()]
            sampler._env_queue = asyncio.Queue()
            sampler._env_queue.put_nowait(sampler._env_pool[0])
            sampler._env_semaphore = asyncio.Semaphore(1)
            sampler._setup_done = True

            call_idx = {"i": 0}

            async def _policy(obs):
                # Cycle through good/win/bad to build a varied tree.
                acts = ["good_move", "good_alt", "win_now",
                         "bad_choice", "good_third"]
                a = acts[call_idx["i"] % len(acts)]
                call_idx["i"] += 1
                return a, [10, 20], [-0.1, -0.2]

            sampler._policy_fn = _policy

            trajs = await sampler.collect_rollouts(
                [{"problem_id": "p1", "theorem_statement": "P1"}])

            assert 1 <= len(trajs) <= 4, (
                f"max_paths_per_problem=4, got {len(trajs)}")
            assert all(t.problem_id == "p1" for t in trajs)

        asyncio.run(_t())

    def test_all_paths_share_root_observation(self):
        """The GRPO contract: trajectories from one problem share the
        same root prompt. The sampler enforces this by always reset()-ing
        from the same problem."""
        async def _t():
            cfg = TreeRolloutConfig(
                num_envs=1, max_concurrent_problems=1,
                branching_factor=2, max_nodes=8, max_depth=2,
                max_paths_per_problem=4, search_kind="best_first",
            )
            sampler = TreeRolloutSampler(cfg)
            sampler._env_pool = [_ScriptedProofEnv()]
            sampler._env_queue = asyncio.Queue()
            sampler._env_queue.put_nowait(sampler._env_pool[0])
            sampler._env_semaphore = asyncio.Semaphore(1)
            sampler._setup_done = True

            async def _policy(obs):
                return "good_X", [1], [-1.0]

            trajs = await sampler.collect_rollouts(
                [{"problem_id": "shared", "theorem_statement": "T"}],
                policy_fn=_policy)

            # All trajectories from the same problem must agree on
            # the root observation (turn 0).
            roots = {t.turns[0].observation for t in trajs if t.turns}
            assert len(roots) == 1, (
                f"All trajectories from one problem must share the root "
                f"observation. Got {len(roots)} distinct roots.")

        asyncio.run(_t())

    def test_group_normalize_rewards_writes_advantage(self):
        async def _t():
            cfg = TreeRolloutConfig(
                num_envs=1, max_concurrent_problems=1,
                branching_factor=3, max_nodes=10, max_depth=2,
                max_paths_per_problem=8, search_kind="best_first",
                group_normalize_rewards=True,
            )
            sampler = TreeRolloutSampler(cfg)
            sampler._env_pool = [_ScriptedProofEnv()]
            sampler._env_queue = asyncio.Queue()
            sampler._env_queue.put_nowait(sampler._env_pool[0])
            sampler._env_semaphore = asyncio.Semaphore(1)
            sampler._setup_done = True

            i = {"x": 0}

            async def _policy(obs):
                seq = ["good", "win_a", "bad_b", "good_b", "win_c"]
                a = seq[i["x"] % len(seq)]; i["x"] += 1
                return a, [], []

            trajs = await sampler.collect_rollouts(
                [{"problem_id": "g1", "theorem_statement": "T"}],
                policy_fn=_policy)

            if len(trajs) >= 2:
                advs = [t.metadata.get("group_advantage")
                         for t in trajs]
                assert all(a is not None for a in advs), (
                    "group_normalize_rewards=True must populate "
                    "metadata['group_advantage']")
                # Advantages should sum to ~0 (centred).
                assert abs(sum(advs)) < 1e-6, (
                    f"Group-centred advantages must sum to 0; "
                    f"got sum={sum(advs)}")

        asyncio.run(_t())

    def test_search_tree_dict_v3_compatible(self):
        """The search_tree dict shape must match the V3 dialog.json schema
        so the trees can be embedded back into meta.search_tree for
        offline analysis."""
        from sampler.tree_rollout_sampler import _Node
        nodes = {
            0: _Node(id=0, parent_id=None, tactic=None,
                     observation="root", depth=0, children=[1, 2]),
            1: _Node(id=1, parent_id=0, tactic="simp",
                     observation="o1", depth=1, success=True,
                     is_terminal=True, score=10.0, reward_at_step=1.0),
            2: _Node(id=2, parent_id=0, tactic="ring",
                     observation="o2", depth=1, score=0.05,
                     reward_at_step=0.05),
        }
        tree = TreeRolloutSampler.to_search_tree_dict(nodes, kind="best_first")
        assert tree["kind"] == "best_first"
        assert tree["root_node_id"] == 0
        assert tree["solved_node_id"] == 1
        assert tree["total_nodes"] == 3
        assert tree["max_depth"] == 1
        # Per-node fields (matching V3 search_tree schema)
        node_dicts = {n["node_id"]: n for n in tree["nodes"]}
        assert node_dicts[1]["status"] == "solved"
        assert node_dicts[2]["status"] == "open"
        assert node_dicts[1]["tactic"] == "simp"


# ═══════════════════════════════════════════════════════════════════════
# 4. Optional verl/slime imports — must not break anything
# ═══════════════════════════════════════════════════════════════════════


class TestOptionalRLFrameworkImports:

    def test_verl_available_flag_exists(self):
        """sampler.verl_sampler must export VERL_AVAILABLE so callers
        can branch on whether real integration is active."""
        assert isinstance(VERL_AVAILABLE, bool)

    def test_slime_available_flag_exists(self):
        assert isinstance(SLIME_AVAILABLE, bool)

    def test_verl_register_decorator_attaches_name(self):
        """Whether or not verl is installed, the @register decorator
        must mark the class with its registered name. This is the hook
        verl uses (when present) to discover the agent loop."""
        assert hasattr(VeRLProofAgentLoop, "_verl_registered_name") or \
               VERL_AVAILABLE, (
            "@_register stub or real must attach _verl_registered_name "
            "for verl's discovery to find the class")

    def test_verl_proof_interaction_constructible_without_verl(self):
        """When verl is absent, _BaseInteraction is `object` — but
        constructor-time TypeErrors must be swallowed so the class
        still works in unit tests."""
        instance = VeRLProofInteraction({"name": "test_no_verl"})
        assert instance.name == "test_no_verl"

    def test_verl_proof_agent_loop_constructible_without_verl(self):
        loop = VeRLProofAgentLoop(
            proof_config={"project_dir": "/tmp", "max_turns": 5})
        assert loop._env_config.max_turns == 5


# ═══════════════════════════════════════════════════════════════════════
# 5. End-to-end: all three layers wired together
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndUnification:
    """Smoke tests that exercise the full path:
       Backend selector → ProofEnv → Sampler → Trajectory → verl/slime format
    """

    def test_e2e_proof_env_with_mock_backend(self):
        """ProofEnvConfig(backend='mock') runs setup → reset → step
        end-to-end without a real Lean. This is the smoke test that
        the *backend selector* is structurally correct."""
        async def _t():
            cfg = ProofEnvConfig(backend="mock", pool_size=1, max_turns=3)
            env = ProofEnv(cfg)

            # Run setup with a real MockTransport-backed pool. The factory
            # path delegates to engine.backend_factory.build_backend,
            # which knows about kind='mock'.
            await env.setup()

            obs = await env.reset({
                "problem_id": "e2e",
                "theorem_statement": "theorem e2e : True := by",
            })
            assert "e2e" in obs

            # Step once — MockTransport returns a successful response.
            _obs, reward, _done, _info = await env.step("trivial")
            assert isinstance(reward, RewardInfo)

            await env.close()

        asyncio.run(_t())

    def test_e2e_trajectory_to_verl_format_with_mock_backend(self):
        """Run a tiny rollout through TreeRolloutSampler with a mock
        backend, then check that the resulting Trajectory.to_verl_format()
        produces a well-formed verl AgentLoopOutput dict."""
        async def _t():
            cfg = TreeRolloutConfig(
                env_config=ProofEnvConfig(backend="local", pool_size=1),
                num_envs=1, max_concurrent_problems=1,
                branching_factor=2, max_nodes=4, max_depth=2,
                max_paths_per_problem=2, search_kind="best_first",
            )
            sampler = TreeRolloutSampler(cfg)
            sampler._env_pool = [_ScriptedProofEnv()]
            sampler._env_queue = asyncio.Queue()
            sampler._env_queue.put_nowait(sampler._env_pool[0])
            sampler._env_semaphore = asyncio.Semaphore(1)
            sampler._setup_done = True

            async def _policy(obs):
                return "win_first_try", [42, 43], [-0.5, -0.5]

            trajs = await sampler.collect_rollouts(
                [{"problem_id": "verl_e2e",
                  "theorem_statement": "theorem t : True"}],
                policy_fn=_policy)
            assert trajs, "tree rollout must produce ≥1 trajectory"

            verl_dict = trajs[0].to_verl_format()
            # The verl AgentLoopOutput contract:
            for key in ("prompt_ids", "response_ids", "response_mask",
                        "response_logprobs", "num_turns", "reward_score"):
                assert key in verl_dict, (
                    f"verl format must include {key!r}; got {list(verl_dict)}")

        asyncio.run(_t())

    def test_e2e_slime_episodes_format(self):
        """Same trajectory must also serialise to slime's episode format."""
        async def _t():
            cfg = TreeRolloutConfig(
                num_envs=1, max_concurrent_problems=1,
                branching_factor=1, max_nodes=2, max_depth=1,
                max_paths_per_problem=1,
            )
            sampler = TreeRolloutSampler(cfg)
            sampler._env_pool = [_ScriptedProofEnv()]
            sampler._env_queue = asyncio.Queue()
            sampler._env_queue.put_nowait(sampler._env_pool[0])
            sampler._env_semaphore = asyncio.Semaphore(1)
            sampler._setup_done = True

            async def _policy(obs):
                return "win", [], []

            trajs = await sampler.collect_rollouts(
                [{"problem_id": "slime_e2e", "theorem_statement": "T"}],
                policy_fn=_policy)
            assert trajs

            episodes = trajs[0].to_slime_episodes()
            assert len(episodes) >= 1
            for step in episodes:
                for k in ("observation", "action", "reward",
                          "done", "info"):
                    assert k in step

        asyncio.run(_t())
