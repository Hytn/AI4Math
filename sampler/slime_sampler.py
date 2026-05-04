"""sampler/slime_sampler.py — slime framework integration

slime (https://github.com/THUDM/slime) uses a multi-turn environment
protocol where environments expose reset() / step() and the trainer
collects episodes as lists of (obs, action, reward, done) tuples.

This module wraps ProofEnv as a slime-compatible environment and
provides a sampler that works with slime's training loop.

Usage::

    from sampler.slime_sampler import SlimeSampler, SlimeProofEnvFactory

    # As a slime environment factory
    env_factory = SlimeProofEnvFactory(config)

    # Or as a standalone sampler
    sampler = SlimeSampler(config)
    trajectories = await sampler.collect_rollouts(problems, policy_fn)
    episodes = [t.to_slime_episodes() for t in trajectories]
"""
from __future__ import annotations

import logging
from typing import Any

from sampler.proof_env import ProofEnv, ProofEnvConfig
from sampler.base_sampler import BaseSampler, SamplerConfig, PolicyFn
from sampler.trajectory import Trajectory

logger = logging.getLogger(__name__)

# v7: detect slime availability. SLIME (https://github.com/THUDM/slime)
# typically exposes a CustomGenerator / Env protocol. We probe a few
# known import paths and expose SLIME_AVAILABLE so callers can branch
# on it. When slime is missing, the wrappers in this module remain
# fully functional via the BaseSampler trajectory protocol — they just
# don't auto-register as slime extensions.
SLIME_AVAILABLE = False
try:
    import slime  # type: ignore  # noqa: F401
    SLIME_AVAILABLE = True
    logger.info("sampler.slime_sampler: slime detected, real integration active")
except ImportError:
    logger.debug(
        "sampler.slime_sampler: slime not installed, running in stub mode "
        "(install with `pip install -r requirements-rl.txt`)")


class SlimeSampler(BaseSampler):
    """Sampler adapted for slime's multi-turn RL protocol.

    Extends BaseSampler with slime-specific output formatting
    and episode collection.
    """

    def __init__(self, config: SamplerConfig = None,
                 policy_fn: PolicyFn = None):
        super().__init__(config)
        self._policy_fn = policy_fn

    async def generate_action(
        self, observation: str, problem_id: str, turn_idx: int,
    ) -> tuple[str, list[int], list[float]]:
        """Use the provided policy function."""
        if self._policy_fn:
            return await self._policy_fn(observation)
        raise RuntimeError("No policy_fn provided to SlimeSampler")

    async def collect_episodes(
        self, problems: list[dict[str, Any]],
        policy_fn: PolicyFn = None,
    ) -> list[list[dict[str, Any]]]:
        """Collect episodes in slime's native format.

        Returns:
            List of episodes, where each episode is a list of step dicts:
            [{"observation": str, "action": str, "reward": float,
              "done": bool, "info": dict}, ...]
        """
        if policy_fn:
            self._policy_fn = policy_fn
        trajectories = await self.collect_rollouts(problems, policy_fn)
        return [t.to_slime_episodes() for t in trajectories]


class SlimeProofEnvFactory:
    """Factory for creating slime-compatible proof environments.

    slime expects environment factories that produce env instances
    with reset() and step() methods. This factory wraps ProofEnv
    to match that protocol.

    Usage::

        factory = SlimeProofEnvFactory(env_config)

        # In slime's env creation:
        env = await factory.create()
        obs = await env.reset(problem)
        obs, reward, done, info = await env.step(action)
    """

    def __init__(self, config: ProofEnvConfig = None):
        self.config = config or ProofEnvConfig()
        self._shared_pool = None

    async def setup_shared_pool(self):
        """Initialize a shared Lean pool for all environments.

        v7: honour ``ProofEnvConfig.backend``. Previously this method
        ignored the backend selector and always built a LocalTransport
        pool — meaning slime users couldn't reach Kimina/Pantograph/
        LooKeng even after they set ``backend="kimina"`` on the config.
        That bug let the slime side silently degrade to local for any
        non-trivial RL throughput run.

        We reuse ``ProofEnv._make_transport_factory()`` so backend
        semantics live in exactly one place — anything ProofEnv learns
        to do (new backend, fallback rules, error handling) the slime
        factory inherits for free.
        """
        if self._shared_pool is not None:
            return

        from engine.async_lean_pool import AsyncLeanPool

        transport_factory = None
        if self.config.backend not in ("", "local", None):
            tmp_env = ProofEnv(self.config)
            transport_factory = tmp_env._make_transport_factory()

        self._shared_pool = AsyncLeanPool(
            pool_size=self.config.pool_size,
            project_dir=self.config.project_dir,
            preamble=self.config.preamble,
            timeout_seconds=self.config.lean_timeout_s,
            transport_factory=transport_factory,
        )
        await self._shared_pool.start()

    async def create(self) -> SlimeProofEnv:
        """Create a slime-compatible proof environment instance."""
        await self.setup_shared_pool()
        env = SlimeProofEnv(self.config, self._shared_pool)
        await env._setup_verifier()
        return env

    async def close(self):
        if self._shared_pool:
            await self._shared_pool.shutdown()
            self._shared_pool = None


class SlimeProofEnv:
    """slime-compatible proof environment.

    Wraps ProofEnv with the exact interface slime expects:
      - reset(problem) -> observation
      - step(action) -> (observation, reward, done, info)
      - close()
    """

    def __init__(self, config: ProofEnvConfig, shared_pool=None):
        self._inner = ProofEnv(config)
        self._shared_pool = shared_pool

    async def _setup_verifier(self):
        """Set up verification using the shared pool."""
        if self._shared_pool is None:
            await self._inner.setup()
            return

        from engine.async_verification_scheduler import AsyncVerificationScheduler
        from engine.prefilter import PreFilter
        from engine.error_intelligence import ErrorIntelligence
        from engine.broadcast import BroadcastBus

        self._inner._pool = self._shared_pool
        self._inner._prefilter = PreFilter()
        self._inner._error_intel = ErrorIntelligence()
        self._inner._broadcast = BroadcastBus()
        self._inner._verifier = AsyncVerificationScheduler(
            prefilter=self._inner._prefilter,
            lean_pool=self._shared_pool,
            error_intel=self._inner._error_intel,
            broadcast=self._inner._broadcast,
            project_dir=self._inner.config.project_dir,
        )

    async def reset(self, problem: dict[str, Any]) -> str:
        return await self._inner.reset(problem)

    async def step(self, action: str) -> tuple[str, float, bool, dict]:
        obs, reward_info, done, info = await self._inner.step(action)
        return obs, reward_info.scalar, done, {
            **info,
            "reward_info": reward_info,
        }

    def get_trajectory(self) -> Trajectory:
        return self._inner.get_trajectory()

    async def close(self):
        # Don't close shared pool
        pass
