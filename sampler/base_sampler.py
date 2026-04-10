"""sampler/base_sampler.py — Framework-agnostic sampler base class

Provides the abstract interface that all RL framework adapters implement.
The sampler orchestrates batch proof episodes: for each problem in a batch,
it runs a multi-turn loop where the RL policy generates tactics and
ProofEnv verifies them.

The key abstraction: the sampler does NOT own the LLM. The RL framework
provides the generation function (policy). The sampler provides the
environment (ProofEnv) and trajectory collection.
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

from sampler.proof_env import ProofEnv, ProofEnvConfig
from sampler.trajectory import Trajectory, Turn, RewardInfo, TerminationReason

logger = logging.getLogger(__name__)


# Type alias for the policy function provided by the RL framework.
# Given an observation string, returns (action_text, token_ids, log_probs).
PolicyFn = Callable[
    [str],  # observation
    Awaitable[tuple[str, list[int], list[float]]]
]


@dataclass
class SamplerConfig:
    """Configuration for the sampler."""
    env_config: ProofEnvConfig = field(default_factory=ProofEnvConfig)

    # Parallelism
    num_envs: int = 8              # Number of concurrent environments
    max_concurrent_problems: int = 64  # Max problems in flight

    # Sampling
    num_samples_per_problem: int = 1  # Number of independent attempts per problem
    temperature: float = 0.9

    # Token-level
    tokenizer_name: str = ""        # For token id/logprob capture
    max_prompt_len: int = 2048
    max_response_len: int = 4096


class BaseSampler(ABC):
    """Framework-agnostic sampler for multi-turn proof rollouts.

    Subclasses implement `generate_action()` to connect their RL framework's
    policy to the proof environment. The base class handles:
      - Environment pool management
      - Batch rollout orchestration
      - Trajectory collection and formatting

    Lifecycle::

        sampler = MySampler(config)
        await sampler.setup()

        trajectories = await sampler.collect_rollouts(problems, policy_fn)

        await sampler.teardown()
    """

    def __init__(self, config: SamplerConfig = None):
        self.config = config or SamplerConfig()
        self._env_pool: list[ProofEnv] = []
        self._env_semaphore: Optional[asyncio.Semaphore] = None
        self._setup_done = False

    async def setup(self):
        """Initialize the environment pool."""
        if self._setup_done:
            return
        self._env_pool = [
            ProofEnv(self.config.env_config)
            for _ in range(self.config.num_envs)
        ]
        await asyncio.gather(*(env.setup() for env in self._env_pool))
        self._env_semaphore = asyncio.Semaphore(self.config.num_envs)
        self._setup_done = True
        logger.info("BaseSampler: %d environments ready", self.config.num_envs)

    async def teardown(self):
        """Release all environments."""
        await asyncio.gather(*(env.close() for env in self._env_pool))
        self._env_pool.clear()
        self._setup_done = False

    @abstractmethod
    async def generate_action(
        self, observation: str, problem_id: str, turn_idx: int,
    ) -> tuple[str, list[int], list[float]]:
        """Generate an action from the RL policy.

        Subclasses implement this to call their framework's policy/model.

        Args:
            observation: Text observation from the environment.
            problem_id: Current problem identifier.
            turn_idx: Current turn index within the episode.

        Returns:
            (action_text, token_ids, log_probs)
        """
        ...

    async def collect_rollouts(
        self, problems: list[dict[str, Any]],
        policy_fn: PolicyFn = None,
    ) -> list[Trajectory]:
        """Run multi-turn proof rollouts for a batch of problems.

        Args:
            problems: List of problem dicts (problem_id, theorem_statement, ...).
            policy_fn: Optional override for generate_action.

        Returns:
            List of completed Trajectory objects.
        """
        if not self._setup_done:
            await self.setup()

        # Expand by num_samples_per_problem
        expanded = []
        for p in problems:
            for sample_idx in range(self.config.num_samples_per_problem):
                expanded.append({
                    **p,
                    "_sample_idx": sample_idx,
                })

        # Concurrent rollout with semaphore
        sem = asyncio.Semaphore(self.config.max_concurrent_problems)

        async def _run_one(problem: dict) -> Trajectory:
            async with sem:
                return await self._single_rollout(problem, policy_fn)

        trajectories = await asyncio.gather(
            *[_run_one(p) for p in expanded],
            return_exceptions=True,
        )

        # Filter out exceptions
        results = []
        for i, t in enumerate(trajectories):
            if isinstance(t, Exception):
                logger.error("Rollout %d failed: %s", i, t)
            else:
                results.append(t)

        return results

    async def _single_rollout(
        self, problem: dict, policy_fn: PolicyFn = None,
    ) -> Trajectory:
        """Run a single multi-turn episode."""
        # Acquire an environment from the pool
        async with self._env_semaphore:
            env = self._env_pool[0]  # Simple round-robin; improve if needed
            # In practice, use a proper pool. This is simplified.
            for e in self._env_pool:
                if not getattr(e, "_in_use", False):
                    env = e
                    break
            env._in_use = True

        try:
            obs = await env.reset(problem)
            done = False

            while not done:
                t0 = time.time()

                # Generate action via RL policy
                if policy_fn:
                    action, token_ids, log_probs = await policy_fn(obs)
                else:
                    action, token_ids, log_probs = await self.generate_action(
                        obs, problem["problem_id"], env._turn_idx)

                gen_ms = int((time.time() - t0) * 1000)

                # Step environment
                obs, reward, done, info = await env.step(action)

                # Enrich the last turn with token-level data
                traj = env.get_trajectory()
                if traj and traj.turns:
                    last = traj.turns[-1]
                    last.action_token_ids = token_ids
                    last.action_log_probs = log_probs
                    last.action_mask = [1] * len(token_ids)
                    last.generation_ms = gen_ms

            return env.get_trajectory()

        finally:
            env._in_use = False

    # ── Convenience: synchronous entry point ──────────────────────────

    def collect_rollouts_sync(
        self, problems: list[dict], policy_fn: PolicyFn = None,
    ) -> list[Trajectory]:
        """Synchronous wrapper for collect_rollouts."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.collect_rollouts(problems, policy_fn))
        finally:
            loop.close()

    # ── Batch statistics ──────────────────────────────────────────────

    @staticmethod
    def batch_stats(trajectories: list[Trajectory]) -> dict[str, Any]:
        """Compute aggregate statistics over a batch of trajectories."""
        if not trajectories:
            return {}
        successes = sum(1 for t in trajectories if t.success)
        total = len(trajectories)
        avg_turns = sum(t.num_turns for t in trajectories) / total
        avg_reward = sum(t.total_reward for t in trajectories) / total
        avg_time = sum(t.wall_time_s for t in trajectories) / total
        return {
            "total": total,
            "success_rate": round(successes / total, 4),
            "avg_turns": round(avg_turns, 2),
            "avg_reward": round(avg_reward, 4),
            "avg_wall_time_s": round(avg_time, 2),
            "termination_dist": _count_terminations(trajectories),
        }


def _count_terminations(trajs: list[Trajectory]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in trajs:
        k = t.termination.value
        counts[k] = counts.get(k, 0) + 1
    return counts
