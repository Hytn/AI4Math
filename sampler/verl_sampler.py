"""sampler/verl_sampler.py — veRL framework integration

Two integration modes for veRL:

1. **VeRLProofInteraction** (BaseInteraction)
   - Plugs into veRL's existing ToolAgentLoop as a custom Interaction
   - The LLM server is managed by veRL; this class provides the environment
   - Minimal integration: just set interaction_config_path in veRL config
   - Best for: using veRL's existing multi-turn infrastructure

2. **VeRLProofAgentLoop** (AgentLoopBase)
   - Custom AgentLoop that directly manages the prove loop
   - Gives full control over observation formatting, reward shaping,
     turn management, and early termination
   - Best for: maximum performance and customization

Configuration example (veRL yaml)::

    # Mode 1: Interaction
    rollout:
      multi_turn:
        enable: true
        interaction_config_path: configs/proof_interaction.yaml
        max_assistant_turns: 32

    # Mode 2: Custom AgentLoop
    rollout:
      agent:
        default_agent_loop: proof_agent
        agent_loop_config_path: configs/proof_agent_loop.yaml
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

from sampler.proof_env import ProofEnv, ProofEnvConfig
from sampler.trajectory import Trajectory, RewardInfo, TerminationReason

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Mode 1: BaseInteraction — minimal integration with veRL ToolAgentLoop
# ═══════════════════════════════════════════════════════════════════════

class VeRLProofInteraction:
    """veRL BaseInteraction implementation for formal theorem proving.

    This wraps ProofEnv as a veRL Interaction, allowing veRL's existing
    ToolAgentLoop to drive multi-turn proof generation.

    The LLM (RL policy) is managed by veRL. This class only handles:
      - Receiving the model's tactic output
      - Running Lean verification
      - Returning (should_terminate, feedback, score, info)

    Registration in interaction config yaml::

        interactions:
          - name: lean_prover
            class_path: sampler.verl_sampler.VeRLProofInteraction
            config:
              project_dir: /path/to/mathlib
              pool_size: 4
              max_turns: 32
              reward_success: 1.0
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name = config.get("name", "lean_prover")

        # Build ProofEnvConfig from veRL interaction config
        env_cfg = ProofEnvConfig(
            project_dir=config.get("project_dir", "."),
            pool_size=config.get("pool_size", 4),
            max_turns=config.get("max_turns", 32),
            lean_timeout_s=config.get("lean_timeout_s", 30),
            preamble=config.get("preamble", "import Mathlib"),
            reward_success=config.get("reward_success", 1.0),
            reward_goal_closed=config.get("reward_goal_closed", 0.1),
            reward_l1_pass=config.get("reward_l1_pass", 0.05),
            reward_l0_reject=config.get("reward_l0_reject", -0.02),
            reward_sorry=config.get("reward_sorry", -0.5),
        )

        self._env_config = env_cfg
        # Per-instance environments (keyed by instance_id)
        self._envs: dict[str, ProofEnv] = {}
        self._setup_done = False
        self._shared_pool = None

    async def _ensure_setup(self):
        """Lazy setup of shared Lean pool."""
        if self._setup_done:
            return
        # Create a shared pool that environments reference
        from engine.async_lean_pool import AsyncLeanPool
        self._shared_pool = AsyncLeanPool(
            pool_size=self._env_config.pool_size,
            project_dir=self._env_config.project_dir,
            timeout_seconds=self._env_config.lean_timeout_s,
        )
        await self._shared_pool.start(preamble=self._env_config.preamble)
        self._setup_done = True

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        """Called by veRL ToolAgentLoop at the start of each episode.

        kwargs should contain:
          - problem_id: str
          - theorem_statement: str
          - header: str (optional)
        """
        await self._ensure_setup()

        if instance_id is None:
            from uuid import uuid4
            instance_id = uuid4().hex

        env = ProofEnv(self._env_config)
        # Share the pool instead of creating a new one
        env._pool = self._shared_pool
        env._verifier = None  # Will be set up on first use

        # Initialize verifier with shared pool
        from engine.async_verification_scheduler import AsyncVerificationScheduler
        from engine.prefilter import PreFilter
        from engine.error_intelligence import ErrorIntelligence
        from engine.broadcast import BroadcastBus

        env._prefilter = PreFilter()
        env._error_intel = ErrorIntelligence()
        env._broadcast = BroadcastBus()
        env._verifier = AsyncVerificationScheduler(
            prefilter=env._prefilter,
            lean_pool=self._shared_pool,
            error_intel=env._error_intel,
            broadcast=env._broadcast,
            project_dir=self._env_config.project_dir,
        )

        problem = {
            "problem_id": kwargs.get("problem_id", instance_id),
            "theorem_statement": kwargs.get("theorem_statement", ""),
            "header": kwargs.get("header", ""),
        }
        await env.reset(problem)
        self._envs[instance_id] = env

        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Called by veRL ToolAgentLoop after each LLM generation.

        Extracts the tactic from the last assistant message, runs Lean
        verification, and returns the result.

        Returns:
            (should_terminate, response_content, turn_score, additional_data)
        """
        env = self._envs.get(instance_id)
        if env is None:
            return True, "Error: unknown instance", 0.0, {}

        # Extract tactic from last assistant message
        tactic = self._extract_tactic(messages)
        if not tactic:
            return False, "Please provide a Lean 4 tactic.", 0.0, {}

        # Step the environment
        obs, reward, done, info = await env.step(tactic)

        return done, obs, reward.scalar, {
            "verification_level": reward.verification_level,
            "error_class": reward.error_class,
            "goals_remaining": reward.goals_remaining,
            "fix_hint": reward.fix_hint,
            "turn_idx": env._turn_idx,
        }

    async def calculate_score(self) -> float:
        return 0.0

    async def finalize_interaction(self) -> None:
        """Release per-instance state."""
        # Don't close the shared pool — just remove instance refs
        self._envs.clear()

    def _extract_tactic(self, messages: list[dict]) -> str:
        """Extract Lean tactic from the last assistant message."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Try to extract code blocks
                if "```lean" in content:
                    start = content.index("```lean") + len("```lean")
                    end = content.index("```", start)
                    return content[start:end].strip()
                if "```" in content:
                    start = content.index("```") + 3
                    end = content.index("```", start)
                    return content[start:end].strip()
                # Raw tactic
                return content.strip()
        return ""


# ═══════════════════════════════════════════════════════════════════════
# Mode 2: Custom AgentLoop — full control over the prove loop
# ═══════════════════════════════════════════════════════════════════════

class VeRLProofAgentLoop:
    """Custom veRL AgentLoop for formal theorem proving.

    Unlike VeRLProofInteraction (which integrates with ToolAgentLoop),
    this class IS the agent loop, giving full control over:
      - Observation/prompt construction
      - Turn management and early termination
      - Reward computation and token masking
      - Metrics collection

    To register with veRL, in your config set::

        rollout.agent.default_agent_loop: proof_agent

    And add to your entrypoint::

        from sampler.verl_sampler import VeRLProofAgentLoop
        # veRL will discover it via the register decorator

    Note: This class follows veRL's AgentLoopBase interface but does not
    inherit from it directly to avoid hard dependency on veRL imports.
    In a real deployment, add the veRL import and @register decorator.
    """

    def __init__(self, *args, **kwargs):
        # In real deployment, call super().__init__(*args, **kwargs)
        # to get server_manager, tokenizer, etc. from veRL
        self._server_manager = kwargs.get("server_manager")
        self._tokenizer = kwargs.get("tokenizer")
        self._rollout_config = kwargs.get("rollout_config", {})

        # Proof environment config
        proof_config = kwargs.get("proof_config", {})
        self._env_config = ProofEnvConfig(
            project_dir=proof_config.get("project_dir", "."),
            pool_size=proof_config.get("pool_size", 4),
            max_turns=proof_config.get("max_turns", 32),
        )

        self._env: Optional[ProofEnv] = None
        self._setup_done = False

    async def _ensure_env(self):
        if self._env is None:
            self._env = ProofEnv(self._env_config)
            await self._env.setup()

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> dict:
        """Main agent loop entry point.

        Called by veRL's AgentLoopWorker for each problem in the batch.

        Args:
            sampling_params: LLM sampling config (temperature, top_p, etc.)
            **kwargs: Must contain 'raw_prompt' (list of messages) and
                      'extra_info' with problem metadata.

        Returns:
            Dict compatible with veRL's AgentLoopOutput.
        """
        await self._ensure_env()

        messages = list(kwargs.get("raw_prompt", []))
        extra_info = kwargs.get("extra_info", {})
        problem = {
            "problem_id": extra_info.get("problem_id", "unknown"),
            "theorem_statement": extra_info.get("theorem_statement",
                                                 self._extract_theorem(messages)),
        }

        # Reset environment
        obs = await self._env.reset(problem)

        # Accumulators
        prompt_ids = []
        response_ids = []
        response_mask = []
        response_logprobs = []
        turn_scores = []
        metrics = {"generate_sequences": 0, "tool_calls": 0, "num_preempted": 0}

        # Tokenize initial prompt
        if self._tokenizer:
            prompt_ids = self._tokenizer.encode(
                self._build_system_prompt() + "\n" + obs)

        done = False
        turn = 0

        while not done:
            t_gen = time.time()

            # Call the LLM via veRL's server manager
            response_text, gen_ids, gen_logprobs = await self._call_llm(
                obs, sampling_params)

            gen_time = time.time() - t_gen
            metrics["generate_sequences"] += gen_time

            # Step the environment
            t_tool = time.time()
            obs, reward, done, info = await self._env.step(response_text)
            tool_time = time.time() - t_tool
            metrics["tool_calls"] += tool_time

            # Record token data
            response_ids.extend(gen_ids)
            response_mask.extend([1] * len(gen_ids))
            response_logprobs.extend(gen_logprobs)
            turn_scores.append(reward.scalar)

            if not done and self._tokenizer:
                # Tokenize feedback (non-trainable)
                fb_ids = self._tokenizer.encode(obs)
                response_ids.extend(fb_ids)
                response_mask.extend([0] * len(fb_ids))
                response_logprobs.extend([0.0] * len(fb_ids))

            turn += 1

        traj = self._env.get_trajectory()

        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_mask": response_mask,
            "response_logprobs": response_logprobs,
            "num_turns": turn,
            "reward_score": traj.total_reward if traj else 0.0,
            "metrics": metrics,
            "extra_fields": {
                "turn_scores": turn_scores,
                "problem_id": problem["problem_id"],
                "success": traj.success if traj else False,
                "termination": traj.termination.value if traj else "unknown",
            },
        }

    async def _call_llm(
        self, observation: str, sampling_params: dict
    ) -> tuple[str, list[int], list[float]]:
        """Call the LLM via veRL's server manager.

        In real deployment, this uses self._server_manager to send
        chat completions to the veRL-managed vLLM/SGLang server.
        """
        if self._server_manager is None:
            # Fallback for testing
            return "sorry", [], []

        # Build messages for the OpenAI-compatible API
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": observation},
        ]

        response = await self._server_manager.chat_completion(
            messages=messages,
            temperature=sampling_params.get("temperature", 0.9),
            max_tokens=sampling_params.get("max_tokens", 2048),
            logprobs=True,
        )

        text = response.choices[0].message.content
        token_ids = []
        log_probs = []

        if self._tokenizer:
            token_ids = self._tokenizer.encode(text)
        if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
            log_probs = [
                lp.logprob for lp in response.choices[0].logprobs.content
            ]

        return text, token_ids, log_probs

    def _build_system_prompt(self) -> str:
        return (
            "You are a formal mathematics expert. Generate Lean 4 tactics "
            "to prove the given theorem. Respond with only the tactic code, "
            "no explanations. Do NOT use sorry."
        )

    def _extract_theorem(self, messages: list[dict]) -> str:
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""


# ═══════════════════════════════════════════════════════════════════════
# Reward function for veRL's reward_function config
# ═══════════════════════════════════════════════════════════════════════

def compute_proof_reward(data_source: str, solution_str: str,
                         ground_truth: dict, extra_info: dict = None) -> float:
    """Standalone reward function compatible with veRL's custom_reward_function.

    Usage in veRL config::

        reward:
          custom_reward_function:
            path: sampler/verl_sampler.py
            name: compute_proof_reward

    Args:
        data_source: Dataset name
        solution_str: Model's generated proof
        ground_truth: Dict with 'theorem_statement' and optional 'expected_proof'
        extra_info: Additional info from the trajectory

    Returns:
        Reward score [0, 1]
    """
    extra = extra_info or {}

    # If trajectory already has a computed reward, use it
    if "reward_score" in extra:
        return extra["reward_score"]

    # Basic heuristic when no Lean verification is available
    if not solution_str.strip():
        return 0.0
    if "sorry" in solution_str.lower():
        return -0.5

    # Check if extra_info contains verification result
    if extra.get("success"):
        return 1.0

    return 0.0
