"""sampler — RL Framework Sampler Abstraction

Bridges the AI4Math formal theorem proving agent with reinforcement learning
frameworks (veRL, slime, etc.) by exposing the multi-turn prove loop as a
standard RL sampling environment.

Architecture:

    RL Trainer (veRL / slime / ...)
         │
         ▼
    ┌─────────────────────┐
    │   BaseSampler       │  ← framework-agnostic interface
    │   (abstract)        │
    └────────┬────────────┘
             │
    ┌────────┴─────────────────────────┐
    │                                  │
    ▼                                  ▼
  VeRLSampler                    SlimeSampler
  (veRL AgentLoop /              (slime env
   Interaction)                   protocol)
             │
             ▼
    ┌─────────────────────┐
    │  ProofEnv            │  ← wraps AI4Math engine
    │  (multi-turn env)    │
    └─────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
  AsyncLeanPool    AsyncLLMProvider
  (verification)   (generation — during RL,
                    replaced by RL policy)

Core concepts:
  - ProofEnv:      Gymnasium-style async environment wrapping Lean verification
  - BaseSampler:   Framework-agnostic sampler producing rollout trajectories
  - VeRLSampler:   veRL-native integration via BaseInteraction + AgentLoop
  - SlimeSampler:  slime-compatible wrapper
  - Trajectory:    Structured rollout data (observations, actions, rewards, masks)
"""

from sampler.trajectory import Trajectory, Turn, RewardInfo
from sampler.proof_env import ProofEnv, ProofEnvConfig
from sampler.base_sampler import BaseSampler, SamplerConfig
from sampler.verl_sampler import (
    VeRLProofInteraction, VeRLProofAgentLoop, VERL_AVAILABLE,
)
from sampler.slime_sampler import (
    SlimeSampler, SLIME_AVAILABLE,
    SlimeProofEnvFactory, SlimeProofEnv,
)
from sampler.tree_rollout_sampler import (
    TreeRolloutSampler, TreeRolloutConfig,
)
from sampler.policy_adapter import (
    MockPolicy, OpenAIPolicy, CallablePolicy, build_policy,
    DEFAULT_SYSTEM_PROMPT,
)
from sampler.batch_export import (
    to_grpo_batch, to_sft_jsonl, to_ppo_batch, save_batch_jsonl,
)

__all__ = [
    # Core types
    "Trajectory", "Turn", "RewardInfo",
    "ProofEnv", "ProofEnvConfig",
    "BaseSampler", "SamplerConfig",
    # RL framework adapters
    "VeRLProofInteraction", "VeRLProofAgentLoop", "VERL_AVAILABLE",
    "SlimeSampler", "SLIME_AVAILABLE",
    "SlimeProofEnvFactory", "SlimeProofEnv",
    # Tree rollout (v7)
    "TreeRolloutSampler", "TreeRolloutConfig",
    # Policy adapters (v7.1)
    "MockPolicy", "OpenAIPolicy", "CallablePolicy",
    "build_policy", "DEFAULT_SYSTEM_PROMPT",
    # Batch export (v7.1)
    "to_grpo_batch", "to_sft_jsonl", "to_ppo_batch", "save_batch_jsonl",
]
