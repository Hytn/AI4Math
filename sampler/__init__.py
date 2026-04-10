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
from sampler.verl_sampler import VeRLProofInteraction, VeRLProofAgentLoop
from sampler.slime_sampler import SlimeSampler

__all__ = [
    "Trajectory", "Turn", "RewardInfo",
    "ProofEnv", "ProofEnvConfig",
    "BaseSampler", "SamplerConfig",
    "VeRLProofInteraction", "VeRLProofAgentLoop",
    "SlimeSampler",
]
