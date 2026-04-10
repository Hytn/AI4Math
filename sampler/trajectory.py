"""sampler/trajectory.py — Rollout trajectory data structures

Framework-agnostic representation of a multi-turn proof attempt trajectory.
Each trajectory captures the full interaction sequence between the RL policy
(LLM) and the Lean verification environment.

A single trajectory:
  Turn 0: observe(theorem_statement) → action(tactic_1) → reward(lean_feedback_1)
  Turn 1: observe(feedback_1)        → action(tactic_2) → reward(lean_feedback_2)
  ...
  Turn N: observe(feedback_N-1)      → action(tactic_N) → reward(final: +1 or 0)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TerminationReason(Enum):
    """Why a trajectory ended."""
    SUCCESS = "success"              # Lean accepted the full proof
    MAX_TURNS = "max_turns"          # Hit turn budget
    MAX_TOKENS = "max_tokens"        # Hit token budget
    UNRECOVERABLE = "unrecoverable"  # Error classified as unrecoverable
    SORRY_DETECTED = "sorry"         # Model used `sorry` — integrity violation
    TIMEOUT = "timeout"              # Wall-clock timeout


@dataclass
class RewardInfo:
    """Structured reward signal from a single verification step.

    RL frameworks consume `scalar` directly. The structured fields
    (verification_level, error_class, etc.) support reward shaping,
    curriculum learning, and diagnostics.
    """
    scalar: float                    # The reward value for this turn
    verification_level: str = ""     # L0 / L1 / L2
    is_terminal: bool = False        # True if proof is complete or failed permanently
    error_class: str = ""            # ProofFailureClass name (if failed)
    goals_remaining: int = -1        # Number of unsolved goals after this tactic
    goals_closed: int = 0            # Goals closed by this tactic
    fix_hint: str = ""               # Suggested fix from ErrorIntelligence
    raw_feedback: str = ""           # Raw Lean output (truncated for context budget)


@dataclass
class Turn:
    """A single turn in the multi-turn proof interaction.

    observation: What the model sees (theorem + accumulated feedback)
    action:      What the model produces (tactic / proof code)
    reward:      Verification result
    token_ids:   For RL frameworks that need token-level data
    """
    turn_idx: int
    observation: str                 # Text observation shown to the model
    action: str                      # Model-generated tactic / proof code
    reward: RewardInfo = field(default_factory=lambda: RewardInfo(scalar=0.0))

    # Token-level data (populated by framework-specific samplers)
    observation_token_ids: list[int] = field(default_factory=list)
    action_token_ids: list[int] = field(default_factory=list)
    action_log_probs: list[float] = field(default_factory=list)
    action_mask: list[int] = field(default_factory=list)  # 1 = trainable token

    # Timing
    generation_ms: int = 0
    verification_ms: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.reward.is_terminal


@dataclass
class Trajectory:
    """A complete multi-turn proof attempt trajectory.

    Designed to be convertible to any RL framework's native format:
      - veRL: DataProto with prompt_ids / response_ids / response_mask
      - slime: list of (obs, act, rew, done) tuples
      - generic: flat token sequence with per-token reward assignment
    """
    problem_id: str                  # Benchmark problem identifier
    theorem_statement: str           # The Lean theorem to prove
    turns: list[Turn] = field(default_factory=list)
    termination: TerminationReason = TerminationReason.MAX_TURNS

    # Aggregate statistics
    total_reward: float = 0.0
    success: bool = False
    wall_time_s: float = 0.0
    total_tokens: int = 0

    # Metadata for diagnostics / logging
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def add_turn(self, turn: Turn):
        """Append a turn and update aggregates."""
        self.turns.append(turn)
        self.total_reward += turn.reward.scalar
        self.total_tokens += len(turn.action_token_ids)
        if turn.reward.is_terminal and turn.reward.scalar > 0:
            self.success = True
            self.termination = TerminationReason.SUCCESS

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    # ── Conversion helpers ──────────────────────────────────────────────

    def to_flat_token_sequence(self) -> dict[str, list]:
        """Flatten to a single token sequence for standard RL.

        Returns dict with keys:
          input_ids:  all observation + action tokens concatenated
          labels:     -100 for observation tokens, token_id for action tokens
          rewards:    0 for all tokens except last action token of each turn
          mask:       1 for action tokens, 0 for observation tokens
        """
        input_ids, labels, rewards, mask = [], [], [], []
        for turn in self.turns:
            # Observation tokens (not trained on)
            input_ids.extend(turn.observation_token_ids)
            labels.extend([-100] * len(turn.observation_token_ids))
            rewards.extend([0.0] * len(turn.observation_token_ids))
            mask.extend([0] * len(turn.observation_token_ids))

            # Action tokens (trained on)
            input_ids.extend(turn.action_token_ids)
            labels.extend(turn.action_token_ids)
            act_rewards = [0.0] * len(turn.action_token_ids)
            if act_rewards:
                act_rewards[-1] = turn.reward.scalar  # assign to last token
            rewards.extend(act_rewards)
            mask.extend(turn.action_mask or [1] * len(turn.action_token_ids))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "rewards": rewards,
            "mask": mask,
        }

    def to_verl_format(self) -> dict[str, Any]:
        """Convert to veRL-compatible dict.

        Maps to veRL's AgentLoopOutput fields:
          prompt_ids, response_ids, response_mask, response_logprobs, num_turns
        """
        prompt_ids = []
        response_ids = []
        response_mask = []
        response_logprobs = []

        for i, turn in enumerate(self.turns):
            if i == 0:
                prompt_ids = list(turn.observation_token_ids)
            else:
                # Subsequent observations become part of the response
                # (environment feedback is non-trainable)
                response_ids.extend(turn.observation_token_ids)
                response_mask.extend([0] * len(turn.observation_token_ids))
                response_logprobs.extend([0.0] * len(turn.observation_token_ids))

            response_ids.extend(turn.action_token_ids)
            response_mask.extend(
                turn.action_mask or [1] * len(turn.action_token_ids))
            response_logprobs.extend(
                turn.action_log_probs or [0.0] * len(turn.action_token_ids))

        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_mask": response_mask,
            "response_logprobs": response_logprobs,
            "num_turns": self.num_turns,
            "reward_score": self.total_reward,
            "success": self.success,
            "extra_fields": {
                "turn_scores": [t.reward.scalar for t in self.turns],
                "problem_id": self.problem_id,
                "termination": self.termination.value,
            },
        }

    def to_slime_episodes(self) -> list[dict[str, Any]]:
        """Convert to slime-compatible episode format.

        Each turn becomes a step in the episode:
          {"observation": str, "action": str, "reward": float,
           "done": bool, "info": dict}
        """
        steps = []
        for turn in self.turns:
            steps.append({
                "observation": turn.observation,
                "action": turn.action,
                "reward": turn.reward.scalar,
                "done": turn.is_terminal,
                "info": {
                    "verification_level": turn.reward.verification_level,
                    "error_class": turn.reward.error_class,
                    "goals_remaining": turn.reward.goals_remaining,
                    "fix_hint": turn.reward.fix_hint,
                },
            })
        return steps

    def summary(self) -> dict[str, Any]:
        """Compact summary for logging."""
        return {
            "problem_id": self.problem_id,
            "success": self.success,
            "num_turns": self.num_turns,
            "total_reward": round(self.total_reward, 4),
            "termination": self.termination.value,
            "wall_time_s": round(self.wall_time_s, 2),
            "total_tokens": self.total_tokens,
        }
