"""sampler/reward_shaping.py — Reward shaping for formal proof RL

Provides configurable reward functions that go beyond binary success/failure.
These can be plugged into ProofEnvConfig or used as standalone reward
transformations in veRL/slime training loops.

Strategies:
  - sparse:       +1 on full proof, 0 otherwise
  - goal_progress: Reward based on goals closed / remaining
  - curriculum:    Adjust rewards based on problem difficulty
  - dense:        Per-turn reward from verification feedback
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sampler.trajectory import Trajectory, RewardInfo


@dataclass
class RewardConfig:
    strategy: str = "dense"          # sparse | goal_progress | curriculum | dense
    success_bonus: float = 1.0
    sorry_penalty: float = -0.5
    per_goal_bonus: float = 0.1
    l1_pass_bonus: float = 0.05
    l0_reject_penalty: float = -0.02
    timeout_penalty: float = -0.1
    # Discount for future turns (encourages shorter proofs)
    turn_discount: float = 0.99
    # Curriculum: scale reward by difficulty
    difficulty_scale: bool = False


def reshape_trajectory(
    trajectory: Trajectory, config: RewardConfig = None
) -> Trajectory:
    """Apply reward shaping to a completed trajectory.

    Modifies rewards in-place and returns the trajectory.
    """
    cfg = config or RewardConfig()

    if cfg.strategy == "sparse":
        _apply_sparse(trajectory, cfg)
    elif cfg.strategy == "goal_progress":
        _apply_goal_progress(trajectory, cfg)
    elif cfg.strategy == "curriculum":
        _apply_curriculum(trajectory, cfg)
    else:  # dense (default — keep per-turn rewards as-is)
        pass

    # Apply turn discount
    if cfg.turn_discount < 1.0:
        for i, turn in enumerate(trajectory.turns):
            turn.reward.scalar *= cfg.turn_discount ** i

    # Recompute total
    trajectory.total_reward = sum(t.reward.scalar for t in trajectory.turns)
    return trajectory


def _apply_sparse(traj: Trajectory, cfg: RewardConfig):
    """Only the final turn gets reward."""
    for turn in traj.turns:
        turn.reward.scalar = 0.0
    if traj.turns:
        if traj.success:
            traj.turns[-1].reward.scalar = cfg.success_bonus
        elif traj.turns[-1].reward.error_class == "sorry":
            traj.turns[-1].reward.scalar = cfg.sorry_penalty


def _apply_goal_progress(traj: Trajectory, cfg: RewardConfig):
    """Reward based on cumulative goal progress."""
    for turn in traj.turns:
        r = 0.0
        if turn.reward.goals_closed > 0:
            r += cfg.per_goal_bonus * turn.reward.goals_closed
        if turn.reward.is_terminal and turn.reward.scalar > 0:
            r += cfg.success_bonus
        turn.reward.scalar = r


def _apply_curriculum(traj: Trajectory, cfg: RewardConfig):
    """Scale rewards by problem difficulty metadata."""
    difficulty = traj.metadata.get("difficulty", 1.0)
    scale = 1.0 + (difficulty - 1.0) * 0.5  # harder = higher reward
    for turn in traj.turns:
        turn.reward.scalar *= scale
    if traj.success and traj.turns:
        traj.turns[-1].reward.scalar += cfg.success_bonus * scale
