"""sampler/batch_export.py — Trajectory batch export helpers

The sampler returns ``list[Trajectory]``; trainers want batched dicts
in their native shape. Per-traj methods (``to_verl_format()``,
``to_slime_episodes()``) already cover the single-traj case. This
module covers the *batch* case, including the GRPO group-shape that
``TreeRolloutSampler`` is built around.

Three shapes:

  * ``to_grpo_batch(trajs)`` — groups by problem_id, computes per-group
    advantages (mean-centred / std-normalised), packs into a
    verl-DataProto-compatible dict-of-lists.
  * ``to_sft_jsonl(trajs, path, *, successful_only=True)`` — the
    SFT-only export.
  * ``to_ppo_batch(trajs)`` — flat per-step batch with token-level
    rewards and the optional value-baseline columns when present.

The dict shapes are documented in line. None of these methods touch
the trainer SDK directly — they emit ``dict[str, list]`` shapes that
verl's ``DataProto.from_dict`` and slime's ``Episode.from_dict``
accept verbatim.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Iterable

from sampler.trajectory import Trajectory

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# GRPO batch — group by problem_id, compute advantages
# ═══════════════════════════════════════════════════════════════════════

def to_grpo_batch(
    trajectories: Iterable[Trajectory],
    *,
    advantage_kind: str = "centered_normalized",
    min_group_size: int = 2,
) -> dict[str, list]:
    """Pack trajectories into a GRPO batch.

    Trajectories with the same ``problem_id`` form a group; advantages
    are computed within-group. Singleton groups (``< min_group_size``)
    get advantage 0 — they're useful for accounting but don't contribute
    a meaningful gradient.

    Args:
        trajectories: input list (e.g. from ``TreeRolloutSampler.collect_rollouts``).
        advantage_kind:
            * ``"centered_normalized"`` (default): ``(R - μ) / σ`` per group
            * ``"centered"``: ``R - μ`` per group, no scaling
            * ``"raw"``: ``R`` per traj, no group adjustment
        min_group_size: groups smaller than this get advantage 0.

    Returns:
        Dict with keys::

            problem_ids:       [str, ...] one per traj
            prompt_ids:        [list[int], ...]
            response_ids:      [list[int], ...]
            response_mask:     [list[int], ...] 1 = trainable, 0 = env feedback
            response_logprobs: [list[float], ...]
            rewards:           [float, ...] per-traj total reward
            advantages:        [float, ...] per-traj GRPO advantage
            num_turns:         [int, ...]
            success:           [bool, ...]
            group_id:          [str, ...] (= problem_id, redundant convenience)
            group_size:        [int, ...] cardinality of each traj's group
            metadata:          [dict, ...] each traj's metadata (search_kind, etc.)
    """
    trajs = list(trajectories)
    if not trajs:
        return _empty_grpo_batch()

    # Group by problem_id
    groups: dict[str, list[Trajectory]] = {}
    for t in trajs:
        groups.setdefault(t.problem_id, []).append(t)

    # Compute per-group statistics
    group_stats: dict[str, tuple[float, float]] = {}  # problem_id -> (mu, sd)
    for pid, group in groups.items():
        if len(group) >= min_group_size:
            rewards = [t.total_reward for t in group]
            mu = sum(rewards) / len(rewards)
            var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
            sd = math.sqrt(var) if var > 1e-9 else 1.0
            group_stats[pid] = (mu, sd)
        else:
            group_stats[pid] = (0.0, 1.0)

    out: dict[str, list] = {
        "problem_ids": [],
        "prompt_ids": [],
        "response_ids": [],
        "response_mask": [],
        "response_logprobs": [],
        "rewards": [],
        "advantages": [],
        "num_turns": [],
        "success": [],
        "group_id": [],
        "group_size": [],
        "metadata": [],
    }

    for t in trajs:
        verl = t.to_verl_format()
        mu, sd = group_stats[t.problem_id]
        if advantage_kind == "raw":
            adv = t.total_reward
        elif advantage_kind == "centered":
            adv = t.total_reward - mu
        else:  # "centered_normalized"
            adv = (t.total_reward - mu) / sd if sd > 0 else 0.0
            if len(groups[t.problem_id]) < min_group_size:
                adv = 0.0  # singleton group

        out["problem_ids"].append(t.problem_id)
        out["prompt_ids"].append(verl["prompt_ids"])
        out["response_ids"].append(verl["response_ids"])
        out["response_mask"].append(verl["response_mask"])
        out["response_logprobs"].append(verl["response_logprobs"])
        out["rewards"].append(t.total_reward)
        out["advantages"].append(adv)
        out["num_turns"].append(t.num_turns)
        out["success"].append(bool(t.success))
        out["group_id"].append(t.problem_id)
        out["group_size"].append(len(groups[t.problem_id]))
        out["metadata"].append(dict(t.metadata))

    return out

def _empty_grpo_batch() -> dict[str, list]:
    return {k: [] for k in (
        "problem_ids", "prompt_ids", "response_ids", "response_mask",
        "response_logprobs", "rewards", "advantages",
        "num_turns", "success", "group_id", "group_size", "metadata",
    )}

# ═══════════════════════════════════════════════════════════════════════
# SFT JSONL — successful-only by default
# ═══════════════════════════════════════════════════════════════════════

def to_sft_jsonl(
    trajectories: Iterable[Trajectory],
    path: str | Path,
    *,
    successful_only: bool = True,
    preset: str = "qwen3",
) -> int:
    """Write trajectories to an SFT-ready JSONL.

    Each line is a single dialog in the standard chat-tuning shape
    ``{"messages": [{"role": ..., "content": ...}, ...]}``.

    Args:
        trajectories: iterable of Trajectory objects.
        path: output JSONL file path.
        successful_only: if True (default for SFT), drops failed
            trajectories. Set False to retain failures (only useful
            for DPO / contrastive setups).
        preset: forwarded to ``agent.persistence.sft_export.dialog_to_sft_record``;
            controls the chat template wrapper.

    Returns:
        Number of records written.
    """
    from agent.persistence.sft_export import dialog_to_sft_sample

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as fout:
        for t in trajectories:
            if successful_only and not t.success:
                continue
            try:
                dialog = t.to_dialog()
                rec = dialog_to_sft_sample(dialog, preset=preset)
                if rec is None:
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False))
                fout.write("\n")
                written += 1
            except Exception as e:
                logger.warning(
                    "to_sft_jsonl: skipping traj %s: %r",
                    t.problem_id, e)
    return written

# ═══════════════════════════════════════════════════════════════════════
# PPO flat-step batch — for token-level credit assignment
# ═══════════════════════════════════════════════════════════════════════

def to_ppo_batch(
    trajectories: Iterable[Trajectory],
    *,
    discount: float = 1.0,
    gae_lambda: float = 1.0,
    bootstrap_value: float = 0.0,
) -> dict[str, list]:
    """Pack trajectories into a flat per-step PPO batch.

    Per-trajectory rewards are placed on the *last action token* of
    each turn (matching ``Trajectory.to_flat_token_sequence``); GAE
    advantages are computed turn-by-turn within each trajectory using
    the supplied ``discount`` and ``gae_lambda``. When no value
    estimator is plugged in, V(s) is taken as ``bootstrap_value`` for
    every step (i.e. the bootstrap is constant — equivalent to plain
    REINFORCE if you set ``gae_lambda=1.0``).

    Returns:
        Dict with::

            input_ids:        [list[int], ...] one per traj (concat obs+act)
            labels:           [list[int], ...] action tokens, -100 elsewhere
            rewards:          [list[float], ...] per-token rewards
            advantages:       [list[float], ...] per-token GAE advantages
            returns:          [list[float], ...] per-token returns (R = A + V)
            mask:             [list[int], ...] 1 = action token
            problem_id:       [str, ...]
            success:          [bool, ...]
    """
    out: dict[str, list] = {
        "input_ids": [], "labels": [], "rewards": [], "advantages": [],
        "returns": [], "mask": [], "problem_id": [], "success": [],
    }

    for t in trajectories:
        flat = t.to_flat_token_sequence()
        rewards = list(flat["rewards"])
        # GAE per turn — but flat() puts reward on the last action token
        # of each turn. So we compute advantages on the same per-token grid.
        advantages = _compute_gae(rewards, discount=discount,
                                     gae_lambda=gae_lambda,
                                     bootstrap=bootstrap_value)
        returns = [a + bootstrap_value for a in advantages]

        out["input_ids"].append(flat["input_ids"])
        out["labels"].append(flat["labels"])
        out["rewards"].append(rewards)
        out["advantages"].append(advantages)
        out["returns"].append(returns)
        out["mask"].append(flat["mask"])
        out["problem_id"].append(t.problem_id)
        out["success"].append(bool(t.success))

    return out

def _compute_gae(
    rewards: list[float],
    *,
    discount: float = 1.0,
    gae_lambda: float = 1.0,
    bootstrap: float = 0.0,
) -> list[float]:
    """Generalised Advantage Estimation (Schulman et al., 2016).

    With ``discount=1.0, gae_lambda=1.0, bootstrap=0.0`` this reduces to
    plain Monte Carlo returns minus a constant baseline (= REINFORCE).
    """
    n = len(rewards)
    if n == 0:
        return []
    advantages = [0.0] * n
    gae = 0.0
    for t in range(n - 1, -1, -1):
        next_v = bootstrap if t == n - 1 else 0.0
        delta = rewards[t] + discount * next_v - bootstrap
        gae = delta + discount * gae_lambda * gae
        advantages[t] = gae
    return advantages

# ═══════════════════════════════════════════════════════════════════════
# Save / load batch dumps for offline pipelines
# ═══════════════════════════════════════════════════════════════════════

def save_batch_jsonl(batch: dict[str, list],
                       path: str | Path) -> int:
    """Save a GRPO/PPO batch as JSONL (one row per trajectory).

    Useful when you want to inspect / debug / shuffle a batch on disk
    before feeding it to verl.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    if not batch:
        path.write_text("")
        return 0
    keys = list(batch.keys())
    n_rows = len(batch[keys[0]])
    with open(path, "w", encoding="utf-8") as fout:
        for i in range(n_rows):
            row = {k: batch[k][i] for k in keys}
            fout.write(json.dumps(row, ensure_ascii=False, default=str))
            fout.write("\n")
    return n_rows
