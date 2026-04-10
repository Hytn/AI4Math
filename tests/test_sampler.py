"""tests/test_sampler.py — Unit tests for the Sampler abstraction"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sampler.trajectory import Trajectory, Turn, RewardInfo, TerminationReason
from sampler.proof_env import ProofEnv, ProofEnvConfig
from sampler.base_sampler import BaseSampler, SamplerConfig
from sampler.verl_sampler import VeRLProofInteraction, compute_proof_reward
from sampler.slime_sampler import SlimeSampler
from sampler.reward_shaping import reshape_trajectory, RewardConfig


# ── Trajectory tests ──────────────────────────────────────────────────

class TestTrajectory:
    def test_empty_trajectory(self):
        t = Trajectory(problem_id="test", theorem_statement="theorem t : True")
        assert t.num_turns == 0
        assert not t.success
        assert t.total_reward == 0.0

    def test_add_turn(self):
        t = Trajectory(problem_id="test", theorem_statement="")
        turn = Turn(
            turn_idx=0, observation="prove True", action="trivial",
            reward=RewardInfo(scalar=1.0, is_terminal=True),
            action_token_ids=[1, 2, 3],
        )
        t.add_turn(turn)
        assert t.num_turns == 1
        assert t.success
        assert t.total_reward == 1.0
        assert t.total_tokens == 3

    def test_to_flat_token_sequence(self):
        t = Trajectory(problem_id="test", theorem_statement="")
        t.add_turn(Turn(
            turn_idx=0, observation="obs", action="act",
            reward=RewardInfo(scalar=0.5),
            observation_token_ids=[10, 11],
            action_token_ids=[20, 21, 22],
        ))
        flat = t.to_flat_token_sequence()
        assert flat["input_ids"] == [10, 11, 20, 21, 22]
        assert flat["labels"] == [-100, -100, 20, 21, 22]
        assert flat["mask"] == [0, 0, 1, 1, 1]
        # Reward only on last action token
        assert flat["rewards"] == [0.0, 0.0, 0.0, 0.0, 0.5]

    def test_to_verl_format(self):
        t = Trajectory(problem_id="test", theorem_statement="")
        t.add_turn(Turn(
            turn_idx=0, observation="obs", action="act",
            reward=RewardInfo(scalar=1.0, is_terminal=True),
            observation_token_ids=[10, 11],
            action_token_ids=[20, 21],
            action_log_probs=[-0.1, -0.2],
        ))
        verl = t.to_verl_format()
        assert verl["prompt_ids"] == [10, 11]
        assert verl["response_ids"] == [20, 21]
        assert verl["response_mask"] == [1, 1]
        assert verl["response_logprobs"] == [-0.1, -0.2]
        assert verl["success"]

    def test_to_slime_episodes(self):
        t = Trajectory(problem_id="test", theorem_statement="")
        t.add_turn(Turn(
            turn_idx=0, observation="obs", action="act",
            reward=RewardInfo(scalar=0.5, verification_level="L1"),
        ))
        t.add_turn(Turn(
            turn_idx=1, observation="fb", action="done",
            reward=RewardInfo(scalar=1.0, is_terminal=True),
        ))
        eps = t.to_slime_episodes()
        assert len(eps) == 2
        assert eps[0]["done"] is False
        assert eps[1]["done"] is True
        assert eps[0]["reward"] == 0.5

    def test_multi_turn_verl_format(self):
        t = Trajectory(problem_id="test", theorem_statement="")
        # Turn 0
        t.add_turn(Turn(
            turn_idx=0, observation="goal", action="intro h",
            reward=RewardInfo(scalar=0.05),
            observation_token_ids=[1, 2, 3],
            action_token_ids=[10, 11],
        ))
        # Turn 1
        t.add_turn(Turn(
            turn_idx=1, observation="feedback", action="exact h",
            reward=RewardInfo(scalar=1.0, is_terminal=True),
            observation_token_ids=[20, 21],
            action_token_ids=[30, 31],
        ))
        verl = t.to_verl_format()
        # Turn 0 obs → prompt_ids
        assert verl["prompt_ids"] == [1, 2, 3]
        # Turn 0 action + Turn 1 obs (non-trainable) + Turn 1 action
        assert verl["response_ids"] == [10, 11, 20, 21, 30, 31]
        assert verl["response_mask"] == [1, 1, 0, 0, 1, 1]


# ── ProofEnv tests ────────────────────────────────────────────────────

class TestProofEnv:
    def test_sorry_detection(self):
        """sorry should terminate with negative reward."""
        async def _test():
            env = ProofEnv(ProofEnvConfig())
            # Mock the pool and verifier
            env._pool = MagicMock()
            env._verifier = MagicMock()
            env._prefilter = MagicMock()
            env._error_intel = MagicMock()
            env._broadcast = MagicMock()

            await env.reset({
                "problem_id": "t1",
                "theorem_statement": "theorem t : True := by",
            })
            obs, reward, done, info = await env.step("sorry")
            assert done
            assert reward.scalar < 0
            assert env.get_trajectory().termination == TerminationReason.SORRY_DETECTED

        asyncio.run(_test())

    def test_max_turns(self):
        """Should terminate when max turns reached."""
        async def _test():
            cfg = ProofEnvConfig(max_turns=2)
            env = ProofEnv(cfg)
            env._pool = MagicMock()

            # Mock verifier that always returns success=False
            mock_vr = MagicMock()
            mock_vr.success = False
            mock_vr.l0_passed = True
            mock_vr.level_reached = "L1"
            mock_vr.feedback = None
            mock_vr.l0_reject_reason = None
            mock_vr.goals_after = 1
            mock_vr.new_env_id = None

            mock_verifier = AsyncMock()
            mock_verifier.verify_tactic = AsyncMock(return_value=mock_vr)
            env._verifier = mock_verifier
            env._prefilter = MagicMock()
            env._error_intel = MagicMock()
            env._broadcast = MagicMock()

            await env.reset({"problem_id": "t2", "theorem_statement": "test"})

            # Turn 0
            obs, r, done, _ = await env.step("simp")
            assert not done
            # Turn 1
            obs, r, done, _ = await env.step("ring")
            assert not done
            # Turn 2 — should hit max_turns
            obs, r, done, _ = await env.step("omega")
            assert done

        asyncio.run(_test())


# ── Reward shaping tests ──────────────────────────────────────────────

class TestRewardShaping:
    def test_sparse_success(self):
        t = Trajectory(problem_id="x", theorem_statement="")
        t.add_turn(Turn(0, "o", "a1", RewardInfo(scalar=0.1)))
        t.add_turn(Turn(1, "o", "a2", RewardInfo(scalar=1.0, is_terminal=True)))
        t.success = True
        t = reshape_trajectory(t, RewardConfig(strategy="sparse", turn_discount=1.0))
        assert t.turns[0].reward.scalar == 0.0
        assert t.turns[1].reward.scalar == 1.0

    def test_sparse_failure(self):
        t = Trajectory(problem_id="x", theorem_statement="")
        t.add_turn(Turn(0, "o", "a1", RewardInfo(scalar=0.1)))
        t.add_turn(Turn(1, "o", "sorry", RewardInfo(scalar=-0.5, error_class="sorry")))
        t = reshape_trajectory(t, RewardConfig(strategy="sparse", turn_discount=1.0))
        assert t.turns[1].reward.scalar == -0.5

    def test_turn_discount(self):
        t = Trajectory(problem_id="x", theorem_statement="")
        t.add_turn(Turn(0, "o", "a1", RewardInfo(scalar=1.0)))
        t.add_turn(Turn(1, "o", "a2", RewardInfo(scalar=1.0)))
        t = reshape_trajectory(t, RewardConfig(strategy="dense", turn_discount=0.5))
        assert t.turns[0].reward.scalar == 1.0
        assert t.turns[1].reward.scalar == 0.5


# ── veRL integration tests ───────────────────────────────────────────

class TestVeRLIntegration:
    def test_compute_proof_reward_sorry(self):
        r = compute_proof_reward("test", "sorry", {})
        assert r == -0.5

    def test_compute_proof_reward_success(self):
        r = compute_proof_reward("test", "exact h", {}, {"success": True})
        assert r == 1.0

    def test_compute_proof_reward_empty(self):
        r = compute_proof_reward("test", "", {})
        assert r == 0.0

    def test_extract_tactic_from_code_block(self):
        interaction = VeRLProofInteraction({"name": "test"})
        messages = [
            {"role": "user", "content": "prove this"},
            {"role": "assistant", "content": "```lean\nexact h\n```"},
        ]
        assert interaction._extract_tactic(messages) == "exact h"

    def test_extract_tactic_raw(self):
        interaction = VeRLProofInteraction({"name": "test"})
        messages = [
            {"role": "assistant", "content": "simp [Nat.add_comm]"},
        ]
        assert interaction._extract_tactic(messages) == "simp [Nat.add_comm]"


# ── SlimeSampler tests ────────────────────────────────────────────────

class TestSlimeSampler:
    def test_episode_format(self):
        t = Trajectory(problem_id="p1", theorem_statement="test")
        t.add_turn(Turn(0, "obs0", "act0", RewardInfo(scalar=0.1)))
        t.add_turn(Turn(1, "obs1", "act1", RewardInfo(scalar=1.0, is_terminal=True)))
        eps = t.to_slime_episodes()
        assert len(eps) == 2
        assert eps[0]["observation"] == "obs0"
        assert eps[0]["action"] == "act0"
        assert eps[0]["reward"] == 0.1
        assert not eps[0]["done"]
        assert eps[1]["done"]
