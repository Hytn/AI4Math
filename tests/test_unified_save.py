"""tests/test_unified_save.py — End-to-end tests for single-file storage

Verifies that EVERY trajectory class in AI4Math produces a
self-contained ``dialog.json`` via ``save_unified()``:

  <task_dir>/
    └── dialog.json   ← {schema_version, meta, messages, result}

The five classes covered:
  1. ``prover.models.ProofTrace``
  2. ``sampler.trajectory.Trajectory``
  3. ``agent.runtime.agent_loop.LoopResult``
  4. ``agent.persistence.session_store.SessionData`` (sidecar)
  5. ``engine.lane.proof_session_store.ProofSessionSnapshot``
"""
from __future__ import annotations

import json
import os
import sys

import pytest

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)


from agent.persistence import (
    DIALOG_FILENAME, SCHEMA_VERSION,
    load_task, validate_dialog, collect_dialogs,
    messages_of, meta_of, result_of,
)


# ─────────────────────────────────────────────────────────────────────────
# 1. ProofTrace.save_unified — single dialog.json with everything
# ─────────────────────────────────────────────────────────────────────────

class TestProofTraceUnified:
    def test_single_file_only(self, tmp_path):
        from prover.models import (
            ProofTrace, ProofAttempt, AttemptStatus,
            LeanError, ErrorCategory,
        )
        a1 = ProofAttempt(
            attempt_number=1,
            generated_proof=":= by ring",
            llm_model="qwen3",
            lean_result=AttemptStatus.LEAN_ERROR,
            lean_errors=[LeanError(
                category=ErrorCategory.TACTIC_FAILED,
                message="ring failed")],
        )
        a2 = ProofAttempt(
            attempt_number=2,
            generated_proof=":= by simp",
            llm_model="qwen3",
            lean_result=AttemptStatus.SUCCESS,
        )
        trace = ProofTrace(
            problem_id="t01", problem_name="add_zero",
            theorem_statement="theorem t (n : ℕ) : n + 0 = n",
        )
        trace.add_attempt(a1)
        trace.add_attempt(a2)

        task_dir = tmp_path / "t01"
        trace.save_unified(
            task_dir, model="qwen3-32b", provider="local",
            system_prompt="You are a Lean prover.",
            tools=[{"name": "lean_verify",
                    "description": "Verify Lean code"}],
        )

        # ONLY dialog.json — no result.json, no meta_config.json
        files = sorted(p.name for p in task_dir.iterdir())
        assert files == [DIALOG_FILENAME]

        # Load and check the wrapped object has everything inline
        d = load_task(task_dir)
        assert d["schema_version"] == SCHEMA_VERSION

        # meta carries problem identity, model, system prompt, tools
        meta = meta_of(d)
        assert meta["problem_id"] == "t01"
        assert meta["theorem_statement"] == \
               "theorem t (n : ℕ) : n + 0 = n"
        assert meta["model"] == "qwen3-32b"
        assert meta["provider"] == "local"
        assert meta["system_prompt"] == "You are a Lean prover."
        assert any(t["name"] == "lean_verify" for t in meta["tools"])

        # messages are well-formed
        assert validate_dialog(d) == []

        # result captures outcome
        res = result_of(d)
        assert res["success"] is True
        assert res["total_attempts"] == 2
        assert res["successful_proof"] == ":= by simp"


# ─────────────────────────────────────────────────────────────────────────
# 2. Trajectory.save_unified
# ─────────────────────────────────────────────────────────────────────────

class TestTrajectoryUnified:
    def test_single_file(self, tmp_path):
        from sampler.trajectory import (
            Trajectory, Turn, RewardInfo, TerminationReason,
        )
        traj = Trajectory(
            problem_id="rl01",
            theorem_statement="n + 0 = n",
        )
        traj.add_turn(Turn(
            turn_idx=0, observation="Prove: n + 0 = n",
            action="rfl",
            reward=RewardInfo(scalar=0.0, verification_level="L1"),
        ))
        traj.add_turn(Turn(
            turn_idx=1, observation="Try again",
            action="by simp",
            reward=RewardInfo(scalar=1.0, verification_level="L2",
                              is_terminal=True),
        ))
        traj.wall_time_s = 1.5
        traj.success = True
        traj.termination = TerminationReason.SUCCESS

        task_dir = tmp_path / "rl01"
        traj.save_unified(task_dir, model="rl-v1",
                          system_prompt="prove tactics")

        # Only dialog.json
        assert sorted(p.name for p in task_dir.iterdir()) \
               == [DIALOG_FILENAME]

        d = load_task(task_dir)
        assert validate_dialog(d) == []
        assert meta_of(d)["problem_id"] == "rl01"
        assert meta_of(d)["theorem_statement"] == "n + 0 = n"
        assert meta_of(d)["system_prompt"] == "prove tactics"
        assert result_of(d)["success"] is True


# ─────────────────────────────────────────────────────────────────────────
# 3. LoopResult.save_unified
# ─────────────────────────────────────────────────────────────────────────

class TestLoopResultUnified:
    def test_single_file(self, tmp_path):
        from agent.runtime.agent_loop import LoopResult, LoopMessage
        result = LoopResult(
            content="Done. ```lean\ntheorem t : True := trivial\n```",
            proof_code="theorem t : True := trivial",
            messages=[
                LoopMessage(role="user", content="Prove True"),
                LoopMessage(role="assistant",
                            content="```lean\ntheorem t : True := trivial\n```"),
            ],
            turns_used=1,
            total_tokens=42,
            total_latency_ms=1200,
            tools_called=[],
            stopped_reason="proof_found",
        )

        task_dir = tmp_path / "loop01"
        result.save_unified(
            task_dir, problem_id="loop01",
            model="claude-opus", provider="anthropic",
            system_prompt="prove things",
        )

        d = load_task(task_dir)
        assert sorted(p.name for p in task_dir.iterdir()) \
               == [DIALOG_FILENAME]
        assert validate_dialog(d) == []
        assert meta_of(d)["problem_id"] == "loop01"
        assert meta_of(d)["model"] == "claude-opus"
        assert meta_of(d)["system_prompt"] == "prove things"
        assert result_of(d)["success"] is True
        assert result_of(d)["termination"] == "proof_found"


# ─────────────────────────────────────────────────────────────────────────
# 4. (TestSessionStoreSidecar deleted in v11: agent/persistence/session_store.py removed.)
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# 5. (TestSnapshotUnified deleted in v9: engine/lane/ removed.)
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# 6. End-to-end: collect & SFT export uses meta.system_prompt
# ─────────────────────────────────────────────────────────────────────────

class TestEndToEndSFT:
    def test_collect_and_sft_export(self, tmp_path):
        from prover.models import ProofTrace, ProofAttempt, AttemptStatus
        from sampler.trajectory import Trajectory, Turn, RewardInfo
        from agent.persistence import dialogs_to_sft_jsonl

        root = tmp_path / "results" / "traces"
        root.mkdir(parents=True)

        # 1. ProofTrace with system prompt baked into meta
        t = ProofTrace(
            problem_id="pt01",
            theorem_statement="theorem t : True := trivial")
        t.add_attempt(ProofAttempt(
            generated_proof=":= trivial",
            lean_result=AttemptStatus.SUCCESS))
        t.save_unified(root / "pt01",
                       model="qwen3", system_prompt="prove A")

        # 2. Trajectory with different system prompt
        traj = Trajectory(problem_id="tr01",
                          theorem_statement="True")
        traj.add_turn(Turn(
            turn_idx=0, observation="Prove True",
            action="trivial",
            reward=RewardInfo(scalar=1.0, is_terminal=True)))
        traj.save_unified(root / "tr01", system_prompt="prove B")

        items = collect_dialogs(root)
        assert len(items) == 2

        # Each one carries its own system prompt
        for _, dialog in items:
            assert validate_dialog(dialog) == []
            assert meta_of(dialog).get("system_prompt") in {
                "prove A", "prove B"}

        # SFT export — system prompts come from each dialog's meta
        out = tmp_path / "sft.jsonl"
        n = dialogs_to_sft_jsonl(
            [d for _, d in items], str(out), preset="qwen3")
        assert n == 2

        lines = out.read_text("utf-8").strip().split("\n")
        # Exactly one sample per dialog, each containing the right
        # system prompt rendered into the chat template
        prompts_seen = set()
        for line in lines:
            sample = json.loads(line)
            assert "<|im_start|>system" in sample["text"]
            for p in ("prove A", "prove B"):
                if p in sample["text"]:
                    prompts_seen.add(p)
        assert prompts_seen == {"prove A", "prove B"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
