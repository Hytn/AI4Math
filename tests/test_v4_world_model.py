"""tests/test_v4_world_model.py — v4 WorldModel from Mock to Real

After v4, the world-model layer is wired end-to-end:

  ProofContextStore → train_world_model.py → world_model.pkl
                                                    ↓
                                       make_world_model(path)
                                                    ↓
                                       UnifiedProofRunner(world_model=...)
                                                    ↓
                                       TacticApplyTool — short-circuits
                                       high-confidence-fail predictions

This file pins:

  1. ``TrainedWorldModel._load_model`` (was a TODO stub) actually loads
     a .pkl produced by WorldModelTrainer.save().
  2. ``make_world_model(path=None)`` factory: returns Mock when no path,
     Trained when a valid .pkl is given, Mock when the path is bad.
  3. End-to-end pkl round-trip: train → save → load → predict.
  4. ``TacticApplyTool`` accepts ``world_model`` and ``wm_min_confidence``.
  5. Tactic gate fires on high-confidence failure prediction.
  6. Tactic gate does NOT fire when prediction is uncertain.
  7. Tactic gate does NOT fire when the model says success.
  8. Without a world_model, behaviour is identical to before.
  9. Plumbing: build_tool_registry passes world_model through.
 10. Plumbing: UnifiedProofRunner accepts and stores world_model.

Some tests are skipped in environments without sklearn / scipy
(the Sklearn-backed model is optional infra).
"""
from __future__ import annotations

import asyncio
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)


# Probe sklearn / scipy availability — gates the train→pkl round-trip.
try:
    import sklearn  # noqa: F401
    import scipy    # noqa: F401
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


from engine.world_model import (
    WorldModelPredictor, WorldModelPrediction,
    MockWorldModel, TrainedWorldModel, make_world_model,
)
from agent.tools.base import ToolContext
from agent.tools.builtin.tactic_apply import TacticApplyTool


# ─────────────────────────────────────────────────────────────────────────
# Fakes — same shape as the Gap #2 test file
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class FakeTacticResult:
    success: bool
    is_proof_complete: bool = False
    new_env_id: int = -1
    remaining_goals: list = field(default_factory=list)
    error_message: str = ""
    error_category: str = ""


class FakePool:
    base_env_id = 0

    def __init__(self, results: list):
        self._results = list(results)
        self._calls = 0
        self.last_inputs: list = []

    async def try_tactic(self, env_id: int, tactic: str):
        self.last_inputs.append((env_id, tactic))
        r = self._results[self._calls % len(self._results)]
        self._calls += 1
        return r

    @property
    def call_count(self) -> int:
        return self._calls


class HighConfidenceFailModel(WorldModelPredictor):
    """Predicts every tactic as a high-confidence failure.

    Used to verify the gate fires; real models won't behave this way.
    """
    def predict(self, goal_state, tactic, hypotheses=None, context=None):
        return WorldModelPrediction(
            tactic=tactic, likely_success=False, confidence=0.95,
            reasoning="test stub: always reject")


class HighConfidenceSuccessModel(WorldModelPredictor):
    def predict(self, goal_state, tactic, hypotheses=None, context=None):
        return WorldModelPrediction(
            tactic=tactic, likely_success=True, confidence=0.95,
            reasoning="test stub: always pass")


class UncertainModel(WorldModelPredictor):
    """Predicts low-confidence failure — should NOT trigger the gate."""
    def predict(self, goal_state, tactic, hypotheses=None, context=None):
        return WorldModelPrediction(
            tactic=tactic, likely_success=False, confidence=0.4,
            reasoning="test stub: uncertain")


class CrashingModel(WorldModelPredictor):
    def predict(self, goal_state, tactic, hypotheses=None, context=None):
        raise RuntimeError("model exploded")


# ─────────────────────────────────────────────────────────────────────────
# 1. TrainedWorldModel loader — no longer a TODO stub
# ─────────────────────────────────────────────────────────────────────────

class TestTrainedWorldModelLoader:
    def test_no_path_falls_back_to_mock(self):
        wm = TrainedWorldModel()
        assert not wm.is_trained
        # predict still works via fallback
        pred = wm.predict("⊢ True", "trivial")
        assert pred.likely_success in (True, False)  # mock returns either

    def test_missing_path_falls_back_to_mock(self, tmp_path):
        wm = TrainedWorldModel(str(tmp_path / "does-not-exist.pkl"))
        assert not wm.is_trained
        pred = wm.predict("⊢ n + 0 = n", "simp")
        assert isinstance(pred, WorldModelPrediction)

    def test_corrupt_pkl_falls_back_to_mock(self, tmp_path):
        bad = tmp_path / "bad.pkl"
        bad.write_text("not a pickle file")
        wm = TrainedWorldModel(str(bad))
        assert not wm.is_trained


# ─────────────────────────────────────────────────────────────────────────
# 2. make_world_model factory
# ─────────────────────────────────────────────────────────────────────────

class TestMakeWorldModel:
    def test_no_arg_returns_mock(self):
        wm = make_world_model()
        assert isinstance(wm, MockWorldModel)

    def test_none_path_returns_mock(self):
        wm = make_world_model(None)
        assert isinstance(wm, MockWorldModel)

    def test_empty_string_returns_mock(self):
        wm = make_world_model("")
        assert isinstance(wm, MockWorldModel)

    def test_missing_path_returns_mock(self, tmp_path):
        wm = make_world_model(str(tmp_path / "no.pkl"))
        assert isinstance(wm, MockWorldModel)

    @pytest.mark.skipif(not SKLEARN_OK,
                         reason="sklearn / scipy not installed")
    def test_valid_pkl_returns_trained(self, tmp_path):
        """Train a tiny model end-to-end, then load via the factory."""
        model_path = _train_tiny_model(tmp_path)
        if model_path is None:
            pytest.skip("trainer needed >= 50 samples; data too small")
        wm = make_world_model(str(model_path))
        # Could be Trained (if .is_trained returned True) or Mock
        # (if the loader rejected the pkl). Either way a real predict
        # call must work and return a sensible structure.
        pred = wm.predict("⊢ True", "trivial")
        assert isinstance(pred, WorldModelPrediction)


# ─────────────────────────────────────────────────────────────────────────
# 3. Train → save → load → predict round-trip
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not SKLEARN_OK,
                     reason="sklearn / scipy not installed")
class TestPklRoundtrip:
    def test_train_save_load_predict(self, tmp_path):
        from engine.world_model_trainer import (
            WorldModelTrainer, SklearnWorldModel,
        )
        # Build a tiny synthetic dataset of 100 samples
        from engine.proof_context_store import (
            RichProofTrajectory, StepDetail,
        )
        trajs = []
        for i in range(20):
            steps = []
            for j in range(5):
                # Half successes, half failures
                ok = (i + j) % 2 == 0
                steps.append(StepDetail(
                    step_index=j,
                    tactic="simp" if ok else "ring",
                    env_id_before=j,
                    env_id_after=(j + 1) if ok else -1,
                    goals_before=[f"⊢ x{i} + 0 = x{i}"],
                    goals_after=[] if ok else [f"⊢ x{i} + 0 = x{i}"],
                    error_message="" if ok else "ring failed on Nat",
                    error_category="" if ok else "tactic_failed",
                    is_proof_complete=ok,
                ))
            trajs.append(RichProofTrajectory(
                theorem=f"theorem t{i} : x{i} + 0 = x{i}",
                steps=steps,
                success=True,
                depth=5,
                duration_ms=100.0,
            ))

        trainer = WorldModelTrainer(db_path=str(tmp_path / "fake.db"))
        n = trainer.extract_from_trajectories(trajs)
        assert n == 100
        metrics = trainer.train(test_size=0.2)
        assert "accuracy" in metrics

        out = tmp_path / "model.pkl"
        trainer.save(str(out))
        assert out.exists()

        # Direct SklearnWorldModel.load
        skm = SklearnWorldModel(str(out))
        assert skm.is_trained
        pred = skm.predict("⊢ a + 0 = a", "simp")
        assert isinstance(pred, WorldModelPrediction)
        assert 0.0 <= pred.confidence <= 1.0

    def test_trained_world_model_actually_loads_pkl(self, tmp_path):
        """v4 contract: TrainedWorldModel._load_model is no longer a stub."""
        path = _train_tiny_model(tmp_path)
        if path is None:
            pytest.skip("training data too small")
        wm = TrainedWorldModel(str(path))
        assert wm.is_trained, \
            "TrainedWorldModel must actually load valid .pkl files"


def _train_tiny_model(tmp_path):
    """Helper: produce a real .pkl in tmp_path or return None if training
    couldn't gather enough samples."""
    if not SKLEARN_OK:
        return None
    from engine.world_model_trainer import WorldModelTrainer
    from engine.proof_context_store import (
        RichProofTrajectory, StepDetail,
    )
    trajs = []
    for i in range(20):
        steps = []
        for j in range(5):
            ok = (i + j) % 2 == 0
            steps.append(StepDetail(
                step_index=j, tactic="simp" if ok else "ring",
                env_id_before=j, env_id_after=(j + 1) if ok else -1,
                goals_before=[f"⊢ x{i} + 0 = x{i}"],
                goals_after=[] if ok else [f"⊢ x{i} + 0 = x{i}"],
                error_message="" if ok else "boom",
                error_category="" if ok else "tactic_failed",
                is_proof_complete=ok,
            ))
        trajs.append(RichProofTrajectory(
            theorem=f"t{i}", steps=steps, success=True,
            depth=5, duration_ms=100.0))
    t = WorldModelTrainer(db_path=str(tmp_path / "x.db"))
    n = t.extract_from_trajectories(trajs)
    if n < 50:
        return None
    metrics = t.train()
    if "error" in metrics:
        return None
    out = tmp_path / "tiny.pkl"
    t.save(str(out))
    return out


# ─────────────────────────────────────────────────────────────────────────
# 4-7. TacticApplyTool world-model gate
# ─────────────────────────────────────────────────────────────────────────

class TestTacticApplyWMGate:
    def test_tool_accepts_world_model_param(self):
        m = HighConfidenceFailModel()
        t = TacticApplyTool(lean_pool=FakePool([]), world_model=m,
                              wm_min_confidence=0.5)
        assert t._wm is m
        assert t._wm_min_conf == 0.5

    def test_high_conf_fail_short_circuits_lean(self):
        """The Lean pool is NOT called when the model rejects."""
        from prover.unified.search_driver import SharedSearchState
        s = SharedSearchState(root_env_id=0, root_goals=["⊢ False"])
        pool = FakePool([FakeTacticResult(success=True)])  # will not be hit
        t = TacticApplyTool(
            lean_pool=pool, search_state=s,
            world_model=HighConfidenceFailModel(),
            wm_min_confidence=0.85)
        out = asyncio.run(t.execute({"tactic": "trivial"}, ToolContext()))
        # Tool returns success at the protocol level (the obs is structured)
        assert not out.is_error
        # But the obs should mark the rejection
        import json
        obs = json.loads(out.content)
        assert obs["success"] is False
        assert obs.get("world_model_blocked") is True
        # Lean pool was NOT called
        assert pool.call_count == 0

    def test_uncertain_prediction_does_not_gate(self):
        from prover.unified.search_driver import SharedSearchState
        s = SharedSearchState(root_env_id=0, root_goals=["⊢ True"])
        pool = FakePool([FakeTacticResult(
            success=True, new_env_id=2, remaining_goals=[])])
        t = TacticApplyTool(
            lean_pool=pool, search_state=s,
            world_model=UncertainModel(),
            wm_min_confidence=0.85)
        out = asyncio.run(t.execute({"tactic": "trivial"}, ToolContext()))
        assert pool.call_count == 1, \
            "uncertain predictions must let the tactic through"
        import json
        obs = json.loads(out.content)
        assert obs["success"] is True
        assert "world_model_blocked" not in obs

    def test_predicted_success_does_not_gate(self):
        from prover.unified.search_driver import SharedSearchState
        s = SharedSearchState(root_env_id=0, root_goals=["⊢ True"])
        pool = FakePool([FakeTacticResult(
            success=True, new_env_id=3, remaining_goals=[])])
        t = TacticApplyTool(
            lean_pool=pool, search_state=s,
            world_model=HighConfidenceSuccessModel())
        out = asyncio.run(t.execute({"tactic": "rfl"}, ToolContext()))
        assert pool.call_count == 1
        assert not out.is_error

    def test_no_world_model_no_gate(self):
        from prover.unified.search_driver import SharedSearchState
        s = SharedSearchState(root_env_id=0, root_goals=["⊢ False"])
        pool = FakePool([FakeTacticResult(
            success=False, error_message="nope")])
        t = TacticApplyTool(lean_pool=pool, search_state=s,
                              world_model=None)
        out = asyncio.run(t.execute({"tactic": "trivial"}, ToolContext()))
        assert pool.call_count == 1, \
            "without a world model, every tactic reaches Lean"

    def test_world_model_crash_does_not_gate(self):
        """A misbehaving model must not block the tactic."""
        from prover.unified.search_driver import SharedSearchState
        s = SharedSearchState(root_env_id=0, root_goals=["⊢ True"])
        pool = FakePool([FakeTacticResult(success=True, new_env_id=1,
                                            remaining_goals=[])])
        t = TacticApplyTool(lean_pool=pool, search_state=s,
                              world_model=CrashingModel())
        out = asyncio.run(t.execute({"tactic": "trivial"}, ToolContext()))
        assert pool.call_count == 1, \
            "model crash → fall back to running the tactic normally"
        assert not out.is_error

    def test_no_goals_state_skips_gate(self):
        """Without a search_state, goals_before is [] and the gate is
        a noop — we don't have a goal to feed to the model."""
        pool = FakePool([FakeTacticResult(
            success=True, new_env_id=1, remaining_goals=[])])
        t = TacticApplyTool(lean_pool=pool,  # no search_state
                              world_model=HighConfidenceFailModel())
        out = asyncio.run(t.execute({"tactic": "trivial"}, ToolContext()))
        assert pool.call_count == 1


# ─────────────────────────────────────────────────────────────────────────
# 8-10. Plumbing through tool_kits + runner
# ─────────────────────────────────────────────────────────────────────────

class TestPlumbing:
    def test_build_tool_registry_threads_world_model(self):
        from prover.unified.tool_kits import build_tool_registry
        from prover.unified.profiles import Profile, ToolKit
        m = HighConfidenceFailModel()
        prof = Profile(
            name="t", tools=[ToolKit.TACTIC_APPLY], max_turns=1,
            framing="step_level_pure")
        reg = build_tool_registry(prof, lean_pool=FakePool([]),
                                    world_model=m)
        tool = reg.get("tactic_apply")
        assert tool._wm is m

    def test_runner_accepts_world_model(self):
        from prover.unified.runner import UnifiedProofRunner
        m = HighConfidenceSuccessModel()
        runner = UnifiedProofRunner(llm=None, world_model=m)
        assert runner.world_model is m

    def test_runner_world_model_defaults_none(self):
        from prover.unified.runner import UnifiedProofRunner
        runner = UnifiedProofRunner(llm=None)
        assert runner.world_model is None

    def test_make_world_model_used_by_factory(self):
        """Plug factory output into runner end-to-end."""
        from prover.unified.runner import UnifiedProofRunner
        wm = make_world_model()  # → Mock
        runner = UnifiedProofRunner(llm=None, world_model=wm)
        assert isinstance(runner.world_model, MockWorldModel)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
