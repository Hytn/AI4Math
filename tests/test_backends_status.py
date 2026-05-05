"""

The four community backend slots (Kimina, Pantograph, LooKeng) all
expose ``is_fallback`` to indicate "I'm constructed, but the real
implementation isn't available so I'm running in degraded mode".
Before V6 this was only visible in debug logs, so a user who set up
``--backend pantograph`` and forgot to install pypantograph would see
the run complete with no warning that they were getting LocalTransport
behaviour all along.

V6 surfaces this as structured data in ``dialog.json``::

    meta:
      backends:
        kimina:     {present: true, is_fallback: true,  is_alive: false}
        pantograph: {present: true, is_fallback: false, mode: "pypantograph"}
        lookeng:    {present: false}
        lean_pool:  {present: true, kind: "AsyncLeanPool", size: 4}

Tests pin:
  1. The collector tolerates None backends.
  2. The collector reads is_fallback / mode / is_alive when present.
  3. The collector returns {} on hard failure (not raise).
  4. ``UnifiedResult.save_unified`` surfaces backends_status as
     ``meta.backends`` only when populated.
  5. Empty backends_status doesn't add a meta.backends key.
  6. The collector reports lean_pool kind as the class name.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.runtime.agent_loop import LoopResult, LoopMessage
from prover.unified.runner import UnifiedProofRunner, UnifiedResult

# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

def _runner(**kwargs) -> UnifiedProofRunner:
    """Build a runner without auto-registering the autoformalizer
    (to keep tests focused on backend status, not LLM side-effects)."""
    return UnifiedProofRunner(
        llm=MagicMock(),
        auto_register_llm_autoformalizer=False,
        **kwargs,
    )

def _make_kimina(*, is_fallback: bool, is_alive: bool = True):
    b = MagicMock()
    type(b).is_fallback = property(lambda self: is_fallback)
    type(b).is_alive = property(lambda self: is_alive)
    return b

def _make_pantograph(*, is_fallback: bool, mode: str = "pypantograph"):
    b = MagicMock()
    type(b).is_fallback = property(lambda self: is_fallback)
    type(b).mode = mode
    return b

def _make_lookeng(*, is_fallback: bool):
    b = MagicMock()
    type(b).is_fallback = property(lambda self: is_fallback)
    return b

# ─────────────────────────────────────────────────────────────────────
# Collector: every slot empty
# ─────────────────────────────────────────────────────────────────────

class TestCollectorEmpty:

    def test_no_backends_present(self):
        runner = _runner()
        status = runner._collect_backend_status()
        # All four slots reported, all "present": False
        assert set(status.keys()) == {
            "kimina", "pantograph", "lookeng", "lean_pool"}
        for slot in ("kimina", "pantograph", "lookeng", "lean_pool"):
            assert status[slot]["present"] is False

    def test_no_backends_no_is_fallback_field(self):
        # When backend is absent, we don't synthesize is_fallback —
        # the contract is "present: false, that's all there is to say".
        runner = _runner()
        status = runner._collect_backend_status()
        assert "is_fallback" not in status["kimina"]

# ─────────────────────────────────────────────────────────────────────
# Collector: each slot present
# ─────────────────────────────────────────────────────────────────────

class TestCollectorKimina:

    def test_kimina_real_backend(self):
        runner = _runner(kimina_backend=_make_kimina(
            is_fallback=False, is_alive=True))
        status = runner._collect_backend_status()
        assert status["kimina"]["present"] is True
        assert status["kimina"]["is_fallback"] is False
        assert status["kimina"]["is_alive"] is True

    def test_kimina_fallback_backend(self):
        runner = _runner(kimina_backend=_make_kimina(
            is_fallback=True, is_alive=False))
        status = runner._collect_backend_status()
        assert status["kimina"]["present"] is True
        assert status["kimina"]["is_fallback"] is True
        assert status["kimina"]["is_alive"] is False

class TestCollectorPantograph:

    def test_pantograph_pypantograph_mode(self):
        runner = _runner(pantograph_backend=_make_pantograph(
            is_fallback=False, mode="pypantograph"))
        status = runner._collect_backend_status()
        assert status["pantograph"]["present"] is True
        assert status["pantograph"]["is_fallback"] is False
        assert status["pantograph"]["mode"] == "pypantograph"

    def test_pantograph_binary_mode(self):
        runner = _runner(pantograph_backend=_make_pantograph(
            is_fallback=False, mode="binary"))
        status = runner._collect_backend_status()
        assert status["pantograph"]["mode"] == "binary"

    def test_pantograph_fallback_mode(self):
        runner = _runner(pantograph_backend=_make_pantograph(
            is_fallback=True, mode="fallback"))
        status = runner._collect_backend_status()
        assert status["pantograph"]["is_fallback"] is True
        assert status["pantograph"]["mode"] == "fallback"

class TestCollectorLooKeng:

    def test_lookeng_real_backend(self):
        runner = _runner(lookeng_backend=_make_lookeng(is_fallback=False))
        status = runner._collect_backend_status()
        assert status["lookeng"]["present"] is True
        assert status["lookeng"]["is_fallback"] is False

    def test_lookeng_fallback(self):
        runner = _runner(lookeng_backend=_make_lookeng(is_fallback=True))
        status = runner._collect_backend_status()
        assert status["lookeng"]["is_fallback"] is True

class TestCollectorLeanPool:

    def test_lean_pool_records_kind_as_class_name(self):
        class FakePool:
            pass
        pool = FakePool()
        runner = _runner(lean_pool=pool)
        status = runner._collect_backend_status()
        assert status["lean_pool"]["present"] is True
        assert status["lean_pool"]["kind"] == "FakePool"

    def test_lean_pool_records_size_when_present(self):
        class FakePool:
            size = 8
        runner = _runner(lean_pool=FakePool())
        status = runner._collect_backend_status()
        assert status["lean_pool"]["size"] == 8

    def test_lean_pool_records_alive_when_present(self):
        class FakePool:
            is_alive = False
        runner = _runner(lean_pool=FakePool())
        status = runner._collect_backend_status()
        assert status["lean_pool"]["is_alive"] is False

    def test_lean_pool_no_size_no_alive_just_kind(self):
        class FakePool:
            pass
        runner = _runner(lean_pool=FakePool())
        status = runner._collect_backend_status()
        # Only kind + present, no synthesized fields
        assert "size" not in status["lean_pool"]
        assert "is_alive" not in status["lean_pool"]

# ─────────────────────────────────────────────────────────────────────
# Collector: robustness
# ─────────────────────────────────────────────────────────────────────

class TestCollectorRobustness:

    def test_backend_with_missing_is_fallback_attribute(self):
        # Custom backend that doesn't expose is_fallback at all.
        # Should yield is_fallback=None (the "we don't know" code).
        custom = MagicMock(spec=[])  # no attrs
        runner = _runner(kimina_backend=custom)
        status = runner._collect_backend_status()
        assert status["kimina"]["present"] is True
        assert status["kimina"]["is_fallback"] is None

    def test_backend_property_raising_does_not_crash(self):
        # A backend whose is_fallback property raises must not crash
        # the collector — it's instrumentation, not gating.
        backend = MagicMock()
        # MagicMock attribute access auto-creates; replace with a
        # PropertyMock that raises.
        type(backend).is_fallback = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("oops")))
        runner = _runner(pantograph_backend=backend)
        status = runner._collect_backend_status()
        # collector swallows; entry still exists with is_fallback=None
        assert status["pantograph"]["present"] is True
        assert status["pantograph"]["is_fallback"] is None

    def test_collector_never_raises(self):
        # Even if every slot has a hostile backend, the collector
        # returns a dict (possibly empty) and never raises.
        hostile = MagicMock()
        type(hostile).is_fallback = property(
            lambda self: (_ for _ in ()).throw(Exception("nope")))
        runner = _runner(
            kimina_backend=hostile,
            pantograph_backend=hostile,
            lookeng_backend=hostile,
        )
        status = runner._collect_backend_status()
        # Returns something — empty dict on total fail, populated dict
        # on partial success.
        assert isinstance(status, dict)

# ─────────────────────────────────────────────────────────────────────
# Surface in dialog.json (via UnifiedResult.save_unified)
# ─────────────────────────────────────────────────────────────────────

def _make_dummy_loop_result() -> LoopResult:
    """Minimal LoopResult so save_unified can write a file."""
    return LoopResult(
        content="done",
        proof_code="by simp",
        messages=[
            LoopMessage(role="user", content="prove this"),
            LoopMessage(role="assistant", content="```lean\nby simp\n```"),
        ],
        turns_used=1,
        stopped_reason="proof_found",
    )

class TestSaveUnifiedSurfacesBackends:

    def test_meta_backends_written_when_status_populated(self, tmp_path: Path):
        ur = UnifiedResult(
            profile_name="whole_proof",
            success=True,
            proof_code="by simp",
            loop_result=_make_dummy_loop_result(),
            backends_status={
                "kimina": {"present": True, "is_fallback": True},
                "lean_pool": {"present": True, "kind": "AsyncLeanPool"},
            },
        )
        out_dir = tmp_path / "trace"
        ur.save_unified(str(out_dir), problem_id="t1")

        dialog_path = out_dir / "dialog.json"
        assert dialog_path.exists()
        d = json.loads(dialog_path.read_text(encoding="utf-8"))
        assert "meta" in d
        assert "backends" in d["meta"]
        assert d["meta"]["backends"]["kimina"]["is_fallback"] is True

    def test_meta_backends_omitted_when_empty(self, tmp_path: Path):
        ur = UnifiedResult(
            profile_name="whole_proof",
            success=True,
            proof_code="by simp",
            loop_result=_make_dummy_loop_result(),
            backends_status={},  # empty → not surfaced
        )
        out_dir = tmp_path / "trace2"
        ur.save_unified(str(out_dir), problem_id="t2")
        d = json.loads((out_dir / "dialog.json").read_text(encoding="utf-8"))
        assert "backends" not in d.get("meta", {})

    def test_search_tree_and_backends_coexist(self, tmp_path: Path):
        # Both meta.search_tree and meta.backends can appear
        # together in the same dialog.
        ur = UnifiedResult(
            profile_name="mcts",
            success=True,
            proof_code="by simp",
            loop_result=_make_dummy_loop_result(),
            search_tree={"kind": "ucb", "nodes": []},
            backends_status={
                "lean_pool": {"present": True, "kind": "Pool"},
            },
        )
        out_dir = tmp_path / "trace3"
        ur.save_unified(str(out_dir), problem_id="t3")
        d = json.loads((out_dir / "dialog.json").read_text(encoding="utf-8"))
        assert "search_tree" in d["meta"]
        assert "backends" in d["meta"]

# ─────────────────────────────────────────────────────────────────────
# Default UnifiedResult construction has empty backends_status
# ─────────────────────────────────────────────────────────────────────

class TestDefaultBackendsStatus:

    def test_default_field_is_empty_dict(self):
        ur = UnifiedResult(profile_name="any", success=False)
        assert ur.backends_status == {}

    def test_legacy_construction_does_not_break(self, tmp_path: Path):
        # Old code that constructs UnifiedResult without
        # backends_status (positional or keyword) should still save
        # fine — V6 added the field as additive.
        ur = UnifiedResult(
            profile_name="whole_proof",
            success=True,
            proof_code="by simp",
            loop_result=_make_dummy_loop_result(),
        )
        out_dir = tmp_path / "legacy"
        ur.save_unified(str(out_dir), problem_id="legacy")
        d = json.loads((out_dir / "dialog.json").read_text(encoding="utf-8"))
        # No backends key — backwards-compatible behaviour
        assert "backends" not in d.get("meta", {})
