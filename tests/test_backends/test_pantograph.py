"""tests/test_backends/test_pantograph.py — Pantograph backend tests.

We test the protocol-translation surface and the mvar/draft helpers.
``pypantograph`` and the ``pantograph`` binary are external; we exercise
the fallback path that's exercised in CI environments without them.
"""
import pytest

from engine.backends.pantograph import (
    PantographBackend, GoalFragment, MVarFocusResult, DraftResult,
    extract_proof_term,
)

# ─── GoalFragment ─────────────────────────────────────────────

def test_goal_fragment_from_wire_full():
    d = {
        "goal": "⊢ P → Q",
        "mvar_id": "?m1",
        "coupled_with": ["?m2", "?m3"],
        "hypotheses": [
            {"name": "h", "type": "P"},
            {"name": "g", "type": "Q → R"},
        ],
        "is_meta": True,
    }
    f = GoalFragment.from_wire(d)
    assert f.goal == "⊢ P → Q"
    assert f.mvar_id == "?m1"
    assert f.coupled_with == ["?m2", "?m3"]
    assert f.hypotheses == [("h", "P"), ("g", "Q → R")]
    assert f.is_meta is True

def test_goal_fragment_accepts_tuple_hypotheses():
    """Some pypantograph versions emit hypotheses as tuples."""
    d = {"goal": "G", "hypotheses": [("h", "P"), ("g", "Q")]}
    f = GoalFragment.from_wire(d)
    assert f.hypotheses == [("h", "P"), ("g", "Q")]

def test_goal_fragment_target_aliases():
    """Some servers emit `type` instead of `goal`."""
    f = GoalFragment.from_wire({"type": "x = x"})
    assert f.goal == "x = x"

def test_goal_fragment_defaults():
    f = GoalFragment(goal="anything")
    assert f.mvar_id == ""
    assert f.coupled_with == []
    assert f.hypotheses == []
    assert f.is_meta is False

# ─── PantographBackend ────────────────────────────────────────

@pytest.mark.asyncio
async def test_backend_falls_back_when_neither_lib_nor_binary_available(
        monkeypatch):
    """No pypantograph + no `pantograph` on PATH → fallback mode."""
    # Force pypantograph init to fail
    pb = PantographBackend()
    monkeypatch.setattr(pb, "_try_pybind_init", lambda: False)
    # Force `which` to find nothing
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    started = await pb.start()
    assert started is True
    assert pb.mode == PantographBackend.MODE_FALLBACK
    assert pb.is_fallback is True

@pytest.mark.asyncio
async def test_backend_send_in_fallback_returns_none(monkeypatch):
    pb = PantographBackend()
    monkeypatch.setattr(pb, "_try_pybind_init", lambda: False)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    await pb.start()
    resp = await pb.send({"cmd": "theorem t : True := trivial"})
    assert resp is None

@pytest.mark.asyncio
async def test_focus_mvar_unavailable_returns_error(monkeypatch):
    pb = PantographBackend()
    monkeypatch.setattr(pb, "_try_pybind_init", lambda: False)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    await pb.start()
    r = await pb.focus_mvar(proof_state=0, mvar_id="?m1")
    assert r.success is False
    assert "unavailable" in r.error.lower() or "not" in r.error.lower()

@pytest.mark.asyncio
async def test_insert_draft_unavailable_returns_error(monkeypatch):
    pb = PantographBackend()
    monkeypatch.setattr(pb, "_try_pybind_init", lambda: False)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    await pb.start()
    r = await pb.insert_draft(proof_state=0, statement="∀ n, P n")
    assert r.success is False

def test_get_stats_includes_mode():
    pb = PantographBackend()
    s = pb.get_stats()
    assert "mode" in s
    assert "mvar_coupling" in s
    assert "drafting" in s

# ─── extract_proof_term helper ────────────────────────────────

def test_extract_proof_term_returns_none_in_fallback():
    pb = PantographBackend()
    # mode is fallback by default until start()
    pb.mode = PantographBackend.MODE_FALLBACK
    out = extract_proof_term(pb, proof_state=42)
    assert out is None

def test_extract_proof_term_open_proof_returns_none(monkeypatch):
    """If goals remain in the state, the proof isn't closed."""
    pb = PantographBackend()
    pb.mode = PantographBackend.MODE_PYBIND
    pb._proof_state_goals = {7: [GoalFragment(goal="⊢ P")]}
    out = extract_proof_term(pb, proof_state=7)
    assert out is None
