"""tests/test_backends/test_infra_tools.py — Infrastructure tools tests.

Covers the five new ``ToolKit`` tools defined in
``prover.unified.tools_infra`` and the ``backend_factory`` entry point.
"""
import json
import pytest

from agent.tools.base import ToolContext, ToolPermission
from prover.unified.tools_infra import (
    BatchVerifyTool, MVarFocusTool, DraftHoleTool,
    LemmaByLemmaTool, NLExistenceBridgeTool,
)
from engine.backend_factory import build_backend, SUPPORTED_BACKENDS


def _ctx():
    return ToolContext(
        agent_name="test",
        theorem_statement="theorem t : True",
        allowed_permissions={
            ToolPermission.READ_ONLY,
            ToolPermission.WRITE_LOCAL,
            ToolPermission.EXTERNAL,
            ToolPermission.DANGEROUS,
        },
    )


# ─── BatchVerifyTool ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_batch_verify_with_no_backend_returns_error():
    tool = BatchVerifyTool(kimina_backend=None)
    r = await tool.execute({"proofs": ["theorem t : True := trivial"]}, _ctx())
    assert r.is_error
    assert "not configured" in r.content.lower()


@pytest.mark.asyncio
async def test_batch_verify_empty_proofs_rejected():
    tool = BatchVerifyTool(kimina_backend=None)
    r = await tool.execute({"proofs": []}, _ctx())
    assert r.is_error
    assert "non-empty" in r.content.lower()


@pytest.mark.asyncio
async def test_batch_verify_in_fallback_returns_error():
    """Backend in fallback mode → tool returns structured error."""
    class _FakeBackend:
        is_fallback = True
    tool = BatchVerifyTool(kimina_backend=_FakeBackend())
    r = await tool.execute({"proofs": ["theorem t : True := trivial"]}, _ctx())
    assert r.is_error
    assert "fallback" in r.content.lower()


@pytest.mark.asyncio
async def test_batch_verify_calls_backend_and_aggregates():
    """Happy path: returns JSON with per-proof + aggregate stats."""
    from engine.backends.kimina_server import BatchVerifyResult, TacticTrace

    class _FakeBackend:
        is_fallback = False
        async def verify_batch(self, proofs, preamble=None):
            return [
                BatchVerifyResult(id=f"b-{i}", success=(i % 2 == 0),
                                   error_messages=[] if i % 2 == 0 else ["err"],
                                   elapsed_ms=10 * (i + 1),
                                   tactic_trace=[
                                       TacticTrace(tactic="trivial",
                                                    goal_before="True",
                                                    goals_after=[],
                                                    is_proof_complete=True),
                                   ])
                for i in range(len(proofs))
            ]

    tool = BatchVerifyTool(kimina_backend=_FakeBackend())
    proofs = ["theorem a := trivial", "theorem b := wrong",
              "theorem c := trivial"]
    r = await tool.execute({"proofs": proofs}, _ctx())
    assert not r.is_error
    payload = json.loads(r.content)
    assert payload["batch_size"] == 3
    assert payload["n_succeeded"] == 2
    assert len(payload["results"]) == 3
    # Tactic trace count makes it through
    assert payload["results"][0]["n_tactics_extracted"] == 1


# ─── MVarFocusTool ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_mvar_focus_no_backend_errors():
    tool = MVarFocusTool(pantograph_backend=None)
    r = await tool.execute(
        {"mvar_id": "?m1", "proof_state": 1}, _ctx())
    assert r.is_error
    assert "not available" in r.content.lower()


@pytest.mark.asyncio
async def test_mvar_focus_happy_path():
    """Simulate Pantograph happily focusing the requested mvar."""
    from engine.backends.pantograph import MVarFocusResult, GoalFragment

    class _FakeP:
        is_fallback = False
        async def focus_mvar(self, proof_state, mvar_id):
            return MVarFocusResult(
                success=True, new_proof_state=99,
                focused=GoalFragment(
                    goal="⊢ Q", mvar_id=mvar_id,
                    coupled_with=["?m2"]),
                remaining=[GoalFragment(goal="⊢ R", mvar_id="?m2")])

    tool = MVarFocusTool(pantograph_backend=_FakeP())
    r = await tool.execute(
        {"mvar_id": "?m1", "proof_state": 0}, _ctx())
    assert not r.is_error
    payload = json.loads(r.content)
    assert payload["new_proof_state"] == 99
    assert payload["focused_mvar"] == "?m1"
    assert payload["coupled_with"] == ["?m2"]
    assert payload["n_remaining_goals"] == 1


# ─── DraftHoleTool ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_draft_hole_no_backend_errors():
    tool = DraftHoleTool(pantograph_backend=None)
    r = await tool.execute(
        {"statement": "P → Q", "proof_state": 0}, _ctx())
    assert r.is_error


@pytest.mark.asyncio
async def test_draft_hole_happy_path():
    from engine.backends.pantograph import DraftResult, GoalFragment

    class _FakeP:
        is_fallback = False
        async def insert_draft(self, proof_state, statement):
            return DraftResult(
                success=True,
                holes=[GoalFragment(goal=statement, is_meta=True,
                                     mvar_id="_draft_42")],
                proof_state=42)

    tool = DraftHoleTool(pantograph_backend=_FakeP())
    r = await tool.execute(
        {"statement": "∀ n, P n", "proof_state": 0}, _ctx())
    assert not r.is_error
    payload = json.loads(r.content)
    assert payload["hole_proof_state"] == 42
    assert payload["hole_goal"] == "∀ n, P n"


# ─── LemmaByLemmaTool ────────────────────────────────────────


@pytest.mark.asyncio
async def test_lemma_by_lemma_no_backend_errors():
    tool = LemmaByLemmaTool(lookeng_backend=None)
    r = await tool.execute({
        "session_id": "x", "name": "step1",
        "proof": "rfl",
    }, _ctx())
    assert r.is_error


@pytest.mark.asyncio
async def test_lemma_by_lemma_passes_through():
    """Tool should forward the input verbatim to the backend."""
    captured = {}
    class _FakeLK:
        async def submit_lemma(self, **kwargs):
            captured.update(kwargs)
            return {"ok": True, "running_context_size": 1}

    tool = LemmaByLemmaTool(lookeng_backend=_FakeLK())
    r = await tool.execute({
        "session_id": "s1",
        "name": "step1",
        "statement": "theorem step1 : True",
        "proof": "trivial",
        "is_final": False,
    }, _ctx())
    assert not r.is_error
    assert captured["session_id"] == "s1"
    assert captured["name"] == "step1"
    assert captured["proof"] == "trivial"
    assert captured["is_final"] is False


# ─── NLExistenceBridgeTool ───────────────────────────────────


@pytest.mark.asyncio
async def test_nl_existence_returns_lean_skeleton():
    tool = NLExistenceBridgeTool()
    r = await tool.execute({
        "nl_problem": "Find all integers n such that n^2 < 10",
        "answer_type": "integer",
    }, _ctx())
    assert not r.is_error
    payload = json.loads(r.content)
    assert "lean_statement" in payload
    assert "ℤ" in payload["lean_statement"]
    assert "ai4math_q" in payload["lean_statement"]
    assert "n^2 < 10" in payload["informal_summary"]


@pytest.mark.asyncio
async def test_nl_existence_handles_unknown_type():
    tool = NLExistenceBridgeTool()
    r = await tool.execute({
        "nl_problem": "Compute the value of foo",
    }, _ctx())
    assert not r.is_error
    payload = json.loads(r.content)
    # Default fallback type is ℕ
    assert "ℕ" in payload["lean_statement"]


@pytest.mark.asyncio
async def test_nl_existence_strips_latex():
    tool = NLExistenceBridgeTool()
    r = await tool.execute({
        "nl_problem": r"Find $\frac{1}{2}$ such that...",
    }, _ctx())
    payload = json.loads(r.content)
    # `$` and `\` stripped from the comment line
    assert "$" not in payload["lean_statement"]


# ─── backend_factory ─────────────────────────────────────────


def test_supported_backends_complete():
    """All advertised backends should be in SUPPORTED_BACKENDS."""
    expected = {"local", "socket", "http", "kimina", "pantograph",
                "lookeng", "mock", "fallback", "auto"}
    assert expected == set(SUPPORTED_BACKENDS)


@pytest.mark.asyncio
async def test_build_backend_mock():
    t = await build_backend("mock")
    assert t.is_alive
    assert not t.is_fallback


@pytest.mark.asyncio
async def test_build_backend_fallback():
    t = await build_backend("fallback")
    assert t.is_alive
    assert t.is_fallback
    # Fallback returns None for everything
    out = await t.send({"cmd": "anything"})
    assert out is None


@pytest.mark.asyncio
async def test_build_backend_unknown_kind_falls_through_to_auto(caplog):
    """Unknown kind shouldn't raise — should fall back to auto."""
    t = await build_backend("frobnicator")
    # auto in CI without Lean → fallback
    assert t.is_alive
