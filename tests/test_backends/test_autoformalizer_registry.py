"""tests/test_backends/test_autoformalizer_registry.py — autoformalizer plugin."""
import json
import pytest

from agent.tools.base import ToolContext, ToolPermission
from prover.unified.tools_infra import (
    NLExistenceBridgeTool,
    register_autoformalizer,
    _get_autoformalizer,
)

def _ctx():
    return ToolContext(
        agent_name="test", theorem_statement="",
        allowed_permissions={ToolPermission.READ_ONLY,
                              ToolPermission.WRITE_LOCAL,
                              ToolPermission.EXTERNAL,
                              ToolPermission.DANGEROUS})

@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the registry between tests so they don't bleed."""
    register_autoformalizer(None)
    yield
    register_autoformalizer(None)

@pytest.mark.asyncio
async def test_default_uses_heuristic_when_unregistered():
    tool = NLExistenceBridgeTool()
    r = await tool.execute({
        "nl_problem": "Compute x.", "answer_type": "integer",
    }, _ctx())
    payload = json.loads(r.content)
    assert payload["autoformalizer"] == "heuristic"
    assert "ai4math_q" in payload["lean_statement"]

@pytest.mark.asyncio
async def test_register_sync_autoformalizer_is_used():
    def my_translator(nl: str, ty: str) -> str:
        return f"theorem real : ∃ ans : {ty}, ans = ans -- from translator"

    register_autoformalizer(my_translator)
    tool = NLExistenceBridgeTool()
    r = await tool.execute({
        "nl_problem": "x", "answer_type": "ℝ",
    }, _ctx())
    payload = json.loads(r.content)
    assert payload["autoformalizer"] == "registered"
    assert "from translator" in payload["lean_statement"]

@pytest.mark.asyncio
async def test_register_async_autoformalizer_is_awaited():
    async def my_async_translator(nl: str, ty: str) -> str:
        return f"theorem async_t : Async {ty}"

    register_autoformalizer(my_async_translator)
    tool = NLExistenceBridgeTool()
    r = await tool.execute({
        "nl_problem": "x", "answer_type": "Set ℕ",
    }, _ctx())
    payload = json.loads(r.content)
    assert payload["autoformalizer"] == "registered"
    assert "Async" in payload["lean_statement"]

@pytest.mark.asyncio
async def test_failing_autoformalizer_falls_back_to_heuristic():
    def broken(nl: str, ty: str) -> str:
        raise ValueError("simulated failure")

    register_autoformalizer(broken)
    tool = NLExistenceBridgeTool()
    r = await tool.execute({
        "nl_problem": "x", "answer_type": "integer",
    }, _ctx())
    payload = json.loads(r.content)
    # Falls back to the heuristic skeleton
    assert payload["autoformalizer"] == "heuristic"
    assert "ai4math_q" in payload["lean_statement"]

@pytest.mark.asyncio
async def test_empty_string_from_autoformalizer_falls_back():
    def empty(nl: str, ty: str) -> str:
        return ""

    register_autoformalizer(empty)
    tool = NLExistenceBridgeTool()
    r = await tool.execute({
        "nl_problem": "x", "answer_type": "integer",
    }, _ctx())
    payload = json.loads(r.content)
    assert payload["autoformalizer"] == "heuristic"

def test_register_none_clears_registry():
    def fn(nl, ty): return "anything"
    register_autoformalizer(fn)
    assert _get_autoformalizer() is fn
    register_autoformalizer(None)
    assert _get_autoformalizer() is None
