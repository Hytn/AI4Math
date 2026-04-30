"""tests/test_backends/test_kimina_knowledge_deposit.py — Kimina→knowledge integration."""
import pytest
from unittest.mock import AsyncMock

from engine.backends.kimina_server import TacticTrace
from knowledge.writer import KnowledgeWriter


class _FakeStore:
    """Minimal store stub that records every upsert it gets."""

    def __init__(self):
        self.tactic_calls = []
        self.error_calls = []

    async def upsert_tactic_effectiveness(self, **kwargs):
        self.tactic_calls.append(kwargs)

    async def upsert_error_pattern(self, **kwargs):
        self.error_calls.append(kwargs)


@pytest.mark.asyncio
async def test_deposit_kimina_trace_empty_returns_zero():
    w = KnowledgeWriter(store=_FakeStore())
    n = await w.deposit_kimina_trace([])
    assert n == 0


@pytest.mark.asyncio
async def test_deposit_kimina_trace_with_dataclass_entries():
    """Pass real ``TacticTrace`` dataclass objects."""
    store = _FakeStore()
    w = KnowledgeWriter(store=store)
    trace = [
        TacticTrace(tactic="intro h", goal_before="P → Q",
                     goals_after=["Q"], is_proof_complete=False),
        TacticTrace(tactic="exact h.elim", goal_before="Q",
                     goals_after=[], is_proof_complete=True),
    ]
    n = await w.deposit_kimina_trace(
        trace, theorem="theorem t : P → Q := by ...")
    assert n == 2
    # Both tactics should have triggered an effectiveness upsert
    assert len(store.tactic_calls) == 2
    # All calls should be marked success (Kimina trace = successful proof)
    assert all(c["success"] is True for c in store.tactic_calls)
    # No error entries — successful proof has no errors
    assert store.error_calls == []
    # Tactic strings made it through verbatim
    tactics = [c["tactic"] for c in store.tactic_calls]
    assert tactics == ["intro h", "exact h.elim"]


@pytest.mark.asyncio
async def test_deposit_kimina_trace_with_dict_entries():
    """Pass plain dicts (the wire form)."""
    store = _FakeStore()
    w = KnowledgeWriter(store=store)
    trace = [
        {"tactic": "rfl", "goal_before": "1 = 1", "goals_after": [],
         "is_proof_complete": True},
    ]
    n = await w.deposit_kimina_trace(trace)
    assert n == 1
    assert store.tactic_calls[0]["tactic"] == "rfl"


@pytest.mark.asyncio
async def test_deposit_kimina_trace_skips_empty_tactic():
    store = _FakeStore()
    w = KnowledgeWriter(store=store)
    trace = [
        TacticTrace(tactic="", goal_before="P"),       # skip
        TacticTrace(tactic="trivial", goal_before=""),  # skip (no goal)
        TacticTrace(tactic="rfl", goal_before="x = x"), # ingest
    ]
    n = await w.deposit_kimina_trace(trace)
    assert n == 1
    assert len(store.tactic_calls) == 1


@pytest.mark.asyncio
async def test_deposit_kimina_trace_never_raises():
    """A misbehaving store must not crash deposit — it logs and skips."""
    class _BadStore:
        async def upsert_tactic_effectiveness(self, **kwargs):
            raise RuntimeError("simulated DB failure")
        async def upsert_error_pattern(self, **kwargs):
            pass
    w = KnowledgeWriter(store=_BadStore())
    trace = [TacticTrace(tactic="rfl", goal_before="x = x")]
    n = await w.deposit_kimina_trace(trace)
    # Each entry's exception is swallowed; method returns 0
    assert n == 0


@pytest.mark.asyncio
async def test_deposit_kimina_trace_passes_domain_and_trace_id():
    store = _FakeStore()
    w = KnowledgeWriter(store=store)
    trace = [TacticTrace(tactic="omega", goal_before="n + 1 > n")]
    await w.deposit_kimina_trace(
        trace, theorem="theorem t : n + 1 > n",
        domain="number_theory", trace_id=42)
    call = store.tactic_calls[0]
    assert call["domain"] == "number_theory"
    assert call["trace_id"] == 42
