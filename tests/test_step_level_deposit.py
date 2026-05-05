"""tests/test_step_level_deposit.py — step-level knowledge deposit

After v4, every successful or failed tactic application by step-level
profiles (reprover / leandojo / mcts / best_first / beam) is auto-
deposited into the knowledge writer. This closes the gap noted in
``REFACTOR_REPORT.md §九.2`` ("Step-level 知识落库").

This file pins:

  1. ``TacticApplyTool`` accepts a ``knowledge_writer`` param
  2. Every successful tactic call → exactly one writer.ingest_step call
  3. Every failed tactic call → exactly one writer.ingest_step call
  4. REPL-level exceptions (try_tactic raises) → still exactly one deposit
  5. The deposited StepDetail has the right shape (tactic, env_id,
     goals_before/after, error fields, elapsed_ms, is_proof_complete)
  6. Step indices are monotonic per tool instance
  7. Writer errors are SWALLOWED — proof loop must never crash on them
  8. ``build_tool_registry`` plumbs the writer through
  9. ``build_tool_registry`` also picks up ``knowledge_store.writer``
     when no explicit writer is passed
 10. Linear profiles without TACTIC_APPLY are unaffected (no writer
     dependency, no overhead)
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

from agent.tools.base import ToolContext
from agent.tools.builtin.tactic_apply import TacticApplyTool
from prover.unified.tool_kits import build_tool_registry
from prover.unified.profiles import Profile, ToolKit
from prover.unified.search_driver import SharedSearchState
from engine.proof_context_store import StepDetail

# ─────────────────────────────────────────────────────────────────────────
# Fakes — minimal, focused on the deposit contract
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class FakeTacticResult:
    """Mimics what async_lean_pool.try_tactic returns."""
    success: bool
    is_proof_complete: bool = False
    new_env_id: int = -1
    remaining_goals: list = field(default_factory=list)
    error_message: str = ""
    error_category: str = ""

class FakePool:
    """Minimal AsyncLeanPool stand-in. Plays back a queue of results."""

    base_env_id = 0

    def __init__(self, results: list, raise_on_call: int = -1):
        self._results = list(results)
        self._raise_on = raise_on_call
        self._calls = 0
        self.last_inputs: list = []

    async def try_tactic(self, env_id: int, tactic: str):
        self.last_inputs.append((env_id, tactic))
        if self._calls == self._raise_on:
            self._calls += 1
            raise RuntimeError("simulated REPL crash")
        r = self._results[self._calls % len(self._results)]
        self._calls += 1
        return r

class RecordingWriter:
    """Captures every ingest_step call. Async, like the real writer."""

    def __init__(self, *, raise_on_call: int = -1):
        self.calls: list[tuple[StepDetail, str]] = []
        self._raise_on = raise_on_call
        self._n = 0

    async def ingest_step(self, step: StepDetail, theorem: str = "",
                           domain: str = "", trace_id: int = 0):
        if self._n == self._raise_on:
            self._n += 1
            raise RuntimeError("writer exploded")
        self._n += 1
        self.calls.append((step, theorem))

# ─────────────────────────────────────────────────────────────────────────
# 1-2. Plumbing + happy path
# ─────────────────────────────────────────────────────────────────────────

class TestStepDepositPlumbing:
    def test_tool_accepts_writer_param(self):
        w = RecordingWriter()
        t = TacticApplyTool(lean_pool=FakePool([]), knowledge_writer=w)
        assert t._kw is w

    def test_tool_works_without_writer(self):
        # Back-compat: omitting knowledge_writer is allowed.
        t = TacticApplyTool(lean_pool=FakePool([]))
        assert t._kw is None

    def test_successful_tactic_deposits_one_step(self):
        pool = FakePool([FakeTacticResult(
            success=True, new_env_id=5,
            remaining_goals=["⊢ True"], is_proof_complete=False)])
        w = RecordingWriter()
        t = TacticApplyTool(lean_pool=pool, knowledge_writer=w)
        ctx = ToolContext(theorem_statement="thm")
        out = asyncio.run(t.execute({"tactic": "intro h"}, ctx))
        assert not out.is_error
        assert len(w.calls) == 1
        step, theorem = w.calls[0]
        assert step.tactic == "intro h"
        assert step.env_id_after == 5
        assert step.goals_after == ["⊢ True"]
        assert step.error_message == ""
        assert theorem == "thm"

    def test_failed_tactic_still_deposits_one_step(self):
        pool = FakePool([FakeTacticResult(
            success=False, error_message="ring failed",
            error_category="tactic_failed")])
        w = RecordingWriter()
        t = TacticApplyTool(lean_pool=pool, knowledge_writer=w)
        out = asyncio.run(t.execute({"tactic": "ring"}, ToolContext()))
        assert not out.is_error  # tool returned, didn't raise
        assert len(w.calls) == 1
        step, _ = w.calls[0]
        assert step.tactic == "ring"
        assert step.env_id_after == -1
        assert step.error_message == "ring failed"
        assert step.error_category == "tactic_failed"

# ─────────────────────────────────────────────────────────────────────────
# 3-4. Edge cases
# ─────────────────────────────────────────────────────────────────────────

class TestStepDepositEdgeCases:
    def test_repl_exception_still_deposits(self):
        """REPL crash → tool returns error, but the failure step is
        still recorded so Layer 1 reflects the infrastructure problem."""
        pool = FakePool([FakeTacticResult(success=True)],
                         raise_on_call=0)
        w = RecordingWriter()
        t = TacticApplyTool(lean_pool=pool, knowledge_writer=w)
        out = asyncio.run(t.execute({"tactic": "trivial"}, ToolContext()))
        assert out.is_error
        assert len(w.calls) == 1
        step, _ = w.calls[0]
        assert step.error_category == "repl_error"
        assert step.env_id_after == -1

    def test_no_writer_no_overhead(self):
        """Without a writer, tool works exactly as before."""
        pool = FakePool([FakeTacticResult(success=True, new_env_id=2)])
        t = TacticApplyTool(lean_pool=pool, knowledge_writer=None)
        out = asyncio.run(t.execute({"tactic": "trivial"}, ToolContext()))
        assert not out.is_error
        # Sanity: tool still ran successfully without depositing.

    def test_writer_exception_does_not_break_loop(self):
        """A misbehaving writer must not bring down the proof loop."""
        pool = FakePool([FakeTacticResult(success=True, new_env_id=2,
                                           remaining_goals=[])])
        w = RecordingWriter(raise_on_call=0)  # writer dies on first call
        t = TacticApplyTool(lean_pool=pool, knowledge_writer=w)
        # Tool still returns success — the writer crash is swallowed.
        out = asyncio.run(t.execute({"tactic": "trivial"}, ToolContext()))
        assert not out.is_error

# ─────────────────────────────────────────────────────────────────────────
# 5. StepDetail shape
# ─────────────────────────────────────────────────────────────────────────

class TestStepDetailShape:
    def test_full_field_population_on_success(self):
        pool = FakePool([FakeTacticResult(
            success=True, new_env_id=7, is_proof_complete=True,
            remaining_goals=[])])
        w = RecordingWriter()
        t = TacticApplyTool(lean_pool=pool, knowledge_writer=w)
        ctx = ToolContext(theorem_statement="theorem t : True")
        asyncio.run(t.execute({"tactic": "trivial"}, ctx))
        step, _ = w.calls[0]
        assert step.tactic == "trivial"
        assert step.env_id_before == 0           # FakePool.base_env_id
        assert step.env_id_after == 7
        assert step.is_proof_complete is True
        assert step.goals_after == []
        assert step.elapsed_ms >= 0
        assert step.step_index == 0

    def test_goals_before_from_search_state(self):
        """When a search_state is wired in, goals_before is captured."""
        s = SharedSearchState(root_env_id=0, root_goals=["⊢ True"])
        pool = FakePool([FakeTacticResult(
            success=True, new_env_id=1, remaining_goals=[],
            is_proof_complete=True)])
        w = RecordingWriter()
        t = TacticApplyTool(lean_pool=pool, search_state=s,
                             knowledge_writer=w)
        asyncio.run(t.execute({"tactic": "trivial"}, ToolContext()))
        step, _ = w.calls[0]
        assert step.goals_before == ["⊢ True"]

# ─────────────────────────────────────────────────────────────────────────
# 6. Monotonic step_index
# ─────────────────────────────────────────────────────────────────────────

class TestMonotonicIndex:
    def test_step_index_increments_across_calls(self):
        pool = FakePool([FakeTacticResult(success=True, new_env_id=1),
                          FakeTacticResult(success=False,
                                            error_message="boom"),
                          FakeTacticResult(success=True, new_env_id=2)])
        w = RecordingWriter()
        t = TacticApplyTool(lean_pool=pool, knowledge_writer=w)

        async def run3():
            for tac in ("a", "b", "c"):
                await t.execute({"tactic": tac}, ToolContext())

        asyncio.run(run3())
        indices = [s.step_index for s, _ in w.calls]
        assert indices == [0, 1, 2]

# ─────────────────────────────────────────────────────────────────────────
# 7-9. Plumbing through tool_kits + runner-side defaults
# ─────────────────────────────────────────────────────────────────────────

class TestToolKitsPlumbing:
    def test_build_tool_registry_threads_writer(self):
        prof = Profile(
            name="test", tools=[ToolKit.TACTIC_APPLY], max_turns=1,
            framing="step_level_pure")
        w = RecordingWriter()
        reg = build_tool_registry(prof, lean_pool=FakePool([]),
                                    knowledge_writer=w)
        tool = reg.get("tactic_apply")
        assert isinstance(tool, TacticApplyTool)
        assert tool._kw is w

    def test_build_tool_registry_picks_writer_from_store(self):
        """Back-compat: knowledge_store.writer is used when the explicit
        kwarg is absent."""
        store = MagicMock()
        store.writer = RecordingWriter()
        prof = Profile(
            name="test2", tools=[ToolKit.TACTIC_APPLY], max_turns=1,
            framing="step_level_pure")
        reg = build_tool_registry(prof, lean_pool=FakePool([]),
                                    knowledge_store=store)
        tool = reg.get("tactic_apply")
        assert tool._kw is store.writer

    def test_build_tool_registry_no_writer_when_neither_set(self):
        prof = Profile(
            name="test3", tools=[ToolKit.TACTIC_APPLY], max_turns=1,
            framing="step_level_pure")
        reg = build_tool_registry(prof, lean_pool=FakePool([]))
        tool = reg.get("tactic_apply")
        assert tool._kw is None

    def test_runner_picks_writer_from_store_attribute(self):
        """UnifiedProofRunner.__init__ pulls .writer off knowledge_store
        when no explicit knowledge_writer is given."""
        from prover.unified.runner import UnifiedProofRunner
        store = MagicMock()
        store.writer = RecordingWriter()
        runner = UnifiedProofRunner(llm=None, knowledge_store=store)
        assert runner.knowledge_writer is store.writer

    def test_runner_explicit_writer_wins(self):
        """Explicit writer kwarg wins over knowledge_store.writer."""
        from prover.unified.runner import UnifiedProofRunner
        store = MagicMock()
        store.writer = RecordingWriter()
        explicit = RecordingWriter()
        runner = UnifiedProofRunner(llm=None, knowledge_store=store,
                                      knowledge_writer=explicit)
        assert runner.knowledge_writer is explicit

# ─────────────────────────────────────────────────────────────────────────
# 10. Linear profiles without TACTIC_APPLY are unaffected
# ─────────────────────────────────────────────────────────────────────────

class TestLinearProfilesUnaffected:
    def test_whole_proof_repair_no_writer_dependency(self):
        """The default profile uses lean_verify, not tactic_apply,
        and must not require a writer to be wired in."""
        from prover.unified.profiles import get_profile
        prof = get_profile("whole_proof_repair")
        # lean_verify is in tools, tactic_apply is not.
        assert ToolKit.TACTIC_APPLY not in prof.tools
        # build_tool_registry succeeds with no writer.
        reg = build_tool_registry(prof, lean_pool=FakePool([]))
        assert reg.get("lean_verify") is not None

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
