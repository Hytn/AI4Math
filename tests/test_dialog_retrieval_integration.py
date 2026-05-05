"""tests/test_dialog_retrieval_integration.py — End-to-end V5 wiring.

These tests pin the contracts that connect :mod:`knowledge.dialog_index`
into the rest of the system:

  * ``KnowledgeReader.attach_dialog_index`` / ``find_similar_dialogs``
    / ``render_similar_dialogs``
  * ``ObservationPolicy.inject_similar_dialogs`` flag is honoured by
    ``UnifiedProofRunner._build_initial_message``
  * The flag defaults to *off* — pre-V5 behaviour is byte-for-byte
    preserved when no DialogIndex is supplied.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from knowledge.dialog_index import DialogIndex
from knowledge.reader import KnowledgeReader
from prover.unified.profiles import ObservationPolicy, get_profile
from prover.unified.runner import UnifiedProofRunner

# ─────────────────────────────────────────────────────────────────────
# Sample dialog factory (kept independent from test_dialog_index)
# ─────────────────────────────────────────────────────────────────────

def make_solved(theorem: str, *, proof: str) -> dict:
    return {
        "schema_version": "3.0",
        "meta": {"theorem_statement": theorem},
        "messages": [
            {"role": "user", "content": "prove"},
            {"role": "assistant", "content": f"```lean\n{proof}\n```"},
        ],
        "result": {"success": True,
                   "successful_proof": proof,
                   "termination": "success"},
    }

# ─────────────────────────────────────────────────────────────────────
# 1. ObservationPolicy default
# ─────────────────────────────────────────────────────────────────────

class TestObservationPolicyDefaults:
    def test_inject_similar_dialogs_defaults_to_false(self):
        """V5 must not change pre-V5 behaviour for existing profiles.

        All built-in PRESETS keep ``inject_similar_dialogs=False`` —
        only profiles that explicitly set it opt in.
        """
        p = ObservationPolicy()
        assert p.inject_similar_dialogs is False
        assert p.n_similar_dialogs == 3
        assert p.similar_dialogs_max_chars == 2000

    def test_all_presets_have_inject_similar_dialogs_off_by_default(self):
        from prover.unified.profiles import PRESETS
        # Hand-audit: every shipped preset should keep this flag off
        # because we don't know whether the caller has populated a
        # DialogIndex. Users explicitly enable it via YAML / inline
        # override.
        # Exception: profiles named "*_knowledge" are explicitly the
        # opt-in profiles for cross-problem knowledge retrieval — by
        # convention their NAME signals the opt-in, and the runner
        # gracefully no-ops when --dialog-index is not configured.
        for name, prof in PRESETS.items():
            if name.endswith("_knowledge"):
                continue
            assert prof.observation.inject_similar_dialogs is False, (
                f"Preset {name!r} silently turns on cross-problem "
                f"dialog injection — should be opt-in")

# ─────────────────────────────────────────────────────────────────────
# 2. KnowledgeReader integration
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def populated_reader():
    """A KnowledgeReader with a DialogIndex pre-populated with three
    related solved dialogs."""
    from knowledge.store import UnifiedKnowledgeStore
    store = UnifiedKnowledgeStore(":memory:")
    reader = KnowledgeReader(store)

    idx = DialogIndex()
    idx.add_dialog(make_solved(
        "theorem nat_add_zero (n : ℕ) : n + 0 = n",
        proof="by simp"))
    idx.add_dialog(make_solved(
        "theorem nat_zero_add (n : ℕ) : 0 + n = n",
        proof="by simp"))
    idx.add_dialog(make_solved(
        "theorem int_add_comm (a b : ℤ) : a + b = b + a",
        proof="by ring"))
    reader.attach_dialog_index(idx, populate_from_store=False)
    return reader

class TestReaderDialogMethods:
    def test_attach_makes_index_accessible(self, populated_reader):
        assert populated_reader.dialog_index is not None
        assert populated_reader.dialog_index.size == 3

    def test_find_similar_dialogs_returns_matches(self, populated_reader):
        async def go():
            return await populated_reader.find_similar_dialogs(
                "theorem t (k : ℕ) : k + 0 = k", top_k=3)
        matches = asyncio.run(go())
        assert len(matches) >= 1
        assert all(m.solved for m in matches)

    def test_render_similar_dialogs_returns_text(self, populated_reader):
        async def go():
            return await populated_reader.render_similar_dialogs(
                "theorem t (k : ℕ) : k + 0 = k",
                top_k=2, max_chars=1000)
        text = asyncio.run(go())
        assert "Past similar work" in text
        assert "```lean" in text

    def test_no_index_returns_empty_results(self):
        from knowledge.store import UnifiedKnowledgeStore
        reader = KnowledgeReader(UnifiedKnowledgeStore(":memory:"))
        async def go():
            matches = await reader.find_similar_dialogs(
                "theorem t : True")
            text = await reader.render_similar_dialogs(
                "theorem t : True")
            return matches, text
        matches, text = asyncio.run(go())
        assert matches == []
        assert text == ""

    def test_attach_with_populate_from_empty_store(self):
        """populate_from_store=True is fail-soft — empty store is fine."""
        from knowledge.store import UnifiedKnowledgeStore
        store = UnifiedKnowledgeStore(":memory:")
        reader = KnowledgeReader(store)
        idx = DialogIndex()
        n = reader.attach_dialog_index(idx, populate_from_store=True)
        assert n == 0
        assert reader.dialog_index is idx

    def test_attach_skips_populate_when_index_already_populated(
            self, populated_reader):
        # Re-attach same idx: should not re-ingest.
        idx = populated_reader.dialog_index
        size_before = idx.size
        n = populated_reader.attach_dialog_index(
            idx, populate_from_store=True)
        assert n == 0
        assert idx.size == size_before

# ─────────────────────────────────────────────────────────────────────
# 3. UnifiedProofRunner — initial-message injection
# ─────────────────────────────────────────────────────────────────────

@dataclass
class _FakeProblem:
    theorem_statement: str = ""
    natural_language: str = ""

class _NullLLM:
    """Minimal LLM stub — runner.__init__ only stores it."""
    def model_name(self): return "null"
    def generate(self, *a, **kw): raise RuntimeError("not used")

class TestRunnerInitialMessage:
    def _build_runner(self, *, dialog_index=None):
        return UnifiedProofRunner(
            llm=_NullLLM(),
            dialog_index=dialog_index,
        )

    def test_runner_accepts_dialog_index_kwarg(self):
        idx = DialogIndex()
        runner = self._build_runner(dialog_index=idx)
        assert runner.dialog_index is idx

    def test_runner_default_dialog_index_is_none(self):
        runner = self._build_runner()
        assert runner.dialog_index is None

    def test_initial_message_unchanged_when_flag_off(self):
        """Without inject_similar_dialogs (default), the initial
        message must not contain similar-work content even when an
        index is attached."""
        idx = DialogIndex()
        idx.add_dialog(make_solved(
            "theorem old (n : ℕ) : n + 0 = n",
            proof="by simp"))
        runner = self._build_runner(dialog_index=idx)
        prof = get_profile("whole_proof")  # flag is False by default
        problem = _FakeProblem(
            theorem_statement="theorem t (m : ℕ) : m + 0 = m")
        msg = runner._build_initial_message(problem, prof)
        assert "Past similar work" not in msg

    def test_initial_message_includes_similar_when_flag_on(self):
        idx = DialogIndex()
        idx.add_dialog(make_solved(
            "theorem old (n : ℕ) : n + 0 = n",
            proof="by simp"))
        runner = self._build_runner(dialog_index=idx)
        prof = get_profile("whole_proof")
        # Mutate observation policy in-place via dataclasses.replace
        from dataclasses import replace
        prof = replace(prof, observation=replace(
            prof.observation, inject_similar_dialogs=True))
        problem = _FakeProblem(
            theorem_statement="theorem t (m : ℕ) : m + 0 = m")
        msg = runner._build_initial_message(problem, prof)
        assert "Past similar work" in msg
        assert "theorem old" in msg
        assert "by simp" in msg

    def test_initial_message_works_when_flag_on_but_no_index(self):
        """Flag-on without DialogIndex must not crash."""
        runner = self._build_runner(dialog_index=None)
        prof = get_profile("whole_proof")
        from dataclasses import replace
        prof = replace(prof, observation=replace(
            prof.observation, inject_similar_dialogs=True))
        problem = _FakeProblem(
            theorem_statement="theorem t : True")
        msg = runner._build_initial_message(problem, prof)
        # Just confirm it doesn't raise and produces a non-empty
        # initial message that does NOT contain the heading.
        assert msg
        assert "Past similar work" not in msg

    def test_initial_message_silent_on_index_exception(self):
        """If the DialogIndex raises during render, the initial
        message must still build cleanly (best-effort principle)."""
        class BrokenIndex:
            def render_for_prompt(self, *a, **kw):
                raise RuntimeError("boom")
        runner = self._build_runner(dialog_index=BrokenIndex())
        from dataclasses import replace
        prof = get_profile("whole_proof")
        prof = replace(prof, observation=replace(
            prof.observation, inject_similar_dialogs=True))
        problem = _FakeProblem(theorem_statement="theorem x : 1 = 1")
        msg = runner._build_initial_message(problem, prof)
        assert "Past similar work" not in msg
        assert "theorem x" in msg or "Theorem to prove" in msg
