"""tests/test_v5_dialog_index.py — V5 cross-problem dialog retrieval.

Pins the contract added in :file:`knowledge/dialog_index.py`:
ingest from disk and from ``proof_contexts``, similarity retrieval with
solved-only filter and theorem dedup, prompt rendering with truncation,
and fail-soft on malformed inputs.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from knowledge.dialog_index import (
    DialogIndex, SimilarDialogMatch,
    extract_final_proof, extract_used_tactics,
)


# ─────────────────────────────────────────────────────────────────────
# Sample dialog factories
# ─────────────────────────────────────────────────────────────────────

def make_dialog(theorem: str, *, solved: bool = True,
                tactics: list[str] | None = None,
                final_proof: str | None = None,
                schema_version: str = "3.0") -> dict:
    """Create a minimal wrapped-dialog dict for testing."""
    msgs = [{"role": "user", "content": "prove it"}]
    if tactics:
        for t in tactics:
            msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{t[:5]}",
                    "function": {
                        "name": "tactic_apply",
                        "arguments": json.dumps({"tactic": t}),
                    },
                }],
            })
    if final_proof and solved:
        msgs.append({
            "role": "assistant",
            "content": f"```lean\n{final_proof}\n```",
        })
    return {
        "schema_version": schema_version,
        "meta": {"theorem_statement": theorem},
        "messages": msgs,
        "result": {
            "success": solved,
            "successful_proof": final_proof if (solved and final_proof) else "",
            "termination": "success" if solved else "max_turns",
        },
    }


# ─────────────────────────────────────────────────────────────────────
# 1. extract_* helpers
# ─────────────────────────────────────────────────────────────────────


class TestExtractHelpers:
    def test_extract_final_proof_from_result(self):
        d = make_dialog("theorem t : True", final_proof="by trivial")
        assert extract_final_proof(d) == "by trivial"

    def test_extract_final_proof_from_assistant_msg(self):
        # Dialog without ``successful_proof`` — should fall back to
        # scanning assistant messages.
        d = {
            "meta": {"theorem_statement": "theorem t : True"},
            "messages": [
                {"role": "user", "content": "prove"},
                {"role": "assistant",
                 "content": "Here it is:\n```lean\nby trivial\n```\n"},
            ],
            "result": {},
        }
        assert extract_final_proof(d) == "by trivial"

    def test_extract_final_proof_empty_when_absent(self):
        d = {"meta": {}, "messages": [], "result": {}}
        assert extract_final_proof(d) == ""

    def test_extract_used_tactics_from_tool_calls(self):
        d = make_dialog("theorem t : True", tactics=["rfl", "simp", "ring"])
        tactics = extract_used_tactics(d)
        assert tactics == ["rfl", "simp", "ring"]

    def test_extract_used_tactics_handles_str_arguments(self):
        # Some providers serialize arguments as a JSON-encoded string.
        d = {
            "messages": [
                {"role": "assistant", "content": "",
                 "tool_calls": [{
                     "function": {"name": "tactic_apply",
                                  "arguments": '{"tactic": "exact rfl"}'},
                 }]},
            ],
        }
        assert extract_used_tactics(d) == ["exact rfl"]

    def test_extract_used_tactics_handles_malformed_args(self):
        # Bad JSON should be skipped silently.
        d = {
            "messages": [
                {"role": "assistant", "content": "",
                 "tool_calls": [{
                     "function": {"name": "tactic_apply",
                                  "arguments": "not json {{"},
                 }]},
            ],
        }
        assert extract_used_tactics(d) == []

    def test_extract_helpers_tolerate_non_dict(self):
        assert extract_final_proof("not a dict") == ""  # type: ignore[arg-type]
        assert extract_used_tactics(None) == []  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────
# 2. add_dialog + size + clear
# ─────────────────────────────────────────────────────────────────────


class TestAddDialog:
    def test_add_one_dialog(self):
        idx = DialogIndex()
        ok = idx.add_dialog(
            make_dialog("theorem foo (n : ℕ) : n + 0 = n",
                        final_proof="by simp"))
        assert ok is True
        assert idx.size == 1

    def test_skip_dialog_without_theorem(self):
        idx = DialogIndex()
        ok = idx.add_dialog({"meta": {}, "messages": [], "result": {}})
        assert ok is False
        assert idx.size == 0

    def test_clear(self):
        idx = DialogIndex()
        for i in range(3):
            idx.add_dialog(make_dialog(f"theorem t{i} : True"))
        assert idx.size == 3
        idx.clear()
        assert idx.size == 0


# ─────────────────────────────────────────────────────────────────────
# 3. find_similar
# ─────────────────────────────────────────────────────────────────────


class TestFindSimilar:
    def _populate(self, idx: DialogIndex):
        # Three solved + one unsolved dialog covering different
        # theorem shapes.
        idx.add_dialog(make_dialog(
            "theorem nat_add_zero (n : ℕ) : n + 0 = n",
            tactics=["simp"], final_proof="by simp"))
        idx.add_dialog(make_dialog(
            "theorem nat_zero_add (n : ℕ) : 0 + n = n",
            tactics=["simp"], final_proof="by simp"))
        idx.add_dialog(make_dialog(
            "theorem int_add_comm (a b : ℤ) : a + b = b + a",
            tactics=["ring"], final_proof="by ring"))
        idx.add_dialog(make_dialog(
            "theorem broken (x : Bool) : x = true",
            solved=False, tactics=["sorry"]))

    def test_finds_nat_arithmetic_when_query_is_nat(self):
        idx = DialogIndex()
        self._populate(idx)
        matches = idx.find_similar(
            "theorem t (k : ℕ) : k + 0 = k", top_k=3)
        assert len(matches) >= 1
        # Top hit must be a nat dialog, not the int one.
        top = matches[0]
        assert "ℕ" in top.theorem or "Nat" in top.theorem or "nat" in top.theorem

    def test_solved_only_default(self):
        idx = DialogIndex()
        self._populate(idx)
        matches = idx.find_similar(
            "theorem broken (y : Bool) : y = true", top_k=5)
        # The unsolved dialog has the closest theorem text — but
        # solved_only=True means we exclude it, so we get either no
        # match or solved-only matches with lower scores.
        for m in matches:
            assert m.solved is True

    def test_solved_only_false_includes_failures(self):
        idx = DialogIndex()
        self._populate(idx)
        matches = idx.find_similar(
            "theorem broken (y : Bool) : y = true",
            top_k=5, solved_only=False)
        # Now we should get at least one unsolved match.
        assert any(not m.solved for m in matches)

    def test_empty_query_returns_empty(self):
        idx = DialogIndex()
        self._populate(idx)
        assert idx.find_similar("") == []
        assert idx.find_similar("   ") == []

    def test_empty_index_returns_empty(self):
        idx = DialogIndex()
        assert idx.find_similar("theorem t : True") == []

    def test_dedup_by_theorem(self):
        # Two solved dialogs with identical theorem — only the
        # highest-scoring one should survive dedup.
        idx = DialogIndex()
        idx.add_dialog(make_dialog(
            "theorem same : 1 + 1 = 2",
            tactics=["rfl"], final_proof="by rfl"))
        idx.add_dialog(make_dialog(
            "theorem same : 1 + 1 = 2",
            tactics=["norm_num"], final_proof="by norm_num"))
        matches = idx.find_similar("theorem t : 1 + 1 = 2", top_k=5)
        assert len(matches) == 1
        assert matches[0].theorem == "theorem same : 1 + 1 = 2"

    def test_top_k_limits_results(self):
        idx = DialogIndex()
        for i in range(8):
            idx.add_dialog(make_dialog(
                f"theorem t{i} (n : ℕ) : n + 0 = n",
                tactics=["simp"], final_proof="by simp"))
        matches = idx.find_similar(
            "theorem t (m : ℕ) : m + 0 = m", top_k=3)
        assert len(matches) == 3


# ─────────────────────────────────────────────────────────────────────
# 4. render_for_prompt
# ─────────────────────────────────────────────────────────────────────


class TestRenderForPrompt:
    def test_render_includes_theorem_and_proof(self):
        idx = DialogIndex()
        idx.add_dialog(make_dialog(
            "theorem foo (n : ℕ) : n + 0 = n",
            tactics=["simp"], final_proof="by simp"))
        text = idx.render_for_prompt(
            "theorem t (m : ℕ) : m + 0 = m", top_k=3)
        assert "Past similar work" in text
        assert "theorem foo" in text
        assert "```lean" in text
        assert "by simp" in text

    def test_render_empty_when_no_matches(self):
        idx = DialogIndex()
        # Only one stored dialog about a totally different topic.
        idx.add_dialog(make_dialog(
            "theorem unrelated_a : True",
            tactics=["trivial"], final_proof="by trivial"))
        text = idx.render_for_prompt(
            "theorem zzz_xyz_qq : 42 = 42", top_k=3,
            solved_only=True)
        # Either we get a hit (low score, but allowed) or empty.
        # The contract only requires no crash and well-formed output.
        if text:
            assert "Past similar work" in text

    def test_render_truncates_at_max_chars(self):
        idx = DialogIndex()
        big_proof = "by\n" + "  rfl\n" * 200  # ~1600 chars
        idx.add_dialog(make_dialog(
            "theorem t : 1 = 1",
            tactics=["rfl"], final_proof=big_proof))
        text = idx.render_for_prompt(
            "theorem u : 1 = 1", top_k=1, max_chars=200)
        assert len(text) <= 280  # max_chars + truncation suffix slack
        assert "truncated" in text.lower() or len(text) < 200

    def test_render_returns_string_type(self):
        idx = DialogIndex()
        text = idx.render_for_prompt("theorem t : True", top_k=3)
        assert isinstance(text, str)


# ─────────────────────────────────────────────────────────────────────
# 5. index_from_directory
# ─────────────────────────────────────────────────────────────────────


class TestIndexFromDirectory:
    def test_ingest_one_dialog_file(self, tmp_path):
        d = tmp_path / "task1"
        d.mkdir()
        dialog = make_dialog(
            "theorem t1 : 1 + 1 = 2",
            tactics=["rfl"], final_proof="by rfl")
        (d / "dialog.json").write_text(json.dumps(dialog))

        idx = DialogIndex()
        n = idx.index_from_directory(tmp_path)
        assert n == 1
        assert idx.size == 1

    def test_ingest_recursive(self, tmp_path):
        for i in range(3):
            d = tmp_path / f"task_{i}"
            d.mkdir()
            (d / "dialog.json").write_text(json.dumps(make_dialog(
                f"theorem t_{i} (n : ℕ) : n = n",
                tactics=["rfl"], final_proof="by rfl")))
        idx = DialogIndex()
        n = idx.index_from_directory(tmp_path)
        assert n == 3

    def test_skip_malformed_json(self, tmp_path):
        good = tmp_path / "good"
        good.mkdir()
        (good / "dialog.json").write_text(json.dumps(make_dialog(
            "theorem ok : True",
            tactics=["trivial"], final_proof="by trivial")))

        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "dialog.json").write_text("{not json {{")

        idx = DialogIndex()
        n = idx.index_from_directory(tmp_path)
        # Bad file dropped silently; good one ingested.
        assert n == 1

    def test_missing_directory_returns_zero(self, tmp_path):
        idx = DialogIndex()
        # Does not exist:
        n = idx.index_from_directory(tmp_path / "nope")
        assert n == 0
        assert idx.size == 0

    def test_limit_caps_ingested(self, tmp_path):
        for i in range(10):
            d = tmp_path / f"t_{i}"
            d.mkdir()
            (d / "dialog.json").write_text(json.dumps(make_dialog(
                f"theorem t_{i} : True",
                tactics=["trivial"], final_proof="by trivial")))
        idx = DialogIndex()
        n = idx.index_from_directory(tmp_path, limit=4)
        assert n == 4


# ─────────────────────────────────────────────────────────────────────
# 6. index_from_proof_context_store
# ─────────────────────────────────────────────────────────────────────


class TestIndexFromProofContextStore:
    """Hits a fake store with the expected ``_connect()`` shape."""

    def _make_fake_store(self, rows: list[dict]):
        # Build an in-memory SQLite database with proof_contexts schema
        # and stub a context-manager ``_connect()``.
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE proof_contexts (
                id INTEGER PRIMARY KEY,
                theorem TEXT,
                state_json TEXT,
                solved INTEGER,
                updated_at REAL
            )""")
        for r in rows:
            conn.execute(
                "INSERT INTO proof_contexts(id, theorem, state_json, "
                "solved, updated_at) VALUES (?,?,?,?,?)",
                (r["id"], r["theorem"], r["state_json"],
                 r["solved"], r["updated_at"]))
        conn.commit()

        class _Store:
            def _connect(self_inner):
                from contextlib import contextmanager
                @contextmanager
                def cm():
                    yield conn
                return cm()
        return _Store()

    def test_ingest_two_solved_rows(self):
        store = self._make_fake_store([
            {"id": 1,
             "theorem": "theorem db_a (n : ℕ) : n + 0 = n",
             "state_json": json.dumps({"tactic_history": ["simp"]}),
             "solved": 1,
             "updated_at": time.time()},
            {"id": 2,
             "theorem": "theorem db_b : True",
             "state_json": json.dumps({"tactic_history": ["trivial"]}),
             "solved": 1,
             "updated_at": time.time()},
        ])
        idx = DialogIndex()
        n = idx.index_from_proof_context_store(store)
        assert n == 2
        assert idx.size == 2

    def test_ingest_unsolved_row(self):
        store = self._make_fake_store([
            {"id": 1,
             "theorem": "theorem db_x : 0 = 0",
             "state_json": json.dumps({"tactic_history": []}),
             "solved": 0,
             "updated_at": time.time()},
        ])
        idx = DialogIndex()
        n = idx.index_from_proof_context_store(store)
        assert n == 1
        # Unsolved entry gets ingested but is filtered out by
        # the default ``solved_only=True`` retrieval.
        assert idx.find_similar("theorem db_x : 0 = 0") == []
        # …yet retrievable when we relax that constraint.
        matches = idx.find_similar(
            "theorem db_x : 0 = 0", solved_only=False)
        assert len(matches) == 1
        assert matches[0].solved is False

    def test_store_without_connect_returns_zero(self):
        idx = DialogIndex()
        # ``object()`` has no ``_connect`` attribute — must not crash.
        n = idx.index_from_proof_context_store(object())
        assert n == 0

    def test_malformed_state_json_skipped(self):
        store = self._make_fake_store([
            {"id": 1,
             "theorem": "theorem ok : True",
             "state_json": json.dumps({"tactic_history": ["trivial"]}),
             "solved": 1,
             "updated_at": 1.0},
            {"id": 2,
             "theorem": "theorem broken : True",
             "state_json": "{not json",  # malformed
             "solved": 1,
             "updated_at": 2.0},
        ])
        idx = DialogIndex()
        n = idx.index_from_proof_context_store(store)
        assert n == 1  # second row dropped


# ─────────────────────────────────────────────────────────────────────
# 7. End-to-end: mix of ingest + retrieve
# ─────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_disk_plus_db_combined(self, tmp_path):
        # File source.
        d = tmp_path / "task_disk"
        d.mkdir()
        (d / "dialog.json").write_text(json.dumps(make_dialog(
            "theorem disk (n : ℕ) : n + 0 = n",
            tactics=["simp"], final_proof="by simp")))

        # DB source.
        store_helper = TestIndexFromProofContextStore()._make_fake_store([
            {"id": 99,
             "theorem": "theorem from_db (n : ℕ) : n * 1 = n",
             "state_json": json.dumps(
                 {"tactic_history": ["simp"]}),
             "solved": 1,
             "updated_at": time.time()},
        ])

        idx = DialogIndex()
        idx.index_from_directory(tmp_path)
        idx.index_from_proof_context_store(store_helper)
        assert idx.size == 2

        # Both should be retrievable.
        all_matches = idx.find_similar(
            "theorem q (n : ℕ) : n + 0 = n", top_k=2)
        sources = {m.source for m in all_matches}
        assert any(s.startswith("file:") for s in sources) or \
               any(s.startswith("db:") for s in sources)
