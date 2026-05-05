"""

Closes V6+ item #1 from INFRA_MERGE_V5_REPORT.md ("DialogIndex
persistence to SQLite"). The test surface pins three things:

  1. Round-trip — every entry attribute survives persist + load
     identically.
  2. Idempotence — re-persisting the same in-memory state to the
     same file does not double-count entries (UPSERT by theorem+source).
  3. Failure modes — missing file, corrupted file, schema-version
     mismatch all degrade as documented.

We do NOT test against an actual long-running process — that's an
integration concern. These tests run entirely on tmp_path.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from knowledge.dialog_index import (
    DialogIndex,
    _DialogEntry,
    _SQLITE_SCHEMA_VERSION,
    _connect_sqlite,
    _ensure_schema,
)

# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

def _mk_dialog(theorem: str, solved: bool = True,
               proof: str = "by simp",
               tactics=("simp", "rfl")) -> dict:
    """Build a wrapped-dialog dict matching DialogIndex's expectations."""
    return {
        "schema_version": "3.0",
        "meta": {"theorem_statement": theorem},
        "messages": [],
        "result": {
            "success": solved,
            "successful_proof": proof if solved else "",
        },
    }

@pytest.fixture
def populated_index() -> DialogIndex:
    """A DialogIndex with three solved + one unsolved entry."""
    idx = DialogIndex()
    idx.add_dialog(_mk_dialog("theorem t1 (n : ℕ) : n + 0 = n"),
                   source="file:test/t1.json", timestamp=100.0)
    idx.add_dialog(_mk_dialog("theorem t2 (n : ℕ) : 0 + n = n",
                              proof="by induction n; simp"),
                   source="file:test/t2.json", timestamp=200.0)
    idx.add_dialog(_mk_dialog("theorem t3 (a b : ℕ) : a + b = b + a",
                              proof="by ring"),
                   source="file:test/t3.json", timestamp=300.0)
    idx.add_dialog(_mk_dialog("theorem t4 (n : ℕ) : n * 0 = 0",
                              solved=False, proof=""),
                   source="file:test/t4.json", timestamp=400.0)
    return idx

# ─────────────────────────────────────────────────────────────────────
# Schema + connection helpers
# ─────────────────────────────────────────────────────────────────────

class TestSchema:

    def test_ensure_schema_is_idempotent(self, tmp_path: Path):
        db = tmp_path / "idx.sqlite"
        with _connect_sqlite(db) as conn:
            _ensure_schema(conn)
            _ensure_schema(conn)
            _ensure_schema(conn)
            row = conn.execute(
                "SELECT value FROM dialog_index_meta "
                "WHERE key='schema_version'"
            ).fetchone()
        assert row["value"] == _SQLITE_SCHEMA_VERSION

    def test_schema_creates_required_tables(self, tmp_path: Path):
        db = tmp_path / "idx.sqlite"
        with _connect_sqlite(db) as conn:
            _ensure_schema(conn)
            tables = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "dialog_entries" in tables
        assert "dialog_index_meta" in tables

    def test_schema_creates_indexes(self, tmp_path: Path):
        db = tmp_path / "idx.sqlite"
        with _connect_sqlite(db) as conn:
            _ensure_schema(conn)
            indexes = {r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()}
        assert "idx_dialog_entries_theorem" in indexes
        assert "idx_dialog_entries_solved" in indexes

    def test_unique_constraint_on_theorem_plus_source(self, tmp_path: Path):
        db = tmp_path / "idx.sqlite"
        with _connect_sqlite(db) as conn:
            _ensure_schema(conn)
            conn.execute(
                "INSERT INTO dialog_entries(theorem, solved, source) "
                "VALUES ('t', 1, 'a')")
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO dialog_entries(theorem, solved, source) "
                    "VALUES ('t', 1, 'a')")

    def test_newer_schema_version_refused(self, tmp_path: Path):
        db = tmp_path / "future.sqlite"
        with _connect_sqlite(db) as conn:
            _ensure_schema(conn)
            conn.execute(
                "INSERT OR REPLACE INTO dialog_index_meta(key, value) "
                "VALUES ('schema_version', '99')")
            conn.commit()
        idx = DialogIndex()
        with pytest.raises(RuntimeError, match="newer than this code"):
            idx.load_from_sqlite(db)

# ─────────────────────────────────────────────────────────────────────
# persist_to_sqlite
# ─────────────────────────────────────────────────────────────────────

class TestPersist:

    def test_persist_returns_row_count(self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        n = populated_index.persist_to_sqlite(db)
        assert n == 4

    def test_persist_creates_file(self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        assert not db.exists()
        populated_index.persist_to_sqlite(db)
        assert db.exists()
        assert db.stat().st_size > 0

    def test_persist_writes_all_entries(self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)
        with _connect_sqlite(db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM dialog_entries"
            ).fetchone()["c"]
        assert count == 4

    def test_persist_idempotent_no_duplication(self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)
        populated_index.persist_to_sqlite(db)
        populated_index.persist_to_sqlite(db)
        with _connect_sqlite(db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM dialog_entries"
            ).fetchone()["c"]
        assert count == 4

    def test_persist_replace_clears_old_entries(self, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        idx1 = DialogIndex()
        idx1.add_dialog(_mk_dialog("old1"), source="src/a")
        idx1.add_dialog(_mk_dialog("old2"), source="src/b")
        idx1.persist_to_sqlite(db)

        idx2 = DialogIndex()
        idx2.add_dialog(_mk_dialog("new1"), source="src/c")
        idx2.persist_to_sqlite(db, replace=True)

        with _connect_sqlite(db) as conn:
            theorems = {r["theorem"] for r in conn.execute(
                "SELECT theorem FROM dialog_entries"
            ).fetchall()}
        assert theorems == {"new1"}

    def test_persist_then_modify_in_memory_does_not_touch_disk(
            self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)

        # Mutate in-memory
        populated_index.add_dialog(
            _mk_dialog("post-persist"), source="file:test/late.json")

        # Disk still has 4
        with _connect_sqlite(db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM dialog_entries"
            ).fetchone()["c"]
        assert count == 4

    def test_persist_serializes_used_tactics_as_json(
            self, populated_index, tmp_path: Path):
        # Add an entry with tactic_history populated via add_dialog —
        # the helper builds messages so used_tactics extraction has
        # something to chew on.
        idx = DialogIndex()
        d = _mk_dialog("with_tactics")
        d["messages"] = [{
            "role": "assistant",
            "content": "Try this:",
            "tool_calls": [{
                "function": {
                    "name": "tactic_apply",
                    "arguments": json.dumps({"tactic": "exact rfl"}),
                }
            }],
        }]
        idx.add_dialog(d, source="src/x")
        db = tmp_path / "p.sqlite"
        idx.persist_to_sqlite(db)

        with _connect_sqlite(db) as conn:
            row = conn.execute(
                "SELECT used_tactics_json FROM dialog_entries "
                "WHERE theorem='with_tactics'"
            ).fetchone()
        tactics = json.loads(row["used_tactics_json"])
        assert tactics == ["exact rfl"]

    def test_persist_handles_unicode_theorem(self, tmp_path: Path):
        idx = DialogIndex()
        idx.add_dialog(_mk_dialog("theorem α (n : ℕ) : ∀ x, x = x"),
                       source="src/u")
        db = tmp_path / "p.sqlite"
        idx.persist_to_sqlite(db)

        with _connect_sqlite(db) as conn:
            row = conn.execute(
                "SELECT theorem FROM dialog_entries"
            ).fetchone()
        assert "ℕ" in row["theorem"]
        assert "∀" in row["theorem"]

    def test_persist_empty_index(self, tmp_path: Path):
        idx = DialogIndex()
        db = tmp_path / "p.sqlite"
        n = idx.persist_to_sqlite(db)
        assert n == 0
        # File still created with schema
        with _connect_sqlite(db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM dialog_entries"
            ).fetchone()["c"]
        assert count == 0

# ─────────────────────────────────────────────────────────────────────
# load_from_sqlite
# ─────────────────────────────────────────────────────────────────────

class TestLoad:

    def test_load_appends_to_existing(self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)

        idx2 = DialogIndex()
        idx2.add_dialog(_mk_dialog("preexisting"), source="src/pre")
        n = idx2.load_from_sqlite(db)
        assert n == 4
        assert idx2.size == 5

    def test_load_missing_file_returns_zero(self, tmp_path: Path):
        idx = DialogIndex()
        n = idx.load_from_sqlite(tmp_path / "nope.sqlite")
        assert n == 0
        assert idx.size == 0

    def test_load_corrupt_file_returns_zero(self, tmp_path: Path):
        bad = tmp_path / "bad.sqlite"
        bad.write_bytes(b"not a sqlite database, just garbage bytes" * 50)
        idx = DialogIndex()
        n = idx.load_from_sqlite(bad)
        assert n == 0

    def test_load_preserves_solved_flag(self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)

        idx2 = DialogIndex()
        idx2.load_from_sqlite(db)

        # Find the unsolved entry
        unsolved = [e for e in idx2._entries if not e.solved]
        assert len(unsolved) == 1
        assert "t4" in unsolved[0].theorem

    def test_load_preserves_timestamps(self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)

        idx2 = DialogIndex()
        idx2.load_from_sqlite(db)
        ts = sorted(e.timestamp for e in idx2._entries)
        assert ts == [100.0, 200.0, 300.0, 400.0]

    def test_load_preserves_source(self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)

        idx2 = DialogIndex()
        idx2.load_from_sqlite(db)
        sources = {e.source for e in idx2._entries}
        assert sources == {
            "file:test/t1.json", "file:test/t2.json",
            "file:test/t3.json", "file:test/t4.json"}

    def test_load_preserves_final_proof(self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)

        idx2 = DialogIndex()
        idx2.load_from_sqlite(db)
        proofs = sorted(e.final_proof for e in idx2._entries if e.solved)
        # All three solved entries had distinct proofs
        assert len(proofs) == 3
        assert "by simp" in proofs
        assert "by ring" in proofs

    def test_load_with_malformed_used_tactics_json_skips_tactics(
            self, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        with _connect_sqlite(db) as conn:
            _ensure_schema(conn)
            conn.execute(
                "INSERT INTO dialog_entries(theorem, solved, "
                "  final_proof, used_tactics_json, source, timestamp) "
                "VALUES ('t', 1, '', 'not json', 's', 0)")
            conn.commit()
        idx = DialogIndex()
        n = idx.load_from_sqlite(db)
        assert n == 1
        assert idx._entries[0].used_tactics == []

    def test_load_skips_blank_theorem_rows(self, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        with _connect_sqlite(db) as conn:
            _ensure_schema(conn)
            conn.execute(
                "INSERT INTO dialog_entries(theorem, solved, source) "
                "VALUES ('   ', 1, 'src/blank')")
            conn.execute(
                "INSERT INTO dialog_entries(theorem, solved, source) "
                "VALUES ('valid', 1, 'src/valid')")
            conn.commit()
        idx = DialogIndex()
        n = idx.load_from_sqlite(db)
        assert n == 1
        assert idx._entries[0].theorem == "valid"

# ─────────────────────────────────────────────────────────────────────
# replace_from_sqlite — snapshot semantics
# ─────────────────────────────────────────────────────────────────────

class TestReplace:

    def test_replace_drops_existing_entries(
            self, populated_index, tmp_path: Path):
        db = tmp_path / "snap.sqlite"
        # First snapshot: only entries 1,2
        snap = DialogIndex()
        snap.add_dialog(_mk_dialog("only1"), source="src/a")
        snap.add_dialog(_mk_dialog("only2"), source="src/b")
        snap.persist_to_sqlite(db)

        # populated_index has 4 entries; replace_from_sqlite should
        # leave it with only 2.
        n = populated_index.replace_from_sqlite(db)
        assert n == 2
        assert populated_index.size == 2
        theorems = {e.theorem for e in populated_index._entries}
        assert theorems == {"only1", "only2"}

    def test_replace_on_missing_file_clears_then_returns_zero(
            self, populated_index, tmp_path: Path):
        n = populated_index.replace_from_sqlite(tmp_path / "absent.sqlite")
        assert n == 0
        # Important contract: in-memory state is dropped even on miss
        assert populated_index.size == 0

# ─────────────────────────────────────────────────────────────────────
# End-to-end: persist + load + retrieval still works
# ─────────────────────────────────────────────────────────────────────

class TestRetrievalAfterRoundTrip:

    def test_find_similar_works_after_load(
            self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)

        idx2 = DialogIndex()
        idx2.load_from_sqlite(db)

        matches = idx2.find_similar(
            "theorem foo (k : ℕ) : k + 0 = k", top_k=2)
        # Should hit at least one of the n+0=n / 0+n=n entries
        assert len(matches) >= 1
        assert any("0" in m.theorem for m in matches)

    def test_render_for_prompt_works_after_load(
            self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)

        idx2 = DialogIndex()
        idx2.load_from_sqlite(db)
        rendered = idx2.render_for_prompt(
            "theorem foo (k : ℕ) : k + 0 = k", top_k=2)
        assert "Past similar work" in rendered
        assert "```lean" in rendered

    def test_load_marks_index_dirty(self, populated_index, tmp_path: Path):
        # Persist then load into a fresh index — first find_similar
        # call should rebuild rather than return [] from a cached
        # empty retriever.
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)
        idx2 = DialogIndex()
        # Force "fresh retriever" state with one find_similar on empty
        assert idx2.find_similar("anything") == []
        idx2.load_from_sqlite(db)
        # After load, the dirty flag must be set so a query rebuilds
        matches = idx2.find_similar("theorem t (n : ℕ) : n + 0 = n",
                                     top_k=1, solved_only=True)
        assert len(matches) >= 1

# ─────────────────────────────────────────────────────────────────────
# sqlite_file_size helper
# ─────────────────────────────────────────────────────────────────────

class TestFileSize:

    def test_size_of_missing_file_is_none(self, tmp_path: Path):
        assert DialogIndex.sqlite_file_size(
            tmp_path / "absent.sqlite") is None

    def test_size_of_persisted_file_is_positive(
            self, populated_index, tmp_path: Path):
        db = tmp_path / "p.sqlite"
        populated_index.persist_to_sqlite(db)
        size = DialogIndex.sqlite_file_size(db)
        assert size is not None
        assert size > 0

# ─────────────────────────────────────────────────────────────────────
# Edge cases the V6 design contract pins
# ─────────────────────────────────────────────────────────────────────

class TestContract:

    def test_two_entries_same_theorem_different_source_both_persist(
            self, tmp_path: Path):
        """The UNIQUE constraint is (theorem, source) — different sources
        for the same theorem should both round-trip."""
        idx = DialogIndex()
        idx.add_dialog(_mk_dialog("same_theorem", proof="by simp"),
                       source="src/a")
        idx.add_dialog(_mk_dialog("same_theorem", proof="by ring"),
                       source="src/b")
        db = tmp_path / "dup.sqlite"
        idx.persist_to_sqlite(db)

        idx2 = DialogIndex()
        idx2.load_from_sqlite(db)
        assert idx2.size == 2
        proofs = {e.final_proof for e in idx2._entries}
        assert proofs == {"by simp", "by ring"}

    def test_repersist_with_updated_proof_overwrites(self, tmp_path: Path):
        """If we persist (t, src) twice with different proofs, the
        second write replaces the first — UPSERT semantics."""
        db = tmp_path / "up.sqlite"

        idx1 = DialogIndex()
        idx1.add_dialog(_mk_dialog("t", proof="old proof"), source="src/x")
        idx1.persist_to_sqlite(db)

        idx2 = DialogIndex()
        idx2.add_dialog(_mk_dialog("t", proof="new proof"), source="src/x")
        idx2.persist_to_sqlite(db)

        idx3 = DialogIndex()
        idx3.load_from_sqlite(db)
        assert idx3.size == 1
        assert idx3._entries[0].final_proof == "new proof"
