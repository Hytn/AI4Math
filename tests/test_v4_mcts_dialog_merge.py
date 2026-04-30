"""tests/test_v4_mcts_dialog_merge.py — v4 MCTS↔dialog 合流契约

After v4, the three tree-search profiles (mcts / best_first / beam)
sit alongside the linear ones in ``PRESETS`` and produce the same
``dialog.json`` (schema 3.0) — same ``messages`` shape, plus an
optional ``meta.search_tree`` block containing the explored DAG.

This file pins:

  1. Schema bump 2.0 → 3.0 (existing 2.0 / 1.0 files still load)
  2. ``DialogBuilder.set_search_tree`` round-trips through save/load
  3. ``validate_dialog`` accepts a well-formed search_tree, flags malformed
  4. ``search_tree_of`` and ``solved_path_of`` accessors
  5. ``SharedSearchState.to_search_tree_dict`` schema
  6. ``SharedSearchState.solved_path_messages`` flattens root→solved
  7. The three tree-search profiles ARE in ``PRESETS``
  8. ``UnifiedResult.search_tree`` propagates into dialog.json on save
  9. dialog.json-only contract holds — search-tree runs still produce
     exactly one file, no new sidecars
"""
from __future__ import annotations

import json
import os
import sys

import pytest

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)


from agent.persistence.dialog_format import (
    SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS,
    DialogBuilder, save_dialog, load_dialog, validate_dialog,
    search_tree_of, solved_path_of,
)
from agent.persistence import DIALOG_FILENAME


# ─────────────────────────────────────────────────────────────────────────
# 1. Schema bump
# ─────────────────────────────────────────────────────────────────────────

class TestSchemaBumpV4:
    def test_schema_version_is_3_0(self):
        assert SCHEMA_VERSION == "3.0"

    def test_supported_schema_versions(self):
        # We must continue accepting older payloads.
        assert "1.0" in SUPPORTED_SCHEMA_VERSIONS
        assert "2.0" in SUPPORTED_SCHEMA_VERSIONS
        assert "3.0" in SUPPORTED_SCHEMA_VERSIONS

    def test_load_v2_dialog_still_works(self, tmp_path):
        """A 2.0 file (no search_tree) loads with no errors."""
        d = {
            "schema_version": "2.0",
            "meta": {"problem_id": "old"},
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "result": {"success": True},
        }
        p = tmp_path / "v2.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        loaded = load_dialog(p)
        assert loaded["schema_version"] == "2.0"
        assert validate_dialog(loaded) == []

    def test_load_v1_legacy_list_still_works(self, tmp_path):
        """A 1.0 file (raw list) auto-wraps."""
        legacy = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        p = tmp_path / "v1.json"
        p.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = load_dialog(p)
        assert loaded["schema_version"] == "1.0"
        assert loaded["messages"] == legacy


# ─────────────────────────────────────────────────────────────────────────
# 2-3. Builder round-trip + validation of search_tree
# ─────────────────────────────────────────────────────────────────────────

class TestSearchTreeRoundtrip:
    def _well_formed_tree(self) -> dict:
        return {
            "kind": "ucb",
            "root_node_id": 0,
            "solved_node_id": 2,
            "total_nodes": 3,
            "max_depth": 2,
            "nodes": [
                {"node_id": 0, "parent_id": None, "tactic": None,
                 "depth": 0, "status": "open",
                 "visit_count": 5, "success_count": 1,
                 "score": 0.0, "messages": []},
                {"node_id": 1, "parent_id": 0, "tactic": "intro h",
                 "depth": 1, "status": "open",
                 "visit_count": 3, "success_count": 1,
                 "score": 0.7,
                 "messages": [{"role": "assistant", "content": "intro h"}]},
                {"node_id": 2, "parent_id": 1, "tactic": "exact h",
                 "depth": 2, "status": "solved",
                 "visit_count": 1, "success_count": 1,
                 "score": 12.0,
                 "messages": [{"role": "assistant", "content": "exact h"}]},
            ],
        }

    def test_set_search_tree_lands_in_meta(self):
        b = DialogBuilder()
        b.set_meta(problem_id="p", theorem_statement="t")
        b.add_user("Prove t")
        b.add_assistant_proof(":= by exact trivial")
        b.set_result(success=True)
        b.set_search_tree(self._well_formed_tree())
        d = b.build()
        assert d["schema_version"] == "3.0"
        assert "search_tree" in d["meta"]
        assert d["meta"]["search_tree"]["kind"] == "ucb"
        assert d["meta"]["search_tree"]["solved_node_id"] == 2

    def test_save_load_search_tree_roundtrip(self, tmp_path):
        b = DialogBuilder()
        b.set_meta(problem_id="p")
        b.add_user("Prove")
        b.add_assistant_proof(":= by simp")
        b.set_search_tree(self._well_formed_tree())
        d_in = b.build()
        path = tmp_path / "dialog.json"
        save_dialog(d_in, path)
        d_out = load_dialog(path)
        assert d_out["meta"]["search_tree"]["solved_node_id"] == 2
        assert validate_dialog(d_out) == []

    def test_validate_clean_tree_no_issues(self):
        b = DialogBuilder()
        b.set_meta(problem_id="p")
        b.add_user("u")
        b.add_assistant("hi")
        b.set_search_tree(self._well_formed_tree())
        assert validate_dialog(b.build()) == []

    def test_validate_orphan_node_flagged(self):
        bad_tree = self._well_formed_tree()
        bad_tree["nodes"][1]["parent_id"] = 999  # orphan
        b = DialogBuilder()
        b.set_meta(problem_id="p")
        b.add_user("u")
        b.add_assistant("a")
        b.set_search_tree(bad_tree)
        issues = validate_dialog(b.build())
        codes = {i.code for i in issues}
        assert "search_tree_orphan" in codes

    def test_validate_dangling_solved_flagged(self):
        bad_tree = self._well_formed_tree()
        bad_tree["solved_node_id"] = 42  # not in nodes
        b = DialogBuilder()
        b.set_meta(problem_id="p")
        b.add_user("u")
        b.add_assistant("a")
        b.set_search_tree(bad_tree)
        codes = {i.code for i in validate_dialog(b.build())}
        assert "search_tree_solved_dangling" in codes

    def test_validate_unknown_kind_flagged(self):
        bad_tree = self._well_formed_tree()
        bad_tree["kind"] = "monte_carlo"
        b = DialogBuilder()
        b.set_meta(problem_id="p")
        b.add_user("u")
        b.add_assistant("a")
        b.set_search_tree(bad_tree)
        codes = {i.code for i in validate_dialog(b.build())}
        assert "search_tree_kind_unknown" in codes


# ─────────────────────────────────────────────────────────────────────────
# 4. Accessors
# ─────────────────────────────────────────────────────────────────────────

class TestAccessors:
    def test_search_tree_of_returns_none_when_absent(self):
        d = {"schema_version": "3.0", "meta": {}, "messages": [],
             "result": {}}
        assert search_tree_of(d) is None

    def test_search_tree_of_returns_dict_when_present(self):
        d = {"schema_version": "3.0",
             "meta": {"search_tree": {"kind": "ucb", "nodes": []}},
             "messages": [], "result": {}}
        out = search_tree_of(d)
        assert out is not None and out["kind"] == "ucb"

    def test_solved_path_of_alias_to_messages(self):
        d = {"schema_version": "3.0", "meta": {},
             "messages": [{"role": "user", "content": "x"}],
             "result": {}}
        path = solved_path_of(d)
        assert path == d["messages"]

    def test_search_tree_of_handles_legacy_list(self):
        # Plain message list (1.0) — no meta, no tree.
        assert search_tree_of([{"role": "user", "content": "x"}]) is None


# ─────────────────────────────────────────────────────────────────────────
# 5-6. SharedSearchState tree dict + solved_path_messages
# ─────────────────────────────────────────────────────────────────────────

class TestSharedSearchStateSerialization:
    def _build_solved_tree(self):
        from prover.unified.search_driver import SharedSearchState
        s = SharedSearchState(root_env_id=0, root_goals=["G"])
        # root → child1 (intro) → child2 (exact, solved)
        c1 = s.expand(parent_node_id=0, tactic="intro h",
                       new_env_id=1, remaining_goals=["G'"],
                       is_complete=False)
        s.nodes[c1].messages.append(
            {"role": "assistant", "content": "intro h"})
        c2 = s.expand(parent_node_id=c1, tactic="exact h",
                       new_env_id=2, remaining_goals=[],
                       is_complete=True)
        s.nodes[c2].messages.append(
            {"role": "assistant", "content": "exact h"})
        return s, c1, c2

    def test_to_search_tree_dict_basic_shape(self):
        s, c1, c2 = self._build_solved_tree()
        tree = s.to_search_tree_dict(kind="ucb")
        assert tree["kind"] == "ucb"
        assert tree["root_node_id"] == 0
        assert tree["solved_node_id"] == c2
        assert tree["total_nodes"] == 3
        assert tree["max_depth"] == 2
        ids = {n["node_id"] for n in tree["nodes"]}
        assert ids == {0, c1, c2}

    def test_to_search_tree_dict_includes_per_node_messages(self):
        s, c1, c2 = self._build_solved_tree()
        tree = s.to_search_tree_dict(kind="ucb")
        node1 = next(n for n in tree["nodes"] if n["node_id"] == c1)
        assert node1["messages"] == [
            {"role": "assistant", "content": "intro h"}]

    def test_solved_path_messages_walks_root_to_solved(self):
        s, c1, c2 = self._build_solved_tree()
        msgs = s.solved_path_messages()
        # Order: root (no msgs) → c1 (intro) → c2 (exact)
        contents = [m["content"] for m in msgs]
        assert contents == ["intro h", "exact h"]

    def test_solved_path_messages_falls_back_when_unsolved(self):
        from prover.unified.search_driver import SharedSearchState
        s = SharedSearchState(root_env_id=0, root_goals=["G"])
        c1 = s.expand(parent_node_id=0, tactic="bad",
                       new_env_id=1, remaining_goals=["G'"],
                       is_complete=False)
        s.nodes[c1].messages.append({"role": "assistant", "content": "bad"})
        s.nodes[c1].score = 5.0
        msgs = s.solved_path_messages()
        # Falls back to deepest+highest-score path even without solve.
        assert len(msgs) == 1
        assert msgs[0]["content"] == "bad"


# ─────────────────────────────────────────────────────────────────────────
# 7. PRESETS contains all three tree-search profiles
# ─────────────────────────────────────────────────────────────────────────

class TestProfilesUpgrade:
    def test_mcts_in_active_presets(self):
        from prover.unified import PRESETS, get_profile
        assert "mcts" in PRESETS
        p = get_profile("mcts")
        assert p.search.kind == "ucb"
        assert p.search.ucb_c == pytest.approx(1.414, abs=1e-3)

    def test_best_first_in_active_presets(self):
        from prover.unified import get_profile
        p = get_profile("best_first")
        assert p.search.kind == "best_first"

    def test_beam_in_active_presets(self):
        from prover.unified import get_profile
        p = get_profile("beam")
        assert p.search.kind == "beam"
        assert p.search.beam_width == 8

    def test_experimental_presets_now_empty(self):
        from prover.unified import EXPERIMENTAL_PRESETS
        assert EXPERIMENTAL_PRESETS == {}

    def test_run_unified_help_advertises_mcts(self):
        """The CLI script's docstring must list mcts as an example."""
        import run_unified
        assert "mcts" in (run_unified.__doc__ or "").lower()


# ─────────────────────────────────────────────────────────────────────────
# 8. UnifiedResult.search_tree → meta.search_tree on save
# ─────────────────────────────────────────────────────────────────────────

class TestUnifiedResultSearchTreeSave:
    def test_save_unified_attaches_search_tree(self, tmp_path):
        from prover.unified.runner import UnifiedResult
        from agent.runtime.agent_loop import LoopResult, LoopMessage

        loop = LoopResult(
            content="proved",
            proof_code=":= by simp",
            messages=[
                LoopMessage(role="user", content="Prove"),
                LoopMessage(role="assistant",
                            content="```lean\n:= by simp\n```"),
            ],
            turns_used=1,
            total_tokens=10,
            stopped_reason="proof_found",
        )
        tree = {
            "kind": "ucb", "root_node_id": 0,
            "solved_node_id": 1, "total_nodes": 2, "max_depth": 1,
            "nodes": [
                {"node_id": 0, "parent_id": None, "tactic": None,
                 "depth": 0, "status": "open",
                 "visit_count": 1, "success_count": 1,
                 "score": 0.0, "messages": []},
                {"node_id": 1, "parent_id": 0, "tactic": "simp",
                 "depth": 1, "status": "solved",
                 "visit_count": 1, "success_count": 1,
                 "score": 12.0,
                 "messages": [{"role": "assistant", "content": "simp"}]},
            ],
        }

        ur = UnifiedResult(
            profile_name="mcts", success=True,
            proof_code=":= by simp", loop_result=loop,
            search_tree=tree,
        )

        out = tmp_path / "task01"
        ur.save_unified(str(out), problem_id="task01",
                          model="qwen3", system_prompt="prove")

        # 1. Exactly one file — dialog.json (the dialog.json-only contract)
        assert sorted(p.name for p in out.iterdir()) == [DIALOG_FILENAME]

        d = load_dialog(out / DIALOG_FILENAME)
        # 2. schema 3.0
        assert d["schema_version"] == "3.0"
        # 3. search_tree under meta
        assert "search_tree" in d["meta"]
        assert d["meta"]["search_tree"]["kind"] == "ucb"
        assert d["meta"]["search_tree"]["solved_node_id"] == 1
        # 4. messages still carry the linear "solved path" view
        assert len(d["messages"]) >= 1
        # 5. validates clean
        assert validate_dialog(d) == []

    def test_save_unified_without_search_tree_unchanged(self, tmp_path):
        """Linear profile run — no search_tree field, no meta.search_tree.
        Confirms back-compat: 2.0-style dialogs are still produced when
        search.kind == 'none'."""
        from prover.unified.runner import UnifiedResult
        from agent.runtime.agent_loop import LoopResult, LoopMessage

        loop = LoopResult(
            content="proved",
            proof_code=":= by trivial",
            messages=[
                LoopMessage(role="user", content="Prove True"),
                LoopMessage(role="assistant",
                            content="```lean\n:= by trivial\n```"),
            ],
            turns_used=1,
            total_tokens=8,
            stopped_reason="proof_found",
        )
        ur = UnifiedResult(
            profile_name="whole_proof", success=True,
            proof_code=":= by trivial", loop_result=loop,
            search_tree=None,
        )

        out = tmp_path / "task02"
        ur.save_unified(str(out), problem_id="task02", model="qwen3",
                          system_prompt="prove")

        d = load_dialog(out / DIALOG_FILENAME)
        assert d["schema_version"] == "3.0"
        assert "search_tree" not in d["meta"], \
            "linear profile must not emit meta.search_tree"


# ─────────────────────────────────────────────────────────────────────────
# 9. dialog.json-only contract still holds for tree-search runs
# ─────────────────────────────────────────────────────────────────────────

class TestDialogJsonOnlyContract:
    def test_only_dialog_json_for_tree_search(self, tmp_path):
        """No sidecar files — even with the search_tree block the run
        must produce exactly task_dir/dialog.json and nothing else."""
        from prover.unified.runner import UnifiedResult
        from agent.runtime.agent_loop import LoopResult, LoopMessage

        ur = UnifiedResult(
            profile_name="mcts", success=False,
            proof_code="", loop_result=LoopResult(
                content="", proof_code="",
                messages=[LoopMessage(role="user", content="x")],
                stopped_reason="search_exhausted",
            ),
            search_tree={
                "kind": "ucb", "root_node_id": 0,
                "solved_node_id": None, "total_nodes": 1, "max_depth": 0,
                "nodes": [{
                    "node_id": 0, "parent_id": None, "tactic": None,
                    "depth": 0, "status": "open",
                    "visit_count": 1, "success_count": 0,
                    "score": 0.0, "messages": []}],
            },
        )

        d = tmp_path / "trace"
        ur.save_unified(str(d), problem_id="trace")
        assert sorted(p.name for p in d.iterdir()) == [DIALOG_FILENAME]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
