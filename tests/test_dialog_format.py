"""tests/test_dialog_format.py — Tests for the self-contained format

Verifies:
  1. ``DialogBuilder.build()`` returns the wrapped object with
     ``schema_version``, ``meta``, ``messages``, ``result``.
  2. Round-trip: build → save → load preserves the schema.
  3. Adapters from each legacy format produce wrapped dialogs.
  4. ``messages_of`` / ``meta_of`` / ``result_of`` work on both wrapped
     and legacy plain-list inputs.
  5. SFT export auto-extracts system prompt from ``meta.system_prompt``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

from agent.persistence.dialog_format import (
    SCHEMA_VERSION,
    DialogBuilder,
    Message,
    ToolCall,
    save_dialog,
    load_dialog,
    validate_dialog,
    messages_of, meta_of, result_of,
    split_dialog_at_markers,
    is_tool_response_user_msg,
    strip_tool_response_wrapper,
)
from agent.persistence.dialog_adapters import (
    from_loop_messages,
    from_trajectory,
    from_proof_trace,
    to_openai_messages,
)
from agent.persistence.sft_export import (
    QWEN3_PRESET,
    AGENTCPM_PRESET,
    OPENAI_PRESET,
    dialog_to_sft_sample,
)


# ────────────────────────────────────────────────────────────────────────
# 1. Builder + I/O
# ────────────────────────────────────────────────────────────────────────

class TestBuilderAndIO:
    def test_minimal_wrapped(self):
        b = DialogBuilder()
        b.set_meta(
            problem_id="t01",
            theorem_statement="theorem t (n : ℕ) : n + 0 = n",
            system_prompt="You are a Lean prover.",
            model="qwen3",
        )
        b.add_user("Prove the theorem")
        b.add_assistant_proof(
            proof_code=":= by simp",
            thought="Use simp.",
        )
        b.set_result(success=True, total_tokens=42)
        d = b.build()

        # Wrapped object structure
        assert d["schema_version"] == SCHEMA_VERSION
        assert "meta" in d and "messages" in d and "result" in d

        # Meta carries everything the agent was given
        assert d["meta"]["problem_id"] == "t01"
        assert d["meta"]["system_prompt"] == "You are a Lean prover."
        assert d["meta"]["model"] == "qwen3"
        assert "started_at" in d["meta"]
        assert "finished_at" in d["meta"]

        # Messages list is AgentCPM-shape
        assert len(d["messages"]) == 2
        assert d["messages"][0]["role"] == "user"
        assert d["messages"][1]["role"] == "assistant"
        assert d["messages"][1]["thought"] == "Use simp."
        assert "```lean" in d["messages"][1]["content"]

        # Result captures outcome
        assert d["result"]["success"] is True
        assert d["result"]["total_tokens"] == 42

    def test_save_load_round_trip(self):
        b = DialogBuilder()
        b.set_meta(model="qwen3", system_prompt="prove things")
        b.add_user("hello")
        b.add_assistant("hi", thought="being friendly")
        b.set_result(success=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dialog.json")
            save_dialog(b, path)
            loaded = load_dialog(path)
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["meta"]["model"] == "qwen3"
        assert loaded["messages"][0]["content"] == "hello"
        assert loaded["result"]["success"] is True

    def test_save_load_legacy_list_upgraded(self):
        # Saving a plain list still works — it's auto-wrapped on save.
        legacy = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dialog.json")
            save_dialog(legacy, path)
            loaded = load_dialog(path)
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["messages"] == legacy

    def test_load_legacy_plain_list_file(self):
        # Files written under schema 1.0 (raw list) auto-upgrade on load.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dialog.json")
            with open(path, "w") as f:
                json.dump([{"role": "user", "content": "q"}], f)
            loaded = load_dialog(path)
        assert loaded["schema_version"] == "1.0"
        assert loaded["messages"][0]["content"] == "q"

    def test_tool_call_round_trip_inside_wrapped(self):
        b = DialogBuilder()
        b.set_meta(problem_id="t02", model="qwen3")
        b.add_user("Prove n + 0 = n")
        b.add_assistant_thinking(
            "Look up Nat.add_zero.",
            followed_by_tool_call=(
                "premise_search",
                {"query": "n + 0 = n"}, "mathlib"),
        )
        b.add_tool_response(
            name="premise_search",
            content='[{"name": "Nat.add_zero"}]',
        )
        d = b.build()

        assert validate_dialog(d) == []
        # arguments is a JSON-string per AgentCPM convention
        tcs = d["messages"][1]["tool_calls"]
        assert isinstance(tcs[0]["function"]["arguments"], str)
        # Tool response matched by id
        assert d["messages"][2]["tool_call_id"] == tcs[0]["id"]


# ────────────────────────────────────────────────────────────────────────
# 2. Accessors work on both forms
# ────────────────────────────────────────────────────────────────────────

class TestAccessors:
    def test_messages_of_wrapped(self):
        d = {"meta": {}, "messages": [{"role": "user", "content": "q"}],
             "result": {}}
        assert messages_of(d) == [{"role": "user", "content": "q"}]

    def test_messages_of_plain_list(self):
        legacy = [{"role": "user", "content": "q"}]
        assert messages_of(legacy) == legacy

    def test_meta_of_plain_list_returns_empty(self):
        assert meta_of([{"role": "user", "content": "q"}]) == {}

    def test_result_of_wrapped(self):
        d = {"meta": {}, "messages": [], "result": {"success": True}}
        assert result_of(d)["success"] is True


# ────────────────────────────────────────────────────────────────────────
# 3. Validation
# ────────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_clean_wrapped_dialog(self):
        b = DialogBuilder()
        b.add_user("q")
        b.add_assistant("a")
        assert validate_dialog(b.build()) == []

    def test_clean_plain_list(self):
        assert validate_dialog([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]) == []

    def test_arguments_must_be_string(self):
        bad = {"schema_version": "2.0", "meta": {}, "result": {},
               "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "",
             "tool_calls": [{
                 "id": "c1",
                 "function": {"name": "foo",
                              "arguments": {"x": 1}},  # ← dict not str
                 "server_id": "default",
             }]},
            {"role": "tool", "tool_call_id": "c1", "name": "foo",
             "content": "", "server_id": "default"},
        ]}
        codes = [i.code for i in validate_dialog(bad)]
        assert "tool_args_not_string" in codes


# ────────────────────────────────────────────────────────────────────────
# 4. Adapters return wrapped dialogs when wrapped=True
# ────────────────────────────────────────────────────────────────────────

class TestLoopAdapter:
    def test_legacy_default_returns_messages(self):
        history = [
            {"role": "user", "content": "Prove foo"},
            {"role": "assistant", "content": "ok"},
        ]
        out = from_loop_messages(history)
        # Default returns the message list (legacy compatible)
        assert isinstance(out, list)
        assert out[0]["content"] == "Prove foo"

    def test_wrapped_returns_dialog_dict(self):
        history = [
            {"role": "user", "content": "Prove foo"},
            {"role": "assistant", "content": "ok"},
        ]
        out = from_loop_messages(
            history, wrapped=True,
            meta={"problem_id": "p", "system_prompt": "be a prover"},
            result={"success": True},
        )
        assert isinstance(out, dict)
        assert out["schema_version"] == SCHEMA_VERSION
        assert out["meta"]["problem_id"] == "p"
        assert out["meta"]["system_prompt"] == "be a prover"
        assert out["result"]["success"] is True

    def test_claude_api_blocks_unfolded(self):
        history = [
            {"role": "user", "content": "Prove n + 0 = n"},
            {"role": "assistant",
             "content": [
                 {"type": "text", "text": "Searching."},
                 {"type": "tool_use", "id": "tu1",
                  "name": "premise_search",
                  "input": {"query": "Nat.add_zero"}},
             ]},
            {"role": "user",
             "content": [
                 {"type": "tool_result", "tool_use_id": "tu1",
                  "content": "[Nat.add_zero]"},
             ]},
        ]
        d = from_loop_messages(history, wrapped=True)
        assert validate_dialog(d) == []
        msgs = messages_of(d)
        assert msgs[1]["tool_calls"][0]["function"]["name"] \
               == "premise_search"
        assert msgs[2]["role"] == "tool"


class TestTrajectoryAdapter:
    def _make_traj(self):
        class _R:
            def __init__(self, scalar=0.0, level="L1", err="",
                         goals=1, terminal=False, hint=""):
                self.scalar = scalar
                self.verification_level = level
                self.error_class = err
                self.goals_remaining = goals
                self.goals_closed = 0
                self.is_terminal = terminal
                self.fix_hint = hint
                self.raw_feedback = ""

        class _T:
            def __init__(self, obs, action, reward):
                self.observation = obs
                self.action = action
                self.reward = reward

        class _Term:
            value = "success"

        class _Traj:
            def __init__(self):
                self.problem_id = "rl01"
                self.theorem_statement = "n + 0 = n"
                self.success = True
                self.total_tokens = 50
                self.wall_time_s = 1.5
                self.total_reward = 1.0
                self.metadata = {}
                self.termination = _Term()
                self.turns = [
                    _T("Prove: n + 0 = n", "rfl",
                       _R(0.0, "L1", "tactic_failed", 1, False)),
                    _T("Try again", "by simp",
                       _R(1.0, "L2", "", 0, True)),
                ]
        return _Traj()

    def test_wrapped_dialog_carries_meta_and_result(self):
        d = from_trajectory(self._make_traj(), wrapped=True)
        assert d["meta"]["problem_id"] == "rl01"
        assert d["meta"]["theorem_statement"] == "n + 0 = n"
        assert d["result"]["success"] is True
        assert d["result"]["total_attempts"] == 2
        assert d["result"]["total_tokens"] == 50
        assert validate_dialog(d) == []


class TestProofTraceAdapter:
    def _make_trace(self):
        class _Status:
            def __init__(self, v): self.value = v

        class _Att:
            def __init__(self, proof, success):
                self.generated_proof = proof
                self.lean_result = _Status(
                    "success" if success else "lean_error")
                self.lean_errors = []
                self.lean_stderr = ""
                self.retrieved_premises = []

        class _Trace:
            def __init__(self):
                self.problem_id = "pt01"
                self.problem_name = "test problem"
                self.theorem_statement = "theorem t : True"
                self.natural_language = ""
                self.solved = True
                self.total_attempts = 1
                self.total_tokens = 10
                self.total_duration_ms = 100
                self.correct_count = 1
                self.successful_proof = ":= trivial"
                self.error_distribution = {}
                self.strategy_path = []
                self.config_snapshot = {}
                self.trace_id = "abc"
                self.attempts = [_Att(":= trivial", True)]
        return _Trace()

    def test_wrapped(self):
        d = from_proof_trace(self._make_trace(), wrapped=True,
                             meta={"model": "qwen3"})
        assert d["meta"]["problem_id"] == "pt01"
        assert d["meta"]["model"] == "qwen3"
        assert d["meta"]["extra"]["trace_id"] == "abc"
        assert d["result"]["success"] is True
        assert d["result"]["successful_proof"] == ":= trivial"


# ────────────────────────────────────────────────────────────────────────
# 5. SFT export uses meta.system_prompt automatically
# ────────────────────────────────────────────────────────────────────────

class TestSFTExportAutoSystemPrompt:
    def _build(self, with_system_prompt=True):
        b = DialogBuilder()
        if with_system_prompt:
            b.set_meta(system_prompt="You are a Lean 4 theorem prover.")
        b.add_user("Prove n + 0 = n")
        b.add_assistant_thinking(
            "Search for Nat.add_zero.",
            followed_by_tool_call=("premise_search",
                                   {"query": "Nat.add_zero"}, "mathlib"),
        )
        b.add_tool_response("premise_search",
                            '[{"name":"Nat.add_zero"}]')
        b.add_assistant_proof(
            proof_code=":= by simp",
            thought="Direct.",
        )
        return b.build()

    def test_qwen3_uses_meta_system_prompt(self):
        d = self._build(with_system_prompt=True)
        sample = dialog_to_sft_sample(d, preset="qwen3")
        assert "<|im_start|>system" in sample["text"]
        assert "You are a Lean 4 theorem prover." in sample["text"]
        # And all the special-token wrapping still applied
        assert "<think>" in sample["text"]
        assert "<tool_call>" in sample["text"]
        assert "<tool_response>" in sample["text"]
        assert "<code>" in sample["text"]

    def test_explicit_override_wins(self):
        d = self._build(with_system_prompt=True)
        sample = dialog_to_sft_sample(
            d, preset="qwen3",
            system_prompt="Custom override prompt")
        assert "Custom override prompt" in sample["text"]
        # Meta version not present
        assert "You are a Lean 4 theorem prover." not in sample["text"]

    def test_legacy_plain_list_still_works(self):
        # SFT export accepts the old plain-list shape too
        legacy_messages = [
            {"role": "user", "content": "Prove True"},
            {"role": "assistant",
             "content": "```lean\n:= trivial\n```"},
        ]
        sample = dialog_to_sft_sample(
            legacy_messages, preset="qwen3",
            system_prompt="be brief")
        assert "be brief" in sample["text"]

    def test_drop_thoughts(self):
        d = self._build()
        sample = dialog_to_sft_sample(
            d, preset="qwen3", drop_thoughts=True)
        assert "<think>" not in sample["text"]
        assert "</think>" not in sample["text"]


# ────────────────────────────────────────────────────────────────────────
# 6. Helpers (unchanged from prior version)
# ────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_strip_tool_response_wrapper(self):
        s = "<tool_response>\nhello\n</tool_response>"
        assert strip_tool_response_wrapper(s) == "hello"

    def test_is_tool_response_user_msg(self):
        assert is_tool_response_user_msg({
            "role": "user",
            "content": "<tool_response>x</tool_response>",
        })
        assert not is_tool_response_user_msg({
            "role": "user", "content": "regular question"})

    def test_split_dialog_at_markers(self):
        full = [
            {"role": "user", "content": "q1"},
            {"role": "__CONTEXT_SPLIT__",
             "next_history_segment": [
                 {"role": "user", "content": "[summary]"},
             ]},
            {"role": "assistant", "content": "a2"},
        ]
        segments = split_dialog_at_markers(full)
        assert len(segments) == 2
        assert segments[1][0]["content"] == "[summary]"

    def test_to_openai_messages_strips_thoughts(self):
        b = DialogBuilder()
        b.add_user("q")
        b.add_assistant("a", thought="hidden reasoning")
        msgs = to_openai_messages(b.build())
        assert msgs[1]["role"] == "assistant"
        assert "thought" not in msgs[1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
