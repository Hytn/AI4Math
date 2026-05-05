"""tests/test_llm_autoformalizer.py — V5 LLM autoformalizer.

Pins the contract added in :file:`prover/unified/llm_autoformalizer.py`:

  * ``make_llm_autoformalizer`` validates inputs, calls ``llm.generate``
    with the right shape, post-processes the response (strips ```lean
    fences, trims whitespace), and raises on empty / mal-formed output
    so the registered-autoformalizer machinery in
    :file:`prover/unified/tools_infra.py` falls back to the heuristic.
  * ``register_llm_autoformalizer`` integrates with the existing
    ``register_autoformalizer`` registry.
  * The ``NLExistenceBridgeTool`` honours the registered LLM
    autoformalizer over the heuristic when both are available.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from prover.unified.llm_autoformalizer import (
    make_llm_autoformalizer, make_llm_autoformalizer_async,
    register_llm_autoformalizer,
    DEFAULT_AUTOFORMALIZER_SYSTEM_PROMPT,
    _strip_lean_fence,
)

# ─────────────────────────────────────────────────────────────────────
# Mock LLM helpers
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FakeResp:
    content: str

class StubLLM:
    """Records every generate() call and returns a fixed response."""
    def __init__(self, response_content: str):
        self.response = response_content
        self.calls: list[dict] = []

    def generate(self, system, user, temperature, tools, max_tokens):
        self.calls.append({
            "system": system, "user": user,
            "temperature": temperature, "tools": tools,
            "max_tokens": max_tokens,
        })
        return FakeResp(content=self.response)

class CrashingLLM:
    def generate(self, *a, **kw):
        raise RuntimeError("network down")

# ─────────────────────────────────────────────────────────────────────
# 1. make_llm_autoformalizer — input validation
# ─────────────────────────────────────────────────────────────────────

class TestInputValidation:
    def test_none_llm_rejected(self):
        with pytest.raises(ValueError):
            make_llm_autoformalizer(None)

    def test_llm_without_generate_rejected(self):
        with pytest.raises(TypeError):
            make_llm_autoformalizer("not an LLM")  # type: ignore[arg-type]

    def test_empty_nl_raises(self):
        fn = make_llm_autoformalizer(
            StubLLM("theorem ai4math_q : ∃ n : ℕ, True := sorry"))
        with pytest.raises(RuntimeError):
            fn("", "integer")
        with pytest.raises(RuntimeError):
            fn("   ", "integer")

# ─────────────────────────────────────────────────────────────────────
# 2. Happy path — correct content extraction
# ─────────────────────────────────────────────────────────────────────

class TestHappyPath:
    def test_returns_lean_theorem(self):
        llm = StubLLM(
            "theorem ai4math_q : ∃ (n : ℕ), n ^ 2 > 100")
        fn = make_llm_autoformalizer(llm)
        out = fn("Find n with n^2 > 100", "integer")
        assert "theorem" in out
        assert "ai4math_q" in out

    def test_strips_lean_fence(self):
        llm = StubLLM(
            "Here is the formalization:\n"
            "```lean\ntheorem ai4math_q : ∃ (n : ℕ), n = 0\n```\n"
            "Hope this helps!")
        fn = make_llm_autoformalizer(llm)
        out = fn("trivial existence question", "integer")
        assert out == "theorem ai4math_q : ∃ (n : ℕ), n = 0"

    def test_strips_lean4_fence_variant(self):
        llm = StubLLM(
            "```lean4\ntheorem ai4math_q : ∃ (k : ℕ), k = 1\n```")
        fn = make_llm_autoformalizer(llm)
        out = fn("nl problem", "integer")
        assert out == "theorem ai4math_q : ∃ (k : ℕ), k = 1"

    def test_default_temperature_is_zero(self):
        llm = StubLLM("theorem ai4math_q : True")
        fn = make_llm_autoformalizer(llm)
        fn("nl", "integer")
        assert llm.calls[0]["temperature"] == 0.0

    def test_uses_default_system_prompt(self):
        llm = StubLLM("theorem ai4math_q : True")
        fn = make_llm_autoformalizer(llm)
        fn("nl", "integer")
        assert llm.calls[0]["system"] == DEFAULT_AUTOFORMALIZER_SYSTEM_PROMPT

    def test_custom_system_prompt_honoured(self):
        llm = StubLLM("theorem ai4math_q : True")
        fn = make_llm_autoformalizer(
            llm, system_prompt="custom prompt")
        fn("nl", "integer")
        assert llm.calls[0]["system"] == "custom prompt"

    def test_user_message_includes_nl_and_type(self):
        llm = StubLLM("theorem ai4math_q : True")
        fn = make_llm_autoformalizer(llm)
        fn("Compute the value of 2 + 3.", "integer")
        user = llm.calls[0]["user"]
        assert "Compute the value of 2 + 3" in user
        assert "integer" in user

# ─────────────────────────────────────────────────────────────────────
# 3. Failure modes
# ─────────────────────────────────────────────────────────────────────

class TestFailureModes:
    def test_empty_response_raises(self):
        fn = make_llm_autoformalizer(StubLLM(""))
        with pytest.raises(RuntimeError):
            fn("nl", "integer")

    def test_response_without_theorem_keyword_raises(self):
        fn = make_llm_autoformalizer(StubLLM("Sorry, I cannot translate."))
        with pytest.raises(RuntimeError):
            fn("nl", "integer")

    def test_llm_exception_wrapped(self):
        fn = make_llm_autoformalizer(CrashingLLM())
        with pytest.raises(RuntimeError) as ei:
            fn("nl", "integer")
        # The wrapped error message must mention "LLM autoformalizer"
        # so logs are searchable.
        assert "autoformaliz" in str(ei.value).lower()

    def test_passthrough_mode_re_raises(self):
        fn = make_llm_autoformalizer(
            CrashingLLM(), timeout_passthrough=True)
        with pytest.raises(RuntimeError) as ei:
            fn("nl", "integer")
        assert "network down" in str(ei.value)

# ─────────────────────────────────────────────────────────────────────
# 4. _strip_lean_fence helper
# ─────────────────────────────────────────────────────────────────────

class TestStripLeanFence:
    def test_no_fence_passthrough(self):
        assert _strip_lean_fence("theorem t : True") == "theorem t : True"

    def test_lean_fence(self):
        assert _strip_lean_fence(
            "```lean\nx := 1\n```") == "x := 1"

    def test_lean4_fence(self):
        assert _strip_lean_fence("```lean4\nfoo\n```") == "foo"

    def test_unfenced_content_with_text(self):
        assert _strip_lean_fence(
            "Sure!\n```lean\nfoo\n```\n").strip() == "foo"

    def test_empty_input(self):
        assert _strip_lean_fence("") == ""
        assert _strip_lean_fence(None or "") == ""

# ─────────────────────────────────────────────────────────────────────
# 5. register_llm_autoformalizer integration
# ─────────────────────────────────────────────────────────────────────

class TestRegistration:
    def teardown_method(self):
        # Always deregister after each test to keep the global state
        # clean for other tests in the suite.
        from prover.unified.tools_infra import register_autoformalizer
        register_autoformalizer(None)

    def test_register_then_get(self):
        from prover.unified.tools_infra import _get_autoformalizer
        llm = StubLLM("theorem ai4math_q : True")
        fn = register_llm_autoformalizer(llm)
        assert _get_autoformalizer() is fn

    def test_registered_fn_callable(self):
        llm = StubLLM("theorem ai4math_q : ∃ n, n = 0")
        fn = register_llm_autoformalizer(llm)
        assert callable(fn)
        assert "theorem" in fn("nl", "integer")

    def test_deregister_clears_registry(self):
        from prover.unified.tools_infra import (
            register_autoformalizer, _get_autoformalizer)
        llm = StubLLM("theorem ai4math_q : True")
        register_llm_autoformalizer(llm)
        register_autoformalizer(None)
        assert _get_autoformalizer() is None

# ─────────────────────────────────────────────────────────────────────
# 6. End-to-end: NLExistenceBridgeTool prefers LLM over heuristic
# ─────────────────────────────────────────────────────────────────────

class TestNLExistenceBridgeIntegration:
    def teardown_method(self):
        from prover.unified.tools_infra import register_autoformalizer
        register_autoformalizer(None)

    def test_tool_uses_llm_when_registered(self):
        from prover.unified.tools_infra import NLExistenceBridgeTool
        from agent.tools.base import ToolContext

        llm = StubLLM("theorem ai4math_q : ∃ (n : ℕ), n = 42")
        register_llm_autoformalizer(llm)

        tool = NLExistenceBridgeTool()
        result = asyncio.run(tool.execute(
            input={"nl_problem": "Find n with n = 42",
                    "answer_type": "integer"},
            ctx=ToolContext(agent_name="t", theorem_statement="")))
        # Tool should report success and the lean_statement should
        # contain the LLM's text.
        # Find the underlying response payload — ToolResult exposes
        # .content; in this project it's the raw string returned to LLM.
        payload = getattr(result, "content", None) or getattr(
            result, "data", None)
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = {"lean_statement": payload}
        else:
            parsed = payload or {}
        assert "lean_statement" in parsed
        assert "n = 42" in parsed["lean_statement"]
        assert parsed.get("autoformalizer") == "registered"

    def test_tool_falls_back_to_heuristic_on_llm_failure(self):
        from prover.unified.tools_infra import NLExistenceBridgeTool
        from agent.tools.base import ToolContext

        register_llm_autoformalizer(CrashingLLM())

        tool = NLExistenceBridgeTool()
        result = asyncio.run(tool.execute(
            input={"nl_problem": "Find the smallest n with n^2 > 100",
                    "answer_type": "integer"},
            ctx=ToolContext(agent_name="t", theorem_statement="")))
        payload = getattr(result, "content", None) or getattr(
            result, "data", None)
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = {"lean_statement": payload}
        else:
            parsed = payload or {}
        # Heuristic kicked in — the smallest pattern adds a ≤ bound.
        assert "lean_statement" in parsed
        assert parsed.get("autoformalizer") == "heuristic"
        assert "theorem ai4math_q" in parsed["lean_statement"]

# ─────────────────────────────────────────────────────────────────────
# 7. Async variant
# ─────────────────────────────────────────────────────────────────────

class TestAsyncFactory:
    def test_async_callable_works_with_sync_llm(self):
        llm = StubLLM("theorem ai4math_q : ∃ (n : ℕ), n = 0")
        fn = make_llm_autoformalizer_async(llm)
        out = asyncio.run(fn("nl", "integer"))
        assert "theorem" in out
        assert "n = 0" in out
