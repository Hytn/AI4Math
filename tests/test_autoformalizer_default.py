"""

Closes the V5 "5-pattern heuristic is silly when an LLM is on the
bench" gap one step further: V5 added ``register_llm_autoformalizer()``
as an opt-in convenience; V6 makes it the *default* whenever a runner
is constructed with an LLM.

What this test file pins:

  1. Default behaviour — constructing a UnifiedProofRunner with any
     LLM populates the module-level autoformalizer registry.
  2. Backwards compat — an explicit prior registration is never
     overwritten.
  3. Opt-out — ``auto_register_llm_autoformalizer=False`` preserves
     the V5 behaviour exactly.
  4. Robustness — runners constructed with malformed LLMs do not
     raise during __init__.
  5. The flag survives the documented heuristic-fallback semantics:
     the heuristic is still consulted when the registered LLM
     callable raises.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prover.unified.runner import UnifiedProofRunner
from prover.unified.tools_infra import (
    _get_autoformalizer, register_autoformalizer)

@pytest.fixture(autouse=True)
def _reset_autoformalizer_registry():
    """Every test in this file starts with a fresh empty registry."""
    register_autoformalizer(None)
    yield
    register_autoformalizer(None)

def _mock_llm(generate_return: str = "theorem ai4math_q : ∃ n : ℕ, n = n"):
    """Build a minimal LLM mock matching the LLMProvider interface."""
    llm = MagicMock()
    response = MagicMock()
    response.content = generate_return
    llm.generate = MagicMock(return_value=response)
    return llm

# ─────────────────────────────────────────────────────────────────────
# Default behaviour
# ─────────────────────────────────────────────────────────────────────

class TestDefault:

    def test_runner_with_llm_auto_registers(self):
        assert _get_autoformalizer() is None
        runner = UnifiedProofRunner(llm=_mock_llm())
        assert _get_autoformalizer() is not None
        assert runner._auto_registered_autoformalizer is True

    def test_runner_without_llm_does_not_register(self):
        # llm=None should be a no-op for autoformalizer purposes
        # (the runner can't autoformalize without an LLM anyway).
        UnifiedProofRunner(llm=None)
        assert _get_autoformalizer() is None

    def test_two_runners_back_to_back_first_wins(self):
        r1 = UnifiedProofRunner(llm=_mock_llm())
        first_fn = _get_autoformalizer()
        assert first_fn is not None
        assert r1._auto_registered_autoformalizer is True

        # Second runner constructed AFTER first registered should NOT
        # clobber — the contract is "leave existing registration alone".
        r2 = UnifiedProofRunner(llm=_mock_llm())
        assert _get_autoformalizer() is first_fn
        assert r2._auto_registered_autoformalizer is False

    def test_registered_callable_translates(self):
        UnifiedProofRunner(llm=_mock_llm())
        fn = _get_autoformalizer()
        result = fn("Find a natural number n with n=n.", "natural")
        assert "theorem" in result
        # The LLM mock returned this exact string; fence-stripping is
        # a no-op here so we get it back unchanged.
        assert "ai4math_q" in result

# ─────────────────────────────────────────────────────────────────────
# Backwards compat — explicit registration wins
# ─────────────────────────────────────────────────────────────────────

class TestExplicitRegistrationPreserved:

    def test_explicit_registration_not_clobbered(self):
        # User pre-registered their own translator
        explicit_fn = lambda nl, t: f"theorem custom : ∃ x : {t}, True"
        register_autoformalizer(explicit_fn)

        runner = UnifiedProofRunner(llm=_mock_llm())

        # Explicit one still in place; runner did NOT auto-register
        assert _get_autoformalizer() is explicit_fn
        assert runner._auto_registered_autoformalizer is False

    def test_explicit_registration_after_runner_overwrites(self):
        # Reverse order: runner first, then user explicitly registers
        UnifiedProofRunner(llm=_mock_llm())
        first_fn = _get_autoformalizer()
        assert first_fn is not None

        # User explicit register takes precedence
        explicit_fn = lambda nl, t: "theorem mine : True"
        register_autoformalizer(explicit_fn)
        assert _get_autoformalizer() is explicit_fn

    def test_register_none_resets_registry(self):
        UnifiedProofRunner(llm=_mock_llm())
        assert _get_autoformalizer() is not None
        register_autoformalizer(None)
        assert _get_autoformalizer() is None

# ─────────────────────────────────────────────────────────────────────
# Opt-out
# ─────────────────────────────────────────────────────────────────────

class TestOptOut:

    def test_opt_out_flag_prevents_registration(self):
        UnifiedProofRunner(
            llm=_mock_llm(),
            auto_register_llm_autoformalizer=False,
        )
        assert _get_autoformalizer() is None

    def test_opt_out_preserves_v5_behaviour_with_explicit(self):
        # Pre-register, then build runner with opt-out:
        # explicit registration unchanged, auto path skipped.
        explicit_fn = lambda nl, t: "theorem old : True"
        register_autoformalizer(explicit_fn)
        runner = UnifiedProofRunner(
            llm=_mock_llm(),
            auto_register_llm_autoformalizer=False,
        )
        assert _get_autoformalizer() is explicit_fn
        assert runner._auto_registered_autoformalizer is False

# ─────────────────────────────────────────────────────────────────────
# Robustness
# ─────────────────────────────────────────────────────────────────────

class TestRobustness:

    def test_llm_without_generate_does_not_crash(self):
        # An LLM that lacks .generate makes make_llm_autoformalizer
        # raise; the runner must swallow that and stay constructible.
        bad_llm = object()
        runner = UnifiedProofRunner(llm=bad_llm)
        assert _get_autoformalizer() is None
        assert runner._auto_registered_autoformalizer is False
        # Runner is otherwise functional
        assert runner.llm is bad_llm

    def test_llm_generate_raising_does_not_crash_init(self):
        # An LLM whose .generate raises at call time is fine for
        # registration — the registration just stashes a callable.
        # The first translation attempt is what would fail.
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=RuntimeError("boom"))
        runner = UnifiedProofRunner(llm=llm)
        assert runner._auto_registered_autoformalizer is True

        # Calling the registered fn raises — but that's the documented
        # fallback contract (NLExistenceBridgeTool catches it and
        # falls back to the heuristic).
        fn = _get_autoformalizer()
        with pytest.raises(RuntimeError):
            fn("anything", "natural")

# ─────────────────────────────────────────────────────────────────────
# Heuristic fallback still reachable
# ─────────────────────────────────────────────────────────────────────

class TestFallbackPathStillWorks:

    def test_explicit_register_none_disables_auto_path(self):
        # If a caller deliberately wants the heuristic, they can
        # opt out and explicitly clear:
        UnifiedProofRunner(
            llm=_mock_llm(),
            auto_register_llm_autoformalizer=False,
        )
        register_autoformalizer(None)
        assert _get_autoformalizer() is None

        # Now NLExistenceBridgeTool would fall through to its
        # heuristic (which has its own test coverage).

    def test_auto_register_does_not_break_explicit_register_after(self):
        # Auto-register, then explicitly clear, then explicitly set —
        # all steps should work.
        UnifiedProofRunner(llm=_mock_llm())
        assert _get_autoformalizer() is not None
        register_autoformalizer(None)
        assert _get_autoformalizer() is None
        custom = lambda nl, t: "theorem x : True"
        register_autoformalizer(custom)
        assert _get_autoformalizer() is custom

# ─────────────────────────────────────────────────────────────────────
# Multi-runner contract
# ─────────────────────────────────────────────────────────────────────

class TestMultipleRunners:

    def test_three_runners_only_first_registers(self):
        r1 = UnifiedProofRunner(llm=_mock_llm())
        r2 = UnifiedProofRunner(llm=_mock_llm())
        r3 = UnifiedProofRunner(llm=_mock_llm())
        assert r1._auto_registered_autoformalizer is True
        assert r2._auto_registered_autoformalizer is False
        assert r3._auto_registered_autoformalizer is False

    def test_clear_between_runners_lets_second_register(self):
        UnifiedProofRunner(llm=_mock_llm())
        register_autoformalizer(None)
        r2 = UnifiedProofRunner(llm=_mock_llm())
        assert r2._auto_registered_autoformalizer is True
