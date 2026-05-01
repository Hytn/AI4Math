"""V6 — proof_loop_legacy.py removal tests.

Closes V5+ gap #4 (file cleanup). The V5 report kept
``prover/pipeline/proof_loop_legacy.py`` because deleting it was
thought to silently change error-handling semantics in
``prover.pipeline.proof_loop.ProofLoop.single_attempt`` — that method's
exception handler used to fall through to ``LegacyLoop`` before
returning a structured ``LLM_ERROR`` ProofAttempt.

V6 deletes the legacy module after auditing:

  * Only one reference site existed (the fallback ``except`` branch).
  * Zero tests covered the legacy fallback path.
  * The user-visible contract (``single_attempt`` always returns a
    ``ProofAttempt``, never raises) is preserved by replacing the
    legacy fallback with a structured ``LLM_ERROR`` return —
    semantically identical to the second-level except branch the V5
    code had as its final safety net.

These tests pin:
  1. The legacy module is gone (``import`` raises ``ModuleNotFoundError``).
  2. ``ProofLoop.single_attempt`` still returns a ProofAttempt when
     the unified runner raises.
  3. The returned attempt has ``LLM_ERROR`` status and a structured
     stderr — same as V5's terminal ``except e2:`` branch produced.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prover.models import AttemptStatus, ProofAttempt
from prover.pipeline.proof_loop import ProofLoop


# ─────────────────────────────────────────────────────────────────────
# Module is gone
# ─────────────────────────────────────────────────────────────────────


class TestLegacyModuleDeleted:

    def test_proof_loop_legacy_import_raises(self):
        with pytest.raises(ModuleNotFoundError):
            import prover.pipeline.proof_loop_legacy  # noqa: F401

    def test_no_repository_references_to_legacy_module(self):
        # The only acceptable references are docstring mentions in
        # proof_loop.py explaining the V6 cleanup. Code-level
        # ``from prover.pipeline.proof_loop_legacy import ...``
        # statements must not exist.
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        offenders = []
        for p in repo.rglob("*.py"):
            if p.name == "test_v6_legacy_cleanup.py":
                continue
            try:
                src = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line in src.splitlines():
                stripped = line.strip()
                # An actual import statement has these prefixes:
                if (stripped.startswith(
                        "from prover.pipeline.proof_loop_legacy") or
                    stripped.startswith(
                        "import prover.pipeline.proof_loop_legacy")):
                    offenders.append(f"{p}: {stripped}")
        assert offenders == [], (
            "Live import references to the deleted legacy module: "
            f"{offenders}")


# ─────────────────────────────────────────────────────────────────────
# ProofLoop external contract preserved
# ─────────────────────────────────────────────────────────────────────


class TestProofLoopFailureContract:
    """``single_attempt`` must always return a ProofAttempt, even when
    the unified runner raises. V5's code path had two except branches
    (legacy fallback + structured failure). V6 collapses to one branch
    that is semantically equivalent to V5's terminal branch."""

    def _make_loop(self):
        # Construct a ProofLoop with mock dependencies. None of them
        # will actually be exercised because we'll force _run_via_unified
        # to raise.
        lean_env = MagicMock()
        llm = MagicMock()
        return ProofLoop(lean_env, llm)

    def test_returns_proof_attempt_when_unified_raises(self):
        loop = self._make_loop()
        with patch.object(loop, "_run_via_unified",
                            side_effect=RuntimeError("boom")):
            problem = MagicMock(theorem_statement="theorem t : True")
            memory = MagicMock(banked_lemmas=[], attempt_history=[])
            result = loop.single_attempt(problem, memory)
        assert isinstance(result, ProofAttempt)

    def test_returned_attempt_has_llm_error_status(self):
        loop = self._make_loop()
        with patch.object(loop, "_run_via_unified",
                            side_effect=RuntimeError("boom")):
            problem = MagicMock(theorem_statement="theorem t : True")
            memory = MagicMock(banked_lemmas=[], attempt_history=[])
            result = loop.single_attempt(problem, memory, attempt_num=42)
        assert result.lean_result == AttemptStatus.LLM_ERROR
        assert result.attempt_number == 42

    def test_returned_attempt_stderr_includes_exception_info(self):
        loop = self._make_loop()
        with patch.object(loop, "_run_via_unified",
                            side_effect=ValueError("specific failure")):
            problem = MagicMock(theorem_statement="theorem t : True")
            memory = MagicMock(banked_lemmas=[], attempt_history=[])
            result = loop.single_attempt(problem, memory)
        assert "specific failure" in result.lean_stderr
        assert "ValueError" in result.lean_stderr or "value" in \
               result.lean_stderr.lower()

    def test_does_not_raise_on_unified_failure(self):
        # Hard contract: callers should not need to wrap single_attempt
        # in their own try. A unified runner crash must not propagate.
        loop = self._make_loop()
        with patch.object(loop, "_run_via_unified",
                            side_effect=Exception("any kind")):
            problem = MagicMock(theorem_statement="theorem t : True")
            memory = MagicMock(banked_lemmas=[], attempt_history=[])
            try:
                loop.single_attempt(problem, memory)
            except Exception:
                pytest.fail(
                    "single_attempt raised; should return LLM_ERROR "
                    "ProofAttempt instead")

    def test_returns_unified_result_on_happy_path(self):
        # Sanity: when _run_via_unified succeeds, its return value
        # is passed through unchanged.
        loop = self._make_loop()
        sentinel = ProofAttempt(attempt_number=1)
        sentinel.lean_result = AttemptStatus.SUCCESS
        with patch.object(loop, "_run_via_unified", return_value=sentinel):
            problem = MagicMock(theorem_statement="theorem t : True")
            memory = MagicMock(banked_lemmas=[], attempt_history=[])
            result = loop.single_attempt(problem, memory)
        assert result is sentinel
        assert result.lean_result == AttemptStatus.SUCCESS


# ─────────────────────────────────────────────────────────────────────
# File-system audit — the file itself must not be present
# ─────────────────────────────────────────────────────────────────────


class TestFilesystemAudit:

    def test_legacy_file_does_not_exist(self):
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        legacy = repo / "prover" / "pipeline" / "proof_loop_legacy.py"
        assert not legacy.exists()
