"""agent/tools/builtin/lean_verify.py — Verify a Lean4 proof snippet


  1. The previous implementation called ``pool.check_proof(code, ...)``
     but ``AsyncLeanPool`` exposes no such method — only
     ``verify_complete(theorem, proof, preamble)``. Every real call hit
     AttributeError and fell into the ``except`` clause, so the
     ``whole_proof_repair`` loop has been running on error strings
     instead of real Lean feedback. This rewrite uses the actual API.

  2. The README advertises "structured AgentFeedback (~100 bits)" but
     it was only wired into the RL sampler path. The main proof path
     was returning ``[str(e)[:300] for e in errors[:5]]`` — basically
     truncated stderr. We now thread the verify result through
     ``ErrorIntelligence`` (when a ``lean_pool`` is present) so the
     LLM sees structured goals + repair candidates + progress score.

The pool argument can be ``AsyncLeanPool`` or its sync wrapper; we
resolve the call site by feature-detecting ``verify_complete``. If
that's missing too we fall through to the prefilter-only path that
v9 used for the no-pool case.
"""
from __future__ import annotations

import inspect
import json
import logging
import re

from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Theorem / proof split — the LLM tends to submit complete theorem
# blocks ("theorem foo : ... := by ..."); ``verify_complete`` wants
# them split. We do a best-effort split here so callers can keep
# passing a single ``code`` string.
# ─────────────────────────────────────────────────────────────────

_PROOF_SPLIT_RE = re.compile(r"(\s*:=\s*by\b|\s*:=\s*(?=fun|⟨|\{|\())", re.DOTALL)

def _split_theorem_and_proof(code: str) -> tuple[str, str]:
    """Best-effort split of ``theorem ... := by ...`` into (statement, proof).

    If the code lacks an obvious split point, the entire string is
    treated as a self-contained theorem block (statement) and the
    proof returns empty — verify_complete will then take the whole
    string as a self-contained Lean source.
    """
    code = code.strip()
    if not code:
        return ("", "")
    m = _PROOF_SPLIT_RE.search(code)
    if not m:
        return (code, "")
    statement = code[:m.start()]
    proof = code[m.start():]
    return (statement, proof)

class LeanVerifyTool(Tool):
    name = "lean_verify"
    description = (
        "Submit a complete Lean4 proof for verification. Returns structured "
        "feedback: success/failure, error messages, remaining goals, and "
        "specific repair suggestions."
    )
    permission = ToolPermission.EXTERNAL
    input_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Lean4 code to verify",
            },
            "quick_check": {
                "type": "boolean",
                "description": "If true, use fast L1 check only (default false)",
            },
        },
        "required": ["code"],
    }

    def __init__(self, lean_pool=None, error_intelligence=None,
                 integrity_strict: bool = False):
        """
        Args:
            lean_pool:  AsyncLeanPool (or sync wrapper). When None, we
                        run the syntax prefilter only.
            error_intelligence:  Optional ErrorIntelligence instance.
                        When provided, failed verifications are re-run
                        through it to produce structured AgentFeedback
                        in the tool result. When None, we still build
                        a minimal AgentFeedback from the verify result
                        so the response shape is consistent.
            integrity_strict:  v12. When True (competition-style
                        benchmarks like PutnamBench/FormalMATH),
                        CRITICAL integrity issues (native_decide,
                        Decidable.decide, set_option maxHeartbeats 0,
                        sorry, axiom, etc.) flip ``verified`` to
                        False even if Lean accepts the proof. When
                        False (mainline benchmarks like miniF2F /
                        ProofNet), they're surfaced as an advisory
                        ``integrity_violations`` field in the result
                        but do not flip ``verified``. False matches
                        Mathlib reality where many legitimate proofs
                        end with ``native_decide``.
        """
        self._pool = lean_pool
        self._ei = error_intelligence
        self._integrity_strict = integrity_strict

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        code = input["code"]

        if not self._pool:
            return self._prefilter_only(code)

        # ── Real verification path ────────────────────────────────
        statement, proof = _split_theorem_and_proof(code)

        try:
            result = await self._call_verify(statement or code, proof)
        except AttributeError as e:
            # Pool exists but doesn't expose verify_complete. Surface a
            # structured error rather than a stack trace.
            return ToolResult.error(
                f"lean_pool does not support verify_complete: {e}. "
                f"Pool kind={type(self._pool).__name__}.")
        except Exception as e:
            return ToolResult.error(f"Verification failed: {e}")

        # `result` is a FullVerifyResult dataclass (engine._core).
        success = bool(getattr(result, "success", False))
        has_sorry = bool(getattr(result, "has_sorry", False))
        elapsed_ms = int(getattr(result, "elapsed_ms", 0))
        goals = list(getattr(result, "goals_remaining", []) or [])
        errors_list = list(getattr(result, "errors", []) or [])
        stderr = str(getattr(result, "stderr", "") or "")

        # ── 
        # Even if Lean accepts the proof, refuse it when the code uses
        # bypasses (axiom, native_decide, set_option maxHeartbeats 0,
        # sorry hidden in nested comments, etc.). Previously this logic
        # existed in prover/verifier/integrity_checker.py but had zero
        # main-path callers;.
        integrity_issues: list[str] = []
        try:
            from prover.verifier.integrity_checker import check_integrity
            report = check_integrity(code)
            if not report.passed:
                integrity_issues = [
                    f"[{i.severity.value}] {i.message}"
                    for i in report.issues if i.severity.value == "critical"
                ]
        except Exception as e:
            logger.debug(f"integrity_checker unavailable/failed: {e}")

        # Build the structured response. We keep the v9 fields
        # ("verified", "goals_remaining", "errors", "sorry_free") for
        # backward compat AND add an "agent_feedback" block carrying
        # the structured signal the LLM actually benefits from.
        sorry_free = (not has_sorry) and ("sorry" not in code) and \
                     ("admit" not in code)

        # In non-strict mode (the default for Mathlib-style benchmarks)
        # they're surfaced advisory in ``integrity_violations`` so
        # downstream consumers can act on them, but Lean's own
        # acceptance + sorry-freedom is the sole source of truth.
        # ``sorry_free`` is checked independently of the integrity
        # checker, so omitting integrity from the verified gate does
        # NOT let sorry-containing proofs through.
        if self._integrity_strict:
            verified = success and sorry_free and not integrity_issues
        else:
            verified = success and sorry_free
        response: dict = {
            "verified": verified,
            "goals_remaining": goals,
            "errors": [self._fmt_error(e) for e in errors_list[:5]],
            "sorry_free": sorry_free,
            "elapsed_ms": elapsed_ms,
        }
        if integrity_issues:
            response["integrity_violations"] = integrity_issues
            if not self._integrity_strict:
                response["integrity_note"] = (
                    "Integrity violations present but verified=True "
                    "because profile.integrity_strict=False. Switch to "
                    "a strict profile to reject these proofs.")

        # Attach AgentFeedback. Use ErrorIntelligence when available
        # for richer repair candidates; otherwise build a minimal
        # feedback directly from the verify result so the schema is
        # always present.
        feedback = self._build_feedback(
            success=verified,
            goals=goals,
            errors_text=self._collect_error_text(errors_list, stderr),
            elapsed_ms=elapsed_ms,
        )
        if integrity_issues:
            feedback["summary"] = (
                "Lean accepted the proof but it violates integrity rules: "
                + "; ".join(integrity_issues[:3])
                + ". Rewrite without these bypasses.")
        response["agent_feedback"] = feedback

        return ToolResult.success(json.dumps(response, indent=2,
                                             ensure_ascii=False))

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    async def _call_verify(self, statement: str, proof: str):
        """Invoke verify_complete on the pool, await if coroutine."""
        verify = getattr(self._pool, "verify_complete", None)
        if verify is None:
            raise AttributeError("verify_complete")
        out = verify(statement, proof, "")
        if inspect.iscoroutine(out):
            out = await out
        return out

    def _prefilter_only(self, code: str) -> ToolResult:
        """No-pool path: do a syntax check and report it cleanly."""
        try:
            from engine.prefilter import SyntaxPrefilter
            pf = SyntaxPrefilter()
            passed, reason = pf.check(code)
        except Exception as e:
            logger.debug(f"prefilter unavailable: {e}")
            passed, reason = (True, "")
        return ToolResult.success(json.dumps({
            "verified": False,
            "syntax_ok": passed,
            "message": reason or
                       "Syntax OK (REPL unavailable for full verification)",
            "agent_feedback": {
                "is_proof_complete": False,
                "remaining_goals": [],
                "error_message": reason or "lean_pool not configured",
                "progress_score": 0.0,
            },
        }, ensure_ascii=False))

    @staticmethod
    def _fmt_error(e) -> str:
        """Errors arrive as either dicts ({line, message, ...}) or strings."""
        if isinstance(e, dict):
            msg = e.get("message") or e.get("error") or str(e)
            line = e.get("line")
            return f"L{line}: {msg}"[:300] if line else str(msg)[:300]
        return str(e)[:300]

    @staticmethod
    def _collect_error_text(errors_list, stderr: str) -> str:
        if errors_list:
            parts = []
            for e in errors_list[:5]:
                if isinstance(e, dict):
                    parts.append(e.get("message") or str(e))
                else:
                    parts.append(str(e))
            return "\n".join(parts)
        return stderr or ""

    def _build_feedback(self, *, success: bool, goals: list,
                          errors_text: str, elapsed_ms: int) -> dict:
        """Produce a JSON-friendly AgentFeedback block.

        When ``self._ei`` is configured we route through ErrorIntelligence
        to get repair candidates and category classification. Otherwise
        we still emit the same shape with empty repair list — schema
        consistency matters for any downstream consumer that parses
        the tool result.
        """
        if success:
            return {
                "is_proof_complete": True,
                "remaining_goals": [],
                "progress_score": 1.0,
                "elapsed_ms": elapsed_ms,
                "repair_candidates": [],
                "summary": "Proof verified — all goals closed, sorry-free.",
            }

        # Failure path: try ErrorIntelligence; on failure, build minimal
        # feedback ourselves.
        if self._ei is not None:
            try:
                from engine._core import TacticFeedback
                stub = TacticFeedback(
                    success=False,
                    tactic="(whole-proof submission)",
                    error_message=errors_text,
                    error_category=self._classify_error(errors_text),
                    remaining_goals=goals,
                    elapsed_ms=elapsed_ms,
                )
                fb = self._ei.analyze(stub, goals_before=max(1, len(goals)),
                                       use_search_tactics=False)
                return {
                    "is_proof_complete": False,
                    "remaining_goals": [g.target_type if hasattr(g, 'target_type')
                                          else str(g)
                                          for g in fb.remaining_goals[:5]],
                    "error_category": fb.error_category,
                    "error_message": fb.error_message,
                    "progress_score": fb.progress_score,
                    "elapsed_ms": fb.elapsed_ms,
                    "repair_candidates": [
                        {"tactic": rc.tactic,
                         "reason": rc.reason,
                         "confidence": rc.confidence,
                         "source": rc.source}
                        for rc in fb.repair_candidates[:5]
                    ],
                    "summary": fb.to_prompt(max_chars=1500),
                }
            except Exception as e:
                logger.debug(f"ErrorIntelligence failed; using minimal "
                             f"feedback: {e}")

        # Minimal feedback (no ErrorIntelligence available)
        return {
            "is_proof_complete": False,
            "remaining_goals": [str(g) for g in goals[:5]],
            "error_category": self._classify_error(errors_text),
            "error_message": errors_text[:500],
            "progress_score": 0.0 if not goals else 0.1,
            "elapsed_ms": elapsed_ms,
            "repair_candidates": [],
            "summary": ("Verification failed. "
                        f"{len(goals)} goals remain. "
                        f"Error: {errors_text[:200]}"),
        }

    @staticmethod
    def _classify_error(text: str) -> str:
        """
        all four legacy stderr→category implementations aligned. Falls
        back to a tiny inline table if engine import fails (e.g. stripped
        env)."""
        try:
            from engine._core import classify_error
            return classify_error(text or "")
        except Exception:
            t = (text or "").lower()
            if "type mismatch" in t:
                return "type_mismatch"
            if "unknown identifier" in t or "unknown constant" in t:
                return "unknown_identifier"
            if "unsolved goals" in t:
                return "unsolved_goals"
            if "tactic" in t and "failed" in t:
                return "tactic_failed"
            if "syntax" in t or "expected" in t:
                return "syntax_error"
            return "unclassified"
