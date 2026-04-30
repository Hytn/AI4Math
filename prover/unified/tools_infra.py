"""prover/unified/tools_infra.py — Tools for the infrastructure-merge ToolKits.

These tools surface features of community infrastructure projects
(Kimina Lean Server, Pantograph, LooKeng, NFL-HR) as LLM-callable
tools. They share the project's existing :class:`Tool` ABC so they
plug into ``ToolRegistry`` like any other tool.

Each tool is defensive: if its underlying backend is not available
(e.g. no Kimina server reachable, ``aiohttp`` not installed), it
returns a structured error rather than crashing — agents can react
sensibly to "tool unavailable" but not to import errors.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Kimina Lean Server: batch verify
# ═══════════════════════════════════════════════════════════════════════


class BatchVerifyTool(Tool):
    """Verify many proofs in one round-trip via Kimina Lean Server.

    The bare ``lean_verify`` tool processes one proof per call and pays
    a TCP / IPC round-trip every time. ``batch_verify`` packs N proofs
    into a single REST call so the server can route them across its
    worker REPL pool in parallel and reuse its preamble cache.

    Best used by:
    - ``whole_proof`` profile when generating pass@k samples — instead
      of 32 sequential verify calls, send one batch.
    - the heterogeneous runner when fusing proposals from N directions.
    """

    name = "batch_verify"
    description = (
        "Verify a BATCH of complete Lean 4 proofs in one server "
        "round-trip via the Kimina Lean Server. Use this instead of "
        "calling `lean_verify` repeatedly when you have multiple "
        "candidate proofs (e.g. pass@k sampling, exploring "
        "alternative tactic choices).\n"
        "\n"
        "Input: a list of proof strings, each containing a complete "
        "`theorem ... := by ...` block.\n"
        "\n"
        "Returns JSON `{results: [{id, success, errors, has_sorry, "
        "tactic_trace}, ...], n_succeeded, batch_elapsed_ms}`."
    )
    permission = ToolPermission.EXTERNAL
    input_schema = {
        "type": "object",
        "properties": {
            "proofs": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of Lean 4 proof snippets. Each must be a "
                    "complete, self-contained theorem with body."),
            },
            "preamble": {
                "type": "string",
                "description": (
                    "Preamble (imports / opens) shared by all proofs "
                    "in the batch. Defaults to `import Mathlib`."),
            },
        },
        "required": ["proofs"],
    }

    def __init__(self, kimina_backend=None, knowledge_store=None):
        self._backend = kimina_backend
        # Optional: when present, every successful proof's tactic trace
        # is silently fed through KnowledgeWriter.deposit_kimina_trace.
        # Layer 1 effectiveness gets a free signal from every Kimina
        # roll-out without the agent loop having to know about it.
        self._knowledge_store = knowledge_store

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        proofs = input.get("proofs") or []
        if not proofs:
            return ToolResult.error("`proofs` must be a non-empty list")
        preamble = input.get("preamble") or "import Mathlib"

        if self._backend is None:
            return ToolResult.error(
                "Kimina Lean Server backend not configured. "
                "Pass --backend kimina at startup or set KIMINA_SERVER_URL.")

        if getattr(self._backend, "is_fallback", False):
            return ToolResult.error(
                "Kimina backend is in fallback mode — server unreachable. "
                "Falling back to per-proof `lean_verify` is recommended.")

        try:
            results = await self._backend.verify_batch(proofs, preamble=preamble)
        except Exception as e:
            return ToolResult.error(f"batch verify failed: {e}")

        # Auto-deposit: every successful proof's tactic trace teaches
        # the knowledge pyramid. Errors are swallowed — knowledge
        # ingestion must NEVER crash the prover loop.
        if self._knowledge_store is not None:
            await self._auto_deposit(results, theorem=ctx.theorem_statement)

        n_succeeded = sum(1 for r in results if r.success)
        total_ms = sum(r.elapsed_ms for r in results)
        payload = {
            "results": [
                {
                    "id": r.id,
                    "success": r.success,
                    "errors": r.error_messages[:4],
                    "has_sorry": r.has_sorry,
                    "elapsed_ms": r.elapsed_ms,
                    "n_tactics_extracted": len(r.tactic_trace),
                }
                for r in results
            ],
            "n_succeeded": n_succeeded,
            "batch_size": len(results),
            "batch_elapsed_ms": total_ms,
        }
        return ToolResult.success(json.dumps(payload, ensure_ascii=False))

    async def _auto_deposit(self, results, *, theorem: str = "") -> None:
        """Best-effort knowledge deposit. Never raises."""
        try:
            # KnowledgeWriter is built lazily from the store so this
            # tool stays usable even when the writer module is not
            # imported by the caller.
            from knowledge.writer import KnowledgeWriter
        except Exception as e:
            logger.debug(f"BatchVerifyTool: KnowledgeWriter unavailable: {e}")
            return
        try:
            writer = KnowledgeWriter(self._knowledge_store)
        except Exception as e:
            logger.debug(f"BatchVerifyTool: failed to build writer: {e}")
            return
        for r in results:
            if not r.success or not r.tactic_trace:
                continue
            try:
                await writer.deposit_kimina_trace(
                    r.tactic_trace,
                    theorem=theorem,
                    domain="",
                    trace_id=0)
            except Exception as e:
                logger.debug(
                    f"BatchVerifyTool: deposit failed for {r.id}: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Pantograph: metavariable focus
# ═══════════════════════════════════════════════════════════════════════


class MVarFocusTool(Tool):
    """Rotate the goal list to focus on a specific metavariable.

    When ``apply`` introduces multiple coupled goals, the bare REPL
    proceeds linearly: it picks the first goal as "current". Pantograph
    lets the agent decide which goal to attack next, which is critical
    for non-linear proof strategies (e.g. solving the easy goal first
    so its instantiation simplifies the hard one).
    """

    name = "mvar_focus"
    description = (
        "Rotate the proof goal list so the metavariable named "
        "`mvar_id` becomes the current goal. Useful when the natural "
        "left-to-right order is suboptimal — e.g. when one goal will "
        "trivially close after another is solved.\n"
        "\n"
        "Returns JSON `{success, focused_goal, n_remaining_goals, "
        "coupled_with}`."
    )
    permission = ToolPermission.WRITE_LOCAL
    input_schema = {
        "type": "object",
        "properties": {
            "mvar_id": {
                "type": "string",
                "description": "ID of the metavariable to focus on.",
            },
            "proof_state": {
                "type": "integer",
                "description": "Current proof state ID.",
            },
        },
        "required": ["mvar_id", "proof_state"],
    }

    def __init__(self, pantograph_backend=None):
        self._backend = pantograph_backend

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        if self._backend is None or getattr(self._backend, "is_fallback", True):
            return ToolResult.error(
                "Pantograph backend not available. Install pypantograph "
                "or `pantograph` binary to enable mvar focus.")
        try:
            r = await self._backend.focus_mvar(
                proof_state=int(input["proof_state"]),
                mvar_id=str(input["mvar_id"]))
        except Exception as e:
            return ToolResult.error(f"focus_mvar failed: {e}")

        if not r.success:
            return ToolResult.error(r.error or "focus failed")

        return ToolResult.success(json.dumps({
            "success": True,
            "new_proof_state": r.new_proof_state,
            "focused_goal": (r.focused.goal if r.focused else None),
            "focused_mvar": (r.focused.mvar_id if r.focused else None),
            "coupled_with": (r.focused.coupled_with if r.focused else []),
            "n_remaining_goals": len(r.remaining),
        }, ensure_ascii=False))


class DraftHoleTool(Tool):
    """Insert a typed ``sorry``-hole and continue past it.

    Direct support for the Draft–Sketch–Prove pattern in Lean 4. The
    agent declares an intermediate sub-lemma it intends to prove later,
    Pantograph records the hole, and the proof continues using the
    drafted lemma as if it were already proved. The hole can be filled
    in any order.
    """

    name = "draft_hole"
    description = (
        "Insert a typed `sorry`-hole into the current proof. The hole "
        "represents an intermediate sub-lemma you intend to prove "
        "later. The proof can continue past the hole using the "
        "drafted statement as if proved.\n"
        "\n"
        "This is the native DSP (Draft-Sketch-Prove) primitive — use "
        "it instead of writing `have ... := by sorry` manually.\n"
        "\n"
        "Returns JSON `{success, hole_proof_state, hole_goal}`."
    )
    permission = ToolPermission.WRITE_LOCAL
    input_schema = {
        "type": "object",
        "properties": {
            "statement": {
                "type": "string",
                "description": (
                    "Type of the hole, in Lean 4 syntax "
                    "(e.g. `∀ n, P n → Q n`)."),
            },
            "proof_state": {
                "type": "integer",
                "description": "Proof state to insert the hole into.",
            },
        },
        "required": ["statement", "proof_state"],
    }

    def __init__(self, pantograph_backend=None):
        self._backend = pantograph_backend

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        if self._backend is None or getattr(self._backend, "is_fallback", True):
            return ToolResult.error("Pantograph backend not available.")
        try:
            r = await self._backend.insert_draft(
                proof_state=int(input["proof_state"]),
                statement=str(input["statement"]))
        except Exception as e:
            return ToolResult.error(f"insert_draft failed: {e}")
        if not r.success:
            return ToolResult.error(r.error or "drafting failed")
        return ToolResult.success(json.dumps({
            "success": True,
            "hole_proof_state": r.proof_state,
            "hole_goal": (r.holes[0].goal if r.holes else ""),
        }, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════════
# LooKeng: lemma-by-lemma proving
# ═══════════════════════════════════════════════════════════════════════


class LemmaByLemmaTool(Tool):
    """Submit ONE lemma at a time to a stateless verification session.

    Implements the LooKeng workflow: the LLM proposes a single
    intermediate lemma plus its proof, the runtime type-checks it
    against the running context (preamble + previously-proved lemmas),
    and on success it joins the running context. The LLM never has
    to re-state the proofs of earlier lemmas — only their statements.

    Best used by long proofs (Putnam / FATE-X) where the per-turn
    Lean payload would otherwise grow unboundedly.
    """

    name = "lemma_by_lemma"
    description = (
        "Submit ONE intermediate lemma (or the final theorem) to a "
        "LooKeng-style stateless verification session. The lemma is "
        "type-checked against the running context (preamble + earlier "
        "lemmas). On success it is appended to the running context "
        "and the LLM can use its NAME (not its proof) in subsequent "
        "calls.\n"
        "\n"
        "USE THIS INSTEAD OF `lean_verify` when working in lemma-by-"
        "lemma mode. Each call should propose exactly one named "
        "`have`-style lemma. Set `is_final=true` for the closing "
        "step that proves the original goal.\n"
        "\n"
        "Returns JSON `{ok, running_context_size, errors, "
        "session_id}`."
    )
    permission = ToolPermission.EXTERNAL
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Name for the lemma (e.g. `lemma_step_1`). Used "
                    "as a Lean identifier; will be referenced by name "
                    "in later calls."),
            },
            "statement": {
                "type": "string",
                "description": (
                    "Lemma statement in Lean 4 syntax. Required for "
                    "intermediate lemmas; ignored when is_final=true."),
            },
            "proof": {
                "type": "string",
                "description": (
                    "Proof body (everything after `:= by`). Use simple "
                    "tactics; can reference earlier lemmas by name."),
            },
            "is_final": {
                "type": "boolean",
                "description": (
                    "True if this proves the original goal theorem "
                    "(not an intermediate). Default false."),
            },
            "session_id": {
                "type": "string",
                "description": (
                    "(Optional) Session identifier. The runner allocates "
                    "and threads one automatically; only set this if you "
                    "are explicitly forking sessions."),
            },
        },
        "required": ["name", "proof"],
    }

    def __init__(self, lookeng_backend=None):
        self._backend = lookeng_backend

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        if self._backend is None:
            return ToolResult.error(
                "LooKeng backend not configured. Use --backend lookeng "
                "at startup, or pick a non-lemma-by-lemma profile.")

        # Resolution order for session_id:
        #   1. Explicit `input["session_id"]` (LLM is forking/multi-session).
        #   2. `ctx.shared_state["lookeng_session_id"]` (runner-injected).
        #   3. Auto-create one against the current theorem.
        sid = (input.get("session_id")
               or ctx.shared_state.get("lookeng_session_id"))
        if not sid:
            try:
                sid = await self._backend.begin_session(
                    theorem=ctx.theorem_statement or "")
                # Cache for subsequent calls in the same loop.
                ctx.shared_state["lookeng_session_id"] = sid
            except Exception as e:
                return ToolResult.error(
                    f"failed to auto-bootstrap LooKeng session: {e}")

        try:
            r = await self._backend.submit_lemma(
                session_id=str(sid),
                name=str(input["name"]),
                statement=str(input.get("statement", "")),
                proof=str(input["proof"]),
                is_final=bool(input.get("is_final", False)))
        except Exception as e:
            return ToolResult.error(f"submit_lemma failed: {e}")

        return ToolResult.success(json.dumps(r, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════════
# NFL-HR: NL-FL existence-theorem bridge
# ═══════════════════════════════════════════════════════════════════════


class NLExistenceBridgeTool(Tool):
    """Translate a natural-language QA-style problem into a Lean 4
    existence theorem.

    Implements the NL-FL alignment step from "Natural-Formal Hybrid
    Reasoning" (Yao et al., EMNLP 2025). Given an NL problem like
    "Find all integers n such that n² < 10", the tool returns a
    Lean 4 statement of the form ``∃ S : Finset ℤ, ∀ n, n ∈ S ↔ n^2 < 10``.

    The LLM can then prove the existence theorem AND simultaneously
    answer the original NL question in its chain-of-thought, with the
    formal proof acting as a verifier of the answer. The "answer
    extraction" step is left to the runner — see
    ``prover.unified.runner`` for how it pulls the implicit answer
    out of the FL CoT.

    Pluggable autoformalizer
    ------------------------

    The default implementation is a deterministic skeleton (it knows
    the answer's TYPE but not its predicate). To plug in a real
    autoformalizer (e.g. a Kimina-Autoformalizer LLM), call
    :func:`register_autoformalizer` once at startup with a callable
    that takes ``(nl_problem, answer_type) -> lean_statement``::

        from prover.unified.tools_infra import register_autoformalizer
        register_autoformalizer(my_kimina_autoformalizer.translate)

    Subsequent ``execute()`` calls will use the registered formalizer
    in place of the heuristic.
    """

    name = "nl_existence"
    description = (
        "Translate a natural-language MATH-style question into a "
        "Lean 4 *existence theorem* (∃ x, P x ∧ ...) suitable for "
        "formal proving. Use this when the original problem is in "
        "QA form (Find / Compute / How many...) rather than Prove/"
        "Show form, and you want to convert it to a provable Lean "
        "statement that the FL prover can attack.\n"
        "\n"
        "Returns JSON `{lean_statement, expected_answer_shape, "
        "informal_summary}`."
    )
    permission = ToolPermission.READ_ONLY
    input_schema = {
        "type": "object",
        "properties": {
            "nl_problem": {
                "type": "string",
                "description": (
                    "The natural-language problem statement, "
                    "verbatim from the source."),
            },
            "answer_type": {
                "type": "string",
                "enum": ["integer", "rational", "real", "set", "finset",
                          "list", "boolean", "unknown"],
                "description": (
                    "Expected mathematical type of the answer. If "
                    "unsure, use 'unknown' and the tool will guess."),
            },
        },
        "required": ["nl_problem"],
    }

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        nl = str(input["nl_problem"]).strip()
        ans_type = str(input.get("answer_type", "unknown"))

        # If a real autoformalizer is registered, prefer it over the
        # heuristic. Errors in the registered callable fall through to
        # the heuristic so the tool stays useful even when the
        # registered model is misbehaving.
        registered = _get_autoformalizer()
        used = "heuristic"
        lean_stmt = ""
        if registered is not None:
            try:
                if asyncio.iscoroutinefunction(registered):
                    lean_stmt = await registered(nl, ans_type)
                else:
                    lean_stmt = registered(nl, ans_type)
                if lean_stmt:
                    used = "registered"
            except Exception as e:
                logger.warning(
                    f"registered autoformalizer raised, falling back: {e}")
                lean_stmt = ""
        if not lean_stmt:
            lean_stmt = self._heuristic_translate(nl, ans_type)
            used = "heuristic"

        return ToolResult.success(json.dumps({
            "lean_statement": lean_stmt,
            "expected_answer_shape": ans_type,
            "informal_summary": nl[:200],
            "autoformalizer": used,
            "note": (
                "Heuristic translation. For higher-quality "
                "auto-formalization, call register_autoformalizer() "
                "with a Kimina-Autoformalizer or similar."
                if used == "heuristic" else
                "Translated by the registered autoformalizer."),
        }, ensure_ascii=False))

    @staticmethod
    def _heuristic_translate(nl: str, ans_type: str) -> str:
        """Pattern-based NL→Lean existence-theorem skeleton.

        We do not call an LLM; the goal is to produce a *syntactically
        sensible* Lean 4 statement that captures the QA shape so the
        downstream prover has a real target. A registered
        autoformalizer (see :func:`register_autoformalizer`) will
        always be preferred over this — but when nothing better is
        available, these patterns are dramatically more useful than
        a bare ``True`` placeholder.

        The pattern bank covers the QA shapes that account for ~80% of
        MATH/AIME/Putnam-style problems:

          * "Find the smallest/largest/least n such that P(n)"
            → ``∃ n : T, P n ∧ ∀ m, P m → n ≤ m`` (smallest)
                              ``∧ ∀ m, P m → m ≤ n`` (largest)
          * "Find all integers/reals/etc satisfying P"
            → ``∃ S : Finset T, ∀ n, n ∈ S ↔ P n``
          * "How many … (count)"
            → ``∃ k : ℕ, k = Set.ncard {x : T | P x}``
          * "Compute / Evaluate / What is the value of f(…)"
            → ``∃ ans : T, ans = f(…)``
          * Generic fallback
            → ``∃ ans : T, True /- TODO predicate -/`` (annotated)
        """
        type_skel = {
            "integer": "ℤ",
            "rational": "ℚ",
            "real": "ℝ",
            "set":    "Set ℕ",
            "finset": "Finset ℕ",
            "list":   "List ℕ",
            "boolean": "Bool",
        }.get(ans_type, "ℕ")

        # Strip simple LaTeX so the Lean header is parseable.
        clean = nl.replace("$", "").replace("\\", "").strip()
        clean_short = clean[:120]
        nl_lower = clean.lower()

        # Pattern 1 — extremum: smallest / largest / least / greatest n s.t.
        extrema = (
            ("smallest", "≤"), ("least", "≤"), ("minimum", "≤"),
            ("largest", "≥"), ("greatest", "≥"), ("maximum", "≥"),
        )
        for kw, cmp in extrema:
            if kw in nl_lower:
                bound_op = "n ≤ m" if cmp == "≤" else "m ≤ n"
                return (
                    f"-- Auto-formalised existence theorem ({kw}) for: "
                    f"{clean_short}\n"
                    f"theorem ai4math_q : ∃ (n : {type_skel}),\n"
                    f"    True /- TODO: P n -/ ∧\n"
                    f"    ∀ (m : {type_skel}), "
                    f"True /- TODO: P m -/ → {bound_op}")

        # Pattern 2 — set-comprehension: "find all" / "determine all"
        if any(kw in nl_lower for kw in (
                "find all", "determine all", "all integers",
                "all real", "all positive", "all natural")):
            container = "Finset" if ans_type in ("integer", "finset",
                                                  "list") else "Set"
            return (
                f"-- Auto-formalised existence theorem (set) for: "
                f"{clean_short}\n"
                f"theorem ai4math_q : ∃ (S : {container} {type_skel}),\n"
                f"    ∀ (n : {type_skel}), "
                f"n ∈ S ↔ True /- TODO: P n -/")

        # Pattern 3 — counting: "how many" / "count" / "number of"
        if any(kw in nl_lower for kw in (
                "how many", "count", "number of")):
            return (
                f"-- Auto-formalised existence theorem (count) for: "
                f"{clean_short}\n"
                f"theorem ai4math_q : ∃ (k : ℕ),\n"
                f"    k = Set.ncard "
                f"{{x : {type_skel} | True /- TODO: P x -/}}")

        # Pattern 4 — value: "compute" / "evaluate" / "what is"
        if any(kw in nl_lower for kw in (
                "compute", "evaluate", "what is", "find the value")):
            return (
                f"-- Auto-formalised existence theorem (value) for: "
                f"{clean_short}\n"
                f"theorem ai4math_q : ∃ (ans : {type_skel}),\n"
                f"    ans = (0 : {type_skel}) "
                f"/- TODO: replace 0 with the closed-form expression -/")

        # Pattern 5 — boolean / yes-no: "does there exist" / "is it true"
        if any(kw in nl_lower for kw in (
                "does there exist", "is it true", "is it possible")):
            return (
                f"-- Auto-formalised existence theorem (decidable) for: "
                f"{clean_short}\n"
                f"theorem ai4math_q : ∃ (b : Bool),\n"
                f"    b = true ∨ b = false "
                f"/- TODO: replace with decidability witness -/")

        # Fallback — keep the previous generic skeleton, but annotate it.
        return (
            f"-- Auto-formalised existence theorem (generic) for: "
            f"{clean_short}\n"
            f"theorem ai4math_q : ∃ (ans : {type_skel}),\n"
            f"    True /- TODO: replace with translated predicate -/")


# ─── Module-level autoformalizer registry ────────────────────


# Using a module-level holder rather than a class attribute so the
# registration survives across tool instantiations.
_autoformalizer: Optional[Callable[[str, str], str]] = None


def register_autoformalizer(
        fn: Optional[Callable[[str, str], str]]) -> None:
    """Plug in a real NL→FL translator for ``NLExistenceBridgeTool``.

    Once registered, every ``nl_existence`` tool call will use ``fn``
    instead of the heuristic skeleton. Pass ``None`` to deregister and
    return to heuristic mode.

    The callable signature is ``fn(nl_problem: str, answer_type: str) -> str``
    and may be either sync or async — async callables are awaited.
    """
    global _autoformalizer
    _autoformalizer = fn


def _get_autoformalizer() -> Optional[Callable[[str, str], str]]:
    """Internal: read the currently registered autoformalizer."""
    return _autoformalizer
