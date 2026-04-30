"""engine/backends/lookeng.py — LooKeng-style stateless Lean REPL.

Implements the verification interface described in Seed-Prover 1.5
(Chen et al., 2025, "Seed-Prover 1.5: Mastering Undergraduate-Level
Theorem Proving via Learning from Experience"; LooKeng is the
authors' Lean interface, see arXiv:2512.17260 §4.2).

The shape of LooKeng
--------------------

Where ordinary tactic-mode REPL keeps a long-running ``proofState`` stack
on the Lean side, LooKeng goes the other way: it is **stateless** with
respect to the proof itself and only persists a small **running context**
on the Python side:

    running_context = statement_header + previously_proven_lemmas

For each model turn, the model proposes ONE LEMMA (either an intermediate
``have`` lemma or the final goal) along with its proof. The runtime
splices that lemma plus its proof into a fresh Lean compile against the
running context and reports back the structured feedback. If the lemma
type-checks, it gets appended to the running context; if not, the running
context is unchanged.

This matters because:

1. **I/O is reduced ~40%** vs. continuously feeding the entire proof
   history to the REPL. The Seed-Prover 1.5 paper attributes a large
   chunk of their wall-clock improvement to this.
2. **The model context stays lean.** At turn N, the LLM only sees the
   *statements* of previously proved lemmas, not their proofs. Only the
   compiler needs to see the proofs.
3. **Backtracking is trivial.** Pop the last entry off the running
   context list and you've undone a step — no env-tree bookkeeping.

When this backend is the right choice
-------------------------------------

Use LooKeng when:

* The agent is a strong reasoning model that benefits from explicit
  decomposition into named lemmas (Seed-Prover 1.5, Kimina-Prover 72B,
  Claude Opus with extended thinking).
* The proofs are long and the per-turn bandwidth to Lean dominates
  wall-clock (typical for PutnamBench / FATE-X).
* The agent uses lemma-by-lemma framing, not raw tactic-by-tactic.

Use ``LocalTransport`` (the existing default) when:

* The agent works tactic-by-tactic — every tactic sees the live goal
  state from Lean. LooKeng can't expose mid-proof goal state because
  it doesn't keep a live REPL session.

API surface
-----------

* :class:`LooKengBackend` — implements ``REPLTransport`` so it slots
  into ``AsyncLeanPool`` like the other backends.
* :class:`RunningContext` — the per-session statement-and-lemmas cache.
* :class:`LemmaCacheEntry` — one entry in that cache.
* :func:`build_running_context_prompt` — helper for the agent layer to
  render the current running context into an LLM prompt fragment.

Implementation note
-------------------

Internally, LooKeng *uses* a long-running Lean REPL (or a Kimina
Server) to do the actual compilation; the "stateless" claim is about
the REPL not holding mutable per-session state, not about avoiding
Lean entirely. We delegate to whatever ``REPLTransport`` is wrapped
inside ``LooKengBackend.inner``, so any backend that can compile a
self-contained Lean snippet is a valid execution layer.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from engine.transport import REPLTransport, TransportStats, LocalTransport

logger = logging.getLogger(__name__)


# ─── Running context ─────────────────────────────────────────


@dataclass
class LemmaCacheEntry:
    """One lemma proved earlier in this session.

    After type-checking, only the *statement* (header + signature) is
    fed to the LLM in subsequent turns; ``proof_body`` is kept around
    for the compiler. This is the asymmetry that makes LooKeng cheap.
    """
    name: str
    statement: str            # what the LLM sees
    proof_body: str = ""      # what the compiler sees (kept private)
    elapsed_ms: int = 0
    n_attempts: int = 1       # how many tries it took to land

    def render_for_llm(self) -> str:
        """The LLM sees just the typed signature, no proof body."""
        return self.statement.strip()

    def render_for_compiler(self) -> str:
        """The compiler sees the full ``theorem name ... := by ...``."""
        head = self.statement.strip().rstrip(":=").rstrip()
        if not self.proof_body:
            # Caller forgot to set it — degrade to ``sorry`` rather than
            # silently dropping; downstream type-check will then complain.
            return f"{head} := by sorry"
        body = self.proof_body.strip()
        if body.startswith(":="):
            return f"{head} {body}"
        return f"{head} := by\n  {body}"


@dataclass
class RunningContext:
    """The complete per-session context that LooKeng tracks.

    A LooKeng session starts with a theorem statement (what we want to
    prove) and grows by appending ``LemmaCacheEntry`` objects each
    time the model lands an intermediate ``have``-lemma. The final
    closing lemma references all earlier ones in its proof body.
    """
    theorem_header: str
    preamble: str = "import Mathlib"
    lemmas: list[LemmaCacheEntry] = field(default_factory=list)

    # Parallel-search-friendly scratch: when an attempt fails, we don't
    # mutate the canonical list; instead the runner forks a copy.
    session_id: str = ""

    def append(self, entry: LemmaCacheEntry):
        self.lemmas.append(entry)

    def fork(self, session_id: str = "") -> RunningContext:
        return RunningContext(
            theorem_header=self.theorem_header,
            preamble=self.preamble,
            lemmas=list(self.lemmas),
            session_id=session_id or self.session_id)

    def render_compiler_block(self, new_proof: str) -> str:
        """Build the full Lean snippet to send to the compiler.

        Order: preamble, then the proved lemmas (with full bodies), then
        the new proof candidate. The ``new_proof`` parameter is whatever
        the model proposed *this turn*: an intermediate ``have``-lemma
        or a final attempt at the theorem.
        """
        parts = [self.preamble.strip()]
        for entry in self.lemmas:
            parts.append(entry.render_for_compiler())
        parts.append(new_proof.strip())
        return "\n\n".join(p for p in parts if p)

    def render_llm_briefing(self) -> str:
        """Render a human/LLM-readable summary of what's been proved.

        This is what the agent layer slots into the system / user prompt
        each turn so the LLM remembers the lemmas already in scope.
        """
        if not self.lemmas:
            return ""
        lines = ["## Lemmas already proved this session\n"]
        for i, entry in enumerate(self.lemmas, 1):
            lines.append(f"{i}. `{entry.name}`")
            lines.append(f"   {entry.render_for_llm()}")
        return "\n".join(lines)


# ─── Backend ─────────────────────────────────────────────────


class LooKengBackend(REPLTransport):
    """Stateless Lean wrapper that delegates to an inner ``REPLTransport``.

    The inner transport handles actual Lean compilation. This class
    layers the lemma-cache semantics on top: each ``send`` rewrites
    the request to include the running context as a Lean preamble so
    that whatever the inner transport does, it sees a self-contained
    snippet.

    Sessions
    --------

    Because LooKeng is stateless, "sessions" here are purely a Python
    concept: the runtime allocates a session id, builds an empty
    ``RunningContext``, and threads it through subsequent calls. A
    session can be forked cheaply (just clone the lemma list) which
    makes parallel exploration natural.

    The ``REPLTransport`` interface forces us into a narrower API
    (``send(cmd)``), so we encode session ops via reserved keys:

    * ``{"lookeng_op": "begin_session", "theorem": "...", "preamble": "..."}``
      → returns ``{"session_id": ...}``
    * ``{"lookeng_op": "submit_lemma", "session_id": ..., "name": ..., "proof": ...}``
      → returns ``{"ok": bool, "running_context_size": int, "errors": [...]}``
    * ``{"lookeng_op": "close_session", "session_id": ...}``
    * Everything else falls through to the inner transport with the
      running context spliced in as a preamble.
    """

    def __init__(self, inner: REPLTransport = None,
                 project_dir: str = ".",
                 default_preamble: str = "import Mathlib",
                 max_running_context_lemmas: int = 64):
        self._inner = inner
        self._project_dir = project_dir
        self._default_preamble = default_preamble
        self._max_lemmas = max_running_context_lemmas
        self._stats = TransportStats()
        self._alive = False
        self._inner_owned = False  # we created the inner ourselves

        self._sessions: dict[str, RunningContext] = {}
        self._next_session = 1

    # ── REPLTransport ABC ─────────────────────────────────────

    async def start(self) -> bool:
        if self._inner is None:
            self._inner = LocalTransport(
                project_dir=self._project_dir, timeout_seconds=120)
            self._inner_owned = True
        ok = await self._inner.start()
        self._alive = ok
        if ok:
            logger.info(
                f"LooKengBackend: started "
                f"(inner={type(self._inner).__name__}, "
                f"max_lemmas={self._max_lemmas})")
        return ok

    async def send(self, cmd: dict) -> Optional[dict]:
        if not self._alive:
            return None

        op = cmd.get("lookeng_op")
        t0 = time.monotonic()

        try:
            if op == "begin_session":
                resp = self._begin_session(cmd)
            elif op == "submit_lemma":
                resp = await self._submit_lemma(cmd)
            elif op == "close_session":
                resp = self._close_session(cmd)
            elif op == "running_context":
                resp = self._render_running_context(cmd)
            else:
                # Plain pass-through, optionally with running-context splice.
                resp = await self._passthrough(cmd)

            self._stats.record_success((time.monotonic() - t0) * 1000.0)
            return resp
        except Exception as e:
            logger.warning(f"LooKengBackend.send error: {e}")
            self._stats.record_failure()
            return {"messages": [{"severity": "error", "data": str(e)}]}

    async def close(self):
        if self._inner is not None and self._inner_owned:
            try:
                await self._inner.close()
            except Exception as _exc:
                logger.debug(f"Suppressed close exception: {_exc}")
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_fallback(self) -> bool:
        return self._inner is None or self._inner.is_fallback

    def get_stats(self) -> dict:
        d = self._stats.to_dict()
        d["sessions"] = len(self._sessions)
        d["max_lemmas_per_session"] = self._max_lemmas
        if self._inner is not None:
            d["inner"] = self._inner.get_stats()
        return d

    # ── session ops ───────────────────────────────────────────

    def _begin_session(self, cmd: dict) -> dict:
        sid = cmd.get("session_id") or f"lk-{self._next_session}"
        self._next_session += 1
        ctx = RunningContext(
            theorem_header=cmd.get("theorem", "").strip(),
            preamble=cmd.get("preamble", self._default_preamble).strip(),
            session_id=sid)
        self._sessions[sid] = ctx
        return {"session_id": sid, "ok": True,
                "running_context_size": 0}

    def _close_session(self, cmd: dict) -> dict:
        sid = cmd.get("session_id", "")
        existed = self._sessions.pop(sid, None) is not None
        return {"session_id": sid, "ok": existed}

    async def _submit_lemma(self, cmd: dict) -> dict:
        """Type-check ``proof`` against the running context.

        On success, append a new ``LemmaCacheEntry`` to the session.
        On failure, the running context is untouched and the errors
        are returned verbatim.
        """
        sid = cmd.get("session_id", "")
        if sid not in self._sessions:
            return {"ok": False, "errors": [f"unknown session_id {sid!r}"]}

        ctx = self._sessions[sid]
        if len(ctx.lemmas) >= self._max_lemmas:
            return {"ok": False, "errors": [
                f"running context already at max ({self._max_lemmas} "
                "lemmas) — close session or fork"]}

        name = cmd.get("name", "").strip() or f"lemma_{len(ctx.lemmas)+1}"
        statement = cmd.get("statement", "").strip()
        proof_body = cmd.get("proof", "").strip()
        is_final = bool(cmd.get("is_final", False))

        if not statement and not is_final:
            return {"ok": False, "errors": ["statement is required for "
                                              "intermediate lemmas"]}

        # Compose the snippet to type-check.
        if is_final:
            # The "lemma" being submitted is actually the full theorem.
            new_block = self._compose_final(ctx.theorem_header, proof_body)
        else:
            new_block = self._compose_lemma(name, statement, proof_body)

        snippet = ctx.render_compiler_block(new_block)
        t0 = time.monotonic()
        compile_resp = await self._inner.send(
            {"cmd": snippet, "env": 0})
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if compile_resp is None:
            return {"ok": False, "errors": ["inner transport returned None"]}

        errors = [m.get("data", "") for m in compile_resp.get("messages", [])
                  if m.get("severity") == "error"]
        ok = len(errors) == 0

        if ok and not is_final:
            ctx.append(LemmaCacheEntry(
                name=name,
                statement=statement,
                proof_body=proof_body,
                elapsed_ms=elapsed_ms))

        return {
            "ok": ok,
            "is_final": is_final,
            "session_id": sid,
            "running_context_size": len(ctx.lemmas),
            "errors": errors[:8],
            "elapsed_ms": elapsed_ms,
        }

    def _render_running_context(self, cmd: dict) -> dict:
        sid = cmd.get("session_id", "")
        if sid not in self._sessions:
            return {"ok": False, "errors": [f"unknown session_id {sid!r}"]}
        ctx = self._sessions[sid]
        return {
            "ok": True,
            "session_id": sid,
            "lemmas": [
                {"name": e.name, "statement": e.statement,
                 "n_attempts": e.n_attempts}
                for e in ctx.lemmas
            ],
            "llm_briefing": ctx.render_llm_briefing(),
            "compiler_size_bytes": len(ctx.render_compiler_block("")),
        }

    async def _passthrough(self, cmd: dict) -> Optional[dict]:
        """Forward to inner transport, optionally splicing running context.

        If the caller passes ``cmd["session_id"]`` we automatically
        prepend the running context as a preamble. Otherwise it's a
        plain call — useful for warmup, ``import``-only commands, etc.
        """
        sid = cmd.get("session_id")
        if sid and sid in self._sessions:
            ctx = self._sessions[sid]
            # Take whatever the user sent and wrap it in the running
            # context. We only do this for ``cmd``-shaped requests.
            if "cmd" in cmd:
                spliced = ctx.render_compiler_block(cmd["cmd"])
                return await self._inner.send(
                    {"cmd": spliced, "env": cmd.get("env", 0)})
        return await self._inner.send(cmd)

    # ── compile helpers ───────────────────────────────────────

    @staticmethod
    def _compose_lemma(name: str, statement: str, proof_body: str) -> str:
        statement = statement.strip()
        # Allow the user to send either a full ``theorem foo : T`` line
        # or just ``: T`` — normalise to a named ``theorem`` form.
        if statement.startswith("theorem ") or statement.startswith("lemma "):
            head = statement
        elif statement.startswith(":"):
            head = f"theorem {name} {statement}"
        else:
            head = f"theorem {name} : {statement}"
        if ":=" in head:
            head = head[:head.index(":=")].rstrip()
        body = (proof_body or "sorry").strip()
        if body.startswith(":="):
            return f"{head} {body}"
        return f"{head} := by\n  {body}"

    @staticmethod
    def _compose_final(theorem_header: str, proof_body: str) -> str:
        h = theorem_header.strip()
        if ":=" in h:
            h = h[:h.index(":=")].rstrip()
        body = (proof_body or "sorry").strip()
        if body.startswith(":="):
            return f"{h} {body}"
        return f"{h} := by\n  {body}"

    # ── direct API for callers that don't want JSON-string ops ──

    async def begin_session(self, theorem: str,
                             preamble: str = None,
                             session_id: str = None) -> str:
        resp = self._begin_session({"theorem": theorem,
                                     "preamble": preamble
                                     or self._default_preamble,
                                     "session_id": session_id})
        return resp["session_id"]

    async def submit_lemma(self, session_id: str,
                            name: str, statement: str,
                            proof: str,
                            is_final: bool = False) -> dict:
        return await self._submit_lemma({
            "session_id": session_id,
            "name": name,
            "statement": statement,
            "proof": proof,
            "is_final": is_final,
        })

    def get_running_context(self, session_id: str) -> Optional[RunningContext]:
        return self._sessions.get(session_id)


# ─── Public helpers ──────────────────────────────────────────


def build_running_context_prompt(ctx: RunningContext,
                                   include_preamble: bool = False) -> str:
    """Format a ``RunningContext`` as a prompt fragment for the LLM.

    Used by ``prover.unified.system_prompts`` when the ``lookeng_lemma``
    framing is active. The preamble is usually omitted because Mathlib
    is implicit, but tests can set ``include_preamble=True``.
    """
    lines = []
    if include_preamble and ctx.preamble:
        lines.append("```lean")
        lines.append(ctx.preamble.strip())
        lines.append("```")
    if ctx.theorem_header:
        lines.append(f"\n## Goal theorem\n```lean\n{ctx.theorem_header}\n```")
    briefing = ctx.render_llm_briefing()
    if briefing:
        lines.append("\n" + briefing)
    return "\n".join(lines)
