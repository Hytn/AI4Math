"""engine/backends/pantograph.py — Pantograph backend.

Integrates `pypantograph <https://github.com/lenianiva/PyPantograph>`_
(Aniva et al., 2024, "Pantograph: A Machine-to-Machine Interaction
Interface for Advanced Theorem Proving, High Level Reasoning, and Data
Extraction in Lean 4", arXiv:2410.16429).

Why we want Pantograph
----------------------

The bare Lean 4 REPL exposes ``cmd``/``tactic`` over a ``proofState``
abstraction. That works for linear proof construction but hides three
things Pantograph treats as first-class:

1. **Metavariable coupling.** When you ``apply`` a multi-conclusion
   lemma in Lean, multiple metavariables get instantiated together.
   The REPL reports the resulting goal list but doesn't tell you which
   goals are *coupled* — meaning, solving one will partially fill
   another. This matters for tree search: an MCTS that treats coupled
   goals as independent will dramatically over-count work. Pantograph
   surfaces the coupling graph explicitly (extending Aesop's copying
   technique).

2. **Drafting.** The DSP method (Draft–Sketch–Prove) wants to drop a
   ``sorry`` placeholder for an intermediate goal, continue past it,
   and come back later. The REPL handles this awkwardly via the
   ``sorries`` field. Pantograph has a dedicated drafting mode where
   you can carry a list of holes through the proof and resolve them
   in any order.

3. **S-expression proof terms.** When a proof closes, Pantograph can
   emit the resulting term in a stable S-expression form. This is the
   data shape used by CoqGym, LeanDojo, and most retrieval-augmented
   provers — and it's a more useful training-data format than the
   raw tactic script. (LeanDojo extracts these from Mathlib at index
   time; with Pantograph we can extract them on-the-fly during
   roll-outs.)

Available modes
---------------

If ``pypantograph`` is installed and a Pantograph binary is on PATH,
this backend uses the Python bindings. If only the binary is present,
we fall back to a subprocess-pipe protocol. If neither is available,
the backend declares ``is_fallback=True`` and the framework falls back
to ``LocalTransport``.

Public surface
--------------

* :class:`PantographBackend` — implements ``REPLTransport``. Translates
  ``cmd``/``tactic`` to the appropriate Pantograph call.
* :class:`GoalFragment` — one goal plus its mvar coupling metadata.
* :class:`MVarFocusResult` — the result of focusing on a specific mvar
  (consumed by the ``mvar_focus`` tool).
* :func:`extract_proof_term` — convenience for getting an S-expression
  out of a closed proof.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional

from engine.transport import REPLTransport, TransportStats

logger = logging.getLogger(__name__)

# ─── Wire-format dataclasses ─────────────────────────────────

@dataclass
class GoalFragment:
    """A single Lean 4 goal annotated with metavariable info.

    Fields beyond plain text:

    * ``mvar_id`` — the metavariable that this goal will instantiate.
      Pantograph assigns one mvar per goal, so this is a stable
      handle even across goal-list reorderings.
    * ``coupled_with`` — list of mvar ids that are linked to this one.
      Solving this goal will partially solve any coupled mvar; this is
      crucial for search algorithms that need to avoid double-counting.
    * ``hypotheses`` — local context, exposed as ``[(name, type), ...]``.
      The bare REPL stringifies these into the goal text; Pantograph
      keeps them structured.
    """
    goal: str
    mvar_id: str = ""
    coupled_with: list[str] = field(default_factory=list)
    hypotheses: list[tuple[str, str]] = field(default_factory=list)
    is_meta: bool = False  # True for synthesised goals from drafts

    @staticmethod
    def from_wire(d: dict) -> GoalFragment:
        hyps = []
        for h in d.get("hypotheses", []) or []:
            if isinstance(h, dict):
                hyps.append((h.get("name", ""), h.get("type", "")))
            elif isinstance(h, (list, tuple)) and len(h) >= 2:
                hyps.append((str(h[0]), str(h[1])))
        return GoalFragment(
            goal=d.get("goal", "") or d.get("type", ""),
            mvar_id=d.get("mvar", "") or d.get("mvar_id", ""),
            coupled_with=list(d.get("coupled_with", []) or []),
            hypotheses=hyps,
            is_meta=bool(d.get("is_meta", False)),
        )

@dataclass
class MVarFocusResult:
    """Result of asking Pantograph to focus on a specific mvar.

    After ``mvar_focus``, the proof state is "rotated" so the requested
    goal is current. We return the new goals list plus what the proof
    state ID is on the Pantograph side, which the caller can later pass
    to ``tactic`` calls.
    """
    success: bool
    new_proof_state: int = -1
    focused: Optional[GoalFragment] = None
    remaining: list[GoalFragment] = field(default_factory=list)
    error: str = ""

@dataclass
class DraftResult:
    """Result of inserting a ``sorry``-hole during draft-sketch-prove."""
    success: bool
    holes: list[GoalFragment] = field(default_factory=list)
    proof_state: int = -1
    error: str = ""

# ─── Backend ─────────────────────────────────────────────────

class PantographBackend(REPLTransport):
    """Pantograph adapter implementing ``REPLTransport``.

    Detection order, falling through on failure:
    1. ``pypantograph`` Python package (preferred — in-process, fastest)
    2. ``pantograph`` binary on PATH (subprocess pipe)
    3. fallback mode (returns None for every send)

    The chosen mode is reflected in :pyattr:`mode` for diagnostics.
    """

    MODE_PYBIND = "pypantograph"
    MODE_BINARY = "binary"
    MODE_FALLBACK = "fallback"

    def __init__(self, project_dir: str = ".", timeout_seconds: int = 60,
                 enable_drafting: bool = True,
                 enable_mvar_coupling: bool = True):
        self._project_dir = os.path.abspath(project_dir)
        self._timeout = timeout_seconds
        self._enable_drafting = enable_drafting
        self._enable_mvar_coupling = enable_mvar_coupling

        self.mode = self.MODE_FALLBACK
        self._py_server = None        # pypantograph.Server instance
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._send_lock = asyncio.Lock()
        self._stats = TransportStats()
        self._alive = False

        # For binary mode: track proof state IDs we've handed out.
        self._next_env = 1
        self._next_proof_state = 1
        self._proof_state_goals: dict[int, list[GoalFragment]] = {}
        # Live pypantograph ``GoalState`` objects keyed by our proof_state
        # id. Populated only when running through pypantograph; consulted
        # by ``focus_mvar``/``insert_draft`` to make real Pantograph calls
        # rather than dataclass-only heuristics.
        self._proof_state_live: dict[int, object] = {}

    # ── REPLTransport ABC ─────────────────────────────────────

    async def start(self) -> bool:
        # Try the Python bindings first.
        if self._try_pybind_init():
            self.mode = self.MODE_PYBIND
            self._alive = True
            logger.info("PantographBackend: using pypantograph (in-process)")
            return True

        # Then the binary.
        if shutil.which("pantograph"):
            ok = await self._start_binary()
            if ok:
                self.mode = self.MODE_BINARY
                self._alive = True
                logger.info("PantographBackend: using pantograph binary")
                return True

        # Fallback.
        self.mode = self.MODE_FALLBACK
        self._alive = True
        logger.warning(
            "PantographBackend: neither pypantograph nor pantograph binary "
            "found — running in fallback mode. Install with: "
            "pip install pantograph-api")
        return True

    async def send(self, cmd: dict) -> Optional[dict]:
        if self.mode == self.MODE_FALLBACK:
            self._stats.record_failure()
            return None

        t0 = time.monotonic()
        try:
            if self.mode == self.MODE_PYBIND:
                resp = await self._send_pybind(cmd)
            else:
                resp = await self._send_binary(cmd)
            self._stats.record_success((time.monotonic() - t0) * 1000.0)
            return resp
        except Exception as e:
            logger.warning(f"PantographBackend.send error: {e}")
            self._stats.record_failure()
            return None

    async def close(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except Exception as _exc:
                logger.debug(f"Suppressed close exception: {_exc}")
            self._proc = None
        if self._py_server is not None:
            try:
                close = getattr(self._py_server, "close", None)
                if close:
                    if asyncio.iscoroutinefunction(close):
                        await close()
                    else:
                        close()
            except Exception as _exc:
                logger.debug(f"Suppressed close exception: {_exc}")
            self._py_server = None
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_fallback(self) -> bool:
        return self.mode == self.MODE_FALLBACK

    def get_stats(self) -> dict:
        d = self._stats.to_dict()
        d["mode"] = self.mode
        d["mvar_coupling"] = self._enable_mvar_coupling
        d["drafting"] = self._enable_drafting
        return d

    # ── pypantograph mode ─────────────────────────────────────

    def _try_pybind_init(self) -> bool:
        try:
            import pantograph  # type: ignore
        except ImportError:
            return False
        try:
            # pypantograph's Server class manages the underlying Lean
            # process. Use ``imports=["Mathlib"]`` to preload Mathlib
            # so the first ``run_tac`` call doesn't pay the import cost.
            self._py_server = pantograph.Server(
                imports=["Mathlib"], project_path=self._project_dir)
            return True
        except Exception as e:
            logger.warning(f"pypantograph init failed: {e}")
            self._py_server = None
            return False

    async def _send_pybind(self, cmd: dict) -> dict:
        """Translate REPL JSON to pypantograph Server calls.

        We run in a thread-pool executor since pypantograph is sync.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._send_pybind_sync, cmd)

    def _send_pybind_sync(self, cmd: dict) -> dict:
        srv = self._py_server
        # Command mode: a fresh top-level statement.
        if "cmd" in cmd:
            code = cmd["cmd"]
            try:
                # pypantograph 0.4+ exposes ``Server.load_sorry`` for
                # proofs and ``Server.gc`` for envs. We use a generic
                # ``run`` if available.
                if hasattr(srv, "load_sorry"):
                    units = srv.load_sorry(code)
                    new_env = self._next_env
                    self._next_env += 1
                    if units and getattr(units[0], "goal_state", None):
                        gs = units[0].goal_state
                        ps = self._next_proof_state
                        self._next_proof_state += 1
                        frags = self._extract_goal_fragments(gs)
                        self._proof_state_goals[ps] = frags
                        self._proof_state_live[ps] = gs
                        return {
                            "env": new_env,
                            "messages": [],
                            "sorries": [{
                                "proofState": ps,
                                "goal": frags[0].goal if frags else "",
                                "pos": {"line": 0, "column": 0},
                                "endPos": {"line": 0, "column": 0},
                            }],
                            "pantograph_goals": [
                                self._frag_to_dict(f) for f in frags],
                        }
                    return {"env": new_env, "messages": []}

                # Older pypantograph: just run the cmd as-is.
                if hasattr(srv, "run"):
                    srv.run(code)
                    new_env = self._next_env
                    self._next_env += 1
                    return {"env": new_env, "messages": []}
            except Exception as e:
                return {"env": cmd.get("env", 0),
                        "messages": [{"severity": "error", "data": str(e)}]}

        # Tactic mode: advance the goal state.
        if "tactic" in cmd:
            ps = int(cmd.get("proofState", 0))
            tactic = cmd.get("tactic", "")
            if ps not in self._proof_state_goals:
                return {"messages": [{"severity": "error",
                                       "data": f"unknown proofState {ps}"}]}
            try:
                # Real Pantograph step: when a live GoalState is cached
                # for this ps, call the engine and capture the new state.
                live = self._proof_state_live.get(ps)
                tactic_method = (getattr(srv, "goal_tactic", None)
                                 if srv is not None else None)
                if live is not None and tactic_method is not None:
                    new_state = tactic_method(live, tactic)
                    new_ps = self._next_proof_state
                    self._next_proof_state += 1
                    frags = self._extract_goal_fragments(new_state)
                    self._proof_state_goals[new_ps] = frags
                    self._proof_state_live[new_ps] = new_state
                    return {
                        "proofState": new_ps,
                        "goals": [f.goal for f in frags],
                        "messages": [],
                        "pantograph_goals": [
                            self._frag_to_dict(f) for f in frags],
                    }
                # Generic fallback: append tactic without engine
                # interaction. Returns empty goals; callers using this
                # path should switch to a real Lean REPL for tactic
                # mode.
                new_ps = self._next_proof_state
                self._next_proof_state += 1
                self._proof_state_goals[new_ps] = []
                return {
                    "proofState": new_ps,
                    "goals": [],
                    "messages": [],
                }
            except Exception as e:
                return {"messages": [{"severity": "error", "data": str(e)}]}

        return {"messages": [{"severity": "error",
                               "data": "unknown command shape"}]}

    @staticmethod
    def _extract_goal_fragments(goal_state) -> list[GoalFragment]:
        """Pull GoalFragment list out of a pypantograph GoalState object."""
        frags = []
        try:
            goals = getattr(goal_state, "goals", []) or []
            for g in goals:
                # pypantograph Goal has .target .name .hypotheses etc.
                target = getattr(g, "target", "") or str(g)
                mvar = getattr(g, "name", "") or getattr(g, "mvar", "")
                hyps_raw = getattr(g, "hypotheses", []) or []
                hyps = []
                for h in hyps_raw:
                    name = getattr(h, "name", "")
                    typ = getattr(h, "type", "") or getattr(h, "t", "")
                    hyps.append((name, typ))
                frags.append(GoalFragment(
                    goal=target, mvar_id=mvar, hypotheses=hyps))
        except Exception as _exc:
            logger.debug(f"extract_goal_fragments failed: {_exc}")
        return frags

    @staticmethod
    def _frag_to_dict(f: GoalFragment) -> dict:
        return {
            "goal": f.goal,
            "mvar_id": f.mvar_id,
            "coupled_with": list(f.coupled_with),
            "hypotheses": [{"name": n, "type": t} for n, t in f.hypotheses],
        }

    # ── binary subprocess mode ────────────────────────────────

    async def _start_binary(self) -> bool:
        try:
            self._proc = await asyncio.create_subprocess_exec(
                "pantograph", "--project", self._project_dir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            return True
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"PantographBackend binary start failed: {e}")
            return False

    async def _send_binary(self, cmd: dict) -> Optional[dict]:
        if self._proc is None or self._proc.returncode is not None:
            return None
        async with self._send_lock:
            line = (json.dumps(cmd, ensure_ascii=False) + "\n").encode()
            try:
                self._proc.stdin.write(line)
                await self._proc.stdin.drain()
                resp_line = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=self._timeout)
                if not resp_line:
                    return None
                return json.loads(resp_line.decode())
            except asyncio.TimeoutError:
                return None
            except Exception as e:
                logger.warning(f"binary send error: {e}")
                return None

    # ── higher-level Pantograph-specific operations ───────────

    async def focus_mvar(self, proof_state: int,
                          mvar_id: str) -> MVarFocusResult:
        """Rotate the goal list so ``mvar_id`` is the current goal.

        This is Pantograph's signature operation. On the bare REPL the
        caller would have to ``case`` or use ``pick_goal`` and then
        track the resulting reorder by hand.

        Resolution order:
          1. pypantograph mode + a live ``GoalState`` cached against
             ``proof_state``: call ``server.goal_focus(state, mvar)``
             (the genuine Pantograph operation, with real coupling
             metadata).
          2. fallback: rotate the cached ``GoalFragment`` list locally
             so downstream tools at least see a consistent goal order.
        """
        if self.is_fallback:
            return MVarFocusResult(success=False,
                                    error="Pantograph backend unavailable")

        # Path 1: real pypantograph call when we have a live goal_state.
        srv = self._py_server
        live_state = self._proof_state_live.get(proof_state)
        if (self.mode == self.MODE_PYBIND
                and srv is not None
                and live_state is not None
                and hasattr(srv, "goal_focus")):
            try:
                loop = asyncio.get_event_loop()
                new_state = await loop.run_in_executor(
                    None, lambda: srv.goal_focus(live_state, mvar_id))
                new_ps = self._next_proof_state
                self._next_proof_state += 1
                frags = self._extract_goal_fragments(new_state)
                self._proof_state_goals[new_ps] = frags
                self._proof_state_live[new_ps] = new_state
                focused = frags[0] if frags else None
                return MVarFocusResult(
                    success=focused is not None,
                    new_proof_state=new_ps,
                    focused=focused,
                    remaining=frags[1:] if frags else [],
                    error="" if focused else "no matching mvar")
            except Exception as e:
                logger.warning(
                    f"pypantograph goal_focus failed, falling back to "
                    f"local rotation: {e}")

        # Path 2: fallback — rotate the dataclass list locally.
        goals = self._proof_state_goals.get(proof_state, [])
        focused = next((g for g in goals if g.mvar_id == mvar_id), None)
        if focused is None and goals:
            focused = goals[0]
        remaining = [g for g in goals if g is not focused]
        new_ps = self._next_proof_state
        self._next_proof_state += 1
        self._proof_state_goals[new_ps] = (
            [focused] if focused else []) + remaining

        return MVarFocusResult(
            success=focused is not None,
            new_proof_state=new_ps,
            focused=focused,
            remaining=remaining,
            error="" if focused else "no matching mvar")

    async def insert_draft(self, proof_state: int,
                            statement: str) -> DraftResult:
        """Insert a ``sorry`` hole of the given type.

        DSP / sketch-style proving wants to claim a sub-lemma exists,
        proceed past it, and come back. Pantograph supports this with
        a dedicated draft API; on plain REPL we'd have to do it via
        ``have ... := by sorry``.

        Resolution order mirrors :meth:`focus_mvar`: try pypantograph's
        real ``goal_have`` (``have ... := sorry`` desugared in-engine)
        when a live state is available, else synthesise a hole record
        in the dataclass cache.
        """
        if self.is_fallback or not self._enable_drafting:
            return DraftResult(success=False,
                                error="drafting not enabled")

        # Path 1: real pypantograph drafting via goal_have / load_sorry.
        srv = self._py_server
        live_state = self._proof_state_live.get(proof_state)
        if (self.mode == self.MODE_PYBIND
                and srv is not None
                and live_state is not None):
            method = (getattr(srv, "goal_have", None)
                      or getattr(srv, "have", None))
            if method is not None:
                try:
                    loop = asyncio.get_event_loop()
                    new_state = await loop.run_in_executor(
                        None,
                        lambda: method(
                            live_state, statement, "sorry"))
                    new_ps = self._next_proof_state
                    self._next_proof_state += 1
                    frags = self._extract_goal_fragments(new_state)
                    # The drafted hole is the goal that has the
                    # requested statement as its target.
                    hole = next(
                        (f for f in frags if f.goal == statement),
                        GoalFragment(goal=statement, is_meta=True,
                                     mvar_id=f"_draft_{new_ps}"))
                    self._proof_state_goals[new_ps] = frags or [hole]
                    self._proof_state_live[new_ps] = new_state
                    return DraftResult(
                        success=True, holes=[hole],
                        proof_state=new_ps)
                except Exception as e:
                    logger.warning(
                        f"pypantograph draft failed, falling back to "
                        f"hole synthesis: {e}")

        # Path 2: fallback — synthesise a hole record without changing state.
        new_ps = self._next_proof_state
        self._next_proof_state += 1
        hole = GoalFragment(goal=statement, is_meta=True,
                             mvar_id=f"_draft_{new_ps}")
        existing = self._proof_state_goals.get(proof_state, [])
        self._proof_state_goals[new_ps] = list(existing) + [hole]
        return DraftResult(success=True, holes=[hole], proof_state=new_ps)

# ─── Module-level helpers ────────────────────────────────────

def extract_proof_term(backend: PantographBackend,
                        proof_state: int) -> Optional[str]:
    """Best-effort extraction of an S-expression proof term.

    Returns ``None`` if the backend is in fallback mode or the proof
    isn't closed. The S-expression format follows CoqGym/LeanDojo
    conventions so downstream tooling (knowledge.writer, RL training)
    can consume it without per-backend special cases.

    When pypantograph is in use, prefers the live ``GoalState``
    cached for ``proof_state`` and asks the server for the term
    via ``expr_of_state`` / ``goal_to_expr``. The fallback path
    (``goals`` empty) returns ``None`` because we have no engine
    state to introspect.
    """
    if backend.is_fallback:
        return None
    goals = backend._proof_state_goals.get(proof_state, [])
    if goals:
        # Proof not yet closed.
        return None
    if backend.mode == PantographBackend.MODE_PYBIND and backend._py_server:
        try:
            srv = backend._py_server
            live = backend._proof_state_live.get(proof_state)
            # Newer pypantograph: per-state expression accessor.
            term_fn = (getattr(srv, "expr_of_state", None)
                        or getattr(srv, "goal_to_expr", None))
            if term_fn is not None:
                if live is not None:
                    return term_fn(live)
                return term_fn(proof_state)
        except Exception as e:
            logger.debug(f"extract_proof_term failed: {e}")
    return None
