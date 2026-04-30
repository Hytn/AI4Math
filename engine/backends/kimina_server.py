"""engine/backends/kimina_server.py — Kimina Lean Server backend.

Integrates the open-source `kimina-lean-server <https://github.com/project-numina/kimina-lean-server>`_
released by Numina/Kimi (Dos Santos et al., 2025, arXiv:2504.21230) as a
first-class verification backend.

What this gives us
------------------

The bare ``LocalTransport`` already in ``engine.transport`` runs ONE Lean 4
REPL process per session. That works fine for an interactive prover but
scales poorly during RL roll-outs or full benchmark sweeps (miniF2F has
488 problems × 32 samples = 15,616 verifications per pass@k cell).

Kimina Lean Server addresses this with three architectural choices we now
mirror:

1. **Server-side REPL pool with REST front-end** — a FastAPI process
   maintains N independent Lean 4 REPL workers and load-balances across
   them. The client sends ``POST /api/verify`` with a list of proofs and
   gets back a list of results. A single TCP connection replaces N stdio
   pipes, and the server can run on a different machine from the client.

2. **LRU import cache** — every distinct preamble (``import Mathlib``,
   ``import Mathlib.Topology.Basic``, etc.) is loaded once into a worker
   and that worker's ``env_id`` is reused across every subsequent request
   that uses the same preamble. The Kimina paper reports 1.5–2× throughput
   gains from this alone.

3. **Infotree extraction** — when a proof succeeds, the server walks the
   Lean infotree and emits each tactic that fired plus the goal state
   before and after. This is the data shape the NuminaMath-LEAN dataset
   was built from, and it's what we want to feed back into our knowledge
   pyramid (see ``knowledge.writer`` ).

API surface
-----------

This module exposes three things:

* :class:`KiminaServerClient` — thin async HTTP client. Talks to the
  upstream server's REST schema (``/api/check``, ``/api/verify``,
  ``/api/extract_tactics``). Backward compatible with both the v1 and
  v2 server.
* :class:`KiminaServerBackend` — implements ``REPLTransport`` so it can
  drop straight into ``AsyncLeanPool``. Routes ``cmd``/``tactic``
  requests through the REST API.
* :class:`BatchVerifyRequest` / :class:`BatchVerifyResult` — dataclasses
  for the batch verify call which has no analog in the stdio protocol.

Falling back gracefully
-----------------------

If ``aiohttp`` is not installed, or the server isn't reachable, the
backend silently degrades to ``is_fallback=True`` like every other
transport. No imports are performed at module load time so this file
is always safe to import in CI.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from engine.transport import REPLTransport, TransportStats

logger = logging.getLogger(__name__)


# ─── Wire-format dataclasses ─────────────────────────────────


@dataclass
class BatchVerifyRequest:
    """One entry in a batch verify call.

    Mirrors the JSON the Kimina server accepts on ``POST /api/verify``::

        [
          {
            "id":      "req-1",
            "proof":   "theorem t : 1+1=2 := by norm_num",
            "preamble": "import Mathlib",
            "last_used_id": "req-0"
          },
          ...
        ]

    The ``id`` is echoed back so the client can correlate responses
    independent of arrival order — useful when the server processes them
    in parallel across worker REPLs.

    ``last_used_id`` is the Kimina v2.x extension for incremental
    preamble replay: the server keeps the env_id from a previous
    request so a follow-up that shares the same preamble (or extends
    it minimally) can skip the heavy Mathlib re-import. The client
    fills it from the most recent response with the matching preamble
    hash.
    """
    id: str
    proof: str
    preamble: str = "import Mathlib"
    timeout_seconds: int = 120
    last_used_id: Optional[str] = None

    def to_wire(self) -> dict:
        d = {
            "id": self.id,
            "proof": self.proof,
            "preamble": self.preamble,
            "timeout": self.timeout_seconds,
        }
        if self.last_used_id:
            d["last_used_id"] = self.last_used_id
        return d


@dataclass
class TacticTrace:
    """One element of the infotree-extracted tactic sequence.

    Each successful proof can be broken into a list of these — what
    tactic fired, what goal it had before, what came after. This feeds
    ``knowledge.writer`` and is the grain at which we emit RL training
    signal.
    """
    tactic: str
    goal_before: str
    goals_after: list[str] = field(default_factory=list)
    is_proof_complete: bool = False
    line: int = -1
    column: int = -1


@dataclass
class BatchVerifyResult:
    """One entry in a batch verify response."""
    id: str
    success: bool
    error_messages: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    tactic_trace: list[TacticTrace] = field(default_factory=list)
    has_sorry: bool = False
    raw: dict = field(default_factory=dict)
    # Kimina v2.x replay handle. The server returns a stable id we can
    # quote on the next request as ``last_used_id`` to skip preamble
    # re-import. ``""`` when the server didn't provide one.
    server_id: str = ""

    @staticmethod
    def from_wire(d: dict) -> BatchVerifyResult:
        traces = []
        for t in d.get("tactic_trace", []):
            traces.append(TacticTrace(
                tactic=t.get("tactic", ""),
                goal_before=t.get("goal_before", ""),
                goals_after=t.get("goals_after", []) or [],
                is_proof_complete=bool(t.get("is_proof_complete", False)),
                line=t.get("line", -1),
                column=t.get("column", -1),
            ))
        return BatchVerifyResult(
            id=d.get("id", ""),
            success=bool(d.get("success", False)),
            error_messages=d.get("errors", []) or d.get("error_messages", []),
            elapsed_ms=int(d.get("elapsed_ms", 0)),
            tactic_trace=traces,
            has_sorry=bool(d.get("has_sorry", False)),
            raw=d,
            server_id=str(d.get("server_id", "")
                          or d.get("env_id", "")
                          or d.get("snapshot_id", "")),
        )


# ─── HTTP client ─────────────────────────────────────────────


def _preamble_key(preamble: str) -> str:
    """Stable, short hash key for ``preamble`` strings.

    Shared between :class:`KiminaServerClient` (replay-handle cache)
    and :class:`_ImportCache` (env-id LRU). Keying on a hash means
    very long preambles don't blow up memory, and equality is exact —
    two preambles with the same content will collide on purpose.
    """
    return hashlib.blake2b(preamble.encode(), digest_size=16).hexdigest()


class KiminaServerClient:
    """Async HTTP client for the Kimina Lean Server.

    Lazy-imports ``aiohttp`` so the module is safe to import even when
    the dependency is missing. If the import fails, ``available`` is
    ``False`` and the caller should fall back to a stdio backend.

    Auth: the upstream server can be deployed behind a header-auth
    proxy. Pass ``api_key=...`` to add an ``Authorization: Bearer ...``
    header to every request.
    """

    DEFAULT_BASE_URL = "http://localhost:8000"
    DEFAULT_TIMEOUT = 300  # generous; Kimina caches preambles so this is rare

    def __init__(self,
                 base_url: str = None,
                 api_key: str = None,
                 timeout_seconds: int = None,
                 max_concurrent: int = 32):
        self.base_url = (base_url
                         or os.environ.get("KIMINA_SERVER_URL")
                         or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("KIMINA_API_KEY", "")
        self.timeout_seconds = timeout_seconds or self.DEFAULT_TIMEOUT
        self._sem = asyncio.Semaphore(max_concurrent)
        self._session = None
        self._aiohttp = None
        self._available = self._lazy_import_aiohttp()
        # preamble-hash → most recent successful server_id. Used to
        # populate ``last_used_id`` on the NEXT request with the same
        # preamble so the server can skip the import phase. v2.x Kimina
        # exposes this; v1.x ignores the field harmlessly.
        self._replay_handles: dict[str, str] = {}

    # ── lifecycle ──────────────────────────────────────────────

    def _lazy_import_aiohttp(self) -> bool:
        try:
            import aiohttp  # noqa: F401
            self._aiohttp = aiohttp
            return True
        except ImportError:
            logger.info(
                "KiminaServerClient: aiohttp not installed — "
                "client unavailable (run `pip install aiohttp` to enable)")
            return False

    @property
    def available(self) -> bool:
        return self._available

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def _ensure_session(self):
        if not self._available:
            return
        if self._session is None or self._session.closed:
            timeout = self._aiohttp.ClientTimeout(total=self.timeout_seconds)
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._session = self._aiohttp.ClientSession(
                timeout=timeout, headers=headers)

    async def close(self):
        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception as _exc:
                logger.debug(f"Suppressed close exception: {_exc}")
        self._session = None

    # ── high-level operations ──────────────────────────────────

    async def health_check(self) -> bool:
        """GET /health — confirm the server is up and import cache is warm."""
        if not self._available:
            return False
        await self._ensure_session()
        try:
            async with self._sem:
                async with self._session.get(
                        f"{self.base_url}/health") as resp:
                    return resp.status == 200
        except Exception as e:
            logger.debug(f"KiminaServerClient.health_check: {e}")
            return False

    async def check_one(self, proof: str,
                         preamble: str = "import Mathlib") -> BatchVerifyResult:
        """Single-proof convenience wrapper.

        Internally still goes through the batch endpoint so the server can
        deduplicate against its preamble cache.
        """
        results = await self.verify_batch([
            BatchVerifyRequest(id="check-0", proof=proof, preamble=preamble)
        ])
        if results:
            return results[0]
        return BatchVerifyResult(id="check-0", success=False,
                                 error_messages=["server returned empty"])

    async def verify_batch(
            self,
            requests: list[BatchVerifyRequest]) -> list[BatchVerifyResult]:
        """POST /api/verify with a batch of proofs.

        The server will load-balance across its internal worker REPLs
        and reuse the import cache wherever ``preamble`` matches a
        previous request.

        For Kimina v2.x replay: any request without an explicit
        ``last_used_id`` gets one auto-filled from the most recent
        successful response that shared its preamble. v1.x servers
        ignore the field, so this is always safe to send.
        """
        if not self._available:
            return [BatchVerifyResult(id=r.id, success=False,
                                       error_messages=["aiohttp unavailable"])
                    for r in requests]
        if not requests:
            return []

        # Auto-fill last_used_id from the per-preamble replay cache.
        for r in requests:
            if not r.last_used_id:
                handle = self._replay_handles.get(_preamble_key(r.preamble))
                if handle:
                    r.last_used_id = handle

        await self._ensure_session()
        payload = {"requests": [r.to_wire() for r in requests]}

        try:
            async with self._sem:
                async with self._session.post(
                        f"{self.base_url}/api/verify",
                        data=json.dumps(payload)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(
                            f"KiminaServer /api/verify HTTP {resp.status}: "
                            f"{body[:200]}")
                        return [BatchVerifyResult(
                            id=r.id, success=False,
                            error_messages=[f"server HTTP {resp.status}"])
                            for r in requests]
                    body = await resp.json()
        except asyncio.TimeoutError:
            return [BatchVerifyResult(id=r.id, success=False,
                                       error_messages=["server timeout"])
                    for r in requests]
        except Exception as e:
            logger.warning(f"KiminaServer /api/verify error: {e}")
            return [BatchVerifyResult(id=r.id, success=False,
                                       error_messages=[f"client error: {e}"])
                    for r in requests]

        # The server returns either {"results": [...]} (v2) or [...] (v1).
        items = body.get("results", body) if isinstance(body, dict) else body
        if not isinstance(items, list):
            return [BatchVerifyResult(id=r.id, success=False,
                                       error_messages=["malformed response"])
                    for r in requests]

        # Re-order by id to keep correlation safe.
        by_id = {it.get("id"): it for it in items if isinstance(it, dict)}
        out = []
        for r in requests:
            it = by_id.get(r.id)
            if it is None:
                out.append(BatchVerifyResult(
                    id=r.id, success=False,
                    error_messages=["no result for id"]))
            else:
                result = BatchVerifyResult.from_wire(it)
                # Memoize the replay handle for subsequent requests
                # with the same preamble.
                if result.success and result.server_id:
                    self._replay_handles[_preamble_key(r.preamble)] = (
                        result.server_id)
                out.append(result)
        return out

    async def extract_tactics(
            self, proof: str,
            preamble: str = "import Mathlib") -> list[TacticTrace]:
        """POST /api/extract_tactics — get the infotree-extracted trace.

        Use this when a proof has already been verified and we want the
        per-tactic goal states for training data, knowledge deposit, or
        replay debugging.
        """
        if not self._available:
            return []
        await self._ensure_session()
        try:
            async with self._sem:
                async with self._session.post(
                        f"{self.base_url}/api/extract_tactics",
                        data=json.dumps({"proof": proof,
                                          "preamble": preamble})) as resp:
                    if resp.status != 200:
                        return []
                    body = await resp.json()
        except Exception as e:
            logger.debug(f"extract_tactics error: {e}")
            return []

        traces = body.get("trace", []) if isinstance(body, dict) else body
        out = []
        for t in traces:
            if not isinstance(t, dict):
                continue
            out.append(TacticTrace(
                tactic=t.get("tactic", ""),
                goal_before=t.get("goal_before", ""),
                goals_after=t.get("goals_after", []) or [],
                is_proof_complete=bool(t.get("is_proof_complete", False)),
                line=t.get("line", -1),
                column=t.get("column", -1),
            ))
        return out


# ─── Local LRU import cache (mirrors server-side cache) ─────


class _ImportCache:
    """Small client-side memo of ``preamble → env_id`` mappings.

    The server has its own LRU; this is just a hint so we don't re-send
    ``import Mathlib`` on every single request. We never make decisions
    on a cache miss — we always fall back to issuing the request — so
    eventual consistency with the server is enough.
    """

    def __init__(self, maxsize: int = 64):
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(preamble: str) -> str:
        # Hash so very long preambles don't blow up memory; collision
        # at this length is astronomically unlikely.
        return hashlib.blake2b(preamble.encode(), digest_size=16).hexdigest()

    def get(self, preamble: str) -> Optional[int]:
        k = self._key(preamble)
        if k in self._cache:
            self.hits += 1
            self._cache.move_to_end(k)
            return self._cache[k]
        self.misses += 1
        return None

    def put(self, preamble: str, env_id: int):
        k = self._key(preamble)
        self._cache[k] = env_id
        self._cache.move_to_end(k)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._cache),
            "hit_rate": round(self.hits / max(1, total), 4),
        }


# ─── Backend = HTTPTransport implementation ─────────────────


class KiminaServerBackend(REPLTransport):
    """Adapter that makes a Kimina Lean Server look like a ``REPLTransport``.

    The bare REPL protocol speaks ``cmd``/``tactic`` JSON over stdio. The
    Kimina server speaks REST. This class translates between them so the
    existing ``AsyncLeanPool`` can use Kimina with no other changes.

    Translation rules
    -----------------

    ``{"cmd": "import Mathlib", "env": 0}``
        Becomes a no-op locally and stores ``preamble = "import Mathlib"``
        in the session — actual import happens server-side, lazily on the
        first ``proof`` call (when the server applies its own LRU cache).
        We synthesise an ``env_id`` so the caller's bookkeeping is
        consistent with the stdio backend.

    ``{"cmd": "<theorem with body>", "env": <env_id>}``
        Becomes ``POST /api/verify`` with that theorem as the proof body
        and the cached preamble. Errors are returned in the response.

    ``{"tactic": "...", "proofState": <id>}``
        The Kimina server (as of v2.0) does not expose a true tactic-mode
        endpoint — it verifies whole proofs. We accumulate tactics into a
        per-session sequence and re-verify the running script on each
        call. The new ``proofState`` is just an integer counter.

    The trade-off: tactic-level mode is slower on Kimina than on a local
    long-running REPL, but whole-proof verify is much faster (and that's
    most of our compute). The Profile system lets users choose: pick
    Kimina for ``whole_proof_repair``-family profiles, pick a local REPL
    for step-level profiles.
    """

    def __init__(self,
                 base_url: str = None,
                 api_key: str = None,
                 default_preamble: str = "import Mathlib",
                 timeout_seconds: int = 300,
                 max_concurrent: int = 16):
        self._client = KiminaServerClient(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_concurrent=max_concurrent)
        self._default_preamble = default_preamble
        self._stats = TransportStats()
        self._alive = False
        self._fallback = not self._client.available
        # Per-session bookkeeping
        self._next_env_id = 1
        self._next_proof_state = 1
        # env_id → preamble string
        self._env_preambles: dict[int, str] = {0: default_preamble}
        # proofState → (theorem_header, tactics_so_far)
        self._proof_state_scripts: dict[int, tuple[str, list[str]]] = {}
        self._import_cache = _ImportCache()

    # ── REPLTransport ABC ─────────────────────────────────────

    async def start(self) -> bool:
        if not self._client.available:
            logger.warning(
                "KiminaServerBackend: aiohttp missing, running in "
                "fallback mode (all requests will fail-soft)")
            self._alive = True
            return True

        ok = await self._client.health_check()
        if not ok:
            logger.warning(
                f"KiminaServerBackend: server at {self._client.base_url} "
                "did not respond to /health — degrading to fallback. "
                "Start the server with: "
                "docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0")
            self._alive = True
            self._fallback = True
            return True

        logger.info(
            f"KiminaServerBackend: connected to {self._client.base_url}")
        self._alive = True
        self._fallback = False
        return True

    async def send(self, cmd: dict) -> Optional[dict]:
        """Translate REPL-protocol JSON to REST and back."""
        if self._fallback:
            self._stats.record_failure()
            return None

        t0 = time.monotonic()

        # Tactic mode: replay the running script with the new tactic appended.
        if "tactic" in cmd and cmd.get("proofState") is not None:
            resp = await self._handle_tactic(cmd)
            self._record(t0, success=resp is not None)
            return resp

        # Command mode
        if "cmd" in cmd:
            resp = await self._handle_command(cmd)
            self._record(t0, success=resp is not None)
            return resp

        return None

    async def close(self):
        await self._client.close()
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_fallback(self) -> bool:
        return self._fallback

    # ── command-mode handler ──────────────────────────────────

    async def _handle_command(self, cmd: dict) -> Optional[dict]:
        code = (cmd.get("cmd") or "").strip()
        env = int(cmd.get("env", 0))

        # Pure import command (no theorem body): cache the preamble locally.
        if self._is_pure_import(code):
            new_env = self._next_env_id
            self._next_env_id += 1
            preamble = self._compose_preamble(env, code)
            self._env_preambles[new_env] = preamble
            cached = self._import_cache.get(preamble)
            if cached is None:
                # Fire-and-forget warm-up so the server caches it for us.
                # Errors are non-fatal — server will lazily import on first
                # proof anyway.
                asyncio.create_task(self._warmup_preamble(preamble, new_env))
            return {"env": new_env, "messages": []}

        # Theorem-with-body: route through batch verify.
        preamble = self._env_preambles.get(env, self._default_preamble)
        req_id = f"v-{int(time.time()*1000)}-{self._next_env_id}"
        result = await self._client.check_one(proof=code, preamble=preamble)
        new_env = self._next_env_id
        self._next_env_id += 1
        self._env_preambles[new_env] = preamble

        return {
            "env": new_env,
            "messages": [
                {"severity": "error", "data": e}
                for e in result.error_messages
            ],
            "sorries": [] if result.success else [],
            # Surface tactic_trace under a non-standard key so Kimina-aware
            # callers can use it without confusing stdio-aware ones.
            "kimina_tactic_trace": [
                {"tactic": t.tactic,
                 "goal_before": t.goal_before,
                 "goals_after": t.goals_after}
                for t in result.tactic_trace
            ],
        }

    async def _warmup_preamble(self, preamble: str, env_id: int):
        """Best-effort preamble warmup — never raises."""
        try:
            # Verify a trivial theorem under this preamble. Server caches
            # the preamble's env_id under its own LRU.
            warm = "theorem _ai4math_warmup : True := trivial"
            await self._client.check_one(proof=warm, preamble=preamble)
            self._import_cache.put(preamble, env_id)
        except Exception as e:
            logger.debug(f"warmup_preamble({preamble[:40]!r}) failed: {e}")

    # ── tactic-mode handler ────────────────────────────────────

    async def _handle_tactic(self, cmd: dict) -> Optional[dict]:
        tactic = (cmd.get("tactic") or "").strip()
        ps = int(cmd.get("proofState", 0))
        if ps not in self._proof_state_scripts:
            return {
                "messages": [{"severity": "error",
                              "data": f"unknown proofState {ps}"}],
                "goals": [],
            }

        header, tactics = self._proof_state_scripts[ps]
        new_tactics = list(tactics) + [tactic]
        full_proof = self._assemble_proof(header, new_tactics)

        result = await self._client.check_one(
            proof=full_proof, preamble=self._default_preamble)

        if not result.success:
            return {
                "proofState": ps,
                "goals": ["⊢ <unknown — Kimina batch mode does not "
                          "expose mid-proof goal state>"],
                "messages": [{"severity": "error", "data": e}
                             for e in result.error_messages],
            }

        # Whole proof compiled — assume the new tactic closed the goal
        # OR we're still in mid-proof (the server can't tell us). Use the
        # tactic trace if present.
        new_ps = self._next_proof_state
        self._next_proof_state += 1
        self._proof_state_scripts[new_ps] = (header, new_tactics)

        if result.tactic_trace:
            last = result.tactic_trace[-1]
            return {
                "proofState": new_ps,
                "goals": last.goals_after,
                "messages": [],
            }
        return {"proofState": new_ps, "goals": [], "messages": []}

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _is_pure_import(code: str) -> bool:
        # Treat a snippet as a "pure import / preamble" if every non-empty
        # line is either an import, an `open`, or `set_option ...` (ie. no
        # theorem/def/example).
        lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
        if not lines:
            return False
        for ln in lines:
            if not (ln.startswith("import ")
                    or ln.startswith("open ")
                    or ln.startswith("set_option ")
                    or ln.startswith("--")):
                return False
        return True

    def _compose_preamble(self, parent_env: int, new_code: str) -> str:
        parent = self._env_preambles.get(parent_env, "")
        if parent and not parent.endswith("\n"):
            parent = parent + "\n"
        return parent + new_code

    @staticmethod
    def _assemble_proof(header: str, tactics: list[str]) -> str:
        header = header.strip()
        if ":=" in header:
            header = header[:header.index(":=")].rstrip()
        body = "\n  ".join(tactics)
        return f"{header} := by\n  {body}"

    def _record(self, t0: float, success: bool):
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if success:
            self._stats.record_success(elapsed_ms)
        else:
            self._stats.record_failure()

    def get_stats(self) -> dict:
        d = self._stats.to_dict()
        d["base_url"] = self._client.base_url
        d["import_cache"] = self._import_cache.stats()
        d["fallback"] = self._fallback
        return d

    # ── batch entrypoint exposed for ToolKit.BATCH_VERIFY ──────

    async def verify_batch(
            self,
            proofs: list[str],
            preamble: str = None) -> list[BatchVerifyResult]:
        """Public API for callers that want true batch behaviour.

        ``AsyncLeanPool.send()`` only knows about per-request semantics;
        the batch path goes around it for higher throughput.
        """
        preamble = preamble or self._default_preamble
        reqs = [BatchVerifyRequest(id=f"b-{i}", proof=p, preamble=preamble)
                for i, p in enumerate(proofs)]
        return await self._client.verify_batch(reqs)
