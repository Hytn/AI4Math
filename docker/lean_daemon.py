#!/usr/bin/env python3
"""docker/lean_daemon.py — Lean4 REPL daemon with persistent sessions

Serves Lean4 verification over Unix domain socket with two modes:

1. REPL mode (default): Manages persistent lean4-repl processes.
   Each client connection gets a dedicated REPL process with full
   env_id/proofState state tracking. Commands are forwarded directly
   to the REPL, enabling interactive tactic-level proving.

2. Compile mode (fallback): Single-shot `lake env lean --stdin` compilation.
   Used when lean4-repl binary is not available.

Protocol: JSON Lines over Unix socket
  Request:  {"cmd": "import Mathlib", "env": 0, "id": "req-001"}
            {"tactic": "simp", "proofState": 3, "id": "req-002"}
  Response: {"id": "req-001", "result": {<lean4-repl response>}, "elapsed_ms": 42}
            {"id": "req-002", "result": {<lean4-repl response>}, "elapsed_ms": 15}

Usage:
  python3 lean_daemon.py [--socket PATH] [--mode repl|compile]
"""
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lean_daemon")

SOCKET_PATH = os.environ.get("LEAN_SOCKET", "/workspace/exchange/lean.sock")
PROJECT_DIR = os.environ.get("LEAN_PROJECT_DIR", "/workspace/lean-project")
MAX_CONCURRENT = int(os.environ.get("LEAN_MAX_CONCURRENT", "8"))
TIMEOUT = int(os.environ.get("LEAN_TIMEOUT", "120"))
REPL_BINARY = os.environ.get("LEAN_REPL_BINARY", "")

# ─── Detect REPL binary ──────────────────────────────────────

def find_repl_binary() -> str:
    """Find lean4-repl binary."""
    if REPL_BINARY and os.path.isfile(REPL_BINARY):
        return REPL_BINARY
    candidates = [
        os.path.join(PROJECT_DIR, ".lake/build/bin/repl"),
        os.path.join(PROJECT_DIR, "lake-packages/repl/.lake/build/bin/repl"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return ""


# ─── REPL Session Pool ───────────────────────────────────────

class REPLSession:
    """A single lean4-repl process."""

    def __init__(self, session_id: int, binary: str, project_dir: str):
        self.session_id = session_id
        self._binary = binary
        self._project_dir = project_dir
        self._process = None
        self._lock = asyncio.Lock()
        self.total_requests = 0

    async def start(self) -> bool:
        try:
            self._process = await asyncio.create_subprocess_exec(
                self._binary,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._project_dir,
            )
            logger.info(f"REPL session {self.session_id} started "
                        f"(pid={self._process.pid})")
            return True
        except Exception as e:
            logger.error(f"REPL session {self.session_id} start failed: {e}")
            return False

    async def send(self, request: dict) -> dict:
        """Forward a request to the REPL and return its response."""
        async with self._lock:
            self.total_requests += 1

            if not self._process or self._process.returncode is not None:
                return {"messages": [{"severity": "error",
                                      "data": "REPL process not running"}]}

            # Build wire request (strip our 'id' field)
            wire = {k: v for k, v in request.items()
                    if k not in ("id",) and v is not None}
            data = (json.dumps(wire, ensure_ascii=False) + "\n").encode()

            try:
                self._process.stdin.write(data)
                await self._process.stdin.drain()

                line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=TIMEOUT)

                if not line:
                    return {"messages": [{"severity": "error",
                                          "data": "REPL returned empty"}]}

                return json.loads(line.decode())

            except asyncio.TimeoutError:
                return {"messages": [{"severity": "error",
                                      "data": f"Timeout after {TIMEOUT}s"}]}
            except Exception as e:
                return {"messages": [{"severity": "error",
                                      "data": str(e)}]}

    async def close(self):
        if self._process and self._process.returncode is None:
            try:
                self._process.kill()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except Exception as _exc:
                logger.debug(f"Suppressed exception: {_exc}")

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None


class REPLPool:
    """Pool of REPL sessions with preamble prewarming.

    Warm cache: sessions are pre-loaded with 'import Mathlib' so the
    first tactic request pays ~0ms import cost instead of ~10-30s.
    """

    PREAMBLE = os.environ.get("LEAN_PREAMBLE", "import Mathlib")

    def __init__(self, binary: str, project_dir: str, max_sessions: int):
        self._binary = binary
        self._project_dir = project_dir
        self._max_sessions = max_sessions
        self._sessions: dict[int, REPLSession] = {}
        self._warm_queue: asyncio.Queue = asyncio.Queue()
        self._next_id = 0
        self._semaphore = asyncio.Semaphore(max_sessions)
        self._prewarm_task = None
        self._warm_env_ids: dict[int, int] = {}  # session_id → prewarmed env_id

    async def prewarm(self, count: int = 0):
        """Pre-create and warm sessions with preamble import.

        Args:
            count: Number of sessions to prewarm (0 = max_sessions)
        """
        count = count or self._max_sessions
        count = min(count, self._max_sessions)
        logger.info(f"REPLPool: prewarming {count} sessions...")
        tasks = [self._create_warm_session() for _ in range(count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ok = sum(1 for r in results if r is True)
        logger.info(f"REPLPool: prewarmed {ok}/{count} sessions")

    async def _create_warm_session(self) -> bool:
        """Create a session and prewarm it with the preamble."""
        await self._semaphore.acquire()
        sid = self._next_id
        self._next_id += 1
        session = REPLSession(sid, self._binary, self._project_dir)
        ok = await session.start()
        if not ok:
            self._semaphore.release()
            return False

        # Send preamble
        if self.PREAMBLE:
            resp = await session.send({"cmd": self.PREAMBLE, "env": 0})
            env_id = resp.get("env", 0) if resp else 0
            self._warm_env_ids[sid] = env_id
            logger.info(
                f"REPLPool: session {sid} prewarmed (env_id={env_id})")

        self._sessions[sid] = session
        await self._warm_queue.put(session)
        return True

    async def acquire(self) -> REPLSession:
        """Get a prewarmed session, or create one on demand."""
        try:
            session = self._warm_queue.get_nowait()
            if session.is_alive:
                return session
        except asyncio.QueueEmpty as _exc:
            logger.debug(f"Suppressed exception: {_exc}")

        # Fallback: create on demand
        await self._semaphore.acquire()
        sid = self._next_id
        self._next_id += 1
        session = REPLSession(sid, self._binary, self._project_dir)
        ok = await session.start()
        if ok:
            self._sessions[sid] = session
            return session
        else:
            self._semaphore.release()
            return None

    async def release(self, session: REPLSession):
        """Return a session to the pool (close it)."""
        await session.close()
        self._sessions.pop(session.session_id, None)
        self._warm_env_ids.pop(session.session_id, None)
        self._semaphore.release()

    async def shutdown(self):
        for s in list(self._sessions.values()):
            await s.close()
        self._sessions.clear()
        self._warm_env_ids.clear()

    def get_warm_env_id(self, session: REPLSession) -> int:
        """Get the prewarmed env_id for a session (post-preamble)."""
        return self._warm_env_ids.get(session.session_id, 0)

    def stats(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "max_sessions": self._max_sessions,
            "warm_available": self._warm_queue.qsize(),
            "total_created": self._next_id,
            "prewarmed_envs": len(self._warm_env_ids),
        }


# ─── Single-shot compile fallback ────────────────────────────

async def verify_compile(code: str) -> dict:
    """Single-shot compilation with `lake env lean --stdin`."""
    t0 = time.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            "lake", "env", "lean", "--stdin",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=PROJECT_DIR,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=code.encode()),
            timeout=TIMEOUT)

        elapsed_ms = int((time.time() - t0) * 1000)
        stderr_text = stderr.decode(errors="replace")

        if proc.returncode == 0:
            return {"env": 1, "messages": [], "goals": [],
                    "elapsed_ms": elapsed_ms}
        else:
            return {
                "env": 0,
                "messages": [{"severity": "error", "data": stderr_text}],
                "goals": [],
                "elapsed_ms": elapsed_ms,
            }
    except asyncio.TimeoutError:
        return {"env": 0,
                "messages": [{"severity": "error",
                              "data": f"Timeout after {TIMEOUT}s"}],
                "goals": []}
    except Exception as e:
        return {"env": 0,
                "messages": [{"severity": "error", "data": str(e)}],
                "goals": []}


# ─── Client handlers ──────────────────────────────────────────

_compile_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def handle_client_repl(reader, writer, pool: REPLPool):
    """Handle a client in REPL mode: one REPL process per connection."""
    session = await pool.acquire()
    if not session:
        error = json.dumps({"id": "error", "error": "No REPL available"})
        writer.write((error + "\n").encode())
        await writer.drain()
        writer.close()
        return

    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            try:
                request = json.loads(line.decode())
            except json.JSONDecodeError as e:
                resp = {"id": "unknown", "error": f"Invalid JSON: {e}"}
                writer.write((json.dumps(resp) + "\n").encode())
                await writer.drain()
                continue

            request_id = request.get("id", "unknown")
            t0 = time.time()

            result = await session.send(request)
            elapsed_ms = int((time.time() - t0) * 1000)

            response = {
                "id": request_id,
                "result": result,
                "elapsed_ms": elapsed_ms,
            }
            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()

            logger.info(f"req={request_id} elapsed={elapsed_ms}ms")

    except (ConnectionResetError, BrokenPipeError) as _exc:
        logger.debug(f"Suppressed exception: {_exc}")
    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        await pool.release(session)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception as _exc:
            logger.debug(f"Suppressed exception: {_exc}")


async def handle_client_compile(reader, writer):
    """Handle a client in compile mode: single-shot per request."""
    try:
        while True:
            line = await reader.readline()
            if not line:
                break

            try:
                request = json.loads(line.decode())
            except json.JSONDecodeError as e:
                resp = {"id": "unknown", "error": f"Invalid JSON: {e}"}
                writer.write((json.dumps(resp) + "\n").encode())
                await writer.drain()
                continue

            code = request.get("cmd", "")
            request_id = request.get("id", "unknown")
            t0 = time.time()

            async with _compile_semaphore:
                result = await verify_compile(code)

            elapsed_ms = int((time.time() - t0) * 1000)
            response = {
                "id": request_id,
                "result": result,
                "elapsed_ms": elapsed_ms,
            }
            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()

            logger.info(f"req={request_id} elapsed={elapsed_ms}ms")

    except (ConnectionResetError, BrokenPipeError) as _exc:
        logger.debug(f"Suppressed exception: {_exc}")
    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception as _exc:
            logger.debug(f"Suppressed exception: {_exc}")


# ─── Main ─────────────────────────────────────────────────────

async def main():
    repl_binary = find_repl_binary()
    use_repl = bool(repl_binary)

    pool = None
    if use_repl:
        pool = REPLPool(repl_binary, PROJECT_DIR, MAX_CONCURRENT)
        logger.info(f"Mode: REPL (binary={repl_binary})")

        # Prewarm sessions with preamble (import Mathlib)
        prewarm_count = int(os.environ.get("LEAN_PREWARM", "2"))
        if prewarm_count > 0:
            await pool.prewarm(prewarm_count)
    else:
        logger.info("Mode: compile (lean4-repl not found, using lake env lean)")

    socket_path = SOCKET_PATH
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    Path(socket_path).parent.mkdir(parents=True, exist_ok=True)

    if use_repl:
        handler = lambda r, w: handle_client_repl(r, w, pool)
    else:
        handler = handle_client_compile

    server = await asyncio.start_unix_server(handler, path=socket_path)
    os.chmod(socket_path, 0o777)

    logger.info(f"Listening on {socket_path}")
    logger.info(f"Max concurrent: {MAX_CONCURRENT}")
    logger.info(f"Project dir: {PROJECT_DIR}")

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(server, pool)))

    async with server:
        await server.serve_forever()


async def shutdown(server, pool):
    logger.info("Shutting down...")
    server.close()
    if pool:
        await pool.shutdown()
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted")
