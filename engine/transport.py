"""engine/transport.py — REPL transport layer with health checks and auto-restart

Transport abstractions for communicating with Lean4 REPL processes.
All transports implement the async REPLTransport interface.

Transport Implementations:
  LocalTransport:     Local subprocess (lean4-repl or lake env lean)
  SocketTransport:    Unix domain socket to lean_daemon.py
  MockTransport:      Testing mock with realistic REPL state simulation
  FallbackTransport:  Always-fail placeholder when no Lean4 is available

Key improvements over v1:
  - Health check heartbeat: periodic #check Nat to verify REPL is alive
  - Auto-restart: transparent restart on process death (up to max_restarts)
  - Request timeout with cancellation
  - Structured protocol: uses REPLRequest/REPLResponse from repl_protocol.py
  - Metrics: tracks latency, error rate, restart count
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from engine._core import which as _which

logger = logging.getLogger(__name__)


# ─── Transport Stats ──────────────────────────────────────────

@dataclass
class TransportStats:
    """Accumulated transport statistics."""
    total_requests: int = 0
    total_errors: int = 0
    total_restarts: int = 0
    total_latency_ms: float = 0.0
    last_heartbeat: float = 0.0
    consecutive_failures: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.total_requests)

    @property
    def error_rate(self) -> float:
        return self.total_errors / max(1, self.total_requests)

    def record_success(self, latency_ms: float):
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.consecutive_failures = 0

    def record_failure(self):
        self.total_requests += 1
        self.total_errors += 1
        self.consecutive_failures += 1

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_restarts": self.total_restarts,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "consecutive_failures": self.consecutive_failures,
        }


# ─── Abstract Base ────────────────────────────────────────────

class REPLTransport(ABC):
    """REPL transport layer abstract base class."""

    @abstractmethod
    async def send(self, cmd: dict) -> Optional[dict]:
        """Send a JSON command and return the response."""
        ...

    @abstractmethod
    async def start(self) -> bool:
        """Initialize the connection."""
        ...

    @abstractmethod
    async def close(self):
        """Close the connection."""
        ...

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        ...

    @property
    def is_fallback(self) -> bool:
        return False

    def get_stats(self) -> dict:
        return {}


# ─── LocalTransport ───────────────────────────────────────────

class LocalTransport(REPLTransport):
    """Local Lean4 REPL process transport.

    Starts a lean4-repl subprocess and communicates via stdin/stdout
    using JSON Lines protocol.

    Features:
    - Auto-detection of REPL binary (.lake/build/bin/repl)
    - Health check heartbeat
    - Auto-restart on process death (up to max_restarts)
    - Request timeout with process kill on stuck REPL
    """

    REPL_CANDIDATES = [
        ".lake/build/bin/repl",
        "lake-packages/repl/.lake/build/bin/repl",
    ]

    def __init__(self, project_dir: str = ".",
                 timeout_seconds: int = 60,
                 repl_binary: str = None,
                 max_restarts: int = 3,
                 heartbeat_interval: float = 30.0):
        self._project_dir = os.path.abspath(project_dir)
        self._timeout = timeout_seconds
        self._repl_binary = repl_binary
        self._max_restarts = max_restarts
        self._heartbeat_interval = heartbeat_interval
        self._process: Optional[asyncio.subprocess.Process] = None
        self._alive = False
        self._fallback = False
        self._single_shot = False
        self._stats = TransportStats()
        self._send_lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        binary = self._repl_binary or self._find_repl_binary()
        if binary:
            self._repl_binary = binary
            ok = await self._start_repl_process()
            if ok:
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop())
                return True

        lean_bin = _which("lean")
        if lean_bin:
            self._repl_binary = lean_bin
            self._single_shot = True
            self._alive = True
            logger.info(f"LocalTransport: single-shot mode with {lean_bin}")
            return True

        logger.warning("LocalTransport: [FALLBACK] No Lean4 binary found")
        self._alive = True
        self._fallback = True
        return True

    async def _start_repl_process(self) -> bool:
        try:
            self._process = await asyncio.create_subprocess_exec(
                self._repl_binary,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._project_dir,
            )
            self._alive = True
            self._fallback = False
            self._single_shot = False
            logger.info(
                f"LocalTransport: REPL started (pid={self._process.pid})")
            return True
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"LocalTransport: {e}")
            return False
        except Exception as e:
            logger.error(f"LocalTransport: start failed: {e}")
            return False

    async def _restart_repl(self) -> bool:
        if self._stats.total_restarts >= self._max_restarts:
            logger.error("LocalTransport: max restarts reached")
            self._alive = False
            return False

        logger.warning(
            f"LocalTransport: restarting REPL "
            f"(attempt {self._stats.total_restarts + 1})")
        await self._kill_process()
        self._stats.total_restarts += 1
        return await self._start_repl_process()

    async def _kill_process(self):
        if self._process and self._process.returncode is None:
            try:
                self._process.kill()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError, OSError) as _exc:
                logger.debug(f"Suppressed exception: {_exc}")

    async def send(self, cmd: dict) -> Optional[dict]:
        if self._fallback:
            return None
        if self._single_shot:
            return await self._send_single_shot(cmd)
        return await self._send_repl(cmd)

    async def _send_repl(self, cmd: dict) -> Optional[dict]:
        async with self._send_lock:
            t0 = time.monotonic()

            if not self._process or self._process.returncode is not None:
                ok = await self._restart_repl()
                if not ok:
                    self._stats.record_failure()
                    return None

            try:
                data = (json.dumps(cmd, ensure_ascii=False) + "\n").encode()
                self._process.stdin.write(data)
                await self._process.stdin.drain()

                line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=self._timeout)

                elapsed_ms = (time.monotonic() - t0) * 1000

                if not line:
                    logger.warning("LocalTransport: empty response (crash?)")
                    self._stats.record_failure()
                    await self._restart_repl()
                    return None

                result = json.loads(line.decode())
                self._stats.record_success(elapsed_ms)
                return result

            except asyncio.TimeoutError:
                logger.warning(f"LocalTransport: timeout after {self._timeout}s")
                self._stats.record_failure()
                if self._stats.consecutive_failures >= 2:
                    await self._restart_repl()
                return None

            except (json.JSONDecodeError, IOError, BrokenPipeError,
                    ConnectionResetError) as e:
                logger.error(f"LocalTransport: send failed: {e}")
                self._stats.record_failure()
                await self._restart_repl()
                return None

    async def _send_single_shot(self, cmd: dict) -> Optional[dict]:
        """Single-shot compilation with `lean --stdin`."""
        code = cmd.get("cmd", "")
        if not code:
            return None

        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                self._repl_binary, "--stdin",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._project_dir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=code.encode()),
                timeout=self._timeout)

            elapsed_ms = (time.monotonic() - t0) * 1000
            stderr_text = stderr.decode(errors="replace")

            if proc.returncode == 0:
                self._stats.record_success(elapsed_ms)
                return {
                    "env": cmd.get("env", 0) + 1,
                    "messages": [],
                    "goals": [],
                }
            else:
                self._stats.record_failure()
                return {
                    "env": cmd.get("env", 0),
                    "messages": [{"severity": "error", "data": stderr_text}],
                    "goals": [],
                }
        except asyncio.TimeoutError:
            self._stats.record_failure()
            return None
        except Exception as e:
            self._stats.record_failure()
            logger.error(f"LocalTransport single-shot: {e}")
            return None

    async def _heartbeat_loop(self):
        while self._alive and not self._fallback and not self._single_shot:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                if not self._alive:
                    break
                if not self._process or self._process.returncode is not None:
                    continue

                resp = await self._send_repl({"cmd": "#check Nat", "env": 0})
                if resp is not None:
                    self._stats.last_heartbeat = time.time()
                else:
                    logger.warning("LocalTransport: heartbeat failed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"LocalTransport: heartbeat error: {e}")

    async def close(self):
        self._alive = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError as _exc:
                logger.debug(f"Suppressed exception: {_exc}")
        await self._kill_process()

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_fallback(self) -> bool:
        return self._fallback

    @property
    def is_single_shot(self) -> bool:
        return self._single_shot

    def get_stats(self) -> dict:
        d = self._stats.to_dict()
        d["mode"] = ("fallback" if self._fallback
                      else "single_shot" if self._single_shot
                      else "repl")
        d["pid"] = (self._process.pid
                    if self._process and self._process.returncode is None
                    else None)
        return d

    def _find_repl_binary(self) -> Optional[str]:
        for candidate in self.REPL_CANDIDATES:
            path = os.path.join(self._project_dir, candidate)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        for name in ["repl", "lean4-repl"]:
            found = _which(name)
            if found:
                return found
        return None


# ─── SocketTransport ──────────────────────────────────────────

class SocketTransport(REPLTransport):
    """Unix domain socket transport — connects to lean_daemon.py.

    Includes auto-reconnect on connection loss.
    """

    def __init__(self, socket_path: str, timeout_seconds: int = 120,
                 max_reconnects: int = 5):
        self._socket_path = socket_path
        self._timeout = timeout_seconds
        self._max_reconnects = max_reconnects
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._alive = False
        self._stats = TransportStats()
        self._send_lock = asyncio.Lock()
        self._request_counter = 0

    async def start(self) -> bool:
        return await self._connect()

    async def _connect(self) -> bool:
        try:
            self._reader, self._writer = await asyncio.open_unix_connection(
                self._socket_path)
            self._alive = True
            logger.info(f"SocketTransport: connected to {self._socket_path}")
            return True
        except (FileNotFoundError, ConnectionRefusedError) as e:
            logger.error(f"SocketTransport: {e}")
            return False
        except Exception as e:
            logger.error(f"SocketTransport: start failed: {e}")
            return False

    async def _reconnect(self) -> bool:
        if self._stats.total_restarts >= self._max_reconnects:
            logger.error("SocketTransport: max reconnects reached")
            self._alive = False
            return False
        self._stats.total_restarts += 1
        if self._writer:
            try:
                self._writer.close()
            except Exception as _exc:
                logger.debug(f"Suppressed exception: {_exc}")
        await asyncio.sleep(0.5)
        return await self._connect()

    async def send(self, cmd: dict) -> Optional[dict]:
        async with self._send_lock:
            if not self._writer or self._writer.is_closing():
                if not await self._reconnect():
                    return None

            self._request_counter += 1
            request_id = f"req-{self._request_counter}"
            t0 = time.monotonic()

            try:
                request = dict(cmd, id=request_id)
                data = (json.dumps(request, ensure_ascii=False) + "\n").encode()
                self._writer.write(data)
                await self._writer.drain()

                line = await asyncio.wait_for(
                    self._reader.readline(), timeout=self._timeout)

                elapsed_ms = (time.monotonic() - t0) * 1000

                if not line:
                    self._stats.record_failure()
                    self._alive = False
                    return None

                result = json.loads(line.decode())
                self._stats.record_success(elapsed_ms)

                # lean_daemon may wrap result
                if "result" in result:
                    return result["result"]
                return result

            except asyncio.TimeoutError:
                self._stats.record_failure()
                return None
            except (json.JSONDecodeError, IOError, BrokenPipeError,
                    ConnectionResetError) as e:
                self._stats.record_failure()
                self._alive = False
                await self._reconnect()
                return None

    async def close(self):
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as _exc:
                logger.debug(f"Suppressed exception: {_exc}")
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive

    def get_stats(self) -> dict:
        d = self._stats.to_dict()
        d["socket_path"] = self._socket_path
        return d


# ─── FallbackTransport ────────────────────────────────────────

class FallbackTransport(REPLTransport):
    """Always returns None. Used when no Lean4 is available."""

    def __init__(self):
        self._alive = True

    async def start(self) -> bool:
        return True

    async def send(self, cmd: dict) -> Optional[dict]:
        return None

    async def close(self):
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_fallback(self) -> bool:
        return True


# ─── MockTransport ────────────────────────────────────────────

class MockTransport(REPLTransport):
    """Realistic mock transport that simulates Lean4 REPL state machine.

    Features:
    - Tracks env_id (auto-increments on success)
    - Configurable error map (pattern → error message)
    - Scripted response queue
    - Records all commands for test assertions

    Usage (basic):
        mock = MockTransport()
        await mock.start()
        resp = await mock.send({"cmd": "import Mathlib", "env": 0})
        assert resp["env"] == 1

    Usage (with errors):
        mock = MockTransport(error_on={"simp": "tactic 'simp' failed"})

    Usage (scripted):
        mock = MockTransport(responses=[...])
    """

    def __init__(self, responses: list[dict] = None,
                 error_on: dict[str, str] = None,
                 latency_ms: float = 0,
                 initial_env: int = 0):
        self._responses = list(responses or [])
        self._error_on = error_on or {}
        self._latency_ms = latency_ms
        self._next_env = initial_env + 1
        self._next_ps = 1
        self._call_count = 0
        self._alive = False
        self._sent_commands: list[dict] = []

    async def start(self) -> bool:
        self._alive = True
        return True

    async def send(self, cmd: dict) -> Optional[dict]:
        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000.0)

        self._sent_commands.append(cmd)
        idx = self._call_count
        self._call_count += 1

        # Scripted responses first
        if idx < len(self._responses):
            resp = self._responses[idx]
            if "env" in resp:
                self._next_env = max(self._next_env, resp["env"] + 1)
            return resp

        # Error map
        cmd_text = cmd.get("cmd", "") or cmd.get("tactic", "")
        for pattern, error_msg in self._error_on.items():
            if pattern in cmd_text:
                return {
                    "env": cmd.get("env", 0),
                    "messages": [{"severity": "error", "data": error_msg}],
                    "goals": [],
                }

        # Default: success
        env = self._next_env
        self._next_env += 1

        # Sorry detection → interactive goals
        if "sorry" in cmd_text and any(
                kw in cmd_text for kw in ("theorem", "lemma", "def")):
            ps = self._next_ps
            self._next_ps += 1
            return {
                "env": env,
                "messages": [],
                "goals": [],
                "sorries": [{
                    "proofState": ps,
                    "pos": {"line": 1, "column": 0},
                    "goal": "⊢ <mock goal>",
                    "endPos": {"line": 1, "column": 5},
                }],
            }

        # Tactic mode
        if "tactic" in cmd:
            ps = self._next_ps
            self._next_ps += 1
            return {"proofState": ps, "goals": [], "messages": []}

        return {"env": env, "messages": [], "goals": []}

    async def close(self):
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def sent_commands(self) -> list[dict]:
        return list(self._sent_commands)

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset(self):
        self._sent_commands.clear()
        self._call_count = 0


# ─── Sync Adapter ─────────────────────────────────────────────

class SyncTransportAdapter:
    """Wraps an async Transport for synchronous code."""

    def __init__(self, transport: REPLTransport):
        self._transport = transport
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def start(self) -> bool:
        return self._get_loop().run_until_complete(self._transport.start())

    def send(self, cmd: dict) -> Optional[dict]:
        return self._get_loop().run_until_complete(self._transport.send(cmd))

    def close(self):
        loop = self._get_loop()
        loop.run_until_complete(self._transport.close())
        loop.close()
        self._loop = None

    @property
    def is_alive(self) -> bool:
        return self._transport.is_alive

    @property
    def is_fallback(self) -> bool:
        return self._transport.is_fallback
