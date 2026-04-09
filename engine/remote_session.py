"""engine/remote_session.py — 远程 REPL 会话代理

将 AsyncLeanSession 的通信协议抽象为 Transport 层,
使 REPL 进程可以运行在本机或远端:

  LocalTransport:   直接调用 asyncio.create_subprocess_exec (当前行为)
  TCPTransport:     通过 TCP 连接远端的 REPL 代理服务
  WebSocketTransport: 通过 WebSocket 连接 (适合穿越防火墙)

架构:

  Agent (本机)                     REPL Worker (远端)
  ┌──────────────┐                ┌──────────────┐
  │ AsyncLean    │  TCP/WS/gRPC   │ REPL Proxy   │
  │ Session      │ ◄────────────► │ Server       │
  │ (Transport)  │  JSON protocol │ (lean4-repl) │
  └──────────────┘                └──────────────┘

协议: JSON Lines over TCP
  请求: {"cmd": "simp", "env": 3}
  响应: {"env": 4, "messages": [], "goals": []}

Usage::

    # 本地模式 (默认, 等同于现有 AsyncLeanSession)
    session = RemoteSession(LocalTransport(project_dir="/path"))
    await session.start()

    # 远程模式
    session = RemoteSession(TCPTransport(host="worker-1", port=9100))
    await session.start()

    # 混合池: 2 本地 + 4 远端
    pool = ElasticPool()
    pool.add_local(2, project_dir="/path")
    pool.add_remote(["worker-1:9100", "worker-2:9100"])
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

from engine._core import (
    TacticFeedback, FullVerifyResult,
    classify_error as _classify_error,
    classify_error_structured as _classify_error_structured,
    extract_expected as _extract_expected,
    extract_actual as _extract_actual,
    assemble_code as _assemble_code,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Transport 抽象层
# ═══════════════════════════════════════════════════════════════

class Transport(ABC):
    """REPL 通信传输层抽象"""

    @abstractmethod
    async def connect(self) -> bool:
        """建立连接"""
        ...

    @abstractmethod
    async def send(self, request: dict) -> Optional[dict]:
        """发送 JSON 请求, 接收 JSON 响应"""
        ...

    @abstractmethod
    async def close(self):
        """关闭连接"""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @property
    def transport_type(self) -> str:
        return self.__class__.__name__


class LocalTransport(Transport):
    """本地传输: 直接管理 asyncio subprocess

    等同于现有 AsyncLeanSession 的行为, 封装为 Transport 接口。
    """

    def __init__(self, project_dir: str = ".",
                 timeout_seconds: int = 30):
        self.project_dir = project_dir
        self._timeout = timeout_seconds
        self._process: Optional[asyncio.subprocess.Process] = None
        self._connected = False

    async def connect(self) -> bool:
        repl_binary = self._find_repl()
        if not repl_binary:
            logger.warning("LocalTransport: no REPL binary found")
            self._connected = True  # fallback mode
            return True
        try:
            self._process = await asyncio.create_subprocess_exec(
                repl_binary,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_dir)
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"LocalTransport: start failed: {e}")
            self._connected = True  # fallback
            return True

    async def send(self, request: dict) -> Optional[dict]:
        if not self._process or self._process.returncode is not None:
            return None
        try:
            data = (json.dumps(request) + "\n").encode()
            self._process.stdin.write(data)
            await self._process.stdin.drain()
            line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self._timeout)
            if line:
                return json.loads(line.decode())
            return None
        except (asyncio.TimeoutError, json.JSONDecodeError,
                IOError, BrokenPipeError) as e:
            logger.warning(f"LocalTransport send error: {e}")
            return None

    async def close(self):
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(
                    self._process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._process.kill()
                except Exception:
                    pass
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _find_repl(self) -> Optional[str]:
        import shutil
        candidates = [
            os.path.join(self.project_dir, ".lake", "build", "bin", "repl"),
            "lean",
        ]
        for c in candidates:
            if os.path.isfile(c) or shutil.which(c):
                return c
        return None


class TCPTransport(Transport):
    """TCP 传输: 连接远端 REPL 代理服务

    协议: JSON Lines over TCP (每行一个 JSON 对象)
    请求和响应都以 '\\n' 结尾。
    """

    def __init__(self, host: str = "localhost", port: int = 9100,
                 timeout_seconds: int = 30):
        self.host = host
        self.port = port
        self._timeout = timeout_seconds
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False

    async def connect(self) -> bool:
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self._timeout)
            self._connected = True
            logger.info(
                f"TCPTransport: connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(
                f"TCPTransport: connect failed to "
                f"{self.host}:{self.port}: {e}")
            return False

    async def send(self, request: dict) -> Optional[dict]:
        if not self._writer or self._writer.is_closing():
            return None
        try:
            data = (json.dumps(request) + "\n").encode()
            self._writer.write(data)
            await self._writer.drain()
            line = await asyncio.wait_for(
                self._reader.readline(),
                timeout=self._timeout)
            if line:
                return json.loads(line.decode())
            return None
        except (asyncio.TimeoutError, json.JSONDecodeError,
                ConnectionError, OSError) as e:
            logger.warning(f"TCPTransport send error: {e}")
            self._connected = False
            return None

    async def close(self):
        if self._writer and not self._writer.is_closing():
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


# ═══════════════════════════════════════════════════════════════
# RemoteSession — 基于 Transport 的会话
# ═══════════════════════════════════════════════════════════════

class RemoteSession:
    """基于 Transport 抽象的 REPL 会话

    接口与 AsyncLeanSession 兼容, 可无缝替换。
    """

    def __init__(self, transport: Transport, session_id: int = 0):
        self.transport = transport
        self.session_id = session_id
        self._base_env_id = 0
        self._alive = False
        self._busy = False
        self._fallback = False
        self._total_requests = 0

    async def start(self, preamble: str = "import Mathlib") -> bool:
        ok = await self.transport.connect()
        if not ok:
            self._fallback = True
            self._alive = True
            return True

        if preamble:
            resp = await self.transport.send({"cmd": preamble, "env": 0})
            if resp and "env" in resp:
                self._base_env_id = resp["env"]
                self._alive = True
                return True

        self._alive = True
        if not self.transport._process if hasattr(self.transport, '_process') else True:
            self._fallback = True
        return True

    async def try_tactic(self, env_id: int, tactic: str) -> TacticFeedback:
        t0 = time.time()
        self._total_requests += 1

        if self._fallback:
            elapsed = int((time.time() - t0) * 1000)
            return TacticFeedback(
                success=False, tactic=tactic,
                error_message="No active REPL (fallback mode)",
                error_category="no_backend",
                elapsed_ms=elapsed, session_id=self.session_id)

        resp = await self.transport.send({"cmd": tactic, "env": env_id})
        elapsed = int((time.time() - t0) * 1000)

        if not resp:
            return TacticFeedback(
                success=False, tactic=tactic,
                error_message="Transport communication failed",
                error_category="internal",
                elapsed_ms=elapsed, session_id=self.session_id)

        messages = resp.get("messages", [])
        errors = [m for m in messages if m.get("severity") == "error"]
        new_env = resp.get("env", env_id)
        goals = resp.get("goals", [])

        if errors:
            category, combined, _ = _classify_error_structured(messages)
            return TacticFeedback(
                success=False, tactic=tactic,
                error_message=combined[:500],
                error_category=category,
                elapsed_ms=elapsed, session_id=self.session_id)

        return TacticFeedback(
            success=True, tactic=tactic,
            new_env_id=new_env,
            remaining_goals=goals,
            is_proof_complete=(len(goals) == 0),
            elapsed_ms=elapsed, session_id=self.session_id)

    async def verify_complete(self, theorem: str, proof: str,
                              preamble: str = "") -> FullVerifyResult:
        t0 = time.time()
        self._total_requests += 1
        code = _assemble_code(theorem, proof, preamble)

        if self._fallback:
            return FullVerifyResult(
                success=False, stderr="Fallback mode",
                elapsed_ms=int((time.time() - t0) * 1000))

        resp = await self.transport.send({"cmd": code, "env": 0})
        elapsed = int((time.time() - t0) * 1000)

        if not resp:
            return FullVerifyResult(
                success=False, stderr="Transport failed",
                elapsed_ms=elapsed)

        messages = resp.get("messages", [])
        errors = [m for m in messages if m.get("severity") == "error"]
        goals = resp.get("goals", [])
        has_sorry = any("sorry" in m.get("data", "") for m in messages)

        if not errors and not goals and not has_sorry:
            return FullVerifyResult(
                success=True, env_id=resp.get("env", -1),
                elapsed_ms=elapsed)

        error_msgs = [e.get("data", "") for e in errors]
        return FullVerifyResult(
            success=False,
            errors=[{"message": m, "category": _classify_error(m)}
                    for m in error_msgs],
            goals_remaining=goals,
            has_sorry=has_sorry,
            elapsed_ms=elapsed)

    async def close(self):
        await self.transport.close()
        self._alive = False

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_fallback(self) -> bool:
        return self._fallback

    @property
    def base_env_id(self) -> int:
        return self._base_env_id


# ═══════════════════════════════════════════════════════════════
# ElasticPool — 混合本地/远程会话的弹性池
# ═══════════════════════════════════════════════════════════════

class ElasticPool:
    """弹性混合连接池

    支持同时管理本地和远程 REPL 会话:
      - 本地会话: 低延迟, 受本机资源限制
      - 远程会话: 高延迟, 但可扩展到集群规模

    会话选择策略: 优先使用本地会话, 本地都忙时使用远程会话。
    """

    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds
        self._sessions: list[RemoteSession] = []
        self._session_available = asyncio.Condition()
        self._started = False
        self.preamble = "import Mathlib"
        self.project_dir = "."

    async def add_local(self, count: int = 1,
                        project_dir: str = "."):
        """添加本地会话"""
        self.project_dir = project_dir
        for i in range(count):
            transport = LocalTransport(
                project_dir=project_dir,
                timeout_seconds=self.timeout)
            session = RemoteSession(
                transport, session_id=len(self._sessions))
            ok = await session.start(self.preamble)
            if ok:
                self._sessions.append(session)
        self._started = True

    async def add_remote(self, addresses: list[str]):
        """添加远程会话

        Args:
            addresses: ["host1:port1", "host2:port2", ...]
        """
        for addr in addresses:
            parts = addr.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 9100
            transport = TCPTransport(
                host=host, port=port,
                timeout_seconds=self.timeout)
            session = RemoteSession(
                transport, session_id=len(self._sessions))
            ok = await session.start(self.preamble)
            if ok:
                self._sessions.append(session)
                logger.info(f"ElasticPool: added remote session {addr}")
            else:
                logger.warning(
                    f"ElasticPool: failed to connect to {addr}")
        self._started = True

    async def try_tactic(self, env_id: int,
                         tactic: str) -> TacticFeedback:
        session = await self._acquire()
        try:
            return await session.try_tactic(env_id, tactic)
        finally:
            await self._release(session)

    async def try_tactics_parallel(self, env_id: int,
                                   tactics: list[str]) -> list[TacticFeedback]:
        if not tactics:
            return []

        async def _run(tactic):
            session = await self._acquire()
            try:
                return await session.try_tactic(env_id, tactic)
            finally:
                await self._release(session)

        results = await asyncio.gather(
            *(_run(t) for t in tactics),
            return_exceptions=True)

        return [
            r if isinstance(r, TacticFeedback)
            else TacticFeedback(
                success=False, tactic=tactics[i],
                error_message=str(r), error_category="internal")
            for i, r in enumerate(results)
        ]

    async def verify_complete(self, theorem: str, proof: str,
                              preamble: str = "") -> FullVerifyResult:
        session = await self._acquire()
        try:
            return await session.verify_complete(theorem, proof, preamble)
        finally:
            await self._release(session)

    async def shutdown(self):
        await asyncio.gather(
            *(s.close() for s in self._sessions),
            return_exceptions=True)
        self._sessions.clear()
        self._started = False

    def stats(self) -> dict:
        local = sum(1 for s in self._sessions
                    if isinstance(s.transport, LocalTransport))
        remote = sum(1 for s in self._sessions
                     if isinstance(s.transport, TCPTransport))
        return {
            "total_sessions": len(self._sessions),
            "local_sessions": local,
            "remote_sessions": remote,
            "busy_sessions": sum(1 for s in self._sessions if s.is_busy),
            "fallback_sessions": sum(1 for s in self._sessions if s.is_fallback),
            "all_fallback": all(s.is_fallback for s in self._sessions) and bool(self._sessions),
            "active_sessions": sum(1 for s in self._sessions if s.is_alive),
        }

    @property
    def base_env_id(self) -> int:
        """Return base env_id from first available session.

        Required by PoolProtocol. All sessions share the same preamble
        so they share the same base env_id semantics.
        """
        for s in self._sessions:
            if s.is_alive:
                return s.base_env_id
        return 0

    async def share_lemma(self, lemma_code: str, *,
                          inject_all: bool = True) -> int:
        """Inject a proven lemma into REPL sessions.

        Broadcasts the lemma to all (or one) sessions so subsequent
        tactic executions can reference it.

        Args:
            lemma_code: Full Lean4 lemma/theorem/def code.
            inject_all: If True, inject into all sessions. If False,
                        inject into one available session.

        Returns:
            Number of sessions successfully injected.
        """
        targets = self._sessions if inject_all else self._sessions[:1]
        injected = 0

        async def _inject(session: RemoteSession) -> bool:
            if not session.is_alive or session.is_fallback:
                return False
            env_id = session.base_env_id
            resp = await session.transport.send(
                {"cmd": lemma_code, "env": env_id})
            if resp and "env" in resp:
                messages = resp.get("messages", [])
                errors = [m for m in messages
                          if m.get("severity") == "error"]
                if not errors:
                    return True
            return False

        results = await asyncio.gather(
            *(_inject(s) for s in targets),
            return_exceptions=True)
        injected = sum(1 for r in results if r is True)

        if injected > 0:
            logger.info(
                f"ElasticPool: shared lemma to {injected}/{len(targets)} "
                f"sessions")
        return injected

    async def add_session(self, remote_addr: str = "") -> bool:
        """Add a session dynamically (for PoolScaler compatibility).

        Args:
            remote_addr: If empty, adds a local session. If "host:port",
                         adds a remote session to the specified worker.
        """
        if remote_addr:
            parts = remote_addr.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 9100
            transport = TCPTransport(
                host=host, port=port,
                timeout_seconds=self.timeout)
        else:
            transport = LocalTransport(
                project_dir=self.project_dir,
                timeout_seconds=self.timeout)

        session = RemoteSession(
            transport, session_id=len(self._sessions))
        session._is_overflow = True
        ok = await session.start(self.preamble)
        if ok:
            async with self._session_available:
                self._sessions.append(session)
                self._session_available.notify()
            kind = f"remote({remote_addr})" if remote_addr else "local"
            logger.info(
                f"ElasticPool: added overflow {kind} session "
                f"(total={len(self._sessions)})")
        return ok

    async def remove_idle_session(self) -> bool:
        """Remove one idle overflow session (for PoolScaler compatibility).

        Only removes dynamically-added sessions, never the initial pool.
        Prefers removing remote overflow sessions first (higher latency).
        """
        async with self._session_available:
            # First pass: prefer remote overflow
            for i, s in enumerate(self._sessions):
                if (not s.is_busy
                        and getattr(s, '_is_overflow', False)
                        and isinstance(s.transport, TCPTransport)):
                    self._sessions.pop(i)
                    await s.close()
                    logger.info(
                        f"ElasticPool: removed idle remote overflow "
                        f"(total={len(self._sessions)})")
                    return True
            # Second pass: any overflow
            for i, s in enumerate(self._sessions):
                if (not s.is_busy and getattr(s, '_is_overflow', False)):
                    self._sessions.pop(i)
                    await s.close()
                    logger.info(
                        f"ElasticPool: removed idle overflow session "
                        f"(total={len(self._sessions)})")
                    return True
        return False

    async def _acquire(self) -> RemoteSession:
        async with self._session_available:
            deadline = time.time() + self.timeout
            while True:
                # 优先本地非忙会话
                for s in self._sessions:
                    if (s.is_alive and not s.is_busy
                            and isinstance(s.transport, LocalTransport)):
                        s._busy = True
                        return s
                # 其次远程非忙会话
                for s in self._sessions:
                    if s.is_alive and not s.is_busy:
                        s._busy = True
                        return s

                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(
                        self._session_available.wait(),
                        timeout=min(remaining, 1.0))
                except asyncio.TimeoutError:
                    continue

            # 超时: 返回第一个可用会话 (即使忙)
            if self._sessions:
                s = self._sessions[0]
                s._busy = True
                return s
            raise RuntimeError("ElasticPool: no sessions available")

    async def _release(self, session: RemoteSession):
        async with self._session_available:
            session._busy = False
            self._session_available.notify()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.shutdown()


# ═══════════════════════════════════════════════════════════════
# REPL Proxy Server (远端侧)
# ═══════════════════════════════════════════════════════════════

async def start_repl_proxy_server(host: str = "0.0.0.0",
                                  port: int = 9100,
                                  project_dir: str = "."):
    """启动 REPL 代理服务 (运行在 REPL worker 节点上)

    将 TCP 连接桥接到本地 lean4-repl 进程。

    Usage (worker 节点):
        python -c "
        import asyncio
        from engine.remote_session import start_repl_proxy_server
        asyncio.run(start_repl_proxy_server(port=9100, project_dir='/path'))
        "
    """
    local = LocalTransport(project_dir=project_dir)
    await local.connect()

    async def handle_client(reader: asyncio.StreamReader,
                            writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        logger.info(f"REPL proxy: client connected from {addr}")
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    request = json.loads(line.decode())
                    response = await local.send(request)
                    resp_data = json.dumps(response or {}) + "\n"
                    writer.write(resp_data.encode())
                    await writer.drain()
                except json.JSONDecodeError:
                    error = json.dumps({"error": "Invalid JSON"}) + "\n"
                    writer.write(error.encode())
                    await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()
            logger.info(f"REPL proxy: client {addr} disconnected")

    server = await asyncio.start_server(
        handle_client, host, port)
    logger.info(f"REPL proxy server listening on {host}:{port}")

    async with server:
        await server.serve_forever()
