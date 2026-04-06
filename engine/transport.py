"""engine/transport.py — REPL 通信传输层协议

将 REPL 进程通信抽象为 Transport 接口, 使 AsyncLeanSession
成为唯一的会话实现, 同时支持本地进程和远程连接:

  LocalTransport:     asyncio.create_subprocess_exec (默认, 替代原有实现)
  TCPTransport:       TCP 连接远端 REPL 代理 (远期)
  MockTransport:      测试用 mock (不启动真实 REPL)

所有 Transport 实现都是异步的。同步调用通过 SyncAdapter 包装。
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from engine._core import which as _which

logger = logging.getLogger(__name__)


class REPLTransport(ABC):
    """REPL 通信传输层抽象基类

    所有子类必须实现 async 的 send / start / close。
    """

    @abstractmethod
    async def send(self, cmd: dict) -> Optional[dict]:
        """发送 JSON 命令并返回响应 (带超时)"""
        ...

    @abstractmethod
    async def start(self) -> bool:
        """初始化连接 (启动进程 / 建立 TCP 连接等)"""
        ...

    @abstractmethod
    async def close(self):
        """关闭连接"""
        ...

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """连接是否存活"""
        ...

    @property
    def is_fallback(self) -> bool:
        """是否为 fallback 模式 (无真实 REPL)"""
        return False


class LocalTransport(REPLTransport):
    """本地 REPL 进程传输层

    通过 asyncio.create_subprocess_exec 启动 lean4-repl 进程,
    用 stdin/stdout JSON Lines 协议通信。

    替代原有 AsyncLeanSession 中内联的进程管理代码。
    """

    def __init__(self, project_dir: str = ".",
                 timeout_seconds: int = 30,
                 repl_binary: str = None):
        self._project_dir = project_dir
        self._timeout = timeout_seconds
        self._repl_binary = repl_binary or self._find_repl_binary(project_dir)
        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._alive = False
        self._fallback = False

    async def start(self) -> bool:
        if not self._repl_binary:
            logger.warning(
                f"LocalTransport: [FALLBACK] No REPL binary found in "
                f"{self._project_dir}")
            self._alive = True
            self._fallback = True
            return True

        try:
            self._process = await asyncio.create_subprocess_exec(
                self._repl_binary,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._project_dir,
            )
            self._writer = self._process.stdin
            self._reader = self._process.stdout
            self._alive = True
            return True
        except FileNotFoundError:
            logger.warning(
                f"LocalTransport: [FALLBACK] REPL binary not found: "
                f"{self._repl_binary}")
            self._alive = True
            self._fallback = True
            return True
        except Exception as e:
            logger.error(f"LocalTransport: start failed: {e}")
            return False

    async def send(self, cmd: dict) -> Optional[dict]:
        if not self._process or self._process.returncode is not None:
            return None
        try:
            data = (json.dumps(cmd) + "\n").encode()
            self._writer.write(data)
            await self._writer.drain()

            line = await asyncio.wait_for(
                self._reader.readline(),
                timeout=self._timeout)

            if line:
                return json.loads(line.decode())
            return None
        except asyncio.TimeoutError:
            logger.warning(
                f"LocalTransport: REPL read timed out after {self._timeout}s")
            return None
        except (json.JSONDecodeError, IOError, BrokenPipeError) as e:
            logger.error(f"LocalTransport: send failed: {e}")
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
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_fallback(self) -> bool:
        return self._fallback

    @staticmethod
    def _find_repl_binary(project_dir: str) -> Optional[str]:
        candidates = [
            os.path.join(project_dir, ".lake", "build", "bin", "repl"),
            "lean",
        ]
        for c in candidates:
            if os.path.isfile(c) or _which(c):
                return c
        return None


class FallbackTransport(REPLTransport):
    """Fallback 传输层: 始终返回失败, 用于无 Lean4 环境时"""

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


class MockTransport(REPLTransport):
    """测试用 mock 传输层: 返回预设的响应序列"""

    def __init__(self, responses: list[dict] = None):
        self._responses = list(responses or [])
        self._call_count = 0
        self._alive = False
        self._sent_commands: list[dict] = []

    async def start(self) -> bool:
        self._alive = True
        return True

    async def send(self, cmd: dict) -> Optional[dict]:
        self._sent_commands.append(cmd)
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        # 默认: 返回成功, 无 goal
        self._call_count += 1
        return {"env": self._call_count, "messages": [], "goals": []}

    async def close(self):
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_fallback(self) -> bool:
        return False


# ═══════════════════════════════════════════════════════════════
# 同步适配器: 让异步 Transport 在同步代码中可用
# ═══════════════════════════════════════════════════════════════

class SyncTransportAdapter:
    """将异步 Transport 包装为同步接口

    在内部维护一个事件循环, 将 async 调用转为同步。
    用于向后兼容同步代码 (如 Orchestrator.prove)。

    Usage::

        transport = LocalTransport(project_dir="/path")
        sync = SyncTransportAdapter(transport)
        sync.start()
        resp = sync.send({"cmd": "import Mathlib", "env": 0})
        sync.close()
    """

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
