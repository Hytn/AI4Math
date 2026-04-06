"""engine/async_lean_pool.py — 异步 Lean4 REPL 连接池

与同步 LeanPool 共享数据类型 (TacticFeedback, FullVerifyResult),
但将所有阻塞操作替换为 asyncio:

  subprocess.Popen     → asyncio.create_subprocess_exec
  proc.stdin.write()   → writer.write() + await writer.drain()
  selectors + readline → await asyncio.wait_for(reader.readline(), timeout)
  threading.Thread     → asyncio.gather()
  threading.Lock       → asyncio.Lock()

性能提升:
  - 同步版: try_tactics_parallel(4 tactics) = 4 个 Thread, GIL 序列化
  - 异步版: try_tactics_parallel(4 tactics) = 1 个事件循环, 4 路非阻塞 I/O
  - LLM API 调用期间不阻塞验证, 验证期间不阻塞 LLM
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from engine._core import (
    TacticFeedback, FullVerifyResult,
    CompileCache as _CompileCache,
    classify_error as _classify_error,
    classify_error_structured as _classify_error_structured,
    extract_expected as _extract_expected,
    extract_actual as _extract_actual,
    assemble_code as _assemble_code,
    which as _which,
    make_cache_key,
)

logger = logging.getLogger(__name__)


class AsyncLeanSession:
    """单个 Lean4 REPL 的异步会话

    通过 Transport 协议与 REPL 进程通信, 支持:
      - LocalTransport:  本地 asyncio subprocess (默认)
      - MockTransport:   测试用 mock
      - TCPTransport:    远程 REPL (远期)

    当未提供 transport 时, 自动创建 LocalTransport (向后兼容)。
    """

    def __init__(self, session_id: int, project_dir: str = ".",
                 timeout_seconds: int = 30,
                 transport: 'REPLTransport' = None):
        self.session_id = session_id
        self.project_dir = project_dir
        self._timeout = timeout_seconds
        self._busy = False
        self._is_overflow = False
        self._base_env_id = 0
        self._total_requests = 0
        self._total_errors = 0

        # Transport: 如果未提供, 延迟到 start() 中创建 LocalTransport
        self._transport = transport
        self._transport_created_internally = transport is None

    async def start(self, preamble: str = "import Mathlib") -> bool:
        """启动 REPL 进程并预加载环境"""
        # 如果没有提供 transport, 创建默认的 LocalTransport
        if self._transport is None:
            from engine.transport import LocalTransport
            self._transport = LocalTransport(
                project_dir=self.project_dir,
                timeout_seconds=self._timeout)
            self._transport_created_internally = True

        ok = await self._transport.start()
        if not ok:
            logger.error(f"AsyncSession {self.session_id}: transport start failed")
            return False

        if self._transport.is_fallback:
            logger.warning(
                f"AsyncSession {self.session_id}: [FALLBACK] No REPL binary")
            return True

        # 预加载 preamble
        if preamble:
            resp = await self._transport.send({"cmd": preamble, "env": 0})
            if resp and "env" in resp:
                self._base_env_id = resp["env"]
                logger.info(
                    f"AsyncSession {self.session_id}: started, "
                    f"env_id={self._base_env_id}")
                return True

        return True

    async def try_tactic(self, env_id: int, tactic: str) -> TacticFeedback:
        """在指定 env_id 上尝试一条 tactic"""
        t0 = time.time()
        self._total_requests += 1

        if self._transport and self._transport.is_alive and not self._transport.is_fallback:
            return await self._try_tactic_repl(env_id, tactic, t0)
        else:
            return self._try_tactic_fallback(env_id, tactic, t0)

    async def verify_complete(self, theorem: str, proof: str,
                              preamble: str = "") -> FullVerifyResult:
        """验证完整的定理+证明"""
        t0 = time.time()
        self._total_requests += 1
        full_code = _assemble_code(theorem, proof, preamble)

        if self._transport and self._transport.is_alive and not self._transport.is_fallback:
            resp = await self._transport.send({"cmd": full_code, "env": 0})
            elapsed = int((time.time() - t0) * 1000)
            return self._parse_verify_response(resp, elapsed)
        else:
            return await self._verify_fallback(full_code, t0)

    async def close(self):
        """关闭会话"""
        if self._transport:
            await self._transport.close()

    # ── 内部方法 ──

    async def _try_tactic_repl(self, env_id: int, tactic: str,
                                t0: float) -> TacticFeedback:
        resp = await self._transport.send({"cmd": tactic, "env": env_id})
        elapsed = int((time.time() - t0) * 1000)

        if not resp:
            return TacticFeedback(
                success=False, tactic=tactic,
                error_message="REPL communication failed",
                error_category="internal",
                elapsed_ms=elapsed, session_id=self.session_id)

        messages = resp.get("messages", [])
        errors = [m for m in messages if m.get("severity") == "error"]
        new_env = resp.get("env", env_id)
        goals = resp.get("goals", [])

        if errors:
            category, combined_msg, meta = _classify_error_structured(messages)
            err = errors[0]
            return TacticFeedback(
                success=False, tactic=tactic,
                error_message=combined_msg[:500] if combined_msg else err.get("data", ""),
                error_category=category,
                expected_type=_extract_expected(err.get("data", "")),
                actual_type=_extract_actual(err.get("data", "")),
                elapsed_ms=elapsed, session_id=self.session_id)

        return TacticFeedback(
            success=True, tactic=tactic,
            new_env_id=new_env,
            remaining_goals=goals,
            is_proof_complete=(len(goals) == 0),
            elapsed_ms=elapsed, session_id=self.session_id)

    def _try_tactic_fallback(self, env_id: int, tactic: str,
                              t0: float) -> TacticFeedback:
        elapsed = int((time.time() - t0) * 1000)
        return TacticFeedback(
            success=False, tactic=tactic,
            error_message="No active REPL session (Lean4 not available)",
            error_category="no_backend",
            elapsed_ms=elapsed, session_id=self.session_id)

    async def _verify_fallback(self, code: str,
                                t0: float) -> FullVerifyResult:
        """Fallback: 用 asyncio subprocess 单次编译"""
        try:
            lean_bin = _which("lean")
            if not lean_bin:
                return FullVerifyResult(
                    success=False, stderr="lean binary not found")

            proc = await asyncio.create_subprocess_exec(
                lean_bin, "--run", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_dir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=code.encode()),
                timeout=self._timeout)

            elapsed = int((time.time() - t0) * 1000)
            stderr_text = stderr.decode(errors="replace")
            success = (proc.returncode == 0
                       and "sorry" not in stderr_text.lower())
            return FullVerifyResult(
                success=success, stderr=stderr_text,
                elapsed_ms=elapsed,
                has_sorry="sorry" in stderr_text.lower())

        except asyncio.TimeoutError:
            elapsed = int((time.time() - t0) * 1000)
            return FullVerifyResult(
                success=False, stderr="Timeout", elapsed_ms=elapsed)
        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            return FullVerifyResult(
                success=False, stderr=str(e), elapsed_ms=elapsed)

    def _parse_verify_response(self, resp: dict,
                                elapsed: int) -> FullVerifyResult:
        if not resp:
            return FullVerifyResult(
                success=False, stderr="REPL communication failed",
                elapsed_ms=elapsed)

        messages = resp.get("messages", [])
        errors = [m for m in messages if m.get("severity") == "error"]
        goals = resp.get("goals", [])
        has_sorry = any("sorry" in m.get("data", "") for m in messages)

        if not errors and not goals and not has_sorry:
            return FullVerifyResult(
                success=True, env_id=resp.get("env", -1),
                elapsed_ms=elapsed)

        error_dicts = [{"message": e.get("data", ""),
                        "category": _classify_error(e.get("data", ""))}
                       for e in errors]
        return FullVerifyResult(
            success=False, errors=error_dicts,
            goals_remaining=goals, has_sorry=has_sorry,
            elapsed_ms=elapsed)

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def is_alive(self) -> bool:
        return self._transport.is_alive if self._transport else False

    @property
    def is_fallback(self) -> bool:
        return self._transport.is_fallback if self._transport else True

    @property
    def base_env_id(self) -> int:
        return self._base_env_id


class AsyncLeanPool:
    """异步 Lean4 REPL 连接池

    核心改进 (vs 同步 LeanPool):
      - try_tactics_parallel: asyncio.gather 替代 N 个 Thread
      - _acquire_session: asyncio.Condition 替代 threading.Condition
      - 单线程事件循环, 无 GIL 争用

    Usage::

        async with AsyncLeanPool(pool_size=4) as pool:
            results = await pool.try_tactics_parallel(
                env_id=0, tactics=["simp", "ring", "omega"])
    """

    def __init__(self, pool_size: int = 4, project_dir: str = ".",
                 preamble: str = "import Mathlib",
                 timeout_seconds: int = 30):
        self.pool_size = pool_size
        self.project_dir = project_dir
        self.preamble = preamble
        self.timeout = timeout_seconds
        self._sessions: list[AsyncLeanSession] = []
        self._session_available = asyncio.Condition()
        self._started = False
        self._total_requests = 0
        self._total_latency_ms = 0
        self._compile_cache = _CompileCache(maxsize=1024)
        self._env_cache: dict[str, int] = {}
        self._next_session_id = pool_size
        self._env_version = 0

    async def start(self) -> bool:
        """启动所有会话"""
        if self._started:
            return True

        logger.info(f"AsyncLeanPool: starting {self.pool_size} sessions...")

        # 并行启动所有会话
        sessions = []
        for i in range(self.pool_size):
            s = AsyncLeanSession(
                session_id=i,
                project_dir=self.project_dir,
                timeout_seconds=self.timeout)
            sessions.append(s)

        results = await asyncio.gather(
            *(s.start(self.preamble) for s in sessions),
            return_exceptions=True)

        for s, ok in zip(sessions, results):
            if ok is True:
                self._sessions.append(s)
                if s.base_env_id > 0:
                    key = hashlib.sha256(
                        self.preamble.encode()).hexdigest()[:16]
                    self._env_cache[key] = s.base_env_id
            else:
                logger.warning(f"AsyncSession {s.session_id} failed: {ok}")

        self._started = len(self._sessions) > 0
        fallback = sum(1 for s in self._sessions if s.is_fallback)
        if fallback == len(self._sessions) and self._sessions:
            logger.warning(
                f"AsyncLeanPool: ALL {fallback} sessions in FALLBACK MODE")
        logger.info(
            f"AsyncLeanPool: {len(self._sessions)}/{self.pool_size} ready")
        return self._started

    async def try_tactic(self, env_id: int, tactic: str) -> TacticFeedback:
        """在空闲会话上尝试一条 tactic"""
        session = await self._acquire_session()
        try:
            result = await session.try_tactic(env_id, tactic)
            self._record_latency(result.elapsed_ms)
            return result
        finally:
            await self._release_session(session)

    async def try_tactics_parallel(self, env_id: int,
                                   tactics: list[str]) -> list[TacticFeedback]:
        """并行尝试多条 tactic — asyncio.gather 替代 Thread

        这是相对同步版最大的性能提升点:
        同步版: N 个 Thread, GIL 序列化, 每个 Thread 阻塞等待 REPL
        异步版: 1 个事件循环, N 路非阻塞 I/O, 真正并行等待
        """
        if not tactics:
            return []

        async def _run(tactic: str) -> TacticFeedback:
            session = await self._acquire_session()
            try:
                result = await session.try_tactic(env_id, tactic)
                self._record_latency(result.elapsed_ms)
                return result
            finally:
                await self._release_session(session)

        results = await asyncio.gather(
            *(_run(t) for t in tactics),
            return_exceptions=True)

        # 将异常转为 TacticFeedback
        final = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                final.append(TacticFeedback(
                    success=False, tactic=tactics[i],
                    error_message=str(r), error_category="internal"))
            else:
                final.append(r)
        return final

    async def verify_complete(self, theorem: str, proof: str,
                              preamble: str = "") -> FullVerifyResult:
        """验证完整证明 (带缓存)"""
        cache_key = make_cache_key(
            theorem, proof, preamble,
            env_fingerprint=f"v{self._env_version}")
        cached = self._compile_cache.get(cache_key)
        if cached is not None:
            return cached

        session = await self._acquire_session()
        try:
            result = await session.verify_complete(theorem, proof, preamble)
            if result.success or "timeout" not in result.stderr.lower():
                import copy
                cacheable = copy.copy(result)
                cacheable.env_id = -1
                self._compile_cache.put(cache_key, cacheable)
            return result
        finally:
            await self._release_session(session)


    async def share_lemma(self, lemma_code: str, *,
                          name: str = "", statement: str = "",
                          proof: str = "") -> list[int]:
        """将已证引理注入所有会话 (结构化参数 + 直接 REPL 注入)"""
        # ── 构建合法的 Lean4 代码 ──
        if name and statement and proof:
            lemma_code = f"lemma {name} : {statement} := by {proof}"
        elif name and statement and not proof:
            lemma_code = f"lemma {name} : {statement} := by sorry"

        if not lemma_code or not lemma_code.strip():
            return []

        code = lemma_code.strip()
        has_decl = any(code.startswith(kw) for kw in
                       ("lemma ", "theorem ", "def ", "instance ",
                        "noncomputable ", "private ", "protected ",
                        "section", "namespace", "open ", "set_option",
                        "#check", "@["))
        if not has_decl:
            logger.warning(
                f"share_lemma: code does not start with a Lean4 declaration "
                f"keyword, skipping: {code[:80]}...")
            return []

        new_env_ids = []

        async def _inject(session):
            try:
                resp = await session._transport.send({
                    "cmd": code,
                    "env": session.base_env_id,
                })
                if resp is None:
                    return -1
                messages = resp.get("messages", [])
                errors = [m for m in messages if m.get("severity") == "error"]
                if errors:
                    logger.warning(
                        f"share_lemma: injection failed on session "
                        f"{session.session_id}: "
                        f"{errors[0].get('data', '')[:200]}")
                    return -1
                new_env = resp.get("env", -1)
                if new_env >= 0:
                    session._base_env_id = new_env
                return new_env
            except Exception as e:
                logger.warning(
                    f"share_lemma failed on session {session.session_id}: {e}")
                return -1

        tasks = []
        for session in self._sessions:
            if session.is_alive:
                tasks.append(_inject(session))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, int) and r >= 0:
                new_env_ids.append(r)
        if new_env_ids:
            self._env_version += 1
        return new_env_ids

    async def add_session(self) -> bool:
        """Add a new session to the pool (for PoolScaler).

        Returns True if the session was successfully started and added.
        Thread-safe: protected by _session_available Condition.
        """
        async with self._session_available:
            sid = self._next_session_id
            self._next_session_id += 1
            session = AsyncLeanSession(
                session_id=sid,
                project_dir=self.project_dir,
                timeout_seconds=self.timeout)
            ok = await session.start(self.preamble)
            if ok:
                self._sessions.append(session)
                self._session_available.notify()
                logger.info(f"AsyncLeanPool: added session {sid}")
                return True
            logger.warning(f"AsyncLeanPool: failed to add session {sid}")
            return False

    async def remove_idle_session(self) -> bool:
        """Remove one idle (non-busy) session from the pool (for PoolScaler).

        Removes from the tail end. Returns True if a session was removed.
        Thread-safe: protected by _session_available Condition.
        """
        async with self._session_available:
            for i in range(len(self._sessions) - 1, -1, -1):
                session = self._sessions[i]
                if session.is_alive and not session.is_busy:
                    await session.close()
                    self._sessions.pop(i)
                    logger.info(
                        f"AsyncLeanPool: removed session {session.session_id}")
                    return True
        return False

    async def shutdown(self):
        """关闭所有会话"""
        await asyncio.gather(
            *(s.close() for s in self._sessions),
            return_exceptions=True)
        self._sessions.clear()
        self._started = False
        logger.info("AsyncLeanPool: shutdown complete")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
        return False

    def stats(self) -> dict:
        avg_latency = (self._total_latency_ms / self._total_requests
                       if self._total_requests else 0)
        fallback = sum(1 for s in self._sessions if s.is_fallback)
        return {
            "pool_size": self.pool_size,
            "active_sessions": sum(1 for s in self._sessions if s.is_alive),
            "busy_sessions": sum(1 for s in self._sessions if s.is_busy),
            "fallback_sessions": fallback,
            "all_fallback": fallback == len(self._sessions) and len(self._sessions) > 0,
            "total_requests": self._total_requests,
            "avg_latency_ms": round(avg_latency, 1),
            "compile_cache": self._compile_cache.stats(),
        }

    # ── 会话调度 ──

    async def _acquire_session(self) -> AsyncLeanSession:
        """异步获取空闲会话 (asyncio.Condition)"""
        async with self._session_available:
            deadline = time.time() + self.timeout
            while True:
                for session in self._sessions:
                    if session.is_alive and not session.is_busy:
                        session._busy = True
                        return session

                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(
                        self._session_available.wait(),
                        timeout=min(remaining, 1.0))
                except asyncio.TimeoutError:
                    continue

            # 超时: overflow (标记 _is_overflow, 用完即关)
            logger.warning("AsyncLeanPool: creating overflow session")
            sid = self._next_session_id
            self._next_session_id += 1
            overflow = AsyncLeanSession(
                session_id=sid,
                project_dir=self.project_dir,
                timeout_seconds=self.timeout)
            await overflow.start(self.preamble)
            overflow._busy = True
            overflow._is_overflow = True
            self._sessions.append(overflow)
            return overflow

    async def _release_session(self, session: AsyncLeanSession):
        """释放会话并通知等待者

        Overflow 会话在释放时自动关闭并移除, 防止列表无限增长。
        """
        async with self._session_available:
            session._busy = False

            if session._is_overflow:
                try:
                    await session.close()
                except Exception as e:
                    logger.warning(f"Failed to close overflow session: {e}")
                try:
                    self._sessions.remove(session)
                except ValueError:
                    pass
                logger.debug(
                    f"AsyncLeanPool: removed overflow session "
                    f"{session.session_id}, pool size={len(self._sessions)}")

            self._session_available.notify()

    def _record_latency(self, ms: int):
        self._total_requests += 1
        self._total_latency_ms += ms


# ═══════════════════════════════════════════════════════════════
# 同步接口: 直接复用 LeanPool (消除 sync/async 代码重复)
# ═══════════════════════════════════════════════════════════════

# SyncLeanPool 现在是 LeanPool 的直接别名。
# 之前 SyncLeanPool 通过 asyncio 事件循环包装 AsyncLeanPool,
# 导致 ~100 行重复代码和 Phase 1 修复只应用于一个版本的问题。
# 现在两个名字指向同一个实现, bug 修复只需改一处。
from engine.lean_pool import LeanPool as SyncLeanPool  # noqa: F401
