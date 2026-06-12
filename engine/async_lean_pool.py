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
import logging
import time
from typing import Callable, Optional

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
from engine.observability_stub import metrics

logger = logging.getLogger(__name__)

def _declaration_code(theorem: str, proof: str = "") -> str:
    thm = (theorem or "").strip()
    prf = (proof or "").strip()
    if not thm:
        return prf
    if thm.rstrip().endswith(":=") and prf:
        if prf.startswith(":="):
            return f"{thm.rstrip()[:-2].rstrip()} {prf}"
        if prf.startswith("by"):
            return f"{thm} {prf}"
        return f"{thm} by\n  {prf}"
    if ":=" in thm:
        return thm
    if not prf:
        return thm
    if prf.startswith(":="):
        return f"{thm} {prf}"
    if prf.startswith("by"):
        return f"{thm} := {prf}"
    return f"{thm} := by\n  {prf}"

def _theorem_with_sorry(theorem: str) -> str:
    """Return a theorem command with one sorry proof hole."""
    stmt = (theorem or "").strip()
    if not stmt:
        return ""
    for marker in (":= by", ":="):
        idx = stmt.find(marker)
        if idx >= 0:
            stmt = stmt[:idx].rstrip()
            break
    return f"{stmt} := by sorry"

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
        self._loaded_preamble = ""
        self._preamble_env_cache: dict[str, int] = {}

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

        # 预加载 preamble. ProofNet uses per-problem JSONL headers, so its
        # pool starts with an empty preamble and loads headers lazily.
        if preamble:
            resp = await self._transport.send({"cmd": preamble, "env": 0})
            if resp and "env" in resp:
                self._base_env_id = resp["env"]
                self._loaded_preamble = preamble.strip()
                if self._loaded_preamble:
                    self._preamble_env_cache[self._loaded_preamble] = self._base_env_id
                logger.info(
                    f"AsyncSession {self.session_id}: started, "
                    f"env_id={self._base_env_id}")
                return True
        else:
            logger.info(
                f"AsyncSession {self.session_id}: started, env_id=0 "
                f"(no startup preamble)")

        return True

    async def _env_for_preamble(
            self, preamble: str) -> tuple[bool, int, str, str]:
        preamble = (preamble or "").strip()
        if not preamble:
            return True, self.base_env_id, "", ""
        cached = self._preamble_env_cache.get(preamble)
        if cached is not None:
            return True, cached, "", ""

        env_id = self.base_env_id
        code = preamble
        loaded = self._loaded_preamble.strip()
        if loaded and env_id > 0 and preamble.startswith(loaded):
            suffix = preamble[len(loaded):].strip()
            if not suffix:
                self._preamble_env_cache[preamble] = env_id
                return True, env_id, "", ""
            code = suffix

        resp = await self._transport.send({"cmd": code, "env": env_id})
        if not resp:
            return False, env_id, "REPL communication failed while loading preamble", "internal"

        messages = resp.get("messages", [])
        errors = [m for m in messages if m.get("severity") == "error"]
        if errors:
            category, combined_msg, _meta = _classify_error_structured(messages)
            return (
                False, env_id,
                (combined_msg or errors[0].get("data", ""))[:500],
                category,
            )
        if "env" not in resp:
            return False, env_id, "REPL did not return an env for preamble", "internal"

        env_id = int(resp["env"])
        self._preamble_env_cache[preamble] = env_id
        return True, env_id, "", ""

    async def start_proof(self, theorem: str, preamble: str = "") -> TacticFeedback:
        """Open a theorem as an interactive proofState via lean4-repl."""
        t0 = time.time()
        self._total_requests += 1
        code = _theorem_with_sorry(theorem)
        if not code:
            return TacticFeedback(
                success=False, tactic="<start_proof>",
                error_message="empty theorem statement",
                error_category="invalid_request",
                elapsed_ms=0, session_id=self.session_id)

        if not (self._transport and self._transport.is_alive
                and not self._transport.is_fallback
                and not getattr(self._transport, "is_single_shot", False)):
            elapsed = int((time.time() - t0) * 1000)
            return TacticFeedback(
                success=False, tactic="<start_proof>",
                error_message="Interactive proofState requires lean4-repl",
                error_category="no_repl",
                elapsed_ms=elapsed, session_id=self.session_id)

        ok, env_id, error_message, error_category = await self._env_for_preamble(preamble)
        if not ok:
            elapsed = int((time.time() - t0) * 1000)
            return TacticFeedback(
                success=False, tactic="<start_proof>",
                error_message=error_message,
                error_category=error_category or "preamble_failed",
                elapsed_ms=elapsed, session_id=self.session_id)

        resp = await self._transport.send({"cmd": code, "env": env_id})
        elapsed = int((time.time() - t0) * 1000)
        if not resp:
            return TacticFeedback(
                success=False, tactic="<start_proof>",
                error_message="REPL communication failed",
                error_category="internal",
                elapsed_ms=elapsed, session_id=self.session_id)

        messages = resp.get("messages", [])
        errors = [m for m in messages if m.get("severity") == "error"]
        if errors:
            category, combined_msg, _meta = _classify_error_structured(messages)
            return TacticFeedback(
                success=False, tactic="<start_proof>",
                error_message=(combined_msg or errors[0].get("data", ""))[:500],
                error_category=category,
                elapsed_ms=elapsed, session_id=self.session_id)

        sorries = resp.get("sorries", []) or []
        if not sorries:
            return TacticFeedback(
                success=False, tactic="<start_proof>",
                error_message="REPL did not return a proofState for sorry",
                error_category="no_proof_state",
                elapsed_ms=elapsed, session_id=self.session_id)

        first = sorries[0]
        proof_state = int(first.get("proofState", -1))
        goal = first.get("goal", "")
        if proof_state < 0:
            return TacticFeedback(
                success=False, tactic="<start_proof>",
                error_message="invalid proofState returned by REPL",
                error_category="no_proof_state",
                elapsed_ms=elapsed, session_id=self.session_id)

        return TacticFeedback(
            success=True, tactic="<start_proof>",
            new_env_id=proof_state,
            remaining_goals=[goal] if goal else [],
            is_proof_complete=False,
            elapsed_ms=elapsed, session_id=self.session_id)

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
            if getattr(self._transport, "is_single_shot", False):
                resp = await self._transport.send({"cmd": full_code, "env": 0})
            else:
                ok, env_id, error_message, error_category = (
                    await self._env_for_preamble(preamble))
                if not ok:
                    elapsed = int((time.time() - t0) * 1000)
                    return FullVerifyResult(
                        success=False,
                        errors=[{
                            "message": error_message,
                            "category": error_category or "preamble_failed",
                        }],
                        stderr=error_message,
                        elapsed_ms=elapsed,
                    )
                code = _declaration_code(theorem, proof)
                resp = await self._transport.send({"cmd": code, "env": env_id})
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
        if getattr(self._transport, "is_single_shot", False):
            resp = await self._transport.send({"cmd": tactic, "env": env_id})
        else:
            resp = await self._transport.send({
                "tactic": tactic,
                "proofState": env_id,
            })
        elapsed = int((time.time() - t0) * 1000)

        if not resp:
            return TacticFeedback(
                success=False, tactic=tactic,
                error_message="REPL communication failed",
                error_category="internal",
                elapsed_ms=elapsed, session_id=self.session_id)

        messages = resp.get("messages", [])
        errors = [m for m in messages if m.get("severity") == "error"]
        new_env = resp.get("proofState", resp.get("env", env_id))
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
    def is_single_shot(self) -> bool:
        return bool(getattr(self._transport, "is_single_shot", False))

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
                 timeout_seconds: int = 30,
                 transport_factory: Optional[
                     Callable[[int], "REPLTransport"]] = None):
        """Initialise the pool.

        Args:
            pool_size:        number of REPL sessions to maintain
            project_dir:      Lean project root (passed to transports)
            preamble:         imports to load on each session start
            timeout_seconds:  per-request timeout
            transport_factory: optional callable ``(session_id) -> REPLTransport``.
                When given, every session — including those added later via
                ``add_session()`` — will be constructed with the transport
                this factory returns. The factory is responsible for
                creating *already-configured but not-yet-started* transports;
                ``AsyncLeanSession.start()`` will call ``.start()`` on them.

                When ``None``, every session falls back
                to ``LocalTransport`` exactly as before. **This kwarg is
                strictly additive: callers that pass only positional args
                or that pass no transport_factory are unaffected.**

                Use cases unlocked by passing a factory:
                  * RL roll-outs that want Kimina batch verify
                    (``factory = lambda i: KiminaServerBackend(...)``)
                  * Pantograph-backed sessions for mvar focusing
                  * LooKeng outer / Kimina inner chains
                See ``sampler.proof_env._make_transport_factory`` and
                ``prover.unified.factory.build_infra_backends`` for
                examples of typical builders.
        """
        self.pool_size = pool_size
        self.project_dir = project_dir
        self.preamble = preamble
        self.timeout = timeout_seconds
        self._transport_factory = transport_factory
        self._sessions: list[AsyncLeanSession] = []
        self._session_available = asyncio.Condition()
        self._condition_loop = None
        self._started = False
        self._total_requests = 0
        self._total_latency_ms = 0
        self._compile_cache = _CompileCache(maxsize=1024)
        self._env_cache: dict[str, int] = {}
        self._proof_state_sessions: dict[int, tuple[AsyncLeanSession, int]] = {}
        self._next_proof_handle = 1_000_000
        self._next_session_id = pool_size
        self._env_version = 0

    def _make_session(self, session_id: int) -> "AsyncLeanSession":
        """Construct one session, threading the transport factory through.

        Centralised so ``start()`` and ``add_session()`` cannot drift on
        whether the factory is honoured.
        """
        transport = None
        if self._transport_factory is not None:
            try:
                transport = self._transport_factory(session_id)
            except Exception as e:
                logger.warning(
                    f"transport_factory({session_id}) raised {e!r}; "
                    f"session will fall back to LocalTransport")
                transport = None
        return AsyncLeanSession(
            session_id=session_id,
            project_dir=self.project_dir,
            timeout_seconds=self.timeout,
            transport=transport)

    async def start(self) -> bool:
        """启动所有会话"""
        if self._started:
            return True

        logger.info(f"AsyncLeanPool: starting {self.pool_size} sessions...")

        sessions = [self._make_session(i) for i in range(self.pool_size)]

        # Startup strategy (集中处理 transport_factory via _make_session):
        # A non-empty preamble triggers a heavy ``import Mathlib`` compile in
        # each session. Launching all sessions concurrently makes them all
        # cold-load Mathlib's .oleans at once, which can wedge on slow storage.
        # So when there is a preamble, start the first session alone and await
        # its compile (warming the OS page cache), then start the rest
        # concurrently — they read warm oleans and finish fast. With no preamble
        # there is no heavy compile, so start everything concurrently as before.
        if self.preamble and self.pool_size > 1:
            first = await asyncio.gather(
                sessions[0].start(self.preamble), return_exceptions=True)
            rest = await asyncio.gather(
                *(s.start(self.preamble) for s in sessions[1:]),
                return_exceptions=True)
            results = [first[0], *rest]
        else:
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

    async def start_proof(self, theorem: str, preamble: str = "") -> TacticFeedback:
        """Create an interactive proofState for a theorem."""
        session = await self._acquire_session()
        try:
            with metrics.timer("repl.start_proof"):
                result = await session.start_proof(theorem, preamble=preamble)
            self._record_latency(result.elapsed_ms)
            if not result.success:
                return result
            handle = self._next_proof_handle
            self._next_proof_handle += 1
            self._proof_state_sessions[handle] = (session, result.new_env_id)
            result.new_env_id = handle
            return result
        finally:
            await self._release_session(session)

    async def try_tactic(self, env_id: int, tactic: str) -> TacticFeedback:
        """在空闲会话或绑定 proofState 的会话上尝试一条 tactic"""
        bound = self._proof_state_sessions.get(env_id)
        if bound is not None:
            session = await self._acquire_specific_session(bound[0])
            actual_state = bound[1]
        else:
            session = await self._acquire_session()
            actual_state = env_id
        try:
            with metrics.timer("repl.try_tactic"):
                result = await session.try_tactic(actual_state, tactic)
            self._record_latency(result.elapsed_ms)
            metrics.increment("repl.try_tactic.total")
            if result.success:
                metrics.increment("repl.try_tactic.success")
                if bound is not None:
                    handle = self._next_proof_handle
                    self._next_proof_handle += 1
                    self._proof_state_sessions[handle] = (
                        session, result.new_env_id)
                    result.new_env_id = handle
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
            metrics.increment("repl.verify.cache_hit")
            return cached

        metrics.increment("repl.verify.cache_miss")
        session = await self._acquire_session()
        try:
            with metrics.timer("repl.verify_complete"):
                result = await session.verify_complete(theorem, proof, preamble)
            metrics.increment("repl.verify.total")
            if result.success:
                metrics.increment("repl.verify.success")
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
            session = self._make_session(sid)
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
        single_shot = sum(1 for s in self._sessions if s.is_single_shot)
        return {
            "pool_size": self.pool_size,
            "active_sessions": sum(1 for s in self._sessions if s.is_alive),
            "busy_sessions": sum(1 for s in self._sessions if s.is_busy),
            "fallback_sessions": fallback,
            "single_shot_sessions": single_shot,
            "all_fallback": fallback == len(self._sessions) and len(self._sessions) > 0,
            "all_single_shot": single_shot == len(self._sessions) and len(self._sessions) > 0,
            "total_requests": self._total_requests,
            "avg_latency_ms": round(avg_latency, 1),
            "compile_cache": self._compile_cache.stats(),
            "env_cache_size": len(self._env_cache),
            "proof_state_handles": len(self._proof_state_sessions),
        }

    @property
    def base_env_id(self) -> int:
        """Base env_id after preamble loading (for SearchCoordinator)."""
        for s in self._sessions:
            if s.is_alive:
                return s._base_env_id
        return 0

    # ── 会话调度 ──

    def _ensure_condition_for_current_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._condition_loop is not loop:
            self._session_available = asyncio.Condition()
            self._condition_loop = loop

    async def _acquire_session(self) -> AsyncLeanSession:
        """异步获取空闲会话 (asyncio.Condition)"""
        self._ensure_condition_for_current_loop()
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
            overflow = self._make_session(sid)
            await overflow.start(self.preamble)
            overflow._busy = True
            overflow._is_overflow = True
            self._sessions.append(overflow)
            return overflow

    async def _acquire_specific_session(
            self, target: AsyncLeanSession) -> AsyncLeanSession:
        """Acquire the session that owns a proofState."""
        self._ensure_condition_for_current_loop()
        async with self._session_available:
            deadline = time.time() + self.timeout
            while True:
                if target.is_alive and not target.is_busy:
                    target._busy = True
                    return target
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"proofState session {target.session_id} is busy")
                try:
                    await asyncio.wait_for(
                        self._session_available.wait(),
                        timeout=min(remaining, 1.0))
                except asyncio.TimeoutError:
                    continue

    async def _release_session(self, session: AsyncLeanSession):
        """释放会话并通知等待者

        Overflow 会话在释放时自动关闭并移除, 防止列表无限增长。
        """
        self._ensure_condition_for_current_loop()
        async with self._session_available:
            session._busy = False

            if session._is_overflow:
                try:
                    await session.close()
                except Exception as e:
                    logger.warning(f"Failed to close overflow session: {e}")
                try:
                    self._sessions.remove(session)
                except ValueError as _exc:
                    logger.debug(f"Suppressed exception: {_exc}")
                logger.debug(
                    f"AsyncLeanPool: removed overflow session "
                    f"{session.session_id}, pool size={len(self._sessions)}")

            self._session_available.notify()

    def _record_latency(self, ms: int):
        self._total_requests += 1
        self._total_latency_ms += ms

# ═══════════════════════════════════════════════════════════════
# SyncLeanPool: 同步包装器 — AsyncLeanPool 的唯一同步入口
# ═══════════════════════════════════════════════════════════════

import threading

class SyncLeanPool:
    """AsyncLeanPool 的同步包装器

    内部维护一个独立线程运行 asyncio 事件循环, 所有同步方法
    通过 ``run_coroutine_threadsafe`` 委托给 AsyncLeanPool。

    与旧的同步 LeanPool 接口完全兼容 (drop-in 替代)。

    Usage::

        pool = SyncLeanPool(pool_size=4, project_dir="/path")
        pool.start()
        result = pool.verify_complete("theorem t : True", ":= by trivial")
        pool.shutdown()
    """

    def __init__(self, pool_size: int = 4, project_dir: str = ".",
                 preamble: str = "import Mathlib",
                 timeout_seconds: int = 30,
                 transport_factory: Optional[
                     Callable[[int], "REPLTransport"]] = None):
        self._async_pool = AsyncLeanPool(
            pool_size=pool_size, project_dir=project_dir,
            preamble=preamble, timeout_seconds=timeout_seconds,
            transport_factory=transport_factory)

        # 独立事件循环线程 — 生命周期与 Pool 绑定
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="SyncLeanPool-EventLoop")
        self._thread.start()

        # 透传属性
        self.pool_size = pool_size
        self.project_dir = project_dir
        self.preamble = preamble
        self.timeout = timeout_seconds

    def _run_loop(self):
        """事件循环线程入口"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coro):
        """在事件循环线程中执行协程并同步等待结果"""
        if self._loop.is_closed():
            raise RuntimeError("SyncLeanPool event loop is closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self.timeout + 60)

    # ── 公共接口 (与旧 LeanPool 签名完全一致) ──

    def start(self) -> bool:
        return self._run(self._async_pool.start())

    def start_proof(self, theorem: str, preamble: str = "") -> 'TacticFeedback':
        return self._run(self._async_pool.start_proof(theorem, preamble=preamble))

    def try_tactic(self, env_id: int, tactic: str) -> 'TacticFeedback':
        return self._run(self._async_pool.try_tactic(env_id, tactic))

    def try_tactics_parallel(self, env_id: int,
                             tactics: list[str]) -> list['TacticFeedback']:
        return self._run(self._async_pool.try_tactics_parallel(env_id, tactics))

    def verify_complete(self, theorem: str, proof: str,
                        preamble: str = "") -> 'FullVerifyResult':
        return self._run(self._async_pool.verify_complete(theorem, proof, preamble))

    def share_lemma(self, lemma_code: str, *,
                    name: str = "", statement: str = "",
                    proof: str = "") -> list[int]:
        return self._run(self._async_pool.share_lemma(
            lemma_code, name=name, statement=statement, proof=proof))

    def shutdown(self):
        try:
            self._run(self._async_pool.shutdown())
        except Exception as e:
            logger.warning(f"SyncLeanPool shutdown error: {e}")
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()

    def stats(self) -> dict:
        return self._async_pool.stats()

    @property
    def base_env_id(self) -> int:
        return self._async_pool.base_env_id

    def get_cached_env_id(self, preamble_str: str):
        """P1-5: 查询环境缓存, 避免重复 import 预加载"""
        key = hashlib.sha256(preamble_str.encode()).hexdigest()[:16]
        return self._async_pool._env_cache.get(key)

    # ── PoolScaler 接口 ──

    def add_session(self) -> bool:
        return self._run(self._async_pool.add_session())

    def remove_idle_session(self) -> bool:
        return self._run(self._async_pool.remove_idle_session())

    # ── 内部属性透传 (向后兼容: 测试中访问的内部状态) ──

    @property
    def _compile_cache(self):
        return self._async_pool._compile_cache

    @property
    def _sessions(self):
        return self._async_pool._sessions

    @_sessions.setter
    def _sessions(self, value):
        self._async_pool._sessions = value

    @property
    def _started(self):
        return self._async_pool._started

    @_started.setter
    def _started(self, value):
        self._async_pool._started = value

    @property
    def _env_cache(self):
        return self._async_pool._env_cache

    @_env_cache.setter
    def _env_cache(self, value):
        self._async_pool._env_cache = value

    @property
    def _env_version(self):
        return self._async_pool._env_version

    @_env_version.setter
    def _env_version(self, value):
        self._async_pool._env_version = value

    @property
    def _next_session_id(self):
        return self._async_pool._next_session_id

    @_next_session_id.setter
    def _next_session_id(self, value):
        self._async_pool._next_session_id = value

    @property
    def _session_available(self):
        return self._async_pool._session_available

    @property
    def _lock(self):
        """Compat: 旧 LeanPool 用 threading.Lock, 新版用 asyncio.Condition"""
        return threading.Lock()

    # ── Context manager ──

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def __del__(self):
        try:
            if self._loop and not self._loop.is_closed():
                self.shutdown()
        except Exception as _exc:
            logger.debug(f"Suppressed exception: {_exc}")
