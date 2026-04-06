"""engine/lean_pool.py — Lean4 REPL 连接池

核心思路: 不自建简化内核, 而是把 Lean4 本身变成 Agent 的高性能交互环境。

启动 N 个 Lean4 REPL 长连接进程, 每个进程预加载 Mathlib 环境。
Agent 的验证请求通过调度器分发到空闲进程, 实现:
  - 延迟: 2-12s → 50-500ms (环境预加载, 增量验证)
  - 吞吐: 串行 → N 路并行 (连接池化)
  - 精度: 100% (Lean4 本身在验证, 不是模拟)

关键机制:
  1. env_id 复用: lean4-repl 每条命令返回 env_id, 后续命令可在此基础上继续
  2. 健康检查: 定期 ping, 异常自动重启
  3. 请求路由: least-busy 策略分发
  4. 环境缓存: 相同 import 语句的环境只加载一次
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

# Import shared types and functions from _core (single source of truth)
from engine._core import (
    CompileCache as _CompileCache,
    TacticFeedback,
    FullVerifyResult,
    which as _which,
    assemble_code as _assemble_code,
    classify_error as _classify_error,
    classify_error_structured as _classify_error_structured,
    extract_expected as _extract_expected,
    extract_actual as _extract_actual,
    make_cache_key,
)

logger = logging.getLogger(__name__)


# _CompileCache, TacticFeedback, FullVerifyResult, and helper functions
# are now imported from engine._core (single source of truth).
# Backward-compatible names are available via the imports above.


@dataclass
class _SessionState:
    """单个 REPL 会话的内部状态"""
    process: Optional[subprocess.Popen] = None
    session_id: int = 0
    current_env_id: int = 0
    busy: bool = False
    alive: bool = False
    fallback_mode: bool = False  # True = 无真实 REPL, 所有验证不可信
    is_overflow: bool = False    # True = 临时创建的溢出会话, 用完即关
    last_heartbeat: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    project_dir: str = "."


class LeanSession:
    """单个 Lean4 REPL 长连接会话

    封装与一个 lean4-repl 进程的通信。
    支持: 发送命令、获取 goal state、尝试 tactic。
    """

    def __init__(self, session_id: int, project_dir: str = ".",
                 timeout_seconds: int = 30):
        self._state = _SessionState(session_id=session_id,
                                    project_dir=project_dir)
        self._timeout = timeout_seconds
        self._lock = threading.Lock()
        self._repl_binary = self._find_repl_binary(project_dir)
        self._selector = None  # 延迟创建, 在 start() 中注册 stdout

    @staticmethod
    def _find_repl_binary(project_dir: str) -> Optional[str]:
        """查找 lean4-repl 可执行文件"""
        candidates = [
            os.path.join(project_dir, ".lake", "build", "bin", "repl"),
            "lean",  # fallback: 直接用 lean 命令
        ]
        for c in candidates:
            if os.path.isfile(c) or _which(c):
                return c
        return None

    def start(self, preamble: str = "import Mathlib") -> bool:
        """启动 REPL 进程并预加载环境

        Args:
            preamble: 预加载的 import 语句 (默认加载 Mathlib)

        Returns:
            是否启动成功
        """
        if not self._repl_binary:
            logger.warning(
                f"Session {self._state.session_id}: [FALLBACK MODE] "
                f"No Lean4 REPL binary found. All verification results are "
                f"UNRELIABLE — install elan + lean4 for real verification. "
                f"See: https://leanprover-community.github.io/get_started.html")
            self._state.alive = True
            self._state.fallback_mode = True
            return True

        try:
            self._state.process = subprocess.Popen(
                [self._repl_binary],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._state.project_dir,
                text=True,
                bufsize=1,
            )
            # 创建持久 selector, 注册 stdout 用于带超时的 readline
            import selectors
            self._selector = selectors.DefaultSelector()
            self._selector.register(
                self._state.process.stdout, selectors.EVENT_READ)

            # 发送预加载命令
            if preamble:
                resp = self._send_raw({"cmd": preamble, "env": 0})
                if resp and "env" in resp:
                    self._state.current_env_id = resp["env"]
                    self._state.alive = True
                    logger.info(
                        f"Session {self._state.session_id}: started, "
                        f"env_id={self._state.current_env_id}")
                    return True

            self._state.alive = True
            return True

        except FileNotFoundError:
            logger.warning(
                f"Session {self._state.session_id}: [FALLBACK MODE] "
                f"REPL binary not found at '{self._repl_binary}'. "
                f"Verification results will be UNRELIABLE.")
            self._state.alive = True
            self._state.fallback_mode = True
            return True
        except Exception as e:
            logger.error(f"Session {self._state.session_id}: start failed: {e}")
            return False

    def try_tactic(self, env_id: int, tactic: str) -> TacticFeedback:
        """在指定 env_id 上尝试一条 tactic

        这是 Agent 搜索的核心操作:
        从一个证明状态 (env_id) 出发, 执行一条 tactic,
        返回结构化的反馈 (新状态/错误/进度)。

        注意: busy 标志由 LeanPool 原子管理, 不在此处设置/清除。
        会话级 _lock 仅用于保护 REPL 进程的 stdin/stdout 串行化。
        """
        t0 = time.time()
        with self._lock:
            self._state.total_requests += 1

        if self._state.process and self._state.process.poll() is None:
            return self._try_tactic_repl(env_id, tactic, t0)
        else:
            return self._try_tactic_fallback(env_id, tactic, t0)

    def verify_complete(self, theorem: str, proof: str,
                        preamble: str = "") -> FullVerifyResult:
        """验证一个完整的定理+证明

        注意: busy 标志由 LeanPool 原子管理, 不在此处设置/清除。
        会话级 _lock 仅用于保护 REPL 进程的 stdin/stdout 串行化。
        """
        t0 = time.time()
        with self._lock:
            self._state.total_requests += 1

        try:
            full_code = _assemble_code(theorem, proof, preamble)

            if self._state.process and self._state.process.poll() is None:
                resp = self._send_raw({
                    "cmd": full_code,
                    "env": 0,
                })
                elapsed = int((time.time() - t0) * 1000)
                return self._parse_verify_response(resp, elapsed)
            else:
                return self._verify_fallback(full_code, t0)
        except Exception:
            raise

    def get_goals(self, env_id: int) -> list[str]:
        """获取指定 env_id 下的当前 goal state"""
        if self._state.process and self._state.process.poll() is None:
            resp = self._send_raw({"cmd": "-- goals", "env": env_id})
            if resp and "goals" in resp:
                return resp["goals"]
        return []

    def close(self):
        """关闭会话"""
        # 先关闭 selector (避免操作已关闭的 fd)
        if self._selector:
            try:
                self._selector.close()
            except Exception:
                pass
            self._selector = None

        if self._state.process:
            try:
                self._state.process.terminate()
                self._state.process.wait(timeout=5)
            except Exception:
                try:
                    self._state.process.kill()
                except Exception:
                    pass
        self._state.alive = False

    # ── 内部方法 ──

    def _try_tactic_repl(self, env_id: int, tactic: str,
                         t0: float) -> TacticFeedback:
        """通过 REPL 进程执行 tactic (P1-9: 结构化错误解析)"""
        resp = self._send_raw({"cmd": tactic, "env": env_id})
        elapsed = int((time.time() - t0) * 1000)

        if not resp:
            return TacticFeedback(
                success=False, tactic=tactic,
                error_message="REPL communication failed",
                error_category="internal",
                elapsed_ms=elapsed, session_id=self._state.session_id)

        messages = resp.get("messages", [])
        errors = [m for m in messages if m.get("severity") == "error"]
        new_env = resp.get("env", env_id)
        goals = resp.get("goals", [])

        if errors:
            # P1-9: 使用结构化分类, 聚合多条错误
            category, combined_msg, meta = _classify_error_structured(messages)
            err = errors[0]
            return TacticFeedback(
                success=False, tactic=tactic,
                error_message=combined_msg[:500] if combined_msg else err.get("data", ""),
                error_category=category,
                expected_type=_extract_expected(err.get("data", "")),
                actual_type=_extract_actual(err.get("data", "")),
                elapsed_ms=elapsed, session_id=self._state.session_id)

        return TacticFeedback(
            success=True, tactic=tactic,
            new_env_id=new_env,
            remaining_goals=goals,
            is_proof_complete=(len(goals) == 0),
            elapsed_ms=elapsed, session_id=self._state.session_id)

    def _try_tactic_fallback(self, env_id: int, tactic: str,
                             t0: float) -> TacticFeedback:
        """Fallback: 无 REPL 进程时, 用模拟模式"""
        elapsed = int((time.time() - t0) * 1000)
        # 在无 Lean4 环境时, 返回"未知"状态, 不误判
        return TacticFeedback(
            success=False, tactic=tactic,
            error_message="No active REPL session (Lean4 not available)",
            error_category="no_backend",
            elapsed_ms=elapsed, session_id=self._state.session_id)

    def _verify_fallback(self, code: str, t0: float) -> FullVerifyResult:
        """Fallback: 用 subprocess 单次编译"""
        try:
            result = subprocess.run(
                ["lean", "--run", "-"],
                input=code, capture_output=True, text=True,
                timeout=self._timeout,
                cwd=self._state.project_dir,
            )
            elapsed = int((time.time() - t0) * 1000)
            success = result.returncode == 0 and "sorry" not in result.stderr
            return FullVerifyResult(
                success=success, stderr=result.stderr,
                elapsed_ms=elapsed,
                has_sorry="sorry" in result.stderr)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
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

    def _send_raw(self, cmd: dict) -> Optional[dict]:
        """向 REPL 进程发送 JSON 命令并读取响应

        使用持久 selectors 实现读超时, 防止 REPL 进程挂起时永久阻塞。
        selector 在 start() 中创建, 在 close() 中销毁, 避免每次调用的开销。
        """
        proc = self._state.process
        if not proc or proc.poll() is not None:
            return None
        try:
            proc.stdin.write(json.dumps(cmd) + "\n")
            proc.stdin.flush()

            # 带超时的 readline — 复用持久 selector
            if self._selector:
                events = self._selector.select(timeout=self._timeout)
                if not events:
                    logger.warning(
                        f"Session {self._state.session_id}: REPL read timed out "
                        f"after {self._timeout}s — the process may be hung")
                    self._state.total_errors += 1
                    return None
                line = proc.stdout.readline()
            else:
                # Fallback: 无 selector 时直接 readline (可能阻塞)
                line = proc.stdout.readline()

            if line:
                return json.loads(line)
            return None
        except (json.JSONDecodeError, IOError, BrokenPipeError) as e:
            logger.error(f"REPL send failed (session {self._state.session_id}): {e}")
            self._state.total_errors += 1
            return None

    @property
    def is_busy(self) -> bool:
        return self._state.busy

    @property
    def is_alive(self) -> bool:
        return self._state.alive

    @property
    def session_id(self) -> int:
        return self._state.session_id

    @property
    def base_env_id(self) -> int:
        return self._state.current_env_id

    @property
    def is_fallback(self) -> bool:
        """True if no real REPL process — verification results are unreliable."""
        return self._state.fallback_mode


class LeanPool:
    """Lean4 REPL 连接池

    管理 N 个预热的 REPL 会话, 支持并行验证。

    Usage::

        pool = LeanPool(pool_size=4, project_dir="/path/to/lean-project")
        pool.start()

        # 并行尝试多条 tactic
        results = pool.try_tactics_parallel(env_id=0, tactics=["simp", "ring", "omega"])

        # 验证完整证明
        result = pool.verify_complete("theorem t : True", ":= by trivial")

        pool.shutdown()
    """

    def __init__(self, pool_size: int = 4, project_dir: str = ".",
                 preamble: str = "import Mathlib",
                 timeout_seconds: int = 30):
        self.pool_size = pool_size
        self.project_dir = project_dir
        self.preamble = preamble
        self.timeout = timeout_seconds
        self._sessions: list[LeanSession] = []
        self._lock = threading.Lock()
        self._session_available = threading.Condition(self._lock)
        self._started = False

        # 统计
        self._total_requests = 0
        self._total_latency_ms = 0

        # P1-5: 环境缓存 — preamble_hash → base_env_id
        self._env_cache: dict[str, int] = {}
        self._env_cache_lock = threading.Lock()

        # P0-1: 编译缓存 (从 lean_repl.py 统一到此处)
        self._compile_cache = _CompileCache(maxsize=1024)

        # 单调递增的 session ID, 避免 overflow session ID 冲突
        self._next_session_id = pool_size

        # 环境版本: share_lemma 每次成功注入时递增,
        # 使 CompileCache 的旧条目自动失效
        self._env_version = 0

    def start(self) -> bool:
        """启动所有会话 (预加载环境, P1-5: 带环境缓存)"""
        if self._started:
            return True

        logger.info(f"LeanPool: starting {self.pool_size} sessions...")
        success_count = 0

        for i in range(self.pool_size):
            session = LeanSession(
                session_id=i,
                project_dir=self.project_dir,
                timeout_seconds=self.timeout,
            )
            if session.start(self.preamble):
                self._sessions.append(session)
                success_count += 1
                # P1-5: 缓存 preamble → env_id 映射
                if session.base_env_id > 0:
                    cache_key = hashlib.sha256(
                        self.preamble.encode()).hexdigest()[:16]
                    with self._env_cache_lock:
                        self._env_cache[cache_key] = session.base_env_id
            else:
                logger.warning(f"Session {i} failed to start")

        self._started = success_count > 0
        fallback_count = sum(1 for s in self._sessions if s.is_fallback)
        if fallback_count == len(self._sessions) and self._sessions:
            logger.warning(
                f"LeanPool: ALL {fallback_count} sessions are in FALLBACK MODE. "
                f"No real Lean4 REPL is available — verification results are "
                f"UNRELIABLE. Install elan + lean4 for real verification.")
        elif fallback_count > 0:
            logger.warning(
                f"LeanPool: {fallback_count}/{len(self._sessions)} sessions "
                f"are in fallback mode")
        logger.info(f"LeanPool: {success_count}/{self.pool_size} sessions ready")
        return self._started

    def try_tactic(self, env_id: int, tactic: str) -> TacticFeedback:
        """在空闲会话上尝试一条 tactic"""
        session = self._acquire_session()
        try:
            result = session.try_tactic(env_id, tactic)
            self._record_latency(result.elapsed_ms)
            return result
        finally:
            self._release_session(session)

    def try_tactics_parallel(self, env_id: int,
                             tactics: list[str]) -> list[TacticFeedback]:
        """并行尝试多条 tactic (每条分配到不同会话)

        这是 MCTS 搜索树展开的核心操作:
        在同一个证明状态 (env_id) 上, 同时尝试 N 条 tactic,
        每条成功的 tactic 产生一个新的 env_id (新的搜索节点)。
        """
        if not tactics:
            return []

        results = [None] * len(tactics)
        threads = []

        def _run(idx, tac):
            session = self._acquire_session()
            try:
                results[idx] = session.try_tactic(env_id, tac)
                self._record_latency(results[idx].elapsed_ms)
            finally:
                self._release_session(session)

        for i, tactic in enumerate(tactics):
            t = threading.Thread(target=_run, args=(i, tactic),
                                 daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=self.timeout)

        # 填充超时的结果
        for i, r in enumerate(results):
            if r is None:
                results[i] = TacticFeedback(
                    success=False, tactic=tactics[i],
                    error_message="Timeout", error_category="timeout")

        return results

    def verify_complete(self, theorem: str, proof: str,
                        preamble: str = "") -> FullVerifyResult:
        """在空闲会话上验证完整证明 (P0-1: 带编译缓存)"""
        # 缓存查找 (env_version 确保 share_lemma 后旧缓存失效)
        cache_key = make_cache_key(
            theorem, proof, preamble,
            env_fingerprint=f"v{self._env_version}")
        cached = self._compile_cache.get(cache_key)
        if cached is not None:
            return cached

        session = self._acquire_session()
        try:
            result = session.verify_complete(theorem, proof, preamble)
            # 只缓存确定性结果 (成功, 或非超时的失败)
            if result.success or "timeout" not in result.stderr.lower():
                # 缓存时将 env_id 置为 -1:
                # env_id 是特定 REPL 会话的状态引用, 跨会话无效
                import copy
                cacheable = copy.copy(result)
                cacheable.env_id = -1
                self._compile_cache.put(cache_key, cacheable)
            return result
        finally:
            self._release_session(session)


    def share_lemma(self, lemma_code: str, *,
                    name: str = "", statement: str = "",
                    proof: str = "") -> list[int]:
        """将已证引理注入所有会话的环境中 (并行)

        这使得跨方向的引理复用成为可能:
        方向 A 证明了辅助引理, 方向 B/C/D 的 REPL 环境中立即可用。

        支持两种调用方式:
          1. share_lemma(完整代码)  — lemma_code 必须是完整的 Lean4 声明
          2. share_lemma("", name=..., statement=..., proof=...)
             — 自动拼装为 ``lemma name : statement := by proof``

        Returns:
            注入成功的 session 中产生的 env_id 列表
        """
        # ── 构建合法的 Lean4 代码 ──
        if name and statement and proof:
            # 结构化参数 → 拼装
            lemma_code = f"lemma {name} : {statement} := by {proof}"
        elif name and statement and not proof:
            lemma_code = f"lemma {name} : {statement} := by sorry"
            logger.warning(
                f"share_lemma: no proof provided for '{name}', "
                f"using sorry placeholder")

        if not lemma_code or not lemma_code.strip():
            logger.warning("share_lemma: empty lemma code, skipping")
            return []

        # ── L0 基础验证: 至少要是合法的 Lean4 声明 ──
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

        if "sorry" in code and "sorry" not in (proof or ""):
            logger.info(
                f"share_lemma: code contains sorry, injecting anyway "
                f"(may be intentional placeholder)")

        alive_sessions = [s for s in self._sessions if s.is_alive]
        if not alive_sessions:
            return []

        results = [None] * len(alive_sessions)

        def _inject(idx, session):
            try:
                # 直接发送声明作为 REPL 命令, 在 session 的当前环境上执行。
                # 使用 _send_raw 而非 verify_complete, 避免 _assemble_code
                # 拼装出 "\n\ncode" 这样缺少声明头的无效代码。
                resp = session._send_raw({
                    "cmd": code,
                    "env": session.base_env_id,
                })
                if resp is None:
                    logger.warning(
                        f"share_lemma: REPL returned None on session "
                        f"{session.session_id}")
                    return
                messages = resp.get("messages", [])
                errors = [m for m in messages if m.get("severity") == "error"]
                if errors:
                    err_text = errors[0].get("data", "")[:200]
                    logger.warning(
                        f"share_lemma: injection failed on session "
                        f"{session.session_id}: {err_text}")
                    return
                new_env = resp.get("env", -1)
                if new_env >= 0:
                    # 更新 session 的 base env_id, 使后续命令在包含此引理的环境上执行
                    session._state.current_env_id = new_env
                results[idx] = new_env
            except Exception as e:
                logger.warning(
                    f"share_lemma failed on session {session.session_id}: {e}")

        threads = []
        for i, session in enumerate(alive_sessions):
            t = threading.Thread(target=_inject, args=(i, session), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=self.timeout)

        successful = [r for r in results if r is not None and r >= 0]
        if successful:
            self._env_version += 1
        return successful

    def shutdown(self):
        """关闭所有会话"""
        for session in self._sessions:
            session.close()
        self._sessions.clear()
        self._started = False
        logger.info("LeanPool: shutdown complete")

    def __enter__(self):
        """Context manager 支持: with LeanPool(...) as pool: ..."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """确保退出时关闭所有 REPL 进程, 防止孤儿进程"""
        self.shutdown()
        return False

    def __del__(self):
        """析构时尝试清理 (不保证被调用, 但作为最后防线)"""
        if self._started:
            try:
                self.shutdown()
            except Exception:
                pass

    @property
    def base_env_id(self) -> int:
        """Base env_id after preamble loading (for SearchCoordinator)."""
        for s in self._sessions:
            if s.is_alive:
                return s._state.current_env_id
        return 0

    def stats(self) -> dict:
        avg_latency = (self._total_latency_ms / self._total_requests
                       if self._total_requests else 0)
        fallback_count = sum(1 for s in self._sessions if s.is_fallback)
        return {
            "pool_size": self.pool_size,
            "active_sessions": sum(1 for s in self._sessions if s.is_alive),
            "busy_sessions": sum(1 for s in self._sessions if s.is_busy),
            "fallback_sessions": fallback_count,
            "all_fallback": fallback_count == len(self._sessions) and len(self._sessions) > 0,
            "total_requests": self._total_requests,
            "avg_latency_ms": round(avg_latency, 1),
            "compile_cache": self._compile_cache.stats(),
            "env_cache_size": len(self._env_cache),
        }

    def get_cached_env_id(self, preamble: str) -> Optional[int]:
        """P1-5: 查询环境缓存, 避免重复 import 预加载"""
        cache_key = hashlib.sha256(preamble.encode()).hexdigest()[:16]
        with self._env_cache_lock:
            return self._env_cache.get(cache_key)

    def _acquire_session(self) -> LeanSession:
        """原子获取一个空闲会话 (条件变量等待)

        线程安全: 通过 Condition 变量保证获取到的会话立即被标记为
        busy, 消除原有的竞态条件。当所有会话都忙时, 等待最多
        timeout 秒; 超时则创建临时 overflow 会话。
        """
        with self._session_available:
            deadline = time.time() + self.timeout

            while True:
                # 优先选择空闲的
                for session in self._sessions:
                    if session.is_alive and not session.is_busy:
                        session._state.busy = True
                        return session

                # 所有会话都忙 — 等待有会话被释放
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._session_available.wait(timeout=min(remaining, 1.0))

            # 超时: 创建临时 overflow 会话 (标记 busy + overflow)
            logger.warning("LeanPool: all sessions busy, creating overflow session")
            sid = self._next_session_id
            self._next_session_id += 1
            overflow = LeanSession(
                session_id=sid,
                project_dir=self.project_dir,
                timeout_seconds=self.timeout)
            overflow.start(self.preamble)
            overflow._state.busy = True
            overflow._state.is_overflow = True
            self._sessions.append(overflow)
            return overflow

    def _release_session(self, session: LeanSession):
        """释放会话, 标记为空闲并通知等待线程

        Overflow 会话在释放时自动关闭并从池中移除,
        防止 session 列表无限增长。
        """
        with self._session_available:
            session._state.busy = False

            if session._state.is_overflow:
                # 关闭并移除 overflow 会话
                try:
                    session.close()
                except Exception as e:
                    logger.warning(f"Failed to close overflow session: {e}")
                try:
                    self._sessions.remove(session)
                except ValueError:
                    pass
                logger.debug(
                    f"LeanPool: removed overflow session "
                    f"{session.session_id}, pool size={len(self._sessions)}")

            self._session_available.notify()

    def _record_latency(self, ms: int):
        with self._lock:
            self._total_requests += 1
            self._total_latency_ms += ms


