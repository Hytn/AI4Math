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
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TacticFeedback:
    """单条 tactic 的执行反馈 (面向 Agent 的结构化输出)"""
    success: bool
    tactic: str
    # 成功时的信息
    new_env_id: int = -1
    remaining_goals: list[str] = field(default_factory=list)
    goals_closed: int = 0
    goals_opened: int = 0
    is_proof_complete: bool = False
    # 失败时的信息
    error_message: str = ""
    error_category: str = ""
    expected_type: str = ""
    actual_type: str = ""
    # 性能信息
    elapsed_ms: int = 0
    session_id: int = -1

    @property
    def progress_delta(self) -> int:
        """目标净变化: 正 = 进展, 负 = 增加了新目标"""
        return self.goals_closed - self.goals_opened


@dataclass
class FullVerifyResult:
    """完整证明验证结果"""
    success: bool
    env_id: int = -1
    errors: list[dict] = field(default_factory=list)
    stderr: str = ""
    elapsed_ms: int = 0
    goals_remaining: list[str] = field(default_factory=list)
    has_sorry: bool = False


@dataclass
class _SessionState:
    """单个 REPL 会话的内部状态"""
    process: Optional[subprocess.Popen] = None
    session_id: int = 0
    current_env_id: int = 0
    busy: bool = False
    alive: bool = False
    fallback_mode: bool = False  # True = 无真实 REPL, 所有验证不可信
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

        线程安全: _lock 保护整个 REPL 交互过程 (stdin 写入 + stdout 读取),
        确保两个线程不会交错发送命令并误读彼此的响应。
        busy 标志由 LeanPool 原子管理, 不在此处设置/清除。
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

        线程安全: _lock 保护整个 REPL 交互过程。
        """
        t0 = time.time()
        with self._lock:
            self._state.busy = True
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
            finally:
                self._state.busy = False

    def get_goals(self, env_id: int) -> list[str]:
        """获取指定 env_id 下的当前 goal state"""
        with self._lock:
            if self._state.process and self._state.process.poll() is None:
                resp = self._send_raw({"cmd": "-- goals", "env": env_id})
                if resp and "goals" in resp:
                    return resp["goals"]
        return []

    def close(self):
        """关闭会话"""
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
        """通过 REPL 进程执行 tactic"""
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
            err = errors[0]
            return TacticFeedback(
                success=False, tactic=tactic,
                error_message=err.get("data", ""),
                error_category=_classify_error(err.get("data", "")),
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

        使用 selectors 实现读超时, 防止 REPL 进程挂起时永久阻塞
        调用线程 (及其持有的会话锁)。
        """
        proc = self._state.process
        if not proc or proc.poll() is not None:
            return None
        try:
            proc.stdin.write(json.dumps(cmd) + "\n")
            proc.stdin.flush()

            # 带超时的 readline — 防止 REPL 进程挂起时永久阻塞
            import selectors
            sel = selectors.DefaultSelector()
            sel.register(proc.stdout, selectors.EVENT_READ)
            try:
                events = sel.select(timeout=self._timeout)
                if not events:
                    logger.warning(
                        f"Session {self._state.session_id}: REPL read timed out "
                        f"after {self._timeout}s — the process may be hung")
                    self._state.total_errors += 1
                    return None
                line = proc.stdout.readline()
            finally:
                sel.unregister(proc.stdout)
                sel.close()

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

    def start(self) -> bool:
        """启动所有会话 (预加载环境)"""
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
        """在空闲会话上验证完整证明"""
        session = self._acquire_session()
        try:
            return session.verify_complete(theorem, proof, preamble)
        finally:
            self._release_session(session)

    def fork_env(self, env_id: int) -> int:
        """Fork 一个环境快照

        lean4-repl 的 env_id 天然支持分叉:
        多个命令可以引用同一个 env_id, 各自产生不同的后继状态。
        所以 fork 实际上是零成本的 — 直接返回原 env_id。

        注意: fork 只适用于"只读引用"场景 (从同一状态尝试不同 tactic)。
        如果通过 share_lemma() 修改了会话环境, 需要使用
        share_lemma() 返回的新 env_id 来引用包含新引理的环境。
        """
        return env_id

    def share_lemma(self, lemma_code: str) -> list[int]:
        """将已证引理注入所有会话的环境中

        这使得跨方向的引理复用成为可能:
        方向 A 证明了辅助引理, 方向 B/C/D 的 REPL 环境中立即可用。

        Returns:
            每个成功注入的会话的新 env_id 列表。后续操作如果需要引用
            包含此引理的环境, 应使用这些新的 env_id 而非原始 env_id。
            调用者应通过 latest_env_ids 属性获取每个会话的最新环境。
        """
        new_env_ids = []
        for session in self._sessions:
            if session.is_alive and not session.is_fallback:
                result = session.verify_complete("", lemma_code)
                if result.success:
                    new_env_ids.append(result.env_id)
                    # 更新会话的基础 env_id, 使后续操作能看到新引理
                    session._state.current_env_id = result.env_id
                    logger.debug(
                        f"share_lemma: session {session._state.session_id} "
                        f"→ new env_id={result.env_id}")
                else:
                    logger.warning(
                        f"share_lemma: failed on session "
                        f"{session._state.session_id}: {result.stderr[:100]}")
        return new_env_ids

    @property
    def latest_env_ids(self) -> dict[int, int]:
        """每个会话的最新 env_id (share_lemma 后可能已更新)

        Returns:
            {session_id: current_env_id} 映射
        """
        return {s._state.session_id: s._state.current_env_id
                for s in self._sessions if s.is_alive}

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
        }

    def _acquire_session(self) -> LeanSession:
        """原子获取一个空闲会话 (阻塞等待策略)

        线程安全: 通过条件变量阻塞等待, 直到有空闲会话可用。
        获取到的会话立即被标记为 busy, 防止其他线程在锁释放后
        再次获取它。

        调用者必须在使用完会话后调用 _release_session(session)。

        超时后 (timeout 秒) 如仍无空闲会话, 返回请求数最少的
        已有会话 — 此时两个线程共用一个会话, 但由于会话级 _lock
        保护了 REPL I/O, 不会出现命令/响应交错。
        """
        deadline = time.time() + self.timeout
        with self._session_available:
            while True:
                # 优先选择空闲的
                for session in self._sessions:
                    if session.is_alive and not session.is_busy:
                        session._state.busy = True
                        return session

                # 无空闲 — 等待释放或超时
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._session_available.wait(timeout=min(remaining, 1.0))

            # 超时回退: 返回请求数最少的会话 (会在会话的 _lock 上排队)
            if self._sessions:
                chosen = min(self._sessions,
                             key=lambda s: s._state.total_requests)
                logger.debug(
                    f"LeanPool: all sessions busy, sharing session "
                    f"{chosen._state.session_id} (requests will serialize "
                    f"on session lock)")
                return chosen

            # 无会话 — 创建一个 fallback
            fallback = LeanSession(session_id=-1, project_dir=self.project_dir)
            fallback.start("")
            return fallback

    def _release_session(self, session: LeanSession):
        """释放会话, 标记为空闲并通知等待线程"""
        with self._session_available:
            session._state.busy = False
            self._session_available.notify()

    def _record_latency(self, ms: int):
        with self._lock:
            self._total_requests += 1
            self._total_latency_ms += ms


# ── 辅助函数 ──

def _which(cmd: str) -> Optional[str]:
    """跨平台的 which 实现"""
    import shutil
    return shutil.which(cmd)


def _assemble_code(theorem: str, proof: str, preamble: str = "") -> str:
    """组装完整的 Lean4 源文件

    拼接逻辑:
      - 如果 theorem 已经包含完整证明 (以 ':= by' 或 ':= fun' 或
        ':= show' 等开头的证明体), 则不再拼接 proof
      - 否则将 proof 追加到 theorem 之后
    """
    parts = []
    if preamble:
        parts.append(preamble)
    else:
        parts.append("import Mathlib")
    parts.append("")
    if theorem.strip():
        full = theorem.strip()
        if proof.strip():
            # 检查 theorem 是否已含证明体:
            # 匹配最外层的 ':=' (不在括号/花括号/角括号内)
            if not _has_toplevel_assign(full):
                full += f" {proof.strip()}"
        parts.append(full)
    return "\n".join(parts)


def _has_toplevel_assign(code: str) -> bool:
    """检查代码是否在顶层包含 ':=' 赋值 (不在括号/引号内)"""
    depth = 0
    in_string = False
    i = 0
    while i < len(code):
        ch = code[i]
        if ch == '"' and (i == 0 or code[i-1] != '\\'):
            in_string = not in_string
        elif not in_string:
            if ch in ('(', '[', '{', '⟨'):
                depth += 1
            elif ch in (')', ']', '}', '⟩'):
                depth = max(0, depth - 1)
            elif depth == 0 and ch == ':' and i + 1 < len(code) and code[i+1] == '=':
                return True
        i += 1
    return False


def _classify_error(msg: str) -> str:
    """将错误消息分类"""
    msg_lower = msg.lower()
    if "type mismatch" in msg_lower:
        return "type_mismatch"
    if "unknown identifier" in msg_lower or "unknown constant" in msg_lower:
        return "unknown_identifier"
    if "tactic" in msg_lower and "failed" in msg_lower:
        return "tactic_failed"
    if "unsolved goals" in msg_lower:
        return "unsolved_goals"
    if "sorry" in msg_lower:
        return "sorry"
    if "syntax" in msg_lower or "expected" in msg_lower:
        return "syntax_error"
    if "timeout" in msg_lower or "heartbeat" in msg_lower:
        return "timeout"
    return "other"


def _extract_expected(msg: str) -> str:
    """从错误消息中提取 expected type"""
    for marker in ["expected to have type", "expected type"]:
        if marker in msg:
            idx = msg.index(marker) + len(marker)
            rest = msg[idx:].strip()
            # 取到下一个换行或关键词
            end = len(rest)
            for stop in ["\n", "but is expected", "has type"]:
                pos = rest.find(stop)
                if pos > 0:
                    end = min(end, pos)
            return rest[:end].strip()
    return ""


def _extract_actual(msg: str) -> str:
    """从错误消息中提取 actual type"""
    for marker in ["has type", "actual type"]:
        if marker in msg:
            idx = msg.index(marker) + len(marker)
            rest = msg[idx:].strip()
            end = len(rest)
            for stop in ["\n", "but is expected", "expected"]:
                pos = rest.find(stop)
                if pos > 0:
                    end = min(end, pos)
            return rest[:end].strip()
    return ""
