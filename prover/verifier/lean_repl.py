"""prover/verifier/lean_repl.py — Lean4 REPL 交互层 (v2)

支持三种后端:
  1. lean4-repl  — 社区 JSON REPL (推荐, https://github.com/leanprover-community/repl)
                   协议: stdin/stdout JSON {"cmd": "..."} → {"env": N, "messages": [...]}
  2. pantograph — LeanDojo 团队的 Lean4 交互后端
  3. subprocess — 降级模式, 每次完整重编译 (原有行为, 最慢)

关键改进 (相对 v1):
  - lean4-repl 模式下保持长连接进程, 单次验证延迟 ~0.1-2s (vs 重编译 2-12s)
  - verify_complete_proof() 作为最常用入口, 支持缓存
  - 进程生命周期管理: 超时自动重启, 异常恢复
  - 编译缓存以 (theorem, proof) 内容哈希为键
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import selectors
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════

@dataclass
class REPLState:
    """Tracks the state of a Lean REPL session."""
    session_id: str = ""
    is_alive: bool = False
    goal_stack: list[str] = field(default_factory=list)
    tactic_history: list[str] = field(default_factory=list)
    error_count: int = 0
    env_id: Optional[int] = None


@dataclass
class REPLResponse:
    """Response from a single REPL interaction."""
    success: bool
    goals: list[str] = field(default_factory=list)
    error: str = ""
    raw_output: str = ""
    is_complete: bool = False
    env_id: Optional[int] = None
    elapsed_ms: int = 0


# ═══════════════════════════════════════════════════════════════
# Compilation cache (shared, thread-safe)
# ═══════════════════════════════════════════════════════════════

class _CompileCache:
    """LRU cache for compilation results. Thread-safe."""

    def __init__(self, maxsize: int = 512):
        self._cache: OrderedDict[str, REPLResponse] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[REPLResponse]:
        with self._lock:
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, key: str, response: REPLResponse):
        with self._lock:
            self._cache[key] = response
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits, "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0,
            "size": len(self._cache),
        }


_global_cache = _CompileCache(maxsize=1024)


# ═══════════════════════════════════════════════════════════════
# Backend detection
# ═══════════════════════════════════════════════════════════════

def detect_best_backend(project_dir: str = ".") -> str:
    """Auto-detect the best available Lean4 REPL backend.

    Priority: lean4-repl > pantograph > subprocess > unavailable
    """
    # 1. lean4-repl binary
    repl_bin = os.path.join(project_dir, ".lake", "build", "bin", "repl")
    if os.path.isfile(repl_bin):
        return "lean4-repl"
    # Check lakefile for repl dependency
    lakefile = os.path.join(project_dir, "lakefile.lean")
    if os.path.isfile(lakefile):
        try:
            with open(lakefile) as f:
                if "leanprover-community/repl" in f.read():
                    return "lean4-repl"
        except OSError as _exc:
            logger.debug(f"Suppressed exception: {_exc}")

    # 2. pantograph
    if shutil.which("pantograph"):
        return "pantograph"

    # 3. subprocess (bare lean/lake)
    if shutil.which("lean") or shutil.which("lake"):
        return "subprocess"

    return "unavailable"


# ═══════════════════════════════════════════════════════════════
# LeanREPL — unified interface
# ═══════════════════════════════════════════════════════════════

class LeanREPL:
    """Lean4 REPL with multiple backend support.

    Typical usage (whole-proof verification)::

        repl = LeanREPL.create(project_dir="/path/to/lean-project")
        resp = repl.verify_complete_proof(
            "theorem t (n : Nat) : n = n",
            ":= by rfl")
        print(resp.is_complete)  # True
        repl.close()

    Step-by-step usage::

        repl = LeanREPL.create(project_dir="/path/to/lean-project")
        repl.start("theorem t (n : Nat) : n + 0 = n := by")
        resp = repl.send_tactic("simp")
        if resp.is_complete:
            print("Proof complete!")
        repl.close()
    """

    def __init__(self, mode: str = "auto", lean_cmd: str = "lean",
                 project_dir: str = ".", timeout: int = 60,
                 lean_pool: 'LeanPool' = None):
        if mode == "auto":
            mode = detect_best_backend(project_dir)
        self.mode = mode
        self.lean_cmd = lean_cmd
        self.project_dir = project_dir
        self.timeout = timeout
        self._state = REPLState()
        self._process: Optional[subprocess.Popen] = None
        self._cache = _global_cache
        self._lock = threading.Lock()

        # ── 统一后端: 优先使用 engine.lean_pool.LeanPool ──
        # LeanPool 提供连接池化、超时保护和原子会话管理。
        # 当 lean_pool 可用时, verify_complete_proof / send_tactic 委托给它,
        # 避免维护两套独立的 REPL 进程管理代码。
        # LeanREPL 仍保留编译缓存作为薄层增值, 以及 subprocess fallback。
        self._lean_pool = lean_pool

    @staticmethod
    def create(project_dir: str = ".", timeout: int = 60,
               lean_pool: 'LeanPool' = None) -> 'LeanREPL':
        """Factory: auto-detect backend and create REPL."""
        return LeanREPL(mode="auto", project_dir=project_dir,
                        timeout=timeout, lean_pool=lean_pool)

    @property
    def is_alive(self) -> bool:
        return self._state.is_alive

    @property
    def backend(self) -> str:
        return self.mode

    # ── Core public API ──

    def start(self, theorem_header: str) -> REPLResponse:
        """Start a new proof session."""
        self._state = REPLState(is_alive=True)
        self._state.goal_stack = [theorem_header]
        logger.info(f"REPL[{self.mode}] session for: {theorem_header[:80]}")

        if self.mode == "lean4-repl":
            return self._lean4repl_start(theorem_header)
        elif self.mode == "pantograph":
            return self._pantograph_start(theorem_header)
        elif self.mode == "subprocess":
            return REPLResponse(success=True, goals=[theorem_header])
        else:
            return REPLResponse(
                success=False,
                error=f"Backend '{self.mode}' not available. "
                      f"Install Lean4 via elan.")

    def send_tactic(self, tactic: str) -> REPLResponse:
        """Send a single tactic. Returns new goal state."""
        if not self._state.is_alive:
            return REPLResponse(success=False, error="REPL not started")

        t0 = time.perf_counter()

        # ── 优先委托给 LeanPool ──
        if self._lean_pool and self._state.env_id is not None:
            pool_result = self._lean_pool.try_tactic(self._state.env_id, tactic)
            if pool_result.success:
                self._state.env_id = pool_result.new_env_id
                self._state.tactic_history.append(tactic)
                resp = REPLResponse(
                    success=True,
                    goals=pool_result.remaining_goals,
                    is_complete=pool_result.is_proof_complete,
                    env_id=pool_result.new_env_id,
                )
            else:
                resp = REPLResponse(
                    success=False,
                    error=pool_result.error_message,
                    goals=pool_result.remaining_goals,
                )
        elif self.mode == "lean4-repl":
            resp = self._lean4repl_tactic(tactic)
        elif self.mode == "subprocess":
            resp = self._subprocess_tactic(tactic)
        else:
            resp = REPLResponse(success=False, error=f"No tactic support for {self.mode}")

        resp.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if resp.success:
            self._state.tactic_history.append(tactic)
        else:
            self._state.error_count += 1
        return resp

    def verify_complete_proof(self, theorem: str, proof: str,
                              preamble: str = "") -> REPLResponse:
        """Verify a complete proof — the primary entry point.

        This is the most common usage pattern: LLM generates a full proof,
        we verify it. Supports caching across calls.

        Backend priority:
          1. Cache hit → instant return
          2. LeanPool (engine layer) → 连接池化, 超时保护, 原子会话管理
          3. lean4-repl (self-managed process) → legacy path
          4. subprocess → 最慢, 每次完整编译
        """
        # 缓存键包含环境版本标识, 防止 Lean4/Mathlib 升级后返回过时结果
        env_tag = self._get_env_version_tag()
        cache_key = hashlib.sha256(
            f"{env_tag}||{preamble}||{theorem}||{proof}".encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        t0 = time.perf_counter()

        # ── 优先委托给 LeanPool (统一后端) ──
        if self._lean_pool:
            pool_result = self._lean_pool.verify_complete(theorem, proof, preamble)
            resp = REPLResponse(
                success=pool_result.success,
                goals=pool_result.goals_remaining,
                error=pool_result.stderr if not pool_result.success else "",
                is_complete=pool_result.success and not pool_result.has_sorry,
                env_id=pool_result.env_id,
            )
        elif self.mode == "lean4-repl":
            resp = self._lean4repl_verify_complete(theorem, proof, preamble)
        elif self.mode == "subprocess":
            resp = self._subprocess_verify_complete(theorem, proof, preamble)
        else:
            resp = self._subprocess_verify_complete(theorem, proof, preamble)

        resp.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        self._cache.put(cache_key, resp)
        return resp

    def undo(self) -> REPLResponse:
        if self._state.tactic_history:
            self._state.tactic_history.pop()
            return REPLResponse(success=True)
        return REPLResponse(success=False, error="Nothing to undo")

    def close(self):
        """Close the REPL session and terminate backend process."""
        self._state.is_alive = False
        if self._process:
            try:
                if self._process.stdin and not self._process.stdin.closed:
                    self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception as _exc:
                    logger.debug(f"Suppressed exception: {_exc}")
            self._process = None

    _env_version_cache: str = ""  # class-level cache

    def _get_env_version_tag(self) -> str:
        """获取 Lean4 + Mathlib 环境版本标识, 用于缓存键.

        读取 lean-toolchain 和 lake-manifest.json (如果存在) 生成版本标签。
        结果缓存在类级别, 避免重复 I/O。
        """
        if LeanREPL._env_version_cache:
            return LeanREPL._env_version_cache

        parts = ["lean"]
        # 读取 lean-toolchain
        toolchain_path = os.path.join(self.project_dir, "lean-toolchain")
        try:
            if os.path.isfile(toolchain_path):
                with open(toolchain_path) as f:
                    parts.append(f.read().strip()[:50])
        except OSError as _exc:
            logger.debug(f"Suppressed exception: {_exc}")

        # 读取 lake-manifest.json 中 mathlib 的 rev
        manifest_path = os.path.join(self.project_dir, "lake-manifest.json")
        try:
            if os.path.isfile(manifest_path):
                import json as _json
                with open(manifest_path) as f:
                    manifest = _json.load(f)
                for pkg in manifest.get("packages", []):
                    if pkg.get("name") == "mathlib":
                        parts.append(pkg.get("rev", "")[:12])
                        break
        except (OSError, ValueError) as _exc:
            logger.debug(f"Suppressed exception: {_exc}")

        tag = "|".join(parts)
        LeanREPL._env_version_cache = tag
        return tag

    def get_history(self) -> list[str]:
        return list(self._state.tactic_history)

    def reset(self):
        header = self._state.goal_stack[0] if self._state.goal_stack else ""
        self._state = REPLState(is_alive=True)
        if header:
            self._state.goal_stack = [header]

    @classmethod
    def cache_stats(cls) -> dict:
        return _global_cache.stats()

    # ═══════════════════════════════════════════════════════════
    # Backend: lean4-repl (long-running process, incremental)
    # https://github.com/leanprover-community/repl
    # Protocol: JSON lines over stdin/stdout
    #   → {"cmd": "...", "env": 0}
    #   ← {"env": 1, "messages": [...], "sorries": [...]}
    # ═══════════════════════════════════════════════════════════

    def _ensure_lean4repl_process(self):
        """Ensure the lean4-repl process is running."""
        if self._process and self._process.poll() is None:
            return

        repl_bin = os.path.join(
            self.project_dir, ".lake", "build", "bin", "repl")
        if os.path.isfile(repl_bin):
            cmd = [repl_bin]
        else:
            cmd = ["lake", "env", "lean", "--run", "Repl"]

        logger.info(f"Starting lean4-repl: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.project_dir,
            text=True,
            bufsize=1,
        )
        # Wait briefly for startup
        time.sleep(0.5)
        if self._process.poll() is not None:
            stderr = ""
            if self._process.stderr:
                stderr = self._process.stderr.read()[:500]
            raise RuntimeError(
                f"lean4-repl failed to start: {stderr}\n"
                f"Setup: cd {self.project_dir} && "
                f"echo 'require Repl from git "
                f"\"https://github.com/leanprover-community/repl\" "
                f"@ \"master\"' >> lakefile.lean && lake update && lake build Repl")

    def _lean4repl_send(self, request: dict) -> dict:
        """Send JSON request, read JSON response."""
        self._ensure_lean4repl_process()
        request_str = json.dumps(request, ensure_ascii=False) + "\n\n"
        logger.debug(f"lean4-repl ← {request_str.strip()[:200]}")

        with self._lock:
            try:
                self._process.stdin.write(request_str)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                logger.warning(f"lean4-repl stdin broken: {e}")
                self._kill_process()
                return {"error": f"lean4-repl pipe error: {e}"}

            # Read response with timeout
            try:
                sel = selectors.DefaultSelector()
                sel.register(self._process.stdout, selectors.EVENT_READ)
                deadline = time.time() + self.timeout
                lines = []
                brace_depth = 0
                started = False

                while time.time() < deadline:
                    remaining = max(0.1, deadline - time.time())
                    events = sel.select(timeout=remaining)
                    if not events:
                        if self._process.poll() is not None:
                            break
                        continue
                    chunk = self._process.stdout.readline()
                    if not chunk:
                        break
                    line = chunk.rstrip()
                    if not line and not started:
                        continue
                    lines.append(line)
                    brace_depth += line.count('{') - line.count('}')
                    if '{' in line:
                        started = True
                    if started and brace_depth <= 0:
                        break

                sel.close()
                full_response = "\n".join(lines).strip()
                if not full_response:
                    return {"error": "Empty response (timeout or process died)"}

                logger.debug(f"lean4-repl → {full_response[:200]}")
                return json.loads(full_response)

            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON: {e}",
                        "raw": "\n".join(lines) if 'lines' in dir() else ""}
            except Exception as e:
                return {"error": str(e)}

    def _lean4repl_start(self, theorem_header: str) -> REPLResponse:
        try:
            self._ensure_lean4repl_process()
            # Establish environment with imports
            resp = self._lean4repl_send({
                "cmd": "import Mathlib\n",
                "env": 0
            })
            env_id = resp.get("env", 0)
            if "error" in resp and isinstance(resp["error"], str):
                # Mathlib not available — try minimal import
                resp = self._lean4repl_send({
                    "cmd": "import Lean\n",
                    "env": 0
                })
                env_id = resp.get("env", 0)

            self._state.env_id = env_id
            return REPLResponse(success=True, goals=[theorem_header],
                                env_id=env_id)
        except Exception as e:
            return REPLResponse(success=False, error=str(e))

    def _lean4repl_tactic(self, tactic: str) -> REPLResponse:
        header = self._state.goal_stack[0] if self._state.goal_stack else ""
        all_tactics = self._state.tactic_history + [tactic]
        full_cmd = header + "\n  " + "\n  ".join(all_tactics)

        resp = self._lean4repl_send({
            "cmd": full_cmd,
            "env": self._state.env_id or 0,
        })
        return self._parse_lean4repl_response(resp)

    def _lean4repl_verify_complete(self, theorem: str, proof: str,
                                    preamble: str = "") -> REPLResponse:
        try:
            self._ensure_lean4repl_process()
        except Exception as e:
            # Fall back to subprocess if REPL can't start
            logger.warning(f"lean4-repl unavailable, falling back to subprocess: {e}")
            return self._subprocess_verify_complete(theorem, proof, preamble)

        cmd = ""
        if preamble:
            cmd += preamble.strip() + "\n\n"
        cmd += f"{theorem} {proof}"

        resp = self._lean4repl_send({"cmd": cmd, "env": 0})
        return self._parse_lean4repl_response(resp)

    def _parse_lean4repl_response(self, resp: dict) -> REPLResponse:
        if "error" in resp and isinstance(resp["error"], str):
            return REPLResponse(success=False, error=resp["error"],
                                raw_output=json.dumps(resp, default=str))

        messages = resp.get("messages", [])
        sorries = resp.get("sorries", [])
        env_id = resp.get("env")

        errors = [m for m in messages if m.get("severity") == "error"]

        if errors:
            error_text = "\n".join(
                m.get("data", m.get("message", str(m))) for m in errors)
            if any("unsolved goals" in str(m) for m in errors):
                goals = self._extract_goals_from_messages(errors)
                return REPLResponse(success=True, goals=goals,
                                    raw_output=json.dumps(resp, default=str),
                                    env_id=env_id)
            return REPLResponse(success=False, error=error_text[:1000],
                                raw_output=json.dumps(resp, default=str),
                                env_id=env_id)

        if sorries:
            return REPLResponse(
                success=False,
                error=f"Proof contains {len(sorries)} sorry(s)",
                raw_output=json.dumps(resp, default=str), env_id=env_id)

        return REPLResponse(success=True, goals=[], is_complete=True,
                            raw_output=json.dumps(resp, default=str),
                            env_id=env_id)

    @staticmethod
    def _extract_goals_from_messages(errors: list) -> list[str]:
        goals = []
        for m in errors:
            data = m.get("data", "")
            for line in data.split("\n"):
                stripped = line.strip()
                if stripped.startswith("⊢"):
                    goals.append(stripped)
        return goals

    # ═══════════════════════════════════════════════════════════
    # Backend: pantograph
    # ═══════════════════════════════════════════════════════════

    def _pantograph_start(self, theorem_header: str) -> REPLResponse:
        try:
            cmd = ["pantograph", "--project", self.project_dir]
            self._process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1)
            time.sleep(1)
            if self._process.poll() is not None:
                stderr = self._process.stderr.read()[:300] if self._process.stderr else ""
                return REPLResponse(success=False,
                                    error=f"pantograph failed: {stderr}")
            return REPLResponse(success=True, goals=[theorem_header])
        except FileNotFoundError:
            return REPLResponse(success=False,
                                error="pantograph not found. Install: pip install pantograph")

    # ═══════════════════════════════════════════════════════════
    # Backend: subprocess (fallback — full recompilation each time)
    # ═══════════════════════════════════════════════════════════

    def _subprocess_tactic(self, tactic: str) -> REPLResponse:
        header = self._state.goal_stack[0] if self._state.goal_stack else ""
        all_tactics = self._state.tactic_history + [tactic]
        full_code = f"import Mathlib\n\n{header}\n  " + "\n  ".join(all_tactics) + "\n"

        cache_key = hashlib.sha256(full_code.encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        resp = self._run_lean_subprocess(full_code)
        self._cache.put(cache_key, resp)
        return resp

    def _subprocess_verify_complete(self, theorem: str, proof: str,
                                     preamble: str = "") -> REPLResponse:
        from prover.codegen.import_resolver import assemble_lean_file
        full_code = assemble_lean_file(theorem, proof, preamble)
        return self._run_lean_subprocess(full_code)

    def _run_lean_subprocess(self, full_code: str) -> REPLResponse:
        try:
            if (shutil.which("lake") and
                    os.path.isfile(os.path.join(self.project_dir, "lakefile.lean"))):
                cmd = ["lake", "env", "lean", "--stdin"]
            elif shutil.which("lean"):
                cmd = [self.lean_cmd, "--stdin"]
            else:
                return REPLResponse(
                    success=False,
                    error="Neither lean nor lake found. Install via elan.")

            result = subprocess.run(
                cmd, input=full_code, capture_output=True, text=True,
                timeout=self.timeout, cwd=self.project_dir)

            raw = (result.stderr or "") + (result.stdout or "")

            if result.returncode == 0:
                # Double-check: Lean4 can return 0 but still have errors in stderr
                has_error = any(
                    marker in raw.lower()
                    for marker in ["error:", "unknown identifier", "type mismatch"]
                )
                if not has_error:
                    return REPLResponse(success=True, goals=[], raw_output=raw,
                                        is_complete=True)

            # Parse goals from unsolved-goals error
            if "unsolved goals" in raw:
                goals = []
                for line in raw.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("⊢"):
                        goals.append(stripped)
                return REPLResponse(success=True, goals=goals, raw_output=raw)

            return REPLResponse(success=False, error=raw[:800], raw_output=raw)

        except subprocess.TimeoutExpired:
            return REPLResponse(success=False,
                                error="Lean4 compilation timed out")
        except FileNotFoundError:
            return REPLResponse(
                success=False,
                error="Lean4 not found. Install: "
                      "curl -sSf https://raw.githubusercontent.com/"
                      "leanprover/elan/master/elan-init.sh | sh")

    # ── Helpers ──

    def _kill_process(self):
        if self._process:
            try:
                self._process.kill()
            except Exception as _exc:
                logger.debug(f"Suppressed exception: {_exc}")
            self._process = None
