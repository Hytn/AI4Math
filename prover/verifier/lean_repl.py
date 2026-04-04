"""prover/verifier/lean_repl.py — Lean4 REPL 交互层

管理与 Lean4 进程的 tactic-by-tactic 交互。
支持本地进程和 Docker 两种模式。

Includes a compilation cache: identical code strings skip subprocess
execution and return the cached result.
"""
from __future__ import annotations
import hashlib
import json
import logging
import subprocess
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class _CompileCache:
    """LRU cache for Lean compilation results.

    Avoids re-compiling identical code strings (common when the REPL
    recompiles the full tactic history after each step).
    Thread-safe.
    """

    def __init__(self, maxsize: int = 256):
        self._cache: OrderedDict[str, REPLResponse] = OrderedDict()
        self._maxsize = maxsize
        self._lock = __import__('threading').Lock()
        self.hits = 0
        self.misses = 0

    def get(self, code: str) -> Optional['REPLResponse']:
        key = hashlib.sha256(code.encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, code: str, response: 'REPLResponse'):
        key = hashlib.sha256(code.encode()).hexdigest()
        with self._lock:
            self._cache[key] = response
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)


@dataclass
class REPLState:
    """Tracks the state of a Lean REPL session."""
    session_id: str = ""
    is_alive: bool = False
    goal_stack: list[str] = field(default_factory=list)
    tactic_history: list[str] = field(default_factory=list)
    error_count: int = 0


@dataclass
class REPLResponse:
    """Response from a single REPL interaction."""
    success: bool
    goals: list[str] = field(default_factory=list)
    error: str = ""
    raw_output: str = ""
    is_complete: bool = False


class LeanREPL:
    """Interface for interacting with Lean4 REPL / language server.

    Usage:
        repl = LeanREPL(mode="subprocess")
        repl.start("theorem t (n : Nat) : n = n := by")
        resp = repl.send_tactic("rfl")
        if resp.is_complete:
            print("Proof complete!")
        repl.close()
    """

    def __init__(self, mode: str = "subprocess", lean_cmd: str = "lean",
                 project_dir: str = ".", timeout: int = 60):
        self.mode = mode
        self.lean_cmd = lean_cmd
        self.project_dir = project_dir
        self.timeout = timeout
        self._state = REPLState()
        self._process: Optional[subprocess.Popen] = None
        self._cache = _CompileCache(maxsize=256)

    @property
    def is_alive(self) -> bool:
        return self._state.is_alive

    def start(self, theorem_header: str) -> REPLResponse:
        """Start a new proof session with the given theorem header."""
        self._state = REPLState(is_alive=True)
        # In production: spawn lean4 REPL process
        # For now: track state internally for the tactic-by-tactic interface
        self._state.goal_stack = [theorem_header]
        logger.info(f"REPL started for: {theorem_header[:80]}")
        return REPLResponse(success=True, goals=[theorem_header])

    def send_tactic(self, tactic: str) -> REPLResponse:
        """Send a single tactic and get the resulting goal state."""
        if not self._state.is_alive:
            return REPLResponse(success=False, error="REPL not started")

        self._state.tactic_history.append(tactic)

        if self.mode == "subprocess":
            return self._send_subprocess(tactic)
        return REPLResponse(success=False, error=f"Unknown mode: {self.mode}")

    def _send_subprocess(self, tactic: str) -> REPLResponse:
        """Execute tactic via subprocess compilation (with caching)."""
        # Build complete proof so far
        tactics_so_far = "\n  ".join(self._state.tactic_history)
        header = self._state.goal_stack[0] if self._state.goal_stack else ""
        full_code = f"import Mathlib\n\n{header}\n  {tactics_so_far}\n"

        # Check cache first — avoids recompiling identical code
        cached = self._cache.get(full_code)
        if cached is not None:
            logger.debug("REPL cache hit (saved a compilation)")
            return cached

        try:
            # Use `lake env lean --stdin` for project-aware compilation
            # This respects lakefile.lean dependencies including Mathlib
            result = subprocess.run(
                ["lake", "env", "lean", "--stdin"],
                input=full_code, capture_output=True, text=True,
                timeout=self.timeout, cwd=self.project_dir)

            raw = result.stderr + result.stdout

            if result.returncode == 0 and "error" not in raw.lower():
                resp = REPLResponse(success=True, goals=[],
                                     raw_output=raw, is_complete=True)
                self._cache.put(full_code, resp)
                return resp

            # Parse remaining goals from error output
            from prover.verifier.goal_extractor import extract_goals
            goals = extract_goals(raw)
            goal_strs = [g.to_string() for g in goals]

            if "unsolved goals" in raw:
                resp = REPLResponse(success=True, goals=goal_strs,
                                     raw_output=raw)
                self._cache.put(full_code, resp)
                return resp

            # Other error
            self._state.error_count += 1
            resp = REPLResponse(success=False, error=raw[:500],
                                 raw_output=raw)
            self._cache.put(full_code, resp)
            return resp

        except subprocess.TimeoutExpired:
            self._state.error_count += 1
            return REPLResponse(success=False, error="Tactic timed out")
        except FileNotFoundError:
            return REPLResponse(success=False,
                                 error="Lean4 not found. Install via elan.")

    def undo(self) -> REPLResponse:
        """Undo the last tactic."""
        if self._state.tactic_history:
            self._state.tactic_history.pop()
            return REPLResponse(success=True)
        return REPLResponse(success=False, error="Nothing to undo")

    def close(self):
        """Close the REPL session."""
        self._state.is_alive = False
        if self._process:
            self._process.terminate()
            self._process = None

    def get_history(self) -> list[str]:
        return list(self._state.tactic_history)

    def reset(self):
        """Reset to initial state (keeping the theorem)."""
        header = self._state.goal_stack[0] if self._state.goal_stack else ""
        self._state = REPLState(is_alive=True)
        if header:
            self._state.goal_stack = [header]
