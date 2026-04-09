"""engine/_core.py — Shared pure functions for sync/async Lean4 backends.

All IO-free logic (error classification, code assembly, type extraction,
compile cache, helper utilities) lives here. Both LeanPool (sync) and
AsyncLeanPool (async) import from this module instead of duplicating code.
"""
from __future__ import annotations
import hashlib
import re
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Compile cache (thread-safe LRU, shared singleton)
# ═══════════════════════════════════════════════════════════════

class CompileCache:
    """Thread-safe LRU cache for compilation results.

    Key = sha256(env_version || preamble || theorem || proof).
    Uses OrderedDict for O(1) move_to_end / popitem.
    """

    def __init__(self, maxsize: int = 512):
        self._cache: OrderedDict[str, FullVerifyResult] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[FullVerifyResult]:
        with self._lock:
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, key: str, result: FullVerifyResult):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = result
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits, "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0,
            "size": len(self._cache),
        }


class AsyncCompileCache:
    """Asyncio-safe LRU cache for compilation results (Fix #7).

    Drop-in replacement for CompileCache in async contexts.
    Uses asyncio.Lock instead of threading.Lock to avoid blocking
    the event loop under high concurrency.

    Usage::

        cache = AsyncCompileCache(maxsize=512)
        result = await cache.get(key)
        if result is None:
            result = await do_verify(...)
            await cache.put(key, result)
    """

    def __init__(self, maxsize: int = 512):
        self._cache: OrderedDict[str, 'FullVerifyResult'] = OrderedDict()
        self._maxsize = maxsize
        self._lock = None  # lazy init (needs running event loop)
        self._init_guard = threading.Lock()  # guards lazy lock creation
        self.hits = 0
        self.misses = 0

    def _ensure_lock(self):
        if self._lock is None:
            with self._init_guard:
                if self._lock is None:  # double-checked locking
                    import asyncio
                    self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional['FullVerifyResult']:
        self._ensure_lock()
        async with self._lock:
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self.misses += 1
            return None

    async def put(self, key: str, result: 'FullVerifyResult'):
        self._ensure_lock()
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = result
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits, "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0,
            "size": len(self._cache),
        }


# ═══════════════════════════════════════════════════════════════
# Shared data types
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Pure helper functions
# ═══════════════════════════════════════════════════════════════

def which(cmd: str) -> Optional[str]:
    """跨平台的 which 实现"""
    return shutil.which(cmd)


def assemble_code(theorem: str, proof: str, preamble: str = "") -> str:
    """组装完整的 Lean4 源文件

    处理以下情况:
      1. theorem 已含完整证明 (含 :=)  → 忽略 proof 参数
      2. theorem 无证明, proof 以 := 开头  → 直接拼接
      3. theorem 无证明, proof 以 by 开头  → 插入 := 再拼接
      4. theorem 无证明, proof 是纯 tactic → 插入 := by 再拼接
      5. theorem 或 proof 为空            → 安全处理
    """
    parts = []
    if preamble:
        parts.append(preamble)
    else:
        parts.append("import Mathlib")
    parts.append("")

    thm = theorem.strip()
    prf = proof.strip()

    if not thm:
        if prf:
            parts.append(prf)
        return "\n".join(parts)

    # 如果 theorem 已经包含 := (含证明体), 不追加 proof
    if ":=" in thm:
        parts.append(thm)
    elif prf:
        # proof 自带 :=
        if prf.startswith(":="):
            parts.append(f"{thm} {prf}")
        # proof 以 by 开头 (tactic block)
        elif prf.startswith("by"):
            parts.append(f"{thm} := {prf}")
        # proof 是裸 tactic 或 term — 假定 tactic mode
        else:
            parts.append(f"{thm} := by\n  {prf}")
    else:
        parts.append(thm)

    return "\n".join(parts)


def classify_error(msg: str) -> str:
    """将 Lean4 错误消息分类, 按优先级匹配。"""
    msg_lower = msg.lower()

    # 高优先级: 精确匹配 (更具体的在前)
    if "application type mismatch" in msg_lower:
        return "app_type_mismatch"
    if "type mismatch" in msg_lower:
        return "type_mismatch"
    if "unknown identifier" in msg_lower or "unknown constant" in msg_lower:
        return "unknown_identifier"
    if "unsolved goals" in msg_lower:
        return "unsolved_goals"
    if "declaration uses 'sorry'" in msg_lower:
        return "sorry"

    # tactic 失败
    if "tactic" in msg_lower and "failed" in msg_lower:
        return "tactic_failed"

    # Lean4 特有错误类型
    if "universe level" in msg_lower:
        return "universe_error"
    if "failed to synthesize" in msg_lower:
        return "instance_not_found"
    if "function expected" in msg_lower:
        return "function_expected"
    if "maximum recursion depth" in msg_lower or "deep recursion" in msg_lower:
        return "recursion_limit"
    if "deterministic timeout" in msg_lower or "heartbeat" in msg_lower:
        return "timeout"
    if "timeout" in msg_lower:
        return "timeout"
    if ("elaboration" in msg_lower or "expected type" in msg_lower) and "error" in msg_lower:
        return "elaboration_error"
    if "ambiguous" in msg_lower:
        return "ambiguous"

    # 低优先级
    if "syntax" in msg_lower or "expected" in msg_lower:
        return "syntax_error"

    return "other"


def classify_error_structured(messages: list[dict]) -> tuple[str, str, dict]:
    """从 REPL 的结构化 JSON messages 中提取分类信息。

    Returns:
        (primary_category, combined_message, metadata)
    """
    errors = [m for m in messages if m.get("severity") == "error"]
    if not errors:
        return ("none", "", {})

    primary = errors[0]
    primary_msg = primary.get("data", "")
    category = classify_error(primary_msg)

    all_msgs = [e.get("data", "") for e in errors]
    combined = "\n".join(all_msgs[:5])

    metadata = {
        "error_count": len(errors),
        "primary_pos": primary.get("pos"),
        "primary_end_pos": primary.get("endPos"),
        "all_categories": list({classify_error(m.get("data", "")) for m in errors}),
    }

    return (category, combined, metadata)


def extract_expected(msg: str) -> str:
    """从错误消息中提取 expected type"""
    for marker in ["expected to have type", "expected type"]:
        if marker in msg:
            idx = msg.index(marker) + len(marker)
            rest = msg[idx:].strip()
            end = len(rest)
            for stop in ["\n", "but is expected", "has type"]:
                pos = rest.find(stop)
                if pos > 0:
                    end = min(end, pos)
            return rest[:end].strip()
    return ""


def extract_actual(msg: str) -> str:
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


def make_cache_key(theorem: str, proof: str, preamble: str = "",
                   env_fingerprint: str = "") -> str:
    """Generate a deterministic cache key for a verification request.

    Args:
        env_fingerprint: Opaque string capturing environment state
            (e.g. injected lemma count, Mathlib version). When the
            environment changes, cached results are automatically
            invalidated.
    """
    return hashlib.sha256(
        f"{env_fingerprint}||{preamble}||{theorem}||{proof}".encode()
    ).hexdigest()
