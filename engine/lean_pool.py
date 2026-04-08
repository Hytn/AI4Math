"""engine/lean_pool.py — 向后兼容 shim

Phase A 统一双栈后, 所有实现均在 async_lean_pool.py 中。
本文件仅保留向后兼容的名称导出, 新代码应直接导入:

    from engine.async_lean_pool import AsyncLeanPool, SyncLeanPool
    from engine._core import TacticFeedback, FullVerifyResult

保留的导出 (向后兼容):
    LeanPool          → SyncLeanPool (AsyncLeanPool 的同步包装)
    LeanSession       → AsyncLeanSession
    TacticFeedback    → engine._core.TacticFeedback
    FullVerifyResult  → engine._core.FullVerifyResult
    _CompileCache     → engine._core.CompileCache
    _SessionState     → 保留 dataclass (测试中引用)
"""
from __future__ import annotations
import subprocess
from dataclasses import dataclass, field
from typing import Optional

# ── 数据类型 (从 _core 统一导入, 保持旧 import 路径可用) ──
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

# ── _SessionState: 保留用于测试兼容 ──

@dataclass
class _SessionState:
    """单个 REPL 会话的内部状态 (保留用于向后兼容)"""
    process: Optional[subprocess.Popen] = None
    session_id: int = 0
    current_env_id: int = 0
    busy: bool = False
    alive: bool = False
    fallback_mode: bool = False
    is_overflow: bool = False
    last_heartbeat: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    project_dir: str = "."


# ── 类导出: 统一指向 async_lean_pool 的实现 ──
from engine.async_lean_pool import SyncLeanPool as LeanPool  # noqa: F401, E402
from engine.async_lean_pool import AsyncLeanSession as LeanSession  # noqa: F401, E402
