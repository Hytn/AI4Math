"""prover/verifier/lean_repl.py — 同步 Lean REPL 包装 (重建)

历史: 本模块在上传版本中不存在, 但 ``run_mcts_eval.py``(line 37) 仍
import ``LeanREPL`` / ``REPLResponse`` —— 与 ``agent.brain.
claude_provider``、``prover.codegen`` 同批的 MCTS 入口 import 即崩
既有缺陷 (run_unified / run_eval 主链路不受影响)。

重建策略: **不重写 REPL 协议**。唯一维护的 Lean 通信实现是
``engine.transport.LocalTransport`` (异步, 心跳/重启/fallback 齐全);
本模块用私有事件循环线程把它包成同步接口, 协议层永远单一事实源。

契约 (由 run_mcts_eval 的调用面反推, 见 tests/test_lean_repl_sync.py):

    repl = LeanREPL.create(project_dir="data/miniF2F", timeout=120)
    repl.backend                      # "local-repl" / "fallback"
    resp = repl.check_tactic_sequence(theorem_header, ["intro n", ...],
                                      preamble="import Mathlib")
    resp.success / resp.is_complete / resp.goals / resp.error / resp.raw_output
    repl.close()

语义: 把 ``theorem_header := by <tactics...>`` 喂给 REPL —
  - 全部 goal 关闭且无错误            → success=True,  is_complete=True
  - 编译通过但有剩余 goal (追加 sorry) → success=True,  is_complete=False,
                                         goals=剩余 proof state
  - 编译失败                          → success=False, error=首条错误
fallback 模式 (无 Lean 工具链) 下所有调用诚实返回 success=False +
error="lean repl unavailable (fallback)" —— 不伪造成功。
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class REPLResponse:
    success: bool = False           # 代码 (允许 sorry) 是否编译通过
    is_complete: bool = False       # 是否零剩余 goal 且无 sorry
    goals: list[str] = field(default_factory=list)
    error: str = ""
    raw_output: str = ""


def parse_repl_output(resp: dict | None,
                      had_trailing_sorry: bool) -> REPLResponse:
    """把 Lean REPL 的 json 响应归一成 REPLResponse (纯函数, 单测覆盖)。"""
    if not isinstance(resp, dict):
        return REPLResponse(success=False, error="no response from REPL")
    raw = json.dumps(resp, ensure_ascii=False)
    messages = resp.get("messages") or []
    errors = [str(m.get("data") or m.get("message") or "")
              for m in messages
              if isinstance(m, dict) and m.get("severity") == "error"]
    sorries = resp.get("sorries") or []
    goals = [str(s.get("goal", "")) for s in sorries
             if isinstance(s, dict) and s.get("goal")]
    if errors:
        return REPLResponse(success=False, error=errors[0][:2000],
                            goals=goals, raw_output=raw)
    if sorries:
        # 编译过但有 sorry: 剩余 goal 即 proof state
        return REPLResponse(success=True, is_complete=False,
                            goals=goals, raw_output=raw)
    # 无错误无 sorry: 若我们主动加了尾部 sorry 却没报 sorries,
    # 视为异常输出; 否则证明完成。
    return REPLResponse(success=True,
                        is_complete=not had_trailing_sorry,
                        goals=[], raw_output=raw)


class LeanREPL:
    """同步外观; 内部私有事件循环线程驱动 LocalTransport。"""

    def __init__(self, transport, loop: asyncio.AbstractEventLoop,
                 thread: threading.Thread, timeout: float):
        self._transport = transport
        self._loop = loop
        self._thread = thread
        self._timeout = timeout
        self._env_cache: dict[str, int] = {}

    # ── 构造/销毁 ────────────────────────────────────────────────

    @classmethod
    def create(cls, project_dir: str = "", timeout: float = 120.0
               ) -> "LeanREPL":
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever,
                                  name="lean-repl-sync", daemon=True)
        thread.start()

        from engine.transport import LocalTransport
        kwargs = {}
        if project_dir:
            kwargs["project_dir"] = project_dir
        try:
            transport = LocalTransport(**kwargs)
        except TypeError:
            transport = LocalTransport()
        fut = asyncio.run_coroutine_threadsafe(transport.start(), loop)
        try:
            fut.result(timeout=timeout)
        except Exception as e:  # noqa: BLE001
            logger.warning("LeanREPL: transport start failed: %s", e)
        return cls(transport, loop, thread, timeout)

    @property
    def backend(self) -> str:
        if getattr(self._transport, "is_fallback", True):
            return "fallback"
        return "local-repl"

    def close(self):
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._transport.close(), self._loop)
            fut.result(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    # ── 核心 API ─────────────────────────────────────────────────

    def check_tactic_sequence(self, theorem_header: str,
                              tactics: list[str],
                              preamble: str = "import Mathlib"
                              ) -> REPLResponse:
        if self.backend == "fallback":
            return REPLResponse(
                success=False,
                error="lean repl unavailable (fallback)")
        env_id = self._env_for(preamble)
        head = theorem_header.strip()
        if head.endswith(":= by sorry"):
            head = head[: -len("sorry")].rstrip()
        elif not head.rstrip().endswith("by"):
            head = head + " := by"
        body = "\n".join(f"  {t.strip()}" for t in tactics if t.strip())
        # 先试"完整证明" (无 sorry)
        code = f"{head}\n{body}" if body else head.rstrip(" by:=") + " := by skip"
        if tactics:
            r = self._send({"cmd": f"{head}\n{body}",
                            **({"env": env_id} if env_id is not None else {})})
            parsed = parse_repl_output(r, had_trailing_sorry=False)
            if parsed.success and parsed.is_complete:
                return parsed
        # 再试 tactic 前缀 + 尾部 sorry, 读剩余 proof state
        code = f"{head}\n{body}\n  sorry" if body else f"{head}\n  sorry"
        r = self._send({"cmd": code,
                        **({"env": env_id} if env_id is not None else {})})
        return parse_repl_output(r, had_trailing_sorry=True)

    # ── 内部 ─────────────────────────────────────────────────────

    def _env_for(self, preamble: str) -> int | None:
        key = (preamble or "").strip()
        if not key:
            return None
        if key in self._env_cache:
            return self._env_cache[key]
        r = self._send({"cmd": key})
        env = r.get("env") if isinstance(r, dict) else None
        if isinstance(env, int):
            self._env_cache[key] = env
            return env
        logger.warning("LeanREPL: preamble env creation failed; "
                       "falling back to inline preamble per call")
        return None

    def _send(self, cmd: dict) -> dict | None:
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._transport.send(cmd), self._loop)
            return fut.result(timeout=self._timeout)
        except Exception as e:  # noqa: BLE001
            logger.warning("LeanREPL send failed: %s", e)
            return None
