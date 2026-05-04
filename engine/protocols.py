"""engine/protocols.py — Engine 层对外接口协议

定义 engine 层暴露给上层 (agent/, prover/, sampler/) 的类型化协议,
使上层依赖协议而不是具体类, 方便 mock 和替换实现。

v13: 精简到实际被引用的 Protocol。``PoolProtocol`` (sync 版)、
``VerifierProtocol``、``BroadcastProtocol`` 在 v12 时定义但 0 处用作
类型注解, 已删除。如果将来需要 sync pool 抽象或 verifier 抽象, 在
此文件按相同模式补回即可。
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable

from engine._core import TacticFeedback, FullVerifyResult


@runtime_checkable
class AsyncPoolProtocol(Protocol):
    """异步 Lean 4 REPL 连接池协议 — AsyncLeanPool 满足此协议

    主路径 (UnifiedProofRunner / sampler.ProofEnv) 通过此协议引用池,
    而非直接依赖 ``AsyncLeanPool`` 类。这让 mock pool 和未来的远程
    pool 实现都能透明替换。
    """

    async def try_tactic(self, env_id: int, tactic: str) -> TacticFeedback: ...

    async def try_tactics_parallel(self, env_id: int,
                                   tactics: list[str]) -> list[TacticFeedback]: ...

    async def verify_complete(self, theorem: str, proof: str,
                              preamble: str = "") -> FullVerifyResult: ...

    async def share_lemma(self, lemma_code: str, **kwargs) -> int: ...

    async def shutdown(self): ...

    def stats(self) -> dict: ...

    @property
    def base_env_id(self) -> int: ...
