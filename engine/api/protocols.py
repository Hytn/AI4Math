"""engine/api/protocols.py — Engine 层公共接口协议

定义 engine 层对外暴露的类型化接口, 使 prover/agent 层
可以通过协议 (Protocol) 依赖 engine 层, 而非具体类。

这些协议是 engine 层的"合同":
  - 只包含 engine 层应该暴露的能力
  - 不包含 AgentFeedback、repair_candidates 等 agent 层关注点
  - 方便测试时 mock engine 层

Usage::

    from engine.api.protocols import VerifierProtocol, PoolProtocol

    class MyOrchestrator:
        def __init__(self, verifier: VerifierProtocol, pool: PoolProtocol):
            ...
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable

from engine._core import TacticFeedback, FullVerifyResult


@runtime_checkable
class PoolProtocol(Protocol):
    """Lean4 REPL 连接池的最小接口

    同步和异步 Pool 都应满足此协议 (通过 SyncLeanPool 包装)。
    """

    def try_tactic(self, env_id: int, tactic: str) -> TacticFeedback: ...

    def try_tactics_parallel(self, env_id: int,
                             tactics: list[str]) -> list[TacticFeedback]: ...

    def verify_complete(self, theorem: str, proof: str,
                        preamble: str = "") -> FullVerifyResult: ...


    def stats(self) -> dict: ...

    def shutdown(self): ...

    @property
    def base_env_id(self) -> int: ...


@runtime_checkable
class VerifierProtocol(Protocol):
    """验证调度器的最小接口

    Orchestrator/HeterogeneousEngine 应通过此协议引用验证器,
    而非直接依赖 VerificationScheduler 类。
    """

    def verify_complete(self, theorem: str, proof: str,
                        direction: str = "",
                        require_l2: bool = False): ...

    def verify_tactic(self, env_id: int, tactic: str,
                      goals_before: int = 1,
                      direction: str = ""): ...

    def stats(self) -> dict: ...


@runtime_checkable
class BroadcastProtocol(Protocol):
    """广播总线的最小接口"""

    def subscribe(self, subscriber_id: str, filter_types=None): ...

    def unsubscribe(self, subscriber_id: str): ...

    def publish(self, msg) -> None: ...

    def render_for_prompt(self, subscriber_id: str,
                          max_messages: int = 10,
                          max_chars: int = 2000) -> str: ...

    def get_recent(self, n: int = 10, msg_type=None) -> list: ...

    def clear(self) -> None: ...
