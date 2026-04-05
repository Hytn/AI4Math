"""agent/hooks/hook_types.py — 证明生命周期事件与钩子协议

在证明流程的 9 个关键时机插入可声明、可组合的检查规则。
替代 Orchestrator 中散落的硬编码 if 判断。
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class HookEvent(str, Enum):
    """证明生命周期中可以插入钩子的 9 个事件点"""

    # 问题级别
    ON_PROBLEM_START = "on_problem_start"
    ON_PROBLEM_END = "on_problem_end"

    # 生成阶段
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"

    # 验证阶段
    PRE_VERIFICATION = "pre_verification"
    POST_VERIFICATION = "post_verification"

    # 修复阶段
    PRE_REPAIR = "pre_repair"

    # 策略阶段
    ON_STRATEGY_SWITCH = "on_strategy_switch"
    ON_ROUND_END = "on_round_end"


class HookAction(str, Enum):
    """钩子返回的动作指令"""
    CONTINUE = "continue"
    MODIFY = "modify"
    SKIP = "skip"
    ABORT = "abort"
    ESCALATE = "escalate"


@dataclass
class HookContext:
    """传递给钩子的上下文信息"""
    event: HookEvent
    theorem_statement: str = ""
    proof: Optional[str] = None
    errors: Optional[list] = None
    attempt_count: int = 0
    dominant_error: str = ""
    strategy_name: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class HookResult:
    """钩子的执行结果"""
    action: HookAction = HookAction.CONTINUE
    modified_proof: Optional[str] = None
    message: str = ""
    inject_context: dict = field(default_factory=dict)


class Hook:
    """钩子基类 — 所有自定义钩子继承此类"""
    name: str = "base_hook"

    def execute(self, ctx: HookContext) -> HookResult:
        return HookResult()
