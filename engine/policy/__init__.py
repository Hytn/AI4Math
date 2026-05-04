"""engine.policy — 可执行策略规则引擎 (v14 回归)

把 ``agent_loop`` 里硬编码的"何时升级 / 何时切角色 / 何时放弃"逻辑挪到
此模块, 用 declarative ``PolicyRule`` 表达。每条规则可单独开关、单独
测试; agent_loop 只调用 ``PolicyEngine.evaluate`` 拿决策。

模块组成:
  task_state.py    ProofTaskStateMachine + ProofFailureClass + TaskEvent
                   (轻量状态机, 记录"连续 N 次同类错误"等历史)
  recovery.py      RecoveryRecipe + RecoveryRegistry
                   (每种 ProofFailureClass 的自动恢复方案: REPL crash→restart,
                    timeout→reduce timeout, ...)
  engine.py        PolicyEngine + 5 条内置规则:
                     - ConsecutiveSameErrorRule (同错误 N 次→升级)
                     - BudgetEscalationRule (预算耗尽→升级温度/换 profile)
                     - BankedLemmaDecomposeRule (lemma 银行有相关→走 decompose)
                     - ReflectionRule (失败 N 次→插入反思 prompt)
                     - InfraRecoveryRule (基础设施级错误→走 recovery recipe)

设计原则:
  - 规则是纯函数 (state + events → PolicyAction), 易测易组合
  - PolicyEngine 是 stateless, 每次 evaluate 读 task_sm 的快照
  - 不依赖 LLM (规则本身是 doctrine, LLM 由 agent_loop 调)

Usage::

    from engine.policy import PolicyEngine, ProofTaskStateMachine
    engine = PolicyEngine.with_default_rules()
    sm = ProofTaskStateMachine(task_id="problem_001")
    # ... agent_loop 在每轮把 verify 结果塞进 sm ...
    decision = engine.evaluate(sm)
    if decision.action == PolicyAction.SWITCH_ROLE:
        # agent_loop 切 sub-profile
        ...
"""
from engine.policy.task_state import (
    TaskStatus,
    ProofFailureClass,
    TaskFailure,
    TaskEvent,
    TaskContext,
    ProofTaskStateMachine,
)
from engine.policy.recovery import (
    RecoveryAction,
    RecoveryRecipe,
    RecoveryRegistry,
)
from engine.policy.engine import (
    PolicyAction,
    PolicyDecision,
    PolicyRule,
    PolicyEngine,
    # 5 内置规则
    ConsecutiveSameErrorRule,
    BudgetEscalationRule,
    BankedLemmaDecomposeRule,
    ReflectionRule,
    InfraRecoveryRule,
)

__all__ = [
    "TaskStatus", "ProofFailureClass", "TaskFailure", "TaskEvent",
    "TaskContext", "ProofTaskStateMachine",
    "RecoveryAction", "RecoveryRecipe", "RecoveryRegistry",
    "PolicyAction", "PolicyDecision", "PolicyRule", "PolicyEngine",
    "ConsecutiveSameErrorRule", "BudgetEscalationRule",
    "BankedLemmaDecomposeRule", "ReflectionRule", "InfraRecoveryRule",
]
