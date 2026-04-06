"""agent/runtime/sub_agent.py — 子智能体: 独立上下文 + 独立角色 + 独立模型选择

每个 SubAgent 拥有自己的 ContextWindow 和 WorkingMemory，
避免并行智能体之间的上下文污染。

用法::

    spec = AgentSpec(name="induction_expert", role=AgentRole.PROOF_GENERATOR,
                     model="claude-sonnet-4-20250514", temperature=0.7)
    agent = SubAgent(spec, llm_factory, tool_registry)
    result = agent.execute(task)
"""
from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

from agent.brain.llm_provider import LLMProvider, LLMResponse
from agent.brain.roles import AgentRole, ROLE_PROMPTS
from agent.brain.response_parser import extract_lean_code
from agent.context.context_window import ContextWindow
from agent.memory.working_memory import WorkingMemory

logger = logging.getLogger(__name__)


@dataclass
class AgentSpec:
    """声明式智能体规格 — 定义一个子智能体的全部配置"""
    name: str
    role: AgentRole
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: list[str] = field(default_factory=list)
    context_budget: int = 50_000
    timeout_seconds: int = 120
    system_prompt_override: str = ""
    few_shot_override: str = ""


@dataclass
class ContextItem:
    """注入到子智能体上下文中的单条信息"""
    key: str
    content: str
    priority: float = 0.5
    category: str = "general"


@dataclass
class AgentTask:
    """子智能体的任务描述"""
    description: str
    injected_context: list[ContextItem] = field(default_factory=list)
    theorem_statement: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    """子智能体的执行结果"""
    agent_name: str
    role: AgentRole
    content: str
    proof_code: str = ""
    tool_calls: list = field(default_factory=list)
    tokens_used: int = 0
    latency_ms: int = 0
    confidence: float = 0.5
    success: bool = False
    error: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """True if this result represents an execution failure (not a proof failure)."""
        return bool(self.error)


class SubAgent:
    """独立的子智能体实例

    每个 SubAgent 拥有:
    - 独立的 ContextWindow (不与其他智能体共享)
    - 独立的 WorkingMemory (只记录自己的尝试)
    - 独立的 LLM 调用参数 (model, temperature)
    - 可选的工具权限白名单
    """

    def __init__(self, spec: AgentSpec, llm: LLMProvider,
                 tool_registry=None):
        self.spec = spec
        self.llm = llm
        self.context = ContextWindow(max_tokens=spec.context_budget)
        self.memory = WorkingMemory()
        self.tool_registry = tool_registry

    def execute(self, task: AgentTask) -> AgentResult:
        """在隔离的上下文中执行任务"""
        start = time.time()

        # 1. 构建隔离的上下文 — 只注入与本任务相关的信息
        self.context.add_entry("task", task.description, priority=1.0,
                               category="theorem_statement",
                               is_compressible=False)

        for ctx_item in task.injected_context:
            self.context.add_entry(
                ctx_item.key, ctx_item.content,
                priority=ctx_item.priority,
                category=ctx_item.category)

        # 2. 构建 prompt
        system = self.spec.system_prompt_override or ROLE_PROMPTS.get(
            self.spec.role, ROLE_PROMPTS[AgentRole.PROOF_GENERATOR])

        user_prompt = self.context.render()

        if self.spec.few_shot_override:
            user_prompt += f"\n\n{self.spec.few_shot_override}"

        # 3. 调用 LLM
        try:
            tools_schema = None
            if self.tool_registry and self.spec.tools:
                tools_schema = self.tool_registry.to_claude_tools_schema()

            resp = self.llm.generate(
                system=system,
                user=user_prompt,
                temperature=self.spec.temperature,
                tools=tools_schema,
                max_tokens=self.spec.max_tokens,
            )

            proof_code = extract_lean_code(resp.content)
            latency = int((time.time() - start) * 1000)

            return AgentResult(
                agent_name=self.spec.name,
                role=self.spec.role,
                content=resp.content,
                proof_code=proof_code,
                tool_calls=resp.tool_calls or [],
                tokens_used=resp.tokens_in + resp.tokens_out,
                latency_ms=latency,
                confidence=self._estimate_confidence(resp, proof_code),
                success=False,  # success 仅在验证通过后由 Orchestrator 设置
                metadata=task.metadata,
            )

        except Exception as e:
            latency = int((time.time() - start) * 1000)
            logger.error(f"SubAgent '{self.spec.name}' failed: {e}")
            return AgentResult(
                agent_name=self.spec.name,
                role=self.spec.role,
                content="",
                error=str(e),
                latency_ms=latency,
                tokens_used=0,
                confidence=0.0,
            )

    def _estimate_confidence(self, resp: LLMResponse,
                             proof_code: str) -> float:
        """估计本次生成的质量 (0.0-1.0)

        这是生成阶段的初步估计, 基于代码结构特征。
        验证完成后应调用 refine_confidence() 用实际反馈更新。
        """
        if not proof_code.strip():
            return 0.0

        score = 0.3  # 有代码就有基础分

        # sorry/admit 检测 — 强烈负面信号
        if "sorry" in proof_code or "admit" in proof_code:
            score *= 0.3

        # 代码长度合理性
        lines = proof_code.strip().split("\n")
        if 2 <= len(lines) <= 30:
            score += 0.15
        elif len(lines) > 50:
            score -= 0.1

        # 结构化证明有 have 步骤 → 更可能正确
        if "have " in proof_code:
            score += 0.1

        # 使用了常见的 automation tactic → 更简洁
        auto_tactics = ["simp", "ring", "omega", "norm_num",
                        "linarith", "decide"]
        if any(t in proof_code for t in auto_tactics):
            score += 0.05

        # 以 := by 开头 → 基本格式正确
        stripped = proof_code.strip()
        if stripped.startswith(":= by") or stripped.startswith("by"):
            score += 0.05

        return min(1.0, max(0.0, score))

    @staticmethod
    def refine_confidence(result: 'AgentResult',
                          feedback: 'AgentFeedback' = None,
                          l0_passed: bool = True,
                          l1_passed: bool = False,
                          l2_passed: bool = False) -> float:
        """用验证反馈更新置信度 (委托给 ConfidenceEstimator)

        .. deprecated:: v4
            直接使用 ConfidenceEstimator.refine_confidence() 代替。
        """
        from agent.strategy.confidence_estimator import ConfidenceEstimator
        return ConfidenceEstimator.refine_confidence(
            result, feedback=feedback,
            l0_passed=l0_passed, l1_passed=l1_passed,
            l2_passed=l2_passed)
