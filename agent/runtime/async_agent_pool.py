"""agent/runtime/async_agent_pool.py — 异步子智能体调度池

与同步版共享 AgentSpec, AgentTask, AgentResult, ContextItem 数据类型。
核心改进:
  - SubAgent.execute → async: LLM 调用期间释放事件循环
  - AgentPool.run_parallel → asyncio.gather: 无 GIL, 真正并行
  - 可与 AsyncVerificationScheduler 在同一事件循环中协作:
    LLM 生成和 REPL 验证交替运行, 互不阻塞

Usage::

    async with AsyncAgentPool(llm=async_provider) as pool:
        results = await pool.run_parallel(specs_and_tasks)
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Optional

from agent.brain.async_llm_provider import AsyncLLMProvider
from agent.brain.llm_provider import LLMResponse
from agent.brain.roles import AgentRole, ROLE_PROMPTS
from agent.brain.response_parser import extract_lean_code
from agent.context.context_window import ContextWindow
from agent.memory.working_memory import WorkingMemory
from agent.runtime.sub_agent import (
    AgentSpec, AgentTask, AgentResult, ContextItem,
)
from agent.runtime.result_fuser import ResultFuser
from agent.strategy.budget_allocator import Budget

logger = logging.getLogger(__name__)


class AsyncSubAgent:
    """异步子智能体 — LLM 调用期间不阻塞事件循环"""

    def __init__(self, spec: AgentSpec, llm: AsyncLLMProvider,
                 tool_registry=None):
        self.spec = spec
        self.llm = llm
        self.context = ContextWindow(max_tokens=spec.context_budget)
        self.tool_registry = tool_registry

    async def execute(self, task: AgentTask) -> AgentResult:
        """异步执行任务"""
        start = time.time()

        # 1. 构建上下文
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

        # 3. 异步 LLM 调用
        try:
            tools_schema = None
            if self.tool_registry and self.spec.tools:
                tools_schema = self.tool_registry.to_claude_tools_schema()

            resp = await self.llm.generate(
                system=system,
                user=user_prompt,
                temperature=self.spec.temperature,
                tools=tools_schema,
                max_tokens=self.spec.max_tokens)

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
                success=bool(proof_code.strip()),
                metadata=task.metadata)

        except Exception as e:
            latency = int((time.time() - start) * 1000)
            logger.error(f"AsyncSubAgent '{self.spec.name}' failed: {e}")
            return AgentResult(
                agent_name=self.spec.name,
                role=self.spec.role,
                content="",
                error=str(e),
                latency_ms=latency,
                tokens_used=0,
                confidence=0.0)

    def _estimate_confidence(self, resp: LLMResponse,
                             proof_code: str) -> float:
        if not proof_code.strip():
            return 0.0
        score = 0.3
        if "sorry" in proof_code or "admit" in proof_code:
            score *= 0.3
        lines = proof_code.strip().split("\n")
        if 2 <= len(lines) <= 30:
            score += 0.15
        elif len(lines) > 50:
            score -= 0.1
        if "have " in proof_code:
            score += 0.1
        auto_tactics = ["simp", "ring", "omega", "norm_num",
                        "linarith", "decide"]
        if any(t in proof_code for t in auto_tactics):
            score += 0.05
        stripped = proof_code.strip()
        if stripped.startswith(":= by") or stripped.startswith("by"):
            score += 0.05
        return min(1.0, max(0.0, score))


class AsyncAgentPool:
    """异步子智能体并行调度池

    核心改进:
      - run_parallel: asyncio.gather 替代 ThreadPoolExecutor
      - 一个事件循环中, 4 个 Agent 的 LLM 调用真正并行:
        Agent A 等 API 时, Agent B/C/D 可同时发送请求
    """

    def __init__(self, llm: AsyncLLMProvider,
                 tool_registry=None, max_workers: int = 4):
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_workers = max_workers
        self.fuser = ResultFuser()

    async def run_single(self, spec: AgentSpec, task: AgentTask,
                         budget: Budget = None) -> AgentResult:
        """运行单个异步子智能体"""
        agent = AsyncSubAgent(spec, self.llm, self.tool_registry)
        result = await agent.execute(task)
        if budget:
            budget.add_tokens(result.tokens_used)
        return result

    async def run_parallel(self, specs_and_tasks: list[tuple[AgentSpec, AgentTask]],
                           budget: Budget = None) -> list[AgentResult]:
        """并行运行多个异构子智能体 — asyncio.gather

        与同步版 ThreadPoolExecutor 的关键差异:
        - 同步: N 个线程, 每个阻塞等待 HTTP → GIL 序列化
        - 异步: 1 个事件循环, N 个 await → 真正并行 I/O
        """
        if not specs_and_tasks:
            return []

        # 用 semaphore 限制并发度
        sem = asyncio.Semaphore(self.max_workers)

        async def _run(spec: AgentSpec, task: AgentTask) -> AgentResult:
            async with sem:
                agent = AsyncSubAgent(spec, self.llm, self.tool_registry)
                return await agent.execute(task)

        results = await asyncio.gather(
            *(_run(spec, task) for spec, task in specs_and_tasks),
            return_exceptions=True)

        # 将异常转为 AgentResult
        final = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                spec, _ = specs_and_tasks[i]
                final.append(AgentResult(
                    agent_name=spec.name,
                    role=spec.role,
                    content="",
                    error=f"Execution error: {r}",
                    confidence=0.0))
            else:
                final.append(r)

        if budget:
            total_tokens = sum(r.tokens_used for r in final)
            budget.add_tokens(total_tokens)

        return final

    async def run_then_fuse(self, specs_and_tasks, budget=None,
                            fusion_strategy="best_confidence"):
        """并行运行 → 融合结果"""
        all_results = await self.run_parallel(specs_and_tasks, budget)
        best = self.fuser.select_best(all_results, strategy=fusion_strategy)
        return best, all_results

    async def run_pipeline(self, stages: list[dict],
                           initial_context: list[ContextItem] = None,
                           budget: Budget = None) -> AgentResult:
        """阶段式管线: 前一阶段输出注入下一阶段上下文"""
        current_context = list(initial_context or [])
        last_result = None

        for i, stage in enumerate(stages):
            spec = stage["spec"]
            task = AgentTask(
                description=stage.get("task_template", ""),
                injected_context=current_context)

            result = await self.run_single(spec, task, budget)
            last_result = result

            current_context.append(ContextItem(
                key=f"stage_{i}_output",
                content=result.content[:2000],
                priority=0.8,
                category="previous_stage"))

        return last_result

    async def inject_cross_agent(self, source_result: AgentResult,
                                 target_spec: AgentSpec,
                                 target_task: AgentTask,
                                 injection_key: str = "teammate_insight",
                                 budget: Budget = None) -> AgentResult:
        """跨智能体信息注入"""
        injection = ContextItem(
            key=injection_key,
            content=(f"A teammate ({source_result.agent_name}) found:\n"
                     f"{source_result.content[:1500]}"),
            priority=0.85,
            category="premise")
        target_task.injected_context.append(injection)
        return await self.run_single(target_spec, target_task, budget)
