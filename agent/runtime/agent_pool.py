"""agent/runtime/agent_pool.py — 子智能体调度池

管理子智能体的并行执行、结果收集和跨智能体信息流。

核心能力:
  1. 并行启动多个异构子智能体 (不同角色/模型/prompt)
  2. 收集结果后通过 ResultFuser 融合
  3. 将一个智能体的发现注入另一个智能体的上下文 (信息流闭环)
"""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from agent.runtime.sub_agent import (
    SubAgent, AgentSpec, AgentTask, AgentResult, ContextItem
)
from agent.runtime.result_fuser import ResultFuser
from agent.brain.llm_provider import LLMProvider
from common.budget import Budget

logger = logging.getLogger(__name__)


class AgentPool:
    """异构子智能体并行调度池"""

    def __init__(self, llm: LLMProvider, tool_registry=None,
                 max_workers: int = 4):
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_workers = max_workers
        self.fuser = ResultFuser()

    def run_single(self, spec: AgentSpec, task: AgentTask,
                   budget: Budget = None) -> AgentResult:
        """运行单个子智能体"""
        agent = SubAgent(spec, self.llm, self.tool_registry)
        result = agent.execute(task)
        if budget:
            budget.add_tokens(result.tokens_used)
        return result

    def run_parallel(self, specs_and_tasks: list[tuple[AgentSpec, AgentTask]],
                     budget: Budget = None) -> list[AgentResult]:
        """并行运行多个异构子智能体

        每个 (spec, task) 对创建一个独立的 SubAgent，
        各自拥有独立的上下文窗口，互不干扰。

        Args:
            specs_and_tasks: [(AgentSpec, AgentTask), ...] 列表
            budget: 共享的预算控制器

        Returns:
            所有子智能体的结果列表
        """
        if not specs_and_tasks:
            return []

        results = []
        workers = min(self.max_workers, len(specs_and_tasks))

        def _run(spec_task):
            spec, task = spec_task
            agent = SubAgent(spec, self.llm, self.tool_registry)
            return agent.execute(task)

        if workers <= 1:
            for idx, st in enumerate(specs_and_tasks):
                try:
                    results.append(_run(st))
                except Exception as e:
                    logger.error(f"SubAgent execution error: {e}")
                    spec, _ = st
                    results.append(AgentResult(
                        agent_name=spec.name,
                        role=spec.role,
                        content="",
                        error=f"Execution error: {e}",
                        confidence=0.0,
                    ))
        else:
            # 使用 index 映射确保异常时结果列表长度与输入一致
            indexed_results = [None] * len(specs_and_tasks)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run, st): i
                           for i, st in enumerate(specs_and_tasks)}
                for f in as_completed(futures):
                    idx = futures[f]
                    try:
                        indexed_results[idx] = f.result()
                    except Exception as e:
                        logger.error(f"SubAgent execution error: {e}")
                        # 追加标记失败的结果, 保持长度一致
                        from agent.runtime.sub_agent import AgentResult
                        spec, _ = specs_and_tasks[idx]
                        indexed_results[idx] = AgentResult(
                            agent_name=spec.name,
                            role=spec.role,
                            content="",
                            error=f"Execution error: {e}",
                            confidence=0.0,
                        )
            results = indexed_results

        # 统一更新预算
        if budget:
            total_tokens = sum(r.tokens_used for r in results)
            budget.add_tokens(total_tokens)

        return results

    def run_then_fuse(self, specs_and_tasks: list[tuple[AgentSpec, AgentTask]],
                      budget: Budget = None,
                      fusion_strategy: str = "best_confidence"
                      ) -> tuple[AgentResult, list[AgentResult]]:
        """并行运行 → 融合结果 → 返回最佳结果 + 全部结果

        Returns:
            (best_result, all_results)
        """
        all_results = self.run_parallel(specs_and_tasks, budget)
        best = self.fuser.select_best(all_results, strategy=fusion_strategy)
        return best, all_results

    def run_pipeline(self, stages: list[dict],
                     initial_context: list[ContextItem] = None,
                     budget: Budget = None) -> AgentResult:
        """阶段式管线: 前一阶段的输出注入下一阶段的上下文

        Args:
            stages: [{"spec": AgentSpec, "task_template": str, "parallel": bool}, ...]
            initial_context: 初始上下文
            budget: 预算

        Returns:
            最后一个阶段的结果
        """
        current_context = list(initial_context or [])
        last_result = None

        for i, stage in enumerate(stages):
            spec = stage["spec"]
            task = AgentTask(
                description=stage.get("task_template", ""),
                injected_context=current_context,
            )

            result = self.run_single(spec, task, budget)
            last_result = result

            # 将本阶段结果注入下一阶段的上下文
            current_context.append(ContextItem(
                key=f"stage_{i}_output",
                content=result.content[:2000],
                priority=0.8,
                category="previous_stage",
            ))

        return last_result

    def inject_cross_agent(self, source_result: AgentResult,
                           target_spec: AgentSpec,
                           target_task: AgentTask,
                           injection_key: str = "teammate_insight",
                           budget: Budget = None) -> AgentResult:
        """将一个智能体的结果注入另一个智能体的上下文

        这是解决"信息流断裂"的核心机制。
        例: 检索智能体找到的引理 → 注入修复智能体的上下文。
        """
        # 构造注入内容
        injection = ContextItem(
            key=injection_key,
            content=(f"A teammate ({source_result.agent_name}) found:\n"
                     f"{source_result.content[:1500]}"),
            priority=0.85,
            category="premise",
        )
        target_task.injected_context.append(injection)

        return self.run_single(target_spec, target_task, budget)
