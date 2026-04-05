"""prover/pipeline/heterogeneous_engine.py — 异构并行证明引擎

当前系统的核心瓶颈: RolloutEngine 用同一个 prompt 生成 N 个样本,
所有样本往往犯同一类错误 (如 ℕ 减法用 ring)。

本引擎的改进: 同时启动多个策略方向完全不同的子智能体,
各有独立的角色/模型/prompt/上下文, 实现真正的策略多样性。

典型的四方向探索::

    方向 A: 自动化探测 (Haiku, 低温, 纯 tactic)
    方向 B: 归纳法专家 (Sonnet, 中温, 领域 prompt)
    方向 C: 代数变换   (Sonnet, 高温, 替代路径)
    方向 D: 引理检索   (Sonnet, 低温, 搜索 Mathlib)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

from agent.runtime.sub_agent import AgentSpec, AgentTask, AgentResult, ContextItem
from agent.runtime.agent_pool import AgentPool
from agent.runtime.result_fuser import ResultFuser
from agent.brain.roles import AgentRole
from agent.hooks.hook_types import HookEvent, HookContext
from agent.hooks.hook_manager import HookManager
from agent.plugins.loader import PluginLoader
from agent.strategy.budget_allocator import Budget
from prover.models import BenchmarkProblem, ProofAttempt, AttemptStatus

logger = logging.getLogger(__name__)


@dataclass
class ProofDirection:
    """一个证明探索方向的完整规格"""
    name: str
    role: AgentRole
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    strategic_hint: str = ""
    selected_premises: list[str] = field(default_factory=list)
    few_shot_override: str = ""
    allowed_tools: list[str] = field(default_factory=list)


class HeterogeneousEngine:
    """异构并行证明引擎

    替代 RolloutEngine 的同质化并行采样:
    - 每个方向是一个独立的 SubAgent
    - 不同方向有不同的角色、模型、温度、prompt
    - ResultFuser 融合结果, 支持跨方向信息注入
    """

    def __init__(self, pool: AgentPool, plugin_loader: PluginLoader = None,
                 hook_manager: HookManager = None, retriever=None):
        self.pool = pool
        self.plugins = plugin_loader or PluginLoader()
        self.hooks = hook_manager or HookManager()
        self.retriever = retriever
        self.fuser = ResultFuser()

    def run_round(self, problem: BenchmarkProblem,
                  classification: dict = None,
                  attempt_history: list = None,
                  budget: Budget = None) -> list[AgentResult]:
        """运行一轮异构并行证明

        Args:
            problem: 待证明的问题
            classification: DomainClassifierHook 的分类结果
            attempt_history: 之前的尝试历史 (用于 repair 方向)
            budget: 预算控制器

        Returns:
            所有方向的结果列表 (按 confidence 降序)
        """
        classification = classification or {}
        attempt_history = attempt_history or []

        # 1. 规划探索方向
        directions = self._plan_directions(
            problem, classification, attempt_history)

        # 2. 构建 (spec, task) 对
        specs_and_tasks = []
        for d in directions:
            spec = AgentSpec(
                name=d.name,
                role=d.role,
                model=d.model,
                temperature=d.temperature,
                few_shot_override=d.few_shot_override,
                tools=d.allowed_tools,
            )

            context_items = [
                ContextItem("theorem", problem.theorem_statement, 1.0,
                            "theorem_statement"),
            ]
            if d.strategic_hint:
                context_items.append(
                    ContextItem("strategy", d.strategic_hint, 0.9,
                                "tactic_hint"))
            if d.selected_premises:
                premises_text = "\n".join(
                    f"- {p}" for p in d.selected_premises[:15])
                context_items.append(
                    ContextItem("premises", premises_text, 0.7, "premise"))

            # 注入 hook 产生的上下文 (如 ℕ 减法警告)
            domain_hints = classification.get("domain_hints", {})
            for hk, hv in domain_hints.items():
                context_items.append(
                    ContextItem(hk, str(hv), 0.85, "premise"))

            task = AgentTask(
                description=self._build_direction_prompt(d, problem),
                injected_context=context_items,
                theorem_statement=problem.theorem_statement,
                metadata={"direction": d.name},
            )
            specs_and_tasks.append((spec, task))

        # 3. 并行执行
        results = self.pool.run_parallel(specs_and_tasks, budget)

        # 4. 按 confidence 排序
        results.sort(key=lambda r: -r.confidence)

        # 5. 尝试跨方向融合 — 如果最佳结果接近成功但缺引理
        if results and not any(r.confidence > 0.9 for r in results):
            fused = self._try_cross_fusion(results, problem, budget)
            if fused:
                results.insert(0, fused)

        return results

    def _plan_directions(self, problem, classification,
                         attempt_history) -> list[ProofDirection]:
        """根据问题特征规划 2-4 个探索方向"""
        directions = []

        # 方向 A: 自动化探测 (快速排除简单题)
        directions.append(ProofDirection(
            name="automation",
            role=AgentRole.PROOF_GENERATOR,
            model="claude-sonnet-4-20250514",
            temperature=0.2,
            strategic_hint=(
                "Try to solve this with simple automation ONLY. "
                "Attempt these tactics in order: decide, norm_num, simp, "
                "omega, ring, aesop. If a single tactic doesn't work, "
                "try 'simp; ring' or 'simp; omega'. "
                "Do NOT attempt induction or complex proof structures."
            ),
        ))

        # 方向 B: 结构化证明 (主力方向)
        techniques = classification.get("techniques", [])
        has_nat_sub = classification.get("has_nat_sub", False)

        hint_b = (
            "Plan the proof structure carefully. "
            "Use `have` statements with explicit types for intermediate steps."
        )
        if "induction" in techniques:
            hint_b += (
                "\n\nThis problem likely requires induction on n. "
                "Structure: `induction n with | zero => ... | succ n ih => ...`"
            )
        if has_nat_sub:
            hint_b += (
                "\n\nCRITICAL: This involves natural number subtraction. "
                "In Lean4, ℕ subtraction truncates to 0. "
                "You MUST prove minuend ≥ subtrahend before subtracting. "
                "Use `Nat.sub_add_cancel` or `tsub_add_cancel_of_le`."
            )

        # 检查领域插件
        matched_plugins = self.plugins.match(
            problem.theorem_statement, classification)
        if matched_plugins:
            plugin = matched_plugins[0]
            if plugin.strategic_hint:
                hint_b += f"\n\nDomain expert hint: {plugin.strategic_hint}"

        premises_b = self._get_premises(problem.theorem_statement)
        if matched_plugins and matched_plugins[0].extra_premises:
            # 合并插件的领域前提
            for p in matched_plugins[0].extra_premises[:10]:
                premises_b.append(p.get("statement", str(p)))

        directions.append(ProofDirection(
            name="structured",
            role=AgentRole.PROOF_GENERATOR,
            temperature=0.7,
            strategic_hint=hint_b,
            selected_premises=premises_b[:15],
            few_shot_override=(
                matched_plugins[0].few_shot_examples
                if matched_plugins else ""
            ),
        ))

        # 方向 C: 替代路径 (与 B 不同的角度)
        hint_c = (
            "Try a fundamentally DIFFERENT approach from standard methods. "
            "Consider: casting to ℤ if working with ℕ, "
            "using `conv` to restructure goals, "
            "or finding a non-obvious Mathlib lemma that solves it directly."
        )
        directions.append(ProofDirection(
            name="alternative",
            role=AgentRole.PROOF_PLANNER,
            temperature=0.9,
            strategic_hint=hint_c,
        ))

        # 方向 D: 反思修复 (仅当有失败历史时)
        if len(attempt_history) >= 2:
            recent_errors = []
            for a in attempt_history[-3:]:
                errs = a.get("errors", [])
                for e in errs[:2]:
                    msg = e.get("message", str(e)) if isinstance(e, dict) else str(e)
                    recent_errors.append(msg[:100])

            directions.append(ProofDirection(
                name="repair_rethink",
                role=AgentRole.CRITIC,
                temperature=0.5,
                strategic_hint=(
                    f"Previous {len(attempt_history)} attempts all failed. "
                    f"Recent errors:\n" +
                    "\n".join(f"  - {e}" for e in recent_errors) +
                    "\n\nAnalyze WHY these approaches fail at a fundamental level. "
                    "Then propose a completely different proof strategy."
                ),
            ))

        return directions

    def _try_cross_fusion(self, results, problem, budget):
        """尝试将一个方向的发现注入另一个方向

        典型场景: 检索方向找到了有用引理, 注入到结构化方向的修复上下文中。
        """
        # 找到最佳结构化结果和有引理发现的结果
        best_proof_result = None
        lemma_results = []

        for r in results:
            if r.proof_code and r.confidence > 0.3:
                if best_proof_result is None:
                    best_proof_result = r
            if r.content and ("lemma" in r.content.lower()
                              or "theorem" in r.content.lower()):
                lemma_results.append(r)

        if not best_proof_result or not lemma_results:
            return None

        # 构建融合修复任务
        lemma_insights = self.fuser.merge_insights(lemma_results, 500)
        useful_lemmas = self.fuser.extract_useful_lemmas(results)

        repair_spec = AgentSpec(
            name="cross_fusion_repair",
            role=AgentRole.REPAIR_AGENT,
            temperature=0.5,
        )

        repair_task = AgentTask(
            description=(
                f"A previous attempt generated this proof:\n"
                f"```lean\n{best_proof_result.proof_code[:1000]}\n```\n\n"
                f"Teammates found these potentially useful insights:\n"
                f"{lemma_insights}\n\n"
                f"Potentially useful Mathlib lemmas: "
                f"{', '.join(useful_lemmas[:10])}\n\n"
                f"Fix the proof using these insights."
            ),
            injected_context=[
                ContextItem("theorem", problem.theorem_statement, 1.0),
            ],
        )

        return self.pool.run_single(repair_spec, repair_task, budget)

    def _get_premises(self, theorem: str) -> list[str]:
        """获取前提引理"""
        if self.retriever:
            results = self.retriever.retrieve(theorem, top_k=10)
            return [r.get("statement", r.get("name", ""))
                    for r in results]
        return []

    def _build_direction_prompt(self, direction, problem) -> str:
        """为每个方向构建定制化的 prompt"""
        parts = [
            f"Prove the following Lean 4 theorem:\n"
            f"```lean\n{problem.theorem_statement}\n```",
        ]

        if direction.strategic_hint:
            parts.append(f"\n## Strategy guidance\n{direction.strategic_hint}")

        if problem.natural_language:
            parts.append(f"\n## Natural language description\n{problem.natural_language}")

        parts.append(
            "\nGenerate a complete proof. Output ONLY the proof body "
            "(starting with `:= by`) inside a single ```lean block. "
            "Do NOT use `sorry`."
        )

        return "\n".join(parts)
