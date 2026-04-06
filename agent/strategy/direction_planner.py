"""agent/strategy/direction_planner.py — 证明方向规划器

从 HeterogeneousEngine 中提取的方向规划逻辑。
根据问题特征、分类结果、历史尝试, 规划 2-4 个探索方向。

每个方向是一个 ProofDirection, 包含:
  - 角色 (proof_generator / planner / critic / repair)
  - 模型和温度
  - 策略提示 (strategic_hint)
  - 前提引理 (selected_premises)
  - few-shot 示例覆盖

规划策略可扩展: 继承 DirectionPlanner 并覆盖 plan() 方法。

Usage::

    planner = DirectionPlanner(retriever=retriever, plugin_loader=plugins)
    directions = planner.plan(problem, classification, attempt_history)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

from agent.brain.roles import AgentRole
from prover.models import BenchmarkProblem

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


class DirectionPlanner:
    """证明方向规划器

    根据问题特征规划异构探索方向。
    可通过继承定制规划策略。
    """

    def __init__(self, retriever=None, plugin_loader=None):
        self.retriever = retriever
        self.plugins = plugin_loader

    def plan(self, problem: BenchmarkProblem,
             classification: dict = None,
             attempt_history: list = None) -> list[ProofDirection]:
        """规划 2-4 个探索方向

        Args:
            problem: 待证明的问题
            classification: DomainClassifierHook 的分类结果
            attempt_history: 之前的尝试历史

        Returns:
            ProofDirection 列表
        """
        classification = classification or {}
        attempt_history = attempt_history or []
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
        hint_b, premises_b, few_shot_b = self._build_structured_direction(
            problem, classification)

        directions.append(ProofDirection(
            name="structured",
            role=AgentRole.PROOF_GENERATOR,
            temperature=0.7,
            strategic_hint=hint_b,
            selected_premises=premises_b[:15],
            few_shot_override=few_shot_b,
        ))

        # 方向 C: 替代路径
        directions.append(ProofDirection(
            name="alternative",
            role=AgentRole.PROOF_PLANNER,
            temperature=0.9,
            strategic_hint=(
                "Try a fundamentally DIFFERENT approach from standard methods. "
                "Consider: casting to ℤ if working with ℕ, "
                "using `conv` to restructure goals, "
                "or finding a non-obvious Mathlib lemma that solves it directly."
            ),
        ))

        # 方向 D: 反思修复 (仅当有失败历史时)
        if len(attempt_history) >= 2:
            repair_dir = self._build_repair_direction(attempt_history)
            if repair_dir:
                directions.append(repair_dir)

        return directions

    def _build_structured_direction(self, problem, classification):
        """构建结构化证明方向的提示和前提"""
        techniques = classification.get("techniques", [])
        has_nat_sub = classification.get("has_nat_sub", False)

        hint = (
            "Plan the proof structure carefully. "
            "Use `have` statements with explicit types for intermediate steps."
        )
        if "induction" in techniques:
            hint += (
                "\n\nThis problem likely requires induction on n. "
                "Structure: `induction n with | zero => ... | succ n ih => ...`"
            )
        if has_nat_sub:
            hint += (
                "\n\nCRITICAL: This involves natural number subtraction. "
                "In Lean4, ℕ subtraction truncates to 0. "
                "You MUST prove minuend ≥ subtrahend before subtracting. "
                "Use `Nat.sub_add_cancel` or `tsub_add_cancel_of_le`."
            )

        # 检查领域插件
        few_shot = ""
        if self.plugins:
            matched = self.plugins.match(
                problem.theorem_statement, classification)
            if matched:
                plugin = matched[0]
                if plugin.strategic_hint:
                    hint += f"\n\nDomain expert hint: {plugin.strategic_hint}"
                few_shot = plugin.few_shot_examples or ""

        # 获取前提
        premises = self._get_premises(problem.theorem_statement)
        if self.plugins:
            matched = self.plugins.match(
                problem.theorem_statement, classification)
            if matched and matched[0].extra_premises:
                for p in matched[0].extra_premises[:10]:
                    premises.append(p.get("statement", str(p)))

        return hint, premises, few_shot

    def _build_repair_direction(self, attempt_history):
        """构建反思修复方向"""
        recent_errors = []
        for a in attempt_history[-3:]:
            errs = a.get("errors", [])
            for e in errs[:2]:
                msg = e.get("message", str(e)) if isinstance(e, dict) else str(e)
                recent_errors.append(msg[:100])

        if not recent_errors:
            return None

        return ProofDirection(
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
        )

    def _get_premises(self, theorem: str) -> list[str]:
        """获取前提引理"""
        if not self.retriever:
            return []
        try:
            results = self.retriever.retrieve(theorem, top_k=10)
            if not results:
                return []
            if isinstance(results[0], str):
                return results
            return [r.get("statement", r.get("name", ""))
                    for r in results if isinstance(r, dict)]
        except Exception as e:
            logger.warning(f"Premise retrieval failed: {e}")
            return []


def build_direction_prompt(direction: ProofDirection,
                           problem: BenchmarkProblem) -> str:
    """为方向构建定制化 prompt (纯函数, 可独立测试)"""
    parts = [
        f"Prove the following Lean 4 theorem:\n"
        f"```lean\n{problem.theorem_statement}\n```",
    ]

    if direction.strategic_hint:
        parts.append(f"\n## Strategy guidance\n{direction.strategic_hint}")

    if problem.natural_language:
        parts.append(
            f"\n## Natural language description\n{problem.natural_language}")

    parts.append(
        "\nGenerate a complete proof. Output ONLY the proof body "
        "(starting with `:= by`) inside a single ```lean block. "
        "Do NOT use `sorry`."
    )

    return "\n".join(parts)
