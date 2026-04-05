"""agent/hooks/builtin_hooks.py — 内置钩子集合

开箱即用的钩子, 解决当前系统中最常见的失败模式。

DomainClassifierHook:   ON_PROBLEM_START — 分类问题领域, 匹配插件
RepetitionDetectorHook: ON_ROUND_END     — 检测重复错误, 触发升级
NatSubSafetyHook:       PRE_VERIFICATION — ℕ减法安全检查
ReflectionCloserHook:   ON_STRATEGY_SWITCH — 将反思结论结构化注入
"""
from __future__ import annotations
import re
from agent.hooks.hook_types import (
    Hook, HookContext, HookResult, HookAction
)


class DomainClassifierHook(Hook):
    """ON_PROBLEM_START: 零成本领域分类

    通过关键词匹配判断定理所属数学分支和可能的证明技术,
    结果写入 metadata 供后续模块 (PluginLoader, AgentPool) 使用。
    """
    name = "domain_classifier"

    DOMAIN_KEYWORDS = {
        "number_theory": [
            r"\bdvd\b", r"\bprime\b", r"\bgcd\b", r"\bmod\b",
            r"Nat\.\w*prime", r"Int\.\w*mod", r"Finset\.range",
            r"∣", r"≡",
        ],
        "algebra": [
            r"\bGroup\b", r"\bRing\b", r"\bField\b", r"\bModule\b",
            r"Subgroup", r"RingHom", r"Ideal", r"Polynomial",
        ],
        "analysis": [
            r"\bReal\b", r"\bℝ\b", r"Filter\.", r"Tendsto",
            r"∫", r"deriv", r"ContinuousOn", r"MeasureTheory",
        ],
        "combinatorics": [
            r"Finset\.", r"Fintype", r"card\b", r"choose\b",
            r"∑.*Finset", r"Nat\.choose",
        ],
        "topology": [
            r"TopologicalSpace", r"IsOpen", r"Continuous",
            r"Compact", r"Connected",
        ],
    }

    TECHNIQUE_PATTERNS = {
        "induction": [
            r"∀\s*n", r"Nat\b.*→.*Nat\b", r"\(n\s*\+\s*1\)",
            r"succ", r"Nat\.rec",
        ],
        "contradiction": [r"¬", r"False", r"absurd"],
        "cases": [r"∨", r"Or\b", r"Decidable"],
    }

    NAT_SUB_PATTERN = re.compile(
        r"2\s*\^\s*.*-|"
        r"\w+\s*\^\s*\w+\s*-\s*|"
        r":\s*ℕ\b.*-\s*\(|"
        r"Nat\b.*-\s*",
        re.IGNORECASE,
    )

    def execute(self, ctx: HookContext) -> HookResult:
        stmt = ctx.theorem_statement
        if not stmt:
            return HookResult()

        # 领域分类
        domains = []
        for domain, patterns in self.DOMAIN_KEYWORDS.items():
            for pat in patterns:
                if re.search(pat, stmt):
                    domains.append(domain)
                    break

        # 证明技术判断
        techniques = []
        for tech, patterns in self.TECHNIQUE_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, stmt):
                    techniques.append(tech)
                    break

        # ℕ 减法检测
        has_nat_sub = bool(self.NAT_SUB_PATTERN.search(stmt))

        classification = {
            "domains": domains or ["general"],
            "techniques": techniques,
            "has_nat_sub": has_nat_sub,
        }

        return HookResult(
            action=HookAction.CONTINUE,  # 不中断流程, 只注入信息
            inject_context={"classification": classification},
            message=f"Classified: {classification}",
        )


class RepetitionDetectorHook(Hook):
    """ON_ROUND_END: 检测重复错误模式

    当同一类错误连续出现 N 次时, 触发 ESCALATE 动作,
    而不是被动等待 rounds_completed 达到阈值。
    """
    name = "repetition_detector"

    def __init__(self, threshold: int = 4):
        self.threshold = threshold

    def execute(self, ctx: HookContext) -> HookResult:
        if not ctx.dominant_error or ctx.dominant_error == "none":
            return HookResult()

        repeat_count = ctx.metadata.get("dominant_error_count", 0)

        if repeat_count >= self.threshold:
            return HookResult(
                action=HookAction.ESCALATE,
                message=(f"Error '{ctx.dominant_error}' repeated "
                         f"{repeat_count} times — forcing escalation"),
                inject_context={
                    "escalation_reason": f"stuck_on_{ctx.dominant_error}",
                    "hint": (f"Repeated {ctx.dominant_error} errors suggest "
                             f"the current approach is fundamentally wrong. "
                             f"Try a completely different proof strategy."),
                },
            )
        return HookResult()


class NatSubSafetyHook(Hook):
    """PRE_VERIFICATION: ℕ 减法安全检查

    在提交 Lean4 验证前, 检查证明中是否包含自然数减法
    且缺少相应的 ≤ 证明。如果检测到风险, 注入修复提示。
    """
    name = "nat_sub_safety"

    _SUB_PATTERN = re.compile(r'(\w+)\s*-\s*(\w+)')
    _SAFETY_PATTERNS = [
        r'tsub_add_cancel', r'Nat\.sub_add_cancel',
        r'le_of_lt', r'Nat\.le', r'≤',
        r'Nat\.sub_le', r'omega',
    ]

    def execute(self, ctx: HookContext) -> HookResult:
        if not ctx.proof:
            return HookResult()

        # 检查是否有 ℕ 减法
        has_sub = bool(self._SUB_PATTERN.search(ctx.proof))
        if not has_sub:
            return HookResult()

        # 检查是否有相应的安全处理
        has_safety = any(
            re.search(pat, ctx.proof) for pat in self._SAFETY_PATTERNS
        )

        if has_safety:
            return HookResult()

        # 有减法但没有安全处理 → 注入警告
        return HookResult(
            action=HookAction.MODIFY,
            message="ℕ subtraction without ≤ guard detected",
            inject_context={
                "nat_sub_warning": (
                    "WARNING: This proof contains natural number subtraction "
                    "without proving minuend ≥ subtrahend. "
                    "In Lean4, ℕ subtraction truncates to 0. "
                    "Use `Nat.sub_add_cancel` or `tsub_add_cancel_of_le` "
                    "after establishing the ≤ relationship."
                ),
            },
        )


class ReflectionCloserHook(Hook):
    """ON_STRATEGY_SWITCH: 将反思结论结构化注入

    解决当前系统中 Reflector 分析结果只打 log 不闭环的问题。
    """
    name = "reflection_closer"

    def execute(self, ctx: HookContext) -> HookResult:
        reflection = ctx.metadata.get("reflection_text", "")
        if not reflection:
            return HookResult()

        return HookResult(
            action=HookAction.CONTINUE,
            inject_context={
                "reflection_insight": (
                    f"## Strategic insight from self-reflection\n"
                    f"{reflection[:800]}\n\n"
                    f"Use this analysis to choose a fundamentally "
                    f"different approach in the next round."
                ),
            },
            message="Reflection insight injected into next round context",
        )
