"""knowledge/reader.py — 统一知识检索接口

替换现有碎片化检索：
  PersistentKnowledge.render_for_prompt()  → reader.render_for_prompt()
  LemmaBank.to_prompt_context()           → reader.find_lemmas()
  EpisodicMemory.retrieve_similar()       → reader.suggest_strategy()

Usage::

    reader = KnowledgeReader(store)

    # 推荐 tactic
    suggestions = await reader.suggest_tactics("⊢ n + 0 = n", domain="nat_arithmetic")

    # 获取领域简报 (综合注入)
    briefing = await reader.get_domain_briefing("nat_arithmetic", goal="⊢ n + 0 = n")
    prompt_text = briefing.render(max_chars=1500)

    # 直接获取 prompt 注入文本
    text = await reader.render_for_prompt(goal="⊢ n + 0 = n", theorem="...")
"""
from __future__ import annotations

import logging
from typing import Optional

from knowledge.goal_normalizer import (
    normalize_goal_for_key, classify_domain, extract_keywords,
)
from knowledge.store import UnifiedKnowledgeStore
from knowledge.types import (
    TacticSuggestion, StrategySuggestion, LemmaMatch,
    DomainBriefing,
)

logger = logging.getLogger(__name__)


class KnowledgeReader:
    """统一知识检索 — 从四层知识金字塔中检索相关知识"""

    def __init__(self, store: UnifiedKnowledgeStore):
        self.store = store

    async def suggest_tactics(
            self, goal: str, domain: str = "",
            top_k: int = 8) -> list[TacticSuggestion]:
        """推荐 tactic (来自 Layer 1)

        返回按置信度排序的建议列表，包括正面推荐和负面警告。
        """
        goal_pattern = normalize_goal_for_key(goal)
        if not goal_pattern:
            return []

        # 正面: 高成功率的 tactic
        effective = await self.store.query_tactic_effectiveness(
            goal_pattern, domain, top_k=top_k * 2)

        suggestions = []
        seen_tactics = set()

        for te in effective:
            if te.tactic in seen_tactics:
                continue
            seen_tactics.add(te.tactic)

            if te.total < 2:
                continue  # 样本太少，不推荐

            if te.success_rate >= 0.5:
                suggestions.append(TacticSuggestion(
                    tactic=te.tactic,
                    confidence=te.effective_confidence,
                    source="knowledge",
                    reason=f"{te.successes}/{te.total} succeeded "
                           f"in {te.avg_time_ms:.0f}ms avg",
                    avoid=False))
            elif te.success_rate < 0.2 and te.total >= 3:
                suggestions.append(TacticSuggestion(
                    tactic=te.tactic,
                    confidence=1.0 - te.effective_confidence,
                    source="knowledge",
                    reason=f"failed {te.failures}/{te.total} times",
                    avoid=True))

        # 负面: 已知错误模式
        errors = await self.store.query_error_patterns(
            goal_pattern=goal_pattern, top_k=5)
        for ep in errors:
            if ep.tactic and ep.tactic not in seen_tactics and ep.frequency >= 3:
                seen_tactics.add(ep.tactic)
                suggestions.append(TacticSuggestion(
                    tactic=ep.tactic,
                    confidence=0.8,
                    source="error_pattern",
                    reason=f"{ep.error_category} {ep.frequency}x"
                           + (f", try {ep.typical_fix}" if ep.typical_fix else ""),
                    avoid=True))
                # Also suggest the fix if available
                if ep.typical_fix and ep.fix_success_rate > 0.3:
                    if ep.typical_fix not in seen_tactics:
                        seen_tactics.add(ep.typical_fix)
                        suggestions.append(TacticSuggestion(
                            tactic=ep.typical_fix,
                            confidence=ep.fix_success_rate,
                            source="error_pattern",
                            reason=f"fixes {ep.tactic} "
                                   f"({ep.fix_success_rate:.0%} rate)",
                            avoid=False))

        # 排序: 正面推荐按 confidence 降序，负面警告按 confidence 降序
        positives = sorted(
            [s for s in suggestions if not s.avoid],
            key=lambda s: -s.confidence)
        negatives = sorted(
            [s for s in suggestions if s.avoid],
            key=lambda s: -s.confidence)

        return (positives[:top_k] + negatives[:3])

    async def suggest_strategy(
            self, problem_features: str,
            domain: str = "",
            top_k: int = 3) -> list[StrategySuggestion]:
        """推荐证明策略 (来自 Layer 2)"""
        patterns = await self.store.query_strategy_patterns(
            domain=domain, top_k=top_k)

        return [StrategySuggestion(
            name=p.name,
            tactic_template=p.tactic_template,
            confidence=p.confidence * p.decay_factor,
            reason=f"{p.times_succeeded}/{p.times_applied} succeeded",
            domain=p.domain,
        ) for p in patterns if p.total > 0 or p.confidence > 0.5]

    async def find_lemmas(
            self, goal: str = "", theorem: str = "",
            domain: str = "",
            top_k: int = 5) -> list[LemmaMatch]:
        """检索相关引理 (来自 Layer 1)"""
        keywords = extract_keywords(f"{theorem} {goal}")
        goal_pattern = normalize_goal_for_key(goal) if goal else ""

        return await self.store.search_lemmas(
            keywords=keywords,
            domain=domain,
            goal_pattern=goal_pattern,
            top_k=top_k)

    async def get_domain_briefing(
            self, domain: str = "",
            goal: str = "",
            theorem: str = "") -> DomainBriefing:
        """获取领域简报 — 综合 Layer 1 + Layer 2 的知识注入"""
        if not domain:
            domain = classify_domain(goal, theorem)

        all_suggestions = await self.suggest_tactics(
            goal, domain, top_k=10)
        strategies = await self.suggest_strategy(
            theorem, domain, top_k=3)
        lemmas = await self.find_lemmas(
            goal, theorem, domain, top_k=5)

        return DomainBriefing(
            domain=domain,
            top_tactics=[s for s in all_suggestions if not s.avoid],
            avoid_tactics=[s for s in all_suggestions if s.avoid],
            relevant_lemmas=lemmas,
            strategy_hints=strategies,
        )

    async def render_for_prompt(
            self, goal: str = "", theorem: str = "",
            domain: str = "",
            max_chars: int = 1500) -> str:
        """生成可直接注入 prompt 的知识文本

        这是对外的核心接口 — 替代现有碎片化的 render_for_prompt。
        """
        briefing = await self.get_domain_briefing(domain, goal, theorem)
        text = briefing.render(max_chars=max_chars)
        return text
