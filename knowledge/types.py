"""knowledge/types.py — 统一知识系统的共享数据类型

所有层共用的数据结构，定义了知识系统内部和对外的接口合同。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Layer 1: Tactical Knowledge 类型
# ═══════════════════════════════════════════════════════════════

@dataclass
class TacticEffectiveness:
    """tactic 在特定 goal pattern 上的效能统计"""
    id: int = 0
    tactic: str = ""
    goal_pattern: str = ""
    domain: str = ""
    successes: int = 0
    failures: int = 0
    avg_time_ms: float = 0.0
    last_seen: float = 0.0
    confidence: float = 0.5
    decay_factor: float = 1.0
    sample_traces: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.total)

    @property
    def effective_confidence(self) -> float:
        return self.confidence * self.decay_factor


@dataclass
class ErrorPattern:
    """错误模式：特定 tactic+goal 组合的失败规律"""
    id: int = 0
    error_category: str = ""
    goal_pattern: str = ""
    tactic: str = ""
    frequency: int = 1
    typical_fix: str = ""
    fix_success_rate: float = 0.0
    last_seen: float = 0.0
    description: str = ""


@dataclass
class LemmaRecord:
    """已证引理（统一表示，合并原 LemmaBank + PersistentLemmaBank）"""
    id: int = 0
    name: str = ""
    statement: str = ""
    proof: str = ""
    statement_hash: str = ""
    # 来源追溯
    source_problem: str = ""
    source_trace_id: int = 0
    # 质量信号
    verified: bool = False
    times_cited: int = 0
    last_cited_at: float = 0.0
    # 检索辅助
    keywords: list[str] = field(default_factory=list)
    domain: str = ""
    goal_types: list[str] = field(default_factory=list)
    # 生命周期
    created_at: float = 0.0
    stale: bool = False
    decay_factor: float = 1.0

    def to_lean(self) -> str:
        return f"{self.statement} {self.proof}"


# ═══════════════════════════════════════════════════════════════
# Layer 2: Strategy Patterns 类型
# ═══════════════════════════════════════════════════════════════

@dataclass
class StrategyPattern:
    """归纳出的证明策略模式"""
    id: int = 0
    name: str = ""
    domain: str = ""
    problem_pattern: str = ""
    tactic_template: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    times_applied: int = 0
    times_succeeded: int = 0
    avg_depth: float = 0.0
    confidence: float = 0.5
    source_episodes: list[int] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    decay_factor: float = 1.0

    @property
    def success_rate(self) -> float:
        return self.times_succeeded / max(1, self.times_applied)


# ═══════════════════════════════════════════════════════════════
# Layer 3: Intuition Graph 类型
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConceptNode:
    """概念图谱节点"""
    id: int = 0
    name: str = ""
    domain: str = ""
    description: str = ""
    difficulty_est: float = 0.5
    encounter_count: int = 0
    created_at: float = 0.0


@dataclass
class ConceptEdge:
    """概念图谱边"""
    id: int = 0
    source_id: int = 0
    target_id: int = 0
    relation_type: str = ""  # prerequisite, analogy, generalizes, often_co_occurs
    weight: float = 1.0
    evidence_count: int = 1
    created_at: float = 0.0


# ═══════════════════════════════════════════════════════════════
# 检索结果类型
# ═══════════════════════════════════════════════════════════════

@dataclass
class TacticSuggestion:
    """检索返回的 tactic 建议"""
    tactic: str
    confidence: float
    source: str = "knowledge"  # "knowledge", "error_pattern", "broadcast"
    reason: str = ""
    avoid: bool = False  # True = 建议避免此 tactic

    def to_prompt_line(self) -> str:
        prefix = "AVOID" if self.avoid else "Try"
        return f"  - {prefix} `{self.tactic}` ({self.reason}) [{self.confidence:.0%}]"


@dataclass
class StrategySuggestion:
    """检索返回的策略建议"""
    name: str
    tactic_template: list[str]
    confidence: float
    reason: str = ""
    domain: str = ""


@dataclass
class LemmaMatch:
    """检索返回的引理匹配"""
    name: str
    statement: str
    proof: str
    relevance_score: float
    times_cited: int = 0

    def to_lean(self) -> str:
        return f"{self.statement} {self.proof}"

    def to_prompt_line(self) -> str:
        return f"  - {self.name}: `{self.statement}` (cited {self.times_cited}x)"


@dataclass
class DomainBriefing:
    """领域简报：综合知识注入"""
    domain: str
    top_tactics: list[TacticSuggestion] = field(default_factory=list)
    avoid_tactics: list[TacticSuggestion] = field(default_factory=list)
    relevant_lemmas: list[LemmaMatch] = field(default_factory=list)
    strategy_hints: list[StrategySuggestion] = field(default_factory=list)
    error_warnings: list[str] = field(default_factory=list)

    def render(self, max_chars: int = 1500) -> str:
        """渲染为可注入 prompt 的文本"""
        parts: list[str] = []
        total = 0

        if self.top_tactics:
            parts.append(f"## Effective tactics for {self.domain or 'this problem'}\n")
            for t in self.top_tactics[:5]:
                line = t.to_prompt_line() + "\n"
                if total + len(line) > max_chars:
                    break
                parts.append(line)
                total += len(line)

        if self.avoid_tactics:
            parts.append("\n## Known pitfalls\n")
            for t in self.avoid_tactics[:3]:
                line = t.to_prompt_line() + "\n"
                if total + len(line) > max_chars:
                    break
                parts.append(line)
                total += len(line)

        if self.relevant_lemmas:
            parts.append("\n## Relevant proved lemmas\n")
            for lm in self.relevant_lemmas[:5]:
                line = lm.to_prompt_line() + "\n"
                if total + len(line) > max_chars:
                    break
                parts.append(line)
                total += len(line)

        if self.strategy_hints:
            parts.append("\n## Suggested proof strategies\n")
            for s in self.strategy_hints[:3]:
                seq = " → ".join(s.tactic_template[:6])
                line = f"  - {s.name}: {seq} [{s.confidence:.0%}]\n"
                if total + len(line) > max_chars:
                    break
                parts.append(line)
                total += len(line)

        return "".join(parts)
