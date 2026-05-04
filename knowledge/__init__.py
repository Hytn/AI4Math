"""knowledge — 统一知识系统

四层知识金字塔，统一管理证明经验：
  Layer 0: 原始轨迹 (继承 engine.ProofContextStore)
  Layer 1: 战术级知识 (tactic效能 · 已证引理 · 错误模式)
  Layer 2: 策略模式 (归纳出的证明策略模板)
  Layer 3: 直觉图谱 (跨领域概念关联)

v10 删除了无主路径调用的 4 个子模块:
  knowledge.evolver       (decay/gc 子系统)
  knowledge.broadcaster   (BroadcastBus 桥接)
  knowledge.retriever     (旧 Mathlib 检索器)
  knowledge.backend       (Protocol 抽象层)
保留: store / writer / reader / types / goal_normalizer / dialog_index /
      tfidf_retriever — 都被主路径或活子模块引用。
"""
__version__ = "0.3.0"

from knowledge.store import UnifiedKnowledgeStore
from knowledge.writer import KnowledgeWriter
from knowledge.reader import KnowledgeReader
from knowledge.types import (
    TacticEffectiveness, ErrorPattern, LemmaRecord,
    StrategyPattern, ConceptNode, ConceptEdge,
    TacticSuggestion, StrategySuggestion, LemmaMatch,
    DomainBriefing,
)
from knowledge.goal_normalizer import (
    normalize_level1, normalize_goal_for_key,
    classify_domain, extract_keywords, statement_hash,
)
