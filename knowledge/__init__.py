"""knowledge — 统一知识系统

四层知识金字塔，统一管理证明经验：
  Layer 0: 原始轨迹 (继承 engine.ProofContextStore)
  Layer 1: 战术级知识 (tactic效能 · 已证引理 · 错误模式)
  Layer 2: 策略模式 (归纳出的证明策略模板)
  Layer 3: 直觉图谱 (跨领域概念关联)
"""
__version__ = "0.2.0"

from knowledge.store import UnifiedKnowledgeStore
from knowledge.writer import KnowledgeWriter
from knowledge.reader import KnowledgeReader
from knowledge.broadcaster import KnowledgeBroadcaster
from knowledge.evolver import KnowledgeEvolver
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
