"""knowledge/backend.py — Storage backend Protocol

Defines the abstract interface that any knowledge storage backend must
implement. The default implementation is SQLite (in store.py). Future
backends (PostgreSQL, Redis+PG, etc.) implement the same Protocol.

This enables swapping storage engines without changing any caller code::

    # Development / single-machine RL
    store = SqliteKnowledgeBackend("proofs.db")

    # Distributed RL training (future)
    store = PostgresKnowledgeBackend(dsn="postgresql://...")

    # Both satisfy the same Protocol
    writer = KnowledgeWriter(store)
    evolver = KnowledgeEvolver(store)
"""
from __future__ import annotations

from typing import Protocol, Optional, runtime_checkable

from knowledge.types import (
    TacticEffectiveness, ErrorPattern, LemmaRecord,
    StrategyPattern, ConceptNode, ConceptEdge,
    TacticSuggestion, LemmaMatch, StrategySuggestion,
)


@runtime_checkable
class KnowledgeBackend(Protocol):
    """Abstract storage backend for the unified knowledge system.

    All methods are async to support both local (SQLite via executor)
    and remote (PostgreSQL via asyncpg) backends uniformly.

    Layers:
      L0: Raw proof trajectories (save/load/export)
      L1: Tactical knowledge (tactics, errors, lemmas)
      L2: Strategy patterns
      L3: Concept graph
      Aux: Episodes, sessions, changelog
    """

    # ── L1: Tactic Effectiveness ──

    async def upsert_tactic_effectiveness(
        self, tactic: str, goal_pattern: str, success: bool,
        elapsed_ms: float = 0.0, domain: str = "",
        trace_id: int = 0,
    ) -> None: ...

    async def query_tactic_effectiveness(
        self, goal_pattern: str, domain: str = "",
        top_k: int = 10,
    ) -> list[TacticEffectiveness]: ...

    # ── L1: Error Patterns ──

    async def upsert_error_pattern(
        self, error_category: str, goal_pattern: str,
        tactic: str = "", fix_tactic: str = "",
        fix_succeeded: bool = False,
    ) -> None: ...

    async def query_error_patterns(
        self, goal_pattern: str = "", tactic: str = "",
        top_k: int = 10,
    ) -> list[ErrorPattern]: ...

    # ── L1: Proved Lemmas ──

    async def add_lemma(self, lemma: LemmaRecord) -> int: ...

    async def search_lemmas(
        self, keywords: list[str] = None, domain: str = "",
        goal_pattern: str = "", top_k: int = 10,
        verified_only: bool = False,
    ) -> list[LemmaMatch]: ...

    # ── L2: Strategy Patterns ──

    async def add_strategy_pattern(self, pattern: StrategyPattern) -> int: ...

    async def query_strategy_patterns(
        self, domain: str = "", top_k: int = 5,
    ) -> list[StrategyPattern]: ...

    # ── L3: Concept Graph ──

    async def upsert_concept(
        self, name: str, domain: str = "", description: str = "",
    ) -> int: ...

    async def add_concept_edge(
        self, source_name: str, target_name: str,
        relation_type: str, weight: float = 1.0,
    ) -> None: ...

    # ── Episodic Memory ──

    async def add_episode(
        self, problem_type: str, difficulty: str,
        winning_strategy: str, key_tactics: list[str],
        key_insight: str, solve_time_ms: int,
    ) -> int: ...

    async def query_episodes(
        self, problem_type: str = "", difficulty: str = "",
        top_k: int = 5,
    ) -> list[dict]: ...

    # ── Persistent Knowledge (failure/success patterns) ──

    async def record_failure(
        self, tactic: str, goal_type: str = "",
        error_category: str = "", domain: str = "",
    ) -> None: ...

    async def record_success(
        self, domain: str, tactics: list[str],
        theorem_type: str = "",
    ) -> None: ...

    async def get_suggestions(
        self, domain: str = "", goal_type: str = "",
        max_items: int = 5,
    ) -> list[str]: ...

    # ── Trajectory Export (for RL training) ──

    async def export_trajectories_batch(
        self, min_depth: int = 1, limit: int = 10000,
        format: str = "dict",
    ) -> list[dict]: ...

    # ── Stats & Lifecycle ──

    async def knowledge_stats(self) -> dict: ...

    async def decay_all(self, decay_rate: float = 0.95,
                        min_samples: int = 3) -> dict: ...

    async def reinforce(self, entity_type: str, entity_id: int,
                        reward: float) -> None: ...

    async def gc_stale(self, threshold: float = 0.1) -> dict: ...

    def close(self) -> None: ...
