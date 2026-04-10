"""knowledge/store.py — 统一知识存储后端

单一 SQLite 数据库管理四层知识金字塔：
  Layer 0: 原始轨迹 (继承 ProofContextStore)
  Layer 1: 战术级知识 (tactic_effectiveness, proved_lemmas, error_patterns)
  Layer 2: 策略模式 (strategy_patterns)
  Layer 3: 直觉图谱 (concept_nodes, concept_edges)

设计原则：
  - 继承 ProofContextStore 的全部 Layer 0 能力 (零回归风险)
  - 新增表通过 _ensure_tables() 惰性创建 (向后兼容旧数据库)
  - 所有写入通过 run_in_executor 异步化
  - WAL 模式支持并发读
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

from engine.proof_context_store import ProofContextStore
from knowledge.types import (
    TacticEffectiveness, ErrorPattern, LemmaRecord,
    StrategyPattern, ConceptNode, ConceptEdge,
    TacticSuggestion, LemmaMatch,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Layer 1/2/3 Schema
# ═══════════════════════════════════════════════════════════════

_KNOWLEDGE_SCHEMA = """
-- Layer 1: tactic effectiveness
CREATE TABLE IF NOT EXISTS tactic_effectiveness (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tactic          TEXT NOT NULL,
    goal_pattern    TEXT NOT NULL,
    domain          TEXT DEFAULT '',
    successes       INTEGER DEFAULT 0,
    failures        INTEGER DEFAULT 0,
    avg_time_ms     REAL DEFAULT 0.0,
    last_seen       REAL NOT NULL,
    confidence      REAL DEFAULT 0.5,
    decay_factor    REAL DEFAULT 1.0,
    sample_traces   TEXT DEFAULT '[]',
    UNIQUE(tactic, goal_pattern)
);
CREATE INDEX IF NOT EXISTS idx_te_tactic ON tactic_effectiveness(tactic);
CREATE INDEX IF NOT EXISTS idx_te_goal ON tactic_effectiveness(goal_pattern);
CREATE INDEX IF NOT EXISTS idx_te_domain ON tactic_effectiveness(domain);

-- Layer 1: proved lemmas
CREATE TABLE IF NOT EXISTS proved_lemmas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    statement       TEXT NOT NULL,
    proof           TEXT NOT NULL,
    statement_hash  TEXT NOT NULL UNIQUE,
    source_problem  TEXT DEFAULT '',
    source_trace_id INTEGER DEFAULT 0,
    verified        INTEGER DEFAULT 0,
    times_cited     INTEGER DEFAULT 0,
    last_cited_at   REAL DEFAULT 0.0,
    keywords        TEXT DEFAULT '[]',
    domain          TEXT DEFAULT '',
    goal_types      TEXT DEFAULT '[]',
    created_at      REAL NOT NULL,
    stale           INTEGER DEFAULT 0,
    decay_factor    REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_pl_hash ON proved_lemmas(statement_hash);
CREATE INDEX IF NOT EXISTS idx_pl_domain ON proved_lemmas(domain);
CREATE INDEX IF NOT EXISTS idx_pl_verified ON proved_lemmas(verified);

-- Layer 1: error patterns
CREATE TABLE IF NOT EXISTS error_patterns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    error_category  TEXT NOT NULL,
    goal_pattern    TEXT NOT NULL,
    tactic          TEXT DEFAULT '',
    frequency       INTEGER DEFAULT 1,
    typical_fix     TEXT DEFAULT '',
    fix_success_rate REAL DEFAULT 0.0,
    last_seen       REAL NOT NULL,
    description     TEXT DEFAULT '',
    UNIQUE(error_category, goal_pattern, tactic)
);
CREATE INDEX IF NOT EXISTS idx_ep_category ON error_patterns(error_category);
CREATE INDEX IF NOT EXISTS idx_ep_goal ON error_patterns(goal_pattern);

-- Layer 2: strategy patterns
CREATE TABLE IF NOT EXISTS strategy_patterns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    domain          TEXT DEFAULT '',
    problem_pattern TEXT NOT NULL,
    tactic_template TEXT NOT NULL,
    preconditions   TEXT DEFAULT '[]',
    times_applied   INTEGER DEFAULT 0,
    times_succeeded INTEGER DEFAULT 0,
    avg_depth       REAL DEFAULT 0.0,
    confidence      REAL DEFAULT 0.5,
    source_episodes TEXT DEFAULT '[]',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    decay_factor    REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_sp_domain ON strategy_patterns(domain);

-- Layer 3: concept graph
CREATE TABLE IF NOT EXISTS concept_nodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    domain          TEXT DEFAULT '',
    description     TEXT DEFAULT '',
    difficulty_est  REAL DEFAULT 0.5,
    encounter_count INTEGER DEFAULT 0,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       INTEGER NOT NULL,
    target_id       INTEGER NOT NULL,
    relation_type   TEXT NOT NULL,
    weight          REAL DEFAULT 1.0,
    evidence_count  INTEGER DEFAULT 1,
    created_at      REAL NOT NULL,
    FOREIGN KEY (source_id) REFERENCES concept_nodes(id),
    FOREIGN KEY (target_id) REFERENCES concept_nodes(id),
    UNIQUE(source_id, target_id, relation_type)
);

-- Knowledge changelog (for evolution audit trail)
CREATE TABLE IF NOT EXISTS knowledge_changelog (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    layer           TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    entity_id       INTEGER NOT NULL,
    action          TEXT NOT NULL,
    old_value       TEXT DEFAULT '',
    new_value       TEXT DEFAULT '',
    reason          TEXT DEFAULT '',
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_kcl_layer ON knowledge_changelog(layer);

-- Episodic memory (replaces episodic_memory.jsonl)
CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_type    TEXT NOT NULL,
    difficulty      TEXT DEFAULT '',
    winning_strategy TEXT DEFAULT '',
    key_tactics     TEXT DEFAULT '[]',
    key_insight     TEXT DEFAULT '',
    solve_time_ms   INTEGER DEFAULT 0,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ep_type ON episodes(problem_type);
CREATE INDEX IF NOT EXISTS idx_ep_diff ON episodes(difficulty);

-- Persistent knowledge: failure patterns (replaces knowledge_base.json)
CREATE TABLE IF NOT EXISTS pk_failures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tactic          TEXT NOT NULL,
    goal_type       TEXT NOT NULL,
    error_category  TEXT DEFAULT '',
    domain          TEXT DEFAULT '',
    count           INTEGER DEFAULT 1,
    last_seen       REAL NOT NULL,
    UNIQUE(tactic, goal_type)
);

-- Persistent knowledge: success patterns
CREATE TABLE IF NOT EXISTS pk_successes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    domain          TEXT NOT NULL,
    tactic_combo    TEXT NOT NULL,
    theorem_type    TEXT DEFAULT '',
    count           INTEGER DEFAULT 1,
    last_seen       REAL NOT NULL,
    UNIQUE(domain, tactic_combo)
);
"""


class UnifiedKnowledgeStore(ProofContextStore):
    """统一知识存储 — 继承 Layer 0, 新增 Layer 1/2/3

    使用方式与 ProofContextStore 完全兼容 (is-a 关系),
    额外提供 Layer 1/2/3 的 CRUD 操作。
    """

    def __init__(self, db_path: str = ":memory:"):
        super().__init__(db_path)
        self._ensure_knowledge_tables()

    def _ensure_knowledge_tables(self):
        """创建 Layer 1/2/3 表 (幂等)"""
        try:
            with self._connect() as conn:
                conn.executescript(_KNOWLEDGE_SCHEMA)
            logger.debug("UnifiedKnowledgeStore: knowledge tables ensured")
        except Exception as e:
            logger.warning(f"UnifiedKnowledgeStore: table creation warning: {e}")

    # ═══════════════════════════════════════════════════════════
    # Layer 1: Tactic Effectiveness
    # ═══════════════════════════════════════════════════════════

    async def upsert_tactic_effectiveness(
            self, tactic: str, goal_pattern: str,
            success: bool, elapsed_ms: float = 0.0,
            domain: str = "",
            trace_id: int = 0) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._upsert_te_sync,
            tactic, goal_pattern, success, elapsed_ms, domain, trace_id)

    def _upsert_te_sync(self, tactic, goal_pattern, success,
                         elapsed_ms, domain, trace_id):
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, successes, failures, avg_time_ms, sample_traces "
                "FROM tactic_effectiveness "
                "WHERE tactic=? AND goal_pattern=?",
                (tactic, goal_pattern)).fetchone()

            if row:
                s = row["successes"] + (1 if success else 0)
                f = row["failures"] + (0 if success else 1)
                total = s + f
                # Running average of elapsed time
                old_avg = row["avg_time_ms"]
                new_avg = old_avg + (elapsed_ms - old_avg) / max(1, total)
                # Bayesian confidence update
                confidence = s / max(1, total)
                # Track recent traces (keep last 10)
                traces = json.loads(row["sample_traces"] or "[]")
                if trace_id > 0:
                    traces.append(trace_id)
                    traces = traces[-10:]

                conn.execute(
                    "UPDATE tactic_effectiveness SET "
                    "successes=?, failures=?, avg_time_ms=?, confidence=?, "
                    "last_seen=?, decay_factor=1.0, sample_traces=?, domain=? "
                    "WHERE id=?",
                    (s, f, new_avg, confidence, now,
                     json.dumps(traces),
                     domain or "", row["id"]))
            else:
                s = 1 if success else 0
                f = 0 if success else 1
                traces = [trace_id] if trace_id > 0 else []
                conn.execute(
                    "INSERT INTO tactic_effectiveness "
                    "(tactic, goal_pattern, domain, successes, failures, "
                    "avg_time_ms, last_seen, confidence, decay_factor, "
                    "sample_traces) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0, ?)",
                    (tactic, goal_pattern, domain, s, f,
                     elapsed_ms, now, s / max(1, s + f),
                     json.dumps(traces)))

    async def query_tactic_effectiveness(
            self, goal_pattern: str, domain: str = "",
            top_k: int = 10) -> list[TacticEffectiveness]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._query_te_sync, goal_pattern, domain, top_k)

    def _query_te_sync(self, goal_pattern, domain, top_k):
        with self._connect() as conn:
            # Exact match first, then LIKE fallback
            rows = conn.execute(
                "SELECT * FROM tactic_effectiveness "
                "WHERE goal_pattern=? "
                "ORDER BY (confidence * decay_factor) DESC LIMIT ?",
                (goal_pattern, top_k)).fetchall()

            if not rows and domain:
                rows = conn.execute(
                    "SELECT * FROM tactic_effectiveness "
                    "WHERE domain=? "
                    "ORDER BY (confidence * decay_factor) DESC LIMIT ?",
                    (domain, top_k)).fetchall()

            return [TacticEffectiveness(
                id=r["id"], tactic=r["tactic"],
                goal_pattern=r["goal_pattern"], domain=r["domain"],
                successes=r["successes"], failures=r["failures"],
                avg_time_ms=r["avg_time_ms"], last_seen=r["last_seen"],
                confidence=r["confidence"], decay_factor=r["decay_factor"],
                sample_traces=json.loads(r["sample_traces"] or "[]"),
            ) for r in rows]

    # ═══════════════════════════════════════════════════════════
    # Layer 1: Error Patterns
    # ═══════════════════════════════════════════════════════════

    async def upsert_error_pattern(
            self, error_category: str, goal_pattern: str,
            tactic: str = "", fix_tactic: str = "",
            fix_succeeded: bool = False) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._upsert_ep_sync,
            error_category, goal_pattern, tactic, fix_tactic, fix_succeeded)

    def _upsert_ep_sync(self, error_category, goal_pattern,
                         tactic, fix_tactic, fix_succeeded):
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, frequency, typical_fix, fix_success_rate "
                "FROM error_patterns "
                "WHERE error_category=? AND goal_pattern=? AND tactic=?",
                (error_category, goal_pattern, tactic)).fetchone()

            if row:
                freq = row["frequency"] + 1
                old_fix = row["typical_fix"]
                old_rate = row["fix_success_rate"]
                # Update fix info if we have a new fix
                new_fix = fix_tactic or old_fix
                if fix_tactic and fix_succeeded:
                    new_rate = old_rate + (1.0 - old_rate) / freq
                elif fix_tactic:
                    new_rate = old_rate + (0.0 - old_rate) / freq
                else:
                    new_rate = old_rate

                conn.execute(
                    "UPDATE error_patterns SET "
                    "frequency=?, typical_fix=?, fix_success_rate=?, "
                    "last_seen=? WHERE id=?",
                    (freq, new_fix, new_rate, now, row["id"]))
            else:
                conn.execute(
                    "INSERT INTO error_patterns "
                    "(error_category, goal_pattern, tactic, frequency, "
                    "typical_fix, fix_success_rate, last_seen) "
                    "VALUES (?, ?, ?, 1, ?, ?, ?)",
                    (error_category, goal_pattern, tactic,
                     fix_tactic, 1.0 if fix_succeeded else 0.0, now))

    async def query_error_patterns(
            self, goal_pattern: str = "", tactic: str = "",
            top_k: int = 10) -> list[ErrorPattern]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._query_ep_sync, goal_pattern, tactic, top_k)

    def _query_ep_sync(self, goal_pattern, tactic, top_k):
        conditions = ["1=1"]
        params: list = []
        if goal_pattern:
            conditions.append("goal_pattern=?")
            params.append(goal_pattern)
        if tactic:
            conditions.append("tactic=?")
            params.append(tactic)

        where = " AND ".join(conditions)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM error_patterns "
                f"WHERE {where} "
                f"ORDER BY frequency DESC LIMIT ?",
                (*params, top_k)).fetchall()
            return [ErrorPattern(
                id=r["id"], error_category=r["error_category"],
                goal_pattern=r["goal_pattern"], tactic=r["tactic"],
                frequency=r["frequency"], typical_fix=r["typical_fix"],
                fix_success_rate=r["fix_success_rate"],
                last_seen=r["last_seen"],
                description=r["description"],
            ) for r in rows]

    # ═══════════════════════════════════════════════════════════
    # Layer 1: Proved Lemmas
    # ═══════════════════════════════════════════════════════════

    async def add_lemma(self, lemma: LemmaRecord) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._add_lemma_sync, lemma)

    def _add_lemma_sync(self, lemma: LemmaRecord) -> int:
        from knowledge.goal_normalizer import statement_hash as _hash
        sh = lemma.statement_hash or _hash(lemma.statement)
        now = time.time()

        with self._connect() as conn:
            # Check for duplicate
            existing = conn.execute(
                "SELECT id FROM proved_lemmas WHERE statement_hash=?",
                (sh,)).fetchone()
            if existing:
                # Update citation count
                conn.execute(
                    "UPDATE proved_lemmas SET times_cited = times_cited + 1, "
                    "last_cited_at=?, decay_factor=1.0 WHERE id=?",
                    (now, existing["id"]))
                return existing["id"]

            cur = conn.execute(
                "INSERT INTO proved_lemmas "
                "(name, statement, proof, statement_hash, source_problem, "
                "source_trace_id, verified, times_cited, last_cited_at, "
                "keywords, domain, goal_types, created_at, stale, "
                "decay_factor) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1.0)",
                (lemma.name, lemma.statement, lemma.proof, sh,
                 lemma.source_problem, lemma.source_trace_id,
                 int(lemma.verified), lemma.times_cited, now,
                 json.dumps(lemma.keywords), lemma.domain,
                 json.dumps(lemma.goal_types), now))
            lemma_id = cur.lastrowid
            self._log_change(
                conn, layer="L1", entity_type="proved_lemmas",
                entity_id=lemma_id, action="create",
                new_value=f"{lemma.name}: {lemma.statement[:100]}",
                reason=f"from {lemma.source_problem[:60]}")
            return lemma_id

    async def search_lemmas(
            self, keywords: list[str] = None,
            domain: str = "",
            goal_pattern: str = "",
            top_k: int = 10,
            verified_only: bool = False) -> list[LemmaMatch]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._search_lemmas_sync,
            keywords or [], domain, goal_pattern, top_k, verified_only)

    def _search_lemmas_sync(self, keywords, domain, goal_pattern,
                             top_k, verified_only):
        with self._connect() as conn:
            # Fetch candidates
            conditions = ["stale=0"]
            params: list = []
            if verified_only:
                conditions.append("verified=1")
            if domain:
                conditions.append("domain=?")
                params.append(domain)

            where = " AND ".join(conditions)
            rows = conn.execute(
                f"SELECT * FROM proved_lemmas "
                f"WHERE {where} "
                f"ORDER BY (times_cited * decay_factor) DESC "
                f"LIMIT ?",
                (*params, top_k * 3)).fetchall()  # over-fetch for scoring

        # Score by keyword overlap
        results = []
        kw_set = set(k.lower() for k in keywords)
        for r in rows:
            stored_kw = set(
                k.lower() for k in json.loads(r["keywords"] or "[]"))
            score = len(kw_set & stored_kw) if kw_set else 0
            # Boost by citation count
            score += r["times_cited"] * 0.1
            # Boost by decay
            score *= r["decay_factor"]
            # Boost if goal_pattern matches stored goal_types
            if goal_pattern:
                stored_goals = json.loads(r["goal_types"] or "[]")
                if any(goal_pattern in g or g in goal_pattern
                       for g in stored_goals):
                    score += 2.0

            results.append((score, r))

        results.sort(key=lambda x: -x[0])
        return [LemmaMatch(
            name=r["name"], statement=r["statement"],
            proof=r["proof"], relevance_score=score,
            times_cited=r["times_cited"],
        ) for score, r in results[:top_k]]

    # ═══════════════════════════════════════════════════════════
    # Layer 2: Strategy Patterns (Phase 4 placeholder)
    # ═══════════════════════════════════════════════════════════

    async def add_strategy_pattern(self, pattern: StrategyPattern) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._add_sp_sync, pattern)

    def _add_sp_sync(self, pattern: StrategyPattern) -> int:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO strategy_patterns "
                "(name, domain, problem_pattern, tactic_template, "
                "preconditions, times_applied, times_succeeded, "
                "avg_depth, confidence, source_episodes, "
                "created_at, updated_at, decay_factor) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0)",
                (pattern.name, pattern.domain, pattern.problem_pattern,
                 json.dumps(pattern.tactic_template),
                 json.dumps(pattern.preconditions),
                 pattern.times_applied, pattern.times_succeeded,
                 pattern.avg_depth, pattern.confidence,
                 json.dumps(pattern.source_episodes),
                 now, now))
            sp_id = cur.lastrowid
            self._log_change(
                conn, layer="L2", entity_type="strategy_patterns",
                entity_id=sp_id, action="create",
                new_value=f"{pattern.name}: {' → '.join(pattern.tactic_template[:4])}",
                reason=f"domain={pattern.domain}")
            return sp_id

    async def query_strategy_patterns(
            self, domain: str = "", top_k: int = 5
    ) -> list[StrategyPattern]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._query_sp_sync, domain, top_k)

    def _query_sp_sync(self, domain, top_k):
        conditions = ["1=1"]
        params: list = []
        if domain:
            conditions.append("domain=?")
            params.append(domain)
        where = " AND ".join(conditions)

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM strategy_patterns "
                f"WHERE {where} "
                f"ORDER BY (confidence * decay_factor) DESC LIMIT ?",
                (*params, top_k)).fetchall()
            return [StrategyPattern(
                id=r["id"], name=r["name"], domain=r["domain"],
                problem_pattern=r["problem_pattern"],
                tactic_template=json.loads(r["tactic_template"]),
                preconditions=json.loads(r["preconditions"]),
                times_applied=r["times_applied"],
                times_succeeded=r["times_succeeded"],
                avg_depth=r["avg_depth"],
                confidence=r["confidence"],
                source_episodes=json.loads(r["source_episodes"]),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                decay_factor=r["decay_factor"],
            ) for r in rows]

    # ═══════════════════════════════════════════════════════════
    # Layer 3: Concept Graph (Phase 5 placeholder)
    # ═══════════════════════════════════════════════════════════

    async def upsert_concept(self, name: str,
                              domain: str = "",
                              description: str = "") -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._upsert_concept_sync, name, domain, description)

    def _upsert_concept_sync(self, name, domain, description):
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM concept_nodes WHERE name=?",
                (name,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE concept_nodes SET encounter_count = "
                    "encounter_count + 1 WHERE id=?",
                    (row["id"],))
                return row["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO concept_nodes "
                    "(name, domain, description, difficulty_est, "
                    "encounter_count, created_at) "
                    "VALUES (?, ?, ?, 0.5, 1, ?)",
                    (name, domain, description, now))
                return cur.lastrowid

    async def add_concept_edge(self, source_name: str,
                                target_name: str,
                                relation_type: str,
                                weight: float = 1.0) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._add_edge_sync,
            source_name, target_name, relation_type, weight)

    def _add_edge_sync(self, source_name, target_name, relation, weight):
        now = time.time()
        with self._connect() as conn:
            src = conn.execute(
                "SELECT id FROM concept_nodes WHERE name=?",
                (source_name,)).fetchone()
            tgt = conn.execute(
                "SELECT id FROM concept_nodes WHERE name=?",
                (target_name,)).fetchone()
            if not src or not tgt:
                return

            existing = conn.execute(
                "SELECT id, evidence_count FROM concept_edges "
                "WHERE source_id=? AND target_id=? AND relation_type=?",
                (src["id"], tgt["id"], relation)).fetchone()

            if existing:
                conn.execute(
                    "UPDATE concept_edges SET "
                    "evidence_count = evidence_count + 1, "
                    "weight=? WHERE id=?",
                    (weight, existing["id"]))
            else:
                conn.execute(
                    "INSERT INTO concept_edges "
                    "(source_id, target_id, relation_type, weight, "
                    "evidence_count, created_at) "
                    "VALUES (?, ?, ?, ?, 1, ?)",
                    (src["id"], tgt["id"], relation, weight, now))

    # ═══════════════════════════════════════════════════════════
    # Changelog (audit trail)
    # ═══════════════════════════════════════════════════════════

    def _log_change(self, conn, layer: str, entity_type: str,
                    entity_id: int, action: str,
                    old_value: str = "", new_value: str = "",
                    reason: str = ""):
        conn.execute(
            "INSERT INTO knowledge_changelog "
            "(layer, entity_type, entity_id, action, "
            "old_value, new_value, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (layer, entity_type, entity_id, action,
             old_value, new_value, reason, time.time()))

    # ═══════════════════════════════════════════════════════════
    # Stats
    # ═══════════════════════════════════════════════════════════

    async def knowledge_stats(self) -> dict:
        """统一知识库统计 (含 Layer 0 stats)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._knowledge_stats_sync)

    def _knowledge_stats_sync(self) -> dict:
        base = self._stats_sync()  # Layer 0 stats from parent

        with self._connect() as conn:
            def _count(table, where=""):
                q = f"SELECT COUNT(*) as c FROM {table}"
                if where:
                    q += f" WHERE {where}"
                return conn.execute(q).fetchone()["c"]

            base.update({
                # Layer 1
                "tactic_patterns": _count("tactic_effectiveness"),
                "error_patterns": _count("error_patterns"),
                "proved_lemmas": _count("proved_lemmas"),
                "verified_lemmas": _count("proved_lemmas", "verified=1"),
                # Layer 2
                "strategy_patterns": _count("strategy_patterns"),
                # Layer 3
                "concept_nodes": _count("concept_nodes"),
                "concept_edges": _count("concept_edges"),
                # Changelog
                "changelog_entries": _count("knowledge_changelog"),
                # Episodes (was JSONL)
                "episodes": _count("episodes"),
                # Persistent knowledge (was JSON)
                "pk_failure_patterns": _count("pk_failures"),
                "pk_success_patterns": _count("pk_successes"),
            })

        return base

    # ═══════════════════════════════════════════════════════════
    # Episodic Memory (replaces agent/memory/episodic_memory.py JSONL)
    # ═══════════════════════════════════════════════════════════

    async def add_episode(
            self, problem_type: str, difficulty: str,
            winning_strategy: str, key_tactics: list[str],
            key_insight: str, solve_time_ms: int) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._add_episode_sync,
            problem_type, difficulty, winning_strategy,
            key_tactics, key_insight, solve_time_ms)

    def _add_episode_sync(self, problem_type, difficulty,
                           winning_strategy, key_tactics,
                           key_insight, solve_time_ms) -> int:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO episodes "
                "(problem_type, difficulty, winning_strategy, "
                "key_tactics, key_insight, solve_time_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (problem_type, difficulty, winning_strategy,
                 json.dumps(key_tactics), key_insight,
                 solve_time_ms, now))
            return cur.lastrowid

    async def query_episodes(
            self, problem_type: str = "", difficulty: str = "",
            top_k: int = 5) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._query_episodes_sync,
            problem_type, difficulty, top_k)

    def _query_episodes_sync(self, problem_type, difficulty, top_k):
        with self._connect() as conn:
            conditions = ["1=1"]
            params: list = []
            if problem_type:
                conditions.append("problem_type=?")
                params.append(problem_type)
            if difficulty:
                conditions.append("difficulty=?")
                params.append(difficulty)
            where = " AND ".join(conditions)
            rows = conn.execute(
                f"SELECT * FROM episodes WHERE {where} "
                f"ORDER BY created_at DESC LIMIT ?",
                (*params, top_k)).fetchall()
            return [{
                "id": r["id"],
                "problem_type": r["problem_type"],
                "difficulty": r["difficulty"],
                "winning_strategy": r["winning_strategy"],
                "key_tactics": json.loads(r["key_tactics"] or "[]"),
                "key_insight": r["key_insight"],
                "solve_time_ms": r["solve_time_ms"],
                "created_at": r["created_at"],
            } for r in rows]

    # ═══════════════════════════════════════════════════════════
    # Persistent Knowledge (replaces knowledge_base.json)
    # ═══════════════════════════════════════════════════════════

    async def record_failure(
            self, tactic: str, goal_type: str = "",
            error_category: str = "", domain: str = "") -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._record_failure_sync,
            tactic, goal_type or error_category or "unknown", domain)

    def _record_failure_sync(self, tactic, goal_type, domain):
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, count FROM pk_failures "
                "WHERE tactic=? AND goal_type=?",
                (tactic, goal_type[:80])).fetchone()
            if row:
                conn.execute(
                    "UPDATE pk_failures SET count=?, last_seen=? WHERE id=?",
                    (row["count"] + 1, now, row["id"]))
            else:
                conn.execute(
                    "INSERT INTO pk_failures "
                    "(tactic, goal_type, domain, count, last_seen) "
                    "VALUES (?, ?, ?, 1, ?)",
                    (tactic, goal_type[:80], domain, now))

    async def record_success(
            self, domain: str, tactics: list[str],
            theorem_type: str = "") -> None:
        loop = asyncio.get_event_loop()
        combo = " → ".join(tactics[:5])
        await loop.run_in_executor(
            None, self._record_success_sync, domain or "general", combo)

    def _record_success_sync(self, domain, combo):
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, count FROM pk_successes "
                "WHERE domain=? AND tactic_combo=?",
                (domain, combo)).fetchone()
            if row:
                conn.execute(
                    "UPDATE pk_successes SET count=?, last_seen=? WHERE id=?",
                    (row["count"] + 1, now, row["id"]))
            else:
                conn.execute(
                    "INSERT INTO pk_successes "
                    "(domain, tactic_combo, count, last_seen) "
                    "VALUES (?, ?, 1, ?)",
                    (domain, combo, now))

    async def get_suggestions(
            self, domain: str = "", goal_type: str = "",
            max_items: int = 5) -> list[str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._get_suggestions_sync, domain, goal_type, max_items)

    def _get_suggestions_sync(self, domain, goal_type, max_items):
        suggestions = []
        with self._connect() as conn:
            if domain:
                rows = conn.execute(
                    "SELECT tactic_combo, count FROM pk_successes "
                    "WHERE domain=? ORDER BY count DESC LIMIT ?",
                    (domain, max_items)).fetchall()
                for r in rows:
                    suggestions.append(
                        f"Proven effective ({r['count']}x): {r['tactic_combo']}")

            rows = conn.execute(
                "SELECT tactic, goal_type, count FROM pk_failures "
                "WHERE count >= 3 ORDER BY count DESC LIMIT ?",
                (max_items,)).fetchall()
            for r in rows:
                if not goal_type or goal_type in r["goal_type"]:
                    suggestions.append(
                        f"AVOID `{r['tactic']}` on {r['goal_type']} "
                        f"(failed {r['count']}x)")
        return suggestions[:max_items]

    # ═══════════════════════════════════════════════════════════
    # RL-aligned: Reward-based reinforcement
    # ═══════════════════════════════════════════════════════════

    async def reinforce(self, entity_type: str, entity_id: int,
                        reward: float) -> None:
        """Reinforce or weaken a knowledge entry based on RL reward.

        Instead of pure time-based decay, this allows the RL training
        loop to strengthen knowledge that helped produce successful
        proofs and weaken knowledge that led to failures.

        Args:
            entity_type: "tactic", "lemma", "strategy", "error_pattern"
            entity_id: Row ID in the corresponding table
            reward: Positive = strengthen, negative = weaken.
                    Clamped to [-1, 1], applied as multiplicative
                    adjustment to decay_factor.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._reinforce_sync, entity_type, entity_id, reward)

    def _reinforce_sync(self, entity_type, entity_id, reward):
        table_map = {
            "tactic": "tactic_effectiveness",
            "lemma": "proved_lemmas",
            "strategy": "strategy_patterns",
        }
        table = table_map.get(entity_type)
        if not table:
            return

        # Clamp reward to [-1, 1]
        reward = max(-1.0, min(1.0, reward))
        # Convert to multiplicative factor: reward=1 → 1.2x, reward=-1 → 0.8x
        factor = 1.0 + reward * 0.2

        with self._connect() as conn:
            row = conn.execute(
                f"SELECT id, decay_factor FROM {table} WHERE id=?",
                (entity_id,)).fetchone()
            if row:
                new_decay = max(0.01, min(2.0, row["decay_factor"] * factor))
                conn.execute(
                    f"UPDATE {table} SET decay_factor=? WHERE id=?",
                    (new_decay, entity_id))
                self._log_change(
                    conn, layer="RL", entity_type=table,
                    entity_id=entity_id, action="reinforce",
                    old_value=f"decay={row['decay_factor']:.3f}",
                    new_value=f"decay={new_decay:.3f}",
                    reason=f"reward={reward:.3f}")

    # ═══════════════════════════════════════════════════════════
    # RL-aligned: Batch trajectory export
    # ═══════════════════════════════════════════════════════════

    async def export_trajectories_batch(
            self, min_depth: int = 1, limit: int = 10000,
            format: str = "dict") -> list[dict]:
        """Export proof trajectories for RL training.

        Returns flat dicts suitable for conversion to DataFrame/Arrow.

        Args:
            min_depth: Minimum tactic sequence length
            limit: Max rows to return
            format: "dict" (default) or "flat" (flattened for tabular)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._export_batch_sync, min_depth, limit, format)

    def _export_batch_sync(self, min_depth, limit, format):
        with self._connect() as conn:
            # Check if proof_traces table exists (from parent ProofContextStore)
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='proof_traces'").fetchone()
            if not tables:
                return []

            try:
                rows = conn.execute(
                    "SELECT pt.*, pc.theorem, pc.solved "
                    "FROM proof_traces pt "
                    "JOIN proof_contexts pc ON pt.context_id = pc.id "
                    "WHERE json_array_length(pt.tactic_sequence_json) >= ? "
                    "ORDER BY pt.created_at DESC LIMIT ?",
                    (min_depth, limit)).fetchall()
            except Exception:
                return []

            results = []
            for r in rows:
                try:
                    tactics = json.loads(r["tactic_sequence_json"] or "[]")
                except (json.JSONDecodeError, TypeError):
                    tactics = []

                if format == "flat":
                    # Flatten for tabular: one row per tactic step
                    for step_idx, step in enumerate(tactics):
                        results.append({
                            "trace_id": r["id"],
                            "context_id": r["context_id"],
                            "theorem": r["theorem"],
                            "solved": bool(r["solved"]),
                            "step_idx": step_idx,
                            "total_steps": len(tactics),
                            "tactic": step.get("tactic", "") if isinstance(step, dict) else str(step),
                            "success": step.get("success", False) if isinstance(step, dict) else False,
                            "goal_before": step.get("goal_before", "") if isinstance(step, dict) else "",
                            "duration_ms": r["duration_ms"],
                        })
                else:
                    results.append({
                        "trace_id": r["id"],
                        "context_id": r["context_id"],
                        "theorem": r["theorem"],
                        "solved": bool(r["solved"]),
                        "tactic_sequence": tactics,
                        "num_steps": len(tactics),
                        "duration_ms": r["duration_ms"],
                        "success": bool(r["success"]),
                    })
            return results

    def export_to_parquet(self, path: str, min_depth: int = 1,
                          limit: int = 50000) -> int:
        """Export trajectories to Parquet file for RL framework consumption.

        Requires pyarrow. Returns number of rows written.
        """
        rows = self._export_batch_sync(min_depth, limit, format="flat")
        if not rows:
            return 0

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist(rows)
            pq.write_table(table, path, compression="snappy")
            logger.info(f"Exported {len(rows)} trajectory rows to {path}")
            return len(rows)
        except ImportError:
            # Fallback: write as JSONL
            import os
            jsonl_path = path.replace(".parquet", ".jsonl")
            os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
            with open(jsonl_path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            logger.info(
                f"pyarrow not available; exported {len(rows)} rows "
                f"to {jsonl_path} instead")
            return len(rows)

    # ═══════════════════════════════════════════════════════════
    # Unified decay & GC (for KnowledgeEvolver)
    # ═══════════════════════════════════════════════════════════

    async def decay_all(self, decay_rate: float = 0.95,
                        min_samples: int = 3) -> dict:
        """Apply time-based decay to all knowledge layers.

        Entries with fewer than min_samples observations are protected.
        Returns stats about how many rows were decayed per layer.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._decay_all_sync, decay_rate, min_samples)

    def _decay_all_sync(self, decay_rate, min_samples):
        stats = {}
        with self._connect() as conn:
            # L1: tactic_effectiveness
            r = conn.execute(
                "UPDATE tactic_effectiveness "
                "SET decay_factor = decay_factor * ? "
                "WHERE (successes + failures) >= ?",
                (decay_rate, min_samples))
            stats["tactics_decayed"] = r.rowcount

            # L1: proved_lemmas
            r = conn.execute(
                "UPDATE proved_lemmas "
                "SET decay_factor = decay_factor * ? "
                "WHERE times_cited >= ?",
                (decay_rate, min_samples))
            stats["lemmas_decayed"] = r.rowcount

            # L2: strategy_patterns
            r = conn.execute(
                "UPDATE strategy_patterns "
                "SET decay_factor = decay_factor * ? "
                "WHERE times_applied >= ?",
                (decay_rate, min_samples))
            stats["strategies_decayed"] = r.rowcount

        return stats

    async def gc_stale(self, threshold: float = 0.1) -> dict:
        """Mark entries with decay_factor below threshold as stale."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._gc_stale_sync, threshold)

    def _gc_stale_sync(self, threshold):
        stats = {}
        with self._connect() as conn:
            r = conn.execute(
                "UPDATE proved_lemmas SET stale=1 "
                "WHERE decay_factor < ? AND stale=0",
                (threshold,))
            stats["lemmas_staled"] = r.rowcount

            # For tactics/strategies: delete low-value entries
            r = conn.execute(
                "DELETE FROM tactic_effectiveness "
                "WHERE decay_factor < ? AND (successes + failures) < 5",
                (threshold,))
            stats["tactics_removed"] = r.rowcount

            r = conn.execute(
                "DELETE FROM strategy_patterns "
                "WHERE decay_factor < ? AND times_applied < 3",
                (threshold,))
            stats["strategies_removed"] = r.rowcount

        return stats

    def close(self):
        """Close the database connection."""
        if hasattr(self, '_shared_conn') and self._shared_conn:
            try:
                self._shared_conn.close()
            except Exception:
                pass
            self._shared_conn = None
