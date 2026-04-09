"""engine/proof_context_store.py — Persistent proof context storage

Serializes ProofSessionState to SQLite for cross-session recovery.
This enables:
  1. Resume interrupted proofs after process restart
  2. Share proof states across distributed workers
  3. Build a proof trajectory database for training world models

Storage schema:
  proof_contexts: id, theorem_hash, theorem, state_json, created_at, updated_at, solved
  env_snapshots:  id, context_id, env_id, parent_env_id, tactic, goals_json, depth
  proof_traces:   id, context_id, tactic_sequence_json, duration_ms, success, created_at

Usage::

    store = ProofContextStore("/path/to/proofs.db")

    # Save proof state
    session_state = proof_session.state
    ctx_id = await store.save(session_state)

    # Resume later
    restored = await store.load(ctx_id)
    session = ProofSession(restored, pool)

    # Query proof history
    recent = await store.list_recent(limit=50)
    solved = await store.list_solved(theorem_pattern="Nat.%")

    # Export trajectories for training
    trajectories = await store.export_trajectories(min_depth=3)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from engine.proof_session import EnvNode, ProofSessionState

logger = logging.getLogger(__name__)


# ─── Serialization helpers ───────────────────────────────────

def _serialize_state(state: ProofSessionState) -> str:
    """Serialize ProofSessionState to JSON string."""
    d = {
        "theorem": state.theorem,
        "root_env_id": state.root_env_id,
        "theorem_env_id": state.theorem_env_id,
        "current_env_id": state.current_env_id,
        "tactic_history": state.tactic_history,
        "best_depth": state.best_depth,
        "solved": state.solved,
        "proof_path": state.proof_path,
        "nodes": {
            str(k): {
                "env_id": v.env_id,
                "parent_env_id": v.parent_env_id,
                "tactic": v.tactic,
                "goals": v.goals,
                "is_proof_complete": v.is_proof_complete,
                "children": v.children,
                "created_at": v.created_at,
                "depth": v.depth,
            }
            for k, v in state.nodes.items()
        },
    }
    return json.dumps(d, ensure_ascii=False)


def _deserialize_state(json_str: str) -> ProofSessionState:
    """Deserialize JSON string to ProofSessionState."""
    d = json.loads(json_str)
    nodes = {}
    for k, v in d.get("nodes", {}).items():
        nodes[int(k)] = EnvNode(
            env_id=v["env_id"],
            parent_env_id=v["parent_env_id"],
            tactic=v["tactic"],
            goals=v.get("goals", []),
            is_proof_complete=v.get("is_proof_complete", False),
            children=v.get("children", []),
            created_at=v.get("created_at", 0.0),
            depth=v.get("depth", 0),
        )
    return ProofSessionState(
        theorem=d["theorem"],
        root_env_id=d["root_env_id"],
        theorem_env_id=d["theorem_env_id"],
        current_env_id=d["current_env_id"],
        nodes=nodes,
        tactic_history=d.get("tactic_history", []),
        best_depth=d.get("best_depth", 0),
        solved=d.get("solved", False),
        proof_path=d.get("proof_path", []),
    )


def _theorem_hash(theorem: str) -> str:
    return hashlib.sha256(theorem.strip().encode()).hexdigest()[:16]


# ─── Data classes for query results ─────────────────────────

@dataclass
class ProofContextInfo:
    """Metadata about a stored proof context."""
    context_id: int
    theorem_hash: str
    theorem: str
    solved: bool
    best_depth: int
    num_nodes: int
    num_tactics: int
    created_at: float
    updated_at: float


@dataclass
class ProofTrajectory:
    """A proof trajectory for training."""
    theorem: str
    tactic_sequence: list[str]
    success: bool
    depth: int
    duration_ms: float


@dataclass
class StepDetail:
    """Per-step detail for world model training.

    Captures the full state transition for each tactic application:
      (env_before, goals_before) --tactic--> (env_after, goals_after, error?)
    """
    step_index: int
    tactic: str
    env_id_before: int
    env_id_after: int              # -1 if tactic failed
    goals_before: list[str]
    goals_after: list[str]
    error_message: str = ""
    error_category: str = ""
    elapsed_ms: float = 0.0
    is_proof_complete: bool = False


@dataclass
class RichProofTrajectory:
    """Full-detail proof trajectory for world model training.

    Unlike ProofTrajectory (which stores only tactic names),
    this captures per-step goal states, env_id transitions,
    and error information — everything needed to learn the
    state-transition dynamics of the Lean4 proof environment.
    """
    theorem: str
    steps: list[StepDetail]
    success: bool
    depth: int
    duration_ms: float
    # Optional metadata
    theorem_hash: str = ""
    context_id: int = 0


# ─── Store implementation ────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proof_contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    theorem_hash TEXT NOT NULL,
    theorem TEXT NOT NULL,
    state_json TEXT NOT NULL,
    best_depth INTEGER DEFAULT 0,
    num_nodes INTEGER DEFAULT 0,
    num_tactics INTEGER DEFAULT 0,
    solved INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pc_theorem_hash ON proof_contexts(theorem_hash);
CREATE INDEX IF NOT EXISTS idx_pc_solved ON proof_contexts(solved);
CREATE INDEX IF NOT EXISTS idx_pc_updated ON proof_contexts(updated_at);

CREATE TABLE IF NOT EXISTS proof_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id INTEGER NOT NULL,
    tactic_sequence TEXT NOT NULL,
    step_details TEXT DEFAULT '[]',
    depth INTEGER DEFAULT 0,
    duration_ms REAL DEFAULT 0.0,
    success INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    FOREIGN KEY (context_id) REFERENCES proof_contexts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pt_context ON proof_traces(context_id);
CREATE INDEX IF NOT EXISTS idx_pt_success ON proof_traces(success);
"""

_MIGRATION_V2 = """
ALTER TABLE proof_traces ADD COLUMN step_details TEXT DEFAULT '[]';
"""


class ProofContextStore:
    """SQLite-backed persistent proof context storage.

    Thread-safe: uses WAL mode for file DBs, shared connection for :memory:.
    For async usage, all I/O is offloaded to a thread executor.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._initialized = False
        self._lock = asyncio.Lock()
        self._is_memory = (db_path == ":memory:")

        # Ensure directory exists for file-based databases
        if not self._is_memory:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # For :memory:, keep a persistent connection (with cross-thread access)
        if self._is_memory:
            self._shared_conn = sqlite3.connect(
                ":memory:", check_same_thread=False)
            self._shared_conn.execute("PRAGMA foreign_keys=ON")
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.executescript(_SCHEMA)
            self._shared_conn.commit()
        else:
            self._shared_conn = None
            self._init_schema()

    @contextmanager
    def _connect(self):
        if self._is_memory:
            # Reuse shared connection for :memory:
            try:
                yield self._shared_conn
                self._shared_conn.commit()
            except Exception:
                self._shared_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Auto-migrate: add step_details column if missing
            self._migrate_v2(conn)
        self._initialized = True

    def _migrate_v2(self, conn):
        """Add step_details column to proof_traces if it doesn't exist."""
        try:
            cols = [row[1] for row in
                    conn.execute("PRAGMA table_info(proof_traces)").fetchall()]
            if "step_details" not in cols:
                conn.execute(
                    "ALTER TABLE proof_traces "
                    "ADD COLUMN step_details TEXT DEFAULT '[]'")
                logger.info("ProofContextStore: migrated to v2 "
                            "(added step_details column)")
        except Exception as e:
            logger.warning(f"ProofContextStore: v2 migration skipped: {e}")

    # ─── Core CRUD ──────────────────────────────────────────

    async def save(self, state: ProofSessionState,
                   context_id: Optional[int] = None) -> int:
        """Save or update a proof context. Returns context_id."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._save_sync, state, context_id)

    def _save_sync(self, state: ProofSessionState,
                   context_id: Optional[int]) -> int:
        now = time.time()
        state_json = _serialize_state(state)
        th = _theorem_hash(state.theorem)

        with self._connect() as conn:
            if context_id is not None:
                conn.execute(
                    "UPDATE proof_contexts SET "
                    "state_json=?, best_depth=?, num_nodes=?, num_tactics=?, "
                    "solved=?, updated_at=? WHERE id=?",
                    (state_json, state.best_depth, len(state.nodes),
                     len(state.tactic_history), int(state.solved),
                     now, context_id))
                return context_id
            else:
                cur = conn.execute(
                    "INSERT INTO proof_contexts "
                    "(theorem_hash, theorem, state_json, best_depth, "
                    "num_nodes, num_tactics, solved, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (th, state.theorem, state_json, state.best_depth,
                     len(state.nodes), len(state.tactic_history),
                     int(state.solved), now, now))
                return cur.lastrowid

    async def load(self, context_id: int) -> Optional[ProofSessionState]:
        """Load a proof context by ID."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._load_sync, context_id)

    def _load_sync(self, context_id: int) -> Optional[ProofSessionState]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM proof_contexts WHERE id=?",
                (context_id,)).fetchone()
            if row:
                return _deserialize_state(row["state_json"])
        return None

    async def load_by_theorem(self, theorem: str,
                              latest: bool = True) -> Optional[ProofSessionState]:
        """Load proof context by theorem text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._load_by_theorem_sync, theorem, latest)

    def _load_by_theorem_sync(self, theorem: str,
                              latest: bool) -> Optional[ProofSessionState]:
        th = _theorem_hash(theorem)
        order = "DESC" if latest else "ASC"
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT state_json FROM proof_contexts "
                f"WHERE theorem_hash=? ORDER BY updated_at {order} LIMIT 1",
                (th,)).fetchone()
            if row:
                return _deserialize_state(row["state_json"])
        return None

    async def delete(self, context_id: int) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._delete_sync, context_id)

    def _delete_sync(self, context_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM proof_contexts WHERE id=?", (context_id,))
            return cur.rowcount > 0

    # ─── Trace recording ────────────────────────────────────

    async def record_trace(self, context_id: int,
                           tactic_sequence: list[str],
                           success: bool,
                           depth: int = 0,
                           duration_ms: float = 0.0) -> int:
        """Record a proof attempt trajectory."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._record_trace_sync,
            context_id, tactic_sequence, success, depth, duration_ms)

    def _record_trace_sync(self, context_id: int,
                           tactic_sequence: list[str],
                           success: bool, depth: int,
                           duration_ms: float) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO proof_traces "
                "(context_id, tactic_sequence, depth, duration_ms, "
                "success, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (context_id, json.dumps(tactic_sequence),
                 depth, duration_ms, int(success), time.time()))
            return cur.lastrowid

    async def record_rich_trace(self, context_id: int,
                                steps: list[StepDetail],
                                success: bool,
                                duration_ms: float = 0.0) -> int:
        """Record a proof trajectory with per-step state transition details.

        This is the preferred method for recording trajectories destined
        for world model training. Each StepDetail captures the full
        (state_before, action, state_after) transition tuple.

        Args:
            context_id: ID from save().
            steps: Per-step details including goals and env_id transitions.
            success: Whether the proof completed.
            duration_ms: Total wall-clock time.

        Returns:
            Trace ID.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._record_rich_trace_sync,
            context_id, steps, success, duration_ms)

    def _record_rich_trace_sync(self, context_id: int,
                                steps: list[StepDetail],
                                success: bool,
                                duration_ms: float) -> int:
        tactic_seq = [s.tactic for s in steps]
        details_json = json.dumps([
            {
                "i": s.step_index,
                "tac": s.tactic,
                "env_before": s.env_id_before,
                "env_after": s.env_id_after,
                "goals_before": s.goals_before,
                "goals_after": s.goals_after,
                "error": s.error_message,
                "error_cat": s.error_category,
                "ms": s.elapsed_ms,
                "complete": s.is_proof_complete,
            }
            for s in steps
        ], ensure_ascii=False)
        depth = len(steps)

        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO proof_traces "
                "(context_id, tactic_sequence, step_details, depth, "
                "duration_ms, success, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (context_id, json.dumps(tactic_seq), details_json,
                 depth, duration_ms, int(success), time.time()))
            return cur.lastrowid

    async def export_rich_trajectories(
            self, min_depth: int = 0,
            success_only: bool = False,
            limit: int = 10000) -> list[RichProofTrajectory]:
        """Export trajectories with full per-step details for world model training.

        Only returns traces that have step_details populated (recorded via
        record_rich_trace). Traces recorded via the legacy record_trace
        method are excluded.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._export_rich_trajectories_sync,
            min_depth, success_only, limit)

    def _export_rich_trajectories_sync(
            self, min_depth: int, success_only: bool,
            limit: int) -> list[RichProofTrajectory]:
        conditions = ["pt.depth >= ?",
                       "pt.step_details != '[]'"]
        params: list = [min_depth]
        if success_only:
            conditions.append("pt.success = 1")
        where = "WHERE " + " AND ".join(conditions)

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT pc.id as ctx_id, pc.theorem_hash, pc.theorem, "
                f"pt.step_details, pt.success, pt.depth, pt.duration_ms "
                f"FROM proof_traces pt "
                f"JOIN proof_contexts pc ON pt.context_id = pc.id "
                f"{where} ORDER BY pt.created_at DESC LIMIT ?",
                (*params, limit)).fetchall()

            results = []
            for r in rows:
                raw_steps = json.loads(r["step_details"])
                steps = [
                    StepDetail(
                        step_index=s["i"],
                        tactic=s["tac"],
                        env_id_before=s["env_before"],
                        env_id_after=s["env_after"],
                        goals_before=s.get("goals_before", []),
                        goals_after=s.get("goals_after", []),
                        error_message=s.get("error", ""),
                        error_category=s.get("error_cat", ""),
                        elapsed_ms=s.get("ms", 0.0),
                        is_proof_complete=s.get("complete", False),
                    )
                    for s in raw_steps
                ]
                results.append(RichProofTrajectory(
                    theorem=r["theorem"],
                    steps=steps,
                    success=bool(r["success"]),
                    depth=r["depth"],
                    duration_ms=r["duration_ms"],
                    theorem_hash=r["theorem_hash"],
                    context_id=r["ctx_id"],
                ))
            return results

    # ─── Query ──────────────────────────────────────────────

    async def list_recent(self, limit: int = 50,
                          solved_only: bool = False) -> list[ProofContextInfo]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._list_recent_sync, limit, solved_only)

    def _list_recent_sync(self, limit: int,
                          solved_only: bool) -> list[ProofContextInfo]:
        where = "WHERE solved=1" if solved_only else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, theorem_hash, theorem, solved, best_depth, "
                f"num_nodes, num_tactics, created_at, updated_at "
                f"FROM proof_contexts {where} "
                f"ORDER BY updated_at DESC LIMIT ?",
                (limit,)).fetchall()
            return [ProofContextInfo(
                context_id=r["id"],
                theorem_hash=r["theorem_hash"],
                theorem=r["theorem"],
                solved=bool(r["solved"]),
                best_depth=r["best_depth"],
                num_nodes=r["num_nodes"],
                num_tactics=r["num_tactics"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            ) for r in rows]

    async def export_trajectories(self, min_depth: int = 0,
                                  success_only: bool = False,
                                  limit: int = 10000) -> list[ProofTrajectory]:
        """Export proof trajectories for world model training."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._export_trajectories_sync,
            min_depth, success_only, limit)

    def _export_trajectories_sync(self, min_depth: int,
                                  success_only: bool,
                                  limit: int) -> list[ProofTrajectory]:
        conditions = ["pt.depth >= ?"]
        params: list = [min_depth]
        if success_only:
            conditions.append("pt.success = 1")
        where = "WHERE " + " AND ".join(conditions)

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT pc.theorem, pt.tactic_sequence, pt.success, "
                f"pt.depth, pt.duration_ms "
                f"FROM proof_traces pt "
                f"JOIN proof_contexts pc ON pt.context_id = pc.id "
                f"{where} ORDER BY pt.created_at DESC LIMIT ?",
                (*params, limit)).fetchall()
            return [ProofTrajectory(
                theorem=r["theorem"],
                tactic_sequence=json.loads(r["tactic_sequence"]),
                success=bool(r["success"]),
                depth=r["depth"],
                duration_ms=r["duration_ms"],
            ) for r in rows]

    async def stats(self) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._stats_sync)

    def _stats_sync(self) -> dict:
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as c FROM proof_contexts").fetchone()["c"]
            solved = conn.execute(
                "SELECT COUNT(*) as c FROM proof_contexts "
                "WHERE solved=1").fetchone()["c"]
            traces = conn.execute(
                "SELECT COUNT(*) as c FROM proof_traces").fetchone()["c"]
            succ_traces = conn.execute(
                "SELECT COUNT(*) as c FROM proof_traces "
                "WHERE success=1").fetchone()["c"]
            avg_depth = conn.execute(
                "SELECT COALESCE(AVG(best_depth), 0) as a "
                "FROM proof_contexts").fetchone()["a"]
            return {
                "total_contexts": total,
                "solved_contexts": solved,
                "solve_rate": solved / max(1, total),
                "total_traces": traces,
                "successful_traces": succ_traces,
                "avg_best_depth": round(avg_depth, 1),
            }
