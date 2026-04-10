"""agent/persistence/session_store.py — Persistent session storage

Inspired by Claude Code's sessionStorage.ts:
  - Save/load complete proof session state
  - Indexed by session_id with timestamps
  - Supports multiple storage backends (file, SQLite)
  - Atomic writes with crash recovery

Usage::

    store = FileSessionStore(base_dir="./sessions")

    # Save session
    session_id = store.save(SessionData(
        theorem="theorem foo ...",
        messages=[...],
        session_memory=memory.to_dict(),
        proof_state={...},
    ))

    # Resume later
    data = store.load(session_id)

    # List recent sessions
    sessions = store.list_sessions(limit=10)
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Complete state of a proof session."""
    session_id: str = ""
    # Problem
    theorem: str = ""
    problem_id: str = ""
    benchmark: str = ""
    # Conversation
    messages: list[dict] = field(default_factory=list)
    # Agent state
    session_memory: dict = field(default_factory=dict)
    working_memory: dict = field(default_factory=dict)
    # Proof state
    best_proof: str = ""
    proof_verified: bool = False
    green_level: str = "NONE"
    # Statistics
    total_tokens: int = 0
    total_turns: int = 0
    total_latency_ms: int = 0
    tools_used: list[str] = field(default_factory=list)
    # Metadata
    model: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "in_progress"  # "in_progress", "succeeded", "failed", "paused"
    config: dict = field(default_factory=dict)
    # Events (for replay)
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.session_id:
            self.session_id = f"sess_{uuid.uuid4().hex[:12]}"


class SessionStore:
    """Abstract base for session storage backends."""

    def save(self, data: SessionData) -> str:
        raise NotImplementedError

    def load(self, session_id: str) -> Optional[SessionData]:
        raise NotImplementedError

    def list_sessions(self, limit: int = 20, status: str = None) -> list[dict]:
        raise NotImplementedError

    def delete(self, session_id: str) -> bool:
        raise NotImplementedError

    def update(self, session_id: str, **fields) -> bool:
        raise NotImplementedError


class FileSessionStore(SessionStore):
    """File-based session storage.

    Each session is stored as a JSON file in the base directory.
    An index file tracks all sessions for fast listing.
    """

    def __init__(self, base_dir: str = "./sessions"):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._base_dir / "_index.json"
        self._index = self._load_index()

    def save(self, data: SessionData) -> str:
        """Save session data to disk. Returns session_id."""
        data.updated_at = time.time()
        session_id = data.session_id

        # Write session file atomically
        session_path = self._session_path(session_id)
        tmp_path = session_path.with_suffix(".tmp")

        try:
            serializable = self._to_serializable(data)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            tmp_path.rename(session_path)
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        # Update index
        self._index[session_id] = {
            "session_id": session_id,
            "problem_id": data.problem_id,
            "theorem": data.theorem[:100],
            "status": data.status,
            "created_at": data.created_at,
            "updated_at": data.updated_at,
            "total_tokens": data.total_tokens,
            "model": data.model,
        }
        self._save_index()

        logger.info(f"Session saved: {session_id} ({data.status})")
        return session_id

    def load(self, session_id: str) -> Optional[SessionData]:
        """Load session data from disk."""
        session_path = self._session_path(session_id)
        if not session_path.exists():
            logger.warning(f"Session not found: {session_id}")
            return None

        try:
            with open(session_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return self._from_serializable(raw)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def list_sessions(
        self,
        limit: int = 20,
        status: str = None,
    ) -> list[dict]:
        """List recent sessions, newest first."""
        entries = list(self._index.values())

        if status:
            entries = [e for e in entries if e.get("status") == status]

        entries.sort(key=lambda e: e.get("updated_at", 0), reverse=True)
        return entries[:limit]

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
        self._index.pop(session_id, None)
        self._save_index()
        return True

    def update(self, session_id: str, **fields) -> bool:
        """Update specific fields of a session."""
        data = self.load(session_id)
        if not data:
            return False
        for key, value in fields.items():
            if hasattr(data, key):
                setattr(data, key, value)
        self.save(data)
        return True

    def find_by_theorem(self, theorem_prefix: str) -> list[dict]:
        """Find sessions by theorem prefix."""
        results = []
        for entry in self._index.values():
            if theorem_prefix.lower() in entry.get("theorem", "").lower():
                results.append(entry)
        return results

    # ── Internal ──────────────────────────────────────────────────────────

    def _session_path(self, session_id: str) -> Path:
        return self._base_dir / f"{session_id}.json"

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self):
        try:
            tmp = self._index_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(self._index, f, indent=2)
            tmp.rename(self._index_path)
        except Exception as e:
            logger.error(f"Failed to save session index: {e}")

    @staticmethod
    def _to_serializable(data: SessionData) -> dict:
        """Convert SessionData to a JSON-serializable dict."""
        d = {}
        for fld in data.__dataclass_fields__:
            val = getattr(data, fld)
            d[fld] = val
        return d

    @staticmethod
    def _from_serializable(raw: dict) -> SessionData:
        """Convert dict back to SessionData."""
        known_fields = set(SessionData.__dataclass_fields__.keys())
        filtered = {k: v for k, v in raw.items() if k in known_fields}
        return SessionData(**filtered)
