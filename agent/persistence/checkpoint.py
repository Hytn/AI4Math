"""agent/persistence/checkpoint.py — Automatic state checkpointing

Saves proof session state at key milestones so that interrupted sessions
can resume from the latest checkpoint instead of starting over.

Checkpoint triggers:
  - After each successful verification (L1 or L2 pass)
  - After each strategy switch
  - After discovering a useful lemma
  - Periodically (every N turns)

Usage::

    cp = CheckpointManager(store=FileSessionStore("./sessions"))

    # Auto-checkpoint in the proof loop
    cp.maybe_checkpoint(session_data, trigger="verification_pass")

    # Resume from latest checkpoint
    data = cp.resume_latest(problem_id="nat_add_comm")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from agent.persistence.session_store import SessionStore, SessionData

logger = logging.getLogger(__name__)


@dataclass
class CheckpointPolicy:
    """When to create checkpoints."""
    on_verification_pass: bool = True     # After L1/L2 verification succeeds
    on_strategy_switch: bool = True       # After policy engine switches strategy
    on_lemma_discovered: bool = True      # After a useful lemma is banked
    periodic_turns: int = 5               # Checkpoint every N turns (0 = disabled)
    periodic_seconds: float = 60.0        # Checkpoint every N seconds (0 = disabled)
    max_checkpoints_per_session: int = 20 # Limit total checkpoints


class CheckpointManager:
    """Manages automatic checkpointing of proof sessions."""

    def __init__(
        self,
        store: SessionStore,
        policy: CheckpointPolicy = None,
    ):
        self._store = store
        self._policy = policy or CheckpointPolicy()
        self._last_checkpoint_time: float = 0
        self._last_checkpoint_turn: int = 0
        self._checkpoint_count: int = 0

    def maybe_checkpoint(
        self,
        session: SessionData,
        trigger: str = "",
        turn: int = 0,
        force: bool = False,
    ) -> bool:
        """Conditionally create a checkpoint based on policy.

        Args:
            session: Current session state
            trigger: What triggered this check
            turn: Current turn number
            force: Force checkpoint regardless of policy

        Returns:
            True if checkpoint was created
        """
        if not force and not self._should_checkpoint(trigger, turn):
            return False

        if (self._checkpoint_count >= self._policy.max_checkpoints_per_session
                and not force):
            return False

        return self._do_checkpoint(session, trigger)

    def _should_checkpoint(self, trigger: str, turn: int) -> bool:
        """Check if we should create a checkpoint now."""
        p = self._policy

        if trigger == "verification_pass" and p.on_verification_pass:
            return True
        if trigger == "strategy_switch" and p.on_strategy_switch:
            return True
        if trigger == "lemma_discovered" and p.on_lemma_discovered:
            return True

        # Periodic by turns
        if (p.periodic_turns > 0
                and turn - self._last_checkpoint_turn >= p.periodic_turns):
            return True

        # Periodic by time
        if (p.periodic_seconds > 0
                and time.time() - self._last_checkpoint_time >= p.periodic_seconds):
            return True

        return False

    def _do_checkpoint(self, session: SessionData, trigger: str) -> bool:
        """Create the checkpoint."""
        try:
            session.updated_at = time.time()
            # Add checkpoint event
            session.events.append({
                "type": "checkpoint",
                "trigger": trigger,
                "timestamp": time.time(),
                "turn": session.total_turns,
                "tokens": session.total_tokens,
                "status": session.status,
                "best_proof_len": len(session.best_proof),
            })

            self._store.save(session)
            self._last_checkpoint_time = time.time()
            self._last_checkpoint_turn = session.total_turns
            self._checkpoint_count += 1

            logger.debug(
                f"Checkpoint #{self._checkpoint_count} for {session.session_id} "
                f"(trigger={trigger})")
            return True

        except Exception as e:
            logger.error(f"Checkpoint failed: {e}")
            return False

    def resume_latest(
        self,
        problem_id: str = "",
        session_id: str = "",
    ) -> Optional[SessionData]:
        """Resume from the latest checkpoint.

        Args:
            problem_id: Find latest session for this problem
            session_id: Resume a specific session

        Returns:
            SessionData if found, None otherwise
        """
        if session_id:
            data = self._store.load(session_id)
            if data:
                logger.info(
                    f"Resuming session {session_id} from checkpoint "
                    f"(turn {data.total_turns}, tokens {data.total_tokens})")
                data.status = "in_progress"
                data.events.append({
                    "type": "resumed",
                    "timestamp": time.time(),
                })
            return data

        if problem_id:
            sessions = self._store.list_sessions(limit=50)
            for entry in sessions:
                if entry.get("problem_id") == problem_id:
                    return self.resume_latest(
                        session_id=entry["session_id"])

        # Resume most recent in-progress session
        sessions = self._store.list_sessions(limit=1, status="in_progress")
        if sessions:
            return self.resume_latest(session_id=sessions[0]["session_id"])

        return None

    def finalize(self, session: SessionData, success: bool):
        """Mark session as completed and save final state."""
        session.status = "succeeded" if success else "failed"
        session.updated_at = time.time()
        session.events.append({
            "type": "finalized",
            "success": success,
            "timestamp": time.time(),
            "total_tokens": session.total_tokens,
            "total_turns": session.total_turns,
        })
        self._store.save(session)
        logger.info(
            f"Session {session.session_id} finalized: "
            f"{'success' if success else 'failed'}")
