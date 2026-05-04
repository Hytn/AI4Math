"""sampler/proof_env.py — Multi-turn proof environment

Gymnasium-style async environment that wraps AI4Math's Lean verification
engine. The RL policy acts as the LLM generating tactics; the environment
provides verification feedback and reward signals.

State machine per episode:

    RESET → AWAITING_ACTION → VERIFYING → FEEDBACK_READY
                  ↑                           │
                  └───────────────────────────┘
                        (if not terminal)

Usage::

    env = ProofEnv(config)
    await env.setup()  # start Lean pool

    obs = await env.reset(problem)
    while not done:
        obs, reward, done, info = await env.step(tactic_code)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from sampler.trajectory import (
    Trajectory, Turn, RewardInfo, TerminationReason,
)

logger = logging.getLogger(__name__)


@dataclass
class ProofEnvConfig:
    """Configuration for the proof environment."""
    # Turn / token budgets
    max_turns: int = 32
    max_tokens_per_turn: int = 2048
    max_total_tokens: int = 32768
    timeout_per_turn_s: float = 60.0
    timeout_total_s: float = 600.0

    # Lean verification
    project_dir: str = "."
    pool_size: int = 4
    preamble: str = "import Mathlib"
    lean_timeout_s: int = 30

    # ── v7: Verification backend selection ───────────────────────────
    # Picks which Lean transport backs every session in the pool.
    # ``"local"`` (default) preserves V1–V6 behaviour; the others reach
    # the community backends merged in V1–V6 but previously unreachable
    # from the RL sampler. Each pool session gets an independent
    # transport built by ``_make_transport_factory`` below.
    #
    # Supported values:
    #   "local"      — bare Lean 4 REPL subprocess (legacy default)
    #   "socket"     — Unix socket to docker/lean_daemon.py
    #   "kimina"     — Kimina Lean Server REST API (recommended for RL)
    #   "http"       — alias of "kimina"
    #   "pantograph" — Pantograph backend (mvar focus, drafting)
    #   "lookeng"    — LooKeng outer wrapper (lemma cache)
    #   "mock"       — deterministic in-process transport (tests)
    #   "auto"       — probe local→socket→http and pick the first to start
    backend: str = "local"
    backend_url: Optional[str] = None        # for kimina/http
    backend_api_key: Optional[str] = None    # for kimina/http
    backend_socket_path: Optional[str] = None  # for socket
    # When backend == "lookeng", what to use as the inner verifier.
    # None = LooKeng's default LocalTransport; "kimina" = production chain.
    backend_inner_kind: Optional[str] = None

    # Reward shaping
    reward_success: float = 1.0        # Full proof accepted
    reward_goal_closed: float = 0.1    # Per goal closed
    reward_l1_pass: float = 0.05       # Tactic passed L1 REPL check
    reward_l0_reject: float = -0.02    # Syntax / prefilter rejection
    reward_sorry: float = -0.5         # sorry detected
    reward_timeout: float = -0.1       # Turn timed out

    # Observation formatting
    include_error_feedback: bool = True
    include_fix_hints: bool = True
    max_feedback_chars: int = 1024
    include_goal_state: bool = True


class ProofEnv:
    """Async multi-turn environment for formal theorem proving.

    Wraps AsyncLeanPool + AsyncVerificationScheduler to expose a
    standard RL environment interface.

    The environment does NOT call the LLM — it only handles verification.
    The RL framework's policy generates the actions (tactics).
    """

    def __init__(self, config: ProofEnvConfig = None):
        self.config = config or ProofEnvConfig()
        self._pool = None
        self._verifier = None
        self._prefilter = None
        self._error_intel = None
        self._broadcast = None

        # Episode state
        self._problem = None
        self._turn_idx = 0
        self._trajectory: Optional[Trajectory] = None
        self._env_id: int = 0
        self._goals_remaining: int = 1
        self._accumulated_feedback: list[str] = []
        self._episode_start: float = 0.0
        self._done = False

    async def setup(self):
        """Initialize Lean pool and verification infrastructure.

        Call once before any reset/step. Safe to call multiple times.

        v7: honours ``ProofEnvConfig.backend`` — when set to anything
        other than ``"local"``, every pool session gets a transport
        built by ``_make_transport_factory`` below. This is the wire
        that finally connects the V1–V6 community backends
        (Kimina/Pantograph/LooKeng) to the RL sampler.
        """
        if self._pool is not None:
            return

        from engine.async_lean_pool import AsyncLeanPool
        from engine.async_verification_scheduler import AsyncVerificationScheduler
        from engine.prefilter import PreFilter
        from engine.error_intelligence import ErrorIntelligence
        from engine.broadcast import BroadcastBus

        self._prefilter = PreFilter()
        self._error_intel = ErrorIntelligence()
        self._broadcast = BroadcastBus()

        # v7: build a transport factory if a non-default backend is requested.
        # The factory is invoked once per session by AsyncLeanPool, so each
        # session ends up with its own transport — safe for stateful backends
        # like LocalTransport that must not be shared across sessions, and
        # equally safe for stateless REST clients like Kimina (each session
        # gets its own client object pointing at the same server).
        transport_factory = None
        if self.config.backend not in ("", "local", None):
            transport_factory = self._make_transport_factory()

        self._pool = AsyncLeanPool(
            pool_size=self.config.pool_size,
            project_dir=self.config.project_dir,
            preamble=self.config.preamble,  # set on the pool, not on start()
            timeout_seconds=self.config.lean_timeout_s,
            transport_factory=transport_factory,
        )
        await self._pool.start()

        self._verifier = AsyncVerificationScheduler(
            prefilter=self._prefilter,
            lean_pool=self._pool,
            error_intel=self._error_intel,
            broadcast=self._broadcast,
            project_dir=self.config.project_dir,
        )
        logger.info(
            "ProofEnv: setup complete, pool_size=%d backend=%s",
            self.config.pool_size, self.config.backend)

    def _make_transport_factory(self):
        """Build a session-id → REPLTransport factory for non-local backends.

        Returns a *sync* factory that constructs the right transport
        class but does NOT start it. ``AsyncLeanSession.start()`` will
        await the transport's ``start()`` itself — this is the
        documented contract on the session, and it neatly avoids the
        "nested event loop" trap (the factory is invoked from inside
        ``AsyncLeanPool.start()`` which is already on an event loop).

        Falls back to None (which AsyncLeanPool handles by constructing
        a default LocalTransport) for unrecognised backends.
        """
        cfg = self.config

        def _factory(session_id: int):
            try:
                if cfg.backend == "mock":
                    from engine.transport import MockTransport
                    return MockTransport()
                if cfg.backend == "fallback":
                    from engine.transport import FallbackTransport
                    return FallbackTransport()
                if cfg.backend in ("kimina", "http"):
                    # v11: HTTPTransport (thin delegating wrapper) was
                    # deleted; use KiminaServerBackend directly. Same
                    # interface, one less indirection.
                    from engine.backends.kimina_server import (
                        KiminaServerBackend)
                    return KiminaServerBackend(
                        base_url=cfg.backend_url,
                        api_key=cfg.backend_api_key,
                        timeout_seconds=cfg.lean_timeout_s * 5)
                if cfg.backend == "socket":
                    from engine.transport import SocketTransport
                    path = cfg.backend_socket_path or \
                        "/workspace/exchange/lean.sock"
                    return SocketTransport(
                        socket_path=path,
                        timeout_seconds=cfg.lean_timeout_s)
                if cfg.backend == "pantograph":
                    from engine.backends.pantograph import PantographBackend
                    return PantographBackend(
                        project_dir=cfg.project_dir,
                        timeout_seconds=cfg.lean_timeout_s)
                if cfg.backend == "lookeng":
                    from engine.backends.lookeng import LooKengBackend
                    # inner_kind support: if user wants the production
                    # chain (LooKeng outer → Kimina inner), build the
                    # inner transport here too. Inner is constructed
                    # un-started, same contract.
                    inner = None
                    if cfg.backend_inner_kind in ("kimina", "http"):
                        from engine.backends.kimina_server import (
                            KiminaServerBackend)
                        inner = KiminaServerBackend(
                            base_url=cfg.backend_url,
                            api_key=cfg.backend_api_key,
                            timeout_seconds=cfg.lean_timeout_s * 5)
                    return LooKengBackend(
                        inner=inner, project_dir=cfg.project_dir)
                # Unknown / "auto" / unsupported here → return None so
                # AsyncLeanPool falls back to LocalTransport. We log so
                # the operator can spot a typo.
                if cfg.backend not in ("local", "auto", "", None):
                    logger.warning(
                        "ProofEnv: unknown backend %r — falling back to local",
                        cfg.backend)
                return None
            except Exception as e:
                logger.error(
                    "ProofEnv backend %r factory(%d) raised %r — "
                    "session will fall back to LocalTransport",
                    cfg.backend, session_id, e)
                return None

        return _factory

    async def reset(self, problem: dict[str, Any]) -> str:
        """Start a new episode for the given problem.

        Args:
            problem: Dict with at least 'problem_id' and 'theorem_statement'.
                     Optionally 'header' (Lean imports/setup).

        Returns:
            Initial observation string (the theorem statement + context).
        """
        self._problem = problem
        self._turn_idx = 0
        self._done = False
        self._goals_remaining = 1
        self._accumulated_feedback = []
        self._episode_start = time.time()

        self._trajectory = Trajectory(
            problem_id=problem["problem_id"],
            theorem_statement=problem["theorem_statement"],
        )

        # Build initial observation
        obs = self._format_initial_observation(problem)

        # If pool provides env_id from header compilation, use it
        self._env_id = getattr(self._pool, "base_env_id", 0)

        return obs

    async def step(self, action: str) -> tuple[str, RewardInfo, bool, dict]:
        """Execute one turn: verify the action (tactic) in Lean.

        Args:
            action: Lean tactic code generated by the RL policy.

        Returns:
            (observation, reward_info, done, info_dict)
        """
        if self._done:
            raise RuntimeError("Episode already terminated. Call reset().")

        t0 = time.time()
        cfg = self.config

        # ── Check budgets ───────────────────────────────────────────────
        elapsed = time.time() - self._episode_start
        if elapsed > cfg.timeout_total_s:
            return self._terminate(TerminationReason.TIMEOUT, action,
                                   cfg.reward_timeout)

        if self._turn_idx >= cfg.max_turns:
            return self._terminate(TerminationReason.MAX_TURNS, action, 0.0)

        # ── Sorry check ─────────────────────────────────────────────────
        action_lower = action.lower().strip()
        if "sorry" in action_lower:
            return self._terminate(TerminationReason.SORRY_DETECTED, action,
                                   cfg.reward_sorry)

        # ── Verify via AsyncVerificationScheduler ────────────────────────
        try:
            vr = await asyncio.wait_for(
                self._verifier.verify_tactic(
                    env_id=self._env_id,
                    tactic=action,
                    goals_before=self._goals_remaining,
                ),
                timeout=cfg.timeout_per_turn_s,
            )
        except asyncio.TimeoutError:
            reward = RewardInfo(
                scalar=cfg.reward_timeout,
                verification_level="TIMEOUT",
                is_terminal=False,
                error_class="TIMEOUT",
            )
            obs = self._format_feedback("Verification timed out.", action)
            turn = Turn(
                turn_idx=self._turn_idx,
                observation=obs, action=action,
                reward=reward,
                verification_ms=int((time.time() - t0) * 1000),
            )
            self._trajectory.add_turn(turn)
            self._turn_idx += 1
            return obs, reward, False, {"verification_result": None}

        verification_ms = int((time.time() - t0) * 1000)

        # ── Build reward ─────────────────────────────────────────────────
        reward = self._compute_reward(vr, action)
        reward_info = reward

        # ── Update env state ─────────────────────────────────────────────
        if vr.success and hasattr(vr, "new_env_id") and vr.new_env_id:
            self._env_id = vr.new_env_id

        if hasattr(vr, "goals_after"):
            goals_closed = max(0, self._goals_remaining - vr.goals_after)
            self._goals_remaining = vr.goals_after
            reward_info.goals_closed = goals_closed
            reward_info.goals_remaining = vr.goals_after

        # ── Check terminal conditions ────────────────────────────────────
        done = reward_info.is_terminal

        # ── Build observation ────────────────────────────────────────────
        feedback_text = self._extract_feedback(vr)
        obs = self._format_feedback(feedback_text, action)
        self._accumulated_feedback.append(feedback_text)

        # ── Record turn ──────────────────────────────────────────────────
        turn = Turn(
            turn_idx=self._turn_idx,
            observation=obs, action=action,
            reward=reward_info,
            verification_ms=verification_ms,
        )
        self._trajectory.add_turn(turn)
        self._turn_idx += 1

        if done:
            self._done = True
            self._trajectory.wall_time_s = time.time() - self._episode_start
            if reward_info.scalar > 0:
                self._trajectory.termination = TerminationReason.SUCCESS

        info = {
            "verification_result": vr,
            "turn_idx": self._turn_idx,
            "goals_remaining": self._goals_remaining,
        }
        return obs, reward_info, done, info

    def get_trajectory(self) -> Trajectory:
        """Return the current trajectory (possibly incomplete)."""
        if self._trajectory:
            self._trajectory.wall_time_s = time.time() - self._episode_start
        return self._trajectory

    async def close(self):
        """Release Lean pool resources."""
        if self._pool:
            await self._pool.shutdown()
            self._pool = None

    # ── Internals ─────────────────────────────────────────────────────────

    def _terminate(self, reason: TerminationReason, action: str,
                   reward_scalar: float):
        """Helper for terminal states."""
        self._done = True
        reward = RewardInfo(
            scalar=reward_scalar,
            is_terminal=True,
            error_class=reason.value,
        )
        obs = f"[TERMINATED: {reason.value}]"
        turn = Turn(
            turn_idx=self._turn_idx,
            observation=obs, action=action, reward=reward,
        )
        self._trajectory.add_turn(turn)
        self._trajectory.termination = reason
        self._trajectory.wall_time_s = time.time() - self._episode_start
        return obs, reward, True, {"termination": reason.value}

    def _compute_reward(self, vr, action: str) -> RewardInfo:
        """Map a VerificationResult to a RewardInfo."""
        cfg = self.config

        # L0 rejection
        if hasattr(vr, "l0_passed") and not vr.l0_passed:
            return RewardInfo(
                scalar=cfg.reward_l0_reject,
                verification_level="L0",
                error_class=getattr(vr, "l0_reject_reason", ""),
                fix_hint=getattr(vr, "l0_fix_hint", ""),
            )

        # L1 failure
        if not vr.success:
            feedback = getattr(vr, "feedback", None)
            error_class = ""
            fix_hint = ""
            if feedback:
                error_class = getattr(feedback, "error_class", "")
                fix_hint = getattr(feedback, "suggested_fix", "")
            return RewardInfo(
                scalar=0.0,
                verification_level=getattr(vr, "level_reached", "L1"),
                error_class=error_class,
                fix_hint=fix_hint,
                raw_feedback=getattr(vr, "feedback_text",
                                     str(feedback) if feedback else ""),
            )

        # L1 passed
        level = getattr(vr, "level_reached", "L1")
        is_complete = (
            level == "L2"
            or getattr(vr, "proof_complete", False)
            or self._goals_remaining <= 0
        )

        if is_complete:
            return RewardInfo(
                scalar=cfg.reward_success,
                verification_level=level,
                is_terminal=True,
                goals_remaining=0,
            )

        return RewardInfo(
            scalar=cfg.reward_l1_pass + cfg.reward_goal_closed * max(0, getattr(vr, "goals_closed", 0)),
            verification_level=level,
            goals_remaining=getattr(vr, "goals_after", self._goals_remaining),
        )

    def _extract_feedback(self, vr) -> str:
        """Extract human-readable feedback from VerificationResult."""
        parts = []
        if hasattr(vr, "feedback") and vr.feedback:
            fb = vr.feedback
            if hasattr(fb, "lean_error") and fb.lean_error:
                parts.append(f"Lean error: {fb.lean_error}")
            if hasattr(fb, "suggested_fix") and fb.suggested_fix:
                parts.append(f"Suggestion: {fb.suggested_fix}")
        if hasattr(vr, "l0_reject_reason") and vr.l0_reject_reason:
            parts.append(f"Pre-filter rejected: {vr.l0_reject_reason}")
            if hasattr(vr, "l0_fix_hint") and vr.l0_fix_hint:
                parts.append(f"Fix: {vr.l0_fix_hint}")
        if vr.success:
            parts.append("Tactic accepted.")
            if hasattr(vr, "goals_after"):
                parts.append(f"Remaining goals: {vr.goals_after}")

        text = "\n".join(parts) if parts else "No feedback available."
        return text[:self.config.max_feedback_chars]

    def _format_initial_observation(self, problem: dict) -> str:
        """Format the initial observation for turn 0."""
        parts = [
            "Prove the following theorem in Lean 4:",
            "",
            problem["theorem_statement"],
        ]
        if problem.get("header"):
            parts.insert(0, problem["header"])
            parts.insert(1, "")
        if problem.get("context"):
            parts.extend(["", "Context:", problem["context"]])
        return "\n".join(parts)

    def _format_feedback(self, feedback: str, last_action: str) -> str:
        """Format an observation for turn > 0."""
        parts = []
        if self.config.include_error_feedback:
            parts.append(f"Feedback on your last tactic:\n{feedback}")
        if self.config.include_goal_state and self._goals_remaining > 0:
            parts.append(f"\nRemaining goals: {self._goals_remaining}")
        parts.append("\nProvide your next tactic:")
        return "\n".join(parts)
