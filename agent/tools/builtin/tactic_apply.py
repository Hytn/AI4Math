"""agent/tools/builtin/tactic_apply.py — Apply ONE tactic at the current node.

This is the step-level proving primitive that the existing toolset lacks.
It is the workhorse tool for ReProver / LeanDojo / MCTS / Best-First style
proving, where the agent advances the proof one tactic at a time.

It takes a single tactic string, applies it via the Lean4 REPL pool, and
returns the resulting goal state as a JSON observation. It transparently
updates a `search_state` object (if provided) so the outer SearchDriver
can track tree expansion.


into the KnowledgeWriter (if one is wired in), giving step-level
profiles (reprover / leandojo / mcts / best_first / beam) the same
auto-teach pipeline that whole-proof profiles already have via
Kimina's batch_verify. The deposit is best-effort — any exception
inside the writer is swallowed so a misbehaving knowledge store can
never crash the proof loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional
from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission

logger = logging.getLogger(__name__)

class TacticApplyTool(Tool):
    name = "tactic_apply"
    description = (
        "Apply EXACTLY ONE Lean 4 tactic at the current proof node and "
        "observe the resulting goal state.\n"
        "\n"
        "Returns JSON with: {success, remaining_goals, num_goals, "
        "is_proof_complete, error_message?, new_node_id}.\n"
        "\n"
        "USE THIS TOOL ONCE PER TURN. Do not chain multiple tactics in one "
        "call (no `;` or `<;>`). To compose tactics, call this tool again "
        "in the next turn with the next tactic.\n"
        "\n"
        "After a successful call, the proof state advances to a new node and "
        "the next call will operate on that new state."
    )
    permission = ToolPermission.EXTERNAL
    input_schema = {
        "type": "object",
        "properties": {
            "tactic": {
                "type": "string",
                "description": (
                    "A single Lean 4 tactic, e.g. `intro h`, `simp`, "
                    "`rw [add_comm]`, `exact rfl`. NO semicolons or `<;>`."
                ),
            },
            "node_id": {
                "type": "integer",
                "description": (
                    "Optional: the search-tree node to apply at. If omitted, "
                    "applies at the current node. Only meaningful when running "
                    "under a search driver."
                ),
            },
        },
        "required": ["tactic"],
    }

    def __init__(self, lean_pool=None, search_state=None,
                 knowledge_writer=None, world_model=None,
                 wm_min_confidence: float = 0.85):
        self._pool = lean_pool
        self._search_state = search_state  # may be None in non-search mode

        # auto-deposited (Layer 1 tactic effectiveness + error patterns).
        self._kw = knowledge_writer

        # "is this tactic likely to fail?" before sending to Lean. Only
        # high-confidence failures (≥ wm_min_confidence) are gated; we
        # never block a tactic the model is uncertain about.
        self._wm = world_model
        self._wm_min_conf = float(wm_min_confidence)
        # Per-tool monotonic step counter. Step indices in deposits are
        # only meaningful relative to a single TacticApplyTool instance,
        # which matches the natural "one tool per agent run" model.
        self._step_index = 0

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        tactic = input["tactic"].strip()
        node_id = input.get("node_id")

        if not self._pool:
            return ToolResult.error(
                "Lean REPL pool not available. tactic_apply requires a live "
                "Lean 4 environment.")

        # Decide which env_id to apply in
        env_id = self._resolve_env_id(node_id)

        # Capture goals_before from the search state (if available)
        # for the step deposit. Without a search_state we have to leave
        # this empty — Layer 1 deposit then becomes a noop (no
        # goal_pattern), which is the correct behaviour.
        goals_before = self._capture_goals_before(node_id)

        # model is highly confident will fail. Only fires when both a
        # world model is wired AND we have a goal_state to feed it.
        gate = self._wm_gate(tactic, goals_before)
        if gate is not None:
            # Predicted failure with high confidence — synthesize the
            # observation Lean would have produced and skip the call.
            obs = {
                "tactic": tactic,
                "success": False,
                "is_proof_complete": False,
                "remaining_goals": list(goals_before),
                "num_goals": len(goals_before),
                "error_category": "world_model_blocked",
                "error_message": (
                    f"WorldModel rejected (p_success={1.0 - gate:.3f}); "
                    f"Lean call skipped"),
                "world_model_blocked": True,
            }
            await self._deposit_step_safe(
                tactic=tactic, env_id_before=env_id, env_id_after=-1,
                goals_before=goals_before, goals_after=[],
                error_message=obs["error_message"],
                error_category="world_model_blocked",
                elapsed_ms=0.0, is_proof_complete=False, ctx=ctx)
            return ToolResult.success(json.dumps(obs, ensure_ascii=False))

        t0 = time.monotonic()
        try:
            r = await self._try_tactic_async(env_id, tactic)
        except Exception as e:
            logger.warning(f"tactic_apply failed: {e}")
            elapsed_ms = (time.monotonic() - t0) * 1000
            # Even infrastructure-level failures get deposited as a step
            # so the Layer 1 error pattern table reflects reality.
            await self._deposit_step_safe(
                tactic=tactic, env_id_before=env_id, env_id_after=-1,
                goals_before=goals_before, goals_after=[],
                error_message=str(e), error_category="repl_error",
                elapsed_ms=elapsed_ms, is_proof_complete=False, ctx=ctx)
            return ToolResult.error(f"REPL error: {e}")
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Build observation
        obs = {
            "tactic": tactic,
            "success": bool(getattr(r, "success", False)),
            "is_proof_complete": bool(getattr(r, "is_proof_complete", False)),
            "remaining_goals": list(getattr(r, "remaining_goals", []) or []),
            "num_goals": len(getattr(r, "remaining_goals", []) or []),
        }

        if not obs["success"]:
            obs["error_category"] = getattr(r, "error_category", "unknown")
            obs["error_message"] = (
                getattr(r, "error_message", "") or "")[:500]

        # Tree-state bookkeeping (only when search driver is active)
        if self._search_state is not None and obs["success"]:
            new_node_id = self._search_state.expand(
                parent_node_id=node_id,
                tactic=tactic,
                new_env_id=getattr(r, "new_env_id", -1),
                remaining_goals=obs["remaining_goals"],
                is_complete=obs["is_proof_complete"],
            )
            obs["new_node_id"] = new_node_id

        # ── 
        # This is fail-soft — _deposit_step_safe swallows every exception.
        await self._deposit_step_safe(
            tactic=tactic,
            env_id_before=env_id,
            env_id_after=getattr(r, "new_env_id", -1) if obs["success"] else -1,
            goals_before=goals_before,
            goals_after=obs["remaining_goals"] if obs["success"] else [],
            error_message=obs.get("error_message", "") if not obs["success"] else "",
            error_category=obs.get("error_category", "") if not obs["success"] else "",
            elapsed_ms=elapsed_ms,
            is_proof_complete=obs["is_proof_complete"],
            ctx=ctx,
        )

        return ToolResult.success(json.dumps(obs, ensure_ascii=False))

    # ── helpers ────────────────────────────────────────────────────────

    def _resolve_env_id(self, node_id):
        """Look up the env_id for a tree node, or return base env if no tree."""
        if self._search_state is not None and node_id is not None:
            return self._search_state.env_id_for(node_id)
        if self._search_state is not None:
            return self._search_state.current_env_id()
        # no search driver — use the pool's base env
        return getattr(self._pool, "base_env_id", 0)

    def _capture_goals_before(self, node_id) -> list[str]:
        """Best-effort lookup of the goals at the node we're about to act on."""
        if self._search_state is None:
            return []
        try:
            cur = node_id
            if cur is None:
                cur = self._search_state.current_node_id
            node = self._search_state.nodes.get(cur)
            return list(node.goals) if node else []
        except Exception:
            return []

    def _wm_gate(self, tactic: str,
                  goals_before: list[str]) -> Optional[float]:
        """If the world model is loaded and predicts this tactic is
        very likely to fail, return ``1 - p_success`` (the rejection
        confidence). Otherwise return None to mean "let it run".

        Conservative: blocks only when *both*
          (a) the model says ``likely_success=False`` AND
          (b) ``confidence >= wm_min_confidence`` (default 0.85).

        Anything below the threshold runs normally so the agent can
        explore tactics the model is uncertain about — that's how
        the model improves.
        """
        if self._wm is None or not goals_before:
            return None
        try:
            goal = goals_before[0]
            pred = self._wm.predict(goal, tactic)
            if (not pred.likely_success
                    and pred.confidence >= self._wm_min_conf):
                return float(pred.confidence)
        except Exception as e:
            logger.debug(f"world_model.predict failed: {e}")
        return None

    async def _try_tactic_async(self, env_id, tactic):
        """Pool may be sync or async; handle both."""
        if hasattr(self._pool, "try_tactic"):
            fn = self._pool.try_tactic
            if asyncio.iscoroutinefunction(fn):
                return await fn(env_id, tactic)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, fn, env_id, tactic)
        raise RuntimeError("lean_pool has no try_tactic method")

    async def _deposit_step_safe(
            self, *, tactic: str,
            env_id_before: int, env_id_after: int,
            goals_before: list[str], goals_after: list[str],
            error_message: str, error_category: str,
            elapsed_ms: float, is_proof_complete: bool,
            ctx: ToolContext) -> None:
        """Fire-and-forget step-level deposit into the knowledge writer.

        Any exception is logged at DEBUG and swallowed — we never let a
        knowledge problem break the proof loop. If no writer is wired,
        this is a fast noop.
        """
        if self._kw is None:
            return
        try:
            from engine.proof_context_store import StepDetail
            step = StepDetail(
                step_index=self._step_index,
                tactic=tactic,
                env_id_before=env_id_before,
                env_id_after=env_id_after,
                goals_before=list(goals_before),
                goals_after=list(goals_after),
                error_message=error_message or "",
                error_category=error_category or "",
                elapsed_ms=elapsed_ms,
                is_proof_complete=bool(is_proof_complete),
            )
            self._step_index += 1
            theorem = getattr(ctx, "theorem_statement", "") or ""
            await self._kw.ingest_step(step, theorem=theorem)
        except Exception as e:
            logger.debug(
                f"tactic_apply step deposit skipped (writer error): {e}")
