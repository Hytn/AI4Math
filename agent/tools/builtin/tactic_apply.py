"""agent/tools/builtin/tactic_apply.py — Apply ONE tactic at the current node.

This is the step-level proving primitive that the existing toolset lacks.
It is the workhorse tool for ReProver / LeanDojo / MCTS / Best-First style
proving, where the agent advances the proof one tactic at a time.

It takes a single tactic string, applies it via the Lean4 REPL pool, and
returns the resulting goal state as a JSON observation. It transparently
updates a `search_state` object (if provided) so the outer SearchDriver
can track tree expansion.
"""
from __future__ import annotations

import json
import logging
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

    def __init__(self, lean_pool=None, search_state=None):
        self._pool = lean_pool
        self._search_state = search_state  # may be None in non-search mode

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        tactic = input["tactic"].strip()
        node_id = input.get("node_id")

        if not self._pool:
            return ToolResult.error(
                "Lean REPL pool not available. tactic_apply requires a live "
                "Lean 4 environment.")

        # Decide which env_id to apply in
        env_id = self._resolve_env_id(node_id)

        try:
            r = await self._try_tactic_async(env_id, tactic)
        except Exception as e:
            logger.warning(f"tactic_apply failed: {e}")
            return ToolResult.error(f"REPL error: {e}")

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

    async def _try_tactic_async(self, env_id, tactic):
        """Pool may be sync or async; handle both."""
        import asyncio
        if hasattr(self._pool, "try_tactic"):
            fn = self._pool.try_tactic
            if asyncio.iscoroutinefunction(fn):
                return await fn(env_id, tactic)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, fn, env_id, tactic)
        raise RuntimeError("lean_pool has no try_tactic method")
