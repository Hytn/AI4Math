"""prover/unified/tools_extra.py — 统一管线特有的几个 Tool

这些工具是为了让某些范式 (DSP / MCTS / 异构) 完整可表达而新增的。
它们的实现会复用现有的 prover.decompose / knowledge / engine.broadcast
模块, 但以"LLM 工具"的形态对外暴露。
"""
from __future__ import annotations
import json
import logging
from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DSP 用: 把当前目标拆成子目标
# ═══════════════════════════════════════════════════════════════════════

class DecomposeSubgoalTool(Tool):
    name = "decompose_subgoal"
    description = (
        "Break the current proof goal into a list of smaller subgoals.\n"
        "\n"
        "Returns a JSON list of subgoal statements. Each subgoal can then be "
        "proved separately (e.g. via `have` blocks in your final proof).\n"
        "\n"
        "Use this when the goal looks like a conjunction, a case split, or "
        "would benefit from structured `have` lemmas. Do NOT call this for "
        "atomic goals already provable in one tactic."
    )
    permission = ToolPermission.READ_ONLY
    input_schema = {
        "type": "object",
        "properties": {
            "goal": {"type": "string",
                     "description": "The Lean 4 goal to decompose."},
        },
        "required": ["goal"],
    }

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        try:
            from prover.decompose.goal_decomposer import GoalDecomposer
            decomposer = GoalDecomposer(None)
            subgoals = decomposer.decompose(input["goal"]) or []
            payload = [
                {"statement": getattr(sg, "statement", str(sg)),
                 "kind": getattr(sg, "kind", "subgoal")}
                for sg in subgoals
            ]
            return ToolResult.success(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            return ToolResult.error(f"decompose failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 项目内已证引理库 (跨问题复用)
# ═══════════════════════════════════════════════════════════════════════

class LemmaBankTool(Tool):
    name = "lemma_bank"
    description = (
        "Search the project's bank of *previously proved* lemmas (across "
        "this run AND prior runs). Useful for reusing proofs of helper "
        "lemmas you already established.\n"
        "\n"
        "Returns matching lemmas as {name, statement, proof, times_cited}."
    )
    permission = ToolPermission.READ_ONLY
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Goal pattern or keywords."},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, knowledge_store=None):
        self._store = knowledge_store

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        if not self._store:
            return ToolResult.success(
                json.dumps([], ensure_ascii=False),
                count=0)
        try:
            from knowledge.reader import KnowledgeReader
            reader = KnowledgeReader(self._store)
            top_k = input.get("top_k", 5)
            lemmas = await reader.find_lemmas(
                goal=input["query"], top_k=top_k)
            payload = [
                {"name": lm.name, "statement": lm.statement,
                 "proof": lm.proof, "times_cited": lm.times_cited,
                 "relevance": lm.relevance_score}
                for lm in lemmas
            ]
            return ToolResult.success(
                json.dumps(payload, ensure_ascii=False),
                count=len(payload))
        except Exception as e:
            return ToolResult.error(f"lemma_bank query failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 异构方向: 跨 agent 共享发现
# ═══════════════════════════════════════════════════════════════════════

class BroadcastTool(Tool):
    name = "broadcast"
    description = (
        "Read recent messages from your teammates (other agents working on "
        "the same theorem in parallel), or share your own discoveries with "
        "them.\n"
        "\n"
        "Action `read`: returns recent broadcasts ({source, kind, content}).\n"
        "Action `share`: publishes a discovery to all teammates."
    )
    permission = ToolPermission.WRITE_LOCAL
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "share"]},
            "kind":   {"type": "string",
                       "enum": ["positive", "negative", "lemma", "partial_proof"],
                       "description": "Only used when action=share."},
            "content": {"type": "string"},
            "max_messages": {"type": "integer", "default": 10},
        },
        "required": ["action"],
    }

    def __init__(self, bus=None):
        self._bus = bus

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        if not self._bus:
            return ToolResult.success("[]")
        action = input["action"]
        if action == "read":
            n = input.get("max_messages", 10)
            msgs = self._bus.get_recent(n=n) if hasattr(self._bus, "get_recent") else []
            payload = [
                {"source": getattr(m, "source", ""),
                 "kind": getattr(m, "msg_type", "info"),
                 "content": getattr(m, "content", str(m))}
                for m in msgs
            ]
            return ToolResult.success(json.dumps(payload, ensure_ascii=False))
        else:  # share
            try:
                from engine.broadcast import BroadcastMessage
                kind = input.get("kind", "positive")
                content = input.get("content", "")
                fn = {
                    "positive": getattr(BroadcastMessage, "positive", None),
                    "negative": getattr(BroadcastMessage, "negative", None),
                }.get(kind)
                if fn:
                    msg = fn(source=ctx.agent_name, discovery=content)
                else:
                    msg = BroadcastMessage(
                        source=ctx.agent_name, content=content,
                        msg_type=kind)
                self._bus.publish(msg)
                return ToolResult.success(json.dumps({"posted": True}))
            except Exception as e:
                return ToolResult.error(f"broadcast share failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 让 LLM 看到搜索树 (MCTS / best-first 的 LLM-driven 变种)
# ═══════════════════════════════════════════════════════════════════════

class TreeViewTool(Tool):
    name = "tree_view"
    description = (
        "View the current state of the proof search tree.\n"
        "\n"
        "Returns: ancestors (path of tactics from root to current node), "
        "siblings (alternative tactics tried at ancestors with their "
        "outcomes), open_leaves (other unexpanded nodes ranked by score). "
        "Use this to avoid retrying tactics that already failed at this goal."
    )
    permission = ToolPermission.READ_ONLY
    input_schema = {
        "type": "object",
        "properties": {
            "node_id": {"type": "integer",
                        "description": "Defaults to current node."},
            "depth": {"type": "integer", "default": 3,
                      "description": "How many ancestors/siblings to show."},
        },
    }

    def __init__(self, search_state=None):
        self._state = search_state

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        if not self._state:
            return ToolResult.success(json.dumps({"tree": "no search active"}))
        node_id = input.get("node_id")
        depth = input.get("depth", 3)
        snapshot = self._state.render_snapshot(
            node_id=node_id, depth=depth)
        return ToolResult.success(
            json.dumps(snapshot, ensure_ascii=False))


class TreeSelectTool(Tool):
    name = "tree_select"
    description = (
        "Select which open node of the search tree to expand next. Use this "
        "ONLY when you want to override the default UCB/best-first selection. "
        "In most runs you should let the search driver pick automatically."
    )
    permission = ToolPermission.WRITE_LOCAL
    input_schema = {
        "type": "object",
        "properties": {
            "node_id": {"type": "integer"},
        },
        "required": ["node_id"],
    }

    def __init__(self, search_state=None):
        self._state = search_state

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        if not self._state:
            return ToolResult.error("No search active.")
        try:
            self._state.set_current(input["node_id"])
            return ToolResult.success(
                json.dumps({"current_node": input["node_id"]}))
        except Exception as e:
            return ToolResult.error(f"select failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# v6: Conjecture-driven proving — propose auxiliary lemmas
# ═══════════════════════════════════════════════════════════════════════
#
# The ``prover/conjecture/`` package has been in the codebase since v2:
# ``ConjectureProposer`` asks an LLM for plausible auxiliary lemmas and
# ``ConjectureVerifier`` filters them by parse-ability and relevance.
# Until v6 it had no first-class place in the agent loop — no Profile
# advertised it as an action the LLM could take. This tool plugs the
# proposer in as a regular Tool, which lets a ``conjecture_driven``
# Profile invoke it from the unified runner the same way every other
# method is invoked.
#
# Design note: the proposer needs an LLM. The Tool signature does NOT
# take one in __init__ in the legacy code path because tools are
# constructed by ``_build_tool`` from the ToolKit enum. v6 threads
# ``llm`` through ``build_tool_registry`` for the clean path. As a
# safety net the tool also reads ``ctx.shared_state['llm']`` at call
# time, so older callers that build registries without the v6 kwarg
# still work as long as the agent loop populates shared_state. If
# neither is present, the tool returns a structured error — same
# fallback contract as the infrastructure tools when their backend
# is absent.
class ConjectureProposeTool(Tool):
    name = "conjecture_propose"
    description = (
        "Propose auxiliary lemma statements that might help prove the "
        "current target theorem.\n"
        "\n"
        "Returns a JSON list of {statement, score} objects. Use this "
        "when you suspect the target needs a non-obvious intermediate "
        "lemma that isn't already in Mathlib. The statements are "
        "filtered for parseability and relevance — but they are NOT "
        "yet proved. After you receive them, prove each one with the "
        "same agent loop (e.g. by writing `have helper : ... := by "
        "...`) before using them in the main proof.\n"
        "\n"
        "DO NOT call this for goals that look provable by direct "
        "tactics (omega/simp/ring/aesop) — conjecturing is expensive "
        "and only pays off on structurally hard goals."
    )
    permission = ToolPermission.READ_ONLY
    input_schema = {
        "type": "object",
        "properties": {
            "theorem": {
                "type": "string",
                "description": "The target theorem statement.",
            },
            "n": {
                "type": "integer",
                "default": 5,
                "description": "Maximum number of conjectures to return.",
            },
            "existing_lemmas": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": (
                    "Lemmas already known/proved in this run. Pass them "
                    "so the proposer doesn't suggest duplicates."),
            },
        },
        "required": ["theorem"],
    }

    def __init__(self, llm=None, lean_env=None):
        # ``llm`` and ``lean_env`` are the two upstream dependencies of
        # ``ConjectureProposer``. We accept them at construction so
        # ``_build_tool`` can wire the runner's LLM in directly. They
        # remain optional so existing code paths that build registries
        # without an LLM (e.g. tests) still construct successfully.
        self._llm = llm
        self._lean_env = lean_env

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        # Resolve LLM in priority order:
        #   1. tool-bound (passed at construction by _build_tool)
        #   2. ctx.shared_state['llm'] — the standard cross-tool wire
        #      that ToolContext exposes via its ``shared_state`` dict
        #   3. legacy ``ctx.llm`` attribute, kept for older callers
        llm = self._llm
        if llm is None:
            ss = getattr(ctx, "shared_state", None) or {}
            llm = ss.get("llm") if isinstance(ss, dict) else None
        if llm is None:
            llm = getattr(ctx, "llm", None)
        if llm is None:
            return ToolResult.error(
                "conjecture_propose: no LLM available "
                "(neither tool-bound nor in ctx.shared_state). "
                "Check that the runner was constructed with `llm=...`.")

        try:
            from prover.conjecture.conjecture_proposer import (
                ConjectureProposer)
        except ImportError as e:
            return ToolResult.error(
                f"conjecture_propose: module unavailable: {e}")

        theorem = input.get("theorem", "").strip()
        if not theorem:
            return ToolResult.error(
                "conjecture_propose: empty 'theorem' input")
        n = int(input.get("n", 5))
        n = max(1, min(n, 20))  # clamp to a reasonable range
        existing = input.get("existing_lemmas") or []
        if not isinstance(existing, list):
            existing = []
        existing = [str(x) for x in existing if str(x).strip()]

        try:
            proposer = ConjectureProposer(llm=llm, lean_env=self._lean_env)
            # We deliberately disable on-tool verification to avoid
            # the proposer's verifier doing a Lean roundtrip — the
            # runner has its own verification gate. We rely on the
            # proposer's textual filtering only.
            statements = proposer.propose(
                theorem=theorem,
                existing_lemmas=existing,
                n=n,
                verify=False,
            )
        except Exception as e:
            return ToolResult.error(
                f"conjecture_propose: proposer failed: {e}")

        # The proposer returns a flat list of strings; package as
        # objects so future extensions (e.g. relevance score) don't
        # need a schema bump.
        payload = [{"statement": s, "score": None}
                   for s in statements if isinstance(s, str)]
        return ToolResult.success(
            json.dumps(payload, ensure_ascii=False))
