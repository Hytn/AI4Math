"""agent/persistence/dialog_adapters.py — Adapters from legacy formats

Bridges the three pre-existing trajectory representations in AI4Math
into the canonical wrapped Dialog format defined in
``dialog_format.py``.

Each adapter produces a full Dialog object with ``meta``, ``messages``,
and ``result`` populated to the extent the source format carries that
information. ``meta.system_prompt`` and ``meta.tools`` may be empty if
the source doesn't track them — call sites can fill them in via
``DialogBuilder.set_meta(...)`` before saving.

Legacy formats covered:

  1. ``agent/runtime/agent_loop.py::LoopMessage`` / ``LoopResult``
        Live in-memory record from the AgentLoop.

  2. ``sampler/trajectory.py::Trajectory`` / ``Turn``
        Multi-turn RL rollout. Each ``Turn`` has ``observation`` and
        ``action``. Reward metadata is JSON-packed into the tool
        response payload.

  3. ``prover/models.py::ProofTrace`` / ``ProofAttempt``
        Multi-attempt single-shot prover trace.

In all three cases the goal is the same: produce the wrapped Dialog
that matches the schema in dialog_format.py — a single self-contained
record of the run.


``agent/persistence/session_store.py``) was removed — 0 production
callers remained after the lane subsystem was deleted.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional, Union

from agent.persistence.dialog_format import (
    DEFAULT_SERVER_MAP,
    DialogBuilder, ToolCall,
    is_tool_response_user_msg, strip_tool_response_wrapper,
    messages_of,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# 1. AgentLoop  →  Dialog
# ─────────────────────────────────────────────────────────────────────────

def from_loop_messages(
    history: list,                           # list[LoopMessage] | list[dict]
    initial_task: Optional[str] = None,
    *,
    wrapped: bool = False,
    meta: Optional[dict] = None,
    result: Optional[dict] = None,
) -> Union[list[dict], dict]:
    """Convert AgentLoop's history into a canonical dialog.

    Args:
      history:      ``LoopMessage`` instances or plain dicts.
      initial_task: Used only if history is empty.
      wrapped:      If True, return the full wrapped Dialog with
                    ``meta`` / ``messages`` / ``result``. If False
                    (legacy default), return just the messages list.
      meta:         Pre-built meta dict to merge into the wrapper.
      result:       Pre-built result dict to merge into the wrapper.
    """
    b = DialogBuilder()

    if initial_task and not history:
        b.add_user(initial_task)
    else:
        pending_tool_calls: list[ToolCall] = []

        for entry in history:
            role, content, tool_calls_raw, tool_results_raw = \
                _normalize_loop_entry(entry)

            if role == "user":
                text_parts, tool_result_blocks = \
                    _split_user_content(content)
                if tool_result_blocks:
                    for blk in tool_result_blocks:
                        tc = _pop_pending(pending_tool_calls,
                                          blk.get("tool_use_id", ""))
                        b.add_tool_response(
                            name=tc.name if tc else "",
                            content=blk.get("content", ""),
                            tool_call_id=blk.get("tool_use_id", ""),
                            server_id=tc.server_id if tc else "default",
                        )
                text = "\n".join(text_parts).strip()
                if text:
                    if is_tool_response_user_msg(
                            {"role": "user", "content": text}):
                        payload = strip_tool_response_wrapper(text)
                        tc = pending_tool_calls.pop(0) \
                            if pending_tool_calls else None
                        b.add_tool_response(
                            name=tc.name if tc else "",
                            content=payload,
                            tool_call_id=tc.id if tc else "",
                            server_id=tc.server_id if tc else "default",
                        )
                    else:
                        b.add_user(text)

            elif role == "assistant":
                text_parts, blk_tool_calls = \
                    _split_assistant_content(content)
                tcs: list[ToolCall] = []
                for bt in blk_tool_calls:
                    tc = ToolCall.new(
                        name=bt["name"],
                        arguments=bt.get("input", {}),
                        server_id=DEFAULT_SERVER_MAP.get(
                            bt["name"], "default"),
                        tool_id=bt.get("id", ""),
                    )
                    tcs.append(tc)
                for raw in tool_calls_raw or []:
                    tc = _coerce_tool_call(raw)
                    if tc:
                        tcs.append(tc)
                pending_tool_calls.extend(tcs)
                b.add_assistant(
                    content="\n".join(text_parts).strip(),
                    tool_calls=tcs or None,
                )

            elif role in ("tool_result", "tool"):
                results = tool_results_raw or []
                if not results and content:
                    results = [content if isinstance(content, str)
                               else json.dumps(content,
                                               ensure_ascii=False)]
                for res in results:
                    tc = pending_tool_calls.pop(0) \
                        if pending_tool_calls else None
                    b.add_tool_response(
                        name=tc.name if tc else "",
                        content=res,
                        tool_call_id=tc.id if tc else "",
                        server_id=tc.server_id if tc else "default",
                    )

    if not wrapped:
        return b.build_messages()

    if meta:
        b.set_meta(**meta)
    if result:
        b.set_result(**result)
    return b.build()

def _normalize_loop_entry(entry: Any) -> tuple[str, Any, Any, Any]:
    if hasattr(entry, "role"):
        return (
            getattr(entry, "role", ""),
            getattr(entry, "content", ""),
            getattr(entry, "tool_calls", None),
            getattr(entry, "tool_results", None),
        )
    if isinstance(entry, dict):
        return (
            entry.get("role", ""),
            entry.get("content", ""),
            entry.get("tool_calls"),
            entry.get("tool_results"),
        )
    return ("", "", None, None)

def _split_user_content(content: Any) -> tuple[list[str], list[dict]]:
    if isinstance(content, str):
        return ([content], [])
    if isinstance(content, list):
        texts: list[str] = []
        tool_results: list[dict] = []
        for blk in content:
            if not isinstance(blk, dict):
                texts.append(str(blk))
                continue
            t = blk.get("type")
            if t == "text":
                texts.append(blk.get("text", ""))
            elif t == "tool_result":
                tool_results.append(blk)
            else:
                texts.append(json.dumps(blk, ensure_ascii=False))
        return (texts, tool_results)
    return ([str(content) if content else ""], [])

def _split_assistant_content(content: Any) -> tuple[list[str], list[dict]]:
    if isinstance(content, str):
        return ([content], [])
    if isinstance(content, list):
        texts: list[str] = []
        tool_uses: list[dict] = []
        for blk in content:
            if not isinstance(blk, dict):
                texts.append(str(blk))
                continue
            t = blk.get("type")
            if t == "text":
                texts.append(blk.get("text", ""))
            elif t == "tool_use":
                tool_uses.append(blk)
            else:
                texts.append(json.dumps(blk, ensure_ascii=False))
        return (texts, tool_uses)
    return ([str(content) if content else ""], [])

def _coerce_tool_call(raw: Any) -> Optional[ToolCall]:
    if not isinstance(raw, dict):
        return None
    if "function" in raw and isinstance(raw["function"], dict):
        fn = raw["function"]
        return ToolCall.new(
            name=fn.get("name", ""),
            arguments=fn.get("arguments", "{}"),
            server_id=raw.get("server_id", ""),
            tool_id=raw.get("id", ""),
        )
    if "name" in raw:
        return ToolCall.new(
            name=raw.get("name", ""),
            arguments=raw.get("input", {}),
            server_id=raw.get("server_id", ""),
            tool_id=raw.get("id", ""),
        )
    return None

def _pop_pending(pending: list[ToolCall],
                 tool_use_id: str) -> Optional[ToolCall]:
    if tool_use_id:
        for i, tc in enumerate(pending):
            if tc.id == tool_use_id:
                return pending.pop(i)
    return pending.pop(0) if pending else None

# ─────────────────────────────────────────────────────────────────────────
# 2. sampler.Trajectory  →  Dialog
# ─────────────────────────────────────────────────────────────────────────

def from_trajectory(
    trajectory: Any,
    *,
    include_reward_in_tool_payload: bool = True,
    proof_tool_name: str = "lean_verify",
    wrapped: bool = False,
    meta: Optional[dict] = None,
    result: Optional[dict] = None,
) -> Union[list[dict], dict]:
    """Convert a ``sampler.trajectory.Trajectory`` to a Dialog.

    See module docstring for the per-turn mapping.
    """
    b = DialogBuilder()
    turns = list(getattr(trajectory, "turns", []) or [])

    if not turns:
        stmt = getattr(trajectory, "theorem_statement", "") or ""
        if stmt:
            b.add_user(_format_initial_user_text(stmt))
    else:
        b.add_user(turns[0].observation)
        for turn in turns:
            action = (turn.action or "").strip()
            tc = ToolCall.new(
                name=proof_tool_name,
                arguments={"code": action},
                server_id=DEFAULT_SERVER_MAP.get(proof_tool_name, "lean"),
            )
            b.add_assistant(
                content="```lean\n" + action + "\n```" if action else "",
                tool_calls=[tc],
            )

            reward = getattr(turn, "reward", None)
            payload: dict[str, Any] = {}
            if reward is not None:
                payload["raw_feedback"] = (
                    getattr(reward, "raw_feedback", "")
                    or getattr(reward, "fix_hint", ""))
                if include_reward_in_tool_payload:
                    payload["reward"] = {
                        "scalar": float(getattr(reward, "scalar", 0.0)),
                        "verification_level":
                            getattr(reward, "verification_level", ""),
                        "error_class": getattr(reward, "error_class", ""),
                        "goals_remaining":
                            int(getattr(reward, "goals_remaining", -1)),
                        "goals_closed":
                            int(getattr(reward, "goals_closed", 0)),
                        "is_terminal":
                            bool(getattr(reward, "is_terminal", False)),
                        "fix_hint": getattr(reward, "fix_hint", ""),
                    }
            if not payload and turn.observation:
                payload = {"raw_feedback": turn.observation}

            b.add_tool_response(
                name=proof_tool_name,
                content=json.dumps(payload, ensure_ascii=False)
                        if payload else "",
                server_id=DEFAULT_SERVER_MAP.get(
                    proof_tool_name, "lean"),
            )

    if not wrapped:
        return b.build_messages()

    # Auto-fill meta / result from trajectory attrs
    auto_meta = {
        "problem_id": getattr(trajectory, "problem_id", ""),
        "theorem_statement":
            getattr(trajectory, "theorem_statement", ""),
    }
    if meta:
        auto_meta.update(meta)
    b.set_meta(**auto_meta)

    auto_result = {
        "success": bool(getattr(trajectory, "success", False)),
        "total_attempts":
            len(getattr(trajectory, "turns", []) or []),
        "total_tokens":
            int(getattr(trajectory, "total_tokens", 0) or 0),
        "total_duration_ms":
            int(float(getattr(trajectory, "wall_time_s", 0.0) or 0) * 1000),
        "termination": getattr(getattr(trajectory, "termination", None),
                               "value", ""),
        "extra": {
            "total_reward": round(
                float(getattr(trajectory, "total_reward", 0.0)), 4),
            "turn_scores": [
                getattr(t.reward, "scalar", 0.0) for t in turns
            ],
            "metadata": dict(getattr(trajectory, "metadata", {}) or {}),
        },
    }
    if result:
        auto_result.update(result)
    b.set_result(**auto_result)
    return b.build()

def _format_initial_user_text(theorem: str) -> str:
    return "Prove the following theorem in Lean 4:\n\n" + theorem.strip()

# ─────────────────────────────────────────────────────────────────────────
# 3. prover.ProofTrace  →  Dialog
# ─────────────────────────────────────────────────────────────────────────

def from_proof_trace(
    trace: Any,
    *,
    include_premise_search_turns: bool = True,
    proof_tool_name: str = "lean_verify",
    premise_tool_name: str = "premise_search",
    wrapped: bool = False,
    meta: Optional[dict] = None,
    result: Optional[dict] = None,
) -> Union[list[dict], dict]:
    """Convert a ``prover.models.ProofTrace`` to a Dialog."""
    b = DialogBuilder()
    theorem = getattr(trace, "theorem_statement", "") or ""
    nl = getattr(trace, "natural_language", "") or ""

    user_lines = [_format_initial_user_text(theorem)]
    if nl:
        user_lines.append("\nInformal statement:\n" + nl)
    b.add_user("\n".join(user_lines))

    for attempt in getattr(trace, "attempts", []) or []:
        premises = list(getattr(attempt, "retrieved_premises", []) or [])
        if include_premise_search_turns and premises:
            search_call = ToolCall.new(
                name=premise_tool_name,
                arguments={
                    "query": _premise_query_from_theorem(theorem)},
                server_id=DEFAULT_SERVER_MAP.get(
                    premise_tool_name, "mathlib"),
            )
            b.add_assistant(content="", tool_calls=[search_call])
            b.add_tool_response(
                name=premise_tool_name,
                content=json.dumps(
                    [_normalize_premise(p) for p in premises[:20]],
                    ensure_ascii=False),
                server_id=search_call.server_id,
            )

        proof = (getattr(attempt, "generated_proof", "") or "").strip()
        if not proof:
            err = getattr(attempt, "lean_stderr", "") or "(empty proof)"
            b.add_assistant(content=f"(generation failed: {err})")
            continue

        verify_call = ToolCall.new(
            name=proof_tool_name,
            arguments={"code": proof},
            server_id=DEFAULT_SERVER_MAP.get(proof_tool_name, "lean"),
        )
        b.add_assistant(
            content="```lean\n" + proof + "\n```",
            tool_calls=[verify_call],
        )
        b.add_tool_response(
            name=proof_tool_name,
            content=_proof_attempt_payload(attempt),
            server_id=verify_call.server_id,
        )

    if not wrapped:
        return b.build_messages()

    auto_meta = {
        "problem_id": getattr(trace, "problem_id", ""),
        "problem_name": getattr(trace, "problem_name", ""),
        "theorem_statement": theorem,
        "informal_statement": nl,
        "extra": {
            "trace_id": getattr(trace, "trace_id", ""),
            "config_snapshot":
                dict(getattr(trace, "config_snapshot", {}) or {}),
        },
    }
    if meta:
        for k, v in meta.items():
            if k == "extra" and isinstance(v, dict):
                auto_meta["extra"].update(v)
            elif v is not None:
                auto_meta[k] = v
    b.set_meta(**auto_meta)

    auto_result = {
        "success": bool(getattr(trace, "solved", False)),
        "total_attempts": int(getattr(trace, "total_attempts", 0) or 0),
        "total_tokens": int(getattr(trace, "total_tokens", 0) or 0),
        "total_duration_ms":
            int(getattr(trace, "total_duration_ms", 0) or 0),
        "successful_proof":
            getattr(trace, "successful_proof", "") or "",
        "error_distribution":
            dict(getattr(trace, "error_distribution", {}) or {}),
        "extra": {
            "correct_count":
                int(getattr(trace, "correct_count", 0) or 0),
            "strategy_path":
                list(getattr(trace, "strategy_path", []) or []),
        },
    }
    if result:
        for k, v in result.items():
            if k == "extra" and isinstance(v, dict):
                auto_result["extra"].update(v)
            elif v is not None:
                auto_result[k] = v
    b.set_result(**auto_result)
    return b.build()

def _premise_query_from_theorem(theorem: str) -> str:
    s = theorem.strip()
    head = s.split(":", 1)
    return head[-1].strip() if len(head) == 2 else s

def _normalize_premise(p: Any) -> dict:
    if isinstance(p, str):
        return {"name": p}
    if isinstance(p, dict):
        return p
    return {"name": str(p)}

def _proof_attempt_payload(attempt: Any) -> str:
    status = getattr(attempt, "lean_result", None)
    status_val = getattr(status, "value",
                         str(status) if status else "")
    payload = {
        "verified": status_val == "success",
        "status": status_val,
        "errors": [],
        "stderr": (getattr(attempt, "lean_stderr", "") or "")[:1000],
    }
    for e in getattr(attempt, "lean_errors", []) or []:
        cat = getattr(e, "category", None)
        cat_val = getattr(cat, "value", str(cat) if cat else "")
        payload["errors"].append({
            "category": cat_val,
            "message": (getattr(e, "message", "") or "")[:500],
            "line": getattr(e, "line", None),
            "column": getattr(e, "column", None),
            "suggestions":
                list(getattr(e, "suggestions", []) or []),
        })
    return json.dumps(payload, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────────────────
# 4. SessionData.messages  →  Dialog
# ─────────────────────────────────────────────────────────────────────────

def to_openai_messages(dialog: Any) -> list[dict]:
    """Project a dialog to the OpenAI / Claude messages-array shape.
    Drops ``thought`` (training-only signal). Accepts wrapped or plain."""
    out: list[dict] = []
    for m in messages_of(dialog):
        role = m.get("role")
        if role == "user":
            out.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            d: dict = {"role": "assistant",
                       "content": m.get("content", "")}
            if m.get("tool_calls"):
                d["tool_calls"] = m["tool_calls"]
            out.append(d)
        elif role == "tool":
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", ""),
                "name": m.get("name", ""),
                "content": content,
            })
    return out

__all__ = [
    "from_loop_messages", "from_trajectory", "from_proof_trace",
    "to_openai_messages",
]
