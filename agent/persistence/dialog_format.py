"""agent/persistence/dialog_format.py — Self-contained trajectory format

A single ``dialog.json`` file losslessly captures everything an LLM
agent saw and produced during one run:

  * **What the agent was told** — system prompt, available tools, task
  * **What the agent saw** — every user / tool message in order
  * **What the agent thought** — reasoning / planning text
  * **What the agent did** — tool calls with arguments
  * **What came back** — tool responses, including the Lean verifier's output
  * **What happened** — final outcome, tokens, timing

There is no separate ``meta_config.json`` or ``result.json``. To debug
or re-train from a run, you open exactly one file and see everything.

────────────────────────────────────────────────────────────────────────
Schema (version 2.0)
────────────────────────────────────────────────────────────────────────

    Dialog = {
        "schema_version": "2.0",

        "meta": {
            # Task identity
            "task_id":            str,
            "problem_id":         str,
            "problem_name":       str,
            "theorem_statement":  str,
            "informal_statement": str,    # natural-language form, optional

            # Execution
            "model":              str,    # "qwen3-32b", "claude-…", …
            "provider":           str,    # "local", "anthropic", …
            "started_at":         str,    # ISO-8601 UTC
            "finished_at":        str,    # ISO-8601 UTC

            # What the agent was given
            "system_prompt":      str,
            "tools": [                    # available tools at run time
                {"name": str,
                 "description": str,
                 "parameters": dict,      # JSON Schema
                 "server_id": str}        # which provider served it
            ],

            # Free-form extras (config snapshot, run tags, etc.)
            "extra": dict,
        },

        "messages": [                     # ← AgentCPM-style turn list
            {"role": "user",      "content": str},
            {"role": "assistant", "content": str,
             "thought": str?,
             "tool_calls": [
                 {"id": str,
                  "function": {"name": str, "arguments": str},
                  "server_id": str}
             ]?},
            {"role": "tool", "tool_call_id": str, "name": str,
             "content": Any, "server_id": str},
            ...
        ],

        "result": {                       # outcome — populated when run ends
            "success":           bool,
            "total_attempts":    int,
            "total_tokens":      int,
            "total_duration_ms": int,
            "successful_proof":  str,
            "termination":       str,     # "success" / "max_turns" / …
            "error_distribution": dict,
            "extra":             dict,
        }
    }

The ``messages`` list itself is byte-for-byte identical to AgentCPM's
``dialog.json`` content — so anything that already consumes AgentCPM
data only needs to read ``dialog["messages"]`` and it just works.

────────────────────────────────────────────────────────────────────────
Backward compatibility
────────────────────────────────────────────────────────────────────────

Functions that take a ``dialog`` argument accept either form:

  * the new wrapped object (``{"schema_version": …, "meta": …, …}``)
  * the legacy plain message list (``[{"role": …}, …]``)

Use ``messages_of(dialog)`` / ``meta_of(dialog)`` / ``result_of(dialog)``
to read regardless of which form you got.

``save_dialog`` always writes the wrapped form. Loading auto-detects.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────

SCHEMA_VERSION = "3.0"
DIALOG_FILENAME = "dialog.json"   # the only file we produce

# v3.0 adds an optional ``meta.search_tree`` block describing the search
# DAG when the run used a tree-search profile (mcts / best_first / beam).
# When present, ``messages`` carries the *solved path* (or the best
# explored path if unsolved) — same shape and SFT semantics as v2.0.
# All v2.0 files are valid v3.0 files (search_tree absent).
SUPPORTED_SCHEMA_VERSIONS = ("1.0", "2.0", "3.0")

# AgentCPM context-split marker. Long sessions can be split into per-window
# dialogs at points marked by this role. See ``split_dialog_at_markers``.
CONTEXT_SPLIT_ROLE = "__CONTEXT_SPLIT__"

# Wrapper used by some runtimes to feed tool responses back into the
# user-turn. We accept either form on input and always normalize to
# role:"tool" on output.
_TOOL_RESPONSE_RE = re.compile(
    r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL,
)

# Tool → server_id default. Overridable per-call.
DEFAULT_SERVER_MAP: dict[str, str] = {
    "premise_search": "mathlib",
    "tactic_suggest": "mathlib",
    "goal_inspect": "lean",
    "lean_verify": "lean",
    "lean_auto": "lean",
    "cas_tool": "cas",
}


# ── Message classes ─────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """One tool invocation inside an assistant message."""
    id: str
    name: str
    arguments: str             # JSON-encoded string, NOT a dict
    server_id: str = "default"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "function": {"name": self.name, "arguments": self.arguments},
            "server_id": self.server_id,
        }

    @classmethod
    def new(cls, name: str, arguments: Union[dict, str],
            server_id: str = "", tool_id: str = "") -> "ToolCall":
        if isinstance(arguments, str):
            args_str = arguments
        else:
            args_str = json.dumps(arguments, ensure_ascii=False)
        return cls(
            id=tool_id or f"call_{uuid.uuid4().hex[:12]}",
            name=name,
            arguments=args_str,
            server_id=server_id or DEFAULT_SERVER_MAP.get(name, "default"),
        )


@dataclass
class Message:
    """A single dialog message (one turn in the timeline)."""
    role: str                                # "user"|"assistant"|"tool"
    content: Any = ""
    thought: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    server_id: Optional[str] = None

    def to_dict(self) -> dict:
        if self.role == "user":
            return {"role": "user", "content": self.content or ""}

        if self.role == "assistant":
            d: dict = {"role": "assistant", "content": self.content or ""}
            if self.thought:
                d["thought"] = self.thought
            if self.tool_calls:
                d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
            return d

        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id or "",
                "name": self.name or "",
                "content": self.content,
                "server_id": self.server_id or "default",
            }

        raise ValueError(f"Unknown role: {self.role!r}")

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str = "",
        thought: Optional[str] = None,
        tool_calls: Optional[list[ToolCall]] = None,
    ) -> "Message":
        return cls(
            role="assistant",
            content=content or "",
            thought=thought or None,
            tool_calls=list(tool_calls) if tool_calls else None,
        )

    @classmethod
    def tool(
        cls, name: str, content: Any,
        tool_call_id: str = "", server_id: str = "",
    ) -> "Message":
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            server_id=server_id or DEFAULT_SERVER_MAP.get(name, "default"),
        )


# ── Builder ─────────────────────────────────────────────────────────────

class DialogBuilder:
    """Build a self-contained dialog incrementally.

    Sets meta / result fields via dedicated setters. The terminal call
    is ``.build()`` which returns the full wrapped object ready to save.

    Example::

        b = DialogBuilder()
        b.set_meta(
            problem_id="nat_add_zero",
            theorem_statement="theorem t (n : ℕ) : n + 0 = n",
            model="qwen3-32b",
            system_prompt="You are a Lean 4 theorem prover.",
            tools=[{"name": "lean_verify", ...}],
        )
        b.add_user("Prove: theorem t (n : ℕ) : n + 0 = n")
        b.add_assistant_proof(
            proof_code="theorem t (n : ℕ) : n + 0 = n := by simp",
            thought="One-line proof via simp.",
        )
        b.set_result(success=True, total_tokens=42)

        save_dialog(b.build(), "results/traces/nat_add_zero/dialog.json")
    """

    def __init__(self):
        self._messages: list[Message] = []
        self._meta: dict[str, Any] = {}
        self._result: dict[str, Any] = {}
        self._started_at: str = _utc_now_iso()

    # ── Meta ──

    def set_meta(self, **fields: Any) -> "DialogBuilder":
        """Set one or more meta fields. Common keys: ``task_id``,
        ``problem_id``, ``problem_name``, ``theorem_statement``,
        ``informal_statement``, ``model``, ``provider``,
        ``system_prompt``, ``tools``, ``extra``.
        """
        for k, v in fields.items():
            if k == "tools" and v is not None:
                self._meta["tools"] = [_normalize_tool_spec(t) for t in v]
            elif v is not None:
                self._meta[k] = v
        return self

    def update_meta_extra(self, **fields: Any) -> "DialogBuilder":
        """Merge fields into ``meta.extra``."""
        extra = self._meta.setdefault("extra", {})
        extra.update(fields)
        return self

    def set_search_tree(self, tree: dict) -> "DialogBuilder":
        """Attach a search-tree summary to ``meta.search_tree``.

        The tree describes the explored DAG when the run used a tree-search
        profile (mcts / best_first / beam). Schema:

            {"kind":   "ucb" | "best_first" | "beam",
             "root_node_id":   0,
             "solved_node_id": int | None,
             "total_nodes":    int,
             "max_depth":      int,
             "nodes": [
                {"node_id":      int,
                 "parent_id":    int | None,
                 "tactic":       str | None,
                 "depth":        int,
                 "status":       "open" | "solved" | "failed" | "pruned",
                 "visit_count":  int,
                 "success_count":int,
                 "score":        float,
                 "messages":     list[dict]   # per-node expansion turns
                }, ...
             ]}

        ``messages`` on the top level still carries the solved-path
        flattening; this block is purely an additional view.
        """
        self._meta["search_tree"] = dict(tree)
        return self

    # ── Result ──

    def set_result(self, **fields: Any) -> "DialogBuilder":
        """Set outcome fields. Common keys: ``success``,
        ``total_attempts``, ``total_tokens``, ``total_duration_ms``,
        ``successful_proof``, ``termination``, ``error_distribution``,
        ``extra``.
        """
        for k, v in fields.items():
            if v is not None:
                self._result[k] = v
        return self

    # ── Messages ──

    def add(self, msg: Message) -> "DialogBuilder":
        self._messages.append(msg)
        return self

    def extend(self, msgs: Iterable[Message]) -> "DialogBuilder":
        self._messages.extend(msgs)
        return self

    def add_user(self, content: str) -> "DialogBuilder":
        return self.add(Message.user(content))

    def add_assistant(
        self, content: str = "",
        thought: Optional[str] = None,
        tool_calls: Optional[list[ToolCall]] = None,
    ) -> "DialogBuilder":
        if not (content or thought or tool_calls):
            return self  # drop empty turns (matches AgentCPM)
        return self.add(Message.assistant(content, thought, tool_calls))

    def add_assistant_thinking(
        self, thought: str,
        followed_by_tool_call: Optional[
            Union[tuple[str, dict], tuple[str, dict, str]]
        ] = None,
    ) -> "DialogBuilder":
        tcs: Optional[list[ToolCall]] = None
        if followed_by_tool_call is not None:
            if len(followed_by_tool_call) == 2:
                name, args = followed_by_tool_call
                server_id = ""
            else:
                name, args, server_id = followed_by_tool_call  # type: ignore
            tcs = [ToolCall.new(name, args, server_id=server_id)]
        return self.add_assistant(
            content="", thought=thought, tool_calls=tcs)

    def add_assistant_proof(
        self, proof_code: str,
        thought: Optional[str] = None,
        prelude: str = "",
    ) -> "DialogBuilder":
        body = proof_code.strip()
        if not body:
            content = prelude
        else:
            content = (
                (prelude.rstrip() + "\n\n" if prelude else "")
                + "```lean\n" + body + "\n```"
            )
        return self.add_assistant(content=content, thought=thought)

    def add_tool_response(
        self, name: str, content: Any,
        tool_call_id: str = "", server_id: str = "",
    ) -> "DialogBuilder":
        if not tool_call_id:
            for m in reversed(self._messages):
                if m.role == "assistant" and m.tool_calls:
                    pending = [
                        tc for tc in m.tool_calls
                        if not self._has_response_for(tc.id)
                    ]
                    for tc in pending:
                        if tc.name == name:
                            tool_call_id = tc.id
                            if not server_id:
                                server_id = tc.server_id
                            break
                    if tool_call_id:
                        break
        return self.add(Message.tool(
            name=name, content=content,
            tool_call_id=tool_call_id, server_id=server_id,
        ))

    def _has_response_for(self, tool_call_id: str) -> bool:
        return any(
            m.role == "tool" and m.tool_call_id == tool_call_id
            for m in self._messages
        )

    # ── Output ──

    def build(self) -> dict:
        """Return the full wrapped Dialog object (the canonical form)."""
        meta = dict(self._meta)
        meta.setdefault("started_at", self._started_at)
        meta.setdefault("finished_at", _utc_now_iso())
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": meta,
            "messages": [m.to_dict() for m in self._messages],
            "result": dict(self._result),
        }

    def build_messages(self) -> list[dict]:
        """Legacy/AgentCPM-compatible: return only the messages list."""
        return [m.to_dict() for m in self._messages]


# ── I/O helpers ─────────────────────────────────────────────────────────

def save_dialog(
    dialog: Any,
    path: Union[str, Path],
    *,
    indent: int = 2,
    create_parents: bool = True,
) -> Path:
    """Write a dialog to disk in the canonical wrapped form.

    Accepts:
      * Wrapped Dialog dict  → written verbatim
      * Plain message list    → wrapped in a default object first
      * ``DialogBuilder``     → ``.build()`` is called
    """
    if isinstance(dialog, DialogBuilder):
        data = dialog.build()
    elif isinstance(dialog, dict) and "messages" in dialog:
        data = dialog
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("meta", {})
        data.setdefault("result", {})
    elif isinstance(dialog, list):
        # Legacy shape — wrap it.
        msgs = [
            m.to_dict() if isinstance(m, Message) else m
            for m in dialog
        ]
        data = {
            "schema_version": SCHEMA_VERSION,
            "meta": {},
            "messages": msgs,
            "result": {},
        }
    else:
        raise TypeError(
            f"Cannot save_dialog: unsupported type {type(dialog).__name__}")

    p = Path(path)
    if create_parents:
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
    tmp.replace(p)  # atomic on POSIX
    return p


def load_dialog(path: Union[str, Path]) -> dict:
    """Read a dialog from disk. Always returns the wrapped form,
    auto-upgrading legacy plain-list files."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        # Legacy AgentCPM-style file.
        return {
            "schema_version": "1.0",
            "meta": {},
            "messages": data,
            "result": {},
        }
    return data


# ── Accessors (work on either wrapped or legacy form) ──────────────────

def messages_of(dialog: Any) -> list[dict]:
    """Return the messages list whether ``dialog`` is wrapped or a
    plain list."""
    if isinstance(dialog, list):
        return dialog
    if isinstance(dialog, dict):
        return list(dialog.get("messages") or [])
    if isinstance(dialog, DialogBuilder):
        return dialog.build_messages()
    return []


def meta_of(dialog: Any) -> dict:
    """Return the meta dict (empty for legacy plain lists)."""
    if isinstance(dialog, dict):
        return dict(dialog.get("meta") or {})
    return {}


def result_of(dialog: Any) -> dict:
    """Return the result dict (empty for legacy plain lists)."""
    if isinstance(dialog, dict):
        return dict(dialog.get("result") or {})
    return {}


def search_tree_of(dialog: Any) -> Optional[dict]:
    """Return ``meta.search_tree`` if present, else None.

    Only set when the run used a tree-search profile. v2.0 files
    (linear / parallel / single-loop) never carry this block.
    """
    if isinstance(dialog, dict):
        meta = dialog.get("meta") or {}
        tree = meta.get("search_tree")
        return dict(tree) if isinstance(tree, dict) else None
    return None


def solved_path_of(dialog: Any) -> list[dict]:
    """Return the path-to-solution: the dialog's main ``messages`` list.

    For tree-search dialogs this is the flattened solved path (same as
    what an SFT loader sees). For linear dialogs this is just every
    message. The accessor exists so downstream code can be explicit
    about wanting the success path rather than the full tree."""
    return messages_of(dialog)


# ── Validation ──────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    index: int
    code: str
    message: str


def validate_dialog(dialog: Any) -> list[ValidationIssue]:
    """Check structural invariants over the messages list.

    Accepts wrapped Dialog OR plain message list. Returns a list of
    issues (empty = OK). Non-fatal — live/streaming traces may
    momentarily break FIFO matching of tool_call ↔ tool_response.
    """
    messages = messages_of(dialog)
    issues: list[ValidationIssue] = []
    pending: list[tuple[str, str]] = []  # (id, name) FIFO

    for i, msg in enumerate(messages):
        role = msg.get("role")

        if role not in ("user", "assistant", "tool"):
            issues.append(ValidationIssue(
                i, "unknown_role", f"role={role!r}"))
            continue

        if role == "assistant":
            tcs = msg.get("tool_calls") or []
            for tc in tcs:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                tc_id = tc.get("id", "")
                if not tc_id:
                    issues.append(ValidationIssue(
                        i, "tool_call_missing_id",
                        f"tool {name!r} has no id"))
                args = fn.get("arguments")
                if not isinstance(args, str):
                    issues.append(ValidationIssue(
                        i, "tool_args_not_string",
                        f"tool {name!r}: arguments must be a JSON-encoded "
                        f"string, got {type(args).__name__}"))
                pending.append((tc_id, name))

        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            name = msg.get("name", "")
            if not pending:
                issues.append(ValidationIssue(
                    i, "tool_response_unmatched",
                    f"tool message for {name!r} has no preceding tool_call"))
                continue
            expected_id, expected_name = pending.pop(0)
            if tc_id and expected_id and tc_id != expected_id:
                issues.append(ValidationIssue(
                    i, "tool_response_id_mismatch",
                    f"expected tool_call_id={expected_id!r}, got {tc_id!r}"))
            if name and expected_name and name != expected_name:
                issues.append(ValidationIssue(
                    i, "tool_response_name_mismatch",
                    f"expected name={expected_name!r}, got {name!r}"))

    if pending:
        issues.append(ValidationIssue(
            len(messages), "tool_calls_without_response",
            f"{len(pending)} unanswered tool_call(s): "
            f"{', '.join(n for _, n in pending)}"))

    # Optional v3.0: structural check on meta.search_tree if present.
    tree = search_tree_of(dialog)
    if tree is not None:
        issues.extend(_validate_search_tree(tree))

    return issues


def _validate_search_tree(tree: dict) -> list[ValidationIssue]:
    """Structural checks on a meta.search_tree block.

    The tree is *informational* — its absence is fine. When present we
    insist:
      • ``kind`` is one of the supported drivers
      • every node references an existing parent (or has parent_id None)
      • ``solved_node_id``, when set, names a real node
      • ``per-node messages`` (if present) are themselves valid dialog
        message dicts — they have a known role
    """
    issues: list[ValidationIssue] = []
    kind = tree.get("kind")
    if kind not in (None, "best_first", "ucb", "beam"):
        issues.append(ValidationIssue(
            -1, "search_tree_kind_unknown",
            f"meta.search_tree.kind={kind!r}"))
    nodes = tree.get("nodes") or []
    by_id: dict[int, dict] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = n.get("node_id")
        if isinstance(nid, int):
            by_id[nid] = n
    for n in nodes:
        if not isinstance(n, dict):
            issues.append(ValidationIssue(
                -1, "search_tree_node_not_dict",
                f"node entry {n!r} is not a dict"))
            continue
        pid = n.get("parent_id")
        if pid is not None and pid not in by_id:
            issues.append(ValidationIssue(
                -1, "search_tree_orphan",
                f"node {n.get('node_id')} parent_id={pid} not in tree"))
        # Per-node messages are optional but must be well-formed if there.
        for m in (n.get("messages") or []):
            if not isinstance(m, dict) or m.get("role") not in (
                    "user", "assistant", "tool"):
                issues.append(ValidationIssue(
                    -1, "search_tree_node_message_invalid",
                    f"bad message inside node {n.get('node_id')}"))
                break
    solved = tree.get("solved_node_id")
    if solved is not None and solved not in by_id:
        issues.append(ValidationIssue(
            -1, "search_tree_solved_dangling",
            f"solved_node_id={solved} not present in tree.nodes"))
    return issues


# ── Live-runtime helpers ────────────────────────────────────────────────

def is_tool_response_user_msg(msg: dict) -> bool:
    if msg.get("role") != "user":
        return False
    c = msg.get("content")
    return (isinstance(c, str)
            and "<tool_response>" in c
            and "</tool_response>" in c)


def strip_tool_response_wrapper(text: str) -> str:
    if not text:
        return ""
    m = _TOOL_RESPONSE_RE.search(text)
    return (m.group(1) or "").strip() if m else text.strip()


def split_dialog_at_markers(full_log: list[dict]) -> list[list[dict]]:
    """Split a long live conversation log on ``__CONTEXT_SPLIT__``
    markers into per-window message lists."""
    segments: list[list[dict]] = []
    current: list[dict] = []
    for msg in full_log:
        if msg.get("role") == CONTEXT_SPLIT_ROLE:
            segments.append(current)
            current = list(msg.get("next_history_segment", []))
        else:
            current.append(msg)
    segments.append(current)
    return segments


# ── Internal helpers ────────────────────────────────────────────────────

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_tool_spec(t: Any) -> dict:
    """Coerce any tool spec into the canonical {name, description,
    parameters, server_id} shape."""
    if isinstance(t, dict):
        if "function" in t and isinstance(t["function"], dict):
            fn = t["function"]
            return {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
                "server_id": t.get("server_id",
                                   DEFAULT_SERVER_MAP.get(
                                       fn.get("name", ""), "default")),
            }
        return {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("parameters",
                                t.get("input_schema", {})),
            "server_id": t.get("server_id",
                               DEFAULT_SERVER_MAP.get(
                                   t.get("name", ""), "default")),
        }
    return {
        "name": getattr(t, "name", ""),
        "description": getattr(t, "description", ""),
        "parameters": getattr(t, "input_schema",
                              getattr(t, "parameters", {})),
        "server_id": getattr(t, "server_id",
                             DEFAULT_SERVER_MAP.get(
                                 getattr(t, "name", ""), "default")),
    }


__all__ = [
    "SCHEMA_VERSION", "SUPPORTED_SCHEMA_VERSIONS",
    "DIALOG_FILENAME", "CONTEXT_SPLIT_ROLE",
    "DEFAULT_SERVER_MAP",
    "ToolCall", "Message", "DialogBuilder", "ValidationIssue",
    "save_dialog", "load_dialog", "validate_dialog",
    "messages_of", "meta_of", "result_of",
    "search_tree_of", "solved_path_of",
    "is_tool_response_user_msg", "strip_tool_response_wrapper",
    "split_dialog_at_markers",
]
