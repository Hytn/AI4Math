"""agent/persistence/sft_export.py — Convert dialogs to SFT samples

Takes a canonical dialog (the AgentCPM-aligned schema from
``dialog_format.py``) and renders it as a single training sample for
supervised fine-tuning, applying the special tokens of the target model
family.

Why this lives separately from ``dialog_format``:
  Storage format ≠ training format. The dialog on disk is model-agnostic
  (no Qwen / MiniCPM / Llama tokens leaked in). At SFT prep time, you
  pick a preset and this module emits the exact text the trainer will
  see, plus per-segment train-on/skip masks so a downstream tokenizer
  can produce the standard ``input_ids`` / ``labels`` arrays.

Supported presets (extensible — each preset is just a
``ChatTemplate`` config object):

  * ``qwen3``  — Qwen3 / Qwen2.5 chat template + ``<tool_call>``,
                 ``<tool_response>``, ``<think>``, ``<code>`` wrapping
  * ``agentcpm`` — MiniCPM / AgentCPM (the format AgentCPM itself trains
                   on, identical to what's emitted by their
                   ``_tool_call_block`` and the ``<tool_response>``
                   wrapper)
  * ``openai``  — OpenAI chat-completions JSONL (one record per dialog;
                  no special tokens — for API-style fine-tuning)

────────────────────────────────────────────────────────────────────────
Output shape
────────────────────────────────────────────────────────────────────────

``dialog_to_sft_sample(dialog, preset="qwen3")`` returns a dict::

    {
        "preset": "qwen3",
        "text": "<full rendered conversation as one string>",
        "segments": [
            {"text": "...", "role": "system",     "trainable": False},
            {"text": "...", "role": "user",       "trainable": False},
            {"text": "...", "role": "assistant",  "trainable": True},
            {"text": "...", "role": "tool",       "trainable": False},
            ...
        ],
    }

The trainable flag tells the tokenizer which segments to keep in
``labels`` and which to mask with -100. ``text`` is the concatenation of
all segments' ``text`` (so ``"".join(s["text"] for s in segments) ==
text``).

For the openai preset, ``dialog_to_sft_sample`` returns the message
array directly (no rendered text, since OpenAI tokenizes server-side).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Union

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Chat-template configuration
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class ChatTemplate:
    """Per-model-family rendering configuration.

    The conversation is rendered as repeated ``turn_open + role +
    role_sep + body + turn_close`` blocks. ``body`` for an assistant
    turn may itself wrap thought / tool_call sections in their own tags.
    """
    name: str

    # Per-turn delimiters
    turn_open: str = "<|im_start|>"
    turn_close: str = "<|im_end|>\n"
    role_sep: str = "\n"

    # Wrappers inside an assistant turn
    think_open: str = "<think>\n"
    think_close: str = "\n</think>\n\n"
    tool_call_open: str = "<tool_call>\n"
    tool_call_close: str = "\n</tool_call>"
    code_open: str = ""           # "" = no special code wrapping; preset
    code_close: str = ""          # may set ``<code>``/``</code>`` or similar

    # Wrappers inside a tool turn (when rendering the tool response back
    # into a user-role wrapper, AgentCPM-style)
    tool_response_open: str = "<tool_response>\n"
    tool_response_close: str = "\n</tool_response>"

    # System prompt placement
    system_role: str = "system"
    user_role: str = "user"
    assistant_role: str = "assistant"
    tool_role: str = "tool"          # may be "user" for models without
                                     # native tool role (AgentCPM does this)
    wrap_tool_in_user: bool = False  # if True, tool messages become
                                     # role:user content wrapped in
                                     # <tool_response>...</tool_response>

    # Where the trainable region begins for assistant turns. The Qwen
    # convention is to train on everything after the role tag
    # (``"assistant\n"``) including the closing ``<|im_end|>``.
    assistant_trainable_includes_role: bool = False
    assistant_trainable_includes_close: bool = True

    # Whether to extract fenced ``lean`` blocks from assistant content
    # and re-wrap them in ``code_open``/``code_close``. For Qwen3 with
    # ``<code></code>`` this is True.
    rewrap_lean_code_blocks: bool = False

    # Whether to embed tool_calls as <tool_call> blocks INSIDE assistant
    # content (Qwen3, AgentCPM) vs. as a separate API-level field
    # (openai). When True, the assistant ``content`` is augmented with
    # ``<tool_call>{name, arguments}</tool_call>`` blocks before the
    # turn is closed.
    inline_tool_calls: bool = True

    # Optional stop-token to append after the final assistant turn so
    # the trainer learns to terminate. Empty = use ``turn_close``.
    final_stop: str = ""


# Built-in presets ───────────────────────────────────────────────────────

QWEN3_PRESET = ChatTemplate(
    name="qwen3",
    turn_open="<|im_start|>",
    turn_close="<|im_end|>\n",
    think_open="<think>\n",
    think_close="\n</think>\n\n",
    tool_call_open="<tool_call>\n",
    tool_call_close="\n</tool_call>",
    code_open="<code>\n",
    code_close="\n</code>",
    tool_response_open="<tool_response>\n",
    tool_response_close="\n</tool_response>",
    tool_role="user",            # Qwen3 emits tool results as role=user
    wrap_tool_in_user=True,
    rewrap_lean_code_blocks=True,
    inline_tool_calls=True,
)

AGENTCPM_PRESET = ChatTemplate(
    name="agentcpm",
    # AgentCPM's runtime uses <tool_call>/<tool_response> blocks inside
    # a chatml-ish frame. See AgentCPM data_test_copy.py:_tool_call_block
    # and the tool_message_to_log construction (rolled into role=user).
    turn_open="<|im_start|>",
    turn_close="<|im_end|>\n",
    think_open="<think>\n",
    think_close="\n</think>\n\n",
    tool_call_open="<tool_call>\n",
    tool_call_close="\n</tool_call>\n\n",
    tool_response_open="<tool_response>\n",
    tool_response_close="\n</tool_response>",
    tool_role="user",            # AgentCPM also routes tool results
    wrap_tool_in_user=True,      # back through role=user
    rewrap_lean_code_blocks=False,
    inline_tool_calls=True,
)

OPENAI_PRESET = ChatTemplate(
    name="openai",
    # OpenAI fine-tuning consumes a JSONL of message arrays — it does
    # not see special tokens at all. For this preset we don't render
    # text; ``dialog_to_sft_sample`` short-circuits.
    turn_open="",
    turn_close="",
    inline_tool_calls=False,     # tool_calls remain a structured field
)

PRESETS: dict[str, ChatTemplate] = {
    "qwen3": QWEN3_PRESET,
    "agentcpm": AGENTCPM_PRESET,
    "openai": OPENAI_PRESET,
}


# ─────────────────────────────────────────────────────────────────────────
# Sample builder
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    """One contiguous slice of the rendered conversation."""
    text: str
    role: str
    trainable: bool

    def to_dict(self) -> dict:
        return {"text": self.text, "role": self.role,
                "trainable": self.trainable}


def dialog_to_sft_sample(
    dialog: Any,
    preset: Union[str, ChatTemplate] = "qwen3",
    system_prompt: str = "",
    *,
    drop_thoughts: bool = False,
    train_on_tool_calls: bool = True,
) -> dict:
    """Render a dialog as a single SFT sample.

    Args:
      dialog: Either a wrapped Dialog dict ({meta, messages, result})
              OR a legacy plain message list. The system prompt is
              auto-extracted from ``dialog.meta.system_prompt`` when
              not supplied explicitly.
      preset: either a string key in ``PRESETS`` or a custom
              ``ChatTemplate``.
      system_prompt: override the prompt baked into the dialog. If not
              given, ``meta.system_prompt`` is used.
      drop_thoughts: if True, omit ``thought`` content entirely.
      train_on_tool_calls: if False, mask the ``<tool_call>`` blocks too.

    Returns:
      ``{"preset": str, "text": str, "segments": [...]}`` for text-based
      presets, or the OpenAI message array for the openai preset.
    """
    from agent.persistence.dialog_format import messages_of, meta_of
    template = preset if isinstance(preset, ChatTemplate) \
        else PRESETS.get(preset)
    if template is None:
        raise ValueError(f"Unknown preset: {preset!r}. "
                         f"Available: {list(PRESETS)}")

    messages = messages_of(dialog)
    meta = meta_of(dialog)
    if not system_prompt:
        system_prompt = meta.get("system_prompt", "") or ""

    if template.name == "openai":
        return _to_openai_sft(messages, system_prompt=system_prompt)

    segments: list[Segment] = []
    if system_prompt:
        segments.extend(_render_turn(
            template, role=template.system_role,
            body=system_prompt, trainable=False,
        ))

    for msg in messages:
        role = msg.get("role")
        if role == "user":
            segments.extend(_render_turn(
                template, role=template.user_role,
                body=msg.get("content", "") or "",
                trainable=False,
            ))
        elif role == "assistant":
            segments.extend(_render_assistant_turn(
                template, msg,
                drop_thoughts=drop_thoughts,
                train_on_tool_calls=train_on_tool_calls,
            ))
        elif role == "tool":
            segments.extend(_render_tool_turn(template, msg))
        else:
            logger.debug("Skipping unknown role %r", role)

    text = "".join(s.text for s in segments)
    return {
        "preset": template.name,
        "text": text,
        "segments": [s.to_dict() for s in segments],
    }


# ─────────────────────────────────────────────────────────────────────────
# Per-role rendering
# ─────────────────────────────────────────────────────────────────────────

def _render_turn(
    t: ChatTemplate, *, role: str, body: str, trainable: bool,
) -> list[Segment]:
    """Render a simple non-assistant turn as one or two segments."""
    head = t.turn_open + role + t.role_sep
    tail = t.turn_close
    if trainable:
        # Ordinarily we don't train on user/system, but this branch is
        # here for symmetry / future use.
        return [Segment(head + body + tail, role, True)]
    return [Segment(head + body + tail, role, False)]


def _render_assistant_turn(
    t: ChatTemplate, msg: dict, *,
    drop_thoughts: bool, train_on_tool_calls: bool,
) -> list[Segment]:
    """Assistant turns are split into a non-trainable head (the role
    tag), a trainable body (thought + content + tool_calls), and a
    closing tag whose trainability depends on the template.

    The split is what lets a downstream tokenizer build the SFT label
    mask: head → -100, body → token IDs, close → token IDs (so the model
    learns to emit the end-of-turn).
    """
    head = t.turn_open + t.assistant_role + t.role_sep
    head_seg = Segment(
        head, "assistant", t.assistant_trainable_includes_role)

    parts: list[str] = []

    # 1. Thought
    thought = msg.get("thought") if not drop_thoughts else None
    if thought:
        parts.append(t.think_open + thought.rstrip() + t.think_close)

    # 2. Visible content (with optional Lean-code rewrapping)
    content = msg.get("content", "") or ""
    if t.rewrap_lean_code_blocks and (t.code_open or t.code_close):
        content = _rewrap_lean_blocks(content, t.code_open, t.code_close)
    if content:
        parts.append(content)

    # 3. Inline tool_call blocks
    tcs = msg.get("tool_calls") or []
    if tcs and t.inline_tool_calls:
        for tc in tcs:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "")
            try:
                args_obj = json.loads(args_raw) \
                    if isinstance(args_raw, str) else args_raw
            except (json.JSONDecodeError, TypeError):
                args_obj = args_raw
            block_payload = json.dumps(
                {"name": name, "arguments": args_obj},
                ensure_ascii=False,
            )
            parts.append(t.tool_call_open + block_payload + t.tool_call_close)

    body_text = "".join(parts)
    body_seg = Segment(body_text, "assistant", train_on_tool_calls or bool(content))
    # If the caller said *don't* train on tool_calls but we have a
    # content/thought portion before them, we need to split the body
    # into a trainable prefix and a non-trainable suffix.
    if tcs and not train_on_tool_calls and t.inline_tool_calls and (thought or content):
        prefix = "".join(parts[:-len(tcs)])
        tool_part = "".join(parts[-len(tcs):])
        body_segs = [
            Segment(prefix, "assistant", True),
            Segment(tool_part, "assistant", False),
        ]
    else:
        body_segs = [body_seg]

    close = t.turn_close
    close_seg = Segment(
        close, "assistant", t.assistant_trainable_includes_close)

    return [head_seg, *body_segs, close_seg]


def _render_tool_turn(t: ChatTemplate, msg: dict) -> list[Segment]:
    """Tool responses can either get their own ``role: "tool"`` turn or
    be folded into a user-role turn wrapped in
    ``<tool_response>...</tool_response>`` (AgentCPM convention)."""
    content = msg.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)

    if t.wrap_tool_in_user:
        body = t.tool_response_open + content + t.tool_response_close
        return _render_turn(
            t, role=t.user_role, body=body, trainable=False,
        )
    # Native tool role
    head = t.turn_open + t.tool_role + t.role_sep
    tail = t.turn_close
    return [Segment(head + content + tail, "tool", False)]


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

_LEAN_BLOCK = "```lean"
_FENCE_END = "```"


def _rewrap_lean_blocks(text: str, open_tag: str, close_tag: str) -> str:
    """Replace ```` ```lean ... ``` ```` fenced blocks with
    ``open_tag ... close_tag``.

    This is what lets a Qwen3-style model see ``<code>...</code>`` in
    training and at inference produce code blocks that the runtime can
    automatically extract & execute (e.g. via Lean verification),
    matching the user's intent of "<code></code> wraps code segments
    that get auto-parsed".
    """
    if _LEAN_BLOCK not in text:
        return text
    out: list[str] = []
    i = 0
    while True:
        start = text.find(_LEAN_BLOCK, i)
        if start == -1:
            out.append(text[i:])
            break
        out.append(text[i:start])
        body_start = start + len(_LEAN_BLOCK)
        # Skip optional trailing newline after ```lean
        if body_start < len(text) and text[body_start] == "\n":
            body_start += 1
        end = text.find(_FENCE_END, body_start)
        if end == -1:
            # Unterminated block — leave the rest as-is.
            out.append(text[start:])
            break
        body = text[body_start:end]
        # Trim a trailing newline if present (the ``` typically follows one)
        if body.endswith("\n"):
            body = body[:-1]
        out.append(open_tag + body + close_tag)
        i = end + len(_FENCE_END)
    return "".join(out)


# ─────────────────────────────────────────────────────────────────────────
# OpenAI / API-style preset
# ─────────────────────────────────────────────────────────────────────────

def _to_openai_sft(dialog: list[dict], system_prompt: str = "") -> dict:
    """Render a dialog as an OpenAI fine-tuning JSONL record."""
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for m in dialog:
        role = m.get("role")
        if role == "user":
            msgs.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            entry: dict = {"role": "assistant",
                           "content": m.get("content", "") or ""}
            if m.get("tool_calls"):
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": tc.get("function", {}),
                    }
                    for tc in m["tool_calls"]
                ]
            msgs.append(entry)
        elif role == "tool":
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            msgs.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", ""),
                "name": m.get("name", ""),
                "content": content,
            })
    return {"preset": "openai", "messages": msgs}


# ─────────────────────────────────────────────────────────────────────────
# Bulk helpers
# ─────────────────────────────────────────────────────────────────────────

def write_sft_jsonl(
    samples: list[dict],
    path: str,
) -> int:
    """Write a list of SFT samples to a .jsonl file. Returns the count."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    return n


def dialogs_to_sft_jsonl(
    dialogs: list,                    # list of wrapped Dialog dicts OR list[list[dict]]
    output_path: str,
    preset: Union[str, ChatTemplate] = "qwen3",
    system_prompt: str = "",
    *,
    drop_thoughts: bool = False,
    skip_invalid: bool = True,
) -> int:
    """End-to-end: take a batch of dialogs and write an SFT-ready
    .jsonl file. Returns the number of samples written.

    Each dialog may be a wrapped Dialog (preferred) or a legacy plain
    message list. When wrapped, ``meta.system_prompt`` is used unless
    explicitly overridden.

    Pair this with the adapters in ``dialog_adapters.py`` to convert
    legacy AI4Math trajectories straight into SFT data:

        from agent.persistence.unified_storage import collect_dialogs
        from agent.persistence.sft_export import dialogs_to_sft_jsonl

        items = collect_dialogs("results/traces")
        dialogs_to_sft_jsonl(
            [d for _, d in items], "data/sft.jsonl", preset="qwen3",
        )
    """
    from agent.persistence.dialog_format import validate_dialog
    samples: list[dict] = []
    for d in dialogs:
        if skip_invalid:
            issues = validate_dialog(d)
            if any(i.code in {"unknown_role",
                              "tool_response_unmatched"}
                   for i in issues):
                logger.warning(
                    "Skipping dialog with %d structural issue(s)",
                    len(issues))
                continue
        samples.append(dialog_to_sft_sample(
            d, preset=preset, system_prompt=system_prompt,
            drop_thoughts=drop_thoughts,
        ))
    return write_sft_jsonl(samples, output_path)


__all__ = [
    "ChatTemplate",
    "QWEN3_PRESET", "AGENTCPM_PRESET", "OPENAI_PRESET",
    "PRESETS",
    "Segment",
    "dialog_to_sft_sample",
    "write_sft_jsonl",
    "dialogs_to_sft_jsonl",
]
