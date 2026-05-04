"""agent/persistence/unified_storage.py — Single-file task storage

The canonical layout:

    <task_dir>/
        dialog.json       — everything (system prompt, tools, every
                            turn, results) in one self-contained file

That is the entire on-disk representation. There is no separate
``meta_config.json`` or ``result.json``. All metadata lives inside the
wrapped Dialog object's ``meta`` and ``result`` blocks. To debug or
re-train from a run, you open exactly one file and see everything an
LLM agent saw and produced.

Usage::

    from agent.persistence.unified_storage import save_task, load_task

    save_task("results/traces/nat_add_zero", dialog)
    loaded = load_task("results/traces/nat_add_zero")
    # → wrapped Dialog dict
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from agent.persistence.dialog_format import (
    DIALOG_FILENAME, save_dialog, load_dialog, validate_dialog,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Save / load — single file
# ─────────────────────────────────────────────────────────────────────────

def save_task(
    task_dir: Union[str, Path],
    dialog: Any,
    *,
    validate: bool = True,
) -> Path:
    """Write a task's dialog.json to ``task_dir/dialog.json``.

    Args:
      task_dir: Directory to write into (created if missing).
      dialog:   Wrapped Dialog dict, plain message list, or
                ``DialogBuilder``. ``save_dialog`` handles each form.
      validate: If True, run structural validation and log any issues.

    Returns:
      The full path to the written ``dialog.json``.
    """
    p = Path(task_dir)
    p.mkdir(parents=True, exist_ok=True)
    if validate:
        issues = validate_dialog(dialog)
        if issues:
            logger.warning(
                "Dialog has %d structural issue(s) (saving anyway): %s",
                len(issues),
                "; ".join(f"[{i.code}]" for i in issues[:5]),
            )
    return save_dialog(dialog, p / DIALOG_FILENAME)


def load_task(task_dir: Union[str, Path]) -> Optional[dict]:
    """Read ``task_dir/dialog.json``, returning the wrapped Dialog
    dict (auto-upgrading any legacy plain-list files). Returns None
    if the file doesn't exist."""
    p = Path(task_dir) / DIALOG_FILENAME
    if not p.exists():
        return None
    return load_dialog(p)


# v13: 删除 ``save_task_outputs`` / ``load_task_outputs`` back-compat
# alias —— v9 删除了所有可能的旧调用方, 这两个 alias 之后 0 处使用。


# ─────────────────────────────────────────────────────────────────────────
# Bulk operations
# ─────────────────────────────────────────────────────────────────────────

def collect_dialogs(
    root_dir: Union[str, Path],
) -> list[tuple[Path, dict]]:
    """Walk a tree of task directories and yield every dialog.json.
    Each item is ``(path, wrapped_dialog_dict)``. Useful for batch SFT
    prep::

        from agent.persistence.unified_storage import collect_dialogs
        from agent.persistence.sft_export import dialogs_to_sft_jsonl

        items = collect_dialogs("results/traces")
        dialogs_to_sft_jsonl(
            [d for _, d in items], "data/sft.jsonl", preset="qwen3",
        )
    """
    root = Path(root_dir)
    found: list[tuple[Path, dict]] = []
    if not root.exists():
        return found
    for dialog_path in sorted(root.rglob(DIALOG_FILENAME)):
        try:
            data = load_dialog(dialog_path)
            found.append((dialog_path, data))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skipping unreadable %s: %s", dialog_path, e)
    return found


# ─────────────────────────────────────────────────────────────────────────
# Convenience builders (used by the trajectory classes' save_unified)
# ─────────────────────────────────────────────────────────────────────────

def build_meta(
    *,
    task_id: str = "",
    problem_id: str = "",
    problem_name: str = "",
    theorem_statement: str = "",
    informal_statement: str = "",
    model: str = "",
    provider: str = "",
    system_prompt: str = "",
    tools: Optional[list] = None,
    extra: Optional[dict] = None,
    started_at: str = "",
    finished_at: str = "",
) -> dict:
    """Construct the ``meta`` block of a Dialog. Empty fields are
    omitted to keep the on-disk output compact."""
    meta: dict[str, Any] = {}
    for k, v in [
        ("task_id", task_id),
        ("problem_id", problem_id),
        ("problem_name", problem_name),
        ("theorem_statement", theorem_statement),
        ("informal_statement", informal_statement),
        ("model", model),
        ("provider", provider),
        ("system_prompt", system_prompt),
    ]:
        if v:
            meta[k] = v
    meta["started_at"] = started_at or _utc_now_iso()
    meta["finished_at"] = finished_at or _utc_now_iso()
    if tools:
        meta["tools"] = list(tools)
    if extra:
        meta["extra"] = dict(extra)
    return meta


def build_result(
    *,
    success: bool = False,
    total_attempts: int = 0,
    total_tokens: int = 0,
    total_duration_ms: int = 0,
    successful_proof: str = "",
    termination: str = "",
    error_distribution: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Construct the ``result`` block of a Dialog."""
    res: dict[str, Any] = {
        "success": bool(success),
        "total_attempts": int(total_attempts),
        "total_tokens": int(total_tokens),
        "total_duration_ms": int(total_duration_ms),
    }
    if successful_proof:
        res["successful_proof"] = successful_proof
    if termination:
        res["termination"] = termination
    if error_distribution:
        res["error_distribution"] = dict(error_distribution)
    if extra:
        res["extra"] = dict(extra)
    return res


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "DIALOG_FILENAME",
    "save_task", "load_task",
    "collect_dialogs",
    "build_meta", "build_result",
]
