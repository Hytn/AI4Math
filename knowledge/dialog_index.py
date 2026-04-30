"""knowledge/dialog_index.py — Cross-problem dialog retrieval.

The project's "Living Knowledge System" has a four-layer pyramid:

  Layer 0 (raw traces) ─→ Layer 1 (tactic effectiveness) ─→ Layer 2 / 3 …

Layers 1-3 already serve the agent loop via ``KnowledgeReader``. Layer 0
is stored — every solved proof leaves a row in ``proof_contexts`` with
its full ``tactic_history``, and every ``run_eval.py`` run drops a
``dialog.json`` file per problem — but no reader has ever pulled those
back into a new run's prompt.

That's the gap this module closes. It indexes past dialogs by their
*theorem text*, then on a new query returns the most-similar past
sessions as in-context demonstrations: "Last time you saw a theorem
shaped like this one, here is the proof you wound up with."

This is the "demo by past success" signal the project's own
:file:`REFACTOR_REPORT.md` § 九.4 calls out as a strong but
unimplemented prompt input.

Two ingest paths are supported:

* **From the SQLite ``proof_contexts`` table** — used in long-running
  agent processes that keep a ``UnifiedKnowledgeStore``. Every save()
  call adds a candidate.
* **From a directory of ``dialog.json`` files** — used after a batch
  ``run_eval.py`` sweep. The eval writes one dialog per problem to
  ``results/traces/<id>/dialog.json``; the next sweep can index those
  on startup so iteration N benefits from iteration N-1.

The index is in-memory and rebuilt on each ``index_*`` call. It uses
the existing :class:`KnowledgeTFIDFRetriever` (BM25 + char-n-gram
TF-IDF) so retrieval semantics match Layer 1 lemma search.

Public API
----------

* :class:`DialogIndex` — the index itself.
* :class:`SimilarDialogMatch` — one retrieved dialog plus extracted
  metadata (final proof, tactic list, solved status, source).
* :func:`extract_final_proof` / :func:`extract_used_tactics` —
  internal helpers exposed so other code can pull the same fields out
  of a dialog.json without re-implementing the parsing.

Usage::

    from knowledge.dialog_index import DialogIndex

    index = DialogIndex()
    index.index_from_directory("results/traces")          # ingest disk dialogs
    index.index_from_proof_context_store(store)            # ingest DB rows

    matches = index.find_similar(
        theorem="theorem foo (n : ℕ) : n + 0 = n", top_k=3)
    for m in matches:
        print(f"{m.score:.3f}  {m.theorem[:40]}…  solved={m.solved}")

    text = index.render_for_prompt(
        theorem="theorem foo (n : ℕ) : n + 0 = n", top_k=3,
        max_chars=2000)
    # → "## Past similar work\\n### 1. ...\\n```lean\\n...\\n```\\n …"

The class is fully fail-soft — if a dialog file is malformed, the
file is skipped with a debug log and ingest continues. If the index
is empty, every query returns ``[]`` / ``""``.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from knowledge.tfidf_retriever import KnowledgeTFIDFRetriever

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Public dataclasses
# ─────────────────────────────────────────────────────────────────────


@dataclass
class SimilarDialogMatch:
    """One past dialog returned by similarity search.

    Attributes:
      theorem:      The theorem statement that dialog was attempting.
      score:        Combined BM25 + TF-IDF similarity (higher is closer).
      solved:       Whether the past attempt finished successfully.
      source:       Provenance string (``'db:42'`` for proof_contexts row 42,
                    ``'file:results/traces/x/dialog.json'`` for disk).
      final_proof:  Best-guess of the ``by ...`` proof body if the dialog
                    was solved. May be empty.
      used_tactics: Sequence of tactic strings observed during the dialog.
                    For DB rows: ``state.tactic_history``. For disk
                    dialogs: extracted from ``tactic_apply`` /
                    ``lean_verify`` tool-call arguments.
      timestamp:    Wall-clock time the dialog was recorded (used when
                    multiple identical-theorem entries exist).
    """
    theorem: str
    score: float
    solved: bool
    source: str
    final_proof: str = ""
    used_tactics: list[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class _DialogEntry:
    """Internal: one row in the DialogIndex."""
    theorem: str
    solved: bool
    final_proof: str = ""
    used_tactics: list[str] = field(default_factory=list)
    source: str = "memory"
    timestamp: float = 0.0


# ─────────────────────────────────────────────────────────────────────
# Helpers (also exposed so other code can reuse the parsing rules)
# ─────────────────────────────────────────────────────────────────────


def extract_final_proof(dialog: dict) -> str:
    """Pull the final ``by …`` proof body out of a wrapped dialog dict.

    Looks (in order) at:
      1. ``dialog["result"]["successful_proof"]`` — the canonical field
         set by ``UnifiedResult.save_unified``.
      2. The last assistant message containing a ```lean``` fence.

    Returns empty string when nothing matches; the caller decides what
    to do.
    """
    if not isinstance(dialog, dict):
        return ""
    result = dialog.get("result") or {}
    final = result.get("successful_proof") or ""
    if isinstance(final, str) and final.strip():
        return final.strip()

    # Fallback: scan messages for the last ```lean``` block.
    msgs = dialog.get("messages") or []
    for msg in reversed(msgs):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        if not isinstance(content, str):
            continue
        # Find ```lean … ```
        marker = "```lean"
        i = content.find(marker)
        if i < 0:
            continue
        j = content.find("```", i + len(marker))
        if j < 0:
            continue
        body = content[i + len(marker):j].strip()
        if body:
            return body
    return ""


def extract_used_tactics(dialog: dict, *,
                         tool_names: tuple[str, ...] = (
                             "tactic_apply", "lean_verify",
                             "tactic_suggest", "lean_auto",
                             "lemma_by_lemma",
                         )) -> list[str]:
    """Pull tactic strings out of a wrapped dialog's tool_calls.

    Walks every assistant message's ``tool_calls`` and, for any call
    whose function name appears in ``tool_names``, pulls the
    ``tactic`` / ``proof`` argument value. Returns the strings in
    order. Duplicates are NOT removed (they may carry different
    contexts).
    """
    out: list[str] = []
    if not isinstance(dialog, dict):
        return out
    for msg in dialog.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            if name not in tool_names:
                continue
            args = fn.get("arguments")
            # ``arguments`` can be a JSON string or a dict.
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(args, dict):
                continue
            for key in ("tactic", "proof", "code"):
                v = args.get(key)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
                    break
    return out


def _dialog_solved(dialog: dict) -> bool:
    """Best-effort solved flag from a wrapped dialog dict."""
    if not isinstance(dialog, dict):
        return False
    result = dialog.get("result") or {}
    if isinstance(result.get("success"), bool):
        return result["success"]
    return result.get("termination") == "success"


def _dialog_theorem(dialog: dict) -> str:
    """Best-effort theorem text from a wrapped dialog dict."""
    if not isinstance(dialog, dict):
        return ""
    meta = dialog.get("meta") or {}
    t = meta.get("theorem_statement")
    if isinstance(t, str) and t.strip():
        return t.strip()
    # Some older runs put it in ``extra``.
    extra = meta.get("extra") or {}
    t = extra.get("theorem_statement") or extra.get("theorem")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return ""


# ─────────────────────────────────────────────────────────────────────
# DialogIndex
# ─────────────────────────────────────────────────────────────────────


class DialogIndex:
    """In-memory similarity index over saved proof dialogs.

    The index treats each past dialog as a *document* whose text is the
    theorem statement. Retrieval is BM25 + char-n-gram TF-IDF over that
    text. We deliberately do NOT index the proof body: similar
    theorems often have wildly different proofs, and we want
    *retrieval to be driven by the new theorem*, not by what the agent
    happened to write last time.

    Multiple entries for the same theorem are kept; ``find_similar``
    deduplicates after scoring (keeping the latest solved entry).
    """

    def __init__(self):
        self._entries: list[_DialogEntry] = []
        self._retriever: Optional[KnowledgeTFIDFRetriever] = None
        self._dirty = False  # set when entries change; cleared on rebuild

    # ── Ingest paths ─────────────────────────────────────────────────

    def add_dialog(self, dialog: dict, *,
                   source: str = "memory",
                   timestamp: Optional[float] = None) -> bool:
        """Add one wrapped-dialog dict.

        Returns True if added, False if the dialog had no theorem text
        (we silently skip those — they can't be matched anyway).
        """
        theorem = _dialog_theorem(dialog)
        if not theorem:
            return False
        entry = _DialogEntry(
            theorem=theorem,
            solved=_dialog_solved(dialog),
            final_proof=extract_final_proof(dialog),
            used_tactics=extract_used_tactics(dialog),
            source=source,
            timestamp=timestamp if timestamp is not None else time.time(),
        )
        self._entries.append(entry)
        self._dirty = True
        return True

    def index_from_directory(
            self, dir_path: Union[str, Path], *,
            recursive: bool = True,
            limit: Optional[int] = None) -> int:
        """Scan a directory tree for ``dialog.json`` files and ingest them.

        Args:
          dir_path:  Root to scan. If absent, returns 0 silently.
          recursive: If True (default), descend into subdirectories.
          limit:     Maximum number of files to ingest (for large sweeps).

        Returns the number of dialogs successfully added. Malformed JSON
        and dialogs without a theorem statement are skipped with a
        debug log.
        """
        root = Path(dir_path)
        if not root.exists() or not root.is_dir():
            return 0

        if recursive:
            paths = list(root.rglob("dialog.json"))
        else:
            paths = list(root.glob("dialog.json"))
        # Sort for deterministic order — matters for test reproducibility.
        paths.sort()

        added = 0
        for p in paths:
            if limit is not None and added >= limit:
                break
            try:
                with open(p, "r", encoding="utf-8") as f:
                    dialog = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.debug(f"DialogIndex: skipping {p}: {e}")
                continue
            if not isinstance(dialog, dict):
                # Legacy plain-list dialogs have no meta block, so no
                # theorem text — can't index them.
                continue
            try:
                ts = p.stat().st_mtime
            except OSError:
                ts = time.time()
            if self.add_dialog(dialog, source=f"file:{p}", timestamp=ts):
                added += 1
        if added:
            logger.info(
                f"DialogIndex: ingested {added} dialog(s) from {root}")
        return added

    def index_from_proof_context_store(self, store) -> int:
        """Pull all rows from ``proof_contexts`` and ingest each one.

        ``store`` must be a ``ProofContextStore`` or a subclass (such as
        ``UnifiedKnowledgeStore``). The method uses the store's private
        ``_connect()`` because there is no public list-all method — but
        if the store doesn't expose ``_connect`` we fail soft and
        return 0.
        """
        connect = getattr(store, "_connect", None)
        if connect is None:
            logger.debug(
                "DialogIndex: store has no _connect(); skipping DB ingest")
            return 0

        try:
            with connect() as conn:
                rows = conn.execute(
                    "SELECT id, theorem, state_json, solved, updated_at "
                    "FROM proof_contexts "
                    "ORDER BY updated_at DESC"
                ).fetchall()
        except Exception as e:
            logger.warning(f"DialogIndex: DB read failed: {e}")
            return 0

        added = 0
        for row in rows:
            try:
                state_json = row["state_json"]
                state = json.loads(state_json)
                tactics = state.get("tactic_history") or []
                if not isinstance(tactics, list):
                    tactics = []
                tactics = [str(t) for t in tactics if t]
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                logger.debug(f"DialogIndex: row {row['id']} malformed: {e}")
                continue
            theorem = row["theorem"] or ""
            if not theorem.strip():
                continue
            solved = bool(row["solved"])
            final_proof = ""
            if solved and tactics:
                # Best reconstruction: ``by`` + tab-indented tactic lines.
                final_proof = "by\n  " + "\n  ".join(tactics)
            entry = _DialogEntry(
                theorem=theorem.strip(),
                solved=solved,
                final_proof=final_proof,
                used_tactics=tactics,
                source=f"db:{row['id']}",
                timestamp=float(row["updated_at"] or 0),
            )
            self._entries.append(entry)
            self._dirty = True
            added += 1

        if added:
            logger.info(
                f"DialogIndex: ingested {added} row(s) from proof_contexts")
        return added

    def clear(self) -> None:
        """Drop all entries. Used by tests."""
        self._entries.clear()
        self._retriever = None
        self._dirty = False

    @property
    def size(self) -> int:
        return len(self._entries)

    # ── Retrieval ────────────────────────────────────────────────────

    def find_similar(
            self, theorem: str, *,
            top_k: int = 3,
            solved_only: bool = True,
            min_score: float = 0.0) -> list[SimilarDialogMatch]:
        """Return the top-K most similar past dialogs.

        Args:
          theorem:      The query theorem statement.
          top_k:        Maximum number of matches to return.
          solved_only:  If True (default), only return solved dialogs —
                        unsolved past attempts are usually a poor demo.
          min_score:    Drop matches below this combined BM25+TF-IDF
                        score. ``0.0`` returns everything ranked.

        Returns matches sorted by score descending. When the index is
        empty or contains no eligible entries, returns ``[]``.
        """
        if not theorem or not theorem.strip():
            return []
        candidates = (
            [e for e in self._entries if e.solved]
            if solved_only else list(self._entries))
        if not candidates:
            return []

        self._rebuild_if_needed(candidates)
        if self._retriever is None:
            return []

        # The retriever returns ScoredLemma objects keyed by ``name``.
        # We assigned unique synthetic names ``__entry_<i>`` at index
        # time so we can map back unambiguously even when multiple
        # entries share the same ``source`` (e.g. several in-memory
        # adds with the default ``source='memory'``).
        scored = self._retriever.search(theorem, top_k=top_k * 4)
        if not scored:
            return []

        by_key: dict[str, _DialogEntry] = {
            f"__entry_{i}": e for i, e in enumerate(candidates)}
        out: list[SimilarDialogMatch] = []
        seen_theorems: set[str] = set()

        for sl in scored:
            if sl.score < min_score:
                continue
            entry = by_key.get(sl.name)
            if entry is None:
                continue
            # Deduplicate by theorem text — keep highest-scoring entry
            # per theorem (which is the one we hit first since sorted
            # by score).
            if entry.theorem in seen_theorems:
                continue
            seen_theorems.add(entry.theorem)
            out.append(SimilarDialogMatch(
                theorem=entry.theorem,
                score=sl.score,
                solved=entry.solved,
                source=entry.source,
                final_proof=entry.final_proof,
                used_tactics=entry.used_tactics,
                timestamp=entry.timestamp,
            ))
            if len(out) >= top_k:
                break
        return out

    # ── Prompt rendering ─────────────────────────────────────────────

    def render_for_prompt(
            self, theorem: str, *,
            top_k: int = 3,
            max_chars: int = 2000,
            solved_only: bool = True,
            heading: str = "## Past similar work") -> str:
        """Render top matches as a markdown block for prompt injection.

        Output shape::

            ## Past similar work

            ### 1. (similarity 0.62, solved)
            ```lean
            theorem old_one ... := by ...
            ```

            ### 2. (similarity 0.41, solved)
            ```lean
            ...
            ```

        Returns empty string when no matches survive filtering, so the
        caller can do ``parts.append(text)`` unconditionally and the
        absence of similar work is a no-op.
        """
        matches = self.find_similar(
            theorem, top_k=top_k, solved_only=solved_only)
        if not matches:
            return ""

        parts: list[str] = [heading, ""]
        for i, m in enumerate(matches, 1):
            status = "solved" if m.solved else "unsolved"
            parts.append(
                f"### {i}. (similarity {m.score:.3f}, {status})")
            # Theorem statement — sometimes long; truncate to keep
            # demos readable.
            theorem_line = m.theorem.strip()
            if len(theorem_line) > 240:
                theorem_line = theorem_line[:240] + "…"
            block = ["```lean", theorem_line]
            if m.final_proof:
                # Preserve original indentation; truncate hugely long
                # proofs (this is a demo, not a recipe).
                fp = m.final_proof
                if len(fp) > 800:
                    fp = fp[:800] + "\n  -- … (truncated)"
                block.append(fp)
            elif m.used_tactics:
                block.append("by")
                for t in m.used_tactics[:12]:
                    block.append(f"  {t}")
                if len(m.used_tactics) > 12:
                    block.append("  -- … (truncated)")
            block.append("```")
            parts.append("\n".join(block))
            parts.append("")  # blank line between entries

        rendered = "\n".join(parts).rstrip() + "\n"
        if len(rendered) > max_chars:
            # Truncate at line boundary so we don't cut a code fence
            # mid-string; if we can't find one, cut at max_chars.
            cut = rendered.rfind("\n", 0, max_chars)
            if cut < 0:
                cut = max_chars
            rendered = rendered[:cut].rstrip() + (
                "\n\n_… (truncated for length)_\n")
        return rendered

    # ── Internal ────────────────────────────────────────────────────

    def _rebuild_if_needed(self, candidates: list[_DialogEntry]) -> None:
        """Build the underlying TF-IDF retriever if dirty.

        We always rebuild over ``candidates`` (not ``self._entries``)
        so ``solved_only=True`` doesn't bleed unsolved entries into the
        IDF statistics. Keys are synthetic ``__entry_<i>`` strings so
        multiple entries that happen to share the same ``source`` (the
        default ``'memory'`` for in-process adds is a common case)
        don't collide on lookup.
        """
        if not self._dirty and self._retriever is not None:
            return
        retriever = KnowledgeTFIDFRetriever()
        docs = [
            {
                "name": f"__entry_{i}",
                "statement": e.theorem,
                "proof": e.final_proof,
                "domain": "",
                "times_cited": 0,
            }
            for i, e in enumerate(candidates)
        ]
        retriever.index_lemmas(docs)
        self._retriever = retriever
        self._dirty = False
