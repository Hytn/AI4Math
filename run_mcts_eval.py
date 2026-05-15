#!/usr/bin/env python3
"""run_mcts_eval.py — full miniF2F MCTS evaluation via LeanREPL.

This runner evaluates the full miniF2F split loaded by benchmarks.loader and
searches tactic trees against real Lean proof states using LeanREPL. The search
policy is MCTS/PUCT-style:

  - selection by UCB/PUCT over tactic-prefix nodes
  - expansion by LLM or heuristic tactic suggestions
  - evaluation via LeanREPL.check_tactic_sequence(...)
  - backpropagation of solved/progress rewards
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from agent.brain.claude_provider import create_provider
from agent.brain.llm_provider import CachedProvider, LLMProvider
from agent.brain.response_parser import extract_lean_from_model_output
from benchmarks.loader import load_benchmark
from benchmarks.metrics import MetricsSummary, compute_metrics
from prover.codegen.code_formatter import extract_proof_body
from prover.models import AttemptStatus, BenchmarkProblem, ProofAttempt, ProofTrace
from prover.verifier.lean_repl import LeanREPL, REPLResponse

TACTIC_PROMPT = """You are a Lean 4 tactic generator inside a Monte-Carlo tree search prover.

Return ONLY a JSON array of Lean 4 tactic strings.

Rules:
- Suggest 4 to 10 plausible next tactics
- Each tactic must be a single Lean tactic line
- Use the exact local hypothesis names that appear in the state
- Do not output markdown, explanations, comments, or code fences
- Do not use `sorry`
- Prefer short executable tactics over long proof scripts

Theorem:
```lean
{theorem_statement}
```

Current proof state:
```text
{state_text}
```

Previous tactics:
{history}
"""

PROOF_FALLBACK_PROMPT = """You are a Lean 4 theorem prover using Mathlib.

Return ONLY one ```lean fenced code block containing a continuation proof body.

Rules:
- No prose, explanations, bullet points, or markdown outside the single fence
- Do not repeat the theorem declaration
- Start with the next tactic or tactic block valid from the CURRENT proof state
- Use the exact local hypothesis names shown in the state
- Do not use `sorry`

Full theorem:
```lean
{theorem_statement}
```

Current proof state:
```text
{state_text}
```

Previous accepted tactics:
```lean
{history}
```
"""

_THEOREM_START_RE = re.compile(r"(?m)^\s*(theorem|lemma)\s+([A-Za-z0-9_']+)\b")
_TACTIC_HEAD_RE = re.compile(
    r"^\s*(?:"
    r"haveI?\b|letI?\b|exact\b|exact_mod_cast\b|simpa?\b|simp(?:_all)?\b|"
    r"rw\b|nth_rewrite\b|conv\b|change\b|show\b|refine\b|apply\b|"
    r"intro\b|intros\b|rintro\b|rcases\b|cases\b|constructor\b|left\b|right\b|"
    r"calc\b|ring(?:_nf)?\b|omega\b|linarith\b|nlinarith\b|norm_num\b|"
    r"field_simp\b|native_decide\b|positivity\b|tauto\b|aesop\b|"
    r"subst\b|rename_i\b|obtain\b|use\b|revert\b|induction\b|"
    r"generalize\b|specialize\b|clear\b|swap\b|all_goals\b|first\b|repeat\b|"
    r"try\b|by_cases\b|by_contra\b|contradiction\b|trivial\b|rfl\b"
    r")\b"
)
_DECL_HEAD_RE = re.compile(r"^\s*(?:theorem|lemma|example|import|open|set_option|#)")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _parse_json_tactics(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw or not raw.lstrip().startswith("["):
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item).strip() for item in data if str(item).strip()]


def _looks_like_tactic_start(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.startswith(("```", "--", "/-")):
        return False
    if _DECL_HEAD_RE.match(stripped):
        return False
    return bool(_TACTIC_HEAD_RE.match(stripped))


def _sanitize_tactic_candidate(text: str) -> str:
    candidate = (text or "").strip()
    if not candidate:
        return ""
    candidate = re.sub(r"^\s*```(?:lean|lean4)?\s*$", "", candidate, flags=re.I | re.M)
    candidate = re.sub(r"^\s*```\s*$", "", candidate, flags=re.M)
    candidate = candidate.strip()
    if not candidate:
        return ""
    first_line = candidate.splitlines()[0].strip()
    if not _looks_like_tactic_start(first_line):
        return ""
    # Reject obvious narrative spills even if the first line happens to start with a tactic keyword.
    for line in candidate.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("--", "/-", "·", "|", "<;>")):
            continue
        if _looks_like_tactic_start(stripped):
            continue
        if stripped.startswith(("by", ":= by")):
            continue
        if re.match(r"^\s*[_|]", stripped):
            continue
        if stripped.endswith(("=>", ":")):
            continue
        return ""
    return candidate


def _chunk_proof_body_into_tactics(proof_body: str) -> list[str]:
    body = (proof_body or "").strip()
    if not body:
        return []
    if body.startswith(":= by"):
        body = body[len(":= by"):].lstrip()
    elif body.startswith("by"):
        body = body[len("by"):].lstrip()
    lines = body.splitlines()
    followup_indents = [
        len(line) - len(line.lstrip(" "))
        for line in lines[1:]
        if line.strip() and (len(line) - len(line.lstrip(" "))) > 0
    ]
    if followup_indents:
        base_indent = min(followup_indents)
        normalized = [lines[0].lstrip()]
        for line in lines[1:]:
            indent = len(line) - len(line.lstrip(" "))
            if line.strip() and indent >= base_indent:
                normalized.append(line[base_indent:])
            else:
                normalized.append(line.lstrip())
        lines = normalized

    chunks: list[str] = []
    current: list[str] = []
    current_indent = 0

    def flush():
        nonlocal current
        if not current:
            return
        candidate = _sanitize_tactic_candidate("\n".join(current))
        if candidate:
            chunks.append(candidate)
        current = []

    for raw in lines:
        if not raw.strip():
            if current:
                current.append(raw)
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        starts_new = _looks_like_tactic_start(stripped) and not stripped.startswith(("·", "|"))
        if current and starts_new and indent <= current_indent:
            flush()
        if not current:
            current = [raw]
            current_indent = indent
        else:
            current.append(raw)
    flush()
    return _dedupe_keep_order(chunks)


def _extract_tactics_from_model_output(content: str, thinking: str = "") -> list[str]:
    raw = (content or "").strip()
    if raw.lstrip().startswith("["):
        direct = [
            _sanitize_tactic_candidate(tactic)
            for tactic in _parse_json_tactics(content)
        ]
        direct = [t for t in direct if t]
        if direct:
            return _dedupe_keep_order(direct)

    if raw:
        linewise: list[str] = []
        all_tactic_like = True
        for line in raw.splitlines():
            stripped = line.strip().strip(",").strip('"').strip("'")
            if not stripped or stripped.startswith("```"):
                continue
            candidate = _sanitize_tactic_candidate(stripped)
            if candidate:
                linewise.append(candidate)
            else:
                all_tactic_like = False
                break
        if linewise and all_tactic_like:
            return _dedupe_keep_order(linewise)

    lean = extract_lean_from_model_output(content or "", thinking or "")
    if not lean.strip():
        return []
    body = extract_proof_body(lean)
    return _chunk_proof_body_into_tactics(body)


def _normalize_state_text(text: str, limit: int = 3000) -> str:
    cleaned = (text or "").replace("\r", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "\n...[truncated]..."


def _extract_state_text(resp: REPLResponse) -> str:
    if resp.raw_output:
        raw = resp.raw_output.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                parsed = json.loads(raw)
                messages = parsed.get("messages", [])
                chunks = []
                for msg in messages:
                    data = msg.get("data") or msg.get("message") or ""
                    if data:
                        chunks.append(str(data))
                if chunks:
                    return _normalize_state_text("\n".join(chunks))
            except json.JSONDecodeError:
                pass
        return _normalize_state_text(raw)
    if resp.error:
        return _normalize_state_text(resp.error)
    if resp.goals:
        return _normalize_state_text("\n".join(resp.goals))
    return ""


def _extract_target(resp: REPLResponse) -> str:
    for goal in resp.goals:
        stripped = goal.strip()
        if stripped.startswith("⊢"):
            return stripped[1:].strip()
    state_text = _extract_state_text(resp)
    for line in state_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("⊢"):
            return stripped[1:].strip()
    return ""


def _extract_hypothesis_names(state_text: str) -> list[str]:
    def _is_probable_local_name(token: str) -> bool:
        if not token or token in {"case", "|", "error", "warning"}:
            return False
        if token[0].isdigit():
            return False
        if any(ch in token for ch in '<>:/\\.=()[]{}"'):
            return False
        return True

    names: list[str] = []
    for line in state_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("⊢") or ":" not in stripped:
            continue
        if re.match(r"^<[^>]+>:\d+:\d+:", stripped):
            continue
        left, _ = stripped.split(":", 1)
        left = left.strip()
        if not left or left.startswith("warning") or left.startswith("error"):
            continue
        for chunk in re.split(r"[\s,]+", left):
            token = chunk.strip()
            if not _is_probable_local_name(token):
                continue
            if token not in names:
                names.append(token)
    return names[:8]


def _infer_shape(target: str) -> str:
    tgt = target or ""
    if "∀" in tgt or "→" in tgt:
        return "forall"
    if "∧" in tgt:
        return "and"
    if "∨" in tgt:
        return "or"
    if "∃" in tgt:
        return "exists"
    if "=" in tgt:
        return "eq"
    return "other"


def _goal_signature(resp: REPLResponse) -> str:
    if resp.is_complete:
        return "complete"
    payload = "\n".join(resp.goals).strip()
    if not payload:
        payload = _extract_state_text(resp) or resp.error or "unknown"
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def _problem_source_path(project_dir: Path, split: str, name: str) -> Path | None:
    split_dir = "Test" if split.lower() == "test" else "Valid"
    direct = project_dir / "MiniF2F" / split_dir / f"{name}.lean"
    if direct.exists():
        return direct
    for candidate in sorted((project_dir / "MiniF2F").rglob(f"{name}.lean")):
        return candidate
    return None


def _problem_preamble(problem_path: Path | None) -> tuple[str, str]:
    default = "import Mathlib"
    if problem_path is None or not problem_path.exists():
        return default, "(default import Mathlib)"

    try:
        content = problem_path.read_text(encoding="utf-8")
    except OSError:
        return default, "(default import Mathlib)"

    match = _THEOREM_START_RE.search(content)
    if not match:
        return default, str(problem_path)

    preamble = content[:match.start()].strip()
    return (preamble or default), str(problem_path)


def _theorem_header(problem: BenchmarkProblem) -> str:
    return problem.theorem_statement.rstrip() + " := by"


@dataclass
class TacticBatch:
    tactics: list[str]
    source: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    raw_text: str = ""


class TacticSuggester:
    def __init__(self, provider_name: str, model_name: str, api_base: str):
        self.provider_name = (provider_name or "heuristic").strip().lower()
        self.model_name = model_name or "heuristic_tactic_engine"
        self._provider: LLMProvider | None = None
        if self.provider_name not in {"", "heuristic"}:
            provider_cfg = {
                "provider": self.provider_name,
                "model": model_name,
                "api_base": api_base,
                "base_url": api_base,
            }
            self._provider = CachedProvider(create_provider(provider_cfg), maxsize=2048)
            self.model_name = getattr(self._provider, "model_name", None) or self.model_name

    def suggest(
        self,
        problem: BenchmarkProblem,
        response: REPLResponse,
        tactics_so_far: list[str],
        max_candidates: int,
    ) -> TacticBatch:
        state_text = _extract_state_text(response)
        heuristics = self._heuristic_tactics(problem, response, tactics_so_far, max_candidates)
        if self._provider is None:
            return TacticBatch(
                tactics=heuristics[:max_candidates],
                source="heuristic",
                raw_text=state_text,
            )

        history_text = "; ".join(tactics_so_far) if tactics_so_far else "(none)"
        prompt = TACTIC_PROMPT.format(
            theorem_statement=problem.theorem_statement,
            state_text=state_text or "\n".join(response.goals) or "(no explicit goals reported)",
            history=history_text,
        )
        try:
            llm_resp = self._provider.generate(
                system=(
                    "You output only JSON arrays of Lean 4 tactics for one next search step."
                ),
                user=prompt,
                temperature=0,
                max_tokens=256,
            )
            llm_tactics = [
                tactic for tactic in _extract_tactics_from_model_output(
                    llm_resp.content, llm_resp.thinking
                )
                if tactic and "sorry" not in tactic
            ]

            proof_fallback_resp = None
            if not llm_tactics:
                fallback_prompt = PROOF_FALLBACK_PROMPT.format(
                    theorem_statement=problem.theorem_statement,
                    state_text=state_text or "\n".join(response.goals) or "(no explicit goals reported)",
                    history=history_text,
                )
                proof_fallback_resp = self._provider.generate(
                    system=(
                        "You output only executable Lean 4 proof continuations. "
                        "Never explain in natural language."
                    ),
                    user=fallback_prompt,
                    temperature=0,
                    max_tokens=512,
                )
                llm_tactics = [
                    tactic for tactic in _extract_tactics_from_model_output(
                        proof_fallback_resp.content, proof_fallback_resp.thinking
                    )
                    if tactic and "sorry" not in tactic
                ]

            merged = _dedupe_keep_order(llm_tactics + heuristics)[:max_candidates]
            total_tokens_in = int(llm_resp.tokens_in or 0)
            total_tokens_out = int(llm_resp.tokens_out or 0)
            total_latency_ms = int(llm_resp.latency_ms or 0)
            raw_parts = [llm_resp.content]
            if proof_fallback_resp is not None:
                total_tokens_in += int(proof_fallback_resp.tokens_in or 0)
                total_tokens_out += int(proof_fallback_resp.tokens_out or 0)
                total_latency_ms += int(proof_fallback_resp.latency_ms or 0)
                raw_parts.append("\n\n[FALLBACK_PROOF]\n")
                raw_parts.append(proof_fallback_resp.content)
            return TacticBatch(
                tactics=merged,
                source=self.provider_name,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                latency_ms=total_latency_ms,
                raw_text="".join(raw_parts),
            )
        except Exception as exc:
            return TacticBatch(
                tactics=heuristics[:max_candidates],
                source="heuristic_fallback",
                raw_text=str(exc),
            )

    def _heuristic_tactics(
        self,
        problem: BenchmarkProblem,
        response: REPLResponse,
        tactics_so_far: list[str],
        max_candidates: int,
    ) -> list[str]:
        state_text = _extract_state_text(response)
        target = _extract_target(response)
        shape = _infer_shape(target)
        hyp_names = _extract_hypothesis_names(state_text)
        depth = len(tactics_so_far)

        tactics: list[str] = []
        if shape == "forall":
            intro_name = f"h{depth}" if depth > 0 else "h"
            tactics.extend([f"intro {intro_name}", "rintro h"])
        if shape == "and":
            tactics.append("constructor")
        if shape == "or":
            tactics.extend(["left", "right"])
        if shape == "eq":
            tactics.extend(["rfl", "ring", "omega", "norm_num"])
        if shape == "exists":
            tactics.extend(["constructor", "refine ⟨?_, ?_⟩"])

        for name in hyp_names:
            tactics.extend([
                f"exact {name}",
                f"apply {name}",
                f"cases {name}",
            ])

        generic = [
            "assumption",
            "simpa",
            "simp",
            "aesop",
            "norm_num",
            "linarith",
            "nlinarith",
            "omega",
            "ring",
            "positivity",
            "tauto",
            "constructor",
            "contradiction",
            "native_decide",
        ]
        if "Nat." in problem.theorem_statement or "ℕ" in problem.theorem_statement:
            generic.extend(["omega", "norm_num", "linarith"])
        if "ℝ" in problem.theorem_statement or "ℤ" in problem.theorem_statement:
            generic.extend(["linarith", "nlinarith", "ring"])

        return _dedupe_keep_order(tactics + generic)[:max_candidates]


@dataclass
class MCTSConfig:
    max_depth: int = 16
    max_nodes: int = 256
    timeout_ms: int = 10000
    max_candidates: int = 10
    exploration_weight: float = 1.4


@dataclass
class SearchNode:
    tactics: tuple[str, ...]
    response: REPLResponse
    signature: str
    parent: SearchNode | None = None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    pending_tactics: list[str] = field(default_factory=list)
    child_priors: dict[str, float] = field(default_factory=dict)
    children: dict[str, SearchNode] = field(default_factory=dict)
    terminal_reason: str = ""

    @property
    def depth(self) -> int:
        return len(self.tactics)

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def is_solved(self) -> bool:
        return bool(self.response.is_complete)

    @property
    def is_dead(self) -> bool:
        return (not self.response.success and not self.response.is_complete) or bool(self.terminal_reason)


@dataclass
class SearchStats:
    is_solved: bool = False
    solution_path: list[str] = field(default_factory=list)
    nodes_expanded: int = 0
    lean_checks: int = 0
    llm_calls: int = 0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    llm_latency_ms: int = 0
    dead_ends: int = 0
    max_depth_reached: int = 0
    time_ms: float = 0.0
    root_error: str = ""
    stop_reason: str = ""
    terminal_reason_counts: dict[str, int] = field(default_factory=dict)
    root_tactic_outcomes: list[dict[str, object]] = field(default_factory=list)
    tactic_batches: list[dict[str, object]] = field(default_factory=list)


class LeanMCTSSearcher:
    def __init__(self, repl: LeanREPL, suggester: TacticSuggester, cfg: MCTSConfig):
        self._repl = repl
        self._suggester = suggester
        self._cfg = cfg

    def run(self, problem: BenchmarkProblem, theorem_header: str, preamble: str) -> SearchStats:
        stats = SearchStats()
        started_at = time.perf_counter()

        root_resp = self._repl.check_tactic_sequence(theorem_header, [], preamble=preamble)
        stats.lean_checks += 1
        if not root_resp.success and not root_resp.is_complete and not root_resp.goals:
            stats.root_error = root_resp.error or _extract_state_text(root_resp)
            stats.stop_reason = "root_error"
            stats.time_ms = (time.perf_counter() - started_at) * 1000
            return stats

        root = SearchNode(tactics=(), response=root_resp, signature=_goal_signature(root_resp))
        best_solution: list[str] = []
        search_started_at = time.perf_counter()

        while stats.nodes_expanded < self._cfg.max_nodes:
            elapsed_ms = (time.perf_counter() - search_started_at) * 1000
            if elapsed_ms >= self._cfg.timeout_ms:
                stats.stop_reason = "timeout"
                break

            path = self._select_path(root)
            leaf = path[-1]

            if leaf.is_solved:
                stats.is_solved = True
                stats.stop_reason = "solved"
                best_solution = list(leaf.tactics)
                self._backpropagate(path, 1.0)
                break

            if not leaf.response.success:
                stats.dead_ends += 1
                self._backpropagate(path, 0.0)
                continue

            if leaf.depth >= self._cfg.max_depth:
                leaf.terminal_reason = "max_depth"
                self._backpropagate(path, self._progress_reward(leaf))
                continue

            if not leaf.expanded:
                batch = self._suggester.suggest(
                    problem=problem,
                    response=leaf.response,
                    tactics_so_far=list(leaf.tactics),
                    max_candidates=self._cfg.max_candidates,
                )
                if len(stats.tactic_batches) < 24:
                    stats.tactic_batches.append({
                        "depth": leaf.depth,
                        "source": batch.source,
                        "tactics": list(batch.tactics),
                        "tokens_in": batch.tokens_in,
                        "tokens_out": batch.tokens_out,
                        "latency_ms": batch.latency_ms,
                        "raw_text": (batch.raw_text or "")[:1000],
                    })
                if batch.source not in {"heuristic", "heuristic_fallback"}:
                    stats.llm_calls += 1
                    stats.llm_tokens_in += batch.tokens_in
                    stats.llm_tokens_out += batch.tokens_out
                    stats.llm_latency_ms += batch.latency_ms

                priors = self._prior_scores(batch.tactics)
                leaf.pending_tactics = list(batch.tactics)
                leaf.child_priors = dict(zip(batch.tactics, priors))
                leaf.expanded = True

                if not leaf.pending_tactics:
                    leaf.terminal_reason = "no_candidates"
                    stats.dead_ends += 1
                    self._backpropagate(path, self._progress_reward(leaf))
                    continue

            child = self._expand_one_child(
                node=leaf,
                theorem_header=theorem_header,
                preamble=preamble,
                stats=stats,
                ancestor_signatures={node.signature for node in path},
            )
            if child is None:
                leaf.terminal_reason = leaf.terminal_reason or "exhausted"
                stats.dead_ends += 1
                self._backpropagate(path, self._progress_reward(leaf))
                continue

            path.append(child)
            if child.is_solved:
                stats.is_solved = True
                stats.stop_reason = "solved"
                best_solution = list(child.tactics)
                self._backpropagate(path, 1.0)
                break

            reward = self._progress_reward(child)
            self._backpropagate(path, reward)

        if not stats.stop_reason:
            if stats.is_solved:
                stats.stop_reason = "solved"
            elif stats.nodes_expanded >= self._cfg.max_nodes:
                stats.stop_reason = "max_nodes"
            else:
                stats.stop_reason = "exhausted"
        stats.solution_path = best_solution
        stats.time_ms = (time.perf_counter() - started_at) * 1000
        return stats

    def _select_path(self, root: SearchNode) -> list[SearchNode]:
        path = [root]
        node = root
        while True:
            if node.is_solved or node.is_dead or node.depth >= self._cfg.max_depth:
                return path
            if self._should_expand_node(node):
                return path
            candidates = [
                child for child in node.children.values()
                if not child.is_dead or child.is_solved
            ]
            if not candidates:
                if node.pending_tactics:
                    return path
                node.terminal_reason = node.terminal_reason or "all_children_dead"
                return path
            node = max(candidates, key=lambda child: self._puct_score(node, child))
            path.append(node)

    @staticmethod
    def _record_terminal_reason(stats: SearchStats, reason: str) -> None:
        if not reason:
            return
        stats.terminal_reason_counts[reason] = stats.terminal_reason_counts.get(reason, 0) + 1

    def _should_expand_node(self, node: SearchNode) -> bool:
        if not node.expanded:
            return True
        if not node.pending_tactics:
            return False
        if not node.children:
            return True
        # Progressive widening: do not exhaust every sibling at the root before
        # we have given promising children a chance to grow deeper.
        revisit_budget = max(2, 2 * len(node.children))
        return node.visits >= revisit_budget

    def _expand_one_child(
        self,
        node: SearchNode,
        theorem_header: str,
        preamble: str,
        stats: SearchStats,
        ancestor_signatures: set[str],
    ) -> SearchNode | None:
        while node.pending_tactics:
            tactic = node.pending_tactics.pop(0)
            child_tactics = list(node.tactics) + [tactic]
            resp = self._repl.check_tactic_sequence(theorem_header, child_tactics, preamble=preamble)
            stats.lean_checks += 1
            stats.nodes_expanded += 1
            stats.max_depth_reached = max(stats.max_depth_reached, len(child_tactics))

            signature = _goal_signature(resp)
            child = SearchNode(
                tactics=tuple(child_tactics),
                response=resp,
                signature=signature,
                parent=node,
                prior=node.child_priors.get(tactic, 0.1),
            )
            if signature in ancestor_signatures and not child.is_solved:
                child.terminal_reason = "cycle"
            elif resp.success and not resp.is_complete and signature == node.signature:
                child.terminal_reason = "no_progress"
            elif not resp.success and not resp.is_complete:
                child.terminal_reason = "lean_error"
            self._record_terminal_reason(stats, child.terminal_reason)
            if node.depth == 0 and len(stats.root_tactic_outcomes) < 32:
                stats.root_tactic_outcomes.append({
                    "tactic": tactic,
                    "success": resp.success,
                    "is_complete": resp.is_complete,
                    "terminal_reason": child.terminal_reason,
                    "goal_signature_changed": signature != node.signature,
                    "error": (resp.error or "")[:300],
                    "state_excerpt": _extract_state_text(resp)[:300],
                })
            node.children[tactic] = child
            return child
        return None

    def _puct_score(self, parent: SearchNode, child: SearchNode) -> float:
        q = child.q_value
        u = (
            self._cfg.exploration_weight
            * child.prior
            * math.sqrt(max(1, parent.visits))
            / (1 + child.visits)
        )
        return q + u

    def _progress_reward(self, node: SearchNode) -> float:
        if node.is_solved:
            return 1.0
        if not node.response.success:
            return 0.0

        goal_count = max(1, len(node.response.goals))
        reward = 0.15 + 0.25 / goal_count
        if node.parent is not None:
            parent_goals = max(1, len(node.parent.response.goals))
            reduction = max(0, parent_goals - goal_count) / parent_goals
            reward += 0.4 * reduction
            if node.signature == node.parent.signature:
                reward *= 0.25
        if node.terminal_reason in {"cycle", "no_progress"}:
            reward *= 0.2
        return max(0.0, min(0.95, reward))

    @staticmethod
    def _backpropagate(path: list[SearchNode], reward: float):
        for node in path:
            node.visits += 1
            node.value_sum += reward

    @staticmethod
    def _prior_scores(tactics: list[str]) -> list[float]:
        if not tactics:
            return []
        total = float(len(tactics))
        return [max(0.05, (total - idx) / total) for idx in range(len(tactics))]


def _existing_solved(trace_path: Path) -> bool:
    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("solved"))


def _build_trace(
    problem: BenchmarkProblem,
    stats: SearchStats,
    cfg: MCTSConfig,
    provider_name: str,
    model_name: str,
    preamble_source: str,
) -> ProofTrace:
    trace = ProofTrace(
        problem_id=problem.problem_id,
        problem_name=problem.name,
        theorem_statement=problem.theorem_statement,
        natural_language=problem.natural_language,
        config_snapshot={
            "engine": "lean_repl",
            "search_strategy": "mcts",
            "llm_mode": provider_name,
            "llm_model": model_name,
            "max_nodes": cfg.max_nodes,
            "max_depth": cfg.max_depth,
            "timeout_ms": cfg.timeout_ms,
            "max_candidates": cfg.max_candidates,
            "exploration_weight": cfg.exploration_weight,
            "difficulty": problem.difficulty,
            "source": problem.source,
            "preamble_source": preamble_source,
        },
    )

    attempt = ProofAttempt(attempt_number=1)
    attempt.generated_proof = (
        "by\n  " + "\n  ".join(stats.solution_path) if stats.solution_path else ""
    )
    attempt.llm_model = model_name
    attempt.llm_tokens_in = stats.llm_tokens_in
    attempt.llm_tokens_out = stats.llm_tokens_out
    attempt.llm_latency_ms = stats.llm_latency_ms
    attempt.lean_check_ms = int(stats.time_ms)
    attempt.lean_result = AttemptStatus.SUCCESS if stats.is_solved else AttemptStatus.LEAN_ERROR
    if not stats.is_solved:
        if stats.root_error:
            attempt.lean_stderr = stats.root_error
        elif stats.stop_reason == "timeout":
            attempt.lean_stderr = "MCTS timed out before finding a complete proof"
        elif stats.stop_reason == "max_nodes":
            attempt.lean_stderr = "MCTS reached max_nodes before finding a complete proof"
        else:
            attempt.lean_stderr = "MCTS search exhausted before finding a complete proof"
    attempt.repair_trace.append({
        "search_strategy": "mcts",
        "solution_path": stats.solution_path,
        "real_solved": stats.is_solved,
        "stop_reason": stats.stop_reason,
        "nodes_expanded": stats.nodes_expanded,
        "lean_checks": stats.lean_checks,
        "llm_calls": stats.llm_calls,
        "llm_tokens_in": stats.llm_tokens_in,
        "llm_tokens_out": stats.llm_tokens_out,
        "llm_latency_ms": stats.llm_latency_ms,
        "dead_ends": stats.dead_ends,
        "max_depth_reached": stats.max_depth_reached,
        "search_time_ms": round(stats.time_ms, 3),
        "root_error": stats.root_error,
        "terminal_reason_counts": stats.terminal_reason_counts,
        "root_tactic_outcomes": stats.root_tactic_outcomes,
        "tactic_batches": stats.tactic_batches,
    })

    trace.add_attempt(attempt)
    trace.total_duration_ms = int(stats.time_ms)
    trace.strategy_path.append("mcts")
    trace.config_snapshot["difficulty"] = problem.difficulty
    trace.config_snapshot["problem_name"] = problem.name
    return trace


def run_problem(
    problem: BenchmarkProblem,
    repl: LeanREPL,
    suggester: TacticSuggester,
    cfg: MCTSConfig,
    project_dir: Path,
    split: str,
) -> ProofTrace:
    problem_path = _problem_source_path(project_dir, split, problem.name)
    preamble, preamble_source = _problem_preamble(problem_path)
    theorem_header = _theorem_header(problem)
    searcher = LeanMCTSSearcher(repl=repl, suggester=suggester, cfg=cfg)
    stats = searcher.run(problem=problem, theorem_header=theorem_header, preamble=preamble)
    return _build_trace(
        problem=problem,
        stats=stats,
        cfg=cfg,
        provider_name=suggester.provider_name or "heuristic",
        model_name=suggester.model_name,
        preamble_source=preamble_source,
    )


def run_problem_isolated(
    problem: BenchmarkProblem,
    cfg: MCTSConfig,
    project_dir: Path,
    split: str,
    provider_name: str,
    model_name: str,
    api_base: str,
    lean_timeout_s: int,
) -> ProofTrace:
    suggester = TacticSuggester(
        provider_name=provider_name,
        model_name=model_name,
        api_base=api_base,
    )
    repl = LeanREPL.create(project_dir=str(project_dir), timeout=lean_timeout_s)
    try:
        return run_problem(
            problem=problem,
            repl=repl,
            suggester=suggester,
            cfg=cfg,
            project_dir=project_dir,
            split=split,
        )
    except Exception as exc:
        failed = SearchStats(root_error=str(exc))
        return _build_trace(
            problem=problem,
            stats=failed,
            cfg=cfg,
            provider_name=suggester.provider_name or "heuristic",
            model_name=suggester.model_name,
            preamble_source="(worker exception)",
        )
    finally:
        repl.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full miniF2F MCTS evaluation via LeanREPL."
    )
    parser.add_argument("--benchmark", default="minif2f")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", default="results_mcts")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--exploration-weight", type=float, default=1.4)
    parser.add_argument("--problem-workers", type=int, default=1)
    parser.add_argument("--provider", default="heuristic")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-base", default="")
    args = parser.parse_args()

    if args.benchmark.lower() != "minif2f":
        print("run_mcts_eval.py currently only supports --benchmark minif2f", file=sys.stderr)
        return 2

    project_dir = Path(_ROOT) / "data" / "miniF2F"
    dataset_problems = load_benchmark("minif2f", args.split, limit=0)
    run_problems = list(dataset_problems)
    if args.offset > 0:
        run_problems = run_problems[args.offset:]
    if args.limit > 0:
        run_problems = run_problems[:args.limit]

    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "traces" / "minif2f"
    eval_dir = output_dir / "evals"
    trace_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    provider_name = (args.provider or "heuristic").strip().lower()
    model_name = (args.model or "").strip() or "heuristic_tactic_engine"
    if provider_name in {"", "heuristic"}:
        model_name = "heuristic_tactic_engine"
    suggester = TacticSuggester(provider_name=provider_name, model_name=model_name, api_base=(args.api_base or "").strip())

    print("=" * 72)
    print("AI4Math — miniF2F MCTS Evaluation (Full Dataset via LeanREPL)")
    print("=" * 72)
    print(f"Dataset loaded:             {len(dataset_problems)} problems")
    print(f"Requested this run:         {len(run_problems)} problems")
    print(f"LLM mode:                   {suggester.provider_name or 'heuristic'}")
    print(f"Problem workers:            {max(1, int(args.problem_workers))}")
    cfg = MCTSConfig(
        max_depth=max(1, int(args.max_depth)),
        max_nodes=max(1, int(args.max_nodes)),
        timeout_ms=max(1, int(args.timeout_ms)),
        max_candidates=max(1, int(args.max_candidates)),
        exploration_weight=max(0.05, float(args.exploration_weight)),
    )

    lean_timeout_s = max(30, int(math.ceil(cfg.timeout_ms / 1000)) + 10)
    print(f"LLM model:                  {suggester.model_name}")
    print(f"Lean project:               {project_dir}")
    probe_repl = LeanREPL.create(project_dir=str(project_dir), timeout=lean_timeout_s)
    print(f"Lean backend:               {probe_repl.backend}")
    probe_repl.close()
    print(f"Output dir:                 {output_dir}")
    print()

    trace_records: list[dict] = []
    skipped = 0
    todo: list[tuple[int, BenchmarkProblem, Path]] = []
    for idx, problem in enumerate(run_problems, 1):
        trace_path = trace_dir / f"{problem.problem_id}.json"
        if args.resume and trace_path.exists() and _existing_solved(trace_path):
            skipped += 1
            try:
                trace_records.append(json.loads(trace_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
            print(f"[{idx}/{len(run_problems)}] {problem.name:<36} skip (resume)")
            continue
        todo.append((idx, problem, trace_path))

    problem_workers = max(1, int(args.problem_workers))
    if problem_workers == 1:
        repl = LeanREPL.create(project_dir=str(project_dir), timeout=lean_timeout_s)
        try:
            for idx, problem, trace_path in todo:
                trace = run_problem(
                    problem=problem,
                    repl=repl,
                    suggester=suggester,
                    cfg=cfg,
                    project_dir=project_dir,
                    split=args.split,
                )
                trace.save(trace_path)
                record = trace.to_dict()
                trace_records.append(record)

                attempt = record["attempts"][0] if record.get("attempts") else {}
                search_meta = (attempt.get("repair_trace") or [{}])[0]
                status = "SOLVED" if record.get("solved") else "FAILED"
                depth = len(search_meta.get("solution_path", []) or [])
                nodes = int(search_meta.get("nodes_expanded", 0) or 0)
                print(
                    f"[{idx}/{len(run_problems)}] {problem.name:<36} "
                    f"{status:<6} depth={depth:<2} nodes={nodes:<4}"
                )
        finally:
            repl.close()
    else:
        with ThreadPoolExecutor(max_workers=problem_workers) as executor:
            futures = {
                executor.submit(
                    run_problem_isolated,
                    problem,
                    cfg,
                    project_dir,
                    args.split,
                    provider_name,
                    model_name,
                    (args.api_base or "").strip(),
                    lean_timeout_s,
                ): (idx, problem, trace_path)
                for idx, problem, trace_path in todo
            }
            for future in as_completed(futures):
                idx, problem, trace_path = futures[future]
                trace = future.result()
                trace.save(trace_path)
                record = trace.to_dict()
                trace_records.append(record)

                attempt = record["attempts"][0] if record.get("attempts") else {}
                search_meta = (attempt.get("repair_trace") or [{}])[0]
                status = "SOLVED" if record.get("solved") else "FAILED"
                depth = len(search_meta.get("solution_path", []) or [])
                nodes = int(search_meta.get("nodes_expanded", 0) or 0)
                print(
                    f"[{idx}/{len(run_problems)}] {problem.name:<36} "
                    f"{status:<6} depth={depth:<2} nodes={nodes:<4}"
                )

    metrics = compute_metrics(trace_records, k_values=[1])
    summary = {
        "benchmark": "minif2f",
        "pipeline": "mcts_lean_repl_full",
        "llm_mode": suggester.provider_name or "heuristic",
        "llm_model": suggester.model_name,
        "dataset_total_available": len(dataset_problems),
        "dataset_total_requested": len(run_problems),
        "resume_skipped": skipped,
        "metrics": metrics,
    }
    summary_path = eval_dir / "minif2f_mcts_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(MetricsSummary("minif2f/mcts_full", metrics).to_table())
    print(f"Summary saved to {summary_path}")
    print(f"Per-problem traces saved under {trace_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
