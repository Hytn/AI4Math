#!/usr/bin/env python3
"""scripts/rl_pipeline.py — End-to-end RL flywheel orchestrator.

Closes the v4 gap noted in REFACTOR_REPORT.md (and §九.4 of the v4 plan):
chains the four stages of the project's "explore → deposit → train →
deploy" loop into a single command:

  1. **eval**          — run a benchmark with a chosen profile,
                         producing one ``dialog.json`` per problem.
  2. **collect**       — walk the trace dir, validate each dialog,
                         emit a single SFT-ready ``.jsonl``.
  3. **train_wm**      — extract step details from successful proofs,
                         train an sklearn ``WorldModelPredictor``, save
                         to ``world_model.pkl``. (Optional; skipped if
                         no successes are available.)
  4. **train_llm**     — SFT/DPO training is delegated to an external
                         framework (TRL, axolotl, slime…). This stage
                         either invokes a user-supplied command via
                         ``--train-cmd`` or stops here so the user can
                         pick up the produced ``sft.jsonl``.

After step 4, the user re-runs the loop from step 1 with the new
weights. Each iteration's outputs land under
``results/rl/iter_<N>/{traces,sft.jsonl,world_model.pkl}``.

The orchestrator is intentionally thin glue around already-tested
components:

  • ``run_eval.py``                          (stage 1)
  • ``agent.persistence.collect_dialogs``    (stage 2 — SFT JSONL)
  • ``scripts/train_world_model.py``         (stage 3)

This means every stage is independently runnable; the orchestrator
just sequences them and prints a uniform summary.

Examples
--------

::

    # One full iteration on builtin (mock LLM, no Lean)
    python scripts/rl_pipeline.py iter \\
        --iter-dir results/rl/iter_0 \\
        --profile whole_proof_repair \\
        --benchmark builtin \\
        --provider mock

    # Collect-only — already have traces from a separate run
    python scripts/rl_pipeline.py collect \\
        --traces-dir results/traces \\
        --output sft.jsonl

    # Closed loop with a user-supplied SFT trainer
    python scripts/rl_pipeline.py loop \\
        --iters 3 \\
        --benchmark builtin \\
        --provider mock \\
        --train-cmd 'echo would-train-here {sft_jsonl} → {model_out}'

The orchestrator is fully fail-soft: any single stage's failure is
reported but does not crash subsequent iterations — the next iteration
simply starts from the most recent succeeded artifact.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("rl_pipeline")


# ─────────────────────────────────────────────────────────────────────────
# Stage outcomes
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    """Common shape for every stage's outcome."""
    stage: str                     # "eval" | "collect" | "train_wm" | "train_llm"
    ok: bool
    duration_s: float = 0.0
    artifact: Optional[str] = None    # primary output path
    metrics: dict = field(default_factory=dict)
    skipped_reason: str = ""

    def short(self) -> str:
        if self.skipped_reason:
            return f"[skip:{self.stage}] {self.skipped_reason}"
        if not self.ok:
            return f"[FAIL:{self.stage}] {self.metrics.get('error', '?')}"
        m = ", ".join(f"{k}={v}" for k, v in self.metrics.items() if k != "error")
        return f"[ok:{self.stage}] {m}" if m else f"[ok:{self.stage}]"


@dataclass
class IterResult:
    iter_idx: int
    iter_dir: str
    stages: list = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(s.ok for s in self.stages)


# ─────────────────────────────────────────────────────────────────────────
# Stage 1: eval
# ─────────────────────────────────────────────────────────────────────────

def stage_eval(*, iter_dir: Path, profile: str, benchmark: str,
                provider: str, limit: int, max_samples: int,
                model: Optional[str], extra_args: list[str]) -> StageResult:
    """Invoke run_eval.py with the given knobs.

    Output: ``iter_dir/traces/<problem_id>/dialog.json``.
    """
    t0 = time.monotonic()
    traces_dir = iter_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(REPO_ROOT / "run_eval.py"),
        "--benchmark", benchmark,
        "--provider", provider,
        "--profile", profile,
        "--output-dir", str(iter_dir),
        "--max-samples", str(max_samples),
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if model:
        cmd.extend(["--model", model])
    cmd.extend(extra_args)

    logger.info(f"[eval] {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False,
                                capture_output=True, text=True,
                                timeout=24 * 3600)
    except subprocess.TimeoutExpired as e:
        return StageResult(
            stage="eval", ok=False,
            duration_s=time.monotonic() - t0,
            metrics={"error": f"timeout: {e}"})

    log_path = iter_dir / "eval.log"
    log_path.write_text(
        proc.stdout + "\n----- stderr -----\n" + proc.stderr,
        encoding="utf-8")

    if proc.returncode != 0:
        return StageResult(
            stage="eval", ok=False,
            duration_s=time.monotonic() - t0,
            artifact=str(traces_dir),
            metrics={"error": f"returncode={proc.returncode}",
                     "log": str(log_path)})

    # Count produced dialog.json files for a quick metric.
    n_dialogs = sum(1 for _ in traces_dir.rglob("dialog.json"))
    return StageResult(
        stage="eval", ok=True,
        duration_s=time.monotonic() - t0,
        artifact=str(traces_dir),
        metrics={"n_dialogs": n_dialogs, "profile": profile,
                 "benchmark": benchmark, "provider": provider})


# ─────────────────────────────────────────────────────────────────────────
# Stage 2: collect (dialogs → SFT JSONL)
# ─────────────────────────────────────────────────────────────────────────

def stage_collect(*, traces_dir: Path, output: Path,
                    preset: str = "qwen3",
                    successful_only: bool = False) -> StageResult:
    """Walk the trace dir, write an SFT-ready JSONL.

    With ``successful_only=True`` we only export dialogs whose
    ``result.success`` is true — the typical setting for SFT training,
    where you don't want to teach the model to fail.
    """
    from agent.persistence.unified_storage import collect_dialogs
    from agent.persistence.sft_export import dialogs_to_sft_jsonl
    from agent.persistence.dialog_format import result_of

    t0 = time.monotonic()
    if not traces_dir.exists():
        return StageResult(
            stage="collect", ok=False,
            metrics={"error": f"traces_dir does not exist: {traces_dir}"})

    items = collect_dialogs(traces_dir)
    if not items:
        return StageResult(
            stage="collect", ok=True,
            duration_s=time.monotonic() - t0,
            skipped_reason="no dialog.json files found",
            metrics={"n_in": 0, "n_out": 0})

    if successful_only:
        items = [(p, d) for (p, d) in items
                  if bool(result_of(d).get("success"))]

    dialogs = [d for (_, d) in items]
    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = dialogs_to_sft_jsonl(
        dialogs, str(output), preset=preset)

    return StageResult(
        stage="collect", ok=True,
        duration_s=time.monotonic() - t0,
        artifact=str(output),
        metrics={"n_in": len(dialogs), "n_out": n_written,
                 "preset": preset,
                 "successful_only": successful_only})


# ─────────────────────────────────────────────────────────────────────────
# Stage 3: train world model
# ─────────────────────────────────────────────────────────────────────────

def stage_train_wm(*, traces_dir: Path, db_path: Optional[Path],
                     output: Path,
                     min_samples: int = 50) -> StageResult:
    """Train a sklearn WorldModel from the proofs in ``db_path`` (when
    given) or from the dialogs in ``traces_dir``.

    For the dialog path we synthesize ``RichProofTrajectory`` objects
    from the search_tree blocks (when present) or from the linear
    messages list (best-effort — we extract ``tactic_apply`` outcomes).

    Skipped (with reason) when:
      • sklearn / scipy are not installed
      • fewer than ``min_samples`` step-level samples are available
    """
    t0 = time.monotonic()
    try:
        import sklearn  # noqa: F401
        import scipy    # noqa: F401
    except ImportError:
        return StageResult(
            stage="train_wm", ok=True,
            duration_s=time.monotonic() - t0,
            skipped_reason="sklearn / scipy not installed")

    from engine.world_model_trainer import WorldModelTrainer

    trainer = WorldModelTrainer(db_path=str(db_path) if db_path else "")

    n = 0
    if db_path and db_path.exists():
        n = trainer.extract_training_data()

    if n < min_samples:
        # Fallback: synthesize trajectories from dialog.json files.
        synth = _trajectories_from_dialogs(traces_dir)
        if synth:
            n = trainer.extract_from_trajectories(synth)
            logger.info(
                f"[train_wm] synthesized {n} samples from "
                f"{len(synth)} dialog trajectories")

    if n < min_samples:
        return StageResult(
            stage="train_wm", ok=True,
            duration_s=time.monotonic() - t0,
            skipped_reason=(
                f"only {n} samples (need ≥ {min_samples})"),
            metrics={"n_samples": n})

    metrics = trainer.train()
    if "error" in metrics:
        return StageResult(
            stage="train_wm", ok=False,
            duration_s=time.monotonic() - t0,
            metrics={"error": metrics["error"]})

    output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(output))

    return StageResult(
        stage="train_wm", ok=True,
        duration_s=time.monotonic() - t0,
        artifact=str(output),
        metrics={k: v for k, v in metrics.items()
                  if k in ("accuracy", "f1", "train_size",
                            "test_size", "positive_rate")})


def _trajectories_from_dialogs(traces_dir: Path) -> list:
    """Best-effort: synthesize RichProofTrajectory objects from dialog.json
    files for world-model training. Reads the v3.0 ``meta.search_tree``
    block when present (richest signal); otherwise tries to extract
    ``tactic_apply`` tool calls from the linear messages list."""
    if not traces_dir.exists():
        return []

    from agent.persistence.unified_storage import collect_dialogs
    from agent.persistence.dialog_format import (
        meta_of, search_tree_of, messages_of, result_of,
    )
    from engine.proof_context_store import (
        RichProofTrajectory, StepDetail,
    )

    out: list = []
    for path, d in collect_dialogs(traces_dir):
        steps = _steps_from_dialog(d)
        if not steps:
            continue
        meta = meta_of(d)
        out.append(RichProofTrajectory(
            theorem=meta.get("theorem_statement", ""),
            steps=steps,
            success=bool(result_of(d).get("success")),
            depth=len(steps),
            duration_ms=float(result_of(d).get("total_duration_ms", 0)),
        ))
    return out


def _steps_from_dialog(d: dict) -> list:
    """Pull StepDetail records out of a single dialog. Strategy:

      1. If ``meta.search_tree.nodes`` is present, use it directly —
         each node's ``tactic`` + ``status`` gives a step.
      2. Otherwise scan ``messages`` for ``tactic_apply`` tool call /
         response pairs and reconstruct as best we can.
    """
    from agent.persistence.dialog_format import (
        search_tree_of, messages_of,
    )
    from engine.proof_context_store import StepDetail

    steps: list = []
    tree = search_tree_of(d)
    if tree:
        for n in tree.get("nodes") or []:
            tactic = (n.get("tactic") or "").strip()
            if not tactic:
                continue
            status = n.get("status", "")
            success = status == "solved" or n.get("success_count", 0) > 0
            steps.append(StepDetail(
                step_index=n.get("node_id", 0),
                tactic=tactic,
                env_id_before=0,
                env_id_after=1 if success else -1,
                goals_before=[],          # the dialog rarely has these
                goals_after=[],
                error_message="" if success else "tree_status=" + status,
                error_category="" if success else status,
                is_proof_complete=bool(n.get("is_complete", False)),
            ))
        return steps

    # Fallback: scan tool_call / tool messages for tactic_apply.
    msgs = messages_of(d)
    for i, m in enumerate(msgs):
        if m.get("role") != "tool" or m.get("name") != "tactic_apply":
            continue
        try:
            obs = json.loads(m.get("content") or "{}")
        except (ValueError, TypeError):
            continue
        tactic = (obs.get("tactic") or "").strip()
        if not tactic:
            continue
        steps.append(StepDetail(
            step_index=len(steps),
            tactic=tactic,
            env_id_before=0,
            env_id_after=1 if obs.get("success") else -1,
            goals_before=[], goals_after=list(obs.get("remaining_goals") or []),
            error_message=obs.get("error_message", "") or "",
            error_category=obs.get("error_category", "") or "",
            is_proof_complete=bool(obs.get("is_proof_complete", False)),
        ))
    return steps


# ─────────────────────────────────────────────────────────────────────────
# Stage 4: train LLM (delegated)
# ─────────────────────────────────────────────────────────────────────────

def stage_train_llm(*, sft_jsonl: Path, model_out: Path,
                      train_cmd: Optional[str]) -> StageResult:
    """Run an external SFT/DPO trainer.

    The user supplies ``--train-cmd`` with two placeholders:

      ``{sft_jsonl}`` — path to the JSONL produced by stage 2
      ``{model_out}``  — directory where the trainer should save weights

    If ``--train-cmd`` is omitted, we skip the stage with a clear
    message — the SFT JSONL is still on disk and the user can hand it
    to TRL / axolotl / slime offline.
    """
    t0 = time.monotonic()
    if not train_cmd:
        return StageResult(
            stage="train_llm", ok=True,
            duration_s=time.monotonic() - t0,
            skipped_reason=("no --train-cmd provided; SFT JSONL ready "
                            f"at {sft_jsonl}"))
    if not sft_jsonl.exists():
        return StageResult(
            stage="train_llm", ok=False,
            metrics={"error": f"sft_jsonl missing: {sft_jsonl}"})

    rendered = train_cmd.format(
        sft_jsonl=str(sft_jsonl), model_out=str(model_out))
    logger.info(f"[train_llm] {rendered}")
    try:
        proc = subprocess.run(rendered, shell=True, cwd=REPO_ROOT,
                                check=False, capture_output=True,
                                text=True)
    except Exception as e:
        return StageResult(
            stage="train_llm", ok=False,
            duration_s=time.monotonic() - t0,
            metrics={"error": f"subprocess failed: {e}"})

    if proc.returncode != 0:
        return StageResult(
            stage="train_llm", ok=False,
            duration_s=time.monotonic() - t0,
            metrics={"error": f"returncode={proc.returncode}",
                     "stderr_tail": proc.stderr[-2000:]})

    return StageResult(
        stage="train_llm", ok=True,
        duration_s=time.monotonic() - t0,
        artifact=str(model_out),
        metrics={"cmd": rendered})


# ─────────────────────────────────────────────────────────────────────────
# Top-level orchestration: one iteration / closed loop
# ─────────────────────────────────────────────────────────────────────────

def run_iteration(args, iter_idx: int, iter_dir: Path) -> IterResult:
    """Run a single full iteration: eval → collect → train_wm → train_llm."""
    iter_dir.mkdir(parents=True, exist_ok=True)
    res = IterResult(iter_idx=iter_idx, iter_dir=str(iter_dir))

    if "eval" in args.stages:
        r = stage_eval(
            iter_dir=iter_dir, profile=args.profile,
            benchmark=args.benchmark, provider=args.provider,
            limit=args.limit, max_samples=args.max_samples,
            model=args.model, extra_args=args.eval_extra)
        res.stages.append(r)
        if not r.ok and not args.keep_going:
            return res

    if "collect" in args.stages:
        r = stage_collect(
            traces_dir=iter_dir / "traces",
            output=iter_dir / "sft.jsonl",
            preset=args.sft_preset,
            successful_only=args.successful_only)
        res.stages.append(r)
        if not r.ok and not args.keep_going:
            return res

    if "train_wm" in args.stages:
        r = stage_train_wm(
            traces_dir=iter_dir / "traces",
            db_path=Path(args.db) if args.db else None,
            output=iter_dir / "world_model.pkl")
        res.stages.append(r)
        if not r.ok and not args.keep_going:
            return res

    if "train_llm" in args.stages:
        r = stage_train_llm(
            sft_jsonl=iter_dir / "sft.jsonl",
            model_out=iter_dir / "model_weights",
            train_cmd=args.train_cmd)
        res.stages.append(r)

    # Persist a per-iteration summary for downstream tooling.
    summary_path = iter_dir / "rl_iter_summary.json"
    summary_path.write_text(json.dumps(
        {"iter_idx": iter_idx,
         "iter_dir": str(iter_dir),
         "stages": [asdict(s) for s in res.stages]},
        indent=2,
    ), encoding="utf-8")

    return res


def cmd_iter(args) -> int:
    iter_dir = Path(args.iter_dir or
                     (REPO_ROOT / "results" / "rl" / "iter_0"))
    res = run_iteration(args, 0, iter_dir)
    print("\n══ Iteration summary ══")
    for s in res.stages:
        print("  " + s.short())
    return 0 if res.all_ok else 1


def cmd_loop(args) -> int:
    """Run multiple iterations sequentially. Each iteration may consume
    artifacts (e.g. trained weights) from the previous one through the
    user's ``--train-cmd``."""
    base = Path(args.out_root)
    base.mkdir(parents=True, exist_ok=True)
    results: list[IterResult] = []
    for i in range(args.iters):
        d = base / f"iter_{i}"
        logger.info(f"\n──── Starting iteration {i} → {d} ────")
        r = run_iteration(args, i, d)
        results.append(r)
        if not r.all_ok and not args.keep_going:
            logger.warning(
                f"iteration {i} had failures and --keep-going off; stopping")
            break

    print("\n══ Loop summary ══")
    for r in results:
        ok_str = "OK" if r.all_ok else "FAIL"
        print(f"  iter_{r.iter_idx} [{ok_str}] {r.iter_dir}")
        for s in r.stages:
            print("    " + s.short())
    return 0 if all(r.all_ok for r in results) else 1


def cmd_collect(args) -> int:
    r = stage_collect(
        traces_dir=Path(args.traces_dir),
        output=Path(args.output),
        preset=args.sft_preset,
        successful_only=args.successful_only)
    print(r.short())
    if r.artifact:
        print(f"  → {r.artifact}")
    return 0 if r.ok else 1


def cmd_train_wm(args) -> int:
    r = stage_train_wm(
        traces_dir=Path(args.traces_dir),
        db_path=Path(args.db) if args.db else None,
        output=Path(args.output))
    print(r.short())
    if r.artifact:
        print(f"  → {r.artifact}")
    return 0 if r.ok else 1


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    # ── shared knobs ──────────────────────────────────────────────
    def _common(sp):
        sp.add_argument("--profile", default="whole_proof_repair")
        sp.add_argument("--benchmark", default="builtin")
        sp.add_argument("--provider", default="mock",
                        choices=["mock", "anthropic"])
        sp.add_argument("--limit", type=int, default=0)
        sp.add_argument("--max-samples", type=int, default=4)
        sp.add_argument("--model", default=None)
        sp.add_argument("--db", default=None,
                        help="Optional SQLite proof store for world-model training")
        sp.add_argument("--sft-preset", default="qwen3",
                        choices=["qwen3", "agentcpm", "openai"])
        sp.add_argument(
            "--successful-only", action="store_true",
            help="Only export successful dialogs to SFT JSONL")
        sp.add_argument(
            "--stages",
            default="eval,collect,train_wm,train_llm",
            help=("Comma-separated stage list. Drop unwanted stages "
                  "(e.g. --stages eval,collect)."))
        sp.add_argument(
            "--train-cmd", default=None,
            help=("Shell template for stage 4 (LLM training). Use "
                  "{sft_jsonl} and {model_out} placeholders. Omit to skip."))
        sp.add_argument(
            "--keep-going", action="store_true",
            help="Continue even if a stage fails")
        sp.add_argument("-v", "--verbose", action="store_true")
        sp.add_argument(
            "eval_extra", nargs="*",
            help="Extra args forwarded to run_eval.py")

    sp_iter = sub.add_parser("iter", help="Run one full iteration")
    sp_iter.add_argument("--iter-dir", default=None,
                          help="Output dir (default: results/rl/iter_0)")
    _common(sp_iter)

    sp_loop = sub.add_parser("loop", help="Run multiple iterations sequentially")
    sp_loop.add_argument("--iters", type=int, default=2)
    sp_loop.add_argument("--out-root", default="results/rl")
    _common(sp_loop)

    sp_coll = sub.add_parser("collect", help="Stage 2 only — dialogs → SFT JSONL")
    sp_coll.add_argument("--traces-dir", default="results/traces")
    sp_coll.add_argument("--output", default="sft.jsonl")
    sp_coll.add_argument("--sft-preset", default="qwen3",
                           choices=["qwen3", "agentcpm", "openai"])
    sp_coll.add_argument("--successful-only", action="store_true")

    sp_wm = sub.add_parser("train-wm", help="Stage 3 only — train world model")
    sp_wm.add_argument("--traces-dir", default="results/traces")
    sp_wm.add_argument("--db", default=None)
    sp_wm.add_argument("--output", default="world_model.pkl")

    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Normalize stages= to a list (only meaningful for iter/loop)
    if hasattr(args, "stages"):
        args.stages = [s.strip() for s in args.stages.split(",")
                        if s.strip()]
    if not hasattr(args, "successful_only"):
        args.successful_only = False

    handlers = {
        "iter":      cmd_iter,
        "loop":      cmd_loop,
        "collect":   cmd_collect,
        "train-wm":  cmd_train_wm,
    }
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
