#!/usr/bin/env python3
"""scripts/train_world_model.py — End-to-end WorldModel training pipeline.

Reads RichProofTrajectory data deposited by the project's runtime
(via ``ProofContextStore`` / ``KnowledgeWriter.ingest_step``) and
trains a LogisticRegression-on-TF-IDF tactic-success classifier. The
output is a single ``.pkl`` file that can be loaded by
``engine.world_model.make_world_model(path)`` and fed into the
``UnifiedProofRunner(world_model=...)`` to filter low-confidence
tactics before the Lean call.

Closes the v4 gap noted in REFACTOR_REPORT.md §九.4 ("WorldModel 从
Mock 升 Real"). Two data sources are supported:

  1. ``--db proofs.db`` — the SQLite store written by ProofContextStore.
                           Default. Walks ``proof_traces.step_details``.
  2. ``--from-trajectories pickle.pkl`` — a Python pickle of a
     ``list[RichProofTrajectory]`` (handy for offline training pipelines).

Examples
--------

::

    # Train from the default SQLite proof store
    python scripts/train_world_model.py \\
        --db results/knowledge/proofs.db \\
        --output models/world_model.pkl

    # Train from an offline trajectory dump (e.g. a prior eval sweep)
    python scripts/train_world_model.py \\
        --from-trajectories data/trajectories.pkl \\
        --output models/world_model_v2.pkl

    # Inspect a saved model
    python scripts/train_world_model.py --inspect models/world_model.pkl

After training, drop the model into the runner::

    from engine.world_model import make_world_model
    from prover.unified import UnifiedProofRunner

    wm = make_world_model("models/world_model.pkl")
    runner = UnifiedProofRunner(llm=..., lean_pool=..., world_model=wm)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

# Repo root on PYTHONPATH so this script runs from anywhere
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

def cmd_train(args) -> int:
    """Extract → train → save. Returns 0 on success, non-zero on failure."""
    from engine.world_model_trainer import WorldModelTrainer

    trainer = WorldModelTrainer(db_path=args.db)

    if args.from_trajectories:
        if not os.path.exists(args.from_trajectories):
            print(f"error: trajectories file not found: "
                  f"{args.from_trajectories}", file=sys.stderr)
            return 2
        with open(args.from_trajectories, "rb") as f:
            trajs = pickle.load(f)
        n = trainer.extract_from_trajectories(trajs)
        print(f"Loaded {n} samples from {args.from_trajectories}")
    else:
        if not os.path.exists(args.db):
            print(f"error: SQLite store not found: {args.db}\n"
                  f"  Run some proofs first, or pass --from-trajectories.",
                  file=sys.stderr)
            return 2
        n = trainer.extract_training_data(
            min_depth=args.min_depth, limit=args.limit)
        print(f"Loaded {n} samples from {args.db} "
              f"(min_depth={args.min_depth}, limit={args.limit})")

    if n < args.min_samples:
        print(f"error: need at least {args.min_samples} samples to train, "
              f"got {n}. Run more proofs or lower --min-samples.",
              file=sys.stderr)
        return 3

    print("Training LogisticRegression on TF-IDF features...")
    metrics = trainer.train(test_size=args.test_size)

    if "error" in metrics:
        print(f"training failed: {metrics}", file=sys.stderr)
        return 4

    print(f"  accuracy:      {metrics['accuracy']:.4f}")
    print(f"  f1:            {metrics['f1']:.4f}")
    print(f"  positive_rate: {metrics['positive_rate']:.4f}")
    print(f"  train_size:    {metrics['train_size']}")
    print(f"  test_size:     {metrics['test_size']}")
    print(f"  n_features:    {metrics['n_features']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out))
    print(f"\n✓ Saved model to {out}")
    print(f"  Use it with:  make_world_model({str(out)!r})")

    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_json).write_text(json.dumps(metrics, indent=2))
        print(f"  Metrics:    {args.metrics_json}")

    return 0

def cmd_inspect(args) -> int:
    """Print summary stats on an existing .pkl model."""
    path = args.inspect
    if not os.path.exists(path):
        print(f"error: not found: {path}", file=sys.stderr)
        return 2

    from engine.world_model import make_world_model

    wm = make_world_model(path)
    print(f"Loaded world model from {path}")
    print(f"  type:       {type(wm).__name__}")
    print(f"  is_trained: "
          f"{getattr(wm, 'is_trained', '(not exposed)')}")

    # Smoke prediction so the user can sanity-check
    pred = wm.predict(
        "⊢ n + 0 = n", "simp", hypotheses=[], context={})
    print(f"\nSample prediction:")
    print(f"  goal: '⊢ n + 0 = n'  tactic: 'simp'")
    print(f"  → likely_success={pred.likely_success}, "
          f"confidence={pred.confidence:.3f}")
    print(f"     reasoning: {pred.reasoning}")
    return 0

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument(
        "--db", default="proofs.db",
        help="SQLite proof store path (default: proofs.db)")
    p.add_argument(
        "--from-trajectories", default=None,
        help="Alternate input: pickled list[RichProofTrajectory]")
    p.add_argument(
        "--output", "-o", default="world_model.pkl",
        help="Output .pkl path (default: world_model.pkl)")

    p.add_argument(
        "--min-depth", type=int, default=1,
        help="Minimum proof depth to include (default: 1)")
    p.add_argument(
        "--limit", type=int, default=50000,
        help="Max traces to read from DB (default: 50000)")
    p.add_argument(
        "--min-samples", type=int, default=50,
        help="Refuse to train below this many samples (default: 50)")
    p.add_argument(
        "--test-size", type=float, default=0.2,
        help="Hold-out fraction (default: 0.2)")

    p.add_argument(
        "--metrics-json", default=None,
        help="Optional: also write metrics to this JSON path")

    p.add_argument(
        "--inspect", default=None, metavar="PKL",
        help="Inspect mode: print summary of an existing .pkl, no training")

    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    _setup_logging(args.verbose)
    if args.inspect:
        sys.exit(cmd_inspect(args))
    sys.exit(cmd_train(args))

if __name__ == "__main__":
    main()
