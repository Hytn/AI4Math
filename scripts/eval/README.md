# Evaluation Scripts

`run_profile_matrix.py` runs a benchmark/profile matrix by delegating each cell
to the existing `run_eval.py` entrypoint.  It does not reimplement proving or
metrics logic.

Example:

```bash
python scripts/eval/run_profile_matrix.py \
  --config config/experiments/profile_matrix/smoke.yaml
```

Summarize a completed matrix:

```bash
python scripts/eval/summarize_profile_matrix.py results/profile_matrix/smoke
```

