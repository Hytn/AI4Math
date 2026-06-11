<div align="center">

# AI4Math

**An agent framework for Lean 4 formal theorem proving**

[![Lean](https://img.shields.io/badge/Lean-4.24.0-blue)]()
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-963%20passed-brightgreen)]()

[简体中文](README.md) · **[English](README_EN.md)**

</div>

---

This project does not try to train a better prover model. Instead it turns "**how to use** a prover model" into a controlled experiment: on the same model weights, methodologies — single-shot whole-proof generation, verify-and-fix loops, cross-problem knowledge accumulation, heterogeneous parallelism — are composed into 20 named profiles. Switch `--profile` to switch the algorithm; `dialog.json` is the unified output format.

Concretely, for the DeepSeek-Prover-V2-7B line (arXiv:2504.21801): the paper reports 7B reaching pass@8192 = 82.0% on miniF2F-test using a single profile and i.i.d. sampling, with no information flow between samples. This framework provides 5 stackable profiles on the same 7B weights, asking the same question:

> **Given a sample budget, can stacked methodology beat the paper's baseline?**

This is an empirical hypothesis, not a theorem. The framework cannot answer it for you — what it does is let you run the comparison.

```
prover/unified/  ─→  profile (tools + framing + max_turns + obs policy)
agent/runtime/   ─→  AgentLoop: LLM calls + tool execution + dialog accumulation
engine/          ─→  Lean 4 REPL pool, three-tier verification, error intelligence
knowledge/       ─→  cross-problem SQLite (lemma validity / dialog snippets / tactic stats)
```

The `dialog.json` schema is uniform across profiles; all evaluations debug from this file.

---

## Quickstart

5-minute smoke test. No Lean, no API key, no GPU required:

```bash
git clone <this-repo> && cd AI4Math-v17
pip install -r requirements.txt
bash reproduce_minif2f.sh --smoke
```

Expected output:

```
miniF2F-test  (unverified)
─────────────────────────────
Solved      : 5/5  (100.0%)
pass@1      : 1.0000

⚠ Run was --lean-mode=skip; numbers only mean the LLM produced non-empty,
  sorry-free code, not that Lean 4 accepted it. Use --lean for real numbers.
```

The 100% is **not** a paper-style pass rate — it only verifies the pipeline is intact: the 244-problem dataset loads, the mock LLM emits non-empty code, the prefilter does not block it, and `dialog.json` writes to disk. To get real numbers you need Lean 4 + Mathlib installed and a real LLM connected.

Single-problem dry run, to inspect a complete dialog:

```bash
python run_unified.py --builtin nat_add_comm --profile dsp_v2_cot --provider mock
# → results/unified/builtin_nat_add_comm/dialog.json
```

---

## A/B testing on DSP-V2-7B

This is the framework's main use case. The **paper's 7B model (arXiv:2504.21801)** runs on a single 80GB H100 via vLLM. It costs roughly an order of magnitude less than the 671B variant and is the recommended starting point for switching methodologies.

### 1. Deploy the model (vLLM, single GPU)

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-Prover-V2-7B \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.92 \
    --max-model-len 32768 --port 8000 \
    --served-model-name DSP-V2-7B &
curl http://localhost:8000/v1/models    # confirm the model has loaded
```

Any OpenAI-compatible endpoint works (`--provider sglang/ollama/openai_compat`); the framework only speaks OpenAI Chat Completions. To run 671B, change `--tensor-parallel-size` to 8 and the model ID to `DeepSeek-Prover-V2-671B`. The rest of the command is unchanged.

### 2. Install Lean 4 + Mathlib (one-time)

miniF2F pins `v4.24.0`. Mismatched versions break Mathlib API compatibility, which makes every proof fail to verify.

```bash
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain leanprover/lean4:v4.24.0
source ~/.elan/env

# Let the script pull the dataset
bash reproduce_minif2f.sh --smoke

# Build mathlib (~5-10 min after `cache get`, otherwise 30-60 min)
cd data/miniF2F && lake exe cache get && lake build && cd ../..
```

### 3. Run the paper baseline (sanity check)

Reproduce the paper's Table 1 7B CoT pass@32 = 75.6% first, to verify your vLLM deployment and the framework wiring:

```bash
bash reproduce_minif2f.sh \
    --provider vllm --api-base http://localhost:8000/v1 \
    --model DSP-V2-7B --profile dsp_v2_cot \
    --samples 32 --temperature 1.0 --lean
```

Expected pass@32 lands in 70–80%. **If it's well below 70%, see Troubleshooting below.**

### 4. Run the 5-profile ablation (the main course)

```bash
bash reproduce_minif2f_7b_ablation.sh \
    --api-base http://localhost:8000/v1 --samples 32
```

Produces this comparison table:

```
═══ A/B comparison — DeepSeek-Prover-V2-7B on miniF2F-test ═══

  Paper baseline (arXiv:2504.21801, Table 1, 7B row):
    non-CoT pass@1=55.5%, pass@32=68.0%, pass@1024=73.2%, pass@8192=75.0%
    CoT     pass@1=58.6%, pass@32=75.6%, pass@1024=79.9%, pass@8192=82.0%

  Profile                         Solved   pass@32
  ──────────────────────────────────────────────
  dsp_v2_non_cot                  ?/244   ?.????   paper A.1 prompt verbatim
  dsp_v2_cot                      ?/244   ?.????   paper A.2 prompt verbatim ← reference
  dsp_v2_repair                   ?/244   ?.????   + verify-and-fix loop
  dsp_v2_repair_knowledge         ?/244   ?.????   + cross-problem lemma bank + dialog index
  dsp_v2_heterogeneous            ?/244   ?.????   + 4-way heterogeneous parallel + broadcast bus
```

The 5 profiles differ only by explicit switches (see `prover/unified/profiles.py::PRESETS["dsp_v2_*"]`):

| Profile | Paper baseline | Repair loop | Cross-problem KB | 4-way parallel | Tools |
|---|:-:|:-:|:-:|:-:|---|
| `dsp_v2_non_cot` | ✓ |   |   |   | (none) |
| `dsp_v2_cot` | ✓ |   |   |   | (none) |
| `dsp_v2_repair` | ✓ | ✓ |   |   | `lean_verify` |
| `dsp_v2_repair_knowledge` | ✓ | ✓ | ✓ |   | `+ lemma_bank` `+ premise_search` |
| `dsp_v2_heterogeneous` | ✓ | ✓ | ✓ | ✓ | `+ broadcast` |

The ablation script runs profiles in weak-to-strong order, accumulating the KB at `results/dsp_v2_7b_kb/`. The last two profiles read this KB as in-context demos — **the order matters; do not change it**.

### 5. Scaling up samples

`--resume` is on by default. First run pass@32 to see which profile is promising, then push that one to 1024:

```bash
python run_eval.py \
    --benchmark minif2f --profile dsp_v2_repair \
    --provider vllm --api-base http://localhost:8000/v1 \
    --model DSP-V2-7B --temperature 1.0 \
    --max-samples 1024 --lean-mode real \
    --project-dir data/miniF2F \
    --output-dir results/abl_run1/dsp_v2_repair \
    --knowledge-db results/dsp_v2_7b_kb/main.sqlite \
    --resume
```

Budget estimates (DSP-V2-7B on a single H100 vLLM, ~50 tok/s, CoT proof averages 4500 tok):

| Sample budget | Total tokens | Single H100 wall-clock |
|---:|---:|---:|
| pass@32 | ~350M | ~2 hours |
| pass@128 | ~1.4B | ~8 hours |
| pass@1024 | ~11B | ~64 hours |

Real vLLM throughput is 3-5× higher because of KV cache reuse and batched pipelining. **Run pass@32 first to see relative ranking — this is more useful than going straight to pass@8192.** `heterogeneous` runs 4-way in parallel, so the same `--samples` value costs 4× the compute of the other profiles.

---

## The 20 profiles

`--profile NAME` is the only knob for switching algorithms. Source of truth: `prover/unified/profiles.py::PRESETS`. The dumped readable view: `config/profiles/<name>.yaml`.

| Profile | Method |
|---|---|
| **DSP-V2 specific (paper reproduction + framework increments)** | |
| `dsp_v2_non_cot`, `dsp_v2_cot` | Paper Appendix A.1 / A.2 verbatim. **Paper 7B SOTA = 82.0%** comes from the `cot` variant. |
| `dsp_v2_repair` | + verify-and-fix loop |
| `dsp_v2_repair_knowledge` | + cross-problem lemma bank + dialog index |
| `dsp_v2_heterogeneous` | + 4-way heterogeneous parallel + broadcast bus |
| **General-purpose** | |
| `whole_proof` / `whole_proof_repair` | Single-shot whole proof / default path for general-purpose LLMs (Claude / GPT) |
| `dsp` | Draft-Sketch-Prove (Jiang 2023) |
| `reprover` | ReProver-style RAG + step-level |
| `leandojo` | Pure step-level, one tactic per turn |
| `heterogeneous` | General 4-way heterogeneous parallel |
| `conjecture_driven` | Actively conjectures auxiliary lemmas (PutnamBench / FATE-X-class problems) |
| `kimina_batch` / `pantograph_dsp` / `lookeng_lemma` | Backend-specific (Kimina / Pantograph / LooKeng) |
| `nfl_hybrid` | NFL-HR (Yao et al., EMNLP 2025) |
| `mcts` / `best_first` / `beam` | Tree search trio |

Adding a new profile = adding `PRESETS["my_method"] = Profile(...)` plus one framing prompt. `runner` / `agent_loop` / `tool_kits` are untouched.

---

## Provider and Backend

| `--provider` | Command fragment | Credential |
|---|---|---|
| `vllm` / `sglang` / `ollama` (self-hosted) | `--api-base http://localhost:8000/v1` | (none) |
| `anthropic` | `--model claude-opus-4-5` | `ANTHROPIC_API_KEY` |
| `openai` | `--model gpt-4o-mini` | `OPENAI_API_KEY` |
| `deepseek` | `--model deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| `openai_compat` | Any OpenAI-compatible endpoint | (per-service) |
| `mock` | (none, offline smoke) | (none) |

The `dsp_v2_*` framings are paper Appendix A verbatim prompts. **They are only meaningful for DSP-V2 models.** For Claude / GPT, use `whole_proof_repair` instead.

| `--backend` | When to use |
|---|---|
| `auto` (default) | Probes local / socket / http automatically |
| `local` | Local Lean 4 + Mathlib |
| `socket` | A process pool started via `docker/lean_daemon.py` |
| `kimina` | Kimina Lean Server (community) |
| `pantograph` | Need mvar focus / drafting |
| `lookeng` | Long proofs (PutnamBench / FATE-X), I/O optimized |
| `mock` | Fully offline |

To check whether the backend is actually running or has fallen back, look at `meta.backends.<name>.is_fallback` in `dialog.json` (`true` = Lean is not actually running).

---

## Command cheatsheet

```bash
# Single-problem dry run, see the full dialog
python run_unified.py --builtin nat_add_comm \
    --profile dsp_v2_repair \
    --provider vllm --api-base http://localhost:8000/v1 \
    --model DSP-V2-7B --temperature 1.0 --lean

# Reproduce paper 7B CoT pass@32 baseline
python run_eval.py \
    --benchmark minif2f --profile dsp_v2_cot \
    --provider vllm --api-base http://localhost:8000/v1 \
    --model DSP-V2-7B --temperature 1.0 \
    --max-samples 32 --lean-mode real \
    --project-dir data/miniF2F

# One-shot 5-profile ablation sweep
bash reproduce_minif2f_7b_ablation.sh \
    --api-base http://localhost:8000/v1 --samples 32

# YAML-driven benchmark × profile matrix
python scripts/eval/run_profile_matrix.py \
    --config config/experiments/profile_matrix/smoke.yaml

# Summarize matrix outputs as CSV / JSON / Markdown
python scripts/eval/summarize_profile_matrix.py \
    results/profile_matrix/smoke

# Inspect what happened on each problem
cat results/<run>/traces/minif2f/<problem_id>/dialog.json | jq '
  { problem: .meta.problem_id, success: .result.success,
    proof: .result.successful_proof[:200], turns: (.messages | length) }'
```

Main CLI flags:

| Flag | Description |
|---|---|
| `--profile NAME` | One of 19 profiles |
| `--temperature T` | Override profile default (DSP-V2 uses 1.0, general LLMs 0.7) |
| `--max-turns N` | Override profile default `max_turns` |
| `--max-samples K` | The K in pass@K |
| `--project-dir DIR` | Lean REPL working directory; auto-inferred from benchmark |
| `--pool-size N` | Lean REPL pool size (default 4) |
| `--knowledge-db PATH` | Shared KB across eval runs (required for A/B sweeps) |
| `--dialog-index PATH` | Cross-problem successful-dialog retrieval |
| `--lemma-bank-db PATH` | Cross-problem lemma bank (BM25) |
| `--world-model PATH` | sklearn world-model for tactic gating in step-level profiles |
| `--plugins-dir DIR` | Domain plugins |
| `--policy-engine` | Enable declarative PolicyEngine (5 default rules) |

---

## Profile Matrix Runs

Use `scripts/eval/run_profile_matrix.py` when you want a reproducible
benchmark × profile sweep from YAML instead of hand-writing many `run_eval.py`
commands. The matrix runner only expands combinations and assigns output
directories; each cell still delegates to `run_eval.py`, so metrics and
`dialog.json` stay identical to ordinary evaluations.

Example smoke config:

```bash
python scripts/eval/run_profile_matrix.py \
    --config config/experiments/profile_matrix/smoke.yaml
```

Example OpenAI-compatible Claude gateway config:

```bash
export PATH=/root/.elan/bin:$PATH
source .venv/bin/activate
export OPENAI_API_KEY=...

python scripts/eval/run_profile_matrix.py \
    --config config/experiments/profile_matrix/minif2f_claude_opus_4_7.yaml
```

Typical config fields:

```yaml
provider: openai_compat
api_base: "http://example.com/v1"
model: "claude-opus-4-7"
omit_temperature: true

lean_mode: real
max_samples: 1
limit: 5
resume: true
no_knowledge: true

benchmarks:
  - name: minif2f
    split: test
    project_dir: data/miniF2F

profiles:
  - whole_proof_repair
```

Notes:

- `max_samples` is the sample budget for pass@k. `max_samples: 1` reports
  pass@1; `max_samples: 32` reports pass@1/5/10/32.
- `limit: 0` means the full split. Use a small `limit` before full runs.
- `lean_mode: real` needs `lean` on `PATH`; if logs say `[FALLBACK] No Lean4
  binary found`, run `export PATH=/root/.elan/bin:$PATH` before starting.
- Some routed Claude/Bedrock models reject the `temperature` field; set
  `omit_temperature: true` for those OpenAI-compatible gateways.

Results are written under the configured `output_dir`, for example:

```text
results/profile_matrix/minif2f_claude_opus_4_7/
  matrix.yaml
  matrix_runs.json
  logs/<benchmark>/<profile>.log
  <benchmark>/<profile>/evals/eval_<benchmark>_<split>.json
  <benchmark>/<profile>/traces/<benchmark>/<problem_id>/dialog.json
```

Summarize a completed matrix:

```bash
python scripts/eval/summarize_profile_matrix.py \
    results/profile_matrix/minif2f_claude_opus_4_7
```

This writes:

```text
summary.csv
summary.json
summary.md
```

---

## Datasets and RL

7 built-in benchmarks: miniF2F (244), PutnamBench (672), ProofNet (360), FATE-H/M/X (100/150/100), FormalMATH (clone manually: `git clone https://github.com/Sphere-AI-Lab/FormalMATH-Bench data/FormalMATH`).

`results/<run>/traces/<benchmark>/<id>/dialog.json` doubles as natural SFT/RL training data. The RL flywheel:

```bash
python scripts/rl_pipeline.py iter \
    --iter-dir results/rl/iter_0 \
    --profile dsp_v2_repair_knowledge --benchmark minif2f \
    --provider vllm --api-base http://localhost:8000/v1
```

Four stages, each independently re-runnable: `eval` → `collect` (SFT-ready jsonl) → `train_wm` (sklearn world-model) → `train_llm` (input for an external trainer: TRL / verl / slime).

---

## Layout

```
prover/unified/    PRESETS source of truth + system_prompts + tool_kits + runner
prover/            conjecture / decompose / premise / lemma_bank / verifier
agent/             AgentLoop, tools, brain (LLM provider), persistence
engine/            Lean REPL pool, search algebra, policy, broadcast bus
knowledge/         SQLite KB + DialogIndex
sampler/           RL trajectory sampling (verl / slime / TRL adapter)
benchmarks/        miniF2F + PutnamBench + ProofNet + FATE + FormalMATH loaders
config/            default.yaml + 19 profile YAML templates
data/              Benchmark problems (1626 built-in + FormalMATH optional)
plugins/           YAML-driven domain plugins
docs/              ARCHITECTURE.md + dialog.json schema
tests/             963 unit / integration tests

reproduce_minif2f.sh                miniF2F-test reproduction (single profile)
reproduce_minif2f_7b_ablation.sh    5-profile sweep
run_unified.py                      Single-problem CLI
run_eval.py                         Batch CLI
scripts/eval/run_profile_matrix.py  YAML benchmark × profile matrix runner
scripts/eval/summarize_profile_matrix.py  Matrix summary writer
eval.sh                             Generic eval entrypoint
```

For source code, start with [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For mathematicians without coding background, start with [`TUTORIAL_CN.md`](TUTORIAL_CN.md) (Chinese).

---

<details>
<summary><b>Honest disclaimers — what this project does not do</b></summary>

<br>

- **It will not make a non-prover LLM suddenly able to prove Lean theorems.** The framework is leverage, not the engine. Model weights are the engine.
- **`pass@k` is meaningless under `--lean-mode skip`.** Every report is tagged `[unverified]` — do not read it as "pass rate".
- **"Framework increments beat the paper baseline" is an empirical hypothesis, not a theorem.** `reproduce_minif2f_7b_ablation.sh` is an experimental tool, not a conclusion. A profile that fails to add value on miniF2F is itself a useful negative finding.
- **The "world model" is currently sklearn LogisticRegression.** It works as a tactic-validity predictor — it is not the "full state-dynamics network" the architecture diagram might suggest. To swap in Qwen / GPT as a world model, edit one function: `engine/world_model.py::predict()`.
- **The `reprover` profile uses BM25 + character n-gram TF-IDF, not a dense neural retriever.** Replacing it with SBERT / ColBERT is a one-file change.
- **`mathlib_core.jsonl` defaults to ~334 entries.** Real Mathlib4 is on the order of 10⁵. Use `scripts/export_mathlib_premises.py` to expand the pool before running real evaluations.
- **`conjecture_driven` lets the LLM invent auxiliary lemmas while proving a given theorem, but it does not invent the theorems themselves.** All problems come from the 7 built-in benchmarks.

</details>

<details>
<summary><b>Troubleshooting</b></summary>

<br>

**Smoke test fails:**

| Symptom | Fix |
|---|---|
| `Python 3.10+` error | Install conda / pyenv |
| `ModuleNotFoundError` | `pip install -r requirements.txt --break-system-packages` |
| Dataset `git clone` fails | Configure proxy or use a mirror |
| Dataset has < 240 problems | `rm -rf data/miniF2F && bash reproduce_minif2f.sh --smoke` |

**`lake build` hangs or fails:**

| Symptom | Fix |
|---|---|
| `lake: command not found` | `source ~/.elan/env` |
| `lean --version` is not v4.24.0 | Inside `data/miniF2F/`: `elan override set leanprover/lean4:v4.24.0` |
| `lake build` runs >60 minutes | Interrupt, then `lake clean && lake exe cache get && lake build` |

**vLLM baseline pass@32 is well below 75.6%**, in decreasing order of likelihood:

1. **Temperature is not 1.0.** Check `messages[].metadata.temperature` in `dialog.json`.
2. **Lean is not actually running.** If `meta.backends.lean_pool.is_fallback` is `true` for any problem, the REPL failed to start; check that `data/miniF2F/.lake/build/bin/repl` exists.
3. **vLLM is serving the wrong model.** `curl http://localhost:8000/v1/models` to see what's actually loaded.
4. **`max-model-len` is too small** and CoT output is being truncated (CoT proofs average 4488 tokens).
5. **The prompt is not paper-verbatim.** `python -c "from prover.unified.system_prompts import _FRAMINGS; print(_FRAMINGS['deepseek_prover_v2_cot'])"` — should start with `Complete the following Lean 4 code:`.

**Ablation crashes mid-run:**

```bash
# --resume is on by default; rerunning with the same --root skips finished problems
bash reproduce_minif2f_7b_ablation.sh \
    --api-base http://localhost:8000/v1 --samples 32 \
    --root results/dsp_v2_7b_ablation_<timestamp>
```

`dialog.json` is the project's flight recorder. Every evaluation debug starts from there.

</details>

---

## Retrieval providers (LeanSearch v2 / Loogle / LeanStateSearch)

`prover/premise/providers/` is a pluggable retrieval layer in front of
`premise_search`. **Off by default — zero behavior change unless enabled**:

```bash
# Online semantic + pattern retrieval, local TF-IDF as fallback
export AI4MATH_PREMISE_PROVIDERS="leansearch_v2,loogle,local_tfidf"

# Reproducible eval: all online lookups go through a snapshot cache
export AI4MATH_RETRIEVAL_CACHE=results/run1/retrieval.jsonl
export AI4MATH_RETRIEVAL_CACHE_MODE=rw     # rw / ro (frozen replay) / off
```

Providers: `leansearch_v2` (leansearch.net, Mathlib semantic SOTA),
`loogle` (type-pattern search, complementary), `leanstatesearch`
(proof-state-conditioned, experimental; pass `goal_state` in the tool
call), `local_tfidf` (existing TF-IDF over `data/premises/*.jsonl`).
Failed/unreachable providers degrade gracefully and are reported in the
tool result's `degraded_providers` metadata.

**Full Mathlib corpus** (~10^5 theorems, replaces the 404-entry seed):

```bash
python scripts/export_mathlib_premises_full.py --lake-project data/miniF2F
```

Measure the retrieval layer itself (Recall@K / nDCG@K / MRR — don't
attribute retriever changes from end-to-end pass@k noise):

```bash
python scripts/eval/eval_retrieval.py --qrels my_qrels.jsonl \
    --providers leansearch_v2,local_tfidf --k 1 5 10
```

## New benchmarks: lean-eval & MathArena

**lean-eval** (comparator paradigm — *not* compile-and-no-sorry):

```bash
git clone https://github.com/leanprover/lean-eval data/LeanEval
python run_eval.py --benchmark leaneval --lean-mode skip ...   # generate
python scripts/eval/run_leaneval.py --from-traces results/<run>/traces/leaneval/
```

A problem counts as solved **iff the comparator accepts the
submission** written into `generated/<id>/Submission.lean`. Plain
`lean_verify` results on this benchmark are pre-filtering only.

**MathArena** (uncontaminated live competitions; data is CC BY-NC-SA
4.0, fetched at runtime, never vendored):

```bash
python scripts/eval/fetch_matharena.py --comp aime_2026
# Line 1 — informal reasoner benchmarking (official avg@4):
python scripts/eval/matharena_informal.py --comp aime_2026 --provider anthropic --model ...
# Line 2 — formal: answer-aware autoformalization, then the normal eval loop:
python scripts/eval/matharena_autoformalize.py --comp aime_2026 --provider anthropic --model ...
python run_eval.py --benchmark matharena --split aime_2026 ...
```

Autoformalized statements carry `autoformalized_unverified` +
`formalizer:<model>` tags; report these alongside any numbers.

## Hilbert-style recursive decomposition (`run_hilbert.py`)

Standalone reproduction of Hilbert (arXiv:2509.22819; reference
open-source implementation: Gödel's Poetry) — a third entry point,
fully decoupled from the profile system because it needs **per-role
models** (informal reasoner + dedicated prover):

```bash
# smoke (no Lean, no API keys)
python run_hilbert.py --statement "theorem t : 1 + 1 = 2 := by sorry" --backend mock --out /tmp/h
# real run
python run_hilbert.py --config config/hilbert.yaml --benchmark minif2f --lean --limit 20
```

Per node: prover whole-proof passes → reasoner shallow-solve (cheap
one-tactic closes) → recursive decomposition into self-contained
lemmas (depth ≤ D=5) → assembly, which must compile as a whole. Traces
land in `hilbert_trace.json` with per-role token accounting; mock runs
are marked `is_mock` and must be excluded from any reported numbers.

## End-to-end walkthrough (实测引导)

每条命令都已通过 `scripts/eval/integration_drill.py` 实跑验收 —— 该脚本
在**无 Lean / 无 API key / 无外网**的最小环境把下面所有模块串通一遍
(任何一环断裂即非零退出), 拿到仓库后建议先跑它做环境体检:

```bash
python scripts/eval/integration_drill.py          # ~30s, 全离线
```

以下为真实环境 (有 Lean 工具链 / API key / 可出网) 的分步引导。

### Walkthrough A — 检索层: 从快照预热到可复现评测

```bash
# A1. 全量 Mathlib 语料 (一次性, 10-30 min; 之后 local_tfidf 自动加载)
python scripts/export_mathlib_premises_full.py --lake-project data/miniF2F

# A2. 预热在线检索快照 (rw 模式; 同一查询只打一次 API, 快照与 top_k 解耦)
export AI4MATH_RETRIEVAL_CACHE=results/run1/retrieval.jsonl
export AI4MATH_RETRIEVAL_CACHE_MODE=rw
python scripts/eval/eval_retrieval.py \
    --qrels data/retrieval_qrels/my_qrels.jsonl \
    --providers leansearch_v2,loogle,local_tfidf \
    --k 1 5 10 --out results/run1/retrieval_eval.json
# qrels 每行: {"query": "...", "relevant": ["Nat.add_comm", ...],
#              "goal_state": "⊢ ..."}   (goal_state 可选)

# A3. 冻结重放 (发布数字必须用这个口径; degraded_queries 必须为 0)
AI4MATH_RETRIEVAL_CACHE_MODE=ro python scripts/eval/eval_retrieval.py ...同上

# A4. 接入 agent loop (对既有 profile 零改动, 环境变量 opt-in)
export AI4MATH_PREMISE_PROVIDERS="leansearch_v2,loogle,local_tfidf"
python run_unified.py --profile repair --benchmark minif2f --lean ...
# 工具结果里 provider 命中排在最前 (source 字段标来源);
# 某 provider 不可用会进 ToolResult.metadata.degraded_providers — 评测
# 后聚合该字段, 避免"整场评测都在降级跑"而不自知。
```

### Walkthrough B — lean-eval (comparator 官方口径)

```bash
git clone https://github.com/leanprover/lean-eval data/LeanEval
cd data/LeanEval && <按其 README 完成 lake build 与 workspace 生成> && cd -

# B1. 生成证明 (本框架口径只是预筛, --lean-mode skip 纯生成更省)
python run_eval.py --benchmark leaneval --profile repair --lean-mode skip ...

# B2. comparator 终审 (官方口径; 自动备份/写入/评分/还原 Submission.lean)
python scripts/eval/run_leaneval.py \
    --from-traces results/<run>/traces/leaneval/ --out results/leaneval
# 输出 leaneval_results.json 带 scoring="comparator" 标记;
# lean-eval 的评分 CLI 若演进, 用 --validate-cmd/--score-cmd 覆盖
# (drill 即用此机制以桩 comparator 验收全流程)。
```

### Walkthrough C — MathArena 双线

```bash
# C1. 运行时拉数据 (CC BY-NC-SA 4.0, 不入库不分发)
pip install datasets huggingface_hub
python scripts/eval/fetch_matharena.py --comp aime_2026

# C2-informal. reasoner 选型基准 (官方 avg@4 口径 + token 成本对账)
python scripts/eval/matharena_informal.py --comp aime_2026 \
    --provider anthropic --model claude-opus-4-5 --n-runs 4 \
    --out results/matharena/aime_2026_informal.json

# C2-formal. answer-aware 自动形式化 → 红线静态检查 → 人工修订
python scripts/eval/matharena_autoformalize.py --comp aime_2026 \
    --provider anthropic --model claude-opus-4-5
# flagged=true 的条目 loader 拒载; 人工修订 statement 后置 flagged=false。
# 报告里必须带 formalizer 模型与 flagged 比例 (tags 已自动携带)。

# C3. 形式化线进正常评测循环
python run_eval.py --benchmark matharena --split aime_2026 --lean ...
```

### Walkthrough D — Hilbert 递归分解

```bash
# D1. 冒烟 (无 Lean / 无 key; trace 标 is_mock, 不可计入真实数字)
python run_hilbert.py --statement "theorem t : 1 + 1 = 2 := by sorry" \
    --backend mock --out /tmp/h

# D2. 配 per-role 模型 (reasoner 优先上强模型 — 论文消融结论)
$EDITOR config/hilbert.yaml

# D3. 真实运行 (可与 C 衔接: --benchmark matharena --split aime_2026)
python run_hilbert.py --config config/hilbert.yaml \
    --benchmark minif2f --split test --limit 20 --lean \
    --out results/hilbert_minif2f
# 产物: <out>/<id>/hilbert_trace.json (递归树 + per-role token 对账)
#       <out>/summary.json (solved/total + 每题 method 归因)
```

## Results

Results table pending. Numbers will only be published here together with
the exact reproduction command, the profile YAML, and the corresponding
`dialog.json` traces (per the honest-disclosure policy above). Until then,
run `reproduce_minif2f_7b_ablation.sh` to produce your own comparison table.

<!-- A previous draft listed `repair pass@32 = 97.13%` with no run
     configuration or traces attached. That number exceeded the
     DSP-V2-671B paper baseline and could not be audited, so it was
     removed rather than risk being quoted as a verified result. -->

## Citation and License

```bibtex
@software{ai4math2026,
  title = {AI4Math: An Agent Operating System for Formal Theorem Proving},
  year  = {2026},
  url   = {https://github.com/ai4math/ai4math}
}
```

[MIT License](LICENSE)
