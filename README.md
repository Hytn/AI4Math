<div align="center">

# AI4Math

### An Agent Operating System for Formal Theorem Proving

**English** · [中文](README_zh.md) · [Interactive Demo ↗](https://ai4math.github.io/ai4math) · [Tutorial (Chinese) ↗](TUTORIAL_CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Lean 4](https://img.shields.io/badge/Lean-4.24.0-orange.svg)](https://lean-lang.org)
[![Tests](https://img.shields.io/badge/Tests-797%20passed-brightgreen.svg)](#testing)
[![Problems](https://img.shields.io/badge/Benchmarks-6%2C826%20problems-purple.svg)](#benchmarks)

<br>

*Others are building better proof generators —<br>AI4Math is building the operating system that proof generators run inside.*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Why AI4Math?](#why-ai4math)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [中文版 (README_zh.md)](README_zh.md)

---

## Overview

AI4Math is an **agent operating system** that enables hundreds of heterogeneous AI mathematicians to collaboratively discover formal proofs in Lean 4. Rather than being yet another proof generator, AI4Math provides the foundational infrastructure — a verification OS, a living knowledge system, a world model, and a multi-agent society — that any LLM can plug into.

> **See it in action →** Our [interactive demo](https://ai4math.github.io/ai4math) walks through a full Putnam competition problem being solved, step by step, with every internal component visible.

## Why AI4Math?

Current state-of-the-art systems (DeepSeek-Prover, Goedel-Prover, Kimina) share a fundamental limitation:

| | Current Paradigm | AI4Math |
|---|---|---|
| **Feedback** | 1 bit per attempt (pass/fail) | ~100 bits structured diagnostics |
| **Communication** | Zero cross-direction sharing | Real-time broadcast across all agents |
| **Learning** | Failed attempts are discarded | Every failure deposits reusable knowledge |
| **Verification** | Full Lean compilation (2–12s) | 3-tier: syntax ~1μs → REPL ~50ms → full ~3s |
| **Architecture** | Monolithic LLM | Composable OS with pluggable components |

These differences compound. On hard problems requiring 50+ attempts, AI4Math's knowledge flywheel means attempt #50 benefits from everything learned in attempts #1–49.

## Key Features

🧠 **Multi-Agent Society** — 11 specialized roles (generator, planner, repairer, critic, decomposer...) explore in parallel with real-time knowledge sharing via a broadcast bus.

⚡ **3-Tier Verification** — L0 syntax prefilter (~1μs) catches 60% of bad proofs instantly. L1 REPL (~50ms) provides structured feedback. L2 full Lean compilation (~3s) gives definitive results. 95% of invalid proofs never reach Lean.

📚 **Living Knowledge System** — A 4-layer pyramid (raw traces → tactic effectiveness → strategy patterns → concept graphs) built on SQLite with WAL. Knowledge decays, self-corrects, and evolves across proof sessions.

🔄 **Policy Engine** — Composable, inspectable strategy rules replace hardcoded thresholds. Budget-aware escalation across sample, token, and wall-time dimensions. Automatic recovery from REPL crashes, API errors, and timeouts.

🏗️ **Proof Pipeline** — State-machine-driven proof lifecycle with checkpoint/resume support. Green Contract verification (NONE → SYNTAX_CLEAN → GOALS_CLOSED → SORRY_FREE). Context compression keeps LLM prompts under budget.

🔒 **Integrity Verification** — Deep sorry/axiom/unsafeCoerce detection prevents proofs that "cheat" through axiom injection or sorry redefinition — a real vulnerability in other systems.

🔌 **Extensible Plugin System** — Domain-specific strategies (algebra, number theory, analysis) are declared in YAML with custom premises and few-shot examples. No source code changes needed.

---

## Installation

### Prerequisites

- **Python 3.12+**
- **Lean 4** (v4.24.0) with **Mathlib** — required for real proof verification
- An **Anthropic API key** — for LLM-powered proof generation

### Step 1: Clone and install Python dependencies

```bash
git clone https://github.com/ai4math/ai4math.git
cd ai4math
pip install -r requirements.txt
```

### Step 2: Set up Lean 4 + Mathlib (for real verification)

If you are a mathematician new to Lean, follow these steps carefully:

<details>
<summary><b>macOS / Linux — Install Lean 4 from scratch</b></summary>
<br>

```bash
# 1. Install elan (the Lean version manager, like rustup for Rust)
curl https://elan-init.github.io/elan/elan-init.sh -sSf | sh
source ~/.profile   # or restart your terminal

# 2. Verify installation
lean --version       # should show: leanprover/lean4:v4.x.x
lake --version       # Lake is Lean's build system

# 3. Clone our Lean project with Mathlib (first build takes ~20–30 min)
cd data/miniF2F
lake build           # downloads and compiles Mathlib — go get coffee ☕

# 4. Verify Mathlib works
echo 'import Mathlib
#check Nat.add_comm' | lean --stdin
# Should print: Nat.add_comm : ∀ (n m : ℕ), n + m = m + n
```

</details>

<details>
<summary><b>Windows — Install via WSL2</b></summary>
<br>

```bash
# 1. Open PowerShell as admin, install WSL2
wsl --install -d Ubuntu-24.04

# 2. Inside WSL, follow the macOS/Linux instructions above
```

</details>

<details>
<summary><b>Docker — Zero-install option (recommended for evaluation)</b></summary>
<br>

```bash
cd docker
docker compose build    # builds Lean4+Mathlib image (~30 min first time)
docker compose up -d    # starts REPL daemon
# See "Docker Deployment" section below for full details
```

</details>

### Step 3: Configure your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Quick Start

### Single theorem — interactive walkthrough

```bash
# Prove a built-in theorem with full pipeline trace:
python run_single_lane.py --builtin nat_add_comm --provider anthropic

# Prove a custom theorem:
python run_single_lane.py \
  --theorem "theorem t (n : Nat) : n + 0 = n" \
  --provider anthropic

# Verbose mode (shows full LLM prompts and responses):
python run_single_lane.py --provider anthropic --builtin nat_add_comm --verbose
```

This walks through all 10 pipeline stages with intermediate output:

```
Step 1:  Problem loading & analysis
Step 2:  Lane runtime assembly (EventBus, PolicyEngine, Knowledge, AgentPool)
Step 3:  Knowledge injection (KnowledgeReader → prompt)
Step 4:  Direction planning (3–4 heterogeneous exploration directions)
Step 5:  Proof loop (generate → verify → policy → recover)
Step 6:  State machine result (event-driven transitions)
Step 7:  Event stream log
Step 8:  Green Contract check (NONE → SYNTAX_CLEAN → GOALS_CLOSED → SORRY_FREE)
Step 9:  Context compression (one-liner summary + prompt injection)
Step 10: Dashboard overview
```

### Benchmark evaluation

```bash
# Quick evaluation — 5 built-in problems (~2 min)
bash eval.sh --real --benchmark builtin

# Quick sweep — 10 problems per benchmark
bash eval.sh --real --quick

# Full miniF2F evaluation (488 problems, 32 samples each)
bash eval.sh --real --benchmark minif2f --samples 32

# Use a specific model
bash eval.sh --real --benchmark builtin --model claude-opus-4-6

# Enable multi-role (Generator ↔ Repairer alternation)
bash eval.sh --real --benchmark builtin --multi-role
```

### Legacy single-problem mode (without Lane runtime)

```bash
python run_single.py --builtin nat_add_comm --provider anthropic
python run_single.py --theorem "theorem test (n : Nat) : n + 0 = n" --provider anthropic
```

---

## Benchmarks

AI4Math ships with **6,826 problems** across 7 benchmarks covering the full difficulty spectrum:

| Benchmark | Problems | Difficulty | Description |
|-----------|----------|------------|-------------|
| **builtin** | 5 | Easy–Medium | Smoke tests (recommended for first-time users) |
| **miniF2F** | 488 | AMC → IMO | Most widely used formal math benchmark |
| **PutnamBench** | 672 | Collegiate | 1962–2024 Putnam competition problems |
| **ProofNet** | 360 | Undergrad | Analysis, algebra, topology core curriculum |
| **FATE-M/H/X** | 350 | Undergrad → PhD | Abstract algebra, full difficulty coverage |
| **FormalMATH** | 5,560 | Mixed | Multi-domain, multi-difficulty |

### Current SOTA comparison (miniF2F-test, 244 problems)

| Method | Pass@32 | Type |
|--------|---------|------|
| Goedel-Prover-V2-32B | **90.4%** | Full-proof generation |
| Kimina-Prover-72B | 84.0% | Full-proof generation |
| DeepSeek-Prover-V2-671B | 82.4% | Full-proof generation |
| **AI4Math (Claude Opus 4.6)** | *Evaluation in progress* | Agent platform |

> AI4Math is an **orthogonal contribution**: it is the platform these generators can plug into, not a competing generator. Any model can serve as AI4Math's proof engine.

---

## Architecture

> **Interactive version →** See the [full architecture visualization](https://ai4math.github.io/ai4math#pillars) with animated data flow and the [data flow diagram](https://ai4math.github.io/ai4math#flowDiagram) with hover-to-explore components.

AI4Math is built on four layers that form a self-reinforcing flywheel:

| Layer | Module | Lines | Purpose |
|-------|--------|-------|---------|
| **④ Agent Society** | `agent/`, `prover/pipeline/` | ~11K | 11 specialized roles, parallel exploration, real-time broadcast |
| **③ World Model** | `engine/world_model.py` | ~1K | Internalized Lean 4 state dynamics, predict tactic effects without calling the prover |
| **② Living Knowledge** | `knowledge/` | ~2.2K | 4-layer pyramid (traces → tactics → strategies → concepts), decay & evolution |
| **① Verification OS** | `engine/` | ~15K | REPL pool, 3-tier verification, elastic scaling, incremental compilation |

**Knowledge flows as a flywheel:**

```
④ explores → ① verifies → ② deposits knowledge → ③ trains world model → ④ uses knowledge → …
```

A static architecture diagram is also available at [`docs/architecture.svg`](docs/architecture.svg).

---

## v3 Unified Pipeline (Profile-driven)

> **TL;DR — Every mainstream theorem-proving method except MCTS is reproducible through `prover.unified` by switching the `--profile` flag. No code changes.**

### Design essence

Modern LLM-based theorem proving methods differ along just three dimensions:
1. **`max_turns`** — how many LLM calls per session
2. **`tools`** — what the LLM can call
3. **`system_prompt` + initial user message** — what work mode the LLM enters

These three live in a single `Profile` dataclass executed by `UnifiedProofRunner`. Same code, same `dialog.json` schema, six different algorithms.

### Active presets

| Profile | Method | `max_turns` | `tools` | Status |
|---|---|---|---|---|
| `whole_proof` | DeepSeek-Prover · Kimina · Goedel | 1 | `[]` | ✅ complete |
| `whole_proof_repair` | Compile-and-fix loop (project default) | 6 | `[lean_verify]` | ✅ complete |
| `dsp` | Draft-Sketch-Prove | 10 | `[decompose, premise_search, lean_verify]` | ✅ complete |
| `reprover` | ReProver (RAG + step-level) | 30 | `[premise_search, tactic_apply, goal_inspect]` | ✅ complete |
| `leandojo` | LeanDojo (pure step-level) | 50 | `[tactic_apply, goal_inspect, lean_auto]` | ✅ complete |
| `heterogeneous` | AI4Math 4-way parallel + broadcast | 4 | sub-profiles + `broadcast` | ✅ complete |

### How each algorithm is parameterised

#### `whole_proof` ↔ DeepSeek-Prover / Kimina / Goedel

```python
Profile(
    tools=[],                            # no tools = forced single-shot
    max_turns=1,
    framing="whole_proof",               # "Output one ```lean block. Do NOT call tools."
    observation=ObservationPolicy(
        inject_few_shot=True,            # 5 representative Mathlib examples
        inject_premises_in_prompt=True,  # top-N retrieved lemmas pre-injected
        auto_inject_lean_compile=True,   # runtime force-verifies emitted code
    ),
)
```
**Coverage**: ✅ single-shot ✅ pass@k via `--max-samples K` (i.i.d.) ✅ few-shot from `common/prompt_builder.py` ✅ premise injection via `PremiseSelector` ✅ Lean4 final verify. ⚠️ No explicit chain-of-thought scaffolding (relies on underlying model's reasoning ability).

#### `whole_proof_repair` ↔ Compile-and-fix loop

```python
Profile(
    tools=[ToolKit.LEAN_VERIFY],
    max_turns=6,
    framing="whole_proof_repair",        # "Submit proof, see errors, fix, repeat."
    observation=ObservationPolicy(
        compress_errors_budget=1200,     # error compression via lane.summary_compressor
        auto_inject_lean_compile=True,
    ),
)
```
**Coverage**: ✅ multi-round closure ✅ structured error feedback ✅ real Lean4 verification ✅ full dialog history.

#### `dsp` ↔ Draft-Sketch-Prove

```python
Profile(
    tools=[ToolKit.DECOMPOSE, ToolKit.PREMISE_SEARCH, ToolKit.LEAN_VERIFY],
    max_turns=10,
    framing="dsp",                        # explicit phases A-E in system prompt
)
```
**Coverage**: ✅ phase A (sketch via comment block) ✅ phase B (`DecomposeSubgoalTool` calls `prover/decompose/goal_decomposer.py`) ✅ phase C (premise search) ✅ phases D-E (formalize + repair). ⚠️ Original DSP used informal-to-formal training data; this framework is zero-shot DSP via system prompt.

#### `reprover` ↔ ReProver

```python
Profile(
    tools=[ToolKit.PREMISE_SEARCH, ToolKit.TACTIC_APPLY, ToolKit.GOAL_INSPECT],
    max_turns=30,
    framing="step_level_with_retrieval",
    observation=ObservationPolicy(
        auto_inject_goal_state=True,      # tactic_apply natively returns new goals
        inject_premises_in_prompt=False,   # ReProver retrieves on-demand
        inject_few_shot=False,
    ),
)
```
**Coverage**: ✅ true step-level via `TacticApplyTool` (real REPL apply, returns `remaining_goals` + `is_proof_complete`) ✅ on-demand retrieval ✅ 30-turn horizon. ⚠️ Retriever is TF-IDF/BM25 hybrid; original ReProver used ColBERT fine-tuned on LeanDojo data.

#### `leandojo` ↔ LeanDojo

```python
Profile(
    tools=[ToolKit.TACTIC_APPLY, ToolKit.GOAL_INSPECT, ToolKit.LEAN_AUTO],
    max_turns=50,
    framing="step_level_pure",
)
```
**Coverage**: ✅ pure step-level (shares `TacticApplyTool` with reprover) ✅ Mathlib hammer (`exact?`/`apply?`/`aesop`/`polyrith`) ✅ 50-turn horizon. ⚠️ Original LeanDojo also had a best-first search wrapper — that part lives in `EXPERIMENTAL_PRESETS["best_first"]` (set aside per scope).

#### `heterogeneous` ↔ AI4Math project flagship

```python
Profile(
    search=SearchConfig(
        kind="parallel",
        parallel_profiles=["whole_proof", "reprover", "leandojo", "whole_proof_repair"],
    ),
    tools=[ToolKit.LEAN_VERIFY, ToolKit.BROADCAST],
)
```
**Coverage**: ✅ true parallelism via `asyncio.gather` ✅ shared `BroadcastBus` for cross-direction discoveries ✅ heterogeneous (4 different framings/tools/turns) ✅ `ResultFuser` cross-fusion ✅ backward-compat with v2 `assembly.py` constructor.

### How the code becomes a unified base

```
┌────────────────────────────────────────────────────────────────┐
│                UnifiedProofRunner (single entry)                 │
│                                                                  │
│   prove(problem, profile)                                        │
│      ── build initial prompt (theorem + few-shot + premises)     │
│      ── AgentLoop                                                 │
│           ├── system_prompt = framing[X]                          │
│           ├── tools = ToolKit → Tool instances                   │
│           ├── max_turns = N                                       │
│           └── multi-turn: LLM → tool call → tool result → ...    │
│      ── auto_verify (safety net)                                  │
│      ── dialog.json (uniform output)                              │
└────────────────────────────────────────────────────────────────┘
```

**One data path for all methods**:
1. **Entry**: `UnifiedProofRunner.run(problem, profile)`
2. **Core loop**: `agent.runtime.AgentLoop`
3. **Tools**: `agent.tools.ToolRegistry`
4. **Persistence**: `dialog.json` schema v2.0
5. **Bridge**: `prover.unified.adapters` for legacy `ProofAttempt`/`AgentResult` compat

**Adding a new method = adding one Profile** in `prover/unified/profiles.py` (or registering from YAML). No changes to runner / loop / tools / pipeline.

### Honest implementation-vs-paper diff

| Profile | Implemented | Differs from paper |
|---|---|---|
| `whole_proof` | full single-shot + few-shot + premise + final verify | no forced CoT scaffold |
| `whole_proof_repair` | full closure | — |
| `dsp` | all 5 phases via tools | zero-shot DSP (no I2F training data) |
| `reprover` | step-level + on-demand retrieval | TF-IDF/BM25 instead of fine-tuned ColBERT |
| `leandojo` | pure step-level + hammer | search wrapper in `EXPERIMENTAL_PRESETS` |
| `heterogeneous` | full parallel + broadcast + fusion | — |
| `mcts` / `beam` / `best_first` | code complete in `EXPERIMENTAL_PRESETS` | not yet merged into linear-dialog mainline (v4) |

### CLI

```bash
# Switch method by changing --profile only
python run_unified.py --builtin nat_add_comm --profile whole_proof
python run_unified.py --builtin nat_add_comm --profile whole_proof_repair
python run_unified.py --builtin nat_add_comm --profile dsp
python run_unified.py --builtin nat_add_comm --profile reprover
python run_unified.py --builtin nat_add_comm --profile leandojo
python run_unified.py --builtin nat_add_comm --profile heterogeneous

# Benchmark eval supports --profile too
python run_eval.py --benchmark minif2f --provider anthropic --profile reprover

# YAML-defined profiles
python run_unified.py --profile-yaml my_method.yaml --profile my_method --builtin nat_add_comm

# Opt-in to experimental search presets
python -c "from prover.unified import enable_experimental_search_presets; enable_experimental_search_presets()"
python run_unified.py --builtin nat_add_comm --profile mcts
```

### Test coverage

```bash
$ python -m pytest tests/ -q
1001 passed, 1 skipped in 7.84s
```

See [`REFACTOR_REPORT.md`](REFACTOR_REPORT.md) for the full migration analysis.

---

## Project Structure

```
ai4math/                        264 source files · 56,000+ lines
├── engine/                     ① Verification OS
│   ├── lane/                      Lane runtime: state machine, policy, recovery, compression
│   ├── async_lean_pool.py         Async REPL connection pool
│   ├── async_verification_scheduler.py
│   └── broadcast.py               Cross-agent real-time communication
├── knowledge/                  ② Living Knowledge System
│   ├── store.py                   SQLite 4-layer pyramid
│   ├── reader.py / writer.py      Read/write pipeline
│   └── evolver.py                 Decay, GC, revive
├── prover/                     Proof orchestration
│   ├── pipeline/                  State-machine-driven proof pipeline
│   ├── verifier/                  Lean checker, sorry detector, integrity
│   ├── repair/                    Error diagnosis + auto-repair
│   ├── premise/                   BM25 + embedding hybrid retrieval
│   ├── decompose/                 Goal decomposition
│   └── codegen/                   Tactic generation, scaffold, import resolver
├── agent/                      ④ Agent Layer
│   ├── brain/                     LLM providers (Claude, mock)
│   ├── runtime/                   Sub-agent pool, result fusion, mailbox
│   ├── strategy/                  Direction planner, meta-controller, reflection
│   └── tools/                     CAS bridge, premise search, lean automation
├── common/                     Shared types
├── benchmarks/                 7 benchmark loaders + metrics
├── data/                       6,826 problems (miniF2F, Putnam, ProofNet, FATE, FormalMATH)
├── tests/                      797 passing tests
├── plugins/                    Domain strategy plugins (algebra, number theory, analysis)
├── docker/                     Lean4+Mathlib Docker setup
├── config/default.yaml         Full configuration schema
├── run_single_lane.py          Single-problem interactive debugger (recommended)
├── run_eval.py                 Batch evaluation entry point
└── eval.sh                     One-command evaluation script
```

### Key entry points

| File | Purpose |
|------|---------|
| `run_single_lane.py` | **Recommended** — single problem, full pipeline trace |
| `eval.sh` | One-command benchmark evaluation |
| `engine/lane/integration.py` | `LaneProofRunner` — main async proof loop |
| `prover/pipeline/proof_pipeline.py` | `ProofPipeline` — sync state-machine proof pipeline |
| `prover/assembly.py` | Full system assembler |

---

## Configuration

All settings are in `config/default.yaml`. Key options:

```yaml
agent:
  brain:
    provider: "anthropic"              # LLM provider
    model: "claude-sonnet-4-20250514"  # Model name
    extended_thinking: true            # Enable Claude extended thinking
  strategy:
    default: "adaptive"                # light → medium → heavy auto-escalation

prover:
  pipeline:
    samples_per_round: 8              # Parallel proof candidates per round
    max_rounds: 4                     # Max rounds before giving up
    max_samples: 128                  # Total sample budget
  verifier:
    mode: "docker"                    # "docker" or "local"
    timeout_seconds: 300
```

Override via environment variables:

```bash
MODEL=claude-opus-4-6 MAX_SAMPLES=64 bash eval.sh --real --benchmark builtin
```

---

## Testing

```bash
# Run all tests
PYTHONPATH=. python -m pytest tests/ -q

# Run specific test suites
PYTHONPATH=. python -m pytest tests/test_lane.py -v               # Lane runtime
PYTHONPATH=. python -m pytest tests/test_all_fixes_v2.py -v       # All recent fixes
PYTHONPATH=. python -m pytest tests/test_prover/ -v               # Prover layer

# Smoke test (verifies all imports and basic wiring)
python scripts/smoke_test.py
```

---

## Docker Deployment

For production evaluation with real Lean 4 verification:

```bash
# 1. Build Lean4+Mathlib image (first time: ~30 min)
cd docker && docker compose build

# 2. Start Lean REPL daemon
docker compose up -d lean

# 3. Run evaluation with real verification
docker compose run --rm agent \
  python run_eval.py \
    --benchmark builtin \
    --provider anthropic \
    --lean-mode real

# 4. One-command full pipeline
docker compose run --rm agent bash eval.sh --real --lean
```

---

## Roadmap

- [ ] Full miniF2F/PutnamBench pass@k benchmarks with real Lean compilation
- [ ] Dense embedding retrieval (replace n-gram fallback with sentence-transformers)
- [ ] World model training using collected proof trajectories
- [ ] Multi-backend support for Coq and Isabelle via `Transport(ABC)`
- [ ] Distributed agent pool across multiple machines
- [ ] Web UI for interactive proof exploration

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository and create a feature branch
2. **Write tests** for new functionality
3. **Run** `PYTHONPATH=. python -m pytest tests/ -q` to verify no regressions
4. **Submit** a pull request with a clear description

Areas where help is especially welcome: Lean 4 tactic integration, new benchmark loaders, dense embedding retrieval, and multi-backend support.

---

## Citation

```bibtex
@software{ai4math2026,
  title   = {AI4Math: An Agent Operating System for Formal Theorem Proving},
  year    = {2026},
  url     = {https://github.com/ai4math/ai4math}
}
```

---

## Acknowledgments

AI4Math builds upon and is inspired by:

- [Lean 4](https://lean-lang.org) and [Mathlib](https://leanprover-community.github.io/) — the formal verification foundation
- [miniF2F](https://github.com/openai/miniF2F), [PutnamBench](https://github.com/trishullab/PutnamBench), [ProofNet](https://github.com/zhangir-azerbayev/ProofNet), [FATE](https://github.com/fate-ubw), [FormalMATH](https://github.com/FormalMATH) — benchmark datasets
- [DeepSeek-Prover](https://github.com/deepseek-ai/DeepSeek-Prover-V2), [Goedel-Prover](https://github.com/Goedel-LM/Goedel-Prover), [Kimina-Prover](https://github.com/MoonshotAI/Kimina) — pioneering proof generation work

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center"><sub>📖 <a href="README_zh.md">中文版</a> · <a href="https://ai4math.github.io/ai4math">Interactive Demo</a></sub></div>
