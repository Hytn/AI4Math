<div align="center">

# AI4Math

### An Agent Operating System for Formal Theorem Proving

[English](#overview) ·  [中文](#概览) · [Interactive Demo ↗](https://ai4math.github.io/ai4math) · [Tutorial (中文) ↗](TUTORIAL_CN.md)

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
- [中文版](#概览)

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

<br>

<div align="center">

# 概览

**[English](#overview)** · **中文**

</div>

## AI4Math 是什么

AI4Math 是一个**智能体操作系统**，让数百个异构 AI 数学家像研究院一样协同工作，自动发现 Lean 4 形式化证明。

它不是一个"更好的证明生成器"——它是证明生成器运行的**基础设施平台**。任何 LLM 都可以作为证明引擎插入这个平台。

> **在线演示 →** 访问 [交互式前端](https://ai4math.github.io/ai4math) 查看完整的 Putnam 竞赛题解题过程，包含每个内部组件的可视化。

## 为什么选择 AI4Math？

| | 现有范式 (DeepSeek/Goedel/Kimina) | AI4Math |
|---|---|---|
| **反馈** | 每次尝试仅 1 bit (pass/fail) | ~100 bits 结构化错误诊断 |
| **通信** | 方向之间零通信 | 所有智能体实时广播共享 |
| **学习** | 失败经验完全丢失 | 每次失败沉淀可复用知识 |
| **验证** | Lean 全编译 (2–12s) | 三级：语法 ~1μs → REPL ~50ms → 全编译 ~3s |

## 快速开始

### 环境准备

```bash
git clone https://github.com/ai4math/ai4math.git && cd ai4math
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

Lean 4 环境安装请参考 [傻瓜式教程 (TUTORIAL_CN.md)](TUTORIAL_CN.md)，内含从零开始的完整步骤。

### 单题证明

```bash
# 内置题目，完整管线追踪：
python run_single_lane.py --builtin nat_add_comm --provider anthropic

# 自定义定理：
python run_single_lane.py --theorem "theorem t (n : Nat) : n + 0 = n" --provider anthropic
```

### 批量评测

```bash
bash eval.sh --real --benchmark builtin              # 5 题快速跑分 (~2 分钟)
bash eval.sh --real --quick                           # 每个 benchmark 取 10 题
bash eval.sh --real --benchmark minif2f --samples 32  # miniF2F 全量
```

### Docker 部署（推荐用于真实 Lean 4 验证）

```bash
cd docker && docker compose build && docker compose up -d
docker compose run --rm agent bash eval.sh --real --lean
```

## 内置基准数据集（6,826 道题）

| 数据集 | 题数 | 难度 | 说明 |
|--------|------|------|------|
| **builtin** | 5 | 入门 | 冒烟测试，推荐首次使用 |
| **miniF2F** | 488 | AMC → IMO | 领域内最广泛使用的基准 |
| **PutnamBench** | 672 | 大学竞赛 | 1962–2024 Putnam 竞赛题 |
| **ProofNet** | 360 | 本科数学 | 分析、代数、拓扑核心课程 |
| **FATE-M/H/X** | 350 | 本科→博士 | 抽象代数全难度覆盖 |
| **FormalMATH** | 5,560 | 混合 | 多领域多难度 |

## 四层架构

> **交互版 →** 访问 [架构可视化](https://ai4math.github.io/ai4math#pillars) 和 [数据流全景](https://ai4math.github.io/ai4math#flowDiagram) 查看动画版本。

| 层级 | 模块 | 定位 |
|------|------|------|
| **④ 数学家社会** | `agent/` | 11 种角色、并行探索、实时广播 |
| **③ 世界模型** | `engine/world_model.py` | 内化 Lean4 状态动力学，预判策略效果 |
| **② 活知识系统** | `knowledge/` | 四层金字塔、衰减遗忘、跨智能体共享 |
| **① 验证 OS** | `engine/` | REPL 池、三级验证、弹性伸缩、增量编译 |

**飞轮：** ④ 探索 → ① 验证 → ② 沉淀知识 → ③ 训练模型 → ④ 注入知识 → 加速探索

## 常见问题

**Q: 和 DeepSeek-Prover 的根本区别？**
它们是"更强的证明生成器"，AI4Math 是"让证明生成器在其中运行的操作系统"。两者正交互补。

**Q: 能支持 Coq / Isabelle 吗？**
REPL 交互通过 `Transport(ABC)` 抽象，知识系统和智能体层不含 Lean4 特定代码。

**Q: Green Contract 是什么？**
验证分级合约。每个证明结果不再是 pass/fail 的 1 bit，而是分 6 级：NONE → SYNTAX_CLEAN → TACTIC_VALID → GOALS_CLOSED → FULL_COMPILE → SORRY_FREE。

**Q: 断点续证怎么用？**
`ProofPipeline` 每轮结束后自动保存 checkpoint。下次传入 `resume=True` 即可从上次中断处恢复。

---

<div align="center">
<sub>MIT License · 264 files · 56K+ lines · 797 tests · 7 benchmarks · 6,826 problems</sub>
</div>
