<p align="center">
  <img src="https://img.shields.io/badge/Lean-4.28.0-blue" alt="Lean 4" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab" alt="Python" />
  <img src="https://img.shields.io/badge/tests-822-green" alt="Tests" />
  <img src="https://img.shields.io/badge/problems-1631-orange" alt="Problems" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
</p>

<p align="center">
  <b>🇬🇧 English</b> ·
  <a href="README_zh.md">🇨🇳 中文</a> ·
  <a href="TUTORIAL_CN.md">📖 Tutorial (CN)</a> ·
  <a href="docs/ARCHITECTURE.md">🏗️ Architecture</a> ·
  <a href="docs/CHANGELOG.md">📝 Changelog</a>
</p>

# AI4Math

A Lean 4 theorem-proving infrastructure built around three things, with three
reserved interfaces on top.

## What this project is

The **core** is exactly three things:

1. **Unified reasoning** — every prover method (DeepSeek-Prover style whole-proof,
   ReProver-style RAG, LeanDojo step-level, MCTS / best-first / beam, DSP,
   Pantograph drafting, conjecture-driven) runs through one `AgentLoop` driven
   by a single `--profile` switch. 14 profiles ship; adding a new method = adding
   one `Profile` entry.

2. **Verification infrastructure** — async Lean 4 REPL pool (`AsyncLeanPool`,
   ~50 ms incremental verification via `env_id` fork), three-level pre-filter
   (L0 syntax / L1 REPL / L2 full compile), four community backends
   (Local / Kimina / Pantograph / LooKeng), structured `AgentFeedback` (~100 bits)
   instead of a 1-bit pass/fail loop.

3. **RL infra interface** — `sampler/` exposes `ProofEnv` + `BaseSampler` to
   the major RL frameworks (verl / slime / TRL / vLLM-driven trainers). Every
   trajectory drops as a standard `dialog.json`; SFT export is one command.

On top of that, the project reserves **three feature interfaces** for extension:

- **A. Knowledge base** (`knowledge/`) — four-layer SQLite pyramid: raw
  trajectory → tactic effectiveness → strategy patterns → concept graph.
  Layer 0/1 wire into the main path; the upper layers' schema is in but
  the deposit path is left for downstream callers.

- **B. World model** (`engine/world_model.py`) — a `tactic-success-prior`
  predictor sits in front of `tactic_apply`. Default impl is a sklearn
  LogisticRegression trained from past successful proofs. The interface is
  one method; replace `MockWorldModel` with anything (transformer, GNN, ...)
  by passing `world_model=` to `UnifiedProofRunner`.

- **C. Multi-agent broadcast** (`engine/broadcast.py` + `BroadcastTool`) —
  cross-direction discovery sharing for parallel rollouts. v13 wired this
  to `heterogeneous` profile: each sub-profile gets the BROADCAST tool and
  shares one bus, so a discovery in any direction reaches all others. The
  longer-term aim: a global bus across problems → a "community of
  mathematicians" where one solver's lemma helps another.

**Anything outside these six items has been removed in v13.**
The repo focuses; if it's there, it's load-bearing.

## Quick start

```bash
git clone https://github.com/ai4math/ai4math.git
cd ai4math
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."

# Prove one theorem; switch method by changing one flag
python run_unified.py --builtin nat_add_comm --profile whole_proof_repair
python run_unified.py --builtin nat_add_comm --profile reprover
python run_unified.py --builtin nat_add_comm --profile heterogeneous
python run_unified.py --builtin nat_add_comm --profile mcts

# Batch evaluation
bash eval.sh --benchmark builtin                       # mock, no Lean
bash eval.sh --real --benchmark minif2f --samples 8    # full
```

Every run produces `results/traces/<problem_id>/dialog.json` — the single
canonical output. Open one file, see everything (system prompt, tools, every
turn, final result, search tree if applicable).

## Community Lean backends

| Backend | Profile | Setup |
|---|---|---|
| Kimina Lean Server | `kimina_batch` | `docker run projectnumina/kimina-lean-server:2.0.0` |
| Pantograph | `pantograph_dsp` | `pip install pypantograph` |
| LooKeng | `lookeng_lemma` | clone seed-prover, `pip install -e ".[lookeng]"` |

`dialog.json` records `meta.backends.<name>.is_fallback` so you can verify the
backend actually engaged (vs silently degrading to local).

## Adding a new method

```python
# 1. prover/unified/profiles.py
PRESETS["my_method"] = Profile(
    name="my_method",
    tools=[ToolKit.LEAN_VERIFY, ToolKit.PREMISE_SEARCH],
    max_turns=8,
    framing="my_framing",
    temperature=0.5,
)

# 2. prover/unified/system_prompts.py
FRAMING_PROMPTS["my_framing"] = "You are a Lean 4 prover. Output..."

# 3. python run_unified.py --profile my_method
```

No changes to `runner.py`, `agent_loop.py`, or `tools/`.

## Layout

```
engine/      Lean 4 REPL pool, three-level verification, error intelligence
             + reserved B (world_model.py) + reserved C (broadcast.py)
agent/       AgentLoop, tools, brain (LLM), persistence (dialog.json)
prover/      Profile-driven runner; conjecture/decompose/premise/verifier
sampler/     RL interfaces (ProofEnv, verl/slime/TRL adapters, tree rollouts)
knowledge/   reserved A — four-layer SQLite pyramid
benchmarks/  miniF2F + PutnamBench + ProofNet + FATE + FormalMATH loaders
config/      profile YAML templates (one per active method)
data/        benchmark problems
docs/        ARCHITECTURE.md + CHANGELOG.md + dialog.json schema
tests/       760 mock tests; live integration tests run when ANTHROPIC_API_KEY set
```

Start reading at [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What this project does *not* do

Honesty over marketing:

- **No open-ended math discovery.** All 1631 theorems come from the 7 built-in
  benchmarks. The `conjecture_driven` profile lets the LLM invent *auxiliary
  lemmas* on the path to a *given* theorem; it does not invent the theorem
  itself.
- **No published SOTA numbers.** AI4Math's pass@k on these benchmarks is
  "evaluation in progress" — we are infrastructure, not a leaderboard
  contender.
- **The RL flywheel is 3/4 closed.** Stages 1–3 (eval → collect → world-model)
  run end-to-end. Stage 4 hands a ready-to-train JSONL to your trainer of
  choice (verl / slime / TRL).
- **The "world model" is sklearn LogisticRegression.** Trainable, works as a
  tactic-success-prior — not the full state-dynamics network the architecture
  diagram suggests. Replace it with one constructor arg.
- **The cross-problem community broadcast is not yet built.** v13 wired
  the *intra-run* heterogeneous broadcast (4 sub-profiles share a bus). The
  global persistent bus + auth layer (the "community of mathematicians" goal)
  is reserved interface C's next step.

## v14 — 备胎复活: 4 项关键模块回归并接通主路径

v13 砍掉死代码后,我们重新审视了初版仓库里所有"未接通备胎",筛选出 4 项
**对应真实问题**且**契合三件核心 + 三个预留接口**的模块,回归到 v14 主路径。
每一项都做到「调用方真在用」,不再是基础设施摆设:

- **项① · `engine/summary_compressor.py`**: Lean 错误 + AgentFeedback + 跨方向广播
  消息的三个 LLM-readable 压缩器。压缩比 ~9× (实测 4520 → 396 chars)。接通到
  `LoopConfig.compress_tool_results=True` (默认开) 和 `BroadcastTool` 收发。
- **项② · `engine/policy/`**: PolicyEngine + RecoveryRegistry +
  ProofTaskStateMachine 三件套 + 5 条内置规则。把 agent_loop 里硬编码的"何时
  升级 / 何时切角色 / 何时放弃"挪到声明式 PolicyRule。接通到
  `AgentLoop(policy_engine=)` + 每轮失败后自动评估。
- **项③ · `prover/lemma_bank/`**: 跨问题/跨会话的 SQLite + BM25 引理库。直接
  对应**预留接口 A 知识库**的 lemma 维度。接通到 `LemmaBankTool` BM25 fallback
  路径 + `ConjectureProposeTool` 后置写入。
- **项④ · `prover/plugins/` + `plugins/strategies/{algebra,analysis,number-theory}/`**:
  YAML-driven 领域插件系统。`framing` 是 profile-level (整套推理风格),
  `plugin` 是 problem-level (按定理领域注入额外引理 + few-shot + 战略提示)。
  接通到 `_build_initial_message` 在 problem 入口拿得分最高的插件,把 few-shot
  / extra_premises / strategic_hint 注入首条 user message。

**未做** (按数据驱动决策原则,等 Sprint 1 真实 pass@k 出来再决定):
- ⑤ 长上下文压缩(基于 Anthropic prompt caching 重写)
- ⑥ DirectionPlanner(只在 heterogeneous 真低于 best-of-4 时回归)
- ⑦ 规则化修复(只在某类错误占失败 >40% 时回归)

完整的迭代计划见 [`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md) (5 名
工程师 × 4 Sprint 的详细分工)。

测试: `760 → 786 passed (+26 v14 smoke, 零回归)`。

## v13 — focus

Removed ~6,500 lines of dead Python + ~6,300 lines of redundant HTML/SVG/docs.
Fixed one more latent bug of the same async-call-from-sync pattern that v10/v11/v12
already squashed 8 of (`GoalDecomposer.decompose` was still sync). Wired the
`heterogeneous` broadcast bus that had been load-bearing-in-name-only since v6.
Slimmed `common/roles.py` from 11 roles to the 2 that have callers.

The bar after v13 is: every line in this repo is on the path to one of the
**three core** items or one of the **three reserved interfaces**. Anything
that isn't, leaves.

See [`docs/CHANGELOG.md`](docs/CHANGELOG.md) for the full diff and migration
notes.

## Citation

```bibtex
@software{ai4math2026,
  title = {AI4Math: An Agent Operating System for Formal Theorem Proving},
  year  = {2026},
  url   = {https://github.com/ai4math/ai4math}
}
```

## License

[MIT](LICENSE)
