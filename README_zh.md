<p align="center">
  <img src="https://img.shields.io/badge/Lean-4.28.0-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJMMyAyMGgxOEwxMiAyeiIgZmlsbD0id2hpdGUiLz48L3N2Zz4=" alt="Lean 4" />
  <img src="https://img.shields.io/badge/Python-3.12+-3776ab?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/tests-938-green?logo=pytest&logoColor=white" alt="Tests" />
  <img src="https://img.shields.io/badge/problems-1%2C631-orange" alt="Problems" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
</p>

<p align="center">
  <a href="README.md">🇬🇧 English</a> ·
  <b>🇨🇳 中文</b> ·
  <a href="https://ai4math.github.io/ai4math">🌐 项目主页</a> ·
  <a href="TUTORIAL_CN.md">📖 零基础教程</a>
</p>

---

# AI4Math

**形式化定理证明的智能体操作系统（Lean 4）。**

AI4Math 不是又一个证明生成器——它是一套完整的基础设施，让数百个异构 AI 智能体协同攻克数学定理。它提供毫秒级验证、自演化知识系统和多智能体协调，形成飞轮效应：证明得越多，系统越聪明。

> **一句话理解** — 当前 SOTA（DeepSeek-Prover、Kimina、Goedel）生成完整证明，获得 1 bit 反馈（通过/失败），独立重试。AI4Math 给每个智能体提供 ~100 bits 的结构化反馈，在智能体间实时广播发现，并积累可复用的数学知识——将蛮力采样变为有引导的协作搜索。

---

## 目录

- [为什么选择 AI4Math？](#为什么选择-ai4math)
- [架构概览](#架构概览)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [基准数据集](#基准数据集)
- [评测](#评测)
- [Docker 部署（Lean 4 验证）](#docker-部署)
- [项目结构](#项目结构)
- [SOTA 对比](#sota-对比)
- [路线图](#路线图)
- [参与贡献](#参与贡献)
- [引用](#引用)
- [致谢](#致谢)
- [许可证](#许可证)

---

## 为什么选择 AI4Math？

<table>
<tr>
<th width="50%">当前范式</th>
<th width="50%">AI4Math 范式</th>
</tr>
<tr>
<td>

- LLM 生成完整证明
- Lean 4 编译（2–12 秒）→ **1 bit**（通过/失败）
- N 次独立重试，方向之间零通信
- 失败经验完全丢失

</td>
<td>

- N 个异构智能体并行探索
- 三级验证 → **~100 bits** 结构化反馈
- 发现在智能体间实时广播
- 知识自动沉淀，跨题目复用

</td>
</tr>
</table>

### 核心特性

| 特性 | 说明 |
|------|------|
| **三级验证** | L0 语法预过滤（~1 μs）→ L1 REPL 快验（~50 ms）→ L2 全编译（~3 s）。每层返回结构化诊断信息。 |
| **Lane 运行时** | 每个证明任务在显式状态机中运行，具备类型化事件、可组合策略规则、自动恢复方案和断点续证支持。 |
| **自演化知识** | 四层知识金字塔（原始轨迹 → 战术 → 策略模式 → 概念图谱），具备写入/读取/衰减生命周期。第 N 题的知识加速第 N+1 题。 |
| **多智能体协调** | 方向规划器每轮生成 3–4 个异构智能体（不同角色、模型、温度），通过类型化事件总线实时广播发现。 |
| **Green Contract** | 验证结果分为 6 级（NONE → SYNTAX_CLEAN → TACTIC_VALID → GOALS_CLOSED → FULL_COMPILE → SORRY_FREE），支持精细策略决策。 |
| **Sorry/Axiom 完整性检查** | 深度检测拒绝包含 `sorry`、自定义公理、`unsafeCoerce` 或 `sorry` 重定义的证明——即使 Lean 编译通过。 |
| **断点续证** | 长时间运行的证明自动保存状态。中断后从上次检查点恢复，保留所有已积累的知识。 |
| **插件系统** | 领域特定策略（代数、数论、分析），包含定制的前提、few-shot 示例和战术建议。 |

---

## 架构概览

AI4Math 分为四层，形成飞轮：

```
┌──────────────────────────────────────────────────────────┐
│  第三层 │ 数学家社会 — 异构智能体群落的自组织协作          │
├──────────────────────────────────────────────────────────┤
│  第二层 │ 世界模型 — 证明状态动力学预测器                 │
├──────────────────────────────────────────────────────────┤
│  第一层 │ 活知识系统 — 自演化的数学记忆                   │
├──────────────────────────────────────────────────────────┤
│  第零层 │ 验证操作系统 — 毫秒级弹性验证引擎              │
├──────────────────────────────────────────────────────────┤
│    ⓪   │ Lean 4 REPL — 真值形式化验证器                 │
└──────────────────────────────────────────────────────────┘
```

**→ 在[项目主页](https://ai4math.github.io/ai4math)查看交互式架构图和实时演示。**

---

## v3 大一统主管线 (Profile 驱动)

> **一句话：除 MCTS 外，所有主流定理证明方法都已通过 `prover.unified` 主管线实现。切换方法 = 切 `--profile` 参数，无需改代码。**

### 设计本质

现代基于 LLM 的定理证明方法之间的差异，可以完全收敛到三个变量：
1. **`max_turns`** — 一次会话允许 LLM 调用多少轮
2. **`tools`** — LLM 能调用的工具集
3. **`system_prompt` + 初始 user message** — 指引 LLM 进入哪种工作范式

这三者构成 `Profile` dataclass，由 `UnifiedProofRunner` 统一执行。同一份代码、同一份 dialog.json schema，跑出 6 个不同的算法。

### Profile 全表 (active presets)

| Profile | 对应方法 | `max_turns` | `tools` | `framing` | 状态 |
|---|---|---|---|---|---|
| `whole_proof` | DeepSeek-Prover · Kimina · Goedel | 1 | `[]` | `whole_proof` | ✅ 完整 |
| `whole_proof_repair` | 项目原默认主路径 (compile-and-fix) | 6 | `[lean_verify]` | `whole_proof_repair` | ✅ 完整 |
| `dsp` | Draft-Sketch-Prove | 10 | `[decompose, premise_search, lean_verify]` | `dsp` | ✅ 完整 |
| `reprover` | ReProver (RAG + step-level) | 30 | `[premise_search, tactic_apply, goal_inspect]` | `step_level_with_retrieval` | ✅ 完整 |
| `leandojo` | LeanDojo (纯 step-level) | 50 | `[tactic_apply, goal_inspect, lean_auto]` | `step_level_pure` | ✅ 完整 |
| `heterogeneous` | AI4Math 异构并行 (项目卖点) | 4 | 4 路 sub-profile + `broadcast` | `whole_proof_repair` | ✅ 完整 |

### 各算法对照如何被参数化复现

#### 1. DeepSeek-Prover / Kimina / Goedel — `whole_proof`

**论文本质**：单轮 LLM 一次性输出完整 Lean 证明，pass@k 由外层独立采样 K 次实现。

**本框架的参数映射**：
```python
Profile(
    name="whole_proof",
    tools=[],                           # 关掉所有工具 → 强制单轮整证
    max_turns=1,                        # 一轮就结束
    framing="whole_proof",              # 系统提示: "Output exactly one ```lean block. Do NOT call any tools."
    observation=ObservationPolicy(
        inject_few_shot=True,           # ✅ 注入 5 个 Mathlib 示例 (DeepSeek/Goedel 训练数据风格)
        inject_premises_in_prompt=True, # ✅ 检索 top-N Mathlib 引理预注入
        auto_inject_lean_compile=True,  # ✅ 即使 LLM 没调 verify, runtime 后置自动跑 Lean4 编译
    ),
    stop=StopCondition(on_text_only=True),  # 一段文字结束就终止
)
```

**实现完整度核查**：
- ✅ 单轮整证：`max_turns=1` + `stop_on_text_only=True`
- ✅ Pass@k：`run_eval.py --max-samples K` 在外层独立采 K 次（满足 i.i.d.）
- ✅ Few-shot：`common/prompt_builder.py::FEW_SHOT_EXAMPLES` 含 5 个有代表性的 Mathlib 证明，自动注入初始 user message
- ✅ Premise 注入：`PremiseSelector.retrieve()` 默认 hybrid mode (TF-IDF + BM25)，注入 top-10 引理
- ✅ 终态验证：`auto_inject_lean_compile=True` 走 LeanPool 全编译
- ⚠️ 与原版差异：DeepSeek-Prover-V2 用了显式的 `<think>` chain-of-thought 训练数据；本框架默认不强制 CoT 但允许通过 `temperature` 和 `model` 调节（推理模型如 `claude-opus-4-7` 会自然 CoT）

#### 2. Repair Loop — `whole_proof_repair`

**论文本质**：生成 → 编译 → 看错误 → 重生成。N 轮后停止。

**本框架的参数映射**：
```python
Profile(
    name="whole_proof_repair",
    tools=[ToolKit.LEAN_VERIFY],        # 唯一工具: 整证编译验证
    max_turns=6,                        # 5 轮修复机会
    framing="whole_proof_repair",       # 系统提示: "Output proof, see errors, fix, repeat. One ```lean block per turn."
    observation=ObservationPolicy(
        compress_errors_budget=1200,    # 错误用 lane.summary_compressor 压到 1200 字
        auto_inject_lean_compile=True,  # 兜底: 若 LLM 不主动调 verify, runtime 自动跑
    ),
)
```

**实现完整度核查**：
- ✅ 多轮闭环：`max_turns=6`，每轮 LLM 看到上轮错误反馈
- ✅ 错误结构化：`engine/lane/error_classifier.py` + `summary_compressor.py` 把 Lean stderr 压缩为可读分类
- ✅ Lean4 真实验证：`LeanVerifyTool` 直接调 LeanPool 完整编译
- ✅ 增量上下文：dialog 历史完整保留，LLM 看到所有过去尝试

#### 3. Draft-Sketch-Prove (DSP) — `dsp`

**论文本质**：分阶段——先非形式化 sketch → 拆子目标 → 找引理 → 形式化 → 修复。

**本框架的参数映射**：
```python
Profile(
    name="dsp",
    tools=[ToolKit.DECOMPOSE,           # 把目标拆成子目标 (prover.decompose.GoalDecomposer)
           ToolKit.PREMISE_SEARCH,      # 给每个子目标查 Mathlib 引理
           ToolKit.LEAN_VERIFY],        # 形式化后的整证验证
    max_turns=10,
    framing="dsp",                       # 系统提示明确五阶段 A-E (Sketch → Decompose → Premises → Formalize → Repair)
)
```

**实现完整度核查**：
- ✅ Sketch (Phase A)：通过 system prompt 明确指示 LLM 在第一轮输出非形式化 sketch（注释块）
- ✅ Decompose (Phase B)：`DecomposeSubgoalTool` 调用 `prover/decompose/goal_decomposer.py`
- ✅ Premise (Phase C)：`PremiseSearchTool` 复用步级范式同款检索器
- ✅ Formalize + Repair (Phase D-E)：`LeanVerifyTool` + 多轮反馈
- ⚠️ 与原版差异：原 DSP 在 LLM 阶段用了 informal-to-formal 的额外训练数据；本框架仅靠 system prompt 引导，效果取决于底层模型的形式化能力

#### 4. ReProver (RAG + step-level) — `reprover`

**论文本质**：每步 (1) 查 Mathlib 引理，(2) 选一个 tactic，(3) REPL apply，(4) 看新 goal。重复至证明完成。

**本框架的参数映射**：
```python
Profile(
    name="reprover",
    tools=[ToolKit.PREMISE_SEARCH,      # 步级按需查引理 (不在初始 prompt 注入)
           ToolKit.TACTIC_APPLY,        # 单步 tactic apply via REPL
           ToolKit.GOAL_INSPECT],       # 显式查看当前 goal
    max_turns=30,
    framing="step_level_with_retrieval",  # 系统提示: "ONE TACTIC PER TURN, retrieve premises as needed"
    observation=ObservationPolicy(
        auto_inject_goal_state=True,    # tactic_apply 工具结果天然含新 goal
        auto_inject_lean_compile=False, # 步级不需要全编译
        inject_premises_in_prompt=False, # ⚠️ 关键: ReProver 是按需检索, 不预注入
        inject_few_shot=False,          # 步级不需要整证示例
    ),
)
```

**实现完整度核查**：
- ✅ 步级 tactic apply：`TacticApplyTool` 通过 LeanPool REPL 实际执行单条 tactic 并返回新 goal state
- ✅ 按需引理检索：`PremiseSearchTool` 由 LLM 在需要时主动调用，结果含 top-K 相关引理
- ✅ Goal state 反馈：tactic_apply 的返回值天然包含 `remaining_goals`、`is_proof_complete`，等价于 ReProver 的 observation
- ✅ 长 horizon：`max_turns=30` 足以覆盖大多数 IMO/Putnam 级证明
- ⚠️ 与原版差异：ReProver 论文用了**特定训练**的 retriever (BM25+ColBERT 在 LeanDojo 数据上微调)；本框架默认用 TF-IDF + 项目自带 hybrid retriever。结构等价，retriever 质量差异由用户替换 retriever 控制

#### 5. LeanDojo (纯 step-level) — `leandojo`

**论文本质**：与 ReProver 类似但不依赖 retriever；只有 tactic apply + goal inspect + 自动化 hammer。

**本框架的参数映射**：
```python
Profile(
    name="leandojo",
    tools=[ToolKit.TACTIC_APPLY,
           ToolKit.GOAL_INSPECT,
           ToolKit.LEAN_AUTO],          # exact?/aesop/polyrith hammer
    max_turns=50,
    framing="step_level_pure",
    observation=ObservationPolicy(
        auto_inject_goal_state=True,
        inject_premises_in_prompt=False,
        inject_few_shot=False,
    ),
)
```

**实现完整度核查**：
- ✅ 纯步级交互：与 reprover 共用 TacticApplyTool，行为完全一致
- ✅ Hammer 集成：`LeanAutoTool` 调用 Mathlib 内置自动化 (`exact?`, `apply?`, `aesop`, `polyrith`)
- ✅ 长 horizon：`max_turns=50`
- ⚠️ 与原版差异：LeanDojo 论文同时报告了 best-first search 配合 LLM expansion 的结果。这一搜索部分属于 MCTS 家族，**已搁置在 `EXPERIMENTAL_PRESETS["best_first"]`**，需 `enable_experimental_search_presets()` 显式启用

#### 6. AI4Math 异构并行 — `heterogeneous`

**项目原创**：4 路独立 agent 并行，每路有不同的 (model, temperature, system_prompt, tools)，通过 `BroadcastBus` 共享发现，`ResultFuser` 跨路融合。

**本框架的参数映射**：
```python
Profile(
    name="heterogeneous",
    search=SearchConfig(
        kind="parallel",
        parallel_profiles=[
            "whole_proof",          # 自动化探测路 (低温, 整证)
            "reprover",             # 检索 + 步级路
            "leandojo",             # 纯步级路
            "whole_proof_repair",   # 修复路
        ],
    ),
    tools=[ToolKit.LEAN_VERIFY, ToolKit.BROADCAST],  # broadcast 工具供 sub-agent 互相看到对方
)
```

**实现完整度核查**：
- ✅ 真正并行：`UnifiedProofRunner._run_parallel` 用 `asyncio.gather` 启动 N 个独立 runner
- ✅ 共享广播总线：`BroadcastBus` 实例由父 runner 创建并注入所有 sub-runner
- ✅ 异构性：4 个 sub-profile 在 `tools` / `framing` / `max_turns` 上完全不同
- ✅ Result fusion：`ResultFuser.extract_useful_lemmas()` + cross-fusion repair attempt
- ✅ 与 v2 兼容：assembly.py 的旧 `HeterogeneousEngine(pool=..., ...)` 构造签名通过 `_SyncToAsyncAdapter` 自动支持

### 架构如何成为大一统基座

```
┌───────────────────────────────────────────────────────────────────────┐
│                    UnifiedProofRunner (单一入口)                        │
│                                                                         │
│   prove(problem, profile) ──→  initial prompt (theorem + few-shot      │
│                                  + premises + briefing)                 │
│                              ──→  AgentLoop                             │
│                                    ├── system_prompt = framing[X]       │
│                                    ├── tools = ToolKit → Tool 实例     │
│                                    ├── max_turns = N                    │
│                                    └── 多轮: LLM → tool call → result   │
│                              ──→  auto_verify (兜底)                    │
│                              ──→  dialog.json (统一格式输出)            │
└───────────────────────────────────────────────────────────────────────┘
            ▲                                                ▲
            │                                                │
   ┌────────┴────────┐                                ┌──────┴───────┐
   │   Profile       │                                │ ToolKit      │
   │                 │                                │              │
   │ tools list      │                                │ LEAN_VERIFY  │
   │ max_turns       │                                │ TACTIC_APPLY │
   │ framing         │                                │ GOAL_INSPECT │
   │ temperature     │                                │ PREMISE_SEARCH│
   │ ObservationPolicy                                │ LEAN_AUTO    │
   │ StopCondition   │                                │ DECOMPOSE    │
   │ SearchConfig    │                                │ BROADCAST    │
   └─────────────────┘                                │ ...          │
                                                      └──────────────┘
```

**所有方法走同一条数据通路**：
1. **入口**：`UnifiedProofRunner.run(problem, profile)` — 唯一执行入口
2. **核心循环**：`agent.runtime.AgentLoop` — 唯一主循环 (LLM ↔ tool 反馈)
3. **工具层**：`agent.tools.ToolRegistry` — 唯一工具调度
4. **持久化**：`agent.persistence.dialog_format` — 唯一输出格式 (`dialog.json` schema v2.0)
5. **兼容桥**：`prover.unified.adapters` — `UnifiedResult` ↔ 旧 `ProofAttempt` / `AgentResult`

**新加方法只需改一个文件**：`prover/unified/profiles.py` 加一项 Profile（或写 YAML 后 `register_profile`），不动 runner / loop / tools / pipeline。

### 已知的"实现 vs 论文"差异（坦诚清单）

| Profile | 实现到位的 | 与原版差异 |
|---|---|---|
| `whole_proof` | 整证 + few-shot + premise + 终态验证 | 没有强制 chain-of-thought scaffolding（依赖底层模型自身的 reasoning 能力） |
| `whole_proof_repair` | 完整闭环 | — |
| `dsp` | 五阶段架构 + 工具齐全 | 原 DSP 用 informal-to-formal 训练数据；本框架靠 system prompt 引导，等价于 zero-shot DSP |
| `reprover` | 步级 + 按需检索 + REPL | retriever 默认 TF-IDF/BM25 hybrid，原 ReProver 用了在 LeanDojo 上微调的 ColBERT |
| `leandojo` | 纯步级 + hammer | 原 LeanDojo 还包括 best-first search wrapper（属 MCTS 家族，搁置在 experimental） |
| `heterogeneous` | 4 路并行 + broadcast + fusion | — |
| `mcts` / `beam` / `best_first` | 代码完整，搁置在 `EXPERIMENTAL_PRESETS` | 与 dialog-linear 主管线尚未合流（v4 工作） |

### CLI 用法

```bash
# 切换方法只改 --profile
python run_unified.py --builtin nat_add_comm --profile whole_proof
python run_unified.py --builtin nat_add_comm --profile whole_proof_repair
python run_unified.py --builtin nat_add_comm --profile dsp
python run_unified.py --builtin nat_add_comm --profile reprover
python run_unified.py --builtin nat_add_comm --profile leandojo
python run_unified.py --builtin nat_add_comm --profile heterogeneous

# 批量评测同样支持 --profile
python run_eval.py --benchmark minif2f --provider anthropic --profile reprover

# 自定义 Profile (YAML)
python run_unified.py --profile-yaml my_method.yaml --profile my_method --builtin nat_add_comm

# 启用实验性搜索 (MCTS / beam / best_first)
python -c "from prover.unified import enable_experimental_search_presets; enable_experimental_search_presets()"
python run_unified.py --builtin nat_add_comm --profile mcts
```

### Python API 用法

```python
from prover.unified import UnifiedProofRunner, get_profile

runner = UnifiedProofRunner(
    llm=async_llm,
    lean_pool=pool,
    knowledge_store=knowledge,    # 可选
    retriever=retriever,           # 可选
    broadcast_bus=bus,             # 异构并行时启用
)

# 直接选 preset
result = await runner.run(problem, profile_name="reprover")

# 或 override 字段
from dataclasses import replace
prof = replace(get_profile("reprover"), max_turns=20, temperature=0.3)
result = await runner.run(problem, profile=prof)

print(result.success, result.proof_code)
result.save_unified("results/traces/my_run", problem_id=problem.problem_id)
```

### 测试覆盖

```bash
$ python -m pytest tests/ -q
1001 passed, 1 skipped in 7.84s
```

包含 16 个专门的 unified-pipeline 测试用例验证：
- 6 个 active preset 的字段约束
- 3 个 experimental preset 的 opt-in 隔离
- HeterogeneousEngine v3 的 legacy 构造兼容
- ProofLoop shim 的语义保持
- adapters 的 Dialog ↔ ProofAttempt 数据无损
- 端到端 mock LLM 跑通 dialog.json 产出

详细迁移说明见 [REFACTOR_REPORT.md](REFACTOR_REPORT.md)。

---

## 环境要求

| 依赖 | 版本 | 用途 |
|------|------|------|
| **Python** | ≥ 3.10 | 核心运行时 |
| **Anthropic API 密钥** | — | LLM 提供者（Claude） |
| **Lean 4** *（可选）* | 4.28.0 | 真实证明验证 |
| **Docker** *（可选）* | ≥ 24.0 | 容器化 Lean 4 环境 |

### 安装 Lean 4（面向数学家的傻瓜教程）

如果你需要真实的证明验证（而不仅是 LLM 生成候选），需要安装 Lean 4 + Mathlib。

> 💡 **完全不懂编程？** 请查看我们的[零基础教程](TUTORIAL_CN.md)，从安装 Python 开始手把手指导。

<details>
<summary><b>macOS</b></summary>

```bash
# 1. 安装 Homebrew（如果没有的话）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装 elan（Lean 版本管理器）和 Lean 4
curl https://elan.lean-lang.org/install.sh -sSf | sh
source ~/.profile  # 或者重启终端

# 3. 验证安装
lean --version  # 应该显示 leanprover/lean4:v4.x.x

# 4. 安装 Git LFS（Mathlib 需要）
brew install git-lfs
git lfs install
```

</details>

<details>
<summary><b>Ubuntu / Debian Linux</b></summary>

```bash
# 1. 安装前置依赖
sudo apt-get update
sudo apt-get install -y curl git

# 2. 安装 elan 和 Lean 4
curl https://elan.lean-lang.org/install.sh -sSf | sh
source ~/.profile

# 3. 验证
lean --version

# 4. 安装 Git LFS
sudo apt-get install -y git-lfs
git lfs install
```

</details>

<details>
<summary><b>Windows（推荐 WSL2）</b></summary>

```powershell
# 1. 安装 WSL2
wsl --install

# 2. 在 WSL2（Ubuntu）中，按照上面的 Linux 步骤操作
```

</details>

<details>
<summary><b>验证 Lean 4 + Mathlib 安装成功</b></summary>

```bash
# 创建测试项目
mkdir lean-test && cd lean-test
lake +leanprover/lean4:v4.28.0 init test mathlib
lake build  # 首次构建约 30 分钟（下载 Mathlib 缓存）

# 测试一个简单证明
echo 'import Mathlib
example : 1 + 1 = 2 := by norm_num' > Test.lean
lake env lean Test.lean  # 应当无输出、静默通过
```

</details>

---

## 安装

```bash
# 克隆仓库
git clone https://github.com/ai4math/ai4math.git
cd ai4math

# 安装 Python 依赖
pip install -r requirements.txt

# 设置 API 密钥
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 快速开始

### 证明单个定理

```bash
# 证明内置问题
python run_single_lane.py --builtin nat_add_comm --provider anthropic

# 证明自定义定理
python run_single_lane.py \
  --theorem "theorem t (n : Nat) : n + 0 = n" \
  --provider anthropic

# 使用不同模型
python run_single_lane.py \
  --builtin nat_add_comm \
  --provider anthropic \
  --model claude-opus-4-6

# 详细输出（显示完整 LLM prompt 和 Lane 状态转换）
python run_single_lane.py --builtin nat_add_comm --provider anthropic --verbose
```

`run_single_lane.py` 会逐步输出全管线的 10 个阶段：

| 步骤 | 内容 |
|------|------|
| 1 | 读题 & 问题加载 |
| 2 | Lane 运行时组件组装（EventBus, PolicyEngine, Knowledge, AgentPool） |
| 3 | 知识注入（从知识金字塔提取） |
| 4 | 方向规划（3–4 个异构探索方向） |
| 5 | 证明生成 + 三级验证循环 |
| 6 | 状态机结果（事件驱动转换） |
| 7 | 事件流（类型化事件日志） |
| 8 | Green Contract 检查（NONE → … → SORRY_FREE） |
| 9 | 状态压缩摘要（one-liner + prompt 注入格式） |
| 10 | Dashboard 全局视图 |

### 运行基准评测

```bash
# 快速评测 — 5 道内置题（约 2 分钟）
bash eval.sh --real --benchmark builtin

# 所有基准各取 10 题
bash eval.sh --real --quick

# miniF2F 完整评测（244 题，pass@8）
bash eval.sh --real --benchmark minif2f --samples 8

# 使用 Opus 模型
bash eval.sh --real --benchmark builtin --model claude-opus-4-6

# 启用多角色（Generator ↔ Repair Agent 交替）
bash eval.sh --real --benchmark builtin --multi-role

# 查看所有选项
bash eval.sh --help
```

---

## 基准数据集

AI4Math 内置 **1,631** 道形式化数学题，覆盖 7 个基准：

| 数据集 | 题数 | 难度 | 领域 |
|--------|------|------|------|
| `builtin` | 5 | 简单–中等 | 冒烟测试 |
| `minif2f` | 244 | AMC → IMO | 竞赛数学 |
| `putnambench` | 672 | 大学竞赛 | Putnam 1962–2024 |
| `proofnet` | 360 | 本科 | 分析、代数、拓扑 |
| `fate-m` | 150 | 本科 | 抽象代数 |
| `fate-h` | 100 | 研究生 | 抽象代数 |
| `fate-x` | 100 | 研究级 | 抽象代数 |

所有题目均为 Lean 4 定理陈述，使用 Mathlib 依赖。

---

## 评测

结果保存在 `results/evals/`（汇总）和 `results/traces/`（每题详情）。

```bash
# 全量评测
bash eval.sh --real

# 断点续跑（跳过已完成的题目）
python run_eval.py --benchmark minif2f --provider anthropic --resume

# 启用 Lean 4 真实验证（需安装 Lean 4 或 Docker）
bash eval.sh --real --lean
```

报告指标：**pass@k**（无偏估计）、总 token 数、总耗时、策略分布。

---

## Docker 部署

使用 Docker 获得可复现的 Lean 4 验证环境：

```bash
cd docker

# 构建镜像（首次约 30 分钟，下载 Mathlib）
docker compose build

# 启动 Lean 4 REPL 守护进程
docker compose up -d lean

# 运行带真实验证的评测
docker compose run --rm agent \
  python run_eval.py \
    --benchmark builtin \
    --provider anthropic \
    --lean-mode real
```

---

## 项目结构

```
274 个 Python 文件 · 57,000+ 行代码 · 938 项测试 · 7 大基准
```

| 目录 | 行数 | 定位 |
|------|------|------|
| `engine/` | ~12,000 | 验证 OS：REPL 池、三级验证、弹性伸缩、广播总线 |
| `engine/lane/` | ~3,500 | Lane 运行时：状态机、策略引擎、恢复、错误分类、压缩、持久化 |
| `knowledge/` | ~2,200 | 活知识系统：四层金字塔、读写管道、衰减遗忘 |
| `prover/` | ~7,600 | 证明编排：异步管线、修复、分解、代码生成、引理银行 |
| `prover/unified/` | ~1,700 | **v3 统一主管线**: Profile · Runner · adapters · system_prompts · tool_kits |
| `agent/` | ~3,700 | 智能体层：11 种角色、策略控制、钩子、插件 |
| `benchmarks/` | ~800 | 评测框架：7 大基准加载器、指标计算 |
| `tests/` | ~4,000 | 1001 项测试覆盖全部核心模块 |
| `data/` | — | 1,631 道形式化题目（miniF2F、PutnamBench、ProofNet、FATE 等） |

### 关键入口文件

| 文件 | 用途 |
|------|------|
| `run_unified.py` | **v3 推荐入口** — 单题, 通过 `--profile` 切换方法学 |
| `run_eval.py` | 批量评测; 支持 `--profile` 走 v3 主管线 |
| `run_single_lane.py` | 单题调试 — 逐步遍历全管线 (v2 兼容路径) |
| `eval.sh` | 一键评测脚本 |
| `engine/lane/integration.py` | `LaneProofRunner` — 异步证明执行主入口 |
| `prover/pipeline/proof_pipeline.py` | `ProofPipeline` — 状态机驱动证明管线，支持断点续证 |
| `prover/unified/runner.py` | **v3 统一 runtime** — 所有方法的执行核心 |
| `prover/unified/profiles.py` | **v3 Profile preset** — 加新方法只改这一个文件 |
| `prover/assembly.py` | 全系统组装器 |

---

## SOTA 对比

### miniF2F-test（244 题）

| 方法 | 通过率 | 类型 |
|------|--------|------|
| Goedel-Prover-V2-32B | **90.4%**（pass@32） | 全证明生成 |
| Kimina-Prover-72B | 84.0%（pass@32） | 全证明生成 |
| DeepSeek-Prover-V2-671B | 82.4%（pass@32） | 全证明生成 |
| **AI4Math (Claude Opus 4.6)** | *评测中* | 智能体平台 |

> **说明：** 以上 SOTA 是独立证明生成器。AI4Math 是编排平台——可以使用*任何* LLM 作为后端，通过结构化验证、知识积累和多智能体协调增加价值。

---

## 路线图

- [x] Lane 运行时：状态机、策略引擎、自动恢复
- [x] 三级验证（L0/L1/L2）
- [x] 活知识系统（四层金字塔）
- [x] 多智能体协调与方向规划
- [x] Green Contract 验证分级
- [x] 摘要压缩（prompt 注入）
- [x] 断点续证（Checkpoint/Resume）
- [x] Sorry/Axiom 完整性检查
- [x] 7 个基准数据集（1,631 题）
- [ ] 世界模型（Lean 4 状态动力学预测器）
- [ ] 稠密嵌入前提检索（sentence-transformers）
- [ ] Coq / Isabelle 后端支持
- [ ] Web UI 交互式定理证明
- [ ] 分布式多节点评测
- [ ] 基于 RL 的智能体训练

---

## 参与贡献

欢迎贡献！请遵循以下流程：

1. **Fork** 仓库并创建功能分支。
2. **编写测试** 覆盖新功能（项目维护 938+ 项测试）。
3. 提交前**运行测试套件**：
   ```bash
   PYTHONPATH=. python -m pytest tests/ -q
   ```
4. **遵循现有代码风格** — 类型标注、文档字符串、日志记录。
5. 提交 **Pull Request** 并清晰描述修改内容和原因。

### 特别欢迎的贡献方向

- 更多模型的基准评测（GPT-4、Gemini、开源模型）
- Coq / Isabelle transport 实现
- 稠密嵌入检索器（前提选择）
- Web UI 交互式证明探索
- 文档改进和翻译

---

## 引用

```bibtex
@software{ai4math2026,
  title   = {AI4Math: An Agent Operating System for Formal Theorem Proving},
  year    = {2026},
  url     = {https://github.com/ai4math/ai4math}
}
```

---

## 致谢

AI4Math 受到以下优秀开源项目的启发：

- [Lean 4](https://leanprover.github.io/) 和 [Mathlib4](https://github.com/leanprover-community/mathlib4) — 形式化验证基础
- [miniF2F](https://github.com/openai/miniF2F) — 最广泛使用的形式化数学基准
- [PutnamBench](https://github.com/trishullab/PutnamBench) — 多语言 Putnam 竞赛题
- [ProofNet](https://github.com/rahul3613/ProofNet) — 本科级形式化数学
- [FATE](https://github.com/fate-ubw/FATE) — 多难度抽象代数基准
- [FormalMATH](https://github.com/FormalMATH/FormalMATH) — 大规模多领域基准

---

## 许可证

[MIT](LICENSE)

---

<div align="center"><sub>📖 <a href="README.md">English version</a> · <a href="https://ai4math.github.io/ai4math">Interactive Demo</a></sub></div>
