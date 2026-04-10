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
| `agent/` | ~3,700 | 智能体层：11 种角色、策略控制、钩子、插件 |
| `benchmarks/` | ~800 | 评测框架：7 大基准加载器、指标计算 |
| `tests/` | ~4,000 | 938 项测试覆盖全部核心模块 |
| `data/` | — | 1,631 道形式化题目（miniF2F、PutnamBench、ProofNet、FATE 等） |

### 关键入口文件

| 文件 | 用途 |
|------|------|
| `run_single_lane.py` | 单题调试 — 逐步遍历全管线 |
| `run_eval.py` | 批量评测入口 |
| `eval.sh` | 一键评测脚本 |
| `engine/lane/integration.py` | `LaneProofRunner` — 异步证明执行主入口 |
| `prover/pipeline/proof_pipeline.py` | `ProofPipeline` — 状态机驱动证明管线，支持断点续证 |
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
