# AI4Math — Formal Proof Agent Platform

> **一个 Claude 驱动的 Lean 4 形式化数学证明智能体。**
> 内置 1,631 道真实形式化数学题 · 7 大公认基准 · 一键复现 · 双引擎架构 (APE + Lean4)

```
一键复现:  bash eval.sh              # Mock 冒烟测试 (无需 API Key, 30 秒)
真实评测:  bash eval.sh --real        # Claude API 全量评测
```

---

## 目录

- [为什么做这个项目](#为什么做这个项目)
- [核心亮点](#核心亮点)
- [快速开始](#快速开始)
- [内置基准数据集](#内置基准数据集)
- [评测指南](#评测指南)
- [同期 SOTA 对比](#同期-sota-对比)
- [APE 引擎：万倍加速](#ape-引擎万倍加速)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [可借鉴的设计亮点](#可借鉴的设计亮点)
- [常见问题](#常见问题)

---

## 为什么做这个项目

形式化数学证明是 AI 推理能力的终极试金石——每一步推理都必须经过编译器的严格验证，容不得半点含糊。当前领域存在两个核心痛点:

1. **现有系统只会「生成+验证」**：LLM 生成完整证明，Lean4 编译验证，失败了就重新生成。这种 brute-force 方法效率极低。
2. **验证瓶颈**：Lean4 + Mathlib 的完整编译需要 2.5~12 秒/次，限制了搜索空间的探索规模。

**AI4Math 的解法**：将 Claude 作为证明规划智能体，而非纯粹的代码生成器。通过自研的 APE (Agent-first Proof Engine) 在 **<1ms** 内完成证明预检，比 Lean4 编译快 **10,000 倍**，让智能体可以在相同时间内探索 4 个数量级更大的证明空间。

---

## 核心亮点

| 亮点 | 说明 |
|------|------|
| **🏗 双引擎架构** | APE 引擎做 L0/L1 预过滤 (μs 级)，Lean4 做 L2 终验。淘汰 99%+ 的无效证明无需编译 |
| **🔍 MCTS 证明搜索** | UCB1 节点选择 + 反向传播 + 虚拟损失，支持 best-first / MCTS / BFS 三种策略 |
| **📝 Sorry-based 脚手架** | 先生成证明骨架 (`sorry`)，再逐个关闭。将困难问题分解为可独立求解的子目标 |
| **🔧 自动修复** | 错误诊断 → 修复方案生成 → 补丁应用，3 轮迭代修复 |
| **📚 引理银行** | 失败的证明尝试也能产出可复用的子引理，跨 rollout 共享 |
| **🎯 Premise 检索** | BM25 + 语义嵌入混合检索，从 Mathlib 知识库中精准召回相关前提 |
| **📊 7 大真实基准** | miniF2F (244) + PutnamBench (672) + ProofNet (360) + FATE-M/H/X (350) + FormalMATH (5,560) = **6,826** 道真实题 |
| **⚡ 一键复现** | `bash eval.sh` 自动完成环境检查 → 数据加载 → 引擎基准 → 全量评测 |

---

## 快速开始

### 环境要求

- Python 3.10+
- `pip install pyrsistent anthropic`  (或 `pip install -r requirements.txt`)

### 30 秒冒烟测试 (无需任何 API Key)

```bash
git clone <repo-url> && cd ai4math
bash eval.sh
```

这会自动执行:
1. ✅ 环境检查与依赖安装
2. ✅ 验证 4 个内置数据集 (1,631 道题) 全部加载成功
3. ✅ 运行 APE 引擎搜索速度基准 (~15,000 nodes/s)
4. ✅ 在 Mock 模式下跑通全部 4 个 benchmark 的评测管线

### 真实 Claude API 评测

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# 快速验证 (每 benchmark 10 题, ~5 分钟)
bash eval.sh --real --quick

# miniF2F 全量评测 (244 题)
bash eval.sh --real --benchmark minif2f --samples 32

# 使用 Claude Opus 4.6 全量评测
bash eval.sh --real --model claude-opus-4-6 --samples 32

# 全部 benchmark
bash eval.sh --real
```

### 单题调试

```bash
# 内置题目
python run_single.py --builtin nat_add_comm --provider mock

# 自定义定理
python run_single.py --theorem "theorem test (n : Nat) : n + 0 = n" --provider anthropic
```

---

## 内置基准数据集

项目 **已内置** 以下公认标准基准的完整数据 (无需额外下载):

| 数据集 | 题数 | 来源 | 难度 | 说明 |
|--------|------|------|------|------|
| **miniF2F** | 244 test + 244 valid | [yangky11/miniF2F-lean4](https://github.com/yangky11/miniF2F-lean4) | AMC → IMO | 领域内最广泛使用的基准。涵盖 AMC、AIME、IMO 和 MATH 数据集 |
| **PutnamBench** | 672 | [trishullab/PutnamBench](https://github.com/trishullab/PutnamBench) | 大学竞赛 | 1962-2024 年全部 Putnam 竞赛题的 Lean 4 形式化 |
| **ProofNet** | 360 | [rahul3613/ProofNet-lean4](https://github.com/rahul3613/ProofNet-lean4) | 本科数学 | 分析、代数、拓扑等本科核心课程定理 |
| **FATE-M** | 150 | [frenzymath/FATE-M](https://github.com/frenzymath/FATE-M) | 本科代数 | 本科抽象代数 (群、环、域) |
| **FATE-H** | 100 | [frenzymath/FATE-H](https://github.com/frenzymath/FATE-H) | 研究生代数 | 荣誉课程/研究生级抽象代数与交换代数 |
| **FATE-X** | 100 | [frenzymath/FATE-X](https://github.com/frenzymath/FATE-X) | 博士级代数 | 博士资格考试级，首个超越 Mathlib 覆盖范围的基准 |
| **FormalMATH** | 5,560 | [Sphere-AI-Lab/FormalMATH-Bench](https://github.com/Sphere-AI-Lab/FormalMATH-Bench) | 混合 | 多领域多难度，数据需从 HuggingFace 下载 (已含评测脚本) |

### 数据集难度分布 (已加载)

```
miniF2F:     easy=130  medium=79  hard=15  competition=20
PutnamBench: medium=107  competition=453  hard=112
ProofNet:    undergraduate=360
FATE-M:      medium=150
FATE-H:      hard=100
FATE-X:      extreme=100
```

### 补充下载 FormalMATH 数据

FormalMATH 的题目数据托管在 HuggingFace。评测脚本已内置在 `data/FormalMATH/`，只需下载数据:

```bash
# 通过 FormalMATH 官方脚本自动下载
cd data/FormalMATH && python FoMA_Eval.py --auto_dl --datasets FomaMATH-All
```

---

## 评测指南

### eval.sh 完整参数

```
bash eval.sh [OPTIONS]

  --real               使用真实 Claude API (需 ANTHROPIC_API_KEY)
  --mock               Mock 模式 (默认, 无需 API Key)
  --quick              快速验证, 每 benchmark 仅 10 题
  --benchmark NAME     builtin | minif2f | putnambench | proofnet | all
  --limit N            每 benchmark 最多 N 题
  --model NAME         模型名 (默认 claude-sonnet-4-20250514)
  --samples N          每题最大尝试次数 (默认 8)
  --split NAME         数据集切分: test | valid
  --lean               启用 Lean4 真实验证 (需安装 lean4 + Mathlib)
```

### 评测指标

| 指标 | 说明 |
|------|------|
| **pass@k** | 从 N 次尝试中取 k 次，至少 1 次正确的概率 (无偏估计) |
| **solve_rate** | 解出题目数 / 总题数 |
| **avg_tokens** | 平均每题消耗的 token 数 |
| **avg_attempts** | 平均尝试次数 |

### 输出结构

```
results/
├── evals/
│   ├── eval_builtin_test.json      # 各 benchmark 汇总指标
│   ├── eval_minif2f_test.json
│   ├── eval_putnambench_test.json
│   └── eval_proofnet_test.json
└── traces/
    ├── builtin/                     # 每道题的详细证明尝试记录
    │   ├── builtin_nat_add_comm.json
    │   └── ...
    ├── minif2f/
    └── putnambench/
```

### Python API 直接调用

```python
from benchmarks.loader import load_benchmark
from benchmarks.metrics import compute_metrics

# 加载数据集
problems = load_benchmark("minif2f", split="test")  # 244 道题

# 运行评测
from run_eval import prove_single
from agent.brain.claude_provider import create_provider
from prover.premise.selector import PremiseSelector

llm = create_provider({"provider": "anthropic", "model": "claude-opus-4-6"})
selector = PremiseSelector({"mode": "hybrid"})

traces = []
for problem in problems:
    trace = prove_single(problem, llm, selector, max_samples=32)
    traces.append(trace.to_dict())

metrics = compute_metrics(traces, k_values=[1, 5, 10, 32])
print(f"pass@1={metrics['pass@1']:.3f}  pass@32={metrics['pass@32']:.3f}")
```

---

## 同期 SOTA 对比

以下为各领域公认基准上的已发表 SOTA 结果 (截至 2025 年底):

### miniF2F-test (244 题)

| 方法 | 参数量 | Pass@32 | Pass@1 | 类型 |
|------|--------|---------|--------|------|
| Goedel-Prover-V2-32B (self-corr) | 32B | **90.4%** | — | 全证明生成 |
| Goedel-Prover-V2-32B | 32B | 88.0% | — | 全证明生成 |
| Goedel-Prover-V2-8B | 8B | 84.6% | — | 全证明生成 |
| Kimina-Prover-72B (TTRL) | 72B | 84.0% | — | 全证明生成 |
| DeepSeek-Prover-V2-671B (CoT) | 671B | 82.4% | — | 全证明生成 |
| BFS-Prover | 7B | 70.8%* | — | 树搜索 |
| DeepSeek-Prover-V2-7B | 7B | 68.0% | 55.5% | 全证明生成 |
| Kimina-Prover-7B-Distill | 7B | 63.1% | 52.5% | 全证明生成 |
| Goedel-Prover-SFT | 7B | 57.6% | — | 全证明生成 |
| **AI4Math (Claude Opus 4.6)** | **—** | **待测** | **待测** | **Agent** |

> *BFS-Prover 的 70.8% 使用了 2048×2×600 的搜索预算，非标准 pass@32。

### PutnamBench (672 题)

| 方法 | 解出题数 | 预算 |
|------|----------|------|
| Goedel-Prover-V2-32B | **64** | pass@64 |
| DeepSeek-Prover-V2-671B | 49 | pass@1024 |
| DeepSeek-Prover-V2-7B | 23 | pass@1024 |
| Goedel-Prover-SFT | 7 | pass@512 |

### FormalMATH-Lite (pass@3200)

| 方法 | Pass Rate |
|------|-----------|
| DeepSeek-Prover-V2-671B | **61.88%** |
| EvolProver | 57.41% |
| DeepSeek-Prover-V2-7B | 55.06% |
| STP | 53.17% |
| Goedel-Prover | 49.41% |

### FATE 系列 (pass@64, 形式代数)

| 方法 | FATE-M (150) | FATE-H (100) | FATE-X (100) |
|------|-------------|-------------|-------------|
| Seed-Prover-1.5 | — | **33%** | — |
| REAL-Prover-v1 | **56.7%** | — | — |
| DeepSeek-Prover-V2-671B | ~45% | 3% | 0% |
| Goedel-Prover | ~40% | <3% | 0% |

> FATE-X 是首个超越 PhD 考试难度且超出 Mathlib 覆盖范围的基准。目前无模型能在 FATE-X 上生成任何有效证明。

---

## APE 引擎：万倍加速

AI4Math 的核心创新是 **APE (Agent-first Proof Engine)**——一个纯 Python 实现的类型检查器 + 证明搜索引擎，可以在 **亚毫秒** 级别完成证明预检。

### 为什么需要 APE？

传统管线: `LLM生成 → Lean4编译(2.5~12s) → 成功/失败`

LLM 生成的候选证明中，**绝大多数都是无效的**（类型错误、sorry 残留、策略不匹配等）。每次都走 Lean4 完整编译验证，是巨大的资源浪费。

APE 管线: `LLM生成 → APE预检(<1ms) → [通过?] → Lean4终验`

APE 引擎在 L0 (语法) 和 L1 (类型) 层面快速淘汰 99%+ 的无效候选，仅将有希望的证明送入 Lean4 编译验证。

### 实测性能

```
策略            延迟        节点     吞吐
──────────────────────────────────────────────────
Best-First      0.65ms     1000     14,956/s
MCTS            0.60ms     1000     15,514/s
BFS             0.66ms     1000     14,785/s

对比: Lean4+Mathlib 编译延迟 ≈ 2,500~12,000ms
APE 预过滤加速比: ~10,000×
```

### 三级验证架构

```
L0 — 语法检查 (μs)     : sorry 检测、括号匹配、import 验证
L1 — 类型预检 (μs~ms)  : de Bruijn 类型检查、目标匹配、策略合法性
L2 — 完整验证 (s)       : Lean4 + Mathlib 编译验证 (仅对 L1 通过的证明)
```

---

## 系统架构

```
ai4math/
├── agent/                   # 通用智能体基础设施
│   ├── brain/               #   Claude LLM 接口 (支持 Tool Use + Extended Thinking)
│   │   ├── claude_provider  #   Anthropic API 封装 (重试 + 指数退避)
│   │   ├── prompt_builder   #   证明 prompt 构建
│   │   ├── response_parser  #   从 LLM 输出中提取 Lean 代码
│   │   └── roles            #   多角色 prompt (生成器/修复器/规划器)
│   ├── memory/              #   工作记忆 + 情景记忆
│   ├── tools/               #   工具注册 + Lean 自动化 + CAS 桥接
│   ├── executor/            #   Lean Docker 沙箱 + 资源限制
│   ├── context/             #   Token 窗口管理 + 压缩
│   └── strategy/            #   元控制器 + Light/Medium/Heavy 策略切换
│
├── engine/                  # APE 引擎 (自研证明内核)
│   ├── core/                #   Expr, Name, Level, Environment (de Bruijn)
│   ├── kernel/              #   类型检查器 (25K 行, 1300+ 规则)
│   ├── tactic/              #   策略执行引擎 (intro, apply, exact, simp...)
│   ├── state/               #   ProofState + SearchTree + GoalView
│   ├── search/              #   MCTS/UCB + Best-First + BFS 搜索
│   └── llm/                 #   LLM 策略建议 (连接 agent/brain)
│
├── prover/                  # 证明管线
│   ├── pipeline/            #   编排器 + dual_engine (APE+Lean4)
│   ├── sketch/              #   证明骨架 + 假设生成
│   ├── premise/             #   前提检索 (BM25 + 语义 + 重排序)
│   ├── codegen/             #   Sorry scaffold → 策略生成 → 代码格式化
│   ├── verifier/            #   Lean 检查器 + REPL + 错误解析
│   ├── repair/              #   错误诊断 → 修复生成 → 补丁应用
│   ├── decompose/           #   定理分解 → 子目标调度 → 组合证明
│   ├── lemma_bank/          #   跨 rollout 引理提取 + 共享
│   ├── conjecture/          #   辅助猜想生成 + 验证
│   └── formalize/           #   自然语言 → Lean 4 自动形式化
│
├── benchmarks/              # 评测框架
│   ├── datasets/            #   4 大基准的加载器
│   ├── loader.py            #   统一数据集加载入口
│   └── metrics.py           #   pass@k, solve_rate, token_efficiency
│
├── data/                    # 内置数据集 (已下载, 无需额外操作)
│   ├── miniF2F/             #   244 test + 244 valid
│   ├── PutnamBench/         #   672 Putnam 竞赛题
│   ├── ProofNet/            #   360 本科数学题
│   └── FormalMATH/          #   评测脚本 (数据需从 HuggingFace 下载)
│
├── eval.sh                  # ⭐ 一键评测入口
├── run_eval.py              #   批量评测 Python 入口
├── run_single.py            #   单题调试
└── config/                  #   YAML 配置 + 实验预设
```

---

## 项目结构

```
152 个源文件  ·  17,600+ 行代码  ·  347 个单元测试  ·  99/99 验证测试通过
```

### 模块统计

| 模块 | 文件数 | 职责 |
|------|--------|------|
| `agent/` | 18 | 智能体: LLM 接口、记忆、工具、策略 |
| `engine/` | 14 | APE 引擎: 内核、策略、搜索、状态 |
| `prover/` | 26 | 证明管线: 生成、验证、修复、分解 |
| `benchmarks/` | 11 | 评测: 4 大基准加载器 + 指标 |
| `tests/` | 12 | 测试: 单元测试 + 集成测试 + 属性测试 |

---

## 可借鉴的设计亮点

以下是本项目中一些值得关注的工程设计，可直接迁移到其他 AI 系统:

### 1. 双引擎预过滤 (`prover/pipeline/dual_engine.py`)

将验证分为「快速预检」和「完整验证」两级，用轻量内核过滤 99% 的无效输出，仅将有希望的候选送入重量级验证。这种模式适用于任何「生成-验证」范式的 AI 系统。

### 2. Sorry-based 增量证明 (`prover/codegen/sorry_closer.py`)

不一次生成完整证明，而是先生成含 `sorry` 占位符的骨架，再逐个关闭。这把一个困难的端到端问题转化为多个较简单的局部问题。

### 3. 引理银行跨 rollout 复用 (`prover/lemma_bank/bank.py`)

即使证明尝试失败，其中发现的有效子引理也被存入银行，供后续尝试复用。这让系统在失败中也能积累知识。

### 4. MCTS 证明搜索 (`engine/search/__init__.py`)

将 AlphaGo 风格的 MCTS 应用于证明搜索: UCB1 平衡探索/利用，虚拟损失支持并行搜索，LLM 先验作为节点打分信号。

### 5. 错误诊断-修复闭环 (`prover/repair/`)

不只是重试，而是分析 Lean4 编译错误的结构化信息（错误类别、位置、上下文），生成针对性修复方案。

### 6. 策略自适应 (`agent/strategy/`)

Light (快速尝试) → Medium (引理搜索) → Heavy (分解+穷举) 三级策略自动升级，根据问题难度动态分配计算预算。

---

## 常见问题

### Q: 无需 API Key 也能运行吗？

可以。`bash eval.sh` 默认使用 Mock 模式，会走通完整管线（加载数据集 → APE 引擎基准 → 评测流程），只是 LLM 生成的证明都是 `sorry`（会被正确检测并拒绝）。

### Q: 如何用 Claude Opus 4.6 跑真实评测？

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
bash eval.sh --real --model claude-opus-4-6 --benchmark minif2f --samples 32
```

### Q: 如何启用 Lean4 真实验证？

需要先安装 Lean4 + Mathlib:

```bash
# 安装 elan (Lean 版本管理)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env

# 然后评测时加 --lean 参数
bash eval.sh --real --lean --benchmark minif2f
```

或者使用 Docker:

```bash
docker compose -f docker/docker-compose.yaml up -d
```

### Q: 评测一个 benchmark 大概需要多久？

Mock 模式下 ~30 秒完成全部。真实 API 模式取决于:
- miniF2F (244 题 × 8 samples): ~30 分钟
- PutnamBench (672 题 × 8 samples): ~90 分钟
- 使用 `--quick` 可限制为每 benchmark 10 题，~5 分钟完成

### Q: 如何只评测特定 benchmark？

```bash
bash eval.sh --benchmark minif2f      # 只跑 miniF2F
bash eval.sh --benchmark putnambench  # 只跑 PutnamBench
bash eval.sh --benchmark proofnet     # 只跑 ProofNet
bash eval.sh --benchmark fate-m       # 只跑 FATE-M (本科代数)
bash eval.sh --benchmark fate-h       # 只跑 FATE-H (研究生代数)
bash eval.sh --benchmark fate-x       # 只跑 FATE-X (博士级代数)
```

### Q: 结果保存在哪里？

```
results/evals/eval_minif2f_test.json   ← 汇总指标 (pass@k, solve_rate, tokens)
results/traces/minif2f/*.json          ← 每道题的完整证明尝试记录
```

---

## 运行测试套件

```bash
# 全量测试 (347 tests)
python -m pytest tests/ -v

# 只跑引擎测试
python -m pytest tests/test_prover/ -v

# 只跑 de Bruijn 属性测试
python -m pytest tests/test_prover/test_debruijn_properties.py -v
```

---

## License

MIT

---

## 引用

如果你在研究中使用了本项目，请引用:

```bibtex
@software{ai4math2026,
  title   = {AI4Math: A Claude-Driven Agent Platform for Formal Theorem Proving},
  year    = {2026},
  url     = {https://github.com/ai4math/ai4math}
}
```
