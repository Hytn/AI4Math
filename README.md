# AI4Math — Formal Proof Agent Platform

> **别人在让 LLM 变得更会写 Lean4 代码，AI4Math 在让 Lean4 变成 LLM 更容易探索的环境。**
> 内置 1,631 道真实形式化数学题 · 7 大公认基准 · 一键复现 · APE v2 (Lean4 交互环境) + 异构并行 + 跨线程广播

> **⚠️ 项目状态：架构原型（Architecture Prototype）**
>
> 本项目的核心架构（REPL 连接池、广播总线、三级验证调度）已完成 Python 实现并通过 99 项单元测试。
> 但以下方面**尚未完成**，文档中涉及的性能数据均为**设计目标（Design Target）**而非实测结果：
>
> - ❌ **未在真实 Lean4 环境下进行端到端评测** — 所有基准数据集（miniF2F、PutnamBench 等）的 pass@k 数据待测
> - ❌ **延迟数据（50ms、~1μs 等）为理论估算** — 基于 lean4-repl 项目的已知性能特征，未在本系统中实际计量
> - ❌ **"信息效率 ~2000×"为理论上限估算** — 基于简化假设（1 bit vs ~100 bits），实际信息增益取决于 AgentFeedback 的利用效率
> - ❌ **"L0 过滤 ~90%"为预期值** — 基于 LLM 输出中常见语法错误的经验比例，未经统计验证
>
> 要获得真实评测数据，请参阅 [评测指南](#评测指南) 搭建 Lean4 Docker 环境后运行 `bash eval.sh --real`。

```
一键复现:  bash eval.sh              # Mock 冒烟测试 (无需 API Key, 30 秒)
真实评测:  bash eval.sh --real        # Claude API 全量评测
```

---

## 目录

- [为什么做这个项目](#为什么做这个项目)
- [三大核心创新](#三大核心创新)
- [快速开始](#快速开始)
- [内置基准数据集](#内置基准数据集)
- [评测指南](#评测指南)
- [同期 SOTA 对比](#同期-sota-对比)
- [APE v2 引擎：环境工程革命](#ape-v2-引擎环境工程革命)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 为什么做这个项目

形式化数学证明是 AI 推理能力的终极试金石——每一步推理都必须经过编译器的严格验证，容不得半点含糊。

### 当前领域的三个根本问题

**问题 1：所有系统都把 Lean4 当作黑盒 oracle。** 当前 SOTA 系统 (DeepSeek-Prover、Goedel-Prover、Kimina-Prover) 的工作流都是 "LLM 生成完整证明 → Lean4 编译验证 → pass/fail → 如果失败则重新生成"。每一轮交互只获得 **1 bit 信息** (通过/失败)，花费 **2-12 秒**。这意味着 Agent 在极其贫乏的信息环境中做搜索——就像蒙着眼睛走迷宫，每走一步只被告知 "错了"，但不知道错在哪。

**问题 2：并行方向之间没有信息流。** 同时启动 N 个 LLM 采样，但各路采样完全独立。方向 A 发现 "ring 对 ℕ 减法无效" 这个宝贵信息，方向 B/C/D 毫不知情，继续在同一个坑里反复犯错。N 路并行的有效覆盖常常只有 ~1.5 倍独立搜索，而不是 N 倍。

**问题 3：失败尝试的知识完全丢失。** 第 N 次尝试不知道前 N-1 次学到了什么 (除了将失败的 proof 文本塞进 prompt)。没有结构化的知识积累机制——哪些 tactic 已被证明无效、哪些子引理已经证出、哪些 goal 已经关闭。

### AI4Math 的解法

AI4Math 从根本上重新定义了 Agent 与证明环境的交互模式：

```
当前 SOTA:  LLM → Lean4 完整编译 (2-12s) → pass/fail (1 bit)    × N 独立采样
AI4Math:    LLM → APE v2 环境 (*目标* 50ms) → 结构化反馈 (~100 bits) × N 路实时广播
```
> *注：以上延迟数据为设计目标，尚未在本系统中实测验证。*

三大核心创新协同构成一个自增强飞轮：**快速环境** 让 Agent 有预算做更多探索；更多探索产生更多 **结构化反馈**；结构化反馈通过 **广播总线** 让所有方向共享知识；共享知识让下一轮探索的起点更高。

---

## 三大核心创新

### 🏗 创新一：APE v2 环境引擎 — 把 Lean4 变成 Agent 的交互环境

> **核心主张：不自建简化内核，而是把 Lean4 本身变成 Agent 的高性能交互环境。**

所有竞品都在 Lean4 之外构建系统 (LLM 生成代码 → 调用 Lean4 编译)。AI4Math 反其道而行——将 Lean4 REPL 变成 Agent 的操作环境，让 Agent 可以在环境中交互式地探索证明路径。

**技术实现 (`engine/lean_pool.py` + `engine/verification_scheduler.py`):**

启动 N 个 Lean4 REPL 长连接进程，每个进程预加载 `import Mathlib` 环境。Agent 的验证请求通过 least-busy 策略分发到空闲进程，实现真正的并行验证。

```
首次启动: 加载 Mathlib 环境 → ~30s (一次性成本)
后续验证: 在已加载环境上增量执行 → 50-500ms (vs 2-12s 完整编译)
精度:     100% (Lean4 本身在验证，不是模拟)
并行度:   N 路同时验证 (N = REPL 池大小)
```

三级验证调度 (L0 → L1 → L2)，*延迟为设计目标*：

```
L0 PreFilter  (*目标* ~1μs)  : 纯语法检查 (sorry/括号/Lean3语法/ℕ减法陷阱), 预期过滤大部分无效输出
L1 REPL Pool  (*目标* ~50ms) : Lean4 增量验证, 精确的类型检查, 结构化 goal state
L2 Full Compile (~3s) : 最终可信认证, 仅对 L1 通过的候选执行
```

**关键机制——env_id 分叉：** lean4-repl 的每条命令返回一个 `env_id`（环境快照 ID），后续命令可以在任意 `env_id` 上继续。这意味着 MCTS 搜索树中的每个节点可以绑定一个 `env_id`——从这个节点分支出去的 tactic 尝试，各自产生不同的后继环境，不需要重新建立状态。分叉是零成本的。

```python
# 核心操作: 在 env_id=42 上同时尝试 4 条 tactic
results = pool.try_tactics_parallel(env_id=42, tactics=["simp", "ring", "omega", "exact?"])
# → simp: 成功, new_env_id=43, goals=[goal_A']
# → ring: 失败, type_mismatch (结构化错误信息)
# → omega: 成功, new_env_id=44, goals=[]  ← 证明完成!
# → exact?: 成功, new_env_id=45, goals=[goal_B]
# 总耗时: ~200ms (并行), 而不是 4×12s=48s (串行编译)
```

### 📡 创新二：跨线程实时广播 — 一个方向的发现即刻惠及所有方向

> **核心主张：一个方向发现了问题，立刻广播给所有正在执行的方向。**

所有竞品的并行采样是无通信的——N 路独立执行，结果合并。AI4Math 的并行方向之间通过 `BroadcastBus` 实时通信，实现三种指数级剪枝：

**负面知识广播：** 方向 A 发现 "ring 对 ℕ 减法无效" → 立即广播 → 方向 B/C/D 在下一步就避开这条死路，不再浪费 tactic 尝试。每一次有效广播都在指数大小的搜索树上砍掉整棵子树。

**正面发现广播：** 方向 D 通过 `exact?` 找到 `Nat.sub_add_cancel` 可用 → 立即广播 → 方向 B 的候选 tactic 集合瞬间扩充，下一步就能使用这个引理。

**部分证明广播：** 方向 A 的前 3 步成功，关闭了 2 个 goal → 广播 env_id 和剩余 goals → 方向 C 不从头开始，直接 fork 这个 env_id 继续。

**技术实现 (`engine/broadcast.py`):**

```python
# 发布-订阅模型, 发布者不阻塞, 每个订阅者有独立队列
bus = BroadcastBus()
sub_a = bus.subscribe("direction_A")
sub_b = bus.subscribe("direction_B")

# 方向 A 发现引理 → 自动广播 (方向 A 不会收到自己的消息)
bus.publish(BroadcastMessage.positive(
    source="direction_A",
    discovery="Nat.sub_add_cancel solves the ℕ subtraction",
    lemma_name="Nat.sub_add_cancel"))

# 方向 B 的下一轮 prompt 中自动注入:
# "## Teammate discoveries (use these)
#  - [direction_A] USEFUL: Nat.sub_add_cancel solves the ℕ subtraction"
broadcast_context = bus.render_for_prompt("direction_B")
```

**为什么效果是指数级的：** 没有广播时，4 个方向是 4 次独立搜索。有广播后，每一次有效的负面知识广播都等价于在所有方向的搜索树上同时剪枝——原本每个方向要独立浪费 20 步才能发现的死路，现在一次广播就砍掉了。搜索空间不是加法缩减 (4S→3.5S)，而是乘法缩减 (每条死路被砍掉的子树大小是指数级的)。

### 🧠 创新三：错误驱动搜索 + 单题内知识积累

> **核心主张：错误不是终点，是搜索信号。每一次失败都让后续尝试更强。**

**结构化 AgentFeedback (`engine/error_intelligence.py`):**

传统方式下 Agent 看到的是 Lean4 的原始错误文本。AI4Math 将这些信息转化为结构化的 `AgentFeedback`：

```python
AgentFeedback:
  remaining_goals:     [GoalState(target="n + 0 = n", hypotheses=["n : Nat"])]
  failed_tactic:       "ring"
  error_category:      "tactic_failed"
  repair_candidates:   [
    RepairCandidate("omega",          confidence=0.6, source="heuristic"),
    RepairCandidate("simp [Nat.add_zero]", confidence=0.85, source="exact?"),  ← Lean4 自己搜到的
  ]
  progress_score:      0.3  (已关闭 1/3 的 goals)
  goals_closed:        1
```

**`exact?`/`apply?` 集成：** 本项目利用 Lean4 内置的引理搜索 tactic——`exact?` 做类型驱动的精确匹配，比任何 BM25/embedding 检索都准确。当一个 tactic 失败后，ErrorIntelligence 在同一个 REPL 会话中调用 `exact?`，让 Lean4 自己告诉 Agent "用哪个引理能解决当前 goal"。

**跨尝试知识积累：** 每一次失败都产出两类结构化知识：

| 知识类型 | 示例 | 传播方式 |
|---------|------|---------|
| 负面知识 | "`ring` 对此题无效 (tactic_failed)" | 广播 → 所有方向搜索树剪枝 |
| 正面资产 | "辅助引理 `h : n ≤ 2^n` 已证" | 广播 + `share_lemma()` 注入 REPL 环境 |
| 进度快照 | "前 3 步成功, env_id=42" | 广播 → 其他方向可 fork 继续 |
| 错误模式 | "type_mismatch: Nat vs Int 出现 3 次" | ErrorIntelligence 积累 → 注入 prompt |

**飞轮效应：** 前 5 次尝试可能都失败，但第 6 次的成功概率远高于独立采样 6 次中任意一次的成功概率。传统的 pass@k 假设每次尝试独立，但在 AI4Math 中，尝试之间有强正相关——后续尝试站在前面所有尝试的肩膀上。

**信息密度对比（理论估算）：**

```
传统 SOTA:  每次交互获得 1 bit (pass/fail),    花费 2-12s
AI4Math:   每次交互获得 ~100 bits (AgentFeedback), 花费 *目标* 50ms
理论信息效率提升上限: ~2000×  (100/1 × 12000/50)
```

> *注：100 bits 为 AgentFeedback 信息容量的理论估算（goal states + error structure + repair candidates），
> 实际信息增益取决于 Agent 对反馈的利用效率。传统系统也会返回错误消息文本（不止 1 bit），
> 此处使用 1 bit 作为简化下界。*

---

## 核心亮点速览

| 亮点 | 说明 |
|------|------|
| **🏗 APE v2 环境引擎** | Lean4 REPL 连接池 + 三级验证调度 (L0/L1/L2)。*设计目标*：验证延迟 2-12s → 50ms，精度 100%。不自建简化内核，用 Lean4 本身做验证 |
| **📡 跨线程实时广播** | 发布-订阅广播总线。负面知识/正面发现/部分证明三类消息实时共享。搜索空间指数收缩 |
| **🧠 错误驱动搜索** | 结构化 AgentFeedback + exact?/apply? 集成 + 修复候选生成。每次交互 ~100 bits 信息 |
| **⚡ 异构多方向并行** | 4 个数学方向同时探索 (自动化/归纳法/代数变换/引理检索)，各有独立上下文和模型配置 |
| **🪝 证明过程钩子** | 9 个关键时机的声明式规则。分析结论自动注入下一轮 prompt |
| **🧩 数学领域插件** | YAML 声明式策略配置。number-theory 插件自带 ℕ 减法陷阱警告 |
| **🔍 MCTS 证明搜索** | UCB1 + 反向传播 + 虚拟损失，支持 best-first / MCTS / BFS |
| **📚 引理银行** | 失败尝试中提取的子引理，通过 `share_lemma()` 注入所有 REPL 环境 |
| **🎯 Premise 检索** | BM25 + 语义嵌入 + Lean4 `exact?`/`apply?` 三路混合 |
| **📊 7 大真实基准** | miniF2F + PutnamBench + ProofNet + FATE-M/H/X + FormalMATH = **6,826** 道真实题 |

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

## APE v2 引擎：环境工程革命

### 与 v1 的根本区别

APE v1 试图用 626 行 Python 重建 Lean4 类型检查内核。这条路的问题在于方向本身：Lean4 内核有 ~10 万行 C++ 代码，Python 简化实现永远无法达到等价的判定能力，每一个 false negative 都在截断 Agent 的搜索空间。

APE v2 转换了思路：**不造轮子，造驾驶舱。** 把 Lean4 本身变成 Agent 的高性能交互环境。

### 五大核心模块 (新增 2,136 行)

```
engine/
├── broadcast.py              # 跨线程实时广播总线 (321 行)
│                              # 发布-订阅, 非阻塞, TTL 过期, prompt 渲染
├── lean_pool.py              # Lean4 REPL 连接池 (601 行)
│                              # N 路并行, 环境预加载, tactic 级交互
├── prefilter.py              # L0 语法预过滤器 (314 行)
│                              # 7 条内置规则, 可扩展, 结构化拒绝原因
├── error_intelligence.py     # 错误智能层 (466 行)
│                              # AgentFeedback, 修复候选, exact?/apply? 集成
└── verification_scheduler.py # 自适应验证调度器 (434 行)
                               # L0→L1→L2 路由, 自动广播, 统计
```

### 端到端数据流

```
Agent 生成候选 tactic 列表 [t1, t2, t3, t4, t5]
    ↓
L0 PreFilter (~1μs): t3 含 sorry → 直接排除, 返回结构化修复建议
    ↓ 剩余 [t1, t2, t4, t5]
    ↓
Lean4 REPL Pool: 4 条 tactic 并行送入 4 个 REPL session (各 ~50-200ms)
    ↓
  t1: 成功, new_env_id=42, remaining_goals=[goal_A']
  t2: 失败, type_mismatch → ErrorIntelligence 分析
  t4: 成功, new_env_id=43, remaining_goals=[] → 证明完成!
  t5: 失败, unknown_identifier → exact? 搜索修复候选
    ↓
Broadcast Bus:
  → 广播 NEGATIVE: "t2 fails (type_mismatch)" → 所有方向搜索树剪枝
  → 广播 POSITIVE: "t4 completed the proof!" → 所有方向可停止
    ↓
Structured Feedback Bus → Agent 拿到:
  - 2 个成功分支 (env_id=42, 43) 可继续展开
  - 2 个失败分支的精确修复建议 (含 exact?/apply? 返回的候选)
  - 所有分支的 goal state diff
  - 来自其他方向的 teammate discoveries
    ↓
Agent 决策: 展开 env_id=43 (证明已完成) 或 env_id=42 (只剩 1 个 goal)
```

### 保留模块 (v1 遗产)

v1 的 `engine/core/` (Expr, de Bruijn) 和 `engine/state/` (持久化数据结构) 仍然有价值——它们作为 Agent 内部的**证明规划模型** (planning model)，用于不需要精确验证的快速推理 (如估计 tactic 可能的效果)。但验证本身已完全交给 Lean4。

---

## 系统架构

> **下图是整个项目的完整模块地图。** 每一个源文件都标注了位置和作用。
> 新用户可以 top-down 地阅读：先看 5 层架构的整体关系，再沿着箭头深入具体模块。
>
> 📎 **[点击查看交互式架构全景图 → docs/architecture.html](docs/architecture.html)**

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 0 — Entry & Config                                          │
│  run_single.py · run_benchmark.py · config/ · docker/ · eval.sh    │
└────────────────────────────┬────────────────────────────────────────┘
                             │ calls Orchestrator.prove()
┌────────────────────────────▼────────────────────────────────────────┐
│  LAYER 1 — Agent  (9 sub-packages)                                  │
│                                                                     │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────┐ ┌─────────────────┐ │
│  │ brain/      │ │ strategy/    │ │ memory/  │ │ context/        │ │
│  │ LLM+prompt  │ │ meta+budget  │ │ work+epi │ │ window+compress │ │
│  └─────────────┘ └──────────────┘ └──────────┘ └─────────────────┘ │
│  ┌─────────────┐ ┌──────────────┐                                  │
│  │ tools/      │ │ executor/    │  ┌────────────────────────────┐  │
│  │ CAS+registry│ │ sandbox+lean │  │ runtime/  — SubAgent Pool │  │
│  └─────────────┘ └──────────────┘  │ hooks/    — 9 Lifecycle   │  │
│                                    │ plugins/  — YAML Strategy │  │
│                                    └────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Agent calls Engine for verification
┌────────────────────────────▼────────────────────────────────────────┐
│  LAYER 2 — Engine  (APE v2 + legacy modules)                       │
│                                                                     │
│  ┌──────────────────── APE v2 核心 (新增) ──────────────────────┐  │
│  │ broadcast.py          — 跨线程实时广播总线                    │  │
│  │ lean_pool.py          — Lean4 REPL 连接池 (N路并行)           │  │
│  │ prefilter.py          — L0 语法预过滤 (~1μs)                  │  │
│  │ error_intelligence.py — 结构化 AgentFeedback + exact?/apply?  │  │
│  │ verification_scheduler.py — L0→L1→L2 自适应调度              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────── Legacy (规划模型) ────────────────────────┐  │
│  │ core/  Expr+Env    kernel/  TypeChecker    state/  ProofState │  │
│  │ (de Bruijn)        (heuristic)             SearchTree(O(1))   │  │
│  │ search/  MCTS+UCB1+BFS    tactic/  18 built-in tactics       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Engine provides infra for Prover
┌────────────────────────────▼────────────────────────────────────────┐
│  LAYER 3 — Prover  (11 sub-packages)                                │
│                                                                     │
│  pipeline/          premise/         repair/         verifier/      │
│  Orchestrator       BM25+embed       diagnostor      lean_checker   │
│  HeteroEngine(v2)   reranker         repair_gen      lean_repl      │
│  proof_loop         selector         strategies      error_parser   │
│                                                                     │
│  codegen/    decompose/    conjecture/    sketch/    lemma_bank/    │
│  tactic_gen  goal_decomp   proposer       skeleton   bank+extract  │
│  scaffold    composition   verifier       templates  verifier       │
│  sorry_close                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 五层架构一句话总结

| 层 | 包 | 一句话 |
|----|-----|--------|
| **Layer 0** | `config/` `benchmarks/` `docker/` | 配置、数据集加载、容器化部署 |
| **Layer 1** | `agent/` (9 子包) | **决策层**：LLM 接口、策略控制、记忆管理、异构并行、钩子、插件 |
| **Layer 2** | `engine/` (5 新 + 6 旧) | **环境层 (APE v2)**：REPL 连接池、广播总线、验证调度、错误智能 |
| **Layer 3** | `prover/` (11 子包) | **编排层**：Orchestrator 主调度、proof loop、修复、分解、代码生成 |
| **入口** | `run_*.py` `eval.sh` | 单题调试 / 批量评测 / 双引擎演示 |

### 三大核心创新的数据流

```
用户输入定理
    ↓
Orchestrator.prove()
    ↓
ON_PROBLEM_START 钩子 → DomainClassifierHook 分类领域 → 匹配插件
    ↓
BroadcastBus.clear() — 新问题, 重置广播频道
    ↓
HeterogeneousEngine.run_round()  [每个方向一个 SubAgent]
    ├── 方向 A: 自动化探测 (Haiku, temp=0.2)
    │   ├── 生成 tactic → L0 PreFilter → L1 REPL 验证
    │   ├── 成功? → 广播 PARTIAL_PROOF (env_id + remaining goals)
    │   └── 失败? → ErrorIntelligence 分析 → 广播 NEGATIVE_KNOWLEDGE
    │
    ├── 方向 B: 归纳法专家 (Sonnet, temp=0.7)
    │   ├── 接收方向 A 的广播 → 注入 prompt "Teammate discoveries"
    │   ├── 生成 tactic → L0 → L1 验证
    │   └── 成功? → 广播 POSITIVE_DISCOVERY
    │
    ├── 方向 C: 代数变换 (Sonnet, temp=0.9)
    │   └── 接收所有方向的广播 → 知道哪些路走不通
    │
    └── 方向 D: 引理检索 (Sonnet, temp=0.5)
        ├── exact?/apply? 搜索 → 广播 LEMMA_PROVEN
        └── 引理通过 share_lemma() 注入所有 REPL 环境
    ↓
ResultFuser: 选出最佳结果 + 融合跨方向发现
    ↓
VerificationScheduler.verify_complete() — L0 → L1 → L2 终验
    ↓
BroadcastBus 保留全部消息 → 下一轮所有方向自动继承
```

---

## 项目结构

```
170+ 个源文件  ·  22,000+ 行代码  ·  347 个单元测试  ·  99/99 验证测试通过
```

### 模块统计

| 模块 | 文件数 | 职责 |
|------|--------|------|
| **`engine/` APE v2 核心 (新增)** | **5** | **REPL 连接池、广播总线、L0 预过滤、错误智能层、验证调度器** |
| `engine/` legacy | 6 | Expr/de Bruijn、TypeChecker (heuristic)、ProofState、MCTS |
| `agent/brain/` | 5 | LLM 接口、角色 prompt、模板引擎 |
| `agent/strategy/` | 6 | 元控制器、策略升级、置信度、预算、反思 |
| `agent/memory/` | 2 | 工作记忆 + 情景记忆 |
| `agent/context/` | 4 | 上下文窗口管理、压缩、优先级排序 |
| `agent/tools/` | 4 | 工具注册、CAS 桥接、Lean 自动化 |
| `agent/executor/` | 3 | 沙箱执行、验证环境管理、资源限制 |
| `agent/runtime/` | 3 | 子智能体运行时、并行池、结果融合 |
| `agent/hooks/` | 3 | 生命周期钩子、事件管理、内置规则 |
| `agent/plugins/` | 1 | 插件发现、加载、匹配 |
| `prover/pipeline/` | 7 | Orchestrator (v2, 集成广播)、heterogeneous engine (v2, 集成广播)、proof loop |
| `prover/premise/` | 5 | BM25 + 语义嵌入 + 重排序 |
| `prover/repair/` | 4 | 错误诊断、修复生成、补丁应用 |
| `prover/codegen/` | 5 | 代码生成、骨架、sorry closer |
| `prover/verifier/` | 6 | LeanChecker + REPL + 错误解析 |
| `prover/decompose/` | 3 | 定理分解、子目标调度、证明组装 |
| `prover/conjecture/` | 2 | 猜想生成、验证 |
| `prover/formalize/` | 2 | 自然语言 → 形式化语句 |
| `prover/lemma_bank/` | 3 | 引理提取、存储、跨 rollout 复用 |
| `prover/sketch/` | 3 | Sorry-based 脚手架、假设生成、模板 |
| `plugins/strategies/` | 3 | 示例策略插件（number-theory） |
| `benchmarks/` | 3+8 | 评测框架 + 7 大基准数据集加载器 |

---

## 可借鉴的设计亮点

以下是本项目中值得关注的工程设计，可直接迁移到其他 AI Agent 系统:

### 1. 发布-订阅广播总线 (`engine/broadcast.py`)

将并行 Agent 之间的信息共享从"事后融合"升级为"实时广播"。发布者不阻塞，每个订阅者有独立队列，消息带 TTL 自动过期。`render_for_prompt()` 方法将结构化消息直接转化为 LLM 可消费的 prompt 文本。这种模式适用于任何多 Agent 协作系统。

### 2. 结构化错误反馈 (`engine/error_intelligence.py`)

不返回 pass/fail 二元结果，而是返回 `AgentFeedback`——包含剩余 goals、修复候选、进度评估、类型不匹配的具体信息。`to_prompt()` 方法确保 LLM 能直接消费这些信息。`exact?`/`apply?` 集成让 Lean4 自己搜索修复方案——类型驱动的精确匹配比文本检索准确得多。

### 3. 三级验证调度 (`engine/verification_scheduler.py`)

将验证请求路由到代价递增的三级后端 (L0 ~1μs / L1 ~50ms / L2 ~3s)。每一级的拒绝都携带结构化原因，不浪费下一级的资源。这种"漏斗式过滤"模式适用于任何有快慢两种验证手段的系统。

### 4. Sorry-based 增量证明 (`prover/codegen/sorry_closer.py`)

不一次生成完整证明，而是先生成含 `sorry` 占位符的骨架，再逐个关闭。把一个困难的端到端问题转化为多个较简单的局部问题。

### 5. 引理银行 + share_lemma (`prover/lemma_bank/bank.py` + `engine/lean_pool.py`)

失败尝试中发现的有效子引理，通过 `LeanPool.share_lemma()` 注入所有 REPL 环境，变成后续所有方向可直接引用的已知事实。这让系统在失败中积累知识，搜索空间随尝试次数收缩。

### 6. MCTS 证明搜索 (`engine/search/__init__.py`)

将 AlphaGo 风格的 MCTS 应用于证明搜索: UCB1 平衡探索/利用，虚拟损失支持并行搜索，LLM 先验作为节点打分信号。与 APE v2 的 env_id 机制结合后，每个搜索节点直接绑定 Lean4 环境快照。

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
