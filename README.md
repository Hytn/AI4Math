# AI4Math — 形式化定理证明的智能体操作系统

> **别人在让 LLM 更会写 Lean4 代码，AI4Math 在构建让智能体群落自主探索数学前沿的基底平台。**

```
30 秒体验:    python run_single_lane.py              # 单题调试, 遍历全管线
冒烟测试:    bash eval.sh                            # Mock 模式 (无需 API Key)
真实评测:    bash eval.sh --real --benchmark builtin  # 5 题快速跑分
```

---

## 一、快速开始

### 环境准备

```bash
git clone <repo-url> && cd ai4math
pip install -r requirements.txt
```

### A. 单题调试 — 遍历完整数据管线 (推荐首次使用)

```bash
# Mock 模式 — 无需 API Key, 10 秒走完全管线
python run_single_lane.py

# 指定内置题目
python run_single_lane.py --builtin nat_add_comm

# 自定义定理
python run_single_lane.py --theorem "theorem t (n : Nat) : n + 0 = n"

# 真实 Claude API
export ANTHROPIC_API_KEY="sk-ant-..."
python run_single_lane.py --provider anthropic --builtin nat_add_comm

# 详细输出 (含 LLM prompt 全文)
python run_single_lane.py --provider anthropic --verbose
```

`run_single_lane.py` 会逐步输出 10 个阶段的中间状态, 对应下方「项目脊柱」中的数据流:

```
Step 1:  读题 & 问题加载
Step 2:  组装 Lane 运行时组件 (EventBus, PolicyEngine, Knowledge, AgentPool...)
Step 3:  知识注入 (KnowledgeReader.render_for_prompt)
Step 4:  方向规划 (DirectionPlanner.plan → 3-4 个异构探索方向)
Step 5:  构建 TaskPacket & 运行证明循环 (LaneProofRunner.run)
Step 6:  状态机结果 (ProofTaskStateMachine — 事件驱动状态转换)
Step 7:  事件流 (ProofEventBus — 类型化事件日志)
Step 8:  Green Contract 检查 (NONE → SYNTAX_CLEAN → GOALS_CLOSED → SORRY_FREE)
Step 9:  压缩状态摘要 (SummaryCompression — one-liner + prompt 注入格式)
Step 10: Dashboard 全局视图
```

### B. 最小 Benchmark 跑分

```bash
# Mock 冒烟 — 5 题内置题, 无需 API Key, 约 30 秒
bash eval.sh --benchmark builtin

# 真实 Claude API — 5 题内置题跑分 (~2 分钟)
export ANTHROPIC_API_KEY="sk-ant-..."
bash eval.sh --real --benchmark builtin

# 快速真实评测 — 每个 benchmark 取 10 题
bash eval.sh --real --quick

# miniF2F 全量评测 (488 题)
bash eval.sh --real --benchmark minif2f --samples 32

# 使用 Opus 模型
bash eval.sh --real --benchmark builtin --model claude-opus-4-6

# 启用多角色 (Generator → Repair 交替)
bash eval.sh --real --benchmark builtin --multi-role
```

### C. 传统单题测试 (非 Lane 模式)

```bash
python run_single.py --builtin nat_add_comm --provider mock
python run_single.py --theorem "theorem test (n : Nat) : n + 0 = n" --provider anthropic
```

---

## 二、项目骨骼：一句话理解 AI4Math

**AI4Math 是什么？**

一个让数百个 AI 数学家像真实的研究院一样协同工作的操作系统。

它不是一个"更好的证明生成器"——它是一整套基础设施：底层提供毫秒级验证能力，中层积累可复用的数学知识，上层协调异构智能体群落分工合作。三层形成飞轮，越用越聪明。

**与当前 SOTA 的本质差异：**

```
当前范式 (DeepSeek/Goedel/Kimina):
    LLM → 生成完整证明 → Lean4 编译 (2-12s) → pass/fail (1 bit) × N 次独立重试
    ❌ 每轮只获得 1 bit 信息  ❌ 方向之间零通信  ❌ 失败经验完全丢失

AI4Math 范式:
    N 个异构智能体 → 验证 OS (目标 50ms) → 结构化反馈 (~100 bits) × 实时广播
    ✅ 丰富的错误诊断指导下一步  ✅ 发现即刻共享  ✅ 知识自动沉淀复用
```

**四根支柱，一个飞轮：**

```
┌─────────────────────────────────────────────────────────────────┐
│  ④ 数学家社会 — 异构智能体群落的自组织协作                         │
│     数百个角色各异的智能体，分工探索、实时通信、合力攻克难题          │
├─────────────────────────────────────────────────────────────────┤
│  ③ 世界模型 — 证明器的心智模拟器                                  │
│     内化 Lean4 状态转移动力学，不调用证明器即可预判策略效果           │
├─────────────────────────────────────────────────────────────────┤
│  ② 活知识系统 — 自演化的数学记忆                                  │
│     从海量证明经验中涌现层次化知识，具备遗忘、修正、重组能力          │
├─────────────────────────────────────────────────────────────────┤
│  ① 验证操作系统 — 无限弹性的算力基底                              │
│     毫秒级响应 · 弹性伸缩 · 增量编译 · 热缓存 · 持久化上下文        │
├─────────────────────────────────────────────────────────────────┤
│  ⓪ Lean4 REPL — 100% 精确的形式化验证器                          │
└─────────────────────────────────────────────────────────────────┘

                     ↑ 知识沉淀 ↓ 知识注入 — 飞轮自转
```

---

## 三、项目脊柱：一道定理的完整生命之旅

理解 AI4Math 最好的方式是追踪一道定理从输入到被证明的全过程。
**运行 `python run_single_lane.py` 可以逐步看到以下每个阶段的真实输出。**

```
用户输入一道定理
       │
       ▼
  ┌─────────────┐     ┌──────────────────────────────────────┐
  │ 读题 & 分析  │────▶│ 领域识别 · 难度评估 · 策略规划          │
  │ (Step 1,4)   │     │ DirectionPlanner.plan()              │
  └──────┬──────┘     └──────────────────────────────────────┘
         │
         ▼
  ┌─────────────┐     ┌──────────────────────────────────────┐
  │ 知识注入     │────▶│ 从知识金字塔提取:                      │
  │ (Step 3)    │     │   KnowledgeReader.render_for_prompt() │
  └──────┬──────┘     │   相关引理 · tactic 建议 · 错误模式     │
         │            └──────────────────────────────────────┘
         ▼
  ┌─────────────┐     ┌──────────────────────────────────────┐
  │ 多方向并行   │────▶│ AsyncAgentPool.run_parallel()         │
  │ (Step 5)    │     │   3-4 个异构方向同时探索               │
  └──────┬──────┘     │   各有独立模型、温度、角色              │
         │            └──────────────────────────────────────┘
         ▼
  ┌─────────────┐     ┌──────────────────────────────────────┐
  │ 三级验证     │────▶│ AsyncVerificationScheduler            │
  │ (Step 5)    │     │ L0: 语法预过滤 (~1μs) — 瞬间淘汰废案   │
  └──────┬──────┘     │ L1: REPL 快验 (~50ms) — 结构化反馈     │
         │            │ L2: 全编译   (~3s)    — 终极确认       │
         ▼            └──────────────────────────────────────┘
  ┌─────────────┐     ┌──────────────────────────────────────┐
  │ 策略决策     │────▶│ PolicyEngine.evaluate()               │
  │ (Step 5,7)  │     │  5 条可组合规则链:                     │
  └──────┬──────┘     │  InfraRecovery > ConsecutiveError >    │
         │            │  BudgetEscalation > Decompose > Reflect│
         ▼            └──────────────────────────────────────┘
  ┌─────────────┐     ┌──────────────────────────────────────┐
  │ Green Contract│───▶│ 验证分级合约 (Step 8):                 │
  │ (Phase 2)   │     │ NONE → SYNTAX_CLEAN → TACTIC_VALID    │
  └──────┬──────┘     │ → GOALS_CLOSED → FULL_COMPILE         │
         │            │ → SORRY_FREE                          │
         ▼            └──────────────────────────────────────┘
  ┌─────────────┐     ┌──────────────────────────────────────┐
  │ 知识沉淀     │────▶│ KnowledgeWriter.ingest_proof_result() │
  │ (Step 5)    │     │ KnowledgeBroadcaster.on_tactic_result()│
  └──────┬──────┘     └──────────────────────────────────────┘
         │
         ▼
  ┌─────────────┐     ┌──────────────────────────────────────┐
  │ 状态压缩     │────▶│ compress_proof_status() (Step 9):     │
  │ (Phase 2)   │     │ one-liner: [VERIFYING r3] best=...    │
  └─────────────┘     │ for_prompt: 注入下轮 LLM (< 200 tok)  │
                      └──────────────────────────────────────┘
         │
         ▼
    下一道题自动继承所有积累的知识 → 飞轮加速
```

---

## 四、项目关节：模块如何彼此连接

### 4.1 Lane 运行时 (`engine/lane/`, Phase 1+2 核心)

证明任务的完整生命周期管理——所有决策经过类型化状态机和策略引擎。

```
  LaneProofRunner.run(packet)    ← 主入口 (engine/lane/integration.py)
       │
       ├── ProofTaskStateMachine  ← 显式状态机 (task_state.py)
       │   CREATED → KNOWLEDGE_LOADING → GENERATING → VERIFYING
       │                                      ↑           ↓
       │                                      └── REPAIRING
       │                                            ↓
       │                          SUCCEEDED / FAILED / GIVEN_UP
       │
       ├── PolicyEngine           ← 可组合策略规则链 (policy.py)
       │   InfraRecovery > ConsecutiveError > BudgetEscalation > Decompose > Reflect
       │
       ├── RecoveryRegistry       ← 自动恢复配方 (recovery.py)
       │   REPL_CRASH → restart   API_ERROR → backoff   TIMEOUT → larger_timeout
       │
       ├── ProofEventBus          ← 类型化事件 (event_bus.py)
       │   task.created · task.generating · task.failure.* · task.succeeded
       │
       ├── GreenContract          ← 验证分级合约 (green_contract.py, Phase 2)
       │   NONE < SYNTAX_CLEAN < TACTIC_VALID < GOALS_CLOSED < SORRY_FREE
       │
       └── SummaryCompression     ← 状态压缩 (summary_compression.py, Phase 2)
           one_liner · for_prompt(200 tok) · to_dict()
```

### 4.2 验证操作系统 (`engine/`)

```
  AsyncVerificationScheduler
       │
       ├── L0: PreFilter          语法预过滤 (~1μs)
       ├── L1: AsyncLeanPool      REPL 快验 (~50ms)
       ├── L2: subprocess lean    全编译 (~3s)
       │
       ├── ErrorIntelligence      结构化错误诊断
       ├── BroadcastBus           跨方向实时通信
       └── Observability          全链路可观测
```

### 4.3 活知识系统 (`knowledge/`)

```
  知识金字塔 (SQLite + WAL)
  ├── L3: 直觉图谱  concept_nodes/edges
  ├── L2: 策略模式  strategy_patterns
  ├── L1: 战术知识  tactic/lemma/error
  └── L0: 原始轨迹  traces/trajectories

  写入: KnowledgeBroadcaster → KnowledgeWriter → SQLite
  读取: KnowledgeReader.render_for_prompt() → 注入 LLM prompt
  生命周期: KnowledgeEvolver.decay_tick() / gc_stale() / revive()
```

### 4.4 多智能体协作 (`agent/` + `prover/pipeline/`)

```
  DirectionPlanner.plan()
       │
       ├── automation    (低温, 纯 tactic 自动化)
       ├── structured    (中温, 归纳/结构化证明)
       ├── alternative   (高温, 替代路径)
       └── repair        (仅当有失败历史)
               │
               ▼
  AsyncAgentPool.run_parallel() → [AgentResult, ...]
               │
               ▼
  MetaController.evaluate(sm) → PolicyDecision
    (内部委托 PolicyEngine 的可组合规则链)
```

---

## 五、内置基准数据集 (6,826 道题)

| 数据集 | 题数 | 难度 | 说明 |
|--------|------|------|------|
| **builtin** | 5 | easy-medium | 内置冒烟测试 (**最小, 推荐首次使用**) |
| **miniF2F** | 488 | AMC → IMO | 领域内最广泛使用的基准 |
| **PutnamBench** | 672 | 大学竞赛 | 1962-2024 Putnam 竞赛题 |
| **ProofNet** | 360 | 本科数学 | 分析、代数、拓扑核心课程 |
| **FATE-M/H/X** | 350 | 本科→博士 | 抽象代数全难度覆盖 |
| **FormalMATH** | 5,560 | 混合 | 多领域多难度 |

```bash
# 最小评测 (5 题, ~30s mock / ~2min real)
bash eval.sh --benchmark builtin

# 中等评测 (每 benchmark 10 题)
bash eval.sh --real --quick

# 全量评测 (所有 benchmark)
bash eval.sh --real
```

---

## 六、项目结构速查

```
240+ 个源文件 · 45,000+ 行代码 · 88 项新增测试 · 7 大基准 6,826 道题
```

| 模块 | 行数 | 一句话定位 |
|------|------|--------------|
| `engine/` | ~12,000 | 验证 OS：REPL 池、三级验证、弹性伸缩、广播总线 |
| `engine/lane/` | ~2,500 | **Lane 运行时：状态机、策略引擎、恢复、Green Contract、状态压缩** |
| `knowledge/` | ~2,200 | 活知识系统：四层金字塔、读写管道、衰减遗忘 |
| `prover/` | ~7,600 | 证明编排：异步管线、修复、分解、代码生成、引理银行 |
| `agent/` | ~3,700 | 智能体层：11 种角色、策略控制(→PolicyEngine)、钩子、插件 |
| `common/` | ~500 | 共享类型：角色、预算、钩子协议 |
| `benchmarks/` | ~800 | 评测框架：7 大基准加载器、指标计算 |
| `tests/` | ~4,000 | 覆盖全部核心模块 (含 88 项 Phase 1+2 新增测试) |
| `data/` | 6,826 题 | miniF2F、PutnamBench、ProofNet、FATE、FormalMATH |

### 关键入口文件

| 文件 | 用途 |
|------|------|
| `run_single_lane.py` | **单题调试 — 逐步遍历全管线** (推荐) |
| `run_single.py` | 传统单题测试 (非 Lane 模式) |
| `run_eval.py` | 批量评测入口 |
| `eval.sh` | 一键评测脚本 |
| `engine/lane/integration.py` | **LaneProofRunner — Lane 证明循环主入口** |
| `agent/strategy/meta_controller.py` | MetaController → PolicyEngine 委托 |
| `prover/assembly.py` | 全系统组装器 |

---

## 七、SOTA 对比

### miniF2F-test (244 题)

| 方法 | Pass@32 | 类型 |
|------|---------|------|
| Goedel-Prover-V2-32B | **90.4%** | 全证明生成 |
| Kimina-Prover-72B | 84.0% | 全证明生成 |
| DeepSeek-Prover-V2-671B | 82.4% | 全证明生成 |
| **AI4Math (Claude Opus 4.6)** | **待测** | **Agent 平台** |

> **关于"待测"：** 上述 SOTA 的 pass@k 均为 Lean4 编译验证。AI4Math 评测需配置 Lean4 + Mathlib 环境并使用 `--lean` 模式。

---

## 八、Docker 真实 Lean4 验证

```bash
# 1. 构建 Lean4 + Mathlib 镜像 (首次约 30 分钟)
cd docker && docker compose build lean4-repl

# 2. 启动 Lean4 REPL 守护进程
docker compose up -d lean4-repl

# 3. 运行带真实验证的评测
docker compose run --rm agent \
  python run_eval.py \
    --benchmark builtin \
    --provider anthropic \
    --lean-mode real

# 4. 或一键全流程
docker compose run --rm agent bash eval.sh --real --lean
```

---

## 常见问题

**Q: 和 DeepSeek-Prover 的根本区别？**
它们是"更强的证明生成器"，AI4Math 是"让证明生成器在其中运行的操作系统"。

**Q: `run_single_lane.py` vs `run_single.py`？**
`run_single_lane.py` 使用新的 Lane 运行时 (Phase 1+2), 有完整的状态机、策略引擎、Green Contract 和状态压缩。`run_single.py` 是传统流程, 不经过 Lane 层。

**Q: 能支持 Coq / Isabelle 吗？**
REPL 交互通过 `Transport(ABC)` 抽象, 知识系统和智能体层不含 Lean4 特定代码。

**Q: Green Contract 是什么？**
借鉴 claw-code 的验证分级概念。每个证明验证结果不再是 pass/fail 的 1 bit, 而是分为 6 级: NONE → SYNTAX_CLEAN → TACTIC_VALID → GOALS_CLOSED → FULL_COMPILE → SORRY_FREE。策略引擎和 Dashboard 可以基于级别做精细决策。

**Q: Summary Compression 有什么用？**
将冗长的事件流压缩为: 当前阶段 + 上次成功检查点 + 当前阻塞 + 建议下一步。可注入 LLM context (< 200 token) 让模型了解当前进展, 也可用于监控和日志。

---

## License

MIT

## 引用

```bibtex
@software{ai4math2026,
  title   = {AI4Math: An Agent Operating System for Formal Theorem Proving},
  year    = {2026},
  url     = {https://github.com/ai4math/ai4math}
}
```
