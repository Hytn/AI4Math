# AI4Math 改进计划 (v14 → v15+)

5 名工程师 · 4 Sprint · 8-10 周。

每个 Sprint 末有验收门 (gate) — 不达标的事项推到下个 Sprint, 不堆积。

---

## 项目核心定位 (再次明确, 防止跑偏)

```
核心三件                                  预留三个接口
─────────                                 ─────────────
① 大一统推理 (14 Profile)                  A. 经验知识库 (knowledge/ + lemma_bank/ + plugins/)
② Lean 验证基础设施 (engine/)              B. 世界模型 (engine/world_model.py)
③ RL infra (sampler/, verl/slime/vLLM)    C. 多智能体广播 (engine/broadcast.py)
```

任何 Sprint 任务必须能在这张图里指出对应位置。**不能定位的任务排除掉**。

---

## Sprint 0 (本次 v14 已完成)

| 完成项 | 接通点 |
|---|---|
| ① engine/summary_compressor.py | LoopConfig.compress_tool_results 默认开 + BroadcastTool 收发都压缩 |
| ② engine/policy/ (规则引擎) | AgentLoop(policy_engine=) + UnifiedProofRunner 透传 |
| ③ prover/lemma_bank/ (SQLite + BM25) | LemmaBankTool fallback + ConjectureProposeTool 后置写入 |
| ④ prover/plugins/ + 3 领域数据 | _build_initial_message 注入 few-shot/premises/hint |

测试: 760 → 786 passed, 零回归。

---

## Sprint 1 — 真实评测 + 反馈循环建立 (3 周)

**目标**: 项目历史上**第一个真实 pass@k 数字**。所有后续 Sprint 的优先级都靠这个数字驱动。

| 工程师 | 任务 | 产出 | 验收 |
|---|---|---|---|
| **#1** | 在 `whole_proof_repair` profile 上验证项①压缩生效 | `dialog.json` 里 tool_result 平均长度 < 1500 字符 | 8 轮 repair 不超 context window |
| **#2** | 给 3 个最常崩的 profile (`whole_proof_repair` / `dsp` / `heterogeneous`) 写默认 PolicyEngine 规则集 | `engine/policy/profiles.py` 含 3 套 ``ProfilePolicy`` | profile-specific 规则触发率 > 0 |
| **#3** | 扩 `plugins/strategies/*/premises.jsonl` 各到 ≥ 50 条 + 加 4 个领域 (combinatorics / topology / linear-algebra / category-theory) | 7 个领域目录, ≥ 350 条 premises | `PluginLoader.match` hit rate ≥ 80% on miniF2F |
| **#4** | **重写**长上下文压缩 (基于 Anthropic prompt caching API), 专门服务 `reprover` / `leandojo` 30+ 轮 profile | `engine/context_cache.py` (新, ~250 行) + `LoopConfig.use_prompt_cache=True` | reprover 30 轮总 token cost 降 30%+ |
| **#5 (PM)** | **跑 miniF2F-test × Sonnet 4.7 × Kimina × pass@8** 完整评测 | `results/v14_baseline.json` + README pass@k 表 | 拿到首个真实数字 |

**Sprint 1 验收门** (硬性要求):
1. `results/v14_baseline.json` 存在且包含 244 题 × pass@1/4/8 的真实数字
2. 测试 ≥ 800 (新增项一定带 smoke test)
3. v14 接通点全部不变 (test_smoke_v14 全过)

未达标的任务推 Sprint 2,**不阻塞下个 Sprint 启动**。

---

## Sprint 2 — Sprint 1 反馈驱动的优化 (2 周)

#5 的 pass@k 数字进来后, 拿数据决定其他人做什么:

### 数据驱动决策矩阵

| 真实观察 | 行动 | 负责人 |
|---|---|---|
| context 爆炸是首要失败模式 (>30%) | 放大项①: 把 budget 调到 800 + 走 prompt caching | #1+#4 |
| 重复同错误 / 死循环失败模式 (>20%) | 放大项②: 加 ``DeadLoopRule`` + 触发后切 framing | #2 |
| `lemma_bank.search` hit rate 低 (<10%) | 放大项③: 把 `LemmaExtractor` 接到 `lean_verify` 后置, 真把每次成功的 have-step 写库 | #3 |
| 领域插件注入 hit rate 低 | 放大项④: 关键词匹配换成 `engine/world_model.py` 的特征向量 | #3 |
| `heterogeneous` pass@8 < `whole_proof_repair` × 4 (best-of-4) | **回归 ⑥ DirectionPlanner**, 用 LLM 动态规划 sub-profile, 不再写死 | #2+#3 |
| 某类错误 (TYPE_MISMATCH 等) 占失败 >40% | **回归 ⑦ 规则化修复**, 写对应 RepairStrategy | #2 |

### 工程师 #5 在 Sprint 2 做什么

启动 RL 飞轮 — 用 Sprint 1 累积的 ~5000+ dialog.json 真做一次 SFT/GRPO 训练, 验证 `sampler/verl_sampler.py` 整条管线在生产负载下的稳定性。这是把"基础设施"变成"研究"的关键一步。

**Sprint 2 验收门**:
1. 至少一个数据驱动决策被 ship (上面矩阵的某一项的"行动"列做完且回归测试不破)
2. RL 训练管线有第一个 SFT loss curve 输出 (不要求 SOTA, 要求"跑通")
3. README pass@k 表更新为 Sprint 2 后的新数字

---

## Sprint 3 — 大一统的最后一块 (2 周)

把 `sampler/` 的搜索算法和 `prover/unified/` 的搜索算法合并 — 这是 v15 路线图最大的工程项,~900 行重复消除。**只在 Sprint 1+2 的数据已稳定后才做**(因为重构会动 hot path)。

| 工程师 | 任务 |
|---|---|
| **#1** | 把 `SharedSearchState` + `BestFirst/UCB/Beam Driver` 提到 `engine/search/` (新目录) |
| **#2** | 改 `prover/unified/search_driver.py` 从 engine/search/ import |
| **#3** | 改 `sampler/tree_rollout_sampler.py` 从 engine/search/ import + 删 `_Node` 那套并行实现 |
| **#4** | 改 `sampler/proof_env.py::step` 内部调用单步 `AgentLoop`, 不再是平行重写 |
| **#5** | 跑回归: Sprint 1 的 pass@k 数字必须不退 (验证大一统不破坏推理) |

**Sprint 3 验收门**:
1. `engine/search/` 唯一权威实现, 其他文件全部 import
2. 测试 ≥ 800, smoke v14 全过, 新加 ``test_smoke_v15.py`` 钉合并不变性
3. pass@k 数字 ≥ Sprint 2 baseline (允许 ±2%)

---

## Sprint 4 — `GoalInspectTool` REPL 单步查询 + 数学家社区雏形 (3 周)

| 工程师 | 任务 |
|---|---|
| **#1** | 重构 `GoalInspectTool` 走 Lean 4 REPL 原生查询 (毫秒级而非秒级整证重编译) |
| **#2** | `engine/policy/` 加 `RecoveryRecipe` 真实现 (REPL crash → restart_session, timeout → reduce_timeout) |
| **#3** | 把 `BroadcastBus` 持久化到 SQLite, 跨 problem / 跨 run 共享 — **预留接口 C 的"数学家 community"第一次跨 problem** |
| **#4** | `engine/context_cache.py` 接到 `heterogeneous` profile, 4 路共享 cache prefix |
| **#5** | 在 PutnamBench 上重测 pass@8 (不只是 miniF2F-test), 看跨数据集稳定性 |

**Sprint 4 验收门**:
1. `leandojo` profile 单步 verify 平均 < 100ms
2. 持久化 BroadcastBus 在 dialog.json 里有跨 run 引用证据
3. PutnamBench pass@8 数字 + 与 miniF2F 数字的对比分析

---

## 决策原则 (写明白防止跑偏)

### 加新功能的判断标准 (按重要性排序)

1. **对应一个真实问题** — Sprint 数据里能指出"这个问题导致了 N% 的失败"
2. **契合三件核心或三个预留接口之一** — 在架构图里指出位置
3. **能在 1 个 Sprint 内做完并接通到主路径** — 不留"基础设施备胎"
4. **有人能写 smoke test 钉住接通点不回退** — 测试是接通的最终证据

### 不做的判断标准

1. 看起来很高级但 Sprint 数据没验证有用 (典型: world model 升级到 transformer)
2. 重写已经工作的代码,且没有数据证明现版本是瓶颈
3. 引入新依赖且不能立刻 retire 旧依赖
4. 抽象层超过两层(直接 prover → engine 比 prover → adapter → factory → engine 好)

### 做了但发现错了怎么办

1. 立刻在下一个 Sprint 回退到上一版,不"留作未来用"
2. 写一条 CHANGELOG entry 说明回退原因 + 数据
3. 在对应 smoke test 里写一条 anti-regression 钉住"不要再做"

---

## 工程师角色

| # | 主线 | 长期负责 |
|---|---|---|
| **#1** | 核心 ② 验证基础设施 (engine/) | LeanPool, prefilter, error_intelligence, summary_compressor |
| **#2** | 核心 ① 大一统推理 (prover/unified/, agent/runtime/) | Profile 体系, AgentLoop, PolicyEngine |
| **#3** | 预留接口 A 知识库 (knowledge/, lemma_bank/, plugins/) | dialog_index, persistent_bank, domain plugins |
| **#4** | 长上下文 + 核心 ③ RL infra 长 profile (sampler/) | context_cache, reprover/leandojo, RL sampler |
| **#5** | PM + 评测 | 真实 benchmark, 数据驱动决策, README 真数字 |

每个 Sprint 末 #5 出"决策报告"指导下个 Sprint。

---

## 看板状态机

```
  [Backlog]   ← 提议 (任何人)
     ↓ Sprint 末 #5 评估
  [Picked]    ← 进了下个 Sprint
     ↓ 工程师写完代码 + smoke test
  [Done]      ← 接通到主路径, smoke 钉住
     ↓ Sprint 后 #5 验证
  [Verified]  ← 真实数据证明有效
     ↓ 数据再驱动
  [Owned]     ← 长期 owner 持续维护

错误路径:
  [Done] → [Reverted]   ← Sprint 数据证明错误, 立刻回退
```

---

## v15 终态愿景

1. **真实 pass@k 进 README**: miniF2F-test pass@8 ≥ 50%, PutnamBench pass@8 ≥ 30%
2. **核心 ③ RL infra 真用过**: 至少完成一次 SFT 训练 + 评测增益验证
3. **预留接口 A 真接通**: lemma_bank.search hit rate ≥ 20%, plugin.match hit rate ≥ 80%
4. **预留接口 C 跨 problem**: 持久化 BroadcastBus, 一道题的 lemma 真能帮另一道题
5. **代码量稳定**: ≤ 45,000 行 Python (v14 是 42K, 允许小幅增长但不再翻倍)
6. **测试 ≥ 1000**: 每一处主路径接通都有 smoke test 钉住
