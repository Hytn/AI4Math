# AI4Math 架构 (v13)

一份给开发者读的导航文档。先看完这一份再去翻代码。
对终端用户的"怎么用"在 [`README.md`](../README.md);
零基础数学家从 [`../TUTORIAL_CN.md`](../TUTORIAL_CN.md) 开始。

---

## 项目核心定位

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  这个智能体的核心 = 三件事:                                         │
│                                                                    │
│   ① 大一统所有推理方式      —  prover/unified/  (14 profile)        │
│   ② Lean 4 验证基础设施      —  engine/         (async pool +      │
│                                                  4-backend +       │
│                                                  error intelligence)│
│   ③ RL infra 接口            —  sampler/        (verl / slime /    │
│                                                  vLLM)             │
│                                                                    │
│  在以上三件核心之上, 预留三个上层功能接口:                              │
│                                                                    │
│   A. 经验知识库的管理检索与归纳总结 — knowledge/                       │
│   B. 世界模型 (tactic-success-prior) — engine/world_model.py        │
│   C. 多智能体 rollout 实时广播       — engine/broadcast.py +         │
│      (数学家 community)                BroadcastTool                │
│                                                                    │
│  v13: 不属于这两组的东西全部清理掉。                                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 一句话

**所有定理证明算法收敛到一个 `AgentLoop`。换算法 = 换 `Profile`。**

```
problem ──┐
profile ──┼──→ UnifiedProofRunner.run()  ──→  dialog.json
LLM    ──┤
LeanPool ─┘
```

`Profile` 是 5 个声明式开关的组合 (`tools` / `max_turns` / `framing` /
`temperature` / `observation`)。换 `--profile whole_proof_repair` 跑
DeepSeek-Prover 风格,换 `--profile reprover` 跑 ReProver 风格,换
`--profile heterogeneous` 跑 4 路并行,换 `--profile mcts` 跑树搜索。
没有任何"如果是 X 算法就走 X 入口"的硬编码。

---

## 分层

代码按依赖方向自下而上分四层。下层不允许 import 上层。
`scripts/check_layers.py` 在 CI 里强制这条约束。

```
┌─────────────────────────────────────────────────────────────────┐
│  入口层 (run_*.py / eval.sh)                                    │
│    run_unified.py        单题, 标准入口                         │
│    run_eval.py           批量评测, 7 个基准                     │
│    eval.sh / setup_and_eval.sh / scripts/rl_loop.sh             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  prover/      证明编排层                                         │
│    unified/   ★ UnifiedProofRunner + 14 个 Profile (核心 ①)     │
│    conjecture/ 主动猜想辅助引理 + 文本级 verifier               │
│    decompose/ 目标拆分 (DSP profile 用)                         │
│    premise/   引理检索 (BM25 / TF-IDF / dense)                  │
│    verifier/  Sorry/Axiom 完整性检查                            │
│    lemma_bank/ 跨问题 SQLite + BM25 引理库 (v14 项③, 预留 A)    │
│    plugins/   YAML-driven 领域插件 loader (v14 项④)             │
│    models.py  ProofTrace / BenchmarkProblem 等数据类型           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  agent/       智能体运行时层                                     │
│    runtime/   AgentLoop ★ (主入口, ~520 行, 单文件; v14 项②接通) │
│    brain/     LLM provider (AsyncClaude / AsyncMock + 缓存装饰器)│
│    tools/     ToolRegistry + 9 个 builtin tool                   │
│    persistence/ dialog.json + SFT export + unified storage      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  knowledge/   预留接口 A — 活知识系统                            │
│    store.py   四层金字塔 (raw → tactic → strategy → concept)     │
│    reader.py  / writer.py     主路径读写                         │
│    types.py   TacticEffectiveness / LemmaRecord / ...            │
│    dialog_index.py            跨问题 dialog 检索 (含 SQLite)     │
│    tfidf_retriever.py         项目内已证引理的轻量检索器         │
│    goal_normalizer.py         goal pattern 标准化                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  engine/      核心 ② — Lean 4 验证 OS                            │
│    async_lean_pool.py         N 路并行 REPL 长连接 ★             │
│    transport.py               LocalTransport / SocketTransport   │
│                               / MockTransport / FallbackTransport│
│    backends/                  Kimina / Pantograph / LooKeng      │
│    error_intelligence.py      stderr → 结构化 AgentFeedback      │
│    summary_compressor.py      Lean 错误/反馈/广播压缩 (v14 项①)  │
│    _core.py                   classify_error + 共享数据类型       │
│    prefilter.py               L0 语法预过滤 (~1μs)                │
│    async_verification_scheduler.py  自适应 L0/L1/L2 三级调度      │
│    proof_context_store.py     步级 trajectory 落盘 (SQLite)       │
│    protocols.py               对外协议 (AsyncPoolProtocol)        │
│    observability_stub.py      no-op metrics shim                 │
│    policy/  task_state + recovery + engine (v14 项②)            │
│             声明式策略规则引擎 + 13 类 ProofFailureClass         │
│                                                                  │
│  ── 预留接口 B/C 也在 engine/ 这一层 (它们的接口而非依赖) ──        │
│    world_model.py             B: 世界模型 (tactic-success-prior) │
│    world_model_trainer.py     B: sklearn 训练管道                 │
│    broadcast.py               C: 多智能体 rollout 广播总线        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                         Lean 4 + Mathlib
```

`★` 标的是 hot path: `run_eval.py` 跑一道题, 字节流走的就是
`AgentLoop` → `AsyncLeanPool` → `Lean4 REPL`。

> **关于 `sampler/` (核心 ③)**: RL 采样路径目前是与 agent 主路径平行的
> 一份实现 (`ProofEnv` / `BaseSampler` / `TreeRolloutSampler`), 不在上面
> 这棵主依赖树里。它复用 `engine/` 的 lean pool 和 backends, 但搜索
> 算法 (best_first/UCB/beam) 和多轮循环都是另写一遍。把这两条收敛
> 到同一个 `SharedSearchState` 是 v15 的目标 (~900 行重复消除)。

---

## 数据流: 一道题怎么被证明

```
1. run_eval.py / run_unified.py
   └─ 解析 --profile  → get_profile(name) → Profile 对象
   └─ build_lean_pool(args) → AsyncLeanPool
   └─ load_world_model / load_dialog_index / load_knowledge
      (共享 prover.unified.factory 的工厂函数)

2. UnifiedProofRunner.run(problem, profile)
   ├─ build_tool_registry(profile)        # 按 ToolKit 装配 tool 实例
   ├─ render_system_prompt(profile.framing) # 选择 framing prompt
   ├─ _build_initial_message(problem)       # 题目 + 检索引理 + few-shot
   └─ AgentLoop.run(system_prompt, initial_message)
       └─ 多轮:
          LLM.chat(messages, tools=registry.specs)
          ↓
          parse tool_use → ToolRegistry.execute(tool_name, args)
          ↓ (例: lean_verify)
          AsyncLeanPool.verify_complete(theorem, proof)
          ↓
          IntegrityChecker.check_integrity(code)
          ↓
          AgentFeedback (~100 bits)  ← error_intelligence 结构化
          ↓
          inject as tool_result message
          ↓ goto LLM.chat (下一轮)

3. LoopResult / UnifiedResult
   └─ result.save_unified("results/traces/<id>")
       └─ dialog.json (schema v3.0, 含 messages + meta + result + search_tree)

4. (可选) 知识沉淀 — 预留接口 A
   └─ knowledge.writer.deposit(dialog) → SQLite
       下题的 prompt 自动注入相似已成功 dialog
```

---

## 14 个 Profile

| Profile | 对应方法 | tools | max_turns |
|---|---|---|---|
| `whole_proof` | DeepSeek-Prover / Kimina / Goedel | `[]` | 1 |
| `whole_proof_repair` | 项目默认 (compile-and-fix) | `[lean_verify]` | 6 |
| `dsp` | Draft-Sketch-Prove | `[decompose, premise, lean_verify]` | 10 |
| `reprover` | ReProver (RAG + step-level) | `[premise, tactic_apply, goal_inspect]` | 30 |
| `leandojo` | LeanDojo (纯 step-level + hammer) | `[tactic_apply, goal_inspect, lean_auto]` | 50 |
| `heterogeneous` | 4 路异构并行 + 共享 broadcast bus | 4 个 sub-profile + `broadcast` | 4 |
| `conjecture_driven` | 主动猜想辅助引理 | `[conjecture_propose, premise, lemma_bank, decompose]` | 15 |
| `kimina_batch` | Kimina Lean Server (高吞吐) | `[lean_verify]` (走 Kimina) | 6 |
| `pantograph_dsp` | Pantograph (mvar + drafting) | `[lean_verify]` (走 Pantograph) | 10 |
| `lookeng_lemma` | LooKeng (stateless lemma) | `[lemma_by_lemma, lean_verify]` | 20 |
| `nfl_hybrid` | NFL 混合 | `[lean_verify, premise]` | 8 |
| `mcts` | MCTS + UCB1 树搜索 | `[tactic_apply, goal_inspect]` | (节点数计) |
| `best_first` | best-first 树搜索 | `[tactic_apply, goal_inspect]` | (节点数计) |
| `beam` | beam search | `[tactic_apply, goal_inspect]` | (beam 计) |

**加新算法 = 在 `prover/unified/profiles.py` 的 `PRESETS` 里加一项**,
不动 `runner.py`, `agent_loop.py`, `tools/` 任何文件。

---

## 三个预留接口的当前状态

| 接口 | 主路径接通 | 详情 |
|---|---|---|
| **A. 知识库** | ✅ Layer 0/1 接通 | 每次 verify 写 `tactic_effectiveness`, 跨问题 dialog 注入。Layer 2/3 (strategy / concept_graph) schema 在但写入路径未实装 — 留给未来按真实数据决定要不要建。|
| **B. 世界模型** | ✅ tool 调用前可选 gate | `tactic_apply` 接受 `world_model=` 参数, 命中时短路掉 `p_success < threshold` 的 tactic。MockWorldModel = 规则启发式; SklearnWorldModel = LR + 手工特征。要替换为 transformer 编码器一行接口。|
| **C. 广播总线** | ✅ v13 接通 heterogeneous | v13 修了 `_run_parallel` 的 `_augment(sp)` 把 `BROADCAST` 注入每个 sub-profile, bus 在所有 sub-runner 共享。一个 sub-profile 写"avoid: ring 在 ℕ 减法上无效", 其他三个 read 到。下一步: 跨 problem (数学家 community) 的全局 bus —— 需要持久化 + 鉴权层。|

---

## 关键文件速查

| 我想... | 看哪个文件 |
|---|---|
| 看一遍数据流主线 | `prover/unified/runner.py::UnifiedProofRunner.run` |
| 看 LLM↔tool 多轮怎么实现的 | `agent/runtime/agent_loop.py::AgentLoop.run` |
| 加一个新 profile | `prover/unified/profiles.py::PRESETS` |
| 加一个新 tool | `agent/tools/builtin/` 加一个文件 + `tool_kits.py` 注册 |
| 加一个新 framing | `prover/unified/system_prompts.py::FRAMING_PROMPTS` |
| 看 Lean REPL 池 | `engine/async_lean_pool.py` |
| 看错误如何变成 ~100 bits | `engine/error_intelligence.py` + `engine/_core.classify_error` |
| 看完整性反作弊检查 | `prover/verifier/integrity_checker.py` |
| 看知识库 schema (预留 A) | `knowledge/store.py::_KNOWLEDGE_SCHEMA` |
| 看世界模型接口 (预留 B) | `engine/world_model.py::make_world_model` |
| 看广播总线接通 (预留 C) | `prover/unified/runner.py::_run_parallel::_augment` |
| 看 dialog.json schema | `agent/persistence/dialog_format.py` |
| 看 RL 飞轮怎么串 | `scripts/rl_pipeline.py` + `scripts/rl_loop.sh` |
| 加一个新 benchmark | `benchmarks/loader.py::load_benchmark` 加分支 |
| 入口共享工厂 | `prover/unified/factory.py` |

---

## v13 相对 v12 的差异

完整记录见 [`CHANGELOG.md`](CHANGELOG.md) V13 段。摘要:

**修了又一个 latent bug** (与 v10/v11/v12 累计 8 个同模式的第 9 个):
`prover/decompose/goal_decomposer.py::decompose` 是 sync 函数, 内部
`self.llm.generate(...)` 在 `AsyncLLMProvider` 下是 async 返 coroutine,
`resp.content` 必触 AttributeError。`dsp` / `pantograph_dsp` /
`conjecture_driven` 三个 profile 在 anthropic provider 下从 v3 起一直
跑不通。v13 改 async + iscoroutine 兼容。

**heterogeneous broadcast 第一次真接通**: v6 的 `BroadcastBus` +
`BroadcastTool` (~500 行) 一直在主路径死代码; v13 在 `_run_parallel` 用
`_augment(sp)` 把 `BROADCAST` 注入每个 sub-profile 的 tools 列表 + bus
在所有 sub-runner 共享。

**ConjectureVerifier 主路径接通**: v12 之前 `ConjectureProposeTool` 用
`verify=False` 主动绕过 verifier (因为 `_type_check` 调一个不存在的
`.compile()` API)。v13 删除 `_type_check`, 把 verifier 精简为纯文本级
过滤, 切回 `verify=True`。`a = a` 这种平凡 conjecture 现在会被丢掉。

**死代码大扫除** (~6,500 行 Python + ~6,300 行 HTML/SVG/重复 doc):
整文件删 `index.html` (116KB 营销页) / `docs/architecture.{html,svg}` /
3 份 `CLEANUP_*.md` (合进 CHANGELOG) / `engine/lean3_to_lean4.py` /
`engine/repl_protocol.py` / `common/prompt_builder.py`。模块精简
`common/roles.py` (174 → 33) / `engine/protocols.py` (110 → 35) /
`prover/conjecture/conjecture_verifier.py` (192 → 110) /
`config/{default.yaml,schema.py}` (323 → 157)。0 调用方 alias 全删。

**`tests/test_smoke_v13.py`** (20 条断言): 钉每一处修改 + 钉每一个被删
文件 + 钉 14 profile + 三个预留接口 (knowledge / world_model / broadcast)
都可 import。回归立即被 CI 抓到。

**测试**:
```
v12 baseline: 749 passed, 1 skipped, 0 failed
v13 final:    760 passed, 1 skipped, 0 failed   (零回归)
```

---

## v14 — 4 项备胎回归并接通主路径

完整记录见 [`CHANGELOG.md`](CHANGELOG.md) V14 段。摘要:

v13 砍掉死代码后,我们重新审视了初版仓库里所有"未接通备胎",筛选出 4 项
**对应真实问题**且**契合三件核心 + 三个预留接口**的模块,回归到 v14 主路径。

| 项 | 模块 | 解决的真实问题 | 主路径接通点 |
|---|---|---|---|
| ① | `engine/summary_compressor.py` | 多轮 repair / 4 路广播 token 爆炸 | `LoopConfig.compress_tool_results` 默认开 + `BroadcastTool` 收发都压缩 |
| ② | `engine/policy/` (~880 行) | agent_loop 终止条件硬编码,无法 declarative | `AgentLoop(policy_engine=)` + `_evaluate_policy` 每轮失败后跑 |
| ③ | `prover/lemma_bank/` (643 行) | 预留接口 A 知识库的 lemma 维度缺 deposit 调用方 | `LemmaBankTool` BM25 fallback + `ConjectureProposeTool` 后置写入 |
| ④ | `prover/plugins/` + 3 领域数据 | 领域 bug (ℕ减法等) 反复出现,framing 是 profile-level 不够细 | `_build_initial_message` 注入领域 few-shot/premises/hint |

**未做** (按计划等 #5 工程师跑出真实 pass@k 数字后再决策,见下方):

- ⑤ 长上下文压缩 — 基于 Anthropic prompt caching 重写,不直接回归
- ⑥ DirectionPlanner — heterogeneous 真低于 best-of-4 时才做
- ⑦ 规则化修复 — 某类错误占失败 >40% 才做

测试: `760 → 786 passed (+26 v14 smoke, 零回归)`。

---

## v15 的目标 (需要真 benchmark 数据才能决策的事项)

- **统一 RL 路径与 agent 路径**: 把 `sampler/tree_rollout_sampler.py` 里平行
  重写的搜索算法收敛到 `prover/unified/search_driver.py`; 把 `ProofEnv.step`
  改成调单步 `AgentLoop`。这是 ~900 行重复消除。实现路径: 把
  `SharedSearchState` + `BestFirst/UCB/Beam Driver` 提到 `engine/` (它们是纯
  数据 + 纯算法, 不依赖 LLM/agent), 让 prover/ 和 sampler/ 都从那里 import。
- **重构 `GoalInspectTool`**: 不再做整证重编译, 改用 REPL 的查询命令 (毫秒
  级而非秒级)。`leandojo` / `reprover` profile 的实际瓶颈。
- **跑出真实 pass@k**: 评测 miniF2F-test + Claude Sonnet 4 + Kimina backend,
  给出 pass@8/32 的真数字, 写进 README。这是项目从"基础设施"上升到"研究"
  的临界点 —— 也是发现下一波 latent bug 的唯一办法。
- 基于 pass@k 决策 v14 留下的 ⑤/⑥/⑦ 是否需要做。
