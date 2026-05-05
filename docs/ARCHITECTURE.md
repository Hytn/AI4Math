# AI4Math 架构

给开发者读的导航文档。先读这一份再翻代码。终端用户的"怎么用"在
[`../README.md`](../README.md);零基础数学家从
[`../TUTORIAL_CN.md`](../TUTORIAL_CN.md) 开始。

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

## 项目核心

```
┌──────────────────────────────────────────────────────────────┐
│  核心三件                                                     │
│   ① 大一统推理       —  prover/unified/  + agent/runtime/    │
│   ② Lean 验证基础    —  engine/  (含 search/ 子系统)          │
│   ③ RL 采样          —  sampler/  (verl / slime / TRL adapter)│
│                                                              │
│  在以上三件之上,预留三个上层接口:                             │
│   A. 经验知识库      —  knowledge/ + lemma_bank/ + plugins/  │
│   B. 世界模型        —  engine/world_model.py                │
│   C. 多智能体广播    —  engine/broadcast.py                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 分层

代码按依赖方向自下而上分四层。下层不允许 import 上层,
`scripts/check_layers.py` 在 CI 里强制这条约束。

```
┌─────────────────────────────────────────────────────────────────┐
│  入口层 (run_*.py / eval.sh)                                    │
│    run_unified.py       单题, 标准入口                           │
│    run_eval.py          批量评测, 7 个基准                       │
│    eval.sh / scripts/rl_loop.sh                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  prover/      证明编排层                                         │
│    unified/   ★ UnifiedProofRunner + 14 个 Profile (核心 ①)      │
│    conjecture/ 主动猜想辅助引理 + 文本级 verifier                │
│    decompose/ 目标拆分                                           │
│    premise/   引理检索 (BM25 / TF-IDF / hybrid)                  │
│    verifier/  Sorry/Axiom 完整性检查                             │
│    lemma_bank/ 跨问题 SQLite + BM25 引理库 (预留 A 的 lemma 维度)│
│    plugins/   YAML-driven 领域插件 loader                        │
│    models.py  ProofTrace / BenchmarkProblem 等数据类型           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  agent/       智能体运行时层                                     │
│    runtime/   AgentLoop ★ (主入口, 单文件)                       │
│    brain/     LLM provider (AsyncClaude / AsyncOpenAI / Mock)    │
│    tools/     ToolRegistry + 9 个 builtin tool                   │
│    persistence/ dialog.json + SFT export + unified storage      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  knowledge/   预留接口 A — 活知识系统                            │
│    store.py   SQLite 存储 (Layer 0/1 已接通, Layer 2/3 预留)     │
│    reader.py / writer.py    主路径读写                           │
│    types.py   TacticEffectiveness / LemmaRecord / ...            │
│    dialog_index.py          跨问题 dialog 检索 (含 SQLite)       │
│    tfidf_retriever.py       项目内已证引理的轻量检索器           │
│    goal_normalizer.py       goal pattern 标准化                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  engine/      核心 ② — Lean 4 验证基础设施                       │
│    async_lean_pool.py       N 路并行 REPL 长连接 ★               │
│    transport.py             Local / Socket / Mock / Fallback     │
│    backends/                Kimina / Pantograph / LooKeng        │
│    error_intelligence.py    stderr → 结构化 AgentFeedback        │
│    summary_compressor.py    Lean 错误/反馈/广播压缩              │
│    _core.py                 classify_error + 共享数据类型        │
│    prefilter.py             L0 语法预过滤                        │
│    async_verification_scheduler.py  自适应 L0/L1/L2 三级调度     │
│    proof_context_store.py   步级 trajectory 落盘 (SQLite)        │
│    protocols.py             对外协议 (AsyncPoolProtocol)         │
│    policy/    PolicyEngine + 5 内置规则 + 13 类失败分类          │
│                                                                  │
│    search/    ★ 搜索代数的唯一权威实现                           │
│       core.py      SearchNode / SearchTree / backprop / extract  │
│       policies.py  BestFirstPolicy / UCBPolicy / BeamPolicy      │
│       runner.py    run_search 调度循环                           │
│                                                                  │
│  ── 预留接口 B/C 也在 engine/ 这一层 ──                          │
│    world_model.py           B: 世界模型 (tactic-success-prior)   │
│    world_model_trainer.py   B: sklearn 训练管道                  │
│    broadcast.py             C: 多智能体 rollout 广播总线         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                         Lean 4 + Mathlib
```

`★` 标的是 hot path: `run_eval.py` 跑一道题, 字节流走的就是
`AgentLoop` → `AsyncLeanPool` → `Lean4 REPL`。

### 关于 sampler/ 与 prover/unified/ 的关系

两条路径**共享 `engine/search/` 这套搜索代数**(SearchNode / 选择策略 /
backprop / 调度循环)。差异只在 expansion:

  * `prover/unified/` 的 `_DriverWrapper` 把 expander = 跑一次 AgentLoop
  * `sampler/tree_rollout_sampler.py` 把 expander = 调一次 policy_fn 加
    一次 ProofEnv.step

两端的"算法部分"完全不重复。对老调用方,
`prover.unified.search_driver.SharedSearchState/TreeNode/BestFirstDriver/
UCBDriver/BeamDriver/make_driver` 的所有接口保留为薄壳别名,无破坏性变化。

---

## 数据流: 一道题怎么被证明

```
1. run_eval.py / run_unified.py
   └─ 解析 --profile  → get_profile(name) → Profile 对象
   └─ build_lean_pool(args) → AsyncLeanPool
   └─ load_world_model / load_dialog_index / load_knowledge
      load_policy_engine / load_plugin_loader / load_persistent_lemma_bank
      (共享 prover.unified.factory 的工厂函数)

2. UnifiedProofRunner.run(problem, profile)
   ├─ build_tool_registry(profile)        # 按 ToolKit 装配 tool 实例
   ├─ render_system_prompt(profile.framing)
   ├─ _build_initial_message(problem)     # 题目 + 检索引理 + few-shot
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
       └─ dialog.json (含 messages + meta + result + search_tree)

4. (可选) 知识沉淀
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
| `heterogeneous` | 4 路异构并行 + 共享 broadcast bus | 4 sub-profile + `broadcast` | 4 |
| `conjecture_driven` | 主动猜想辅助引理 | `[conjecture_propose, premise, lemma_bank, decompose]` | 15 |
| `kimina_batch` | Kimina Lean Server | `[lean_verify]` | 6 |
| `pantograph_dsp` | Pantograph (mvar + drafting) | `[lean_verify]` | 10 |
| `lookeng_lemma` | LooKeng (stateless lemma) | `[lemma_by_lemma, lean_verify]` | 20 |
| `nfl_hybrid` | NFL 混合 | `[lean_verify, premise]` | 8 |
| `mcts` | MCTS + UCB1 树搜索 | `[tactic_apply, goal_inspect]` | (节点数计) |
| `best_first` | best-first 树搜索 | `[tactic_apply, goal_inspect]` | (节点数计) |
| `beam` | beam search | `[tactic_apply, goal_inspect]` | (beam 计) |

**加新算法 = 在 `prover/unified/profiles.py` 的 `PRESETS` 里加一项**,
不动 `runner.py`, `agent_loop.py`, `tools/` 任何文件。

---

## 三个预留接口的状态

| 接口 | 主路径接通 | 详情 |
|---|---|---|
| **A. 知识库** | Layer 0/1 接通 | 每次 verify 写 `tactic_effectiveness`,跨问题 dialog 注入。Layer 2/3 (strategy / concept_graph) schema 在但写入路径未实装 — 留给真实数据驱动的决定。 |
| **B. 世界模型** | tool 调用前可选 gate | `tactic_apply` 接受 `world_model=` 参数,命中时短路掉 `p_success < threshold` 的 tactic。`MockWorldModel` = 规则启发式;`SklearnWorldModel` = LR + 手工特征。要替换为 transformer 编码器一行接口。 |
| **C. 广播总线** | heterogeneous 内部接通 | `_run_parallel` 把 `BROADCAST` 工具注入每个 sub-profile,bus 在 4 个 sub-runner 间共享。一个 sub-profile 写"avoid: ring 在 ℕ 减法上无效",其他三个 read 到。下一步:跨 problem 的全局 bus 需要持久化层。 |

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
| 看搜索代数(节点 / selection / backprop) | `engine/search/` |
| 看错误如何变成 ~100 bits | `engine/error_intelligence.py` + `engine/_core.classify_error` |
| 看完整性反作弊检查 | `prover/verifier/integrity_checker.py` |
| 看知识库 schema (预留 A) | `knowledge/store.py::_KNOWLEDGE_SCHEMA` |
| 看世界模型接口 (预留 B) | `engine/world_model.py::make_world_model` |
| 看广播总线接通 (预留 C) | `prover/unified/runner.py::_run_parallel` |
| 看 dialog.json schema | `agent/persistence/dialog_format.py` |
| 看 RL 飞轮怎么串 | `scripts/rl_pipeline.py` + `scripts/rl_loop.sh` |
| 入口共享工厂 | `prover/unified/factory.py` |

---

## 命名诚实性说明

* **"World model"** 是 tactic-success-rate prior。`MockWorldModel` = 30 条
  正则,`SklearnWorldModel` = LogisticRegression on TF-IDF。它**不**模拟
  Lean kernel 的状态转移。要换成 transformer 编码器,接口一行不变。
* **`reprover` profile 的 RAG** 用的是 word TF-IDF + char n-gram TF-IDF 加权,
  **不是** dense neural retriever。要换成 SBERT/ColBERT,替换
  `prover.premise.embedding_retriever` 的 `_vectorize` 即可。
* **`mathlib_core.jsonl` 当前只有 ~334 条**,真实 Mathlib4 是 10⁵ 量级。
  `scripts/export_mathlib_premises.py` 提供从本地 Mathlib repo 扩池的脚本,
  评测前请先扩池,否则任何检索器的 recall 都被这个数字物理封死。
* **知识金字塔 Layer 2/3** 只有 schema 没有调用方,等真实数据进来再决定
  是否实装。

---

## 工作目录约定

  * `results/unified/<problem_id>/` — `run_unified.py` 单题输出
  * `results/traces/<problem_id>/`  — `run_eval.py` 批量评测输出
  * `results/rl/iter_<N>/`          — `scripts/rl_pipeline.py` RL 飞轮迭代输出
