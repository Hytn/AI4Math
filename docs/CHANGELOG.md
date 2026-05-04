# CHANGELOG

按版本倒序。每条只记**用户可见的变化**;实现细节请看对应 commit。
完整的设计讨论保留在 git 历史里(原 `INFRA_MERGE_V*_REPORT.md` 已归并到此处;
v8/v9/v10 的 `CLEANUP_*.md` 也归并到本文件 V13 段下方)。

---

## V14 — 备胎复活: 4 项未接通模块回归并真上主路径

v13 砍掉 ~6,500 行死代码后明确了"三件核心 + 三个预留接口"的项目定位。
v14 不再是清理,而是**反向操作**:在初版仓库里筛选 4 个高价值"未接通备胎",
回归并接通到主路径。每一项都做到「调用方真在用」,而不是再次成为基础设施
摆设。

**新增 4 项接通到主路径的功能**:

- **项① · `engine/summary_compressor.py`** (490 行回归 / 原 ``engine/lane/``)

  Lean 错误 + AgentFeedback + 跨方向广播消息的三个 LLM-readable 压缩器。压缩
  比 ~9× (实测 4520 chars → 396 chars)。
  - 接通点 1: ``agent/runtime/agent_loop.py::LoopConfig.compress_tool_results``
    默认 ``True``, ``compress_budget=1200``。每轮 verify 失败的 tool_result
    自动压缩再 inject 进下一轮 user message。
  - 接通点 2: ``BroadcastTool.execute`` 的 publish 单条 + get_recent batch
    都走压缩。4 路 sub-profile 累积广播不再塞爆 reader 的 context。

- **项② · `engine/policy/`** (~880 行回归 / 原 ``engine/lane/``)

  PolicyEngine + RecoveryRegistry + ProofTaskStateMachine 三件套。把 agent_loop
  里硬编码的"何时升级 / 何时切角色 / 何时放弃"挪到声明式 ``PolicyRule``,可
  组合可单独测。
  - 5 条内置规则: ``ConsecutiveSameErrorRule`` / ``BudgetEscalationRule`` /
    ``BankedLemmaDecomposeRule`` / ``ReflectionRule`` / ``InfraRecoveryRule``。
  - 13 类 ``ProofFailureClass`` (SYNTAX_ERROR/TYPE_MISMATCH/TIMEOUT/REPL_CRASH/
    INTEGRITY_VIOLATION/...) 把"错误类别"提到一等公民。
  - 接通点: ``AgentLoop(policy_engine=)`` 可选注入,``_evaluate_policy`` 在
    每轮失败后跑;``UnifiedProofRunner(policy_engine=)`` 透传。不传则保持
    v13 行为 (硬 max_turns 终止)。

- **项③ · `prover/lemma_bank/`** (整目录 643 行回归)

  跨问题/跨会话的 SQLite + BM25 引理库。直接对应**预留接口 A 知识库**的
  lemma 维度 — v13 的 ``knowledge/store.py`` 有 schema 但缺 deposit 调用方,
  v14 补上。
  - ``ProvedLemma`` / ``LemmaBank`` (内存版) / ``PersistentLemmaBank`` (SQLite +
    BM25) / ``LemmaExtractor`` (从失败证明里抽 have-step) / ``LemmaVerifier``。
  - 接通点 1: ``LemmaBankTool.persistent_bank`` fallback 路径 — knowledge_store
    没结果时走 BM25 检索跨问题库。
  - 接通点 2: ``ConjectureProposeTool.persistent_bank`` 后置写入 — 提议的
    conjecture 即使没证出来也写入,下次同类问题查到该 statement 时跳过重提议。
  - ``UnifiedProofRunner(persistent_lemma_bank=)`` + ``build_tool_registry``
    + ``_build_tool`` 三层透传完成。

- **项④ · `prover/plugins/` + `plugins/strategies/{algebra,analysis,number-theory}/`**

  YAML-driven 领域插件系统。``framing`` 是 profile-level (整套推理风格),
  ``plugin`` 是 problem-level (按定理领域注入额外引理 + few-shot + 战略提示)。
  - 3 个领域目录,每个含 ``plugin.yaml`` (关键词匹配 + 参数覆盖) +
    ``premises.jsonl`` (领域引理库) + ``few_shot.md`` (领域示范)。
    ``number-theory/few_shot.md`` 第 1 条规则就是 "Lean 4 ℕ 减法 truncates"
    — 项目里反复出现的真实 bug 来源,从硬编码挪成 declarative。
  - 接通点: ``UnifiedProofRunner.plugin_loader`` + ``_build_initial_message``
    在 problem 入口 ``PluginLoader.match(theorem)`` 拿得分最高的插件,把它的
    ``few_shot_examples`` / ``extra_premises`` / ``strategic_hint`` 注入到
    首条 user message。

**未做** (按原计划留给数据驱动决策, 等 #5 工程师跑出 pass@k 后):

- ⑤ 长上下文压缩 (基于 Anthropic prompt caching API 重写, 而非回归)
- ⑥ DirectionPlanner — 仅在 heterogeneous pass@8 显著低于 best-of-4 才回归
- ⑦ 规则化修复 — 仅在某类错误占失败的 >40% 才补对应 RepairStrategy

**测试**:

```
v13 baseline: 760 passed, 1 skipped, 0 failed
v14 final:    786 passed, 1 skipped, 0 failed   (+26 v14 smoke, 零回归)
```

新增 ``tests/test_smoke_v14.py`` 26 条断言。每条钉一处接通点,任何一处接通
被拆掉都会立即被 CI 抓到。

**向后兼容**:v14 三个新参数 (``policy_engine`` / ``persistent_lemma_bank``
/ ``plugin_loader``) 默认 ``None``,不传则 v13 行为完全不变。``TestBackwardCompat``
专门钉这条不变性。

**架构变化**:目录树新增

```
engine/
  policy/                  ← 新增 (项②)
    task_state.py          状态机 + 13 类 ProofFailureClass
    recovery.py            RecoveryRegistry
    engine.py              PolicyEngine + 5 内置规则
  summary_compressor.py    ← 新增 (项①)
prover/
  lemma_bank/              ← 新增 (项③)
    bank.py
    persistent_bank.py     SQLite + BM25
    lemma_extractor.py
    lemma_verifier.py
  plugins/                 ← 新增 (项④)
    loader.py
plugins/                   ← 新增 (项④数据)
  strategies/
    algebra/
    analysis/
    number-theory/
```

**代码量**:

```
v13: 39,801 行 Python
v14: ~42,200 行 Python  (回归 ~2,400 行 + 写 v14 smoke 测试 + 文档)
```

代码量增加但**功能数量增加更多**: 三个预留接口的 lemma 维度、领域维度、
策略维度都第一次有了真调用方。

---

## V13 — 死代码大扫除 + 又一个 latent bug + heterogeneous 真接通

**项目核心定位明确化**: 这个智能体的核心是 (1) 用 14 个 Profile 大一统所有
推理方式; (2) 把 Lean 4 验证 + 错误智能 + 多 backend 抽象成可编程基础设施;
(3) 暴露给 RL infra (verl/slime/vLLM)。在此基础上预留三个上层接口: 知识库、
世界模型、多智能体广播总线 (数学家 community)。**不属于这三件事 + 三个预留
接口的代码全部清掉**。

**修了又一个 latent bug** (跟 v10/v11/v12 修过的 8 个一样的"接口签名变了
但调用方没跟上 + try/except 把它藏起来"模式):

- `prover/decompose/goal_decomposer.py` 的 ``decompose`` 是 sync 函数, 但
  内部 ``self.llm.generate(...)`` 在 ``AsyncLLMProvider`` 下是 async, 返
  coroutine。下一行 ``resp.content`` 必触 ``AttributeError``. v12 给
  ``DecomposeSubgoalTool`` 加的 ``iscoroutine`` 防御只覆盖
  ``decomposer.decompose()`` 自身, 没覆盖内部的 ``llm.generate()``,
  所以 ``dsp`` / ``pantograph_dsp`` / ``conjecture_driven`` 三个 profile 在
  anthropic provider 下从 v3 起一直跑不通。v13 改 ``decompose`` 为 async +
  ``iscoroutine`` 兼容, 与 v11 修过的 ``ConjectureProposer.propose``
  对齐。``DecomposeSubgoalTool.execute`` 同步改 ``await
  decomposer.decompose(...)``。

**heterogeneous broadcast 第一次真接通主路径**: v6 引入 ``BroadcastBus`` +
``BroadcastTool`` (~500 行基础设施), 卖点是"4 路异构并行 + 跨方向共享发现"。
但 ``_run_parallel`` 实例化 sub-profile 用 ``sp.__dict__``, parent profile
的 ``ToolKit.BROADCAST`` 不传播 —— sub-profile 拿不到 BroadcastTool, 整个
广播子系统在主路径死代码, heterogeneous 实际只是 best-of-4。v13 在
``_run_parallel`` 里用 ``_augment(sp)`` 把 BROADCAST 注入每个 sub-profile
的 tools 列表, bus 在所有 sub-runner 之间共享 (``self.broadcast_bus``)。
现在一个 sub-profile 写"avoid: ring 在 ℕ 减法上无效", 其他三个能 read 到。
**三个预留功能接口里的"多智能体广播"第一次真活了**。

**ConjectureVerifier 主路径接通**: v12 之前 ``ConjectureProposeTool`` 用
``verify=False`` 主动绕过 verifier (因为 verifier 的 ``_type_check`` 调一个
不存在的 ``.compile()`` API, 任何调用必崩)。v13 删除 ``_type_check`` (那条
路本来 100% 死), 把 verifier 精简为纯文本级过滤 (parse-ability + 平凡性 +
与目标的 token-相关性), 切回 ``verify=True`` —— ``a = a`` 这种平凡 conjecture
会被丢掉, 不再喂给 LLM 当 distraction。

**死代码大扫除** (~6,500 行):

- 整文件删除:
  - ``docs/architecture.html`` + ``architecture.svg`` (内容与 ``ARCHITECTURE.md``
    重叠, 三份维护一份不维护)
  - ``docs/CLEANUP_v9.md`` + ``CLEANUP_v10.md`` + ``CLEANUP_SUMMARY.md`` (三份
    历次清理报告, 内容已合并进本 CHANGELOG, v8 摘要见下方"v8 摘要")
  - ``engine/lean3_to_lean4.py`` (113 行, v10 抽出"以备复用", 至今 0 主路径调用)
  - ``engine/repl_protocol.py`` (261 行, REPL wire-format types, 主路径不
    import, 仅 1 个测试用)
  - ``common/prompt_builder.py`` (109 行, 主路径只用 ``FEW_SHOT_EXAMPLES`` 一
    个常量; 该常量挪到 ``common/few_shot.py``, ``build_prompt`` /
    ``FIRST_ATTEMPT`` / ``RETRY`` 三个 helper 全部只在测试里调过)

- **保留 (一度误删, 已恢复)**:
  - ``index.html`` —— 这是 GitHub Pages 主页, 不是营销死代码。v13 内容
    已对齐到"三件核心 + 三个预留接口"框架, 数字对齐到 ``760 tests /
    1631 problems / 161 files``。``tests/test_smoke_v13.py`` 加了一条断言
    钉住它的存在 + 内容对齐, 防止后续清理再次误伤。

- 模块精简:
  - ``common/roles.py``: 174 → 33 行 (11 角色 + 11 ROLE_PROMPTS +
    ``MODEL_TIER_OVERRIDES`` 三层 → 2 个真在用的角色: ``DECOMPOSER``,
    ``CONJECTURE_PROPOSER``)。``get_role_prompt()`` 全仓 0 调用方, 删。
  - ``common/response_parser.py``: 26 → 22 行 (``extract_json`` /
    ``extract_sorry_blocks`` 0 主路径调用方, 删)
  - ``engine/protocols.py``: 110 → 35 行 (``PoolProtocol`` /
    ``VerifierProtocol`` / ``BroadcastProtocol`` 0 处用作类型注解, 删;
    保留实际被引用的 ``AsyncPoolProtocol``)
  - ``prover/conjecture/conjecture_verifier.py``: 192 → 110 行 (删旁路的
    ``_type_check``, 见上"主路径接通"段)
  - ``config/default.yaml``: 73 → 27 行 (主路径只读 2 个字段, 70% 字段是
    装饰用的 schema)
  - ``config/schema.py``: 252 → 130 行 (校验逻辑同步精简)

- 0 调用方的 alias / re-export:
  - ``agent.persistence.unified_storage.save_task_outputs`` /
    ``load_task_outputs`` (back-compat alias, v9 删了所有可能的旧调用方)
  - ``agent.persistence.dialog_adapters.from_session_messages`` (适配
    ``SessionData``, ``session_store.py`` v9 已删)

**入口/测试维护**:

- ``.github/workflows/ci.yml``: 移除 ``--ignore=tests/test_v[6-7]_*.py``
  —— v12 把这些测试重命名了, ``--ignore`` 路径已失效但 CI 跑过没人发现。
- ``tests/test_dialog_format.py`` / ``test_seven_items.py`` /
  ``test_fixes.py``: 跟随删除的模块去掉过时 import 与对应测试方法。
- ``tests/test_smoke_v13.py`` (20 条断言): 钉这次的每一处修改 —
  ``GoalDecomposer.decompose`` 必 async; ``DecomposeSubgoalTool`` 必
  ``await``; ``_run_parallel`` 必注入 BROADCAST; 每个被删文件必不再存在;
  核心架构的 14 profile + 三个预留接口 (knowledge / world_model / broadcast)
  必可 import。回归立即被 CI 抓到。

**测试结果**:

```
v12 baseline: 749 passed, 1 skipped, 0 failed
v13 final:    760 passed, 1 skipped, 0 failed   (+11 测试 = v13 smoke 16 - v12
                                                  的 build_prompt 测试 5)
```

**代码量**:

```
v12: 41,664 行 Python  +  ~10,500 行 HTML/SVG/markdown
v13: 35,200 行 Python  +    4,200 行 markdown   (净 -6,500 行 Python +
                                                 -6,300 行 HTML/SVG/重复 doc)
```

**未做 (留给 v14, 与 README "What this project does *not* do" 段一致)**:

- ``sampler/tree_rollout_sampler.py`` 与 ``prover/unified/search_driver.py``
  的 ~900 行搜索算法重复合并 —— 这是项目核心 (统一 AgentLoop) 的最后一块
  双轨。需要把 ``SharedSearchState`` + 三个 ``Driver`` 提到层级允许的位置
  (engine/ 是合理选择: 纯数据结构 + 纯算法, 不依赖 LLM/agent)。
- 真 ``ANTHROPIC_API_KEY`` × 真 Lean 4 REPL 跑 miniF2F-test pass@8 — 项目
  历史上**第一个真实 pass@k 数字**。这是发现下一波 latent bug 的唯一办法。

---

## V12 — 又三个 latent bug + 反作弊误伤修复 + LLM 缓存接通 + 死代码再删一轮

**又三个 latent bug**(同样的"接口变了但调用方没跟上",同样的 `try/except: log.debug` 把它们藏住):

- `agent/tools/builtin/premise_search.py`: TF-IDF fallback 路径整段是死的——
  从写下来就在调一个不存在的 API。`from knowledge.tfidf_retriever import TFIDFRetriever`
  失败 (实际类名 `KnowledgeTFIDFRetriever`),即便修了类名,构造参数也不对
  (实际是 `bm25_weight, tfidf_weight`,不是 `path`),方法也错了
  (实际是 `.search()`,不是 `.retrieve()`),返回类型也错了
  (实际 `list[ScoredLemma]`,代码却 `for name, score in tfidf_results`)。
  v12 重写整段:从 `data/premises/*.jsonl` 加载,用真 API,
  `except Exception` 升 WARNING (不再 silent debug)。
- `prover/unified/tools_extra.py::DecomposeSubgoalTool`:
  `GoalDecomposer(None)` 把 LLM 写死为 None,然后 `.decompose()` 内部
  `self.llm.generate(...)` 必 AttributeError。这条 tool 从 v3 引入到 v11
  在 dsp / pantograph_dsp / conjecture_driven 三个 profile 里**永远**返回
  "decompose failed: 'NoneType' object has no attribute 'generate'"。
  改为通过 `build_tool_registry` 接受 LLM,与 ConjectureProposeTool 对齐。
  顺便修了同一文件里的假字段 `getattr(sg, "kind", "subgoal")`——SubGoal
  数据类没有 `kind`,永远返回默认值。改为输出真字段
  `name/statement/difficulty`。
- `eval.sh --early-stop`: 转发给 `run_eval.py`,但后者在 v9 已删除该参数。
  跟 v11 自己刚修的 `--multi-role` 同模式。改为 warn + ignore。

**修了 ConjectureVerifier._type_check 的接口错配**(虽然主路径不调,但是
是 v11 报告里指出的"已知是死的、不删也不修"那条):
`self.lean_env.compile(code) → (returncode, _, stderr)` 是个**项目里没有
任何 Lean env 暴露过**的 API。`AsyncLeanPool` 的真 API 是
`verify_complete(theorem, proof, preamble)` 且 async。改为 feature-detect
两种 shape,事件循环冲突时静默 skip。`ConjectureProposeTool` 用
`verify=False` 的 workaround 不再必要。

**反作弊检查改为 per-profile 可配置**:`prover/verifier/integrity_checker.py`
把 `native_decide` / `Decidable.decide` 标为 CRITICAL 并直接翻 `verified=False`,
但 Mathlib 大量正确证明合法地用这两条。v12 引入 `Profile.integrity_strict`
旗(默认 `False`):

  * `False` (默认 — Mathlib 风,适合 miniF2F / ProofNet):integrity 违规
    在响应里以 `integrity_violations` + `integrity_note` 的形式作为告知
    出现,但不翻 `verified`。
  * `True` (竞赛风,适合 PutnamBench / FormalMATH):违规直接翻
    `verified=False`,与 v11 行为一致。

`sorry` / `admit` 通过独立的 `sorry_free` 字段拦截,与 `integrity_strict`
正交——任何模式下 `sorry` 都会拒。

**LLM 缓存真接通**:`AsyncCachedProvider` 的 `chat()` 之前没覆盖,但
`AgentLoop` 优先调 `chat()`(`agent/runtime/agent_loop.py`),意味着
v11 之前的多轮主路径**完全绕过缓存**。v12 添加 `chat()` 缓存覆盖,加
`cache_stats()` 方法,`run_unified.py` / `run_eval.py` 加 `--cache` /
`--cache-all` 旗,run 结束打印命中率。pass@k 模式下应该有可观节约。

**收敛 `claude-sonnet-4-20250514` 模型字符串**:
之前 hardcode 在 8 处(`run_eval.py`、`run_unified.py` ×2、
`async_llm_provider.py` ×3、`profiles.py` ×2)。v12 全部读
`common/constants.py::DEFAULT_CLAUDE_MODEL`,可通过 `AI4MATH_DEFAULT_MODEL`
环境变量覆盖。Smoke 测试加了一条**结构性断言**:模型字符串只能出现
在 `common/constants.py` 一处,出现在别处即测试失败——堵死后续再
hardcode 的回归。

**死代码再删一轮**:

- 整目录:`prover/codegen/`(自 v11 起就是只剩一句 docstring)
- 文件:`common/hook_types.py`、`common/budget.py`、
  `common/working_memory.py`(三个 0 主路径调用方,
  `agent.{hooks, plugins, memory}` 在 v9 删除时漏了清这一份的孤儿)
- 字段:`Profile.plugins`、`ObservationPolicy.compress_errors_budget`、
  `ObservationPolicy.visible_history_turns`(都是 0 读者,
  YAML 保留兼容 shim,旧 YAML 仍能加载)
- 测试:14 个 `test_v[4-7]_*.py` 按内容重命名(`test_v4_world_model.py`
  → `test_world_model.py` 等),`test_sampler_v7.py` →
  `test_sampler_async.py`。版本前缀只能命名当前正在加的功能;
  一旦合入主线,该按"测什么"命名,不该按"哪版加的"。

**新增 `tests/test_smoke_v12.py`**:把这次修的每一个 bug 钉一条测试。
后面任何回归会立即被 CI 抓到——这是 v11 报告里指出的最大漏洞
("0 个 live integration test")的第一步弥补。共 16 条断言,包括:

- 三个 latent bug 各一条(确认接口对得上,而不只是"调用没崩")
- 一条结构性断言(模型字符串不能 hardcode 在 `common/constants.py` 之外)
- 死代码删除断言(三个文件、一个目录、三个字段确实没了)
- YAML 兼容 shim 断言(旧 YAML 含 `plugins:` 还能加载)

**未做(留给 v13 — 需要真 benchmark 数据才能安全做)**:

- `sampler/tree_rollout_sampler.py` 与 `prover/unified/search_driver.py`
  的 ~900 行重复合并(架构核心承诺:统一 AgentLoop,但 RL 路径目前
  还是平行重写)
- `GoalInspectTool` 改 REPL 单步查询(目前每次重编译整证,秒级)
- `BroadcastTool` 在 heterogeneous sub-profile 真接进 `tools` 列表
  (目前 438 行 broadcast 基础设施在主路径死代码;heterogeneous
  实际是 best-of-4)
- 真 ANTHROPIC_API_KEY × 真 Lean 4 REPL 跑 miniF2F-test pass@8,
  得到项目历史上**第一个真实 pass@k 数字**

---

## V11 — 三个 latent bug + 文档刷新 + 死代码再清一轮 (2026-05)

**三个 latent bug**(都是 v10 的 ``pool.check_proof`` 那一类问题——
接口变了但调用方没跟上):

- ``agent/tools/builtin/tactic_suggest.py``:
  ``pool.try_tactic(tactic, context=, timeout=)`` 完全错——真签名是
  ``async try_tactic(env_id: int, tactic: str)``。任何启用了 ``tactic_suggest``
  工具的 profile 一调就 TypeError。修。
- ``prover/conjecture/conjecture_proposer.py``:
  ``ConjectureProposer.propose`` 是 sync 函数, 内部 ``self.llm.generate(...)``
  没 ``await``——而 ``AsyncLLMProvider.generate`` 是 async, 直接拿到
  coroutine, 下一行 ``resp.content`` 必 AttributeError。``conjecture_driven``
  profile 自 v6 引入到 v10 一直跑不通。``propose`` 改 async, 用
  ``inspect.iscoroutine`` 兼容旧 sync provider。
- ``run_eval.py --lean-mode real``: 之前喂 ``LeanEnvironment`` 给 runner
  (没有 ``verify_complete`` 方法), 静默退化到 prefilter; v10 加的
  AttributeError 兜底掩盖了这个 silent failure。现直接构造
  ``AsyncLeanPool``, 真正 real-Lean 模式可用。

**修了 ``eval.sh``**(项目最外层入口脚本之前是坏的):

- Step 3 (APE 引擎基准) ``import`` 的 ``engine.core`` / ``engine.search``
  在 v8 已删, 该步从 v8 起每次必 ``ModuleNotFoundError``。整段删除。
- ``--multi-role`` 旗标 v9 已从 ``run_eval.py`` 移除, eval.sh 还在转发,
  导致 ``unrecognized arguments``。该旗一并删除。
- 加 ``--profile`` 旗与 ``run_unified.py`` 一致。

**README 卖点真正生效**: ``IntegrityChecker`` (反作弊 — 嵌套注释藏 sorry,
``native_decide`` 绕过, ``set_option maxHeartbeats 0`` 等) 之前一直是
0 主路径调用, v11 把它接进 ``LeanVerifyTool.execute()``: 即使 Lean 接受
证明, 触发任一 ``CRITICAL`` 完整性问题就把 ``verified`` 标为 false。

**抽了四处重复**:

- ``prover/unified/factory.py`` 新增 ``load_world_model`` /
  ``load_dialog_index`` / ``load_knowledge``。``run_unified.py`` 和
  ``run_eval.py`` 不再各写一遍 (~120 行重复消失)。
- ``benchmarks/datasets/_base.py`` 新增 ``parse_lean_files``;
  miniF2F / PutnamBench / ProofNet / FATE / FormalMATH 五个 loader 复用
  (它们之前各持一份微调过的 ``_THEOREM_RE``, 行为不一致)。
- 4 套 stderr → category 分类合 1 套, 都走 ``engine._core.classify_error``
  (14 类, 比之前最大那套还多 8 类)。``LeanVerifyTool._classify_error`` /
  ``adapters._classify_category`` 现在都是薄包装。
- ``engine/transport.py::HTTPTransport`` 删除 (薄壳, 包 ``KiminaServerBackend``);
  ``sampler/proof_env.py`` 直接用底层 backend, ``SyncTransportAdapter``
  也删除 (0 引用)。

**死代码再清一轮** (v10 cleanup 留下的第三批):

- 整目录: ``prover/sketch/``, ``prover/lemma_bank/``, ``agent/executor/``
- 文件:
  ``prover/verifier/{goal_extractor, error_parser, lean_checker}.py``,
  ``prover/codegen/import_resolver.py`` (testing-only fallback chain),
  ``agent/persistence/session_store.py`` (lane subsystem 已 v9 删,
  这是仅剩的孤儿),
  ``sampler/backend_adapters.py`` (仅测试用,
  生产路径用 ``ProofEnv._make_transport_factory``),
  ``prover/premise/tactic_suggester.py`` (仅测试用)
- 函数: ``EXPERIMENTAL_PRESETS`` 空字典 + ``enable_experimental_search_presets``
  no-op (自 v4 起就是空+空), ``KnowledgeWriter.import_from_lemma_bank``
  (一次性迁移辅助, 目标类已删)。

**保留并加用**: ``prover/verifier/integrity_checker.py``——之前在 v11
  的删除候选里, 但反作弊有真实价值, 改为接进 ``LeanVerifyTool`` 主路径。

**文档**:

- ``docs/ARCHITECTURE.md`` 整体重写。v10 版本列了八个早已删除的目录
  (``engine/lane``, ``agent/strategy``, ``agent/hooks``, ``agent/plugins``,
  ``agent/context``, ``agent/memory``, ``agent/executor``,
  ``knowledge/evolver``, ``knowledge/broadcaster``)。新版本只列实际仓库
  里的目录, 加了 v11 变更段和 v12 路线段。
- ``TUTORIAL_CN.md`` 第 410 行 ``--samples 32`` (run_unified.py 上不存在)
  改 ``--max-samples 32`` 并标明走 ``run_eval.py``。

**配置**: env 变量前缀 ``APE_*`` → ``AI4MATH_*``。``APE_*`` 仍接受 (一轮
deprecation 警告), v12 移除。

**测试**:

```
v10 baseline: 768 通过, 1 跳过, 0 失败
v11 final:    736 通过, 1 跳过, 0 失败  (-32 测试, 1:1 对应已删模块的测试)
```

**代码量**:

```
v10:  44,260 行 Python
v11:  41,664 行 Python  (净 -2,596 行 / -5.9%)
```

---

## V10 — Bug 修复 + 工程债务清算 + 锁起来的功能解锁 (2026-05)

**两个 latent bug**(v9 起就在,代码 review 才发现):

- `LeanVerifyTool` / `GoalInspectTool` / `runner._auto_verify_proof`
  调用的 `pool.check_proof()` 在 `AsyncLeanPool` 上**不存在**——
  正确接口是 `verify_complete(theorem, proof, preamble)`。这意味着
  `whole_proof_repair` 等 profile 的"compile-and-fix loop"自 v3 起
  从未真正接到 Lean,每次都落 except 返回错误字符串给 LLM。三处都已修。
- `run_eval.py` 第 401 行写入 `args.multi_role`,但 `--multi-role` 标志
  早在 v9 删掉,这一行触发 `AttributeError` 让评测末尾的 metrics 落盘失败。

**README 卖点真正生效**:`AgentFeedback` (~100 bits 结构化反馈) 之前只
在 sampler 路径用,主路径返回的是截断 stderr。v10 把 `ErrorIntelligence`
接进 `LeanVerifyTool`,主路径每次失败的 verify 现在产出
`agent_feedback.{remaining_goals, error_category, repair_candidates,
progress_score, summary}`,与 README 描述一致。

**锁起来的功能解锁**:`UnifiedProofRunner` 早就接受
`world_model=` 和 `dialog_index=` 参数,但 CLI 没暴露,从命令行无法
启用。v10 给 `run_unified.py` 和 `run_eval.py` 都加了:

- `--world-model PATH` —— 加载 sklearn world model pickle
- `--dialog-index DB_PATH` —— 加载 SQLite DialogIndex 做跨问题 demo 注入
- `--knowledge-db DB_PATH` (run_unified 新增) —— 单题模式开知识沉淀

**死代码清理**(在 v9 大清理之外又删了 ~3.7k 行):

- 整目录:`prover/repair/`、`prover/formalize/`、`agent/context/`
- 部分文件:`prover/decompose/{composition,subgoal_scheduler}.py`、
  `prover/codegen/{code_formatter,tactic_generator,scaffold_generator}.py`、
  `knowledge/{evolver,broadcaster,retriever,backend}.py`
- `agent/tools/builtin/__init__.py::register_all_builtins`(0 调用方)
- `prover/repair/_fix_identifier` 里的 Lean3→Lean4 重命名表抽到独立的
  `engine/lean3_to_lean4.py` 保留(可被任何路径复用)

**用户可见文档修复**(都是 v9 cleanup 留下的):

- `TUTORIAL_CN.md`: 全文 15 处 `run_single.py` (v9 已删) 全部替换为
  `run_unified.py` + 正确参数。新手照教程跑不再立即报错。
- `docker/Dockerfile.agent`: 注释里的 `run_single.py` 引用已修。
- `docs/ARCHITECTURE.md`: 入口层架构图删除已死的 `run_single_lane.py`。
- `scripts/lean4_smoke_test.py`: 修了导入已删的 `engine.proof_session`
  的 collection error。

**docstring 与代码对齐**:`prover/__init__.py` 和 `agent/__init__.py`
docstring 重写,只列实际活的子模块,把已删的明确归到"v10 删除"段。

**工程基础设施**:

- 加 `pyproject.toml` (ruff + mypy + pytest 配置)
- 加 `.github/workflows/ci.yml` 跑 ruff + 测试
- ruff: 0 errors (所有 113 个 F401 unused import + 1 个 B033 set
  duplicate 已修)
- mypy: 在 `engine/lean3_to_lean4.py` 和 `agent/tools/builtin/lean_verify.py`
  这两个新写的模块上 strict, legacy 模块 permissive

**测试**:

```
v9 baseline: 800 测试方法
v10 final:   713 测试方法 (-87, 1:1 对应已删模块的测试)
            768 passed, 1 skipped, 0 failed   (零回归)
```

---

## V7 — RL 飞轮闭合 (2026-05)

- `sampler/`: 新增 `tree_rollout_sampler.py` —— MCTS / best-first / beam
  现在是一等的 RL roll-out 单元(之前只能离线生成 dialog.json)。
- `sampler/proof_env.py`: 新增 `backend` 字段,RL 采样可以直连
  Kimina/Pantograph/LooKeng;之前默认 LocalTransport,Kimina 的批量验证完全
  够不到。
- `sampler/`: `BaseSampler` 的 env-pool 从 `Semaphore + _in_use bool`
  换成 `asyncio.Queue`,消除 TOCTOU race。
- `sampler/verl_sampler.py`: 真正 `@register("ai4math_proof_agent")` 注册
  到 verl,而不是 ducktyping。新增 `requirements-rl.txt` 列可选依赖。
- 修了三个 sampler 都有的 latent bug:`pool.start(preamble=...)` →
  preamble 应在构造时传,`start()` 不接参数。

## V6 — DialogIndex 持久化 + 猜想驱动 + 后端可见性 (2026-04)

- `knowledge/dialog_index.py`: 新增 SQLite 持久化(`persist_to_sqlite` /
  `load_from_sqlite`),50k+ entries 不再每次进程重启都重建索引。
- `prover/unified/profiles.py`: 新增 `conjecture_driven` profile,
  把 `prover/conjecture/` 接入统一 runner(之前要绕开 runner 直接用)。
- `prover/unified/runner.py`: `auto_register_llm_autoformalizer=True` 默认开,
  V5 引入的"opt-in 一次"忘了配的人会静默落到 regex 启发式。
- `dialog.json` 新增 `meta.backends` 字段,`is_fallback=true` 不再只在 debug log。
- 删除 `prover/pipeline/proof_loop_legacy.py`(187 LOC 死代码,零测试覆盖)。

## V5 — 测试稳健性 + Profile YAML 模板 (2026-03)

- `config/profiles/*.yaml`: 14 个 active preset 全部有对应 YAML 模板,
  通过 `scripts/dump_profile_yamls.py` 从 Python `PRESETS` 自动生成。
- `prover/unified/llm_autoformalizer.py`: NL → Lean 形式化用 LLM 而不是
  5 行正则。仍保留正则作 fallback。
- 测试集多个 fixture 不再硬要求 Lean / pyrsistent / sentence-transformers,
  环境缺包时 graceful skip 而不是 hard fail。

## V4 — RL 飞轮 + 步级知识 + 世界模型 (2026-02)

- `scripts/rl_pipeline.py` + `scripts/rl_loop.sh`: "评测→收 trajectory
  → SFT JSONL → 世界模型" 串成一条命令;最后一步外包给 TRL/verl/slime。
- `engine/proof_context_store.py`: `StepDetail` 落盘,每步都进知识库,
  不仅是整证成功才落。
- `engine/world_model.py` + `world_model_trainer.py`: sklearn 包装的
  `WorldModelPredictor`(预测某 tactic 应用后是否更接近闭目标)。
- MCTS / dialog 格式合流,树搜索结果也产出标准 dialog.json。

## V3 — 大一统 Profile 管线 (2026-01)

- 新增 `prover/unified/`: 单一 `UnifiedProofRunner`,所有方法用 14 个
  Profile 表达。换方法 = 改 `--profile`,不动代码。
- 全系统单一输出格式:`results/traces/<id>/dialog.json` schema v2.0。
- 旧 `prover/pipeline/` 多个 Engine 类通过 `prover/unified/adapters.py`
  反向适配,保留 v2 调用方一段时间。
- 知识闭环:每次尝试结果写入知识库,下轮 prompt 注入累积知识。

## V2 — Lane 运行时 + 三级验证 (2025-12)

- 新增 `engine/lane/`: 每道题一个状态机,带类型化事件总线、可执行策略
  规则、自动恢复方案、断点续证。
- 三级验证:L0 语法预过滤(~1μs)→ L1 REPL 快验(~50ms)→ L2 全编译(~3s)。
- Sorry/Axiom 完整性深检测,即使 Lean 编译通过也会拒绝含 sorry 的证明。
- Green Contract 6 级验证状态。

## V1 — 社区 backend 接入 + 基础设施 (2025-11)

- 接入四个社区 Lean 4 项目作为 `REPLTransport` 实现:Kimina Lean Server
  (HTTP)、PyPantograph(Python 绑定)、LooKeng(stateless lemma)、本地 REPL。
- `engine/async_lean_pool.py`: N 路并行 REPL 长连接,每会话维护
  `env_id` 实现增量验证。
- `engine/error_intelligence.py`: Lean stderr 结构化为 `AgentFeedback`
  (~100 bits) 而不是 1 bit pass/fail。
- `engine/broadcast.py`: 跨 agent 实时发布-订阅总线。

## V0 — 初始原型 (2025-09)

- 单 Engine 实现:`Orchestrator` + `RolloutEngine` + 同步 LLM 调用 +
  本地 Lean 子进程。
- APE v1 自建 CIC kernel(`engine/core/ kernel/ tactic/ state/ search/`)。
  V2 起改为直接用 Lean 4 REPL,这些模块 V7 起完全删除。
- 7 个基准数据集导入(miniF2F、PutnamBench、ProofNet、FATE-{M,H,X}、FormalMATH)。
