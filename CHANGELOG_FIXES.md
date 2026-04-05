# AI4Math-APEv2 严重设计问题修复记录

**修复日期**: 2026-04-05
**影响范围**: 6 个核心文件, +256 / -30 行
**测试状态**: 396/396 通过, 无回归

---

## 修复总览

| 级别 | 问题 | 修复文件 | 新增行数 |
|------|------|----------|---------|
| Critical | `heterogeneous_engine.py` 语法错误 | `prover/pipeline/heterogeneous_engine.py` | -5 |
| Critical | L2 验证与 L1 是同一调用 | `engine/verification_scheduler.py` | +75 |
| Critical | `exact?`/`apply?` 死代码 | `engine/error_intelligence.py` | +12 |
| Critical | `share_lemma()`/`fork_env()` 未被调用 | `prover/pipeline/heterogeneous_engine.py` | +35 |
| Critical | MCTS 搜索与 REPL 池断开 | `prover/pipeline/dual_engine.py` | +5 |
| High | 降级模式静默失败 | `engine/lean_pool.py` | +30 |
| 补充 | `_get_premises()` AttributeError | `prover/pipeline/heterogeneous_engine.py` | +15 |
| 补充 | 反思机制仅策略升级时触发 | `prover/pipeline/orchestrator.py` | +22 |

---

## 详细修复说明

### Fix 1: `heterogeneous_engine.py` 语法错误

**根因**: `_extract_lemma_mentions()` 方法的 `return` 语句之后，有一段从 `run_round()` 方法中错误粘贴的代码残留（第 238-242 行），导致 `unexpected indent` 语法错误，整个异构引擎模块无法加载。

**修复**: 删除孤立代码块（5 行）。

---

### Fix 2: L2 验证层从复用 REPL 改为独立编译

**根因**: `VerificationScheduler.verify_complete()` 中的 L2 路径调用 `self.pool.verify_complete()`，与 L1 路径使用同一个 `LeanPool` 方法——不存在独立的完整编译路径。文档宣称的"L2 = 完整 Lean4 编译，最终可信认证"是虚构的。

**修复**:
1. 新增 `_l2_full_compile()` 方法（67 行），使用 `subprocess.run(["lean", "--run", "-"])` 在全新进程中从零编译完整 `.lean` 文件
2. L2 路径不再依赖 REPL 池，即使池不可用也能执行 L2 认证
3. 新增 `project_dir` 参数传递给 `subprocess.run` 的 `cwd`
4. 明确的错误处理：`lean` 二进制不存在时返回可读的错误信息而非静默失败

---

### Fix 3: `exact?`/`apply?` 搜索从死代码变为可达

**根因**: `ErrorIntelligence.analyze()` 在失败路径中检查 `result.new_env_id >= 0`，但失败时 `new_env_id` 永远为 `-1`（仅在成功时赋值）。因此 `_search_via_lean()` 永远不会被调用。

**修复**:
1. 新增 `parent_env_id` 参数（默认 `-1`），允许调用者传入执行 tactic 前的环境 ID
2. 搜索条件改为 `search_env = parent_env_id if parent_env_id >= 0 else result.new_env_id`
3. 更新 `VerificationScheduler` 中所有 `analyze()` 调用点，传递 `parent_env_id=env_id`

**效果**: `exact?`/`apply?`/`rw?` 现在会在 tactic 失败后，在尚有未解 goal 的父环境上搜索修复候选。

---

### Fix 4: `share_lemma()` 和 `fork_env()` 集成

**根因**: 两个方法在 `LeanPool` 中定义但在整个代码库中无调用点。

**修复**: 在 `HeterogeneousEngine._broadcast_results()` 末尾新增两段逻辑：
1. **`share_lemma()` 集成**: 当广播总线中出现 `LEMMA_PROVEN` 消息时，提取引理代码，调用 `pool.share_lemma()` 注入所有 REPL 会话
2. **`fork_env()` 集成**: 当广播总线中出现 `PARTIAL_PROOF` 消息且含有效 `env_id` 时，调用 `pool.fork_env()` 生成可供其他方向继续的环境快照

---

### Fix 5: MCTS 搜索连接到 REPL 池

**根因**: `SearchCoordinator` 已有 `lean_pool` 参数和 `_try_tactic_via_repl()` 方法，但 `DualEngine` 和 `APEEngine` 创建 `SearchCoordinator` 时从未传递 `lean_pool`。

**修复**:
1. `APEEngine.__init__()` 新增 `lean_pool` 参数，传递给 `SearchCoordinator`
2. `DualEngine.__init__()` 新增 `lean_pool` 参数，传递给 `APEEngine`

---

### Fix 8: 降级模式显式报错

**根因**: 当 Lean4 REPL 二进制不存在时，`LeanSession.start()` 静默返回 `True`（标记为 alive），下游代码无法区分"真正的 REPL 会话"和"虚假的 fallback 会话"。所有验证请求都会静默返回 `success=False`，看起来像是证明失败而非环境不可用。

**修复**:
1. `_SessionState` 新增 `fallback_mode: bool` 字段
2. `LeanSession.start()` 在无 REPL 时设置 `fallback_mode=True` 并发出 WARNING 级别日志（含安装链接）
3. 新增 `LeanSession.is_fallback` 属性
4. `LeanPool.start()` 检测所有会话均为 fallback 时发出汇总警告
5. `LeanPool.stats()` 新增 `fallback_sessions` 和 `all_fallback` 字段

---

### Fix 11: `_get_premises()` 类型安全

**根因**: `KnowledgeRetriever.retrieve()` 返回 `list[str]`，但 `HeterogeneousEngine._get_premises()` 对返回值调用 `.get()` 方法，必然抛出 `AttributeError`。

**修复**: 使用 `isinstance(results[0], str)` 检测返回类型，兼容 `list[str]` 和 `list[dict]` 两种格式。添加 `try/except` 保护，失败时返回空列表而非崩溃。

---

### Fix 12: 反思机制常规轮次触发

**根因**: `Reflector.reflect()` 仅在策略升级（`should_escalate()` 返回非空）时被调用。如果问题始终在同一策略下失败但未触发升级条件，反思永远不会运行。

**修复**: 在 `orchestrator.prove()` 主循环的 `ON_ROUND_END` 钩子之后，新增：
1. **周期性反思**: 每 `reflection_interval`（默认 3）轮自动触发 `_run_reflection()`，将分析结论注入下一轮的 `classification.domain_hints`
2. **验证反馈注入**: 将上一轮的 `last_feedback_text`（来自 `AgentFeedback.to_prompt()`）注入下一轮上下文，使 LLM 能看到上一轮的结构化错误信息
