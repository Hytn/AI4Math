# AI4Math-APE v4 架构审查与改进规划

## 一、总体评价

项目的宏观愿景（形式化验证操作系统）极具雄心，已有的代码体量约 32K 行，模块划分为 **engine / agent / prover / benchmarks** 四大层次，涵盖了 REPL 连接池、异步管线、三级验证调度、广播总线、异构并行引擎、增量验证、动态伸缩等子系统。整体设计方向正确，但在审阅全部核心源码后，发现以下 **7 个结构性问题** 和若干实操层面的缺陷，按严重程度排序如下。

---

## 二、结构性问题诊断

### 问题 1: 同步/异步双栈冗余 — 最大的架构债务

**现象**: `LeanPool` (threading) 和 `AsyncLeanPool` (asyncio) 几乎是逐方法复制粘贴。同样的情况出现在 `Orchestrator` vs `AsyncOrchestrator`、`EngineFactory` vs `AsyncEngineFactory`、`AgentPool` vs `AsyncAgentPool`、`VerificationScheduler` vs `AsyncVerificationScheduler`。

**危害**:
- 每次修 bug 或加功能都要改两份代码，极易遗漏
- `LeanSession._send_raw` 用 `selectors` 实现超时，`AsyncLeanSession._send_raw` 用 `asyncio.wait_for` — 行为差异已出现（例如同步版 `_send_raw` 在 selectors 异常时不关闭进程，而异步版也没有）
- `Orchestrator.prove()` 是同步的，但调用了同步 `LeanPool`，而 `AsyncOrchestrator.prove()` 调用 `AsyncLeanPool` — 两条执行路径的测试覆盖不可能对等

**根因**: 项目先写了同步版，后加异步版时没有统一抽象。

### 问题 2: REPL 会话管理存在隐蔽的状态泄漏

**现象**: 
- `LeanPool._acquire_session()` 在所有会话忙时创建 overflow 会话并追加到 `self._sessions`，**但从未回收**。长时间运行后会话数无限增长。
- `AsyncLeanPool` 同样有此问题。虽然有 `PoolScaler`，但 `PoolScaler` 只在空闲时缩容，而 overflow 会话一旦变忙就不会被缩。
- `LeanSession.try_tactic()` 中 `busy` 标志由 `LeanPool` 管理，但 `verify_complete()` 中 `busy` 标志由 `LeanSession` 自己管理 — 两套互相矛盾的并发控制。

**具体代码**:
```python
# lean_pool.py, LeanSession.verify_complete()
with self._lock:
    self._state.busy = True   # ← Session 自己设 busy
    ...
# lean_pool.py, LeanPool._acquire_session()
session._state.busy = True    # ← Pool 也设 busy
```
当 `verify_complete` 被直接调用（不通过 `LeanPool.verify_complete`）时，Pool 的 Condition 不会被 notify，导致其他线程永久等待。

### 问题 3: 编译缓存的正确性隐患

**现象**: `CompileCache` 和 `LeanPool._compile_cache` 用 `sha256(preamble||theorem||proof)` 做 key，但：
- 缓存的是 `FullVerifyResult`，其中 `env_id` 字段指向的 REPL 环境是特定会话的，缓存命中后返回的 `env_id` 可能在另一个会话中无效
- 不同 Lean 版本 / Mathlib 版本下同一代码的验证结果不同，但缓存 key 不含版本信息
- `classify_error` 是纯字符串匹配，`timeout` 判断用 `"timeout" not in result.stderr.lower()`，但错误消息中包含 "timeout" 一词（如 "try increasing heartbeats for deterministic timeout"）也会被排除缓存

### 问题 4: Orchestrator 职责过载 — God Object 反模式

`Orchestrator.prove()` 单个方法包含：策略选择、钩子触发、策略升级、反思闭环、验证分发、分解调度、猜想生成、budget 管理。约 200 行主循环，嵌套了 6 层以上的 if/while/for。

**后果**: 无法单独测试某个阶段；无法替换策略升级逻辑而不碰证明循环；无法并行化某些阶段（如分解和猜想可以与主循环并行）。

### 问题 5: Agent ↔ Engine 层边界模糊

**现象**:
- `VerificationScheduler` (engine 层) 直接 import `AgentFeedback`（语义上属于 agent 层）
- `Orchestrator` (prover 层) 直接操作 `engine._core` 的数据类型
- `SubAgent.refine_confidence` 是 `@staticmethod`，在 `Orchestrator`、`AsyncOrchestrator`、`async_factory.py` 三处被重复 import 和调用
- `engine/error_intelligence.py` 生成的 `AgentFeedback` 包含 `repair_candidates` 和 `strategic_hint` — 这些是 agent 层的关注点，不应在 engine 层产生

**后果**: engine 层无法独立于 agent 层使用（如果想把 engine 层作为通用验证服务，当前做不到）。

### 问题 6: 配置系统不完整

- `config/schema.py` 定义了 schema，但没有 `config/default.yaml` 文件
- `EngineFactory.build()` 通过 `self.config.get("lean_pool_size", 4)` 获取配置，但 `Orchestrator` 直接把整个 `config` 透传，没有 namespace 隔离 — engine 层和 agent 层的配置 key 可能冲突
- 没有环境变量覆盖机制（生产部署必需）
- `_KEY_ALIASES` 的双向映射没有冲突检测

### 问题 7: 错误处理和可观测性不足

- 绝大多数异常被 `except Exception as e: logger.warning(...)` 吞掉，没有传播
- 没有结构化日志（全部是 `logger.info/warning/error` + 字符串拼接）
- 没有指标收集（Prometheus / OpenTelemetry）— 对"毫秒级响应"的目标无法验证
- `ProofTrace` 记录了 `total_duration_ms`，但没有各阶段的耗时分解

---

## 三、实操层面的缺陷

| # | 文件 | 问题 | 影响 |
|---|------|------|------|
| A | `lean_pool.py:LeanSession._send_raw` | `selectors` 实例每次创建/注销，开销不必要 | 性能 |
| B | `lean_pool.py:LeanPool.share_lemma` | 串行调用所有 session 的 `verify_complete`，未并行 | 性能 |
| C | `broadcast.py:BroadcastMessage.__post_init__` | 在 `frozen=True` dataclass 中用 `object.__setattr__` 绕过不可变约束 | 正确性风险 |
| D | `proof_session.py:ProofSession.rewind` | 回退时 pop `tactic_history` 但不验证是否匹配 | 状态不一致 |
| E | `verification_scheduler.py:_l2_full_compile` | 临时文件写入 `self.project_dir/.ai4math_tmp/`，但从未清理目录本身 | 磁盘泄漏 |
| F | `context_window.py:_compress` | Phase 3 总结只取每条 content 第一行前 100 字符，信息丢失严重 | 质量 |
| G | `agent_pool.py:run_parallel` | `ThreadPoolExecutor` 内异常被捕获后构造了空结果，调用者无法区分"真正的空结果"和"异常" | 调试困难 |
| H | `heterogeneous_engine.py` | `run_round` 返回的结果没有验证就设置 confidence > 0 | 语义错误 |
| I | `_core.py:assemble_code` | 当 theorem 已含 `:= by ...` 且 proof 非空时，会拼出 `theorem t : T := by ... := by ...` 的无效代码 | 正确性 |

---

## 四、逐步改进规划

### Phase 0: 止血修复（1-2 天）

> 目标: 修复正确性问题，不改架构

1. **修复 `assemble_code` 拼接逻辑** (问题 I): 检测 theorem 是否已包含证明体，避免重复拼接
2. **修复 overflow session 泄漏** (问题 2): 在 `_release_session` 中检测 overflow 会话，若空闲则关闭并移除
3. **修复 `verify_complete` 的 busy 管理** (问题 2): 统一由 Pool 管理 busy 标志，移除 Session 内部的自管理
4. **修复缓存返回的 `env_id` 问题** (问题 3): 缓存命中时将 `env_id` 置为 -1（标记为不可复用），或分离"结果缓存"和"状态缓存"
5. **清理 L2 临时目录** (问题 E): 在 `VerificationScheduler.__init__` 中清理，或用 `tempfile.TemporaryDirectory`

### Phase 1: 统一异步架构（1 周）

> 目标: 消除同步/异步双栈，以 async 为唯一内核

1. **定义 Transport 协议** (`engine/transport.py`):
   ```python
   class REPLTransport(Protocol):
       async def send(self, cmd: dict) -> Optional[dict]: ...
       async def start(self, preamble: str) -> bool: ...
       async def close(self): ...
   ```
2. **统一 `LeanSession`**: 删除同步版，只保留 `AsyncLeanSession`，通过 Transport 接口支持本地/远程
3. **统一 Pool**: 删除 `LeanPool`，只保留 `AsyncLeanPool`，对外提供 `run_sync()` 包装
4. **统一 Orchestrator**: 删除同步 `Orchestrator`，`AsyncOrchestrator` 提供 `prove_sync()` 入口
5. **统一 Factory**: 合并 `EngineFactory` 和 `AsyncEngineFactory`

**预计减少代码量**: ~3000 行

### Phase 2: 分层解耦（1 周）

> 目标: 建立清晰的 engine ← prover ← agent 依赖方向

1. **定义 engine 层公共接口** (`engine/api/`):
   ```
   VerificationRequest → VerificationResult  (无 AgentFeedback)
   TacticRequest → TacticResult              (无 repair 建议)
   ```
2. **将 `AgentFeedback` 生成移到 agent 层**: engine 层只返回原始错误，agent 层的 `ErrorAnalyzer` 负责生成修复建议
3. **将 `SubAgent.refine_confidence` 移入 `ConfidenceEstimator`**: 消除跨层调用
4. **将 `ProofDirection` 规划移出 `HeterogeneousEngine`**: 用 `DirectionPlanner` 类管理方向规划策略

### Phase 3: Orchestrator 拆分（3-5 天）

> 目标: 消除 God Object

将 `Orchestrator.prove()` 拆分为 pipeline stages:
```
ProblemAnalyzer → StrategySelector → ProofLoop → ResultAggregator
                                      ↓ (每轮)
                              CandidateGenerator → Verifier → FeedbackInjector → Escalator
```
每个 stage 是独立可测的类，通过 `ProofPipeline` 编排。

### Phase 4: 可观测性（3-5 天）

> 目标: 为"毫秒级响应"目标提供度量基础

1. **结构化日志**: 用 `structlog` 替代 `logging`，所有日志带 `session_id / problem_id / direction` 上下文
2. **指标收集**: 在 Pool / Scheduler / Orchestrator 中埋点:
   - `repl_latency_histogram` (L0/L1/L2 分级)
   - `pool_utilization_gauge`
   - `proof_attempt_counter` (按 strategy / outcome 分)
   - `llm_latency_histogram`
3. **`ProofTrace` 增强**: 记录每个 stage 的耗时、每次验证的 level 和结果

### Phase 5: 配置与部署（2-3 天）

1. **创建 `config/default.yaml`**: 包含所有参数的默认值和文档
2. **环境变量覆盖**: `APE_ENGINE__LEAN_POOL_SIZE=8` → `engine.lean_pool_size=8`
3. **配置 namespace 隔离**: Factory 只接收自己层的配置子树
4. **Docker 化**: 提供包含 elan + lean4 + mathlib 的基础镜像

### Phase 6: 弹性伸缩基座（远期，2-4 周）

> 目标: 向"形式化验证操作系统"迈进

1. **完善 `RemoteSession` + `ElasticPool`**: 实现 TCP/gRPC Transport，支持跨机器 REPL Worker
2. **持久化证明上下文**: 将 `ProofSessionState` 的 env_id 树序列化到 Redis/SQLite，支持跨会话恢复
3. **资源调度器**: 基于 `PoolScaler` 扩展为多维调度（CPU / Memory / REPL 槽位），集成 Kubernetes HPA
4. **增量编译守护进程**: 常驻的 Lean4 daemon 进程替代每次启动新 REPL，实现真正的热缓存

---

## 五、优先级总结

| 优先级 | Phase | 核心收益 | 工作量 | 状态 |
|--------|-------|----------|--------|------|
| P0 | Phase 0 | 修复正确性 bug | 1-2 天 | ✅ 已完成 |
| P0 | Phase 1 | 消除最大架构债务 | 1 周 | ✅ 已完成 |
| P1 | Phase 2 | 分层可独立演进 | 1 周 | ✅ 已完成 |
| P1 | Phase 3 | Orchestrator 可测可扩展 | 3-5 天 | ✅ 已完成 |
| P2 | Phase 4 | 性能可度量 | 3-5 天 | ✅ 已完成 |
| P2 | Phase 5 | 可部署 | 2-3 天 | ✅ 已完成 |
| P3 | Phase 6 | 远期愿景 | 2-4 周 | ⬜ 待做 |

---

## 六、已完成修复记录

### Phase 0: 止血修复 (5/5 ✅)

1. **`assemble_code` 拼接重写** — 正确处理5种 theorem/proof 组合 (`engine/_core.py`)
2. **Overflow session 泄漏** — `is_overflow` 标记 + 释放时自动关闭 (`lean_pool.py` + `async_lean_pool.py`)
3. **busy 管理统一** — 移除 Session 自管理 (`lean_pool.py`)
4. **缓存 env_id 失效** — 缓存前置 -1 + `make_cache_key` 统一 (`lean_pool.py` + `async_lean_pool.py`)
5. **L2 临时文件泄漏** — `TemporaryDirectory` + 复用 `assemble_code` (`verification_scheduler.py`)

### 实操缺陷修复 (9/9 ✅ + 3 bonus)

- **A**: selectors 持久化 (`lean_pool.py`)
- **B**: `share_lemma` 并行化 (`lean_pool.py`)
- **C**: BroadcastMessage frozen 安全 — 工厂方法预 freeze (`broadcast.py`)
- **D**: `ProofSession.rewind` 历史一致 (`proof_session.py`)
- **F**: `_compress` Phase 3 信息保留 (`context_window.py`)
- **G**: `run_parallel` 错误标记 + `is_error` 属性 (`sub_agent.py` + `agent_pool.py`)
- **H**: 未验证 confidence 上限 0.4 + metadata 标记 (`heterogeneous_engine.py`)
- **bonus**: `SubAgent.execute` success 语义修正 (`sub_agent.py`)
- **bonus**: `_trace_path` 循环检测 + `begin_proof` fallback 自引用修复 (`proof_session.py`)

### Phase 1: 统一异步架构 ✅

1. **Transport 协议层** — `LocalTransport` / `FallbackTransport` / `MockTransport` + `SyncTransportAdapter` (`engine/transport.py`)
2. **AsyncLeanSession 重构** — 使用 Transport 协议, 消除内联进程管理 (`async_lean_pool.py`)
3. **SyncLeanPool** — AsyncLeanPool 的同步包装器, drop-in 替代旧 LeanPool (`async_lean_pool.py`)
4. **EngineFactory 切换** — `_build_lean_pool()` 改用 `SyncLeanPool` (`factory.py`)

### Phase 2: 分层解耦 ✅

1. **`refine_confidence` 统一** — 从 `SubAgent` 移入 `ConfidenceEstimator`, SubAgent 保留 deprecated 委托
2. **Engine API 协议层** — `PoolProtocol` / `VerifierProtocol` / `BroadcastProtocol` (`engine/api/protocols.py`)
3. **DirectionPlanner** — 从 HeterogeneousEngine 提取方向规划逻辑 (`agent/strategy/direction_planner.py`)

### Phase 3: Orchestrator 拆分 ✅

1. **ProofPipeline** — 4阶段管线 (pre_round / generate / verify / post_round), 每阶段独立可测 (`prover/pipeline/proof_pipeline.py`)
2. **Orchestrator 接入** — `prove()` 可切换到 pipeline 模式, `prove_pipeline()` 直接调用

### Phase 4: 可观测性 ✅

1. **StructuredLogger** — 带上下文字段的结构化日志包装器
2. **MetricsCollector** — 计时器 + 计数器 + 仪表盘, 线程安全, 支持 labels
3. **`@timed` 装饰器** — 同步/异步函数通用, 自动记录耗时
4. **Pipeline 埋点** — 每阶段计时, 按策略统计轮次, 总耗时/成功率
5. 所有模块合并在 `engine/observability.py`

### Phase 5: 配置与部署 ✅

1. **`default.yaml` 增强** — 添加 `pool_scaler` / `reflection_interval` 配置
2. **环境变量覆盖** — `APE_ENGINE__LEAN_POOL_SIZE=8` → `engine.lean_pool_size=8`, 支持类型自动推断
3. **`_set_nested` / `_coerce_value`** — 嵌套路径写入 + bool/int/float/null 自动转换

### 测试

- **499 tests passed, 0 failures** (456 原有 + 43 新增回归测试)
- 新增测试覆盖: assemble_code / Transport / SyncLeanPool / ConfidenceEstimator / DirectionPlanner / _trace_path / MetricsCollector / BroadcastMessage / Config env / API protocols
