# AI4Math-APEv2 优化修复记录

**修复日期**: 2026-04-05
**影响范围**: 10 个核心文件修改 + 1 个新文件, +463 / -159 行
**测试状态**: 396/396 原有测试通过, 无回归

---

## 修复总览

| 编号 | 级别 | 问题 | 修复文件 | +/- 行 |
|------|------|------|----------|--------|
| #1 | Critical | `LeanSession` REPL I/O 无并发保护 | `engine/lean_pool.py` | +108/-42 |
| #3 | Critical | L2 `lean --run` 语义错误 + 证明拼接脆弱 | `engine/verification_scheduler.py`, `engine/lean_pool.py` | +25/-15 |
| #2 | Critical | `share_lemma()` 不追踪新 env_id | `engine/lean_pool.py` | (含在 #1 中) |
| #5 | High | PreFilter 守卫只检查 proof 不检查 theorem | `engine/prefilter.py` | +30/-10 |
| #6 | High | Lean3 检测器漏报 `nat.add_comm` 等命名空间写法 | `engine/prefilter.py` | (含在 #5 中) |
| #7 | High | `exact?`/`apply?` 输出解析硬编码不健壮 | `engine/error_intelligence.py` | +44/-10 |
| #8 | High | `_backpropagate` 直接修改不可变 PMap | `engine/state/search_tree.py`, `engine/search/__init__.py` | +82/-23 |
| #9 | High | `TacticExistence` 规则永远返回 OK (no-op) | `engine/prefilter.py` | (含在 #5 中) |
| #10 | Medium | 广播队列过期消息挤占有效消息空间 | `engine/broadcast.py` | +10/-1 |
| #11 | Medium | `speed_bonus` 阈值使 REPL 模式丧失区分度 | `engine/search/__init__.py` | (含在 #8 中) |
| #13 | Medium | 异构方向数量/模型/温度全部硬编码 | `prover/pipeline/heterogeneous_engine.py`, `config/default.yaml` | +149/-57 |
| #14 | Medium | `KnowledgeRetriever` 未自动创建导致前提检索断开 | `prover/pipeline/orchestrator.py` | +15/-1 |
| #16 | Low | 缺少真实 Lean4 集成测试基础设施 | `tests/test_integration/test_lean4_real.py` | +206 (新文件) |

---

## 详细修复说明

### Fix #1: LeanSession 并发安全 (Critical)

**根因**: `try_tactic()` 中 `_lock` 仅保护 `total_requests` 计数器, 释放后 REPL
stdin/stdout 读写无任何串行化保护。当多线程共用同一 session 时, 命令-响应交错。

**修复**:
1. 将 `_lock` 作用域扩展到整个 REPL 交互过程 (`try_tactic`, `verify_complete`, `get_goals`)
2. `LeanPool` 新增 `threading.Condition` 条件变量, `_acquire_session()` 在所有 session
   繁忙时阻塞等待而非返回已占用的 session
3. `_release_session()` 通知等待线程
4. 超时后降级: 返回请求数最少的 session, 此时两个线程共用但通过会话级锁串行化

### Fix #3: L2 验证命令 + 证明拼接 (Critical)

**根因**: `lean --run` 的语义是"编译并执行 main", 纯定理证明没有 main 函数会报错。
`_assemble_code` 中 `":=" not in full` 会被定理签名中的 `:=` 误触发。

**修复**:
1. L2 改用临时 `.lean` 文件 + `lean <file>` 编译, 正确做类型检查
2. 新增 `_has_toplevel_assign()`: 通过括号计数检查顶层 `:=`, 忽略嵌套的赋值

### Fix #2: fork_env / share_lemma 语义 (Critical)

**修复**:
1. `share_lemma()` 成功后更新 `session._state.current_env_id`
2. 新增 `latest_env_ids` 属性供调用者获取每个 session 的最新环境
3. `share_lemma()` 跳过 fallback 模式的 session

### Fix #5/#6: PreFilter 逻辑修复 (High)

**修复**:
1. `NatSubtractGuard` / `RingOnNatGuard`: 减法检测从 `proof` 扩展到 `proof + theorem`
2. `Lean3Detector`: 新增 `nat.xxx`, `int.xxx`, `list.xxx` 命名空间模式
3. `TacticExistence`: 从永远返回 OK 改为返回 warning 级别的未知 tactic 报告

### Fix #7: exact?/apply? 解析健壮性 (High)

**修复**: 支持三种输出格式:
- `"Try this: exact <term>"` (标准 Mathlib 格式)
- `"exact <term>"` (无前缀)
- 从 `error_message` 和 `remaining_goals` 两个来源提取建议

### Fix #8: SearchTree 不可变性 (High)

**修复**:
1. `SearchTree` 新增 `update_node()` 和 `backpropagate()` 方法, 返回新 tree 实例
2. `SearchCoordinator._backpropagate()` 改用 `self._tree = self._tree.backpropagate(...)`
3. 不再直接修改 `self._tree._nodes`

### Fix #10: 广播队列过期清理 (Medium)

**修复**: `Subscription.push()` 在队列容量 ≥80% 时主动清理过期消息, 防止有效消息被挤出。

### Fix #11: 速度奖励跨尺度适配 (Medium)

**修复**: 从线性 `threshold/elapsed` 改为对数衰减 `max(0, 1 - log2(elapsed/threshold + 1))`,
使本地模式 (~1μs) 和 REPL 模式 (~50ms) 下的速度差异都能产生有意义的评分梯度。

### Fix #13: 配置化异构方向 (Medium)

**修复**:
1. `default.yaml` 新增 `directions` 配置段, 声明式定义方向的 name/role/model/temperature/hint
2. `HeterogeneousEngine._plan_directions()` 优先从配置读取, 无配置时使用内置默认
3. 新增 `_role_from_string()` 辅助函数映射字符串角色名到枚举

### Fix #14: 自动创建 KnowledgeRetriever (Medium)

**修复**: `Orchestrator.__init__()` 在 `retriever=None` 时自动创建 `KnowledgeRetriever`,
连接 BM25 检索和内置 Mathlib 前提库。

### Fix #16: 真实 Lean4 集成测试 (Low)

**新增**: `tests/test_integration/test_lean4_real.py` (206 行)
- 使用 `@requires_lean4` 标记, 无 Lean4 环境时自动 skip
- 覆盖: REPL 池启动、tactic 执行、L0→L1→L2 管线、广播集成、延迟基准
