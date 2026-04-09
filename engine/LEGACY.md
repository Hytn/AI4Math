# engine/LEGACY.md — Legacy Module Notice

## 废弃模块清单 (2,488 行，不参与活跃代码路径)

以下模块是 APE v1 自建内核的遗留代码。它们实现了一个简化的 CIC type checker
和 18 种 tactic，但在实际评测管道中**完全不被调用**。

| 模块 | 行数 | 原始用途 | 替代方案 |
|------|------|----------|----------|
| `engine/core/` | 616 | 表达式 AST、Universe、Environment | Lean4 REPL 直接处理 |
| `engine/tactic/` | 1,165 | 18 种内置 tactic | Lean4 REPL 执行 tactic |
| `engine/kernel/` | 707 | CIC type checker | Lean4 REPL 类型检查 |

## 依赖这些模块的其他 legacy 组件

| 模块 | 说明 |
|------|------|
| `engine/state/` | 持久化 ProofState (用于 search tree) |
| `engine/search/` | MCTS/UCB1 搜索 (依赖 state + tactic) |
| `engine/lean_bridge/` | miniF2F 问题的 APE 格式定义 |
| `engine/api/` | APE 验证器接口 (部分依赖 core) |

## APE v2 的实际架构

APE v2 的核心价值**不在自建内核**，而在将 Lean4 REPL 变为高性能交互环境：

1. **REPL 连接池** (`async_lean_pool.py`) — N 路并行 REPL 长连接
2. **增量验证** (`incremental_verifier.py`) — env_id fork 实现 50ms 验证
3. **L0 预过滤** (`prefilter.py`) — <10μs 语法检查过滤 90% 无效输出
4. **错误智能层** (`error_intelligence.py`) — ~100 bits 结构化反馈
5. **广播总线** (`broadcast.py`) — 跨智能体实时知识共享

## 不要新增对 legacy 模块的依赖

新代码应直接使用 Lean4 REPL 接口 (`AsyncLeanPool`, `VerificationScheduler`)，
不要导入 `engine.core`、`engine.tactic`、`engine.kernel`。
