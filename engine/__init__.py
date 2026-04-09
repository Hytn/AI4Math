"""APE v2 — Agent-oriented Proof Environment

把 Lean4 本身变成 Agent 的高性能交互环境（不自建简化内核）。

三大核心能力:
  1. Lean4 REPL 连接池 + 增量验证 → 验证延迟 2-12s → 50ms, 精度 100%
  2. 错误智能层 → 每次交互 ~100 bits 结构化反馈 (vs 传统 1 bit pass/fail)
  3. 跨智能体广播 → 失败中提取负面知识, 搜索空间实时剪枝

Active modules (v2-v4):
  broadcast              跨线程实时广播总线 (发布-订阅, 非阻塞)
  async_lean_pool        Lean4 REPL 异步连接池
  prefilter              L0 语法预过滤器 (~1μs, 过滤 ~90% 无效输出)
  error_intelligence     错误智能层 (结构化 AgentFeedback + 修复候选)
  verification_scheduler 自适应三级验证调度 (L0→L1→L2)
  observability          结构化日志 + 指标收集 + Prometheus/JSON 导出
  world_model            世界模型预测器接口 + Mock + Sklearn 实现
  world_model_trainer    世界模型训练管道 (TF-IDF + LogisticRegression)
  lane/                  Proof Lane Runtime (状态机, 事件总线, 策略引擎)

Legacy modules (v1, see engine/LEGACY.md — DO NOT add new dependencies):
  core/       Expr, de Bruijn indices, Universe, Environment
  kernel/     TypeChecker (deprecated)
  tactic/     18 built-in tactics (deprecated)
  state/      ProofState, SearchTree (depends on core)
  search/     MCTS + UCB1 (depends on state + tactic)
"""
__version__ = "0.4.0"

# APE v2 core exports (synchronous)
from engine.broadcast import BroadcastBus, BroadcastMessage, MessageType
from engine.prefilter import PreFilter, FilterResult
from engine.lean_pool import LeanPool, TacticFeedback, FullVerifyResult
from engine.error_intelligence import ErrorIntelligence, AgentFeedback
from engine.verification_scheduler import VerificationScheduler, VerificationResult

# APE v3 async exports
from engine.async_lean_pool import AsyncLeanSession, AsyncLeanPool, SyncLeanPool
from engine.async_verification_scheduler import AsyncVerificationScheduler
from engine.async_factory import AsyncEngineFactory, AsyncEngineComponents

# APE v3 incremental verification + persistent state
from engine.proof_session import ProofSessionManager, ProofSession
from engine.incremental_verifier import IncrementalVerifier, IncrementalResult

# APE v3 elastic scheduling
from engine.pool_scaler import PoolScaler
from engine.resource_scheduler import ResourceScheduler, ResourceBudget, Priority
from engine.remote_session import (
    RemoteSession, LocalTransport, TCPTransport, ElasticPool,
)
from engine.api.protocols import AsyncPoolProtocol

# APE v4 persistent proof context
from engine.proof_context_store import (
    ProofContextStore, StepDetail, RichProofTrajectory,
)

# APE v4 async search
from engine.async_search import AsyncSearchCoordinator
