"""APE v2 — Agent-oriented Proof Environment

不自建简化内核, 而是把 Lean4 本身变成 Agent 的高性能交互环境。

三大核心能力:
  1. Lean4 REPL 连接池 + 跨线程实时广播 → 验证延迟 2-12s → 50ms, 精度 100%
  2. 错误智能层 → 每次交互 ~100 bits 结构化反馈 (vs 传统 1 bit pass/fail)
  3. 单题内知识积累 → 失败中提取负面知识 + 辅助引理, 搜索空间指数收缩

Core modules:
  broadcast              跨线程实时广播总线 (发布-订阅, 非阻塞)
  lean_pool              Lean4 REPL 连接池 (N 路并行, 环境预加载)
  prefilter              L0 语法预过滤器 (~1μs, 过滤 ~90% 无效输出)
  error_intelligence     错误智能层 (结构化 AgentFeedback + 修复候选)
  verification_scheduler 自适应三级验证调度 (L0→L1→L2, 自动广播)

Legacy modules (retained for heuristic planning):
  core/       Expr, de Bruijn indices, Universe, Environment
  kernel/     TypeChecker (heuristic pre-evaluation)
  state/      ProofState, SearchTree (persistent data structures)
  search/     MCTS + UCB1 + search strategies
  tactic/     18 built-in tactics (heuristic scoring)
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
