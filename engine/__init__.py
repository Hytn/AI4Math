"""engine/ — Lean 4 验证引擎层 (Verification OS)

把 Lean 4 REPL 包装成高性能、可编程的交互环境。所有上层 (agent/, prover/)
通过此层与 Lean 交互, 不直接调 Lean 子进程。

核心模块:
  async_lean_pool                Lean 4 REPL 异步连接池 + AsyncLeanSession
  async_verification_scheduler   自适应三级验证调度 (L0 → L1 → L2)
  prefilter                      L0 语法预过滤 (~1μs)
  error_intelligence             stderr → 结构化 AgentFeedback (~100 bits)
  transport                      Lean transport 抽象 (Local/Socket/Mock/Fallback)
  backends/                      社区 Lean 4 backend (Kimina/Pantograph/LooKeng)
  proof_context_store            步级 trajectory 落盘 (SQLite)
  _core                          共享数据类型 + classify_error 一份

预留接口模块 (核心三件套之外、明确保留的功能位):
  broadcast                      多智能体 rollout 广播总线 (发布-订阅)
  world_model                    世界模型预测器 (sklearn 包装, 留给训练替换)
"""
__version__ = "0.10.0"

from engine._core import (
    TacticFeedback, FullVerifyResult, VerificationResult,
)
from engine.prefilter import PreFilter, FilterResult
from engine.error_intelligence import ErrorIntelligence, AgentFeedback
from engine.async_lean_pool import AsyncLeanSession, AsyncLeanPool, SyncLeanPool
from engine.async_verification_scheduler import AsyncVerificationScheduler
from engine.proof_context_store import (
    ProofContextStore, StepDetail, RichProofTrajectory,
)

# Reserved interfaces (intentionally exposed):
from engine.broadcast import BroadcastBus, BroadcastMessage, MessageType
from engine.world_model import (
    WorldModelPredictor, WorldModelPrediction,
    MockWorldModel, make_world_model,
)

# Backward-compat alias: legacy callers used `from engine import LeanPool`
LeanPool = SyncLeanPool
