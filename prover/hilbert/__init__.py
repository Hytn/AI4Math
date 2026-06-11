"""prover.hilbert — Hilbert (arXiv:2509.22819) 递归分解证明的独立复现

与 unified profile 体系**零耦合**的独立子系统:
  - 入口: 仓库根目录 ``run_hilbert.py`` (不是 run_unified --profile);
  - 配置: ``config/hilbert.yaml`` (per-role 模型, 不动 Profile dataclass);
  - 依赖面: agent.brain 的 provider 工厂 + 注入式 verifier 回调。

代码参照: Gödel's Poetry (arXiv:2512.14252) — Hilbert 官方未开源,
该工作是当前最接近的开源复现。
"""
from prover.hilbert.config import HilbertConfig, RoleModel
from prover.hilbert.orchestrator import (
    HilbertOrchestrator,
    ProofNode,
    assemble,
    extract_proof_body,
    parse_decomposition,
    statement_head,
)

__all__ = [
    "HilbertConfig",
    "HilbertOrchestrator",
    "ProofNode",
    "RoleModel",
    "assemble",
    "extract_proof_body",
    "parse_decomposition",
    "statement_head",
]
