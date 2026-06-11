"""prover.premise.providers — 可插拔检索 provider 层 (纯加法模块)

四个内置 provider:
    local_tfidf      — 既有 KnowledgeTFIDFRetriever + data/premises/*.jsonl
    leansearch_v2    — LeanSearch (leansearch.net) 语义检索, Mathlib SOTA
    loogle           — Loogle 类型签名/模式匹配检索 (与语义检索互补)
    leanstatesearch  — proof-state-conditioned 检索 (experimental)

启用方式 (默认全关, 不配置 = 旧行为零变化):

    # 方式 1: 环境变量 (零代码改动)
    export AI4MATH_PREMISE_PROVIDERS="leansearch_v2,loogle,local_tfidf"

    # 方式 2: 显式注入
    from prover.premise.providers import build_providers
    tool = PremiseSearchTool(providers=build_providers("leansearch_v2"))

评测可复现: 在线检索一律过 RetrievalCache 快照 —
    AI4MATH_RETRIEVAL_CACHE=results/run1/retrieval.jsonl   # 快照路径
    AI4MATH_RETRIEVAL_CACHE_MODE=ro                        # 冻结重放
"""
from prover.premise.providers.base import (
    MultiRetriever,
    RetrievedPremise,
    RetrieverProvider,
    build_providers,
    provider_names,
    register_provider,
)
from prover.premise.providers.cache import RetrievalCache

__all__ = [
    "MultiRetriever",
    "RetrievedPremise",
    "RetrieverProvider",
    "RetrievalCache",
    "build_providers",
    "provider_names",
    "register_provider",
]
