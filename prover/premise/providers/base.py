"""prover/premise/providers/base.py — Retriever provider 抽象层

设计目标 (与既有代码完全解耦):

  - 既有 ``PremiseSearchTool`` 的默认行为 **零改动**: 不配置 provider
    时, 工具仍走 knowledge_store → 本地 TF-IDF → heuristic 三级降级。
  - Provider 是纯加法: ``PremiseSearchTool(providers=[...])`` 显式注入,
    或经 ``AI4MATH_PREMISE_PROVIDERS`` 环境变量在 tool_kits 组装时注入。
  - 每个 provider 自报告 ``available()``; 不可用时被静默跳过 (但会在
    结果 meta 里记录 degraded 列表, 供评测层聚合"降级比例")。
  - 所有在线 provider 强制走 :class:`RetrievalCache` 快照, 评测可复现。

新增 provider = 新文件 + ``@register_provider`` 装饰器, 不碰任何旧文件。
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPremise:
    """统一的检索结果条目 (所有 provider 的输出归一到此结构)。"""
    name: str                      # 全限定名, e.g. "Nat.add_comm"
    statement: str = ""            # 形式化陈述 (pp 后的类型)
    score: float = 0.0             # provider 自己的相关性分 (越大越相关)
    source: str = ""               # provider name, e.g. "leansearch_v2"
    module: str = ""               # 所在 module, e.g. "Mathlib.Algebra.Group.Basic"
    informal: str = ""             # 非形式化描述 (LeanSearch 系返回)
    kind: str = ""                 # theorem / def / instance / ...

    def to_tool_result_entry(self) -> dict:
        """转成 PremiseSearchTool 既有输出格式的一条 (保持 schema 兼容)。"""
        entry = {
            "name": self.name,
            "type_signature": self.statement,
            "relevance": float(self.score),
            "source": self.source,
        }
        if self.module:
            entry["module"] = self.module
        if self.informal:
            entry["informal"] = self.informal
        return entry


class RetrieverProvider(ABC):
    """单个检索后端的最小契约。

    实现者注意:
      - ``search`` 不允许抛异常到调用方 —— 网络错误内部捕获,
        返回空列表并记 WARNING (premise_search 曾因静默吞错
        隐藏 bug 多个版本, 此处沿用"降级必须可见"的原则)。
      - ``goal_state`` 是可选的形式化证明状态 (LeanStateSearch 等
        state-conditioned 检索器使用; 语义检索器忽略即可)。
    """

    #: 注册名, 子类必须覆写
    name: str = ""

    @abstractmethod
    def search(self, query: str, top_k: int = 10, *,
               goal_state: str = "") -> list[RetrievedPremise]:
        ...

    def available(self) -> bool:
        """provider 当前是否可用 (默认可用; 在线 provider 可探测)。"""
        return True


# ─── 注册表 ──────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[RetrieverProvider]] = {}


def register_provider(cls: type[RetrieverProvider]):
    """类装饰器: 把 provider 类登记进注册表。"""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty .name")
    if cls.name in _REGISTRY:
        raise ValueError(f"Duplicate retriever provider name: {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def provider_names() -> list[str]:
    _ensure_builtin_loaded()
    return sorted(_REGISTRY)


def _ensure_builtin_loaded():
    """惰性 import 内置 provider 模块以触发注册 (避免循环 import)。"""
    # noqa: import 副作用即注册
    from prover.premise.providers import (  # noqa: F401
        local_tfidf, leansearch_v2, loogle, leanstatesearch)


def build_providers(spec: str | list[str] | None = None,
                    *, cache_path: str = "",
                    cache_mode: str = "") -> list[RetrieverProvider]:
    """按 spec 构造 provider 实例列表。

    Args:
        spec: 逗号分隔字符串或列表, e.g. "leansearch_v2,local_tfidf"。
              None/空 → 读 ``AI4MATH_PREMISE_PROVIDERS`` 环境变量;
              环境变量也为空 → 返回 [] (即: 完全保持旧行为)。
        cache_path: 在线检索快照路径 (默认读
              ``AI4MATH_RETRIEVAL_CACHE``, 再默认 results/retrieval_cache.jsonl)。
        cache_mode: "rw" (默认, 读写) / "ro" (只读重放, 评测复现用) /
              "off" (不缓存)。默认读 ``AI4MATH_RETRIEVAL_CACHE_MODE``。

    未知 provider 名记 WARNING 并跳过 —— 不让配置 typo 炸掉主流程。
    """
    if spec is None or spec == "":
        spec = os.environ.get("AI4MATH_PREMISE_PROVIDERS", "")
    if isinstance(spec, str):
        names = [s.strip() for s in spec.split(",") if s.strip()]
    else:
        names = [s.strip() for s in spec if s and s.strip()]
    if not names:
        return []

    _ensure_builtin_loaded()

    from prover.premise.providers.cache import RetrievalCache
    cache = RetrievalCache.from_env(path=cache_path, mode=cache_mode)

    out: list[RetrieverProvider] = []
    for n in names:
        cls = _REGISTRY.get(n)
        if cls is None:
            logger.warning(
                "Unknown retriever provider %r (available: %s) — skipped",
                n, sorted(_REGISTRY))
            continue
        try:
            inst = cls(cache=cache) if _accepts_cache(cls) else cls()
            out.append(inst)
        except Exception as e:  # noqa: BLE001 — 配置错误不应炸主流程
            logger.warning("Failed to construct provider %r: %s", n, e)
    return out


def _accepts_cache(cls) -> bool:
    import inspect
    try:
        return "cache" in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


@dataclass
class MultiRetriever:
    """把多个 provider 串成一个查询面: 依序查询、去重、合并。

    去重规则: 同名 premise 保留先出现者 (provider 顺序即优先级)。
    ``degraded`` 记录本次查询中不可用/查询失败的 provider 名 —
    评测层应聚合此字段, 避免"整场评测都在降级跑"而不自知。
    """
    providers: list[RetrieverProvider] = field(default_factory=list)

    def search(self, query: str, top_k: int = 10, *,
               goal_state: str = "") -> tuple[list[RetrievedPremise], list[str]]:
        results: list[RetrievedPremise] = []
        seen: set[str] = set()
        degraded: list[str] = []
        for p in self.providers:
            if len(results) >= top_k:
                break
            try:
                if not p.available():
                    degraded.append(p.name)
                    continue
                hits = p.search(query, top_k=top_k - len(results),
                                goal_state=goal_state)
            except Exception as e:  # noqa: BLE001
                logger.warning("Provider %r search failed: %s", p.name, e)
                degraded.append(p.name)
                continue
            if not hits:
                continue
            for h in hits:
                if h.name and h.name not in seen:
                    seen.add(h.name)
                    results.append(h)
        return results[:top_k], degraded
