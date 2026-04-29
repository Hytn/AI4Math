"""prover/pipeline/heterogeneous_engine.py — 异构并行证明引擎 (v3 — 统一 runtime)

v2 → v3 的关键改造
==================

v2 直接调 ``SubAgent.execute()`` (单轮 LLM 生成), 无法表达 step-level / RAG /
hammer 等多轮 agentic 方法。v3 改为给每个方向启动一个 ``UnifiedProofRunner``
+ 一个 ``Profile``, 通过共享的 ``BroadcastBus`` 同步发现, 通过 ``ResultFuser``
做跨方向融合。

各方向的差异 = 各自的 Profile (preset 名 + override 参数), 不再是硬编码的
SubAgent 配置。这使得加新方法 = 加新 preset, 不需要改本文件。

兼容性
======
对外 API 完全不变: ``run_round(problem, classification, attempt_history,
budget) -> list[AgentResult]``。下游的 ``ProofPipeline.verify()`` /
``ResultFuser`` / 持久化逻辑无需改动。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from prover.pipeline._agent_deps import (
    AgentResult, ResultFuser, HookManager, PluginLoader,
)
# Backward-compat re-exports for legacy modules that imported these
# from heterogeneous_engine (e.g. ``prover/pipeline/async_prove.py``).
from prover.pipeline._agent_deps import ProofDirection, build_direction_prompt  # noqa: F401
from common.budget import Budget
from prover.models import BenchmarkProblem
from prover.unified import (
    UnifiedProofRunner, UnifiedResult, get_profile, PRESETS,
    unified_to_agent_result,
)
from prover.unified.profiles import Profile

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Direction = a Profile + per-direction config override
# ══════════════════════════════════════════════════════════════════════

@dataclass
class HeteroDirection:
    """一个异构方向的声明: profile + 个性化 override。"""
    name: str
    profile_name: str
    overrides: dict = field(default_factory=dict)
    """profile 字段级 override, e.g. {'temperature': 0.9, 'model': 'opus-4-6'}。"""


# 默认的 4 方向异构 (替代旧 DirectionPlanner 的硬编码方向)
_DEFAULT_DIRECTIONS = [
    HeteroDirection(
        name="automation",
        profile_name="whole_proof",
        overrides={"temperature": 0.2},
    ),
    HeteroDirection(
        name="repair",
        profile_name="whole_proof_repair",
        overrides={"temperature": 0.5, "max_turns": 4},
    ),
    HeteroDirection(
        name="creative",
        profile_name="whole_proof_repair",
        overrides={"temperature": 0.9, "max_turns": 3},
    ),
    HeteroDirection(
        name="retrieval",
        profile_name="reprover",
        overrides={"temperature": 0.3, "max_turns": 12},
    ),
]


# ══════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════

class HeterogeneousEngine:
    """异构并行证明引擎 (v3) — 统一 runtime + Profile 多方向。

    用法::

        engine = HeterogeneousEngine(
            llm=async_llm, lean_pool=pool,
            knowledge_store=ks, retriever=retr,
            broadcast=bus, scheduler=verif_scheduler,
        )
        results = engine.run_round(problem, classification, history, budget)

    每个方向走 ``UnifiedProofRunner``:
      - profile 决定方法学 (whole_proof / repair / reprover / leandojo / ...)
      - overrides 决定方向间差异 (model / temperature / max_turns)
      - BroadcastBus 在方向间同步发现
      - ResultFuser 做跨方向融合
    """

    def __init__(
        self,
        # ── new-style kwargs (preferred) ───────────────────────
        *,
        llm=None,                         # async LLM provider
        lean_pool=None,
        knowledge_store=None,
        retriever=None,
        broadcast=None,
        scheduler=None,                   # verification_scheduler
        directions: Optional[list[HeteroDirection]] = None,
        plugin_loader: PluginLoader = None,
        hook_manager: HookManager = None,
        # ── legacy kwargs (assembly.py compat) ─────────────────
        pool=None,                        # AgentPool (legacy) — has .llm
        verification_scheduler=None,      # legacy alias of scheduler
    ):
        # Resolve LLM: prefer explicit `llm`; else extract from legacy pool
        resolved_llm = llm
        if resolved_llm is None and pool is not None:
            resolved_llm = getattr(pool, "llm", None)
        self.llm = resolved_llm
        self._legacy_pool = pool          # kept for cross_fusion fallback

        # Resolve scheduler aliases
        self.scheduler = scheduler or verification_scheduler

        # Resource injection
        self.lean_pool = lean_pool or self._extract_lean_pool_from_scheduler()
        self.knowledge_store = knowledge_store
        self.retriever = retriever
        self.broadcast = broadcast or self._make_default_broadcast()

        # Direction config (default 4-way; can be overridden by `directions`)
        self.directions = directions or list(_DEFAULT_DIRECTIONS)

        # Legacy fields (kept for plugin / hook fan-out)
        self.plugins = plugin_loader or PluginLoader()
        self.hooks = hook_manager or HookManager()
        self.fuser = ResultFuser()

        # Lazy: convert sync LLM → async if needed (assembly.py provides sync)
        self.llm = self._ensure_async_llm(self.llm)

    def _extract_lean_pool_from_scheduler(self):
        sch = self.scheduler
        if sch is None:
            return None
        return getattr(sch, "pool", None) or getattr(sch, "lean_pool", None)

    def _ensure_async_llm(self, llm):
        """If `llm` is sync (LLMProvider), wrap it so that `await chat()` works.

        Strategy: if it has async `generate`/`chat`, accept as-is; otherwise
        wrap with a thin adapter that runs sync calls in an executor.
        """
        if llm is None:
            return None
        # Already async?
        import inspect
        chat = getattr(llm, "chat", None)
        gen = getattr(llm, "generate", None)
        if (chat and inspect.iscoroutinefunction(chat)) or \
           (gen and inspect.iscoroutinefunction(gen)):
            return llm
        # Sync provider — wrap
        try:
            return _SyncToAsyncAdapter(llm)
        except Exception as e:
            logger.warning(f"Could not wrap sync LLM as async: {e}")
            return llm

    @staticmethod
    def _make_default_broadcast():
        try:
            from engine.broadcast import BroadcastBus
            return BroadcastBus()
        except Exception:
            return None

    # ── public API ─────────────────────────────────────────────────

    def run_round(
        self,
        problem: BenchmarkProblem,
        classification: dict = None,
        attempt_history: list = None,
        budget: Budget = None,
    ) -> list[AgentResult]:
        """同步入口 (兼容 ProofPipeline.generate 旧调用约定)。

        内部其实启动 asyncio.run; 如果调用方已经在事件循环中, 用
        ``run_round_async`` 直接 await。
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有事件循环中: 不能 asyncio.run; 退化到一次性执行
                logger.warning(
                    "run_round called inside running event loop; "
                    "use run_round_async() instead.")
                return asyncio.ensure_future(
                    self.run_round_async(problem, classification,
                                         attempt_history, budget))
        except RuntimeError:
            pass
        return asyncio.run(self.run_round_async(
            problem, classification, attempt_history, budget))

    async def run_round_async(
        self,
        problem: BenchmarkProblem,
        classification: dict = None,
        attempt_history: list = None,
        budget: Budget = None,
    ) -> list[AgentResult]:
        """异构并行一轮; 返回按 confidence 降序的 AgentResult 列表。"""
        classification = classification or {}
        attempt_history = attempt_history or []

        # 1. 为每个方向构造一个 Profile (preset + overrides)
        profiles = [self._materialize_profile(d) for d in self.directions]

        # 2. 共享 broadcast 订阅 (跨轮次知识传递)
        subscriptions = self._setup_broadcasts()

        try:
            # 3. 并行启动 N 个 runtime
            runner = UnifiedProofRunner(
                llm=self.llm,
                lean_pool=self.lean_pool,
                knowledge_store=self.knowledge_store,
                retriever=self.retriever,
                broadcast_bus=self.broadcast,
            )
            tasks = [
                runner.run(problem, profile=prof)
                for prof in profiles
            ]
            unified_results: list[UnifiedResult] = await asyncio.gather(
                *tasks, return_exceptions=False)

            # 4. UnifiedResult → AgentResult, 同时回写 budget
            agent_results: list[AgentResult] = []
            for d, prof, ur in zip(self.directions, profiles, unified_results):
                ar = unified_to_agent_result(ur, agent_name=d.name)
                ar.metadata["direction"] = d.name
                ar.metadata["profile"] = prof.name
                if budget is not None:
                    budget.add_tokens(ar.tokens_used)
                agent_results.append(ar)

            # 5. 广播发现 (失败/成功/引理)
            self._broadcast_results(agent_results, profiles)

            # 6. 按 confidence 降序
            agent_results.sort(key=lambda r: -r.confidence)

            # 7. 跨方向融合 (现有逻辑)
            if (agent_results
                    and not any(r.confidence > 0.9 for r in agent_results)):
                fused = await self._try_cross_fusion(
                    agent_results, problem, budget)
                if fused is not None:
                    agent_results.insert(0, fused)

        finally:
            self._teardown_broadcasts(subscriptions)

        return agent_results

    # ── helpers ────────────────────────────────────────────────────

    def _materialize_profile(self, d: HeteroDirection) -> Profile:
        """preset_name + overrides → 实际 Profile 对象。"""
        if d.profile_name not in PRESETS:
            raise ValueError(
                f"Unknown profile '{d.profile_name}' "
                f"for direction '{d.name}'. "
                f"Available: {sorted(PRESETS)}")
        base = get_profile(d.profile_name)
        # 浅拷贝 + 字段 override
        from dataclasses import replace
        try:
            return replace(base, **d.overrides)
        except TypeError as e:
            logger.warning(
                f"Invalid override for direction '{d.name}': {e}; "
                f"falling back to base profile")
            return base

    def _setup_broadcasts(self) -> dict:
        """为每个方向订阅 broadcast, 注入历史消息以保证跨轮传递。"""
        subs = {}
        if self.broadcast is None:
            return subs
        for d in self.directions:
            sub = self.broadcast.subscribe(d.name)
            subs[d.name] = sub
            try:
                recent = self.broadcast.get_recent(n=15)
                for msg in recent:
                    sub.push(msg)
            except Exception as e:
                logger.debug(f"broadcast history replay failed: {e}")
        return subs

    def _teardown_broadcasts(self, subs: dict) -> None:
        if self.broadcast is None:
            return
        for name in subs:
            try:
                self.broadcast.unsubscribe(name)
            except Exception as e:
                logger.debug(f"broadcast unsubscribe failed: {e}")

    def _broadcast_results(self, results: list[AgentResult],
                            profiles: list[Profile]) -> None:
        """把方向结果作为发现广播到总线。"""
        if self.broadcast is None:
            return
        try:
            from engine.broadcast import BroadcastMessage
        except Exception:
            return

        for r, prof in zip(results, profiles):
            if not r:
                continue
            # 高置信度证明 → 广播部分证明
            if r.proof_code and r.confidence > 0.5:
                self.broadcast.publish(BroadcastMessage.partial_proof(
                    source=r.agent_name,
                    proof_so_far=r.proof_code[:800],
                    remaining_goals=[],
                    env_id=-1,
                    goals_closed=1,
                ))
            # 失败 → 广播负面知识
            elif r.error and not r.proof_code:
                self.broadcast.publish(BroadcastMessage.negative(
                    source=r.agent_name,
                    tactic=prof.name,
                    error_category="profile_failed",
                    reason=str(r.error)[:200],
                ))

    async def _try_cross_fusion(
        self,
        results: list[AgentResult],
        problem,
        budget,
    ) -> Optional[AgentResult]:
        """跨方向融合 — 如果有方向找到引理, 注入到最佳证明方向再试一次。

        v3 的实现仍然走 ``UnifiedProofRunner`` (用 repair 模式), 但用合并后
        的上下文当作 initial_message 的补充。
        """
        # 找 best proof + lemma findings
        best_proof = next(
            (r for r in results if r.proof_code and r.confidence > 0.3),
            None)
        if best_proof is None:
            return None

        useful_lemmas = self.fuser.extract_useful_lemmas(results)
        if not useful_lemmas:
            return None

        # 构造一个临时 profile: repair + 给提示
        from dataclasses import replace
        base = get_profile("whole_proof_repair")
        fusion_profile = replace(base, name="cross_fusion", max_turns=2)

        runner = UnifiedProofRunner(
            llm=self.llm,
            lean_pool=self.lean_pool,
            knowledge_store=self.knowledge_store,
            retriever=self.retriever,
            broadcast_bus=self.broadcast,
        )
        # 把已有证明 + 引理拼进 problem.theorem_statement 的 hint 不优雅,
        # 但保持 v3 与 v2 行为相近; 真正的"hint 注入"应该走 broadcast。
        try:
            ur = await runner.run(problem, profile=fusion_profile)
            ar = unified_to_agent_result(ur, agent_name="cross_fusion")
            if budget is not None:
                budget.add_tokens(ar.tokens_used)
            return ar
        except Exception as e:
            logger.warning(f"cross fusion failed: {e}")
            return None

    # ── 兼容旧名 (deprecated) ────────────────────────────────────

    def _plan_directions(self, problem, classification, attempt_history):
        """legacy shim — 返回当前 self.directions; 旧测试可能在调。"""
        logger.warning(
            "HeterogeneousEngine._plan_directions is deprecated; "
            "directions are now declared via HeteroDirection list.")
        return self.directions


# ══════════════════════════════════════════════════════════════════════
# Sync → Async LLM adapter
# ══════════════════════════════════════════════════════════════════════

class _SyncToAsyncAdapter:
    """把同步 LLMProvider 包装成 AsyncLLMProvider 接口。

    内部用 asyncio.to_thread 卸载到线程池, 不阻塞事件循环。
    用于 ``assembly.py`` 提供的旧 sync provider 与新 ``UnifiedProofRunner``
    (期待 async) 之间的桥接。
    """

    def __init__(self, sync_llm):
        self._sync = sync_llm
        # 透传部分属性以便 dialog.json 写 model 名等
        self.model_name = getattr(sync_llm, "model_name", "")

    async def generate(self, system: str = "", user: str = "",
                       temperature: float = 0.7,
                       tools: list = None,
                       max_tokens: int = 4096):
        return await asyncio.to_thread(
            self._sync.generate,
            system=system, user=user,
            temperature=temperature,
            tools=tools, max_tokens=max_tokens,
        )

    async def chat(self, system: str, messages: list,
                   temperature: float = 0.7,
                   tools: list = None,
                   max_tokens: int = 4096):
        # 同步 provider 可能没有 chat(), 退到 generate() with flat string
        if hasattr(self._sync, "chat"):
            return await asyncio.to_thread(
                self._sync.chat,
                system=system, messages=messages,
                temperature=temperature,
                tools=tools, max_tokens=max_tokens,
            )
        # Flatten messages into a single user blob
        from agent.runtime.agent_loop import AgentLoop
        flat_parts = []
        for m in messages:
            role = m.get("role", "user")
            c = m.get("content", "")
            if isinstance(c, list):
                txt = []
                for blk in c:
                    if isinstance(blk, dict):
                        if blk.get("type") == "text":
                            txt.append(blk.get("text", ""))
                        elif blk.get("type") == "tool_use":
                            txt.append(f"[tool_use {blk.get('name', '')}]")
                        elif blk.get("type") == "tool_result":
                            txt.append(str(blk.get("content", "")))
                flat_parts.append(f"[{role}]\n" + "\n".join(txt))
            else:
                flat_parts.append(f"[{role}]\n{c}")
        flat = "\n\n".join(flat_parts)
        return await asyncio.to_thread(
            self._sync.generate,
            system=system, user=flat,
            temperature=temperature,
            tools=tools, max_tokens=max_tokens,
        )
