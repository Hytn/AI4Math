"""prover/pipeline/heterogeneous_engine.py — 异构并行证明引擎

当前系统的核心瓶颈: RolloutEngine 用同一个 prompt 生成 N 个样本,
所有样本往往犯同一类错误 (如 ℕ 减法用 ring)。

本引擎的改进: 同时启动多个策略方向完全不同的子智能体,
各有独立的角色/模型/prompt/上下文, 实现真正的策略多样性。

典型的四方向探索::

    方向 A: 自动化探测 (Haiku, 低温, 纯 tactic)
    方向 B: 归纳法专家 (Sonnet, 中温, 领域 prompt)
    方向 C: 代数变换   (Sonnet, 高温, 替代路径)
    方向 D: 引理检索   (Sonnet, 低温, 搜索 Mathlib)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

from prover.pipeline._agent_deps import AgentSpec, AgentTask, AgentResult, ContextItem
from prover.pipeline._agent_deps import AgentPool
from prover.pipeline._agent_deps import ResultFuser
from common.roles import AgentRole
from common.hook_types import HookEvent, HookContext
from prover.pipeline._agent_deps import HookManager
from prover.pipeline._agent_deps import PluginLoader
from common.budget import Budget
from prover.models import BenchmarkProblem, ProofAttempt, AttemptStatus

logger = logging.getLogger(__name__)


# ProofDirection 定义已移至 agent.strategy.direction_planner
# 此处 re-export 以保持向后兼容
from prover.pipeline._agent_deps import ProofDirection, build_direction_prompt


class HeterogeneousEngine:
    """异构并行证明引擎 (v2 — 集成广播总线 + 验证调度器)

    替代 RolloutEngine 的同质化并行采样:
    - 每个方向是一个独立的 SubAgent
    - 不同方向有不同的角色、模型、温度、prompt
    - ResultFuser 融合结果, 支持跨方向信息注入

    v2 新增:
    - BroadcastBus: 一个方向的发现实时广播给所有其他方向
    - VerificationScheduler: L0/L1/L2 三级验证, 结构化反馈
    - 知识积累: 负面知识 + 辅助引理跨方向共享
    """

    def __init__(self, pool: AgentPool, plugin_loader: PluginLoader = None,
                 hook_manager: HookManager = None, retriever=None,
                 broadcast: 'BroadcastBus' = None,
                 verification_scheduler: 'VerificationScheduler' = None,
                 direction_planner: 'DirectionPlanner' = None):
        self.pool = pool
        self.plugins = plugin_loader or PluginLoader()
        self.hooks = hook_manager or HookManager()
        self.retriever = retriever
        self.fuser = ResultFuser()

        # ── APE v2 集成 ──
        from engine.broadcast import BroadcastBus
        from engine.verification_scheduler import VerificationScheduler
        self.broadcast = broadcast or BroadcastBus()
        self.scheduler = verification_scheduler

        # ── 方向规划器 (v3: 独立可替换) ──
        if direction_planner:
            self.planner = direction_planner
        else:
            from agent.strategy.direction_planner import DirectionPlanner
            self.planner = DirectionPlanner(
                retriever=retriever, plugin_loader=plugin_loader)

    def run_round(self, problem: BenchmarkProblem,
                  classification: dict = None,
                  attempt_history: list = None,
                  budget: Budget = None) -> list[AgentResult]:
        """运行一轮异构并行证明 (v2 — 集成广播总线)

        v2 改进:
        - 每个方向启动前, 注入来自广播总线的跨方向知识
        - 每个方向完成后, 将发现/失败广播给所有其他方向
        - 下一轮所有方向自动获得上一轮的全部发现

        Args:
            problem: 待证明的问题
            classification: DomainClassifierHook 的分类结果
            attempt_history: 之前的尝试历史 (用于 repair 方向)
            budget: 预算控制器

        Returns:
            所有方向的结果列表 (按 confidence 降序)
        """
        classification = classification or {}
        attempt_history = attempt_history or []

        # 1. 规划探索方向 (委托给 DirectionPlanner)
        directions = self.planner.plan(
            problem, classification, attempt_history)

        # 2. 为每个方向注册广播订阅
        #    P1-8: 新订阅者注入历史消息, 确保跨轮次知识传递
        subscriptions = {}
        for d in directions:
            sub = self.broadcast.subscribe(d.name)
            subscriptions[d.name] = sub
            # 将之前轮次的历史消息补充到新订阅的队列中
            recent = self.broadcast.get_recent(n=15)
            for msg in recent:
                sub.push(msg)

        # 3. 构建 (spec, task) 对 — 注入广播消息
        specs_and_tasks = []
        try:
            for d in directions:
                spec = AgentSpec(
                    name=d.name,
                    role=d.role,
                    model=d.model,
                    temperature=d.temperature,
                    few_shot_override=d.few_shot_override,
                    tools=d.allowed_tools,
                )

                context_items = [
                    ContextItem("theorem", problem.theorem_statement, 1.0,
                                "theorem_statement"),
                ]
                if d.strategic_hint:
                    context_items.append(
                        ContextItem("strategy", d.strategic_hint, 0.9,
                                    "tactic_hint"))
                if d.selected_premises:
                    premises_text = "\n".join(
                        f"- {p}" for p in d.selected_premises[:15])
                    context_items.append(
                        ContextItem("premises", premises_text, 0.7, "premise"))

                # 注入 hook 产生的上下文 (如 ℕ 减法警告)
                domain_hints = classification.get("domain_hints", {})
                for hk, hv in domain_hints.items():
                    context_items.append(
                        ContextItem(hk, str(hv), 0.85, "premise"))

                # ── v2: 注入来自广播总线的跨方向知识 ──
                broadcast_context = self.broadcast.render_for_prompt(
                    d.name, max_messages=8, max_chars=1500,
                    current_goal=problem.theorem_statement)
                if broadcast_context:
                    context_items.append(
                        ContextItem("teammate_discoveries", broadcast_context,
                                    0.95, "premise"))

                # ── v2: 注入错误智能层的积累知识 ──
                if self.scheduler and self.scheduler.error_intel:
                    dead_ends = self.scheduler.error_intel.get_accumulated_knowledge(5)
                    if dead_ends:
                        context_items.append(
                            ContextItem("known_dead_ends", dead_ends,
                                        0.9, "premise"))

                task = AgentTask(
                    description=self._build_direction_prompt(d, problem),
                    injected_context=context_items,
                    theorem_statement=problem.theorem_statement,
                    metadata={"direction": d.name},
                )
                specs_and_tasks.append((spec, task))

            # 4. 并行执行
            results = self.pool.run_parallel(specs_and_tasks, budget)

            # 4.5 ── v2: 验证每个结果并更新置信度 ──
            pool_stats = self.scheduler.pool.stats() if (
                self.scheduler and self.scheduler.pool) else {}
            has_real_repl = self.scheduler and not pool_stats.get("all_fallback", True)

            if has_real_repl:
                for i, (result, direction) in enumerate(zip(results, directions)):
                    if result.proof_code and result.proof_code.strip():
                        try:
                            vr = self.scheduler.verify_complete(
                                theorem=problem.theorem_statement,
                                proof=result.proof_code,
                                direction=direction.name,
                            )
                            from prover.pipeline._agent_deps import ConfidenceEstimator
                            result.confidence = ConfidenceEstimator.refine_confidence(
                                result,
                                feedback=vr.feedback,
                                l0_passed=vr.l0_passed,
                                l1_passed=(vr.level_reached in ("L1", "L2") and vr.success),
                                l2_passed=vr.l2_verified,
                            )
                            result.metadata["verification"] = {
                                "success": vr.success,
                                "level": vr.level_reached,
                                "feedback_text": vr.feedback.to_prompt(max_chars=500),
                                "env_id": vr.l1_env_id,
                                "goals_remaining": vr.l1_goals_remaining,
                            }
                            if vr.success:
                                result.success = True
                        except Exception as e:
                            logger.warning(
                                f"Verification failed for {direction.name}: {e}")
                            result.metadata["verification"] = {
                                "success": False,
                                "level": "error",
                                "error": str(e),
                            }
            else:
                for result in results:
                    result.metadata["verification"] = {
                        "success": False,
                        "level": "none",
                        "reason": "no_real_repl",
                    }
                    result.confidence = min(result.confidence, 0.4)

            # 5. ── v2: 分析结果并广播发现 ──
            self._broadcast_results(results, directions)

            # 6. 按 confidence 排序
            results.sort(key=lambda r: -r.confidence)

            # 7. 尝试跨方向融合
            if results and not any(r.confidence > 0.9 for r in results):
                fused = self._try_cross_fusion(results, problem, budget)
                if fused:
                    results.insert(0, fused)

        finally:
            # 8. 清理订阅 (即使上面抛出异常也保证执行)
            for name in subscriptions:
                try:
                    self.broadcast.unsubscribe(name)
                except Exception:
                    pass

        return results

    def _broadcast_results(self, results: list[AgentResult],
                           directions: list[ProofDirection]):
        """分析结果并通过广播总线共享发现

        这是跨方向知识共享的核心机制:
        - 方向 A 生成了高置信度的证明 → 广播 PARTIAL_PROOF
        - 方向 D 的内容中提到了有用引理 → 广播 POSITIVE_DISCOVERY
        - 方向 C 完全失败且明确了原因 → 广播 NEGATIVE_KNOWLEDGE
        """
        from engine.broadcast import BroadcastMessage

        for result, direction in zip(results, directions):
            if not result:
                continue

            name = direction.name

            # 高置信度证明 → 广播部分证明 (含 env_id 供其他方向 fork)
            if result.proof_code and result.confidence > 0.5:
                # 从验证结果中提取 env_id 和剩余 goals
                vr = result.metadata.get("verification", {})
                env_id = -1
                remaining_goals = []
                if vr.get("success"):
                    # L1 验证成功时, feedback_text 中可能包含 env_id
                    # 但更可靠的方式是从 VerificationResult 中获取
                    pass
                # 尝试从 metadata 中获取 L1 返回的 env_id
                if isinstance(vr, dict):
                    env_id = vr.get("env_id", -1)

                self.broadcast.publish(BroadcastMessage.partial_proof(
                    source=name,
                    proof_so_far=result.proof_code[:800],
                    remaining_goals=remaining_goals,
                    env_id=env_id,
                    goals_closed=1,
                ))

            # 内容中包含引理发现 → 提取并广播
            if result.content:
                lemmas = self._extract_lemma_mentions(result.content)
                for lemma in lemmas[:3]:
                    self.broadcast.publish(BroadcastMessage.positive(
                        source=name,
                        discovery=f"Useful lemma: {lemma}",
                        lemma_name=lemma,
                    ))

            # 失败且有明确错误 → 广播负面知识
            if result.error and not result.proof_code:
                self.broadcast.publish(BroadcastMessage.negative(
                    source=name,
                    tactic=result.metadata.get("direction", name),
                    error_category="strategy_failed",
                    reason=result.error[:200],
                ))

        # ── 集成 share_lemma(): 将已验证引理注入所有 REPL 环境 ──
        # 当广播总线中出现 LEMMA_PROVEN 消息时, 将引理代码注入 REPL 池,
        # 使所有方向在后续轮次中可以直接 `exact lemma_name` 引用。
        if self.scheduler and self.scheduler.pool:
            from engine.broadcast import MessageType
            recent_lemmas = self.broadcast.get_recent(
                n=10, msg_type=MessageType.LEMMA_PROVEN)
            for msg in recent_lemmas:
                lemma_name = msg.structured.get("lemma_name", "")
                lemma_stmt = msg.structured.get("lemma_statement", "")
                lemma_proof = msg.structured.get("lemma_proof", "")
                if lemma_name and lemma_stmt and lemma_proof:
                    new_envs = self.scheduler.pool.share_lemma(
                        "", name=lemma_name,
                        statement=lemma_stmt, proof=lemma_proof)
                    if new_envs:
                        logger.info(
                            f"share_lemma: injected '{lemma_name}' "
                            f"into {len(new_envs)} REPL sessions"
                        )
                elif lemma_stmt and lemma_proof:
                    # Fallback: no name, send as raw code
                    full_code = f"{lemma_stmt} {lemma_proof}".strip()
                    new_envs = self.scheduler.pool.share_lemma(full_code)
                    if new_envs:
                        logger.info(
                            f"share_lemma: injected unnamed lemma "
                            f"into {len(new_envs)} REPL sessions"
                        )


    def _extract_lemma_mentions(self, content: str) -> list[str]:
        """从 LLM 输出中提取提到的 Mathlib 引理名"""
        import re
        # 匹配 Namespace.lemma_name 模式
        pattern = r'\b([A-Z][a-zA-Z]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b'
        matches = re.findall(pattern, content)
        # 过滤掉太短或太通用的
        return [m for m in set(matches) if len(m) > 5 and "." in m][:5]

    def _plan_directions(self, problem, classification,
                         attempt_history) -> list[ProofDirection]:
        """根据问题特征规划 2-4 个探索方向

        .. deprecated:: v4
            直接使用 self.planner.plan() 代替。
        """
        return self.planner.plan(problem, classification, attempt_history)

    def _try_cross_fusion(self, results, problem, budget):
        """尝试将一个方向的发现注入另一个方向

        典型场景: 检索方向找到了有用引理, 注入到结构化方向的修复上下文中。
        """
        # 找到最佳结构化结果和有引理发现的结果
        best_proof_result = None
        lemma_results = []

        for r in results:
            if r.proof_code and r.confidence > 0.3:
                if best_proof_result is None:
                    best_proof_result = r
            if r.content and ("lemma" in r.content.lower()
                              or "theorem" in r.content.lower()):
                lemma_results.append(r)

        if not best_proof_result or not lemma_results:
            return None

        # 构建融合修复任务
        lemma_insights = self.fuser.merge_insights(lemma_results, 500)
        useful_lemmas = self.fuser.extract_useful_lemmas(results)

        repair_spec = AgentSpec(
            name="cross_fusion_repair",
            role=AgentRole.REPAIR_AGENT,
            temperature=0.5,
        )

        repair_task = AgentTask(
            description=(
                f"A previous attempt generated this proof:\n"
                f"```lean\n{best_proof_result.proof_code[:1000]}\n```\n\n"
                f"Teammates found these potentially useful insights:\n"
                f"{lemma_insights}\n\n"
                f"Potentially useful Mathlib lemmas: "
                f"{', '.join(useful_lemmas[:10])}\n\n"
                f"Fix the proof using these insights."
            ),
            injected_context=[
                ContextItem("theorem", problem.theorem_statement, 1.0),
            ],
        )

        return self.pool.run_single(repair_spec, repair_task, budget)

    def _get_premises(self, theorem: str) -> list[str]:
        """获取前提引理

        兼容两种 retriever 返回格式:
          - KnowledgeRetriever.retrieve() → list[str]  ("name: statement")
          - PremiseSelector.retrieve()    → list[dict] ({"name": ..., "statement": ...})
        """
        if not self.retriever:
            return []
        try:
            results = self.retriever.retrieve(theorem, top_k=10)
            if not results:
                return []
            # 如果返回的已经是字符串列表, 直接使用
            if isinstance(results[0], str):
                return results
            # 如果返回的是字典列表, 提取 statement
            return [r.get("statement", r.get("name", ""))
                    for r in results if isinstance(r, dict)]
        except Exception as e:
            logger.warning(f"Premise retrieval failed: {e}")
            return []

    def _build_direction_prompt(self, direction, problem) -> str:
        """为每个方向构建定制化的 prompt

        .. deprecated:: v4
            直接使用 agent.strategy.direction_planner.build_direction_prompt()。
        """
        from agent.strategy.direction_planner import build_direction_prompt
        return build_direction_prompt(direction, problem)
