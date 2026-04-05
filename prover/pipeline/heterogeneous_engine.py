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

from agent.runtime.sub_agent import AgentSpec, AgentTask, AgentResult, ContextItem
from agent.runtime.agent_pool import AgentPool
from agent.runtime.result_fuser import ResultFuser
from agent.brain.roles import AgentRole
from agent.hooks.hook_types import HookEvent, HookContext
from agent.hooks.hook_manager import HookManager
from agent.plugins.loader import PluginLoader
from agent.strategy.budget_allocator import Budget
from prover.models import BenchmarkProblem, ProofAttempt, AttemptStatus

logger = logging.getLogger(__name__)


def _role_from_string(role_str: str) -> AgentRole:
    """将配置文件中的字符串角色名映射到 AgentRole 枚举"""
    mapping = {
        "proof_generator": AgentRole.PROOF_GENERATOR,
        "proof_planner": AgentRole.PROOF_PLANNER,
        "critic": AgentRole.CRITIC,
        "repair_agent": AgentRole.REPAIR_AGENT,
    }
    result = mapping.get(role_str.lower())
    if result is None:
        logger.warning(f"Unknown role '{role_str}', defaulting to PROOF_GENERATOR")
        return AgentRole.PROOF_GENERATOR
    return result


@dataclass
class ProofDirection:
    """一个证明探索方向的完整规格"""
    name: str
    role: AgentRole
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    strategic_hint: str = ""
    selected_premises: list[str] = field(default_factory=list)
    few_shot_override: str = ""
    allowed_tools: list[str] = field(default_factory=list)


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
                 config: dict = None):
        self.pool = pool
        self.plugins = plugin_loader or PluginLoader()
        self.hooks = hook_manager or HookManager()
        self.retriever = retriever
        self.fuser = ResultFuser()
        self.config = config or {}

        # ── APE v2 集成 ──
        from engine.broadcast import BroadcastBus
        from engine.verification_scheduler import VerificationScheduler
        self.broadcast = broadcast or BroadcastBus()
        self.scheduler = verification_scheduler

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

        # 1. 规划探索方向
        directions = self._plan_directions(
            problem, classification, attempt_history)

        # 2. 为每个方向注册广播订阅
        subscriptions = {}
        for d in directions:
            subscriptions[d.name] = self.broadcast.subscribe(d.name)

        # 3. 构建 (spec, task) 对 — 注入广播消息
        specs_and_tasks = []
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
                d.name, max_messages=8, max_chars=1500)
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
        # 在广播和排序之前, 将每个证明候选通过 VerificationScheduler
        # 进行 L0/L1 验证, 并用实际反馈更新 confidence。
        # 这确保 ResultFuser 的排序基于验证结果, 而不仅是代码结构特征。
        #
        # 重要: 仅当真实 Lean4 REPL 可用时才执行此步骤。
        # 在 fallback 模式下, 验证始终返回 success=False, 会错误地
        # 降低所有候选的置信度。
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
                        # 用验证结果更新置信度
                        from agent.runtime.sub_agent import SubAgent
                        result.confidence = SubAgent.refine_confidence(
                            result,
                            feedback=vr.feedback,
                            l0_passed=vr.l0_passed,
                            l1_passed=(vr.level_reached in ("L1", "L2") and vr.success),
                            l2_passed=vr.l2_verified,
                        )
                        # 将验证反馈注入 metadata 供下游使用
                        result.metadata["verification"] = {
                            "success": vr.success,
                            "level": vr.level_reached,
                            "feedback_text": vr.feedback.to_prompt(max_chars=500),
                        }
                        if vr.success:
                            result.success = True
                    except Exception as e:
                        logger.warning(
                            f"Verification failed for {direction.name}: {e}")

        # 5. ── v2: 分析结果并广播发现 ──
        self._broadcast_results(results, directions)

        # 6. 按 confidence 排序
        results.sort(key=lambda r: -r.confidence)

        # 7. 尝试跨方向融合 — 如果最佳结果接近成功但缺引理
        if results and not any(r.confidence > 0.9 for r in results):
            fused = self._try_cross_fusion(results, problem, budget)
            if fused:
                results.insert(0, fused)

        # 8. 清理订阅
        for name in subscriptions:
            self.broadcast.unsubscribe(name)

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

            # 高置信度证明 → 广播部分证明
            if result.proof_code and result.confidence > 0.5:
                self.broadcast.publish(BroadcastMessage.partial_proof(
                    source=name,
                    proof_so_far=result.proof_code[:800],
                    remaining_goals=[],
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
                lemma_code = msg.structured.get("lemma_proof", "")
                lemma_stmt = msg.structured.get("lemma_statement", "")
                if lemma_code and lemma_stmt:
                    full_lemma = f"{lemma_stmt} {lemma_code}"
                    new_envs = self.scheduler.pool.share_lemma(full_lemma)
                    if new_envs:
                        logger.info(
                            f"share_lemma: injected '{msg.structured.get('lemma_name', '?')}' "
                            f"into {len(new_envs)} REPL sessions"
                        )

        # ── 集成 fork_env(): 将部分证明的 env_id 广播给其他方向 ──
        # fork_env() 在 lean4-repl 中是零成本的 (直接复用 env_id),
        # 其他方向可以从这个 env_id 继续, 而不必从头开始。
        if self.scheduler and self.scheduler.pool:
            from engine.broadcast import MessageType
            partial_proofs = self.broadcast.get_recent(
                n=5, msg_type=MessageType.PARTIAL_PROOF)
            for msg in partial_proofs:
                env_id = msg.structured.get("env_id", -1)
                if env_id >= 0:
                    # fork_env 在 lean4-repl 中是零成本的, 直接返回原 env_id
                    forked = self.scheduler.pool.fork_env(env_id)
                    logger.debug(
                        f"fork_env: env_id={env_id} forked as {forked} "
                        f"for continuation by other directions"
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

        优先从 self.config["directions"] 读取方向配置;
        如果无配置则使用内置默认方向。
        无论哪种来源, 都会根据问题特征动态增强策略提示。
        """
        config_directions = self.config.get("directions")
        if config_directions:
            directions = self._directions_from_config(
                config_directions, problem, classification)
        else:
            directions = self._default_directions(
                problem, classification)

        # 动态增强: 根据问题特征为 "structured" 方向补充提示
        self._enrich_structured_direction(
            directions, problem, classification)

        # 方向 D: 反思修复 (仅当有失败历史时)
        repair_cfg = self.config.get("repair_direction", {})
        min_failures = repair_cfg.get("min_failures", 2)
        if len(attempt_history) >= min_failures:
            recent_errors = []
            for a in attempt_history[-3:]:
                errs = a.get("errors", [])
                for e in errs[:2]:
                    msg = e.get("message", str(e)) if isinstance(e, dict) else str(e)
                    recent_errors.append(msg[:100])

            directions.append(ProofDirection(
                name=repair_cfg.get("name", "repair_rethink"),
                role=_role_from_string(repair_cfg.get("role", "critic")),
                model=repair_cfg.get("model", "claude-sonnet-4-20250514"),
                temperature=repair_cfg.get("temperature", 0.5),
                strategic_hint=(
                    f"Previous {len(attempt_history)} attempts all failed. "
                    f"Recent errors:\n" +
                    "\n".join(f"  - {e}" for e in recent_errors) +
                    "\n\nAnalyze WHY these approaches fail at a fundamental level. "
                    "Then propose a completely different proof strategy."
                ),
            ))

        return directions

    def _directions_from_config(self, config_list: list,
                                 problem, classification) -> list[ProofDirection]:
        """从 YAML 配置构建方向列表"""
        directions = []
        for d_cfg in config_list:
            directions.append(ProofDirection(
                name=d_cfg.get("name", f"direction_{len(directions)}"),
                role=_role_from_string(d_cfg.get("role", "proof_generator")),
                model=d_cfg.get("model", "claude-sonnet-4-20250514"),
                temperature=d_cfg.get("temperature", 0.7),
                strategic_hint=d_cfg.get("strategic_hint", ""),
            ))
        return directions

    def _default_directions(self, problem, classification) -> list[ProofDirection]:
        """内置默认方向 (无配置时使用)"""
        directions = []

        # 方向 A: 自动化探测 (快速排除简单题)
        directions.append(ProofDirection(
            name="automation",
            role=AgentRole.PROOF_GENERATOR,
            model="claude-sonnet-4-20250514",
            temperature=0.2,
            strategic_hint=(
                "Try to solve this with simple automation ONLY. "
                "Attempt these tactics in order: decide, norm_num, simp, "
                "omega, ring, aesop. If a single tactic doesn't work, "
                "try 'simp; ring' or 'simp; omega'. "
                "Do NOT attempt induction or complex proof structures."
            ),
        ))

        # 方向 B: 结构化证明 (主力方向)
        directions.append(ProofDirection(
            name="structured",
            role=AgentRole.PROOF_GENERATOR,
            temperature=0.7,
            strategic_hint=(
                "Plan the proof structure carefully. "
                "Use `have` statements with explicit types for "
                "intermediate steps."
            ),
        ))

        # 方向 C: 替代路径
        directions.append(ProofDirection(
            name="alternative",
            role=AgentRole.PROOF_PLANNER,
            temperature=0.9,
            strategic_hint=(
                "Try a fundamentally DIFFERENT approach from standard methods. "
                "Consider: casting to ℤ if working with ℕ, "
                "using `conv` to restructure goals, "
                "or finding a non-obvious Mathlib lemma that solves it directly."
            ),
        ))

        return directions

    def _enrich_structured_direction(self, directions, problem, classification):
        """根据问题特征动态增强 'structured' 方向的策略提示和前提"""
        structured = None
        for d in directions:
            if d.name == "structured":
                structured = d
                break
        if not structured:
            return

        techniques = classification.get("techniques", [])
        has_nat_sub = classification.get("has_nat_sub", False)

        if "induction" in techniques:
            structured.strategic_hint += (
                "\n\nThis problem likely requires induction on n. "
                "Structure: `induction n with | zero => ... | succ n ih => ...`"
            )
        if has_nat_sub:
            structured.strategic_hint += (
                "\n\nCRITICAL: This involves natural number subtraction. "
                "In Lean4, ℕ subtraction truncates to 0. "
                "You MUST prove minuend ≥ subtrahend before subtracting. "
                "Use `Nat.sub_add_cancel` or `tsub_add_cancel_of_le`."
            )

        # 检查领域插件
        matched_plugins = self.plugins.match(
            problem.theorem_statement, classification)
        if matched_plugins:
            plugin = matched_plugins[0]
            if plugin.strategic_hint:
                structured.strategic_hint += (
                    f"\n\nDomain expert hint: {plugin.strategic_hint}")

        premises = self._get_premises(problem.theorem_statement)
        if matched_plugins and matched_plugins[0].extra_premises:
            for p in matched_plugins[0].extra_premises[:10]:
                premises.append(p.get("statement", str(p)))
        structured.selected_premises = premises[:15]

        if matched_plugins:
            structured.few_shot_override = (
                matched_plugins[0].few_shot_examples or "")

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
        """为每个方向构建定制化的 prompt"""
        parts = [
            f"Prove the following Lean 4 theorem:\n"
            f"```lean\n{problem.theorem_statement}\n```",
        ]

        if direction.strategic_hint:
            parts.append(f"\n## Strategy guidance\n{direction.strategic_hint}")

        if problem.natural_language:
            parts.append(f"\n## Natural language description\n{problem.natural_language}")

        parts.append(
            "\nGenerate a complete proof. Output ONLY the proof body "
            "(starting with `:= by`) inside a single ```lean block. "
            "Do NOT use `sorry`."
        )

        return "\n".join(parts)
