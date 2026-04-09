"""knowledge/writer.py — 知识写入管道

将验证结果自动写入知识金字塔的各层：

  验证结果 ──→ ingest_step()
                 ├→ Layer 0: proof_traces (via ProofContextStore)
                 ├→ Layer 1: tactic_effectiveness (增量更新)
                 └→ Layer 1: error_patterns (如果失败)

  证明完成 ──→ ingest_proof_result()
                 ├→ Layer 0: record_rich_trace
                 ├→ Layer 1: 批量更新所有 tactic effectiveness
                 ├→ Layer 1: 提取并注册 proved_lemmas
                 └→ Layer 1: 更新 error_patterns 的修复信息

Usage::

    writer = KnowledgeWriter(store)

    # 每次 tactic 验证后
    await writer.ingest_step(step_detail, theorem="...", domain="...")

    # 证明完成/失败后
    await writer.ingest_proof_result(
        context_id=ctx_id, steps=steps,
        success=True, theorem="...", duration_ms=150)
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from engine.proof_context_store import StepDetail
from knowledge.goal_normalizer import (
    normalize_goal_for_key, classify_domain, extract_keywords,
    statement_hash,
)
from knowledge.store import UnifiedKnowledgeStore
from knowledge.types import LemmaRecord

logger = logging.getLogger(__name__)


class KnowledgeWriter:
    """知识写入管道 — 从验证结果到四层知识金字塔"""

    def __init__(self, store: UnifiedKnowledgeStore):
        self.store = store

    async def ingest_step(
            self, step: StepDetail,
            theorem: str = "",
            domain: str = "",
            trace_id: int = 0) -> None:
        """写入单步验证结果

        每次 tactic 验证（成功或失败）后调用。
        自动更新 Layer 1 的 tactic_effectiveness 和 error_patterns。

        Args:
            step: 步级详情 (来自 engine.proof_context_store.StepDetail)
            theorem: 当前定理 (用于领域推断)
            domain: 领域标签 (如已知则传入，否则自动推断)
            trace_id: 关联的 trace ID (可选)
        """
        if not step.tactic or not step.tactic.strip():
            return

        # 推断领域
        goal_text = " ".join(step.goals_before)
        if not domain:
            domain = classify_domain(goal_text, theorem)

        # 规范化 goal pattern
        goal_pattern = normalize_goal_for_key(goal_text)
        if not goal_pattern:
            return

        # 判断成功/失败
        success = (step.env_id_after >= 0 and not step.error_message)

        # Layer 1: tactic effectiveness
        await self.store.upsert_tactic_effectiveness(
            tactic=step.tactic.strip(),
            goal_pattern=goal_pattern,
            success=success,
            elapsed_ms=step.elapsed_ms,
            domain=domain,
            trace_id=trace_id)

        # Layer 1: error patterns (仅失败时)
        if not success and step.error_category:
            await self.store.upsert_error_pattern(
                error_category=step.error_category,
                goal_pattern=goal_pattern,
                tactic=step.tactic.strip())

    async def ingest_proof_result(
            self, context_id: int,
            steps: list[StepDetail],
            success: bool,
            theorem: str = "",
            duration_ms: float = 0.0,
            domain: str = "") -> int:
        """写入完整证明结果

        一次证明尝试（成功或失败）结束后调用。

        1. 写入 Layer 0 rich trace
        2. 批量更新 Layer 1 tactic effectiveness
        3. 如果成功，提取 have-步骤作为 proved_lemmas
        4. 分析错误序列，更新 error patterns 的修复建议

        Args:
            context_id: 来自 store.save() 的上下文 ID
            steps: 完整的步级详情列表
            success: 证明是否成功
            theorem: 定理文本
            duration_ms: 总耗时
            domain: 领域标签

        Returns:
            trace_id
        """
        if not steps:
            return 0

        if not domain:
            goal_text = " ".join(steps[0].goals_before) if steps else ""
            domain = classify_domain(goal_text, theorem)

        # Layer 0: record rich trace
        trace_id = await self.store.record_rich_trace(
            context_id=context_id,
            steps=steps,
            success=success,
            duration_ms=duration_ms)

        # Layer 1: batch update tactic effectiveness
        for step in steps:
            await self.ingest_step(
                step, theorem=theorem, domain=domain,
                trace_id=trace_id)

        # Layer 1: extract lemmas from successful proof
        if success:
            await self._extract_lemmas_from_steps(
                steps, theorem, domain, trace_id)

        # Layer 1: analyze error→fix patterns
        await self._analyze_fix_patterns(steps, domain)

        return trace_id

    async def _extract_lemmas_from_steps(
            self, steps: list[StepDetail],
            theorem: str, domain: str, trace_id: int):
        """从成功的证明步骤中提取 have 子引理"""
        for step in steps:
            if not step.tactic.strip().startswith("have "):
                continue
            if step.env_id_after < 0:
                continue  # have 步骤自身失败，跳过

            # 解析 have 声明
            lemma_info = _parse_have_tactic(step.tactic.strip())
            if not lemma_info:
                continue

            name, type_str = lemma_info
            statement = f"lemma {name} : {type_str}"
            keywords = extract_keywords(f"{statement} {type_str}")

            lemma = LemmaRecord(
                name=name,
                statement=statement,
                proof=":= by sorry",  # 实际 proof 需要从完整代码中提取
                statement_hash=statement_hash(statement),
                source_problem=theorem[:200],
                source_trace_id=trace_id,
                verified=False,  # 需要后续独立验证
                keywords=keywords,
                domain=domain,
                goal_types=[normalize_goal_for_key(
                    " ".join(step.goals_before))],
            )

            try:
                await self.store.add_lemma(lemma)
            except Exception as e:
                logger.debug(f"KnowledgeWriter: lemma extraction skipped: {e}")

    async def _analyze_fix_patterns(
            self, steps: list[StepDetail], domain: str):
        """分析连续失败→成功的 tactic 序列，提取修复模式

        模式: tactic_A 在 goal_X 上失败 → tactic_B 在 goal_X 上成功
        推断: tactic_B 是 tactic_A 在 goal_X 上的修复候选
        """
        for i in range(len(steps) - 1):
            current = steps[i]
            next_step = steps[i + 1]

            # 当前步失败，下一步在相同 goal 上成功
            if (current.error_message
                    and not next_step.error_message
                    and next_step.env_id_after >= 0
                    and current.goals_before == next_step.goals_before):

                goal_pattern = normalize_goal_for_key(
                    " ".join(current.goals_before))

                await self.store.upsert_error_pattern(
                    error_category=current.error_category or "unknown",
                    goal_pattern=goal_pattern,
                    tactic=current.tactic.strip(),
                    fix_tactic=next_step.tactic.strip(),
                    fix_succeeded=True)

    async def import_from_persistent_knowledge(
            self, pk) -> int:
        """从旧版 PersistentKnowledge 迁移数据

        Args:
            pk: agent.memory.persistent_knowledge.PersistentKnowledge 实例

        Returns:
            迁移的条目数
        """
        count = 0
        now = time.time()

        # 迁移失败模式
        for tactic, goals in pk._failures.items():
            for goal_type, freq in goals.items():
                for _ in range(freq):
                    await self.store.upsert_tactic_effectiveness(
                        tactic=tactic, goal_pattern=goal_type,
                        success=False)
                count += 1

        # 迁移成功模式
        for domain, combos in pk._successes.items():
            for combo, freq in combos.items():
                tactics = combo.split(" → ")
                for t in tactics:
                    for _ in range(freq):
                        await self.store.upsert_tactic_effectiveness(
                            tactic=t.strip(), goal_pattern="(migrated)",
                            success=True, domain=domain)
                count += 1

        logger.info(f"KnowledgeWriter: migrated {count} entries "
                    f"from PersistentKnowledge")
        return count

    async def import_from_lemma_bank(self, bank) -> int:
        """从旧版 LemmaBank 迁移数据

        Args:
            bank: prover.lemma_bank.bank.LemmaBank 实例

        Returns:
            迁移的引理数
        """
        count = 0
        for lemma in bank.lemmas:
            keywords = extract_keywords(
                f"{lemma.statement} {lemma.name}")
            record = LemmaRecord(
                name=lemma.name,
                statement=lemma.statement,
                proof=lemma.proof,
                statement_hash=statement_hash(lemma.statement),
                verified=lemma.verified,
                keywords=keywords,
            )
            await self.store.add_lemma(record)
            count += 1

        logger.info(f"KnowledgeWriter: migrated {count} lemmas "
                    f"from LemmaBank")
        return count

    async def import_from_episodic_memory(self, em) -> int:
        """从旧版 EpisodicMemory 迁移数据

        Args:
            em: agent.memory.episodic_memory.EpisodicMemory 实例

        Returns:
            迁移的 episode 数
        """
        from knowledge.types import StrategyPattern
        count = 0

        for ep in em.episodes:
            pattern = StrategyPattern(
                name=f"{ep.problem_type}_{ep.winning_strategy}",
                domain=ep.problem_type,
                problem_pattern=f"{ep.problem_type} ({ep.difficulty})",
                tactic_template=ep.key_tactics,
                confidence=0.7,  # 历史数据给予中等置信度
            )
            await self.store.add_strategy_pattern(pattern)
            count += 1

        logger.info(f"KnowledgeWriter: migrated {count} episodes "
                    f"from EpisodicMemory")
        return count


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

_HAVE_PATTERN = re.compile(
    r'^have\s+(\w+)\s*:\s*(.+?)(?:\s*:=|\s*$)')


def _parse_have_tactic(tactic: str) -> Optional[tuple[str, str]]:
    """解析 have 声明，提取名称和类型

    "have h1 : n + 0 = n := by omega"
    → ("h1", "n + 0 = n")
    """
    m = _HAVE_PATTERN.match(tactic)
    if m:
        return m.group(1), m.group(2).strip()
    return None
