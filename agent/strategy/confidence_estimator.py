"""agent/strategy/confidence_estimator.py — 置信度评估与主动放弃

改进:
  1. 基于错误模式多样性而非纯线性衰减
  2. 区分"有进展但未完成"和"完全卡住"
  3. 可配置的衰减速率, 与 max_samples 预算对齐
  4. 将 banked_lemmas 和 partial progress 视为正面信号
"""
from __future__ import annotations
from agent.memory.working_memory import WorkingMemory

class ConfidenceEstimator:

    def __init__(self, max_samples: int = 128, base_threshold: float = 0.05):
        self.max_samples = max_samples
        self.base_threshold = base_threshold

    def estimate(self, memory: WorkingMemory) -> float:
        """多维度置信度评估.

        信号:
          - 错误多样性: 多种不同错误 = 搜索空间仍在探索 → 保持信心
          - 错误重复度: 同一错误反复出现 = 卡住了 → 降低信心
          - 进展信号: banked_lemmas, partial goals closed → 提升信心
          - 尝试次数: 相对于 max_samples 预算的比例 → 渐进衰减
        """
        if memory.solved:
            return 1.0
        if not memory.attempt_history:
            return 0.5

        n = len(memory.attempt_history)

        # 基础分: 随尝试次数相对于预算渐进衰减
        budget_ratio = n / max(1, self.max_samples)
        base = 0.5 * (1.0 - budget_ratio)

        # 错误多样性奖励: 多种不同错误 = 搜索空间仍在有效探索
        unique_errors = len(memory.error_patterns)
        diversity_bonus = min(0.2, unique_errors * 0.04)

        # 错误重复度惩罚: 最近 N 次中主导错误出现的比例
        dom = memory.get_dominant_error()
        if dom and dom != "none":
            recent = memory.attempt_history[-8:]
            dom_count = sum(1 for a in recent
                           if dom in str(a.get("errors", [])))
            repetition_penalty = 0.05 * dom_count
        else:
            repetition_penalty = 0.0

        # 进展奖励: banked_lemmas 说明在积累有用的中间结果
        lemma_bonus = min(0.15, len(memory.banked_lemmas) * 0.05)

        score = base + diversity_bonus + lemma_bonus - repetition_penalty
        return max(0.0, min(1.0, score))

    def should_abstain(self, memory: WorkingMemory,
                       threshold: float = None) -> bool:
        t = threshold if threshold is not None else self.base_threshold
        return self.estimate(memory) < t

    @staticmethod
    def refine_confidence(result: 'AgentResult',
                          feedback: 'AgentFeedback' = None,
                          l0_passed: bool = True,
                          l1_passed: bool = False,
                          l2_passed: bool = False) -> float:
        """用验证阶段的实际反馈更新置信度

        统一的置信度精化逻辑 — 在验证完成后调用。
        之前分散在 SubAgent 中, 现在统一到 ConfidenceEstimator。

        置信度分级:
          0.0-0.2: 生成了代码但可能有语法问题
          0.2-0.4: L0 通过, 语法正确但未验证
          0.4-0.7: L1 有部分进展 (关闭了一些 goal)
          0.7-0.9: L1 通过, 所有 goal 关闭
          0.9-1.0: L2 通过, 完整编译认证
        """
        base = result.confidence

        if not l0_passed:
            return min(base, 0.15)

        if l2_passed:
            return 0.95

        if l1_passed:
            return max(base, 0.80)

        if feedback:
            if feedback.is_proof_complete:
                return max(base, 0.85)

            if feedback.progress_score > 0:
                progress_bonus = feedback.progress_score * 0.3
                base = max(base, 0.3 + progress_bonus)

            if feedback.repair_candidates:
                high_conf = [r for r in feedback.repair_candidates
                             if r.confidence > 0.7]
                if high_conf:
                    base = max(base, 0.35)

            if feedback.error_category == "type_mismatch":
                base *= 0.8
            elif feedback.error_category == "unknown_identifier":
                base *= 0.85

        return min(1.0, max(0.0, base))
