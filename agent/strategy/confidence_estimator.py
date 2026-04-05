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
