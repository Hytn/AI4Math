"""engine/world_model.py — 世界模型预测器接口 (Fix #8)

提供证明状态转移的预测能力，让智能体在心智空间中模拟策略效果，
无需调用 Lean4 REPL 即可预判 tactic 的可能结果。

当前状态:
  - 接口已定义
  - MockWorldModel 提供基于规则的启发式预测
  - 训练数据管道通过 ProofContextStore.export_rich_trajectories() 就绪
  - 未来可训练 transformer-based 模型替换 MockWorldModel

架构位置:
  ProofContextStore (Layer 0 数据) → 训练 → WorldModelPredictor
  SearchCoordinator / Agent → WorldModelPredictor.predict() → 预筛选

Usage::

    predictor = MockWorldModel()  # 或 TrainedWorldModel("model.pt")

    prediction = predictor.predict(
        goal_state="⊢ n + 0 = n",
        tactic="simp [Nat.add_zero]",
    )
    if prediction.likely_success and prediction.confidence > 0.7:
        # 值得提交给真正的 REPL 验证
        result = await repl.try_tactic(env_id, tactic)
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorldModelPrediction:
    """世界模型对单步 tactic 的预测结果"""
    tactic: str
    likely_success: bool
    confidence: float  # 0.0 ~ 1.0
    predicted_goals: list[str] = field(default_factory=list)
    predicted_error: str = ""
    goals_delta: int = 0  # >0 = 减少目标, <0 = 增加目标
    reasoning: str = ""

    @property
    def worth_trying(self) -> bool:
        """是否值得提交给真正的 REPL 验证"""
        return self.likely_success or self.confidence < 0.5


class WorldModelPredictor(ABC):
    """世界模型抽象接口

    子类实现 predict() 即可插入搜索流程。
    """

    @abstractmethod
    def predict(self, goal_state: str, tactic: str,
                hypotheses: list[str] = None,
                context: dict = None) -> WorldModelPrediction:
        """预测 tactic 在给定 goal 上的效果。

        Args:
            goal_state: 当前目标 (如 "⊢ n + 0 = n")
            tactic: 要预测的 tactic (如 "simp [Nat.add_zero]")
            hypotheses: 当前假设列表
            context: 额外上下文 (深度、领域等)

        Returns:
            WorldModelPrediction 包含预测结果和置信度
        """
        ...

    def predict_batch(self, goal_state: str,
                      tactics: list[str],
                      hypotheses: list[str] = None,
                      context: dict = None) -> list[WorldModelPrediction]:
        """批量预测多个 tactic，按预期效果排序。"""
        predictions = [
            self.predict(goal_state, t, hypotheses, context)
            for t in tactics
        ]
        # 排序: 高置信度成功优先, 低置信度失败靠后
        predictions.sort(key=lambda p: (
            -int(p.likely_success), -p.confidence))
        return predictions

    def filter_tactics(self, goal_state: str,
                       tactics: list[str],
                       min_confidence: float = 0.3,
                       hypotheses: list[str] = None) -> list[str]:
        """过滤掉世界模型认为必定失败的 tactic。

        保守策略: 只剔除高置信度的预测失败项。
        """
        predictions = self.predict_batch(
            goal_state, tactics, hypotheses)
        return [
            p.tactic for p in predictions
            if p.likely_success or p.confidence < min_confidence
        ]


class MockWorldModel(WorldModelPredictor):
    """基于规则的启发式世界模型 (Fix #8)

    不依赖训练数据，使用简单的模式匹配预测 tactic 效果。
    作为真正世界模型训练完成前的占位实现。
    """

    # 几乎总是成功的 tactic (在合适的 goal 上)
    _HIGH_SUCCESS = {
        "rfl": [r"⊢\s*\S+\s*=\s*\S+$"],  # 等式且两边相同
        "trivial": [r"⊢\s*True", r"⊢\s*\S+\s*=\s*\S+$"],
        "assumption": [],  # 需要检查假设
    }

    # 按 goal 形状推荐的 tactic
    _SHAPE_TACTICS = {
        r"∀|→|⊢\s*\(\S+\s*→": ["intro"],
        r"∃": ["use", "exact ⟨"],
        r"∧": ["constructor", "exact ⟨"],
        r"∨": ["left", "right"],
        r"¬": ["intro", "push_neg"],
        r"Nat|ℕ": ["omega", "simp", "induction"],
        r"Int|ℤ": ["omega", "linarith"],
        r"Real|ℝ": ["linarith", "nlinarith", "norm_num"],
        r"Finset": ["simp", "decide"],
    }

    def predict(self, goal_state: str, tactic: str,
                hypotheses: list[str] = None,
                context: dict = None) -> WorldModelPrediction:
        hypotheses = hypotheses or []
        tactic_base = tactic.split()[0] if tactic.strip() else ""

        # sorry 总是 "成功" 但不可接受
        if tactic_base == "sorry":
            return WorldModelPrediction(
                tactic=tactic, likely_success=True, confidence=0.99,
                reasoning="sorry always closes goals but is not a proof")

        # 检查高成功率 tactic 的 pattern 匹配
        if tactic_base in self._HIGH_SUCCESS:
            patterns = self._HIGH_SUCCESS[tactic_base]
            if not patterns:
                # assumption: 检查假设中是否有匹配
                if hypotheses:
                    return WorldModelPrediction(
                        tactic=tactic, likely_success=True,
                        confidence=0.6, goals_delta=1,
                        reasoning="hypothesis available")
            for pat in patterns:
                if re.search(pat, goal_state):
                    return WorldModelPrediction(
                        tactic=tactic, likely_success=True,
                        confidence=0.8, goals_delta=1,
                        reasoning=f"pattern match: {pat}")

        # 检查 goal 形状 → 推荐 tactic 匹配度
        for shape_pat, good_tactics in self._SHAPE_TACTICS.items():
            if re.search(shape_pat, goal_state):
                if tactic_base in good_tactics:
                    return WorldModelPrediction(
                        tactic=tactic, likely_success=True,
                        confidence=0.5, goals_delta=0,
                        reasoning=f"shape {shape_pat} matches {tactic_base}")

        # intro 在 forall/arrow goal 上大概率成功
        if tactic_base == "intro" and ("→" in goal_state or "∀" in goal_state
                                        or "⊢ (" in goal_state):
            return WorldModelPrediction(
                tactic=tactic, likely_success=True, confidence=0.7,
                goals_delta=0,
                reasoning="intro on forall/arrow goal")

        # simp 是泛用 tactic, 给中等置信度
        if tactic_base in ("simp", "simp?", "norm_num", "ring", "omega",
                           "linarith", "decide"):
            return WorldModelPrediction(
                tactic=tactic, likely_success=True, confidence=0.35,
                goals_delta=0,
                reasoning=f"{tactic_base} is a general automation tactic")

        # 默认: 不确定
        return WorldModelPrediction(
            tactic=tactic, likely_success=False, confidence=0.2,
            reasoning="no pattern match, uncertain")


class TrainedWorldModel(WorldModelPredictor):
    """基于训练数据的世界模型 (占位实现)

    训练数据来自 ProofContextStore.export_rich_trajectories(),
    格式为 (state, action, next_state, error, goal_diff) 五元组。

    未来实现方案:
      1. 简单方案: goal embedding + tactic embedding → MLP 分类器
      2. 完整方案: Transformer encoder 处理 (goal, hypotheses, tactic)
         → 预测 (success, new_goals, error_category)
    """

    def __init__(self, model_path: str = ""):
        self._model_path = model_path
        self._fallback = MockWorldModel()
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        """加载训练好的模型权重。"""
        # TODO: 实际加载 PyTorch/ONNX 模型
        pass

    def predict(self, goal_state: str, tactic: str,
                hypotheses: list[str] = None,
                context: dict = None) -> WorldModelPrediction:
        # 目前降级到 MockWorldModel
        return self._fallback.predict(
            goal_state, tactic, hypotheses, context)
