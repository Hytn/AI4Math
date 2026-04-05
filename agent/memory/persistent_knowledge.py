"""agent/memory/persistent_knowledge.py — 跨问题持久化知识库

解决的问题: 当前系统的知识积累仅限于单题内, 问题间不共享。
一个包含 1000 道题的评测跑完后, 学到的经验全部丢失。

本模块提供轻量级的 JSON 文件持久化, 跨问题积累三类知识:
  1. 错误模式: 哪些 tactic 在哪类 goal 上经常失败
  2. 有效策略: 哪些 tactic 组合在哪类问题上经常成功
  3. 领域经验: 特定领域 (如 ℕ 减法) 的注意事项

Usage::

    kb = PersistentKnowledge("knowledge_base.json")
    kb.load()

    # 记录一次成功
    kb.record_success(domain="number_theory", tactics=["omega", "simp"],
                      theorem_type="divisibility")

    # 记录一次失败
    kb.record_failure(tactic="ring", goal_type="ℕ subtraction",
                      error_category="tactic_failed")

    # 查询建议
    suggestions = kb.get_suggestions(domain="number_theory")
    # → ["Prefer omega over ring for ℕ arithmetic", ...]

    kb.save()
"""
from __future__ import annotations
import json
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """单条知识记录"""
    category: str          # "failure", "success", "insight"
    domain: str = ""       # 数学领域
    tactic: str = ""       # 相关 tactic
    goal_type: str = ""    # goal 类型描述
    error_category: str = ""
    description: str = ""
    count: int = 1         # 出现次数


class PersistentKnowledge:
    """跨问题持久化知识库

    数据结构:
      - failures: {tactic -> {goal_type -> count}}
      - successes: {domain -> {tactic_combo -> count}}
      - insights: [str]  自由形式的经验总结
    """

    def __init__(self, filepath: str = "knowledge_base.json"):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._failures: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._successes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._insights: list[str] = []
        self._dirty = False

    def load(self) -> bool:
        """从文件加载知识库"""
        if not os.path.isfile(self.filepath):
            return False
        try:
            with open(self.filepath) as f:
                data = json.load(f)
            with self._lock:
                for tac, goals in data.get("failures", {}).items():
                    for goal, count in goals.items():
                        self._failures[tac][goal] = count
                for domain, tactics in data.get("successes", {}).items():
                    for tac, count in tactics.items():
                        self._successes[domain][tac] = count
                self._insights = data.get("insights", [])
            logger.info(f"PersistentKnowledge: loaded from {self.filepath} "
                        f"({sum(len(v) for v in self._failures.values())} failures, "
                        f"{sum(len(v) for v in self._successes.values())} successes)")
            return True
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"PersistentKnowledge: load failed: {e}")
            return False

    def save(self) -> bool:
        """保存知识库到文件"""
        with self._lock:
            if not self._dirty:
                return True
            data = {
                "failures": {k: dict(v) for k, v in self._failures.items()},
                "successes": {k: dict(v) for k, v in self._successes.items()},
                "insights": self._insights[-200:],  # 保留最近 200 条
            }
        try:
            os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
            with open(self.filepath, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            with self._lock:
                self._dirty = False
            return True
        except OSError as e:
            logger.warning(f"PersistentKnowledge: save failed: {e}")
            return False

    def record_failure(self, tactic: str, goal_type: str = "",
                       error_category: str = "", domain: str = ""):
        """记录一次 tactic 失败"""
        key = goal_type[:80] or error_category or "unknown"
        with self._lock:
            self._failures[tactic][key] += 1
            self._dirty = True

    def record_success(self, domain: str, tactics: list[str],
                       theorem_type: str = ""):
        """记录一次成功的 tactic 组合"""
        combo = " → ".join(tactics[:5])
        with self._lock:
            self._successes[domain or "general"][combo] += 1
            self._dirty = True

    def record_insight(self, insight: str):
        """记录一条自由形式的经验"""
        with self._lock:
            if insight not in self._insights[-50:]:
                self._insights.append(insight)
                self._dirty = True

    def get_suggestions(self, domain: str = "", goal_type: str = "",
                        max_items: int = 5) -> list[str]:
        """获取基于历史经验的建议"""
        suggestions = []
        with self._lock:
            # 推荐成功率高的 tactic 组合
            domain_successes = self._successes.get(domain, {})
            if domain_successes:
                top = sorted(domain_successes.items(),
                             key=lambda x: -x[1])[:max_items]
                for combo, count in top:
                    suggestions.append(
                        f"Proven effective ({count}x): {combo}")

            # 警告高频失败的 tactic
            for tactic, goals in self._failures.items():
                for gt, count in goals.items():
                    if count >= 3 and (not goal_type or goal_type in gt):
                        suggestions.append(
                            f"AVOID `{tactic}` on {gt} (failed {count}x)")

        return suggestions[:max_items]

    def render_for_prompt(self, domain: str = "", goal_type: str = "",
                          max_chars: int = 800) -> str:
        """渲染为可注入 prompt 的文本"""
        suggestions = self.get_suggestions(domain, goal_type, max_items=8)
        if not suggestions:
            return ""
        parts = ["## Historical experience from previous problems\n"]
        total = 0
        for s in suggestions:
            line = f"- {s}\n"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)
        return "".join(parts)

    def stats(self) -> dict:
        with self._lock:
            return {
                "failure_patterns": sum(len(v) for v in self._failures.values()),
                "success_patterns": sum(len(v) for v in self._successes.values()),
                "insights": len(self._insights),
                "dirty": self._dirty,
            }
