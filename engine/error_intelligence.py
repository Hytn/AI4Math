"""engine/error_intelligence.py — 错误智能层

将 Lean4 的原始错误信息转化为 Agent 可直接消费的结构化反馈。

核心理念: 错误不是终点, 是搜索信号。每一次 Lean4 报错都携带了
丰富的类型论信息 — 期望类型 vs 实际类型、当前 goal state、可用假设。
本模块将这些信息提取、分类、并生成具体的修复候选。

信息密度对比:
  - 传统方式: Lean4 返回 pass/fail → 1 bit / 2-12s
  - 本模块:   返回 AgentFeedback → ~100 bits / 50ms
    包含: goal states + error structure + repair candidates + progress score

关键能力:
  1. 结构化错误分类 (不是正则匹配, 而是语义理解)
  2. 修复候选生成 (利用 Lean4 的 exact?/apply?/rw?)
  3. 局部进度评估 (哪些 goal 已解决, 哪些新增了)
  4. 上下文可用性分析 (当前有哪些假设/引理可用)
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from engine.lean_pool import LeanPool, TacticFeedback

logger = logging.getLogger(__name__)


@dataclass
class RepairCandidate:
    """一个具体的修复候选"""
    tactic: str
    confidence: float  # 0.0-1.0
    reason: str
    source: str = "heuristic"  # heuristic, exact?, apply?, rw?

    def to_prompt_line(self) -> str:
        return f"  - Try `{self.tactic}` ({self.reason}) [{self.source}]"


@dataclass
class GoalState:
    """单个 goal 的结构化表示"""
    index: int
    target_type: str
    hypotheses: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        hyps = "\n".join(f"    {h}" for h in self.hypotheses) if self.hypotheses else "    (none)"
        return f"  Goal {self.index}:\n{hyps}\n    ⊢ {self.target_type}"


@dataclass
class AgentFeedback:
    """Agent 可直接消费的结构化反馈

    这是错误智能层与 Agent 层的核心接口。
    Agent 不需要解析原始错误文本, 直接得到:
    - 当前证明状态的完整快照
    - 最近一步的精确诊断
    - 具体的修复候选
    - 局部进度评估
    """

    # ── 证明状态快照 ──
    remaining_goals: list[GoalState] = field(default_factory=list)
    solved_goals_count: int = 0
    total_goals_count: int = 0

    # ── 最近一步的诊断 ──
    failed_tactic: str = ""
    error_category: str = ""      # type_mismatch, unknown_identifier, ...
    error_message: str = ""       # 简化后的错误描述
    expected_type: str = ""
    actual_type: str = ""

    # ── 修复候选 ──
    repair_candidates: list[RepairCandidate] = field(default_factory=list)

    # ── 上下文 ──
    available_hypotheses: list[str] = field(default_factory=list)
    relevant_lemmas: list[str] = field(default_factory=list)

    # ── 进度评估 ──
    progress_score: float = 0.0   # 0.0-1.0
    goals_closed_this_step: int = 0
    goals_opened_this_step: int = 0
    is_proof_complete: bool = False

    # ── 原始信息 (调试用) ──
    raw_error: str = ""
    elapsed_ms: int = 0

    def to_prompt(self, max_chars: int = 3000) -> str:
        """渲染为可直接注入 LLM prompt 的文本

        这是本模块最重要的输出方法。
        将结构化的反馈转化为 LLM 可理解的自然语言。
        """
        parts = []

        # 状态摘要
        if self.is_proof_complete:
            return "✓ Proof is complete! All goals closed."

        parts.append(f"## Proof state ({self.solved_goals_count} solved, "
                     f"{len(self.remaining_goals)} remaining)\n")

        # 失败诊断
        if self.failed_tactic:
            parts.append(f"Last tactic `{self.failed_tactic}` failed: "
                         f"{self.error_message}")
            if self.expected_type and self.actual_type:
                parts.append(f"  Expected: {self.expected_type}")
                parts.append(f"  Actual:   {self.actual_type}")
            parts.append("")

        # 修复候选
        if self.repair_candidates:
            parts.append("Suggested fixes:")
            for rc in self.repair_candidates[:5]:
                parts.append(rc.to_prompt_line())
            parts.append("")

        # 当前 goals
        if self.remaining_goals:
            parts.append("Current goals:")
            for g in self.remaining_goals[:3]:
                parts.append(g.to_prompt())
            if len(self.remaining_goals) > 3:
                parts.append(f"  ... and {len(self.remaining_goals) - 3} more")

        result = "\n".join(parts)
        return result[:max_chars]

    @staticmethod
    def from_success(goals: list[str], goals_before: int,
                     elapsed_ms: int = 0) -> AgentFeedback:
        """从成功的 tactic 结果构造反馈"""
        goal_states = [GoalState(i, g) for i, g in enumerate(goals)]
        closed = max(0, goals_before - len(goals))
        opened = max(0, len(goals) - goals_before + 1)  # -1 because original goal replaced
        progress = 1.0 if not goals else (closed / max(1, goals_before))

        return AgentFeedback(
            remaining_goals=goal_states,
            solved_goals_count=closed,
            total_goals_count=len(goals),
            goals_closed_this_step=closed,
            goals_opened_this_step=opened,
            progress_score=min(1.0, progress),
            is_proof_complete=(len(goals) == 0),
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def from_failure(tactic: str, error_msg: str, error_cat: str,
                     expected: str = "", actual: str = "",
                     goals: list[str] = None,
                     elapsed_ms: int = 0) -> AgentFeedback:
        """从失败的 tactic 结果构造反馈"""
        goal_states = [GoalState(i, g) for i, g in enumerate(goals or [])]
        return AgentFeedback(
            remaining_goals=goal_states,
            failed_tactic=tactic,
            error_category=error_cat,
            error_message=error_msg[:500],
            expected_type=expected,
            actual_type=actual,
            raw_error=error_msg,
            elapsed_ms=elapsed_ms,
        )


class ErrorIntelligence:
    """错误智能分析引擎

    核心能力:
    1. 将 TacticFeedback 转化为 AgentFeedback (丰富化)
    2. 生成修复候选 (heuristic + Lean4 exact?/apply?)
    3. 跨尝试积累错误模式知识

    Usage::

        ei = ErrorIntelligence(lean_pool)
        feedback = ei.analyze(tactic_result, goals_before=2)
        prompt_text = feedback.to_prompt()
    """

    def __init__(self, lean_pool: LeanPool = None,
                 premise_index=None):
        self.pool = lean_pool
        self.premises = premise_index
        # 跨尝试的错误模式积累
        self._error_history: list[dict] = []
        self._known_failures: set[str] = set()
        # exact?/apply?/rw? 调用计数 (防止每个问题无限搜索)
        self._search_calls: int = 0
        self._max_search_calls: int = 15  # 每个问题最多 15 次搜索
        self._search_timeout: int = 5     # 每次搜索最多 5 秒

    def analyze(self, result: TacticFeedback,
                goals_before: int = 1,
                use_search_tactics: bool = True,
                parent_env_id: int = -1) -> AgentFeedback:
        """分析一个 tactic 执行结果, 生成结构化反馈

        Args:
            result: Lean4 REPL 返回的 tactic 执行结果
            goals_before: 执行前的 goal 数量
            use_search_tactics: 是否调用 exact?/apply? 搜索修复
            parent_env_id: 执行 tactic 之前的环境 ID (用于在失败时
                           调用 exact?/apply? 搜索修复候选)

        Returns:
            AgentFeedback: Agent 可直接消费的结构化反馈
        """
        if result.success:
            fb = AgentFeedback.from_success(
                result.remaining_goals, goals_before, result.elapsed_ms)
            return fb

        # 构造基础失败反馈
        fb = AgentFeedback.from_failure(
            tactic=result.tactic,
            error_msg=result.error_message,
            error_cat=result.error_category,
            expected=result.expected_type,
            actual=result.actual_type,
            goals=result.remaining_goals,
            elapsed_ms=result.elapsed_ms,
        )

        # 记录失败模式
        self._record_failure(result)

        # 生成修复候选
        fb.repair_candidates = self._generate_repairs(result, goals_before)

        # 如果有 REPL 连接池, 尝试用 Lean4 的搜索 tactic 找候选
        # 关键修复: 使用 parent_env_id (失败前的环境) 而非 result.new_env_id
        # (失败时 new_env_id == -1), 因为搜索 tactic 应在尚有 goal 的
        # 父环境上运行, 才能搜索到可解决当前 goal 的引理。
        search_env = parent_env_id if parent_env_id >= 0 else result.new_env_id
        if use_search_tactics and self.pool and search_env >= 0:
            lean_candidates = self._search_via_lean(search_env)
            fb.repair_candidates.extend(lean_candidates)

        # 按 confidence 排序
        fb.repair_candidates.sort(key=lambda r: -r.confidence)

        return fb

    def analyze_batch(self, results: list[TacticFeedback],
                      goals_before: int = 1) -> list[AgentFeedback]:
        """批量分析多个 tactic 结果"""
        return [self.analyze(r, goals_before) for r in results]

    def get_accumulated_knowledge(self, max_items: int = 10) -> str:
        """获取跨尝试积累的错误模式知识 (注入 prompt)"""
        if not self._known_failures:
            return ""

        parts = ["## Known dead ends (avoid these)\n"]
        for failure in list(self._known_failures)[:max_items]:
            parts.append(f"- {failure}")
        return "\n".join(parts)

    # ── 修复候选生成 ──

    def _generate_repairs(self, result: TacticFeedback,
                          goals_before: int) -> list[RepairCandidate]:
        """基于错误类型生成启发式修复候选"""
        candidates = []
        cat = result.error_category
        msg = result.error_message

        if cat == "type_mismatch":
            candidates.extend(self._repairs_for_type_mismatch(
                result.expected_type, result.actual_type, result.tactic))

        elif cat == "unknown_identifier":
            candidates.extend(self._repairs_for_unknown_id(msg, result.tactic))

        elif cat == "tactic_failed":
            candidates.extend(self._repairs_for_tactic_failed(
                result.tactic, msg))

        elif cat == "unsolved_goals":
            candidates.extend(self._repairs_for_unsolved(msg))

        elif cat == "timeout":
            candidates.extend(self._repairs_for_timeout(result.tactic))

        elif cat == "syntax_error":
            candidates.extend(self._repairs_for_syntax(msg))

        return candidates

    def _repairs_for_type_mismatch(self, expected: str, actual: str,
                                   tactic: str) -> list[RepairCandidate]:
        candidates = []

        # Numeric type coercion
        if any(t in (expected + actual) for t in ["Nat", "Int", "ℕ", "ℤ", "ℝ"]):
            candidates.append(RepairCandidate(
                "push_cast", 0.7,
                "Numeric type mismatch — try pushing casts", "heuristic"))
            candidates.append(RepairCandidate(
                "norm_cast", 0.6,
                "Normalize numeric casts", "heuristic"))

        # Equality direction
        if "Eq" in expected or "=" in expected:
            candidates.append(RepairCandidate(
                f"symm; {tactic}", 0.5,
                "Try reversing the equality direction", "heuristic"))

        # General rewrite
        candidates.append(RepairCandidate(
            "simp only [*]", 0.4,
            "Simplify with all hypotheses", "heuristic"))

        return candidates

    def _repairs_for_unknown_id(self, msg: str,
                                tactic: str) -> list[RepairCandidate]:
        candidates = []

        # 提取未知标识符名
        m = re.search(r"unknown (?:identifier|constant) '([^']+)'", msg)
        name = m.group(1) if m else ""

        if name:
            # Lean3 → Lean4 名称映射
            lean4_names = {
                "nat.add_comm": "Nat.add_comm",
                "nat.succ": "Nat.succ",
                "int.coe_nat": "Int.ofNat",
                "list.nil": "List.nil",
            }
            if name.lower() in lean4_names:
                new_name = lean4_names[name.lower()]
                candidates.append(RepairCandidate(
                    tactic.replace(name, new_name), 0.9,
                    f"Lean3→Lean4 rename: {name} → {new_name}", "heuristic"))

            # 建议用 exact? 搜索
            candidates.append(RepairCandidate(
                "exact?", 0.6,
                f"Search for the correct lemma name", "heuristic"))

        candidates.append(RepairCandidate(
            "apply?", 0.5,
            "Search for applicable lemmas", "heuristic"))

        return candidates

    def _repairs_for_tactic_failed(self, tactic: str,
                                   msg: str) -> list[RepairCandidate]:
        candidates = []
        tac_name = tactic.split()[0] if tactic else ""

        # ring 失败 → 试 omega 或 linarith
        if tac_name == "ring":
            candidates.append(RepairCandidate(
                "omega", 0.6, "ring failed; try omega for linear arithmetic",
                "heuristic"))
            candidates.append(RepairCandidate(
                "linarith", 0.5, "ring failed; try linarith", "heuristic"))

        # simp 失败 → 试更具体的 simp
        elif tac_name == "simp":
            candidates.append(RepairCandidate(
                "simp?", 0.7,
                "Ask Lean4 which simp lemmas would work", "heuristic"))
            candidates.append(RepairCandidate(
                "simp_all", 0.5,
                "Simplify with all hypotheses", "heuristic"))

        # omega 失败 → 可能不是线性算术
        elif tac_name == "omega":
            candidates.append(RepairCandidate(
                "ring", 0.5, "omega failed; try ring for polynomial goals",
                "heuristic"))
            candidates.append(RepairCandidate(
                "norm_num", 0.6, "omega failed; try norm_num for numeric goals",
                "heuristic"))

        # 通用: 试 aesop
        candidates.append(RepairCandidate(
            "aesop", 0.3, "General-purpose automation", "heuristic"))

        return candidates

    def _repairs_for_unsolved(self, msg: str) -> list[RepairCandidate]:
        return [
            RepairCandidate("assumption", 0.5,
                            "Check if goal matches a hypothesis", "heuristic"),
            RepairCandidate("exact?", 0.6,
                            "Search for exact match", "heuristic"),
            RepairCandidate("simp_all", 0.4,
                            "Simplify remaining goals", "heuristic"),
        ]

    def _repairs_for_timeout(self, tactic: str) -> list[RepairCandidate]:
        tac_name = tactic.split()[0] if tactic else ""
        candidates = []

        if tac_name == "simp":
            candidates.append(RepairCandidate(
                "simp only [...]", 0.8,
                "Use simp only with specific lemmas to avoid search explosion",
                "heuristic"))
        candidates.append(RepairCandidate(
            "-- break into smaller steps with `have`", 0.6,
            "Complex expressions cause elaboration timeout; decompose",
            "heuristic"))
        return candidates

    def _repairs_for_syntax(self, msg: str) -> list[RepairCandidate]:
        candidates = []
        if "expected" in msg.lower():
            candidates.append(RepairCandidate(
                "-- check indentation and `:= by` prefix", 0.7,
                "Lean4 is indentation-sensitive", "heuristic"))
        return candidates

    def _search_via_lean(self, env_id: int) -> list[RepairCandidate]:
        """利用 Lean4 的 exact?/apply?/rw? 搜索修复候选

        这是本模块最强大的能力:
        让 Lean4 自己搜索可用的引理和 tactic, 而不是靠 BM25 文本匹配。
        exact? 做的是类型驱动的精确匹配, 比任何检索方法都准确。

        安全措施:
          - 每次搜索独立超时 (默认 5s), 防止 REPL 阻塞
          - 每个问题最多 max_search_calls 次搜索, 防止资源耗尽
        """
        candidates = []

        if not self.pool:
            return candidates

        if self._search_calls >= self._max_search_calls:
            logger.debug(
                f"exact?/apply? search skipped: limit reached "
                f"({self._search_calls}/{self._max_search_calls})")
            return candidates

        import threading

        # 尝试 exact? — 搜索精确匹配当前 goal type 的引理
        for search_tac in ["exact?", "apply?", "rw?"]:
            if self._search_calls >= self._max_search_calls:
                break
            self._search_calls += 1

            # 用独立线程 + join(timeout) 实现搜索超时,
            # 避免 exact? 在复杂 goal 上阻塞整个 REPL session 几十秒。
            search_result = [None]
            search_error = [None]

            def _do_search():
                try:
                    search_result[0] = self.pool.try_tactic(env_id, search_tac)
                except Exception as e:
                    search_error[0] = e

            t = threading.Thread(target=_do_search, daemon=True)
            t.start()
            t.join(timeout=self._search_timeout)

            if t.is_alive():
                logger.info(
                    f"{search_tac} search timed out after "
                    f"{self._search_timeout}s on env_id={env_id}")
                # 线程会在后台继续, 但我们不等它了
                continue

            if search_error[0]:
                logger.debug(f"{search_tac} search failed: {search_error[0]}")
                continue

            result = search_result[0]
            if result and result.success and result.remaining_goals is not None:
                # exact?/apply? 返回的 "goal" 其实是建议列表
                for suggestion in result.remaining_goals[:3]:
                    if suggestion.startswith("Try this:"):
                        tac = suggestion.replace("Try this:", "").strip()
                        candidates.append(RepairCandidate(
                            tac, 0.85,
                            f"Lean4 {search_tac} found this match",
                            source=search_tac))

        return candidates

    def _record_failure(self, result: TacticFeedback):
        """记录失败模式, 用于跨尝试知识积累"""
        key = f"`{result.tactic}` fails ({result.error_category})"
        self._known_failures.add(key)
        self._error_history.append({
            "tactic": result.tactic,
            "category": result.error_category,
            "message": result.error_message[:200],
        })
        # 限制历史长度
        if len(self._error_history) > 100:
            self._error_history = self._error_history[-50:]

    def clear(self):
        """重置 (新问题开始时调用)"""
        self._error_history.clear()
        self._known_failures.clear()
        self._search_calls = 0
