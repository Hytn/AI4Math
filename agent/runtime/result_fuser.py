"""agent/runtime/result_fuser.py — 多智能体结果融合

从多个子智能体的异构结果中选择最佳、合并洞察。

策略:
  best_confidence: 选置信度最高的
  first_success:   选第一个有有效证明代码的
  merge_insights:  合并所有智能体的文本洞察为摘要
"""
from __future__ import annotations
from agent.runtime.sub_agent import AgentResult


class ResultFuser:
    """融合多个子智能体的结果"""

    def select_best(self, results: list[AgentResult],
                    strategy: str = "best_confidence") -> AgentResult:
        if not results:
            return AgentResult(agent_name="empty", role=None, content="",
                               confidence=0.0)

        if strategy == "first_success":
            for r in results:
                if r.proof_code.strip() and "sorry" not in r.proof_code:
                    return r
            # fallback to best_confidence
            return max(results, key=lambda r: r.confidence)

        # best_confidence (default)
        return max(results, key=lambda r: r.confidence)

    def merge_insights(self, results: list[AgentResult],
                       max_per_agent: int = 300) -> str:
        """合并多个智能体的分析洞察

        用于构建下一轮 prompt 的上下文, 而不是直接选最佳结果。
        """
        parts = []
        for r in results:
            if not r.content.strip():
                continue
            summary = r.content[:max_per_agent].strip()
            parts.append(
                f"[{r.agent_name}] (confidence={r.confidence:.2f}): "
                f"{summary}")
        return "\n\n".join(parts)

    def extract_useful_lemmas(self, results: list[AgentResult]) -> list[str]:
        """从所有结果中提取可能有用的引理名称"""
        import re
        lemmas = set()
        for r in results:
            # 匹配 Mathlib 引理名 (如 Nat.add_comm, List.map_nil)
            found = re.findall(
                r'\b([A-Z][a-zA-Z0-9]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b',
                r.content)
            lemmas.update(found)
        return sorted(lemmas)

    def get_proof_candidates(self, results: list[AgentResult]) -> list[str]:
        """提取所有非空的证明代码候选"""
        candidates = []
        for r in sorted(results, key=lambda x: -x.confidence):
            if r.proof_code.strip():
                candidates.append(r.proof_code)
        return candidates
