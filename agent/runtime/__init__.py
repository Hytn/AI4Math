"""agent/runtime — Agent 执行运行时

主路径只保留:
  - AgentLoop:    多轮 agentic 循环 (LLM ↔ tool feedback), 是
                   ``UnifiedProofRunner`` 的主执行内核。

历史模块 (
  - sub_agent / async_agent_pool / result_fuser / agent_tool / mailbox:
    这些是 Lane / 多智能体子系统的组件, 主入口
    (``run_unified.py``, ``run_eval.py --profile``) 完全不调用。
    异构并行 (``heterogeneous`` profile) 实际用
    ``UnifiedProofRunner._run_parallel`` 的 ``asyncio.gather``,
    不走 SubAgent 路径。
"""
