"""agent/ — 智能体运行时层

LLM ↔ tool 的多轮交互内核, 上层 prover/ 通过 ``UnifiedProofRunner``
组装 agent 实例。本层不直接对外暴露 CLI;最终用户从 run_*.py 进入。

核心子模块:
    agent.runtime       AgentLoop ★ (主入口唯一调用)
    agent.brain         LLM provider (Async Claude / Mock / Cached)
    agent.tools         ToolRegistry + 9 个 builtin tool
    agent.persistence   dialog.json + SFT export + unified storage

历史模块:
    agent.hooks         生命周期 hook
    agent.plugins       领域策略插件
    agent.memory        episodic memory + persistent knowledge
    agent.context       上下文窗口管理 + 错误压缩
    agent.executor      Lean 子进程执行器
"""
