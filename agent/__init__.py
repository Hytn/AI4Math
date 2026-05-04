"""agent/ — 智能体运行时层

LLM ↔ tool 的多轮交互内核, 上层 prover/ 通过 ``UnifiedProofRunner``
组装 agent 实例。本层不直接对外暴露 CLI;最终用户从 run_*.py 进入。

核心子模块:
    agent.runtime       AgentLoop ★ (主入口唯一调用)
    agent.brain         LLM provider (Async Claude / Mock / Cached)
    agent.tools         ToolRegistry + 9 个 builtin tool
    agent.persistence   dialog.json + SFT export + unified storage

历史模块 (v9/v10/v11/v12 清理后已删除, 0 主路径调用方):
    agent.hooks         生命周期 hook (v9 删; v12 连同 common/hook_types.py 一并删)
    agent.plugins       领域策略插件 (v9 删; v12 把 Profile.plugins 字段也删了)
    agent.memory        episodic memory + persistent knowledge
                        (主代码用 common/working_memory.py 取代; v12 二者一起删除)
    agent.context       上下文窗口管理 + 错误压缩 (v10 删; v12 把
                        ObservationPolicy.{compress_errors_budget,
                        visible_history_turns} 配套字段也删了)
    agent.executor      Lean 子进程执行器 (v11 删 — LeanEnvironment 不
                        暴露 verify_complete, 引导 run_eval.py 走静默
                        坏路径; 现在 real-Lean 直接用 AsyncLeanPool)
"""
