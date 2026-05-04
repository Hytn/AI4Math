"""agent.brain — LLM provider 抽象 (v9: async-only)

支持的 provider:
    AsyncClaudeProvider     — Anthropic Claude (production)
    AsyncMockProvider       — 测试用 (确定性输出, 不依赖网络)
    AsyncCachedProvider     — 缓存装饰器 (LRU + tool-call key)

工厂函数:
    create_async_provider(config)   — 由 config dict 构造 provider

历史模块 (v9 删除, 0 主入口调用):
    llm_provider.py / claude_provider.py    — 同步 ABC + sync ClaudeProvider/MockProvider/CachedProvider
    sync_to_async_adapter.py                — 同步→异步桥 (run_eval 用)
所有路径现已直接使用 ``AsyncLLMProvider``。
"""
