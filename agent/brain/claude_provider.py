"""agent/brain/claude_provider.py — 同步 provider 兼容层 (重建)

历史: 本文件在某次清理中被删除, 但 ``run_mcts_eval.py``(line 30) 仍
``from agent.brain.claude_provider import create_provider`` —— 导致
MCTS 评测入口自上传版本起就 import 即崩 (与本仓其它入口无关的
既有缺陷, 非新功能引入)。

重建策略: 不复刻旧的同步 ClaudeProvider 实现, 而是把唯一维护的
``create_async_provider``(async_llm_provider.py) 包成同步
:class:`LLMProvider` —— 单一事实源, provider 名/默认值与异步路径
永远一致 (anthropic/mock/openai/deepseek/vllm/sglang/ollama/
openai_compat)。

仅 ``run_mcts_eval.py`` 的 TacticSuggester 使用; 其它入口
(run_unified / run_eval) 直接走异步 provider, 不经过本模块。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging

from agent.brain.async_llm_provider import (
    AsyncLLMProvider, LLMResponse, create_async_provider)
from agent.brain.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class SyncProviderAdapter(LLMProvider):
    """把 AsyncLLMProvider 适配成同步 LLMProvider。

    事件循环策略: 调用方无运行中 loop 时直接 ``asyncio.run``;
    已在 loop 内 (例如被异步代码误用) 则丢到独立线程的私有 loop,
    避免 "asyncio.run() cannot be called from a running event loop"。
    """

    def __init__(self, inner: AsyncLLMProvider, model_name: str = ""):
        self._inner = inner
        self._model_name = model_name or getattr(inner, "_model", "") \
            or getattr(inner, "model", "")

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, system: str = "", user: str = "",
                 temperature: float = 0.7, tools: list = None,
                 max_tokens: int = 4096,
                 model: str | None = None) -> LLMResponse:
        # ``model`` 形参属于 ABC 契约; 异步 provider 在构造期绑定模型,
        # 此处仅接受并忽略 (与 CachedProvider 的透传行为一致即可)。
        coro = self._inner.generate(
            system=system, user=user, temperature=temperature,
            tools=tools, max_tokens=max_tokens)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # 罕见路径: 同步接口被运行中的 loop 调用
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()


def create_provider(config: dict) -> LLMProvider:
    """同步工厂 — config dict 契约与 ``create_async_provider`` 相同。

    兼容旧字段名: ``base_url`` 视为 ``api_base`` 的别名
    (run_mcts_eval 传两者, 以 api_base 优先)。
    """
    cfg = dict(config or {})
    if not cfg.get("api_base") and cfg.get("base_url"):
        cfg["api_base"] = cfg["base_url"]
    inner = create_async_provider(cfg)
    return SyncProviderAdapter(inner, model_name=cfg.get("model", ""))
