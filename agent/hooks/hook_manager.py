"""agent/hooks/hook_manager.py — 钩子注册与调度中心

管理所有已注册的钩子，按事件类型和优先级分发执行。

用法::

    manager = HookManager()
    manager.register(HookEvent.POST_VERIFICATION, NatSubSafetyHook())
    manager.register(HookEvent.ON_ROUND_END, RepetitionDetectorHook())

    # 在证明流程中调用
    result = manager.fire(HookEvent.POST_VERIFICATION, context)
    if result.action == HookAction.SKIP:
        # 预过滤拒绝, 跳过 Lean4 编译
        ...
"""
from __future__ import annotations
import logging
import re
from collections import defaultdict

from agent.hooks.hook_types import (
    Hook, HookEvent, HookAction, HookContext, HookResult
)

logger = logging.getLogger(__name__)


class HookManager:
    """生命周期钩子管理器"""

    def __init__(self):
        self._hooks: dict[HookEvent, list[tuple[int, Hook]]] = defaultdict(list)

    def register(self, event: HookEvent, hook: Hook, priority: int = 50):
        """注册钩子 (priority 越小越先执行)"""
        self._hooks[event].append((priority, hook))
        self._hooks[event].sort(key=lambda x: x[0])
        logger.debug(f"Hook registered: {hook.name} → {event.value} "
                      f"(priority={priority})")

    def register_from_plugin(self, plugin_hooks: dict):
        """从插件清单的 hooks 字段批量注册

        plugin_hooks 格式::
            {
                "on_error": [
                    {"pattern": "ring.*failed.*Nat",
                     "action": "inject_context",
                     "content": "ℕ减法问题..."}
                ]
            }
        """
        for event_key, rules in plugin_hooks.items():
            event = self._map_event(event_key)
            if not event:
                continue
            for rule in rules:
                hook = PatternMatchHook(
                    pattern=rule.get("pattern", ""),
                    action_type=rule.get("action", "inject_context"),
                    content=rule.get("content", ""),
                )
                self.register(event, hook, priority=60)

    def fire(self, event: HookEvent, ctx: HookContext) -> HookResult:
        """按优先级依次执行钩子, 遇到非 CONTINUE 即停止"""
        hooks = self._hooks.get(event, [])
        for priority, hook in hooks:
            try:
                result = hook.execute(ctx)
                if result.action != HookAction.CONTINUE:
                    logger.info(
                        f"Hook '{hook.name}' fired on {event.value}: "
                        f"{result.action.value} — {result.message[:100]}")
                    return result
            except Exception as e:
                logger.warning(f"Hook '{hook.name}' error: {e}")
        return HookResult()

    def list_hooks(self) -> dict[str, list[str]]:
        """列出所有已注册的钩子 (用于调试)"""
        return {
            event.value: [h.name for _, h in hooks]
            for event, hooks in self._hooks.items()
            if hooks
        }

    @staticmethod
    def _map_event(key: str) -> HookEvent | None:
        mapping = {
            "on_error": HookEvent.POST_VERIFICATION,
            "on_problem_start": HookEvent.ON_PROBLEM_START,
            "pre_generation": HookEvent.PRE_GENERATION,
            "post_generation": HookEvent.POST_GENERATION,
            "pre_verification": HookEvent.PRE_VERIFICATION,
            "on_round_end": HookEvent.ON_ROUND_END,
            "on_strategy_switch": HookEvent.ON_STRATEGY_SWITCH,
        }
        return mapping.get(key)


class PatternMatchHook(Hook):
    """基于正则模式匹配的通用钩子

    当错误信息或证明代码匹配指定模式时触发。
    支持 inject_context (注入上下文) 和 skip (跳过) 两种动作。
    """

    def __init__(self, pattern: str, action_type: str = "inject_context",
                 content: str = ""):
        self.pattern = pattern
        self.action_type = action_type
        self.content = content
        self.name = f"pattern:{pattern[:30]}"
        self._compiled = re.compile(pattern, re.IGNORECASE) if pattern else None

    def execute(self, ctx: HookContext) -> HookResult:
        if not self._compiled:
            return HookResult()

        # 在错误信息中匹配
        search_text = ctx.dominant_error
        if ctx.errors:
            search_text += " ".join(
                str(e.get("message", "") if isinstance(e, dict)
                    else str(e))
                for e in ctx.errors[:5]
            )

        if not self._compiled.search(search_text):
            return HookResult()

        if self.action_type == "skip":
            return HookResult(
                action=HookAction.SKIP,
                message=f"Pattern '{self.pattern}' matched, skipping")

        # inject_context (default)
        return HookResult(
            action=HookAction.MODIFY,
            message=f"Pattern '{self.pattern}' matched, injecting hint",
            inject_context={"domain_hint": self.content})
