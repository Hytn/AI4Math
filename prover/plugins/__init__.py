"""prover.plugins — YAML-driven 领域插件

原位置 ``agent/plugins/`` ( 重新放到 ``prover/`` 层符合实际作用 ——
这是 prover 层的"领域注入"机制, 与 system_prompts 的 framing 互补:
framing 是 profile-level (整套推理风格), 插件是 problem-level (按定理领域
注入额外引理 + few-shot + 战略提示)。

对应数据目录 ``plugins/strategies/{algebra,analysis,number-theory}/``:
  plugin.yaml      元数据 + 关键词匹配 + 参数覆盖
  premises.jsonl   领域专用引理库 (BM25/embedding 检索基础)
  few_shot.md      领域特定的证明示例 (注入到 LLM prompt)
  tactics.yaml     推荐 tactic 列表 (system_prompts 注入)

主路径接通点:
  - prover/unified/runner.py::_build_initial_message: 在 problem 入口先调
    PluginLoader.match(theorem) → 把得分最高的插件的 few_shot + premises
    注入到首条 user message。
  - 这是 ℕ 减法 / 域同构 / 极限 等领域 bug 的统一防御 —— ``number-theory/
    few_shot.md`` 第 1 条规则就是 "Lean 4 ℕ 减法 truncates"。

设计原则:
  - 插件是声明式数据, 不是 Python 代码 (避免代码注入风险)
  - hooks 字段保留但不消费 (agent/hooks/ 已下线; 留给未来 PolicyEngine 规则集)
  - 关键词匹配是字符串包含; 未来可以改用 engine.world_model 的特征提取做更准
"""
from prover.plugins.loader import StrategyPlugin, PluginLoader

__all__ = ["StrategyPlugin", "PluginLoader"]
