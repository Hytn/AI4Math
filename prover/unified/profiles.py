"""prover/unified/profiles.py — 统一证明范式的开关盘

每个定理证明算法 = 一组 (tools, max_turns, system_prompt, search) 旋钮的取值。
这个文件就是那个开关盘 —— 不写算法逻辑, 只写"算法 X 长什么样"。

实际的执行由 prover/unified/runner.py 消费这个 Profile, 实例化 AgentLoop
(可选外加 SearchDriver) 完成。

设计原则:
  - 完全声明式: profile 不含任何 if/else, 只是数据
  - 与现有 AgentLoop 同构: max_turns/tools 直接喂 LoopConfig
  - 与现有 SearchCoordinator 同构: search.kind 直接选 best_first/ucb
  - 任何新算法 = 加一个 Profile, 不改主代码
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════
# Tool Kit ID — 每个 ID 对应 tool_kits.py 里一个 register_* 函数
# ═══════════════════════════════════════════════════════════════════════

class ToolKit(str, Enum):
    """声明式工具集. 每个 kit 决定 LLM 能看到什么、能做什么。"""

    # ─ 单工具 (可任意组合) ─
    LEAN_VERIFY     = "lean_verify"       # 编译完整证明 → {success, errors}
    TACTIC_APPLY    = "tactic_apply"      # 单步 tactic → {ok, new_goals, env_id}
    GOAL_INSPECT    = "goal_inspect"      # 查看当前 goal state
    TACTIC_SUGGEST  = "tactic_suggest"    # 试一批 auto-tactic
    PREMISE_SEARCH  = "premise_search"    # Mathlib 引理检索
    LEAN_AUTO       = "lean_auto"         # exact?/apply?/aesop hammer
    DECOMPOSE       = "decompose_subgoal" # 把目标拆成子目标
    LEMMA_BANK      = "lemma_bank"        # 查/存项目内已证引理
    CAS             = "cas_compute"       # 数值/符号计算
    BROADCAST       = "broadcast"         # 跨 agent 共享发现 (异构方向)

    # ─ 搜索专用工具 (仅当 search.kind != none 时启用) ─
    TREE_VIEW       = "tree_view"         # 看当前搜索树状态
    TREE_SELECT     = "tree_select"       # 选下一个要展开的节点 (LLM-driven 才用)


# ═══════════════════════════════════════════════════════════════════════
# Search Driver Config — 外部搜索算法 (可选)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SearchConfig:
    """是否在 AgentLoop 外面包一层搜索 driver, 以及哪种。

    none      —— 不包. 主管线就是单次 AgentLoop 调用. 适用于 whole-proof / repair.
    best_first —— 优先扩展启发式分高的节点. driver 调用 loop 做单步 expansion.
    ucb       —— MCTS-UCB1 selection + backprop. driver 调用 loop 做单步 expansion.
    beam      —— 每深度保留 top-K 节点, 每个节点跑一次 loop.
    parallel  —— 异构 N 个 profile 同时跑 (不同 system_prompt, 不同 model).
    """
    kind: Literal["none", "best_first", "ucb", "beam", "parallel"] = "none"

    # 通用搜索参数
    max_nodes: int = 200          # 全局节点上限
    max_depth: int = 25           # 树最大深度
    beam_width: int = 8           # beam / parallel 的宽度
    ucb_c: float = 1.414          # UCB 探索常数
    expansion_max_turns: int = 1  # 每个节点 expansion 时, agent loop 的 max_turns
                                  # (设 1 = 纯 tactic 生成; 设 >1 = 节点内允许调工具)

    # parallel 模式专用: 多个子 profile (异构方向)
    parallel_profiles: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Observation Policy — 观测如何返回给 LLM
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ObservationPolicy:
    """工具结果 (observation) 怎么塞回 LLM 的上下文。

    auto_inject_lean_compile —— 如果 LLM 输出了 ```lean 块且没显式调 lean_verify,
                                自动调用 verify 并把结果作为下一轮 observation 注入.
                                (你说的"Lean 代码生成 → 自动 parse → 送 Lean → 反馈")
    auto_inject_goal_state ——  apply tactic 成功后, 自动把 new_goals 注入 (即使 LLM 没问).
    compress_errors ——        长错误压缩到 N 字 (lane.summary_compressor).
    visible_history_turns —— 上下文窗口可见的历史轮数 (-1 = 全部).
    inject_premises_in_prompt —— 初始 user message 是否预先粘上检索到的引理 (top-N).
    n_premises —— 初始注入的引理条数上限.
    inject_few_shot —— 初始 user message 是否附上 few-shot 示例.
    """
    auto_inject_lean_compile: bool = True
    auto_inject_goal_state: bool = False
    compress_errors_budget: int = 1200
    visible_history_turns: int = -1
    include_search_state_in_prompt: bool = False  # 把当前搜索树注入 prompt
    include_knowledge_briefing: bool = True       # KnowledgeReader 简报
    inject_premises_in_prompt: bool = True        # v3 新增: 检索引理预注入
    n_premises: int = 10                           # v3 新增
    inject_few_shot: bool = True                   # v3 新增: few-shot 注入


# ═══════════════════════════════════════════════════════════════════════
# Stop Condition
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StopCondition:
    on_proof_found: bool = True       # 找到 sorry-free 证明立即停
    on_text_only: bool = False        # LLM 不再调工具就停 (whole-proof 风格)
    on_all_goals_closed: bool = True  # step-level: 所有 goal 闭合即停
    max_total_tokens: int = 200_000
    timeout_seconds: float = 300.0


# ═══════════════════════════════════════════════════════════════════════
# 顶层 Profile —— 一个算法的完整定义
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Profile:
    """一个完整的定理证明算法 = 一个 Profile."""
    name: str
    description: str = ""

    # 1. 工具集 —— 算法的"动作空间"
    tools: list[ToolKit] = field(default_factory=list)

    # 2. 主循环参数 —— 算法的"思考节奏"
    max_turns: int = 1
    temperature: float = 0.7
    model: str = "claude-sonnet-4-20250514"

    # 3. System prompt 引导 —— 算法的"思维框架"
    #    具体 prompt 在 system_prompts.py 中按 framing 名字查表
    framing: str = "whole_proof"

    # 4. 外部搜索 (可选)
    search: SearchConfig = field(default_factory=SearchConfig)

    # 5. observation 怎么回流
    observation: ObservationPolicy = field(default_factory=ObservationPolicy)

    # 6. 终止条件
    stop: StopCondition = field(default_factory=StopCondition)

    # 7. (可选) plugin 钩子 —— 用现有 agent.plugins 体系做领域微调
    plugins: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# 8 个 Built-in Preset —— 覆盖现存所有主流方法
# ═══════════════════════════════════════════════════════════════════════

PRESETS: dict[str, Profile] = {

    # ─ Family 1: Whole-proof generation ────────────────────────────────
    "whole_proof": Profile(
        name="whole_proof",
        description="DeepSeek-Prover/Kimina/Goedel 路线: 一次输出完整证明",
        tools=[],                       # 空工具集 = 关掉所有工具
        max_turns=1,
        framing="whole_proof",
        observation=ObservationPolicy(
            auto_inject_lean_compile=False,   # 严格单轮, 不回灌
            include_knowledge_briefing=True,  # 但仍可注入 few-shot 和检索的引理
        ),
        stop=StopCondition(on_text_only=True),
    ),

    "whole_proof_repair": Profile(
        name="whole_proof_repair",
        description="单轮生成失败后, 编译反馈循环修复 (现项目主路径)",
        tools=[ToolKit.LEAN_VERIFY],
        max_turns=6,
        framing="whole_proof_repair",
        observation=ObservationPolicy(
            auto_inject_lean_compile=True,    # 关键: 没显式调 verify 也自动调
        ),
    ),

    # ─ Family 2: Sketch-then-prove (DSP) ───────────────────────────────
    "dsp": Profile(
        name="dsp",
        description="先非形式化 sketch → 分解子目标 → 逐个形式化",
        tools=[ToolKit.DECOMPOSE, ToolKit.PREMISE_SEARCH, ToolKit.LEAN_VERIFY],
        max_turns=10,
        framing="dsp",
    ),

    # ─ Family 3: Retrieval-augmented (ReProver) ────────────────────────
    "reprover": Profile(
        name="reprover",
        description="ReProver 风格: 检索 Mathlib + 单步 tactic 应用",
        tools=[ToolKit.PREMISE_SEARCH, ToolKit.TACTIC_APPLY,
               ToolKit.GOAL_INSPECT],
        max_turns=30,
        framing="step_level_with_retrieval",
        observation=ObservationPolicy(
            auto_inject_goal_state=True,
            auto_inject_lean_compile=False,   # 步级范式不需要全编译
            inject_premises_in_prompt=False,   # ReProver: 步级按需检索
            inject_few_shot=False,             # 步级 prompt 不需要整证示例
        ),
    ),

    # ─ Family 4: Pure step-level (LeanDojo) ────────────────────────────
    "leandojo": Profile(
        name="leandojo",
        description="纯逐步证明: apply 一个 tactic, 看 goal, 再 apply",
        tools=[ToolKit.TACTIC_APPLY, ToolKit.GOAL_INSPECT, ToolKit.LEAN_AUTO],
        max_turns=50,
        framing="step_level_pure",
        observation=ObservationPolicy(
            auto_inject_goal_state=True,
            inject_premises_in_prompt=False,   # 纯步级, 通过 tool 探索
            inject_few_shot=False,
        ),
    ),

    # ─ Family 5: Heterogeneous parallel (项目现存特色) ─────────────────
    "heterogeneous": Profile(
        name="heterogeneous",
        description="N 个不同 framing/model/temp 的 sub-profile 并行 + 广播总线",
        tools=[ToolKit.LEAN_VERIFY, ToolKit.BROADCAST],
        max_turns=4,
        framing="whole_proof_repair",
        search=SearchConfig(
            kind="parallel",
            beam_width=4,
            parallel_profiles=[
                # 这些必须是已注册的 profile name. 不写就用默认 4 方向
                "whole_proof",          # 自动化探测
                "reprover",             # 检索路径
                "leandojo",             # 步级展开
                "whole_proof_repair",   # 修复路径
            ],
        ),
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Experimental: 树搜索范式 (best_first / mcts / beam) — v3 暂搁置
# ═══════════════════════════════════════════════════════════════════════
#
# 这些 preset 依赖 ``search_driver.py`` 的 SharedSearchState + driver 实现,
# 整套机制完整但与"线性 dialog 主管线"还没合流。v3 主管线先把 5 个非搜索
# 范式 (whole_proof / repair / dsp / reprover / leandojo / heterogeneous)
# 大一统; MCTS 系列下个版本再抬升到 dialog-tree 之上。
#
# 用户可显式 opt-in: ``register_profile`` + EXPERIMENTAL_PRESETS[name]。
EXPERIMENTAL_PRESETS: dict[str, Profile] = {

    "best_first": Profile(
        name="best_first",
        description="(实验) 外部 best-first driver 选节点, 每节点唤起 max_turns=1 的 agent expansion",
        tools=[ToolKit.TACTIC_APPLY],
        max_turns=1,
        framing="step_level_pure",
        search=SearchConfig(
            kind="best_first",
            max_nodes=200, max_depth=25,
            expansion_max_turns=1,
        ),
        observation=ObservationPolicy(auto_inject_goal_state=True),
    ),

    "mcts": Profile(
        name="mcts",
        description="(实验) UCB1 selection + backprop, agent 只做 expansion",
        tools=[ToolKit.TACTIC_APPLY],
        max_turns=1,
        framing="step_level_pure",
        search=SearchConfig(
            kind="ucb", ucb_c=1.414,
            max_nodes=400, max_depth=30,
            expansion_max_turns=1,
        ),
        observation=ObservationPolicy(
            auto_inject_goal_state=True,
            include_search_state_in_prompt=True,  # 让 LLM 看到祖先链 + 兄弟分支
        ),
    ),

    "beam": Profile(
        name="beam",
        description="(实验) 每深度保留 top-K 节点的 best-first 变体",
        tools=[ToolKit.TACTIC_APPLY],
        max_turns=1,
        framing="step_level_pure",
        search=SearchConfig(kind="beam", beam_width=8, max_depth=20),
    ),
}


def enable_experimental_search_presets() -> None:
    """显式注册实验性树搜索 preset (best_first / mcts / beam)。

    主管线默认不暴露这些, 因为它们与 dialog-linear 主路径还没合流。
    需要做 MCTS 实验时调用本函数; 之后 ``get_profile("mcts")`` 可用。
    """
    for name, prof in EXPERIMENTAL_PRESETS.items():
        PRESETS[name] = prof


def get_profile(name: str) -> Profile:
    if name not in PRESETS:
        raise ValueError(
            f"Unknown profile '{name}'. Available: {sorted(PRESETS)}")
    return PRESETS[name]


def register_profile(profile: Profile) -> None:
    """运行时注册一个新 profile (例如从 YAML 加载)."""
    PRESETS[profile.name] = profile


def load_profile_from_yaml(path: str) -> Profile:
    """从 YAML 加载 Profile —— 用户态扩展点, 不动代码就能加新算法."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return _profile_from_dict(data)


def _profile_from_dict(d: dict) -> Profile:
    search_d = d.get("search", {}) or {}
    obs_d = d.get("observation", {}) or {}
    stop_d = d.get("stop", {}) or {}
    return Profile(
        name=d["name"],
        description=d.get("description", ""),
        tools=[ToolKit(t) for t in d.get("tools", [])],
        max_turns=d.get("max_turns", 1),
        temperature=d.get("temperature", 0.7),
        model=d.get("model", "claude-sonnet-4-20250514"),
        framing=d.get("framing", "whole_proof"),
        search=SearchConfig(**search_d) if search_d else SearchConfig(),
        observation=ObservationPolicy(**obs_d) if obs_d else ObservationPolicy(),
        stop=StopCondition(**stop_d) if stop_d else StopCondition(),
        plugins=d.get("plugins", []),
    )
