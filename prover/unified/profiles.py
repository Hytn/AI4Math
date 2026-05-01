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

    # ─ 基础设施大一统: 来自社区配套工作的新工具 ─
    BATCH_VERIFY    = "batch_verify"      # Kimina Lean Server: 批量编译多个 proof
    MVAR_FOCUS      = "mvar_focus"        # Pantograph: 旋转 goal 列表到指定 mvar
    DRAFT_HOLE      = "draft_hole"        # Pantograph: 插入 sorry-hole (DSP 原生)
    LEMMA_BY_LEMMA  = "lemma_by_lemma"    # LooKeng: 一次提交一个 lemma 而非整证
    NL_EXISTENCE    = "nl_existence"      # NFL-HR: NL→FL existence-theorem 桥接

    # ─ v6: 猜想驱动证明 (主动提出辅助引理) ─
    CONJECTURE_PROPOSE = "conjecture_propose"  # propose auxiliary lemmas via LLM


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
    inject_similar_dialogs —— v5: 初始 user message 是否预先粘上相似的过往对话 demo
                              (来自 KnowledgeReader/DialogIndex). 默认 False —
                              opt-in 以避免对未配置 DialogIndex 的 caller 产生
                              意外上下文长度增长.
    n_similar_dialogs —— v5: 注入相似对话的条数上限.
    similar_dialogs_max_chars —— v5: 注入相似对话块的字符上限.
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
    # v5 — 跨问题对话检索
    inject_similar_dialogs: bool = False           # opt-in: 默认关
    n_similar_dialogs: int = 3
    similar_dialogs_max_chars: int = 2000


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

    # ─ Family 6: 基础设施大一统 (社区配套基建对应的 profile) ───────────
    #
    # 每个 profile 都映射到一个具体的社区基建工作:
    #   kimina_batch    ↔ Kimina Lean Server (Numina/Kimi)
    #   pantograph_dsp  ↔ Pantograph (Numina/Stanford/CMU)
    #   lookeng_lemma   ↔ LooKeng (Seed-Prover 1.5)
    #   nfl_hybrid      ↔ NFL-HR (Yao et al., EMNLP 2025)
    #
    # 它们的公共契约: 每个都是 "推理范式 × 具体基建后端" 的笛卡尔积. 把
    # backend 选择从 verifier 层解耦到 profile 层后, 切基建就和切推理范式
    # 一样, 都只动一个 --profile 旗.

    "kimina_batch": Profile(
        name="kimina_batch",
        description=(
            "Kimina Lean Server 路线: 大批量 pass@k 通过 REST 一次提交. "
            "适合 RL roll-out / benchmark sweep. 推理仍是 whole-proof 风格."
        ),
        tools=[ToolKit.LEAN_VERIFY, ToolKit.BATCH_VERIFY],
        max_turns=2,
        framing="whole_proof_repair",
        observation=ObservationPolicy(
            auto_inject_lean_compile=True,
            inject_premises_in_prompt=True,
            inject_few_shot=True,
        ),
    ),

    "pantograph_dsp": Profile(
        name="pantograph_dsp",
        description=(
            "Pantograph 路线: 显式 mvar coupling + DSP 原生 drafting. "
            "对多 conjunct/case-split goal 收益最大."
        ),
        tools=[ToolKit.DECOMPOSE, ToolKit.PREMISE_SEARCH,
                ToolKit.DRAFT_HOLE, ToolKit.MVAR_FOCUS,
                ToolKit.TACTIC_APPLY, ToolKit.GOAL_INSPECT,
                ToolKit.LEAN_VERIFY],
        max_turns=20,
        framing="pantograph_dsp",
        observation=ObservationPolicy(
            auto_inject_goal_state=True,
            auto_inject_lean_compile=True,
        ),
    ),

    "lookeng_lemma": Profile(
        name="lookeng_lemma",
        description=(
            "LooKeng 路线: stateless REPL + 一次提交一个 lemma. "
            "长证明/PutnamBench/FATE-X 上 I/O 减约 40%."
        ),
        tools=[ToolKit.LEMMA_BY_LEMMA, ToolKit.PREMISE_SEARCH,
                ToolKit.LEMMA_BANK],
        max_turns=40,
        framing="lookeng_lemma",
        observation=ObservationPolicy(
            auto_inject_lean_compile=False,  # lemma_by_lemma 自带验证
            auto_inject_goal_state=False,
            inject_premises_in_prompt=False, # 通过 tool 按需检索
            inject_few_shot=True,
        ),
    ),

    "nfl_hybrid": Profile(
        name="nfl_hybrid",
        description=(
            "NFL-HR 路线: NL-FL hybrid reasoning. NL→FL existence-theorem "
            "桥接, FL prover 在 Long-CoT 里同时回答 NL + 证明 FL."
        ),
        tools=[ToolKit.NL_EXISTENCE, ToolKit.LEAN_VERIFY,
                ToolKit.PREMISE_SEARCH, ToolKit.CAS],
        max_turns=8,
        framing="nfl_hybrid",
        observation=ObservationPolicy(
            auto_inject_lean_compile=True,
            inject_premises_in_prompt=True,
            inject_few_shot=True,
        ),
    ),

    # ─ Family 8 (v6): Conjecture-driven proving ──────────────────────
    #
    # 主动提出辅助引理 → 验证 → 用作 main proof 的 stepping stone.
    # 与 dsp 的区别: dsp 是把目标"切下去"(decomposition); conjecture_driven
    # 是把可能有用的辅助命题"猜上来"(generation). 二者正交, 一个 profile
    # 也可以同时启用 (在 tools 里加 DECOMPOSE).
    #
    # 这条 profile 终于把 ``prover/conjecture/`` 包接上了主管线 ——
    # V6 之前 ConjectureProposer/ConjectureVerifier 只能由测试代码/示例
    # 脚本直接 import, 没有 Profile 把它当作可调用的 tool 暴露给 LLM.
    "conjecture_driven": Profile(
        name="conjecture_driven",
        description=(
            "猜想驱动: LLM 主动提出辅助引理 → 验证 → 作为 stepping "
            "stone 完成主证明. 适合 PutnamBench/FATE-X 等需要非平凡 "
            "中间结论的题目."
        ),
        tools=[
            ToolKit.CONJECTURE_PROPOSE,
            ToolKit.LEAN_VERIFY,
            ToolKit.PREMISE_SEARCH,
            ToolKit.LEMMA_BANK,        # store proved conjectures here
            ToolKit.DECOMPOSE,          # cooperates well with subgoal split
        ],
        max_turns=15,
        framing="conjecture_driven",
        observation=ObservationPolicy(
            auto_inject_lean_compile=True,
            inject_premises_in_prompt=True,
            inject_few_shot=True,
        ),
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Family 7: Tree-search profiles (mcts / best_first / beam) — v4 合流
# ═══════════════════════════════════════════════════════════════════════
#
# v3 把这三个 preset 隔离在 EXPERIMENTAL_PRESETS 中, 因为搜索结构与
# dialog-linear 主管线不兼容. v4 在 dialog.json schema 3.0 中加了
# meta.search_tree 块, 把树状探索元数据原生写入主存储格式 ——
# 既保持线性 messages 兼容下游 SFT, 又保留完整的搜索 DAG 供分析/replay。
#
# 现在它们与其他 9 个 profile 完全平等: 同一份 PRESETS, 同一个
# UnifiedProofRunner.run() 入口, 同一份 dialog.json 输出. 切换还是只
# 改 --profile 一个旗.

PRESETS["best_first"] = Profile(
    name="best_first",
    description=(
        "Best-first search: 外部 driver 选启发式分最高的开放叶子, "
        "每个节点用 max_turns=1 的 agent expansion. v4 起进入 PRESETS, "
        "search_tree 写入 meta.search_tree."
    ),
    tools=[ToolKit.TACTIC_APPLY],
    max_turns=1,
    framing="step_level_pure",
    search=SearchConfig(
        kind="best_first",
        max_nodes=200, max_depth=25,
        expansion_max_turns=1,
    ),
    observation=ObservationPolicy(auto_inject_goal_state=True),
)

PRESETS["mcts"] = Profile(
    name="mcts",
    description=(
        "MCTS-UCB1 search: selection + backprop 由 driver 完成, "
        "agent 只做 expansion. 树结构进 dialog.json 的 meta.search_tree. "
        "(v4 起从 EXPERIMENTAL_PRESETS 升正)"
    ),
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
        include_search_state_in_prompt=True,  # LLM 看到祖先链 + 兄弟分支
    ),
)

PRESETS["beam"] = Profile(
    name="beam",
    description=(
        "Beam search: 每深度保留 top-W 节点的 best-first 变体. "
        "search_tree 写入 meta.search_tree."
    ),
    tools=[ToolKit.TACTIC_APPLY],
    max_turns=1,
    framing="step_level_pure",
    search=SearchConfig(kind="beam", beam_width=8, max_depth=20),
    observation=ObservationPolicy(auto_inject_goal_state=True),
)


# ═══════════════════════════════════════════════════════════════════════
# Experimental: 留作未来不确定方向; 当前为空, 但保留 API.
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENTAL_PRESETS: dict[str, Profile] = {}


def enable_experimental_search_presets() -> None:
    """Back-compat shim: tree-search presets (best_first / mcts / beam)
    are now in PRESETS by default thanks to v4's dialog.json schema 3.0
    merge. This function used to opt-in three EXPERIMENTAL_PRESETS into
    PRESETS; it is kept callable so older scripts don't break, but it
    has no effect now and will be removed in a later version.

    EXPERIMENTAL_PRESETS itself remains as an extension point for any
    *future* presets we want to gate behind explicit opt-in.
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
