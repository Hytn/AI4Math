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
from typing import Literal
from enum import Enum

from common.constants import DEFAULT_CLAUDE_MODEL

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

    # ─ 
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
    inject_premises_in_prompt —— 初始 user message 是否预先粘上检索到的引理 (top-N).
    n_premises —— 初始注入的引理条数上限.
    inject_few_shot —— 初始 user message 是否附上 few-shot 示例.
    inject_similar_dialogs —— 
                              (来自 KnowledgeReader/DialogIndex). 默认 False —
                              opt-in 以避免对未配置 DialogIndex 的 caller 产生
                              意外上下文长度增长.
    n_similar_dialogs —— 
    similar_dialogs_max_chars —— 

    v12 移除: ``compress_errors_budget`` 与 ``visible_history_turns`` —
    这两个字段从来没有被 runner 或 agent_loop 读取过, 是已删 ``agent.context``
    子系统的孤儿. 在 YAML 设它们等同 no-op, 删掉避免误导.
    """
    auto_inject_lean_compile: bool = True
    auto_inject_goal_state: bool = False
    include_search_state_in_prompt: bool = False  # 把当前搜索树注入 prompt
    include_knowledge_briefing: bool = True       # KnowledgeReader 简报
    inject_premises_in_prompt: bool = True        # 
    n_premises: int = 10                           # 
    inject_few_shot: bool = True                   # 

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
    model: str = DEFAULT_CLAUDE_MODEL

    # 3. System prompt 引导 —— 算法的"思维框架"
    #    具体 prompt 在 system_prompts.py 中按 framing 名字查表
    framing: str = "whole_proof"

    # 4. 外部搜索 (可选)
    search: SearchConfig = field(default_factory=SearchConfig)

    # 5. observation 怎么回流
    observation: ObservationPolicy = field(default_factory=ObservationPolicy)

    # 6. 终止条件
    stop: StopCondition = field(default_factory=StopCondition)

    # 7. 
    #    True  → CRITICAL integrity issues mark `verified=False` even
    #            when Lean accepts the proof. Use for competition-
    #            style benchmarks (PutnamBench, FormalMATH) where
    #            ``native_decide`` / ``Decidable.decide`` / large
    #            heartbeats are forbidden.
    #    False → integrity issues become an advisory note in the
    #            response but do not flip ``verified``. Default for
    #            mainline benchmarks (miniF2F, ProofNet) where these
    #            tactics are perfectly legitimate Mathlib usage.
    integrity_strict: bool = False

    # Profile but never read by anyone — ``agent.plugins`` was deleted
    #  and nothing replaced the consumer side.

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

    # ─ Family 8: Conjecture-driven proving ──────────────────────
    #
    # 主动提出辅助引理 → 验证 → 用作 main proof 的 stepping stone.
    # 与 dsp 的区别: dsp 是把目标"切下去"(decomposition); conjecture_driven
    # 是把可能有用的辅助命题"猜上来"(generation). 二者正交, 一个 profile
    # 也可以同时启用 (在 tools 里加 DECOMPOSE).
    #
    # 这条 profile 终于把 ``prover/conjecture/`` 包接上了主管线 ——

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
        "每个节点用 max_turns=1 的 agent expansion.  "
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
        "agent 只做 expansion. 树结构进 dialog.json 的 meta.search_tree."
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
# Family 9: DeepSeek-Prover-V2 specialised presets  (v17)
# ═══════════════════════════════════════════════════════════════════════
#
# These presets are tuned for the open-source DeepSeek-Prover-V2 family
# (7B / 671B). The goal is a head-to-head comparison with the paper's
# own evaluation harness on miniF2F-test, holding the model fixed. The
# paper reports for the 7B model:
#     non-CoT  pass@8192 = 75.0%
#     CoT      pass@8192 = 82.0%
# Our 5 profiles below let an evaluator A/B test what THIS framework
# adds on top of the paper's bare whole-proof pipeline (which is the
# `dsp_v2_cot` baseline below — held-fixed for fair comparison).
#
# IMPORTANT — these profiles assume `--model deepseek-ai/DeepSeek-
# Prover-V2-7B` (or 671B) and `--temperature 1.0`. The framing prompts
# are the verbatim ones from paper Appendix A and won't make sense
# with Claude / GPT-4 (those have their own preferred phrasings).

# 9a. Faithful paper baseline (non-CoT). Use this to verify the
#     framework reproduces the paper's number before you start adding
#     "tricks". If your pass@k here is far below 75%, something is
#     wrong with your vLLM deployment or your sampling temperature.
PRESETS["dsp_v2_non_cot"] = Profile(
    name="dsp_v2_non_cot",
    description=(
        "DeepSeek-Prover-V2 7B/671B 非 CoT 模式 (论文 Appendix A.1 "
        "原 prompt). 单次输出整证. 无 repair 循环."
    ),
    tools=[],
    max_turns=1,
    temperature=1.0,             # paper default for pass@k
    framing="deepseek_prover_v2_non_cot",
    observation=ObservationPolicy(
        auto_inject_lean_compile=False,
        include_knowledge_briefing=False,
        inject_premises_in_prompt=False,
        inject_few_shot=False,
    ),
    stop=StopCondition(on_text_only=True),
)

# 9b. Faithful paper baseline (CoT). The paper's strongest 7B number
#     (82.0% pass@8192) comes from this configuration. Use this as
#     the head-to-head against any of the "trick" profiles below.
PRESETS["dsp_v2_cot"] = Profile(
    name="dsp_v2_cot",
    description=(
        "DeepSeek-Prover-V2 7B/671B CoT 模式 (论文 Appendix A.2 原 "
        "prompt). 输出 proof plan + 整证. 无 repair 循环. "
        "↔ 论文 7B Table 1: pass@8192 = 82.0%."
    ),
    tools=[],
    max_turns=1,
    temperature=1.0,
    framing="deepseek_prover_v2_cot",
    observation=ObservationPolicy(
        auto_inject_lean_compile=False,
        include_knowledge_briefing=False,
        inject_premises_in_prompt=False,
        inject_few_shot=False,
    ),
    stop=StopCondition(on_text_only=True, max_total_tokens=400_000),
)

# 9c. Trick #1 — verify-and-fix loop on top of CoT. Paper does NOT
#     use repair (each sample is independent). Expected gain comes
#     from near-miss proofs: correct strategy, wrong lemma name /
#     wrong type. The Lean error is fed back to the same model and
#     it's asked to rewrite.
PRESETS["dsp_v2_repair"] = Profile(
    name="dsp_v2_repair",
    description=(
        "DSP-V2 CoT + 编译反馈循环. 每 sample 内最多 3 轮 verify+fix. "
        "论文不做 repair, 是本框架的第一层增量."
    ),
    tools=[ToolKit.LEAN_VERIFY],
    max_turns=4,                 # 1 initial + up to 3 repair
    temperature=1.0,
    framing="deepseek_prover_v2_repair",
    observation=ObservationPolicy(
        auto_inject_lean_compile=True,
        include_knowledge_briefing=False,
        inject_premises_in_prompt=False,
        inject_few_shot=False,
    ),
)

# 9d. Trick #2 — knowledge accumulation. Same as 9c but turns on the
#     persistent lemma bank + dialog index. Across the 244 problems,
#     successful lemmas / dialog snippets from earlier problems are
#     retrieved and injected into prompts of later problems. Activate
#     with --knowledge-db, --dialog-index, --lemma-bank-db. Without
#     those CLI flags this profile silently degrades to dsp_v2_repair.
PRESETS["dsp_v2_repair_knowledge"] = Profile(
    name="dsp_v2_repair_knowledge",
    description=(
        "DSP-V2 + repair + 跨题知识沉淀 (lemma bank, dialog index). "
        "需要 --knowledge-db / --lemma-bank-db / --dialog-index 才生效."
    ),
    tools=[ToolKit.LEAN_VERIFY, ToolKit.LEMMA_BANK,
           ToolKit.PREMISE_SEARCH],
    max_turns=4,
    temperature=1.0,
    framing="deepseek_prover_v2_repair",
    observation=ObservationPolicy(
        auto_inject_lean_compile=True,
        include_knowledge_briefing=True,
        inject_premises_in_prompt=True,
        n_premises=4,
        inject_few_shot=False,
        inject_similar_dialogs=True,
    ),
)

# 9e. Trick #3 — heterogeneous parallel ensemble of the 4 above.
#     A shared broadcast bus pipes any partial discovery (lemma
#     proven, error category etc.) across sub-profiles. Effectively
#     trades 4× LLM calls per sample for 4× more diverse strategies
#     per sample.
PRESETS["dsp_v2_heterogeneous"] = Profile(
    name="dsp_v2_heterogeneous",
    description=(
        "DSP-V2 4 路异构并行 (non-CoT + CoT + repair + repair+knowledge) "
        "+ broadcast bus. 任一路成功即整 sample 成功. "
        "样本预算 N 实际等价于单 profile 的 pass@(N×4) 上界."
    ),
    tools=[ToolKit.LEAN_VERIFY, ToolKit.BROADCAST],
    max_turns=4,
    temperature=1.0,
    framing="deepseek_prover_v2_repair",
    search=SearchConfig(
        kind="parallel",
        beam_width=4,
        parallel_profiles=[
            "dsp_v2_non_cot",
            "dsp_v2_cot",
            "dsp_v2_repair",
            "dsp_v2_repair_knowledge",
        ],
    ),
)

# shim were removed. They had been empty / no-op  when MCTS / 
# best_first / beam graduated into PRESETS. New "experimental" gating
# can be re-introduced if and when there's an actual preset to gate.

def get_profile(name: str) -> Profile:
    if name not in PRESETS:
        raise ValueError(
            f"Unknown profile '{name}'. Available: {sorted(PRESETS)}")
    return PRESETS[name]

def register_profile(profile: Profile) -> None:
    """运行时注册一个新 profile (例如从 YAML 加载)."""
    PRESETS[profile.name] = profile

def list_profiles() -> list[str]:
    """列出当前所有可用 profile 名 (含 register_profile 注册的)."""
    return sorted(PRESETS.keys())

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

    # fields themselves are gone; the YAMLs in config/profiles/* may
    # still mention them after auto-regen from older PRESETS.
    obs_d = {k: v for k, v in obs_d.items()
             if k not in {"compress_errors_budget", "visible_history_turns"}}
    if "plugins" in d:
        d = {k: v for k, v in d.items() if k != "plugins"}
    return Profile(
        name=d["name"],
        description=d.get("description", ""),
        tools=[ToolKit(t) for t in d.get("tools", [])],
        max_turns=d.get("max_turns", 1),
        temperature=d.get("temperature", 0.7),
        model=d.get("model", DEFAULT_CLAUDE_MODEL),
        framing=d.get("framing", "whole_proof"),
        search=SearchConfig(**search_d) if search_d else SearchConfig(),
        observation=ObservationPolicy(**obs_d) if obs_d else ObservationPolicy(),
        stop=StopCondition(**stop_d) if stop_d else StopCondition(),
        integrity_strict=d.get("integrity_strict", False),
    )
