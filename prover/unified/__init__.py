"""prover.unified — 统一定理证明主管线

把所有定理证明算法收编到一个 AgentLoop 主管线里。
每个算法 = 一个 Profile (声明式开关组合)。
切换算法 = 换 profile name。

Public API
==========

数据层::

    Profile / get_profile / register_profile
    PRESETS                      —— 内建 8 个范式
    ToolKit / SearchConfig / ObservationPolicy / StopCondition

执行层::

    UnifiedProofRunner           —— 主入口 (Profile → dialog.json)
    UnifiedResult                —— 统一结果类型

兼容桥 (用于把统一 runtime 接入 ProofPipeline / HeterogeneousEngine)::

    unified_to_attempt           —— UnifiedResult → ProofAttempt
    unified_to_agent_result      —— UnifiedResult → AgentResult

典型用法
========

直接调用 (单题调试)::

    runner = UnifiedProofRunner(llm=async_llm, lean_pool=pool)
    result = await runner.run(problem, profile_name="whole_proof_repair")
    result.save_unified("results/traces/<id>")

通过 ProofPipeline (含 Lane 状态机 + checkpoint)::

    # ProofPipeline 内部会把 generate() 路由到 UnifiedProofRunner
    pipeline = ProofPipeline(components, config={"profile": "step_level_pure"})
    trace = pipeline.run(problem)

CLI::

    python run_unified.py --builtin nat_add_comm --profile whole_proof_repair
    python run_eval.py --benchmark minif2f --profile reprover

配置切换 (无代码改动)::

    # repair 风格 (项目原默认)
    pipeline = ProofPipeline(components, config={"profile": "whole_proof_repair"})

    # ReProver 风格
    pipeline = ProofPipeline(components, config={"profile": "reprover"})

    # 异构并行 (项目原卖点)
    pipeline = ProofPipeline(components, config={"profile": "heterogeneous"})
"""
from prover.unified.profiles import (
    Profile, ToolKit, SearchConfig, ObservationPolicy, StopCondition,
    PRESETS, EXPERIMENTAL_PRESETS,
    get_profile, register_profile, load_profile_from_yaml,
    enable_experimental_search_presets,
)
from prover.unified.runner import UnifiedProofRunner, UnifiedResult
from prover.unified.adapters import (
    unified_to_attempt, unified_to_agent_result,
)

__all__ = [
    # data
    "Profile", "ToolKit", "SearchConfig", "ObservationPolicy", "StopCondition",
    "PRESETS", "EXPERIMENTAL_PRESETS",
    "get_profile", "register_profile", "load_profile_from_yaml",
    "enable_experimental_search_presets",
    # runtime
    "UnifiedProofRunner", "UnifiedResult",
    # bridges
    "unified_to_attempt", "unified_to_agent_result",
]
