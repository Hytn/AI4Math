"""prover.unified — 统一定理证明主管线

把所有定理证明算法收编到单一 ``AgentLoop`` 上。每个算法 = 一个 Profile
(声明式开关组合: tools + max_turns + framing)。切换算法 = 换 profile 名。

Public API
==========

数据层::

    Profile / get_profile / register_profile
    PRESETS                      —— 内建 14 个 active preset
    ToolKit / SearchConfig / ObservationPolicy / StopCondition

执行层::

    UnifiedProofRunner           —— 主入口 (Profile → dialog.json)
    UnifiedResult                —— 统一结果类型

兼容桥::

    unified_to_attempt           —— UnifiedResult → ProofAttempt
                                   (run_eval.py 累积 pass@k 用)

YAML profile 加载::

    load_profile_from_yaml / register_profile

典型用法
========

直接调用 (单题)::

    runner = UnifiedProofRunner(llm=async_llm, lean_pool=pool)
    result = await runner.run(problem, profile_name="whole_proof_repair")
    result.save_unified("results/traces/<id>")

加新算法 = 在 ``profiles.PRESETS`` 加一个 entry, 不动 runner / loop / tools。
"""
from prover.unified.profiles import (
    Profile, get_profile, register_profile, list_profiles,
    PRESETS,
    ToolKit, SearchConfig, ObservationPolicy, StopCondition,
    load_profile_from_yaml,
)
from prover.unified.runner import UnifiedProofRunner, UnifiedResult
from prover.unified.adapters import unified_to_attempt
from prover.unified.llm_autoformalizer import (
    register_llm_autoformalizer,
)

__all__ = [
    "Profile", "get_profile", "register_profile", "list_profiles",
    "PRESETS",
    "ToolKit", "SearchConfig", "ObservationPolicy", "StopCondition",
    "load_profile_from_yaml",
    "UnifiedProofRunner", "UnifiedResult",
    "unified_to_attempt",
    "register_llm_autoformalizer",
]
