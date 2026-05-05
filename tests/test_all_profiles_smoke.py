"""tests/test_all_profiles_smoke.py — 14 profile × 5 题端到端冒烟

这是 README 的"5 分钟跑通"承诺的回归保护。

测试矩阵:
  * 14 个 profile (每个 profile 必须 ``success: true``)
  * builtin 内置 5 题 (检验 mock 启发式不是被某条特殊题面挑了便宜)
  * dialog.json 结构完整
  * 含工具的 profile (有 tools) 必须真的至少调过一次工具
    —— 防止 fast-path 跳过 tool 时仍报 success 的悬空成功

任何一格挂掉,意味着 README 宣传的某种方法在某种题型上断了主路径。
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from prover.unified import PRESETS, UnifiedProofRunner
from agent.brain.async_llm_provider import AsyncMockProvider
from benchmarks.datasets.builtin.problems import BUILTIN_PROBLEMS
from engine.async_lean_pool import AsyncLeanPool
from engine.transport import MockTransport


# ── 测试矩阵 ────────────────────────────────────────────────────────────

# 14 profile × 5 题 = 70 case,跑完整体 ~3-5 秒。
# 真实评测请走 run_eval.py + 真 Lean + 真 LLM,这里只保住主路径不断。
_PROFILE_NAMES = sorted(PRESETS.keys())
_PROBLEM_IDS = [p.problem_id for p in BUILTIN_PROBLEMS]

# 把 problem_id → BenchmarkProblem 实例索引一下,parametrize 用。
_BY_ID = {p.problem_id: p for p in BUILTIN_PROBLEMS}


# ── 跑一题的封装 ────────────────────────────────────────────────────────

async def _run_one(profile_name: str, problem_id: str,
                   out_dir: str) -> tuple[dict, bool, list[str]]:
    """跑一个 (profile, problem),返回 (dialog dict, success, tools_called)。"""
    pool = AsyncLeanPool(
        pool_size=2,
        transport_factory=lambda _sid: MockTransport(),
    )
    await pool.start()

    try:
        runner = UnifiedProofRunner(
            llm=AsyncMockProvider(),
            lean_pool=pool,
            knowledge_store=None,
            knowledge_writer=None,
            retriever=None,
        )
        result = await runner.run(_BY_ID[problem_id],
                                  profile_name=profile_name)
        result.save_unified(out_dir, problem_id=problem_id)
        with open(os.path.join(out_dir, "dialog.json")) as f:
            dialog = json.load(f)
        tools_called = (
            dialog.get("result", {})
                  .get("extra", {})
                  .get("tools_called", []))
        return dialog, result.success, list(tools_called)
    finally:
        try:
            await pool.close()
        except Exception:
            pass


# ── 矩阵化测试 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("profile_name", _PROFILE_NAMES)
@pytest.mark.parametrize("problem_id", _PROBLEM_IDS)
def test_profile_x_problem_smoke(profile_name: str, problem_id: str,
                                  tmp_path):
    """每个 (profile, problem) 都能在 mock+mock 下端到端跑通。"""
    out = str(tmp_path / profile_name / problem_id)
    os.makedirs(out, exist_ok=True)
    dialog, success, tools_called = asyncio.run(
        _run_one(profile_name, problem_id, out))

    # ── 1. runner.success 必须 True ───────────────────────────────────
    assert success, (
        f"profile={profile_name} problem={problem_id}: "
        f"runner.success=False — 主路径断了")

    # ── 2. dialog.json 结构齐全 ───────────────────────────────────────
    assert "schema_version" in dialog
    assert "messages" in dialog and len(dialog["messages"]) >= 1
    assert "result" in dialog
    assert dialog["result"].get("success") is True, (
        f"profile={profile_name} problem={problem_id}: "
        f"dialog.result.success != True (mock fast-path 失活?)")


# ── 工具调用闭环验证 ────────────────────────────────────────────────────
# 这条补救上一轮 smoke 的弱点:有 tools 的 profile 也可能因 mock fast-path
# 跳过工具直接出 raw text 而 success。我们要求至少有一类 profile 真的调过
# tool,否则 LLM↔tool 闭环就没被跑过。

# whole_proof 没有 tools,豁免。
# heterogeneous / dsp_v2_heterogeneous 内部跑 4 路 sub-runner,顶层 dialog
# 里 tools_called 可能为空,因为汇总在 sub-dialog 里 — 也豁免。
# 其他 profile 必须有非空 tools_called。
_TOOL_REQUIRED_PROFILES = [
    p for p in _PROFILE_NAMES
    if p not in ("whole_proof", "heterogeneous", "dsp_v2_heterogeneous")
    and PRESETS[p].tools
]


@pytest.mark.parametrize("profile_name", _TOOL_REQUIRED_PROFILES)
def test_profile_actually_calls_tools(profile_name: str, tmp_path):
    """有 tools 配置的 profile 不应在所有 5 题上都走 fast-path 跳过工具。"""
    any_tool_called = False
    seen_tools: set[str] = set()
    for pid in _PROBLEM_IDS:
        out = str(tmp_path / profile_name / pid)
        os.makedirs(out, exist_ok=True)
        _, success, tools_called = asyncio.run(
            _run_one(profile_name, pid, out))
        assert success
        if tools_called:
            any_tool_called = True
            seen_tools.update(tools_called)
    assert any_tool_called, (
        f"profile={profile_name}: 5 题都走 fast-path,从未调过任何工具 — "
        f"LLM↔tool 闭环没被走过,profile 配置可能形同虚设")
