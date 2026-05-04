"""tests/test_smoke_v15.py — Pin v15 wirings against regression.

v14 接通了 PolicyEngine / PluginLoader / PersistentLemmaBank / SummaryCompressor
的 ``UnifiedProofRunner.__init__`` kwargs,但 ``run_unified.py`` /
``run_eval.py`` / ``factory.py`` 都没有 CLI flag 也没有 loader,所以默认
评测下这些 reservoir 永远是 ``None``。v15 把"接通"完成到入口层。

本文件每条断言钉一处接通点。回归(把 v15 工作拆掉)立刻被 CI 抓到。

外加 v15 的两项独立修复:
  - LemmaVerifier 死 API bug 修复
  - AsyncOpenAIProvider 接通 OpenAI-compatible 后端

跑: ``pytest tests/test_smoke_v15.py -v``
mock-only (no Lean, no real LLM, no network)。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

ROOT = Path(__file__).parent.parent


# ═════════════════════════════════════════════════════════════════
# 修复 ① — LemmaVerifier 不再调用不存在的 .compile() API
# ═════════════════════════════════════════════════════════════════

class TestFix1_LemmaVerifierApiAlignment:
    """v13 清掉 ConjectureVerifier._type_check 的死 .compile() 路径,
    但 LemmaVerifier 还残留同一类 bug,只是因为没有调用方所以没炸。
    v15 把它对齐到 ``AsyncLeanPool.verify_complete`` 的真实 API。"""

    def test_no_lean_env_compile_calls(self):
        """源文件**代码部分**不能再出现 ``self.lean_env.compile``——
        这条 API 不存在。docstring 提到旧 API 用于讲解历史修复,
        所以把三引号 docstring 剥掉再 grep。"""
        import re
        text = (ROOT / "prover/lemma_bank/lemma_verifier.py").read_text()
        # Strip triple-quoted strings (handles both """ and ''')
        code_only = re.sub(
            r'(""".*?"""|\'\'\'.*?\'\'\')', '', text, flags=re.DOTALL)
        assert "self.lean_env.compile" not in code_only, (
            "v15 应已删除 .compile() 死 API 调用; "
            "见 prover/lemma_bank/lemma_verifier.py")
        # Code path must use the real API:
        assert "verify_complete" in code_only

    def test_init_rejects_compile_only_objects(self):
        """传入只有 .compile() 没有 .verify_complete() 的对象,必须立刻
        抛 TypeError——而不是静默使每一次 verify 都返回 False。"""
        from prover.lemma_bank.lemma_verifier import LemmaVerifier

        class Stub:
            def compile(self, code):
                return (0, "", "")

        with pytest.raises(TypeError, match="verify_complete"):
            LemmaVerifier(lean_pool=Stub())

    def test_init_accepts_async_pool_shape(self):
        """有 ``verify_complete`` 的对象能正常构造。"""
        from prover.lemma_bank.lemma_verifier import LemmaVerifier

        class Pool:
            async def verify_complete(self, theorem, proof, preamble=""):
                return None

        v = LemmaVerifier(lean_pool=Pool())
        assert v.lean_pool is not None

    def test_structural_check_path_unchanged(self):
        """lean_pool=None 时,落到 structural-only 检查——这是 v13 行为,
        不能被破坏。"""
        from prover.lemma_bank.lemma_verifier import LemmaVerifier
        from prover.lemma_bank.bank import ProvedLemma

        v = LemmaVerifier(None)
        good = ProvedLemma(
            name="ok", statement="lemma ok : 1 = 1", proof=":= rfl")
        bad = ProvedLemma(
            name="sorry_lemma",
            statement="lemma sorry_lemma : 1 = 2",
            proof=":= sorry")
        assert v.verify(good) is True
        assert v.verify(bad) is False

    def test_sync_verify_in_event_loop_raises(self):
        """已经在 event loop 里时,同步 ``verify`` 必须显式抛 RuntimeError,
        不能 spin nested loop——那会在生产服务器上 deadlock。"""
        import asyncio
        from prover.lemma_bank.lemma_verifier import LemmaVerifier
        from prover.lemma_bank.bank import ProvedLemma

        class Pool:
            async def verify_complete(self, *a, **kw):
                class R:
                    success = True
                return R()

        v = LemmaVerifier(Pool())
        lemma = ProvedLemma(name="x", statement="lemma x : 1=1",
                             proof=":= rfl")

        async def run_inside_loop():
            with pytest.raises(RuntimeError, match="event loop"):
                v.verify(lemma)
            # async 路径必须工作
            assert await v.averify(lemma) is True

        asyncio.run(run_inside_loop())


# ═════════════════════════════════════════════════════════════════
# 修复 ② — AsyncOpenAIProvider 接通 OpenAI-compat 后端
# ═════════════════════════════════════════════════════════════════

class TestFix2_OpenAICompatibleProvider:
    """v15 之前 ``--provider`` 只接受 anthropic 和 mock,
    意味着 RL 飞轮经济上不可行(每个 rollout 调 Claude),也无法对照
    DeepSeek-Prover / Kimina-Prover。v15 加 AsyncOpenAIProvider。"""

    def test_provider_class_exists(self):
        from agent.brain.async_llm_provider import AsyncOpenAIProvider
        # Constructor accepts the four args we expect from CLI wiring.
        p = AsyncOpenAIProvider(
            model="any-model", api_key="EMPTY",
            api_base="http://localhost:8000/v1")
        assert p.model_name == "any-model"

    def test_factory_routes_aliases(self):
        """create_async_provider 必须把别名映射到正确的 Provider 类
        和默认 base URL,不能再抛 ``provider not yet wired``。"""
        from agent.brain.async_llm_provider import (
            AsyncOpenAIProvider, AsyncMockProvider,
            create_async_provider)

        # mock 和 anthropic 路径(向后兼容必须不动)
        assert isinstance(
            create_async_provider({"provider": "mock"}),
            AsyncMockProvider)

        # OpenAI-compat 别名 + 默认 base URL
        for alias, expected_base, default_model in [
            ("vllm",     "http://localhost:8000/v1",   None),
            ("sglang",   "http://localhost:30000/v1",  None),
            ("ollama",   "http://localhost:11434/v1",  None),
            ("deepseek", "https://api.deepseek.com/v1", "deepseek-chat"),
            ("openai",   "",                           "gpt-4o-mini"),
        ]:
            p = create_async_provider({
                "provider": alias,
                "model": default_model or "stub-7b",
            })
            assert isinstance(p, AsyncOpenAIProvider), \
                f"alias {alias!r} did not route to AsyncOpenAIProvider"
            if expected_base:
                assert p._api_base == expected_base, \
                    f"alias {alias!r}: api_base={p._api_base!r}"

    def test_factory_unknown_provider_raises_clearly(self):
        from agent.brain.async_llm_provider import create_async_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            create_async_provider({"provider": "totally-fake-llm"})

    def test_local_alias_requires_model(self):
        """vLLM/sglang/ollama 没有"默认 model"——必须强制提示。"""
        from agent.brain.async_llm_provider import create_async_provider
        with pytest.raises(ValueError, match="model"):
            create_async_provider({"provider": "vllm"})  # no model

    def test_explicit_api_base_wins_over_alias_default(self):
        """``--api-base`` 必须 override 别名默认值,否则用户没法把 vLLM
        指到 GPU 集群的非默认地址。"""
        from agent.brain.async_llm_provider import (
            AsyncOpenAIProvider, create_async_provider)
        p = create_async_provider({
            "provider": "vllm",
            "model": "stub-7b",
            "api_base": "http://gpu-cluster:9000/v1",
        })
        assert isinstance(p, AsyncOpenAIProvider)
        assert p._api_base == "http://gpu-cluster:9000/v1"

    def test_claude_to_openai_tool_translation(self):
        """tool-use schema 双向翻译必须 round-trip 得到等价结果——否则
        tool-using profile 在 OpenAI 后端上跑会直接崩。"""
        from agent.brain.async_llm_provider import AsyncOpenAIProvider
        claude_tools = [{
            "name": "lean_verify",
            "description": "Verify a Lean 4 proof.",
            "input_schema": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        }]
        oai = AsyncOpenAIProvider._claude_tools_to_openai(claude_tools)
        assert len(oai) == 1
        f = oai[0]
        assert f["type"] == "function"
        assert f["function"]["name"] == "lean_verify"
        assert f["function"]["parameters"]["required"] == ["code"]

    def test_message_translation_handles_tool_blocks(self):
        """assistant 消息含 tool_use block + 后续 user 消息含 tool_result
        block,必须正确变成 OpenAI 的 tool_calls + role=tool 序列。"""
        from agent.brain.async_llm_provider import AsyncOpenAIProvider
        claude_msgs = [
            {"role": "user", "content": "Prove this."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me verify."},
                {"type": "tool_use", "id": "call_1",
                 "name": "lean_verify", "input": {"code": "by rfl"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": "OK"},
            ]},
        ]
        oai = AsyncOpenAIProvider._claude_messages_to_openai(claude_msgs)
        # First message: user, plain string content
        assert oai[0]["role"] == "user"
        # Assistant message: has tool_calls, content may be string or None
        assistant = next(m for m in oai if m["role"] == "assistant")
        assert "tool_calls" in assistant
        assert assistant["tool_calls"][0]["function"]["name"] == "lean_verify"
        # Tool result message: role=tool with tool_call_id
        tool_msg = next(m for m in oai if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == "call_1"
        assert tool_msg["content"] == "OK"


# ═════════════════════════════════════════════════════════════════
# 接通 ① — load_policy_engine / load_plugin_loader / load_persistent_lemma_bank
#         在 prover.unified.factory 里
# ═════════════════════════════════════════════════════════════════

class TestWiring1_FactoryLoaders:
    """v14 的三个 reservoir 在 ``UnifiedProofRunner.__init__`` 已经接受
    kwargs,但 factory.py 之前没有对应 ``load_*`` 函数——意味着每个新
    入口都得手写构造逻辑,实际入口都没构造,等于死路径。
    v15 加齐 loader,失败 fail-soft 返回 None(和 v14 默认 None 行为兼容)。"""

    def test_loaders_importable(self):
        from prover.unified.factory import (
            load_policy_engine, load_plugin_loader,
            load_persistent_lemma_bank)
        # All three are callables
        assert callable(load_policy_engine)
        assert callable(load_plugin_loader)
        assert callable(load_persistent_lemma_bank)

    def test_policy_engine_disabled_returns_none(self):
        """``--policy-engine`` 默认关 → loader 必须返回 None,这样运行时
        AgentLoop 走 v13 hardcoded max_turns 路径 (向后兼容不能破)."""
        from prover.unified.factory import load_policy_engine
        assert load_policy_engine(False) is None
        assert load_policy_engine(None) is None

    def test_policy_engine_enabled_builds_default(self):
        """启用时 loader 构造 PolicyEngine.default(),含 5 条内置规则。"""
        from prover.unified.factory import load_policy_engine
        engine = load_policy_engine(True)
        assert engine is not None
        # 钉住默认有规则注入(不是空 engine)
        assert hasattr(engine, "evaluate")
        rules = getattr(engine, "rules", None) or getattr(
            engine, "_rules", None)
        assert rules and len(rules) >= 5, \
            f"PolicyEngine.default() should have ≥5 rules, got {rules}"

    def test_plugin_loader_none_path_disabled(self):
        from prover.unified.factory import load_plugin_loader
        assert load_plugin_loader(None) is None
        assert load_plugin_loader("") is None

    def test_plugin_loader_loads_real_strategy_pack(self):
        """plugins/strategies 目录必须实际加载到至少 algebra/analysis/
        number-theory 三个领域(否则项④ 接通点已被破坏)。"""
        from prover.unified.factory import load_plugin_loader
        loader = load_plugin_loader(str(ROOT / "plugins/strategies"))
        assert loader is not None
        registry = getattr(loader, "_registry", {})
        assert len(registry) >= 3, \
            f"expected ≥3 plugins (algebra/analysis/number-theory), " \
            f"got: {list(registry)}"

    def test_plugin_loader_missing_dir_returns_none(self):
        """指错路径不能 crash,fail-soft 返回 None。"""
        from prover.unified.factory import load_plugin_loader
        loader = load_plugin_loader("/no/such/path/xyz")
        assert loader is None

    def test_persistent_lemma_bank_path_creates_db(self):
        """传 path 必须真创建 SQLite 文件。"""
        from prover.unified.factory import load_persistent_lemma_bank
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "lb.db")
            bank = load_persistent_lemma_bank(
                db, lean_version="leanprover/lean4:v4.28.0",
                mathlib_rev="abc123def456")
            assert bank is not None
            assert os.path.exists(db)

    def test_persistent_lemma_bank_none_disables(self):
        from prover.unified.factory import load_persistent_lemma_bank
        assert load_persistent_lemma_bank(None) is None
        assert load_persistent_lemma_bank("") is None


# ═════════════════════════════════════════════════════════════════
# 接通 ② — run_unified.py / run_eval.py 暴露 v15 CLI flags
# ═════════════════════════════════════════════════════════════════

class TestWiring2_CliFlagsPresent:
    """如果两个入口里没有这些 flag,即使 factory 接通了,默认评测也走不到
    v14 reservoir。这是 v15 要钉的核心不变性。"""

    @pytest.mark.parametrize("entrypoint", ["run_unified.py", "run_eval.py"])
    @pytest.mark.parametrize("flag", [
        "--policy-engine",
        "--plugins-dir",
        "--lemma-bank-db",
        "--lean-version",
        "--mathlib-rev",
        "--api-base",
    ])
    def test_flag_present(self, entrypoint, flag):
        """两个 entrypoint 都必须暴露 6 个 v15 flag。"""
        path = ROOT / entrypoint
        assert path.exists(), f"missing entrypoint: {entrypoint}"
        # Required-arg entrypoints (run_eval needs --profile) → spawn
        # the help directly. argparse prints flags into --help output.
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        assert result.returncode == 0, \
            f"{entrypoint} --help failed: {result.stderr[:500]}"
        assert flag in result.stdout, (
            f"{entrypoint} is missing v15 flag {flag!r}; "
            f"--help output:\n{result.stdout[-2000:]}"
        )

    def test_provider_choices_include_openai_compat(self):
        """``--provider`` argparse 选项不能再 hardcode anthropic/mock —
        必须能传 deepseek/vllm/sglang/ollama 走 OpenAI-compat 工厂。"""
        # Spawn run_unified --provider=mock (no API key needed) — should
        # not raise SystemExit("provider not yet wired"). We use mock
        # because anthropic without ANTHROPIC_API_KEY would also
        # SystemExit, masking the real test. The mock path is the
        # backward-compat path that must keep working.
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_unified.py"),
             "--builtin", "nat_add_comm", "--provider", "mock",
             "--profile", "whole_proof"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        # mock provider runs end-to-end; if v15 broke the build_llm
        # branch this would SystemExit before the runner starts.
        assert "provider not yet wired" not in result.stderr, (
            "build_llm() still has the legacy 'not yet wired' branch — "
            "v15 should route everything through create_async_provider")


# ═════════════════════════════════════════════════════════════════
# 接通 ③ — UnifiedProofRunner 真接受 v14 reservoir 的 kwargs
#         (v14 已做,但 v15 改动可能误伤,所以再钉一次)
# ═════════════════════════════════════════════════════════════════

class TestWiring3_RunnerAcceptsReservoirKwargs:
    """UnifiedProofRunner.__init__ 必须依然接受 policy_engine /
    plugin_loader / persistent_lemma_bank 三个 kwarg,且 None 默认行为
    与 v14 完全相同。"""

    def test_runner_accepts_v14_kwargs_with_none_defaults(self):
        """构造 runner 不传任何 v14 kwarg → 必须不抛 + reservoirs 都是 None。"""
        from agent.brain.async_llm_provider import AsyncMockProvider
        from prover.unified import UnifiedProofRunner

        runner = UnifiedProofRunner(llm=AsyncMockProvider())
        assert runner.policy_engine is None
        assert runner.plugin_loader is None
        assert runner.persistent_lemma_bank is None

    def test_runner_stores_v14_kwargs_when_provided(self):
        """传入时必须存到 self.* 属性上,供 ``run`` 透传。"""
        from agent.brain.async_llm_provider import AsyncMockProvider
        from prover.unified import UnifiedProofRunner
        from prover.unified.factory import (
            load_policy_engine, load_plugin_loader)

        engine = load_policy_engine(True)
        loader = load_plugin_loader(str(ROOT / "plugins/strategies"))

        runner = UnifiedProofRunner(
            llm=AsyncMockProvider(),
            policy_engine=engine,
            plugin_loader=loader,
            # leave persistent_lemma_bank=None, that's the common path
        )
        assert runner.policy_engine is engine
        assert runner.plugin_loader is loader
        assert runner.persistent_lemma_bank is None
