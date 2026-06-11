"""tests/test_rebuilt_mcts_deps.py — run_mcts_eval 三个重建依赖的契约测试

上传版本中 ``agent.brain.claude_provider`` / ``prover.codegen`` /
``prover.verifier.lean_repl`` 缺失, 导致 run_mcts_eval import 即崩。
本文件钉死重建后的契约, 防止再次漂移。
"""
from __future__ import annotations

import asyncio
import threading

import pytest


class TestEntryPointImports:
    def test_all_entry_points_importable(self):
        import run_eval        # noqa: F401
        import run_unified     # noqa: F401
        import run_mcts_eval   # noqa: F401
        import run_hilbert     # noqa: F401


class TestSyncProviderAdapter:
    def test_create_provider_mock_roundtrip(self):
        from agent.brain.claude_provider import create_provider
        p = create_provider({"provider": "mock", "model": "m"})
        r = p.generate(system="s", user="prove it", temperature=0.0)
        assert getattr(r, "content", "")
        assert hasattr(r, "tokens_in") and hasattr(r, "tokens_out")

    def test_base_url_alias(self):
        from agent.brain.claude_provider import create_provider
        # run_mcts_eval 同时传 api_base 与 base_url; base_url 是旧别名
        p = create_provider({"provider": "mock", "model": "m",
                             "base_url": "http://x:1/v1"})
        assert p is not None

    def test_works_inside_running_loop(self):
        """同步接口被异步上下文误用时不得抛 'cannot be called…'。"""
        from agent.brain.claude_provider import create_provider
        p = create_provider({"provider": "mock", "model": "m"})

        async def caller():
            return p.generate(system="", user="hi")
        r = asyncio.run(caller())
        assert getattr(r, "content", "")


class TestCodeFormatter:
    def test_extract_proof_body_variants(self):
        from prover.codegen import extract_proof_body
        assert extract_proof_body("```lean\nby simp\n```") == "by simp"
        assert extract_proof_body(
            "theorem t : True := by\n  trivial").startswith("by")
        assert extract_proof_body("Nat.add_zero n") == \
            "by exact (Nat.add_zero n)"

    def test_hilbert_uses_shared_impl(self):
        """单一事实源: hilbert 与 codegen 必须是同一个函数对象。"""
        from prover.codegen.code_formatter import extract_proof_body as a
        from prover.hilbert import extract_proof_body as b
        assert a is b


class TestREPLResponseParsing:
    def test_complete_proof(self):
        from prover.verifier.lean_repl import parse_repl_output
        r = parse_repl_output({"env": 1, "messages": []},
                              had_trailing_sorry=False)
        assert r.success and r.is_complete and r.goals == []

    def test_partial_proof_reads_goals_from_sorries(self):
        from prover.verifier.lean_repl import parse_repl_output
        r = parse_repl_output(
            {"sorries": [{"goal": "n : ℕ\n⊢ n + 0 = n"}]},
            had_trailing_sorry=True)
        assert r.success and not r.is_complete
        assert "n + 0 = n" in r.goals[0]

    def test_error_surfaces_first_message(self):
        from prover.verifier.lean_repl import parse_repl_output
        r = parse_repl_output(
            {"messages": [{"severity": "error", "data": "unknown id foo"},
                          {"severity": "error", "data": "second"}]},
            had_trailing_sorry=True)
        assert not r.success and "unknown id foo" in r.error

    def test_none_response(self):
        from prover.verifier.lean_repl import parse_repl_output
        r = parse_repl_output(None, had_trailing_sorry=True)
        assert not r.success and r.error


class _FakeTransport:
    """脚本化 transport: 记录 cmd, 按序返回响应。"""
    is_fallback = False
    is_alive = True

    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    async def start(self):
        return True

    async def close(self):
        return None

    async def send(self, cmd):
        self.sent.append(cmd)
        return self.responses.pop(0) if self.responses else {}


def _make_repl(transport):
    from prover.verifier.lean_repl import LeanREPL
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return LeanREPL(transport, loop, t, timeout=5)


class TestLeanREPLSync:
    def test_check_tactic_sequence_complete(self):
        tr = _FakeTransport([
            {"env": 7},                      # preamble env
            {"env": 8, "messages": []},      # 无 sorry 一次过
        ])
        repl = _make_repl(tr)
        try:
            r = repl.check_tactic_sequence(
                "theorem t : 1 + 1 = 2 := by sorry", ["norm_num"],
                preamble="import Mathlib")
            assert r.success and r.is_complete
            # preamble env 被复用进后续请求
            assert tr.sent[1].get("env") == 7
            assert "norm_num" in tr.sent[1]["cmd"]
            assert "sorry" not in tr.sent[1]["cmd"]
        finally:
            repl.close()

    def test_check_tactic_sequence_partial(self):
        tr = _FakeTransport([
            {"env": 7},
            {"messages": [{"severity": "error", "data": "unsolved goals"}]},
            {"sorries": [{"goal": "⊢ 2 = 2"}]},   # 加 sorry 后取 state
        ])
        repl = _make_repl(tr)
        try:
            r = repl.check_tactic_sequence(
                "theorem t : 2 = 2 := by sorry", ["have h := rfl"],
                preamble="import Mathlib")
            assert r.success and not r.is_complete
            assert r.goals == ["⊢ 2 = 2"]
            assert tr.sent[2]["cmd"].rstrip().endswith("sorry")
        finally:
            repl.close()

    def test_fallback_is_honest(self):
        tr = _FakeTransport([])
        tr.is_fallback = True
        repl = _make_repl(tr)
        try:
            r = repl.check_tactic_sequence("theorem t : True", ["trivial"])
            assert not r.success and "fallback" in r.error
            assert tr.sent == []        # 绝不伪造成功, 也不发请求
        finally:
            repl.close()

    def test_preamble_env_cached(self):
        tr = _FakeTransport([
            {"env": 7},
            {"sorries": [{"goal": "g"}]},
            {"sorries": [{"goal": "g"}]},
        ])
        repl = _make_repl(tr)
        try:
            repl.check_tactic_sequence("theorem a : True := by sorry", [])
            repl.check_tactic_sequence("theorem b : True := by sorry", [])
            n_preamble = sum(
                1 for c in tr.sent if c["cmd"] == "import Mathlib")
            assert n_preamble == 1      # env 缓存生效
        finally:
            repl.close()
