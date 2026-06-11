"""tests/test_hilbert.py — Hilbert 递归调度器测试 (无网络 / 无 Lean)

用脚本化假 LLM + 关键词假 verifier 覆盖四条通路与纯函数。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from prover.hilbert import (
    HilbertConfig, HilbertOrchestrator, assemble, extract_proof_body,
    parse_decomposition, statement_head)
from prover.hilbert.orchestrator import ProofNode


# ─── 假角色 LLM ─────────────────────────────────────────────────────

class FakeLLM:
    """按序弹出脚本响应; 用尽后回退到 default。"""

    def __init__(self, scripted=None, default=""):
        self.scripted = list(scripted or [])
        self.default = default
        self.calls = []

    async def generate(self, system="", user="", temperature=0.7,
                       tools=None, max_tokens=4096):
        self.calls.append({"system": system[:40], "user": user})
        content = self.scripted.pop(0) if self.scripted else self.default

        class R:
            pass
        r = R()
        r.content = content
        r.tokens_in, r.tokens_out = 10, 20
        return r


def make_verify(accept_substrings):
    """verifier: 代码含任一关键子串即判过。"""
    seen = []

    async def verify(code: str):
        seen.append(code)
        ok = any(s in code for s in accept_substrings)
        return ok, ([] if ok else ["unsolved goals"])
    verify.seen = seen
    return verify


def run(coro):
    return asyncio.run(coro)


def small_cfg(**kw) -> HilbertConfig:
    cfg = HilbertConfig()
    cfg.prover_passes = kw.pop("prover_passes", 1)
    cfg.shallow_passes = kw.pop("shallow_passes", 1)
    cfg.decompose_attempts = kw.pop("decompose_attempts", 1)
    cfg.max_depth = kw.pop("max_depth", 2)
    cfg.check_subgoal_statements = kw.pop("check_subgoal_statements", False)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


STMT = "theorem main (n : ℕ) : n + 0 = n := by sorry"


# ─── 纯函数 ─────────────────────────────────────────────────────────

class TestPureHelpers:
    def test_statement_head(self):
        assert statement_head(STMT).endswith("n + 0 = n")
        assert statement_head("lemma l : True := sorry") == "lemma l : True"
        assert statement_head("lemma l : True") == "lemma l : True"

    def test_extract_proof_body_fenced(self):
        assert extract_proof_body("```lean\nby simp\n```") == "by simp"

    def test_extract_proof_body_full_decl(self):
        out = extract_proof_body(
            "theorem main (n : ℕ) : n + 0 = n := by\n  simp")
        assert out.startswith("by")
        assert "simp" in out

    def test_extract_proof_body_bare_term(self):
        assert extract_proof_body("Nat.add_zero n") == \
            "by exact (Nat.add_zero n)"

    def test_parse_decomposition_roundtrip(self):
        raw = json.dumps({
            "subgoals": [{"name": "aux1",
                          "statement": "lemma aux1 : 1 + 1 = 2 := by sorry"},
                         {"name": "aux2", "statement": "2 + 2 = 4"}],
            "assembly": "by exact aux1 ▸ rfl",
        })
        plan = parse_decomposition(raw, max_subgoals=6)
        assert len(plan["subgoals"]) == 2
        # 末尾 sorry 被剥掉, 裸命题被包成 lemma
        assert plan["subgoals"][0]["statement"] == "lemma aux1 : 1 + 1 = 2"
        assert plan["subgoals"][1]["statement"].startswith("lemma aux2 :")
        assert plan["assembly"].startswith("by")

    def test_parse_decomposition_garbage(self):
        assert parse_decomposition("not json", 6) is None
        assert parse_decomposition('{"subgoals": [], "assembly": "by x"}',
                                   6) is None

    def test_assemble(self):
        root = ProofNode(statement="theorem main : True")
        c = ProofNode(statement="lemma aux1 : 1 = 1")
        c.status, c.proof = "proved", "by rfl"
        root.children = [c]
        root.assembly = "by trivial"
        code = assemble(root)
        assert "lemma aux1 : 1 = 1 := by rfl" in code
        assert code.strip().endswith("theorem main : True := by trivial")


# ─── 调度通路 ───────────────────────────────────────────────────────

class TestOrchestratorPaths:
    def test_prover_direct_success(self):
        prover = FakeLLM(scripted=["by simp"])
        reasoner = FakeLLM(default="NO_SHORT_PROOF")
        verify = make_verify(["by simp"])
        orch = HilbertOrchestrator(small_cfg(), reasoner, prover, verify)
        root = run(orch.prove(STMT))
        assert root.status == "proved" and root.method == "prover"
        assert root.proof == "by simp"
        assert orch.stats["prover"].calls == 1
        assert orch.stats["reasoner"].calls == 0     # 直接成功不打 reasoner

    def test_shallow_solve_after_prover_fails(self):
        prover = FakeLLM(default="by bad_tactic")
        reasoner = FakeLLM(scripted=["by omega"])
        verify = make_verify(["by omega"])
        orch = HilbertOrchestrator(small_cfg(), reasoner, prover, verify)
        root = run(orch.prove(STMT))
        assert root.status == "proved" and root.method == "shallow"
        assert root.failures        # prover 失败被记录

    def test_decompose_recurse_and_assemble(self):
        plan = json.dumps({
            "subgoals": [
                {"name": "aux1", "statement": "lemma aux1 : 1 + 1 = 2"}],
            "assembly": "by exact aux1",
        })
        # reasoner: 根节点 shallow 失败 → 分解 plan → 子节点 shallow 成功
        reasoner = FakeLLM(scripted=["NO_SHORT_PROOF", plan, "by norm_num"])
        prover = FakeLLM(default="by bad")
        # 子目标证明 + 组装代码均判过
        verify = make_verify(["by norm_num", "exact aux1"])
        orch = HilbertOrchestrator(small_cfg(), reasoner, prover, verify)
        root = run(orch.prove(STMT))
        assert root.status == "proved" and root.method == "decompose"
        assert len(root.children) == 1
        assert root.children[0].status == "proved"
        assert "lemma aux1 : 1 + 1 = 2 := by norm_num" in root.assembled_code
        # 组装代码必须整体过 verifier (最后一次 verify 是整块代码)
        assert "theorem main" in verify.seen[-1]

    def test_depth_limit_blocks_decompose(self):
        plan = json.dumps({
            "subgoals": [{"name": "aux1", "statement": "lemma a : True"}],
            "assembly": "by trivial"})
        reasoner = FakeLLM(default=plan)
        prover = FakeLLM(default="by bad")
        verify = make_verify(["NEVER"])
        cfg = small_cfg(max_depth=0)      # 深度 0: 禁止分解
        orch = HilbertOrchestrator(cfg, reasoner, prover, verify)
        root = run(orch.prove(STMT))
        assert root.status == "failed"
        assert root.children == []

    def test_node_budget(self):
        # 无限分解的 reasoner + 永不通过的 verifier → budget 必须兜底
        plan = json.dumps({
            "subgoals": [{"name": "aux1", "statement": "lemma a : True"},
                         {"name": "aux2", "statement": "lemma b : True"}],
            "assembly": "by trivial"})
        reasoner = FakeLLM(default=plan)
        prover = FakeLLM(default="by bad")
        verify = make_verify([])
        cfg = small_cfg(max_depth=5, node_budget=10, decompose_attempts=1)
        orch = HilbertOrchestrator(cfg, reasoner, prover, verify)
        root = run(orch.prove(STMT))
        assert root.status == "failed"
        # 越界量受单层 max_subgoals 限制 (budget 检查在节点入口)
        assert orch._nodes_used <= cfg.node_budget + cfg.max_subgoals

    def test_subgoal_statement_check_feedback(self):
        bad_plan = json.dumps({
            "subgoals": [{"name": "aux1",
                          "statement": "lemma aux1 : BROKEN"}],
            "assembly": "by exact aux1"})
        good_plan = json.dumps({
            "subgoals": [{"name": "aux1",
                          "statement": "lemma aux1 : 1 = 1"}],
            "assembly": "by exact aux1"})
        reasoner = FakeLLM(
            scripted=["NO_SHORT_PROOF", bad_plan, good_plan, "by rfl"])
        prover = FakeLLM(default="by bad")
        # `:= by sorry` 体检: 只有 "1 = 1" 的声明可编译
        verify = make_verify(["1 = 1", "exact aux1"])
        cfg = small_cfg(check_subgoal_statements=True, decompose_attempts=2)
        orch = HilbertOrchestrator(cfg, reasoner, prover, verify)
        root = run(orch.prove(STMT))
        assert root.status == "proved"
        # 第二次分解的反馈 prompt 里带了体检报错
        assert any("failed to compile standalone" in c["user"] or
                   "fix them" in c["user"] for c in reasoner.calls)

    def test_banned_tokens_never_verified(self):
        prover = FakeLLM(scripted=["by sorry", "by native_decide"])
        reasoner = FakeLLM(default="NO_SHORT_PROOF")
        verify = make_verify(["sorry", "native_decide"])   # 即使判过也不该被调用
        cfg = small_cfg(prover_passes=2, max_depth=0)
        orch = HilbertOrchestrator(cfg, reasoner, prover, verify)
        root = run(orch.prove(STMT))
        assert root.status == "failed"
        assert all("sorry" not in c and "native_decide" not in c
                   for c in verify.seen)

    def test_trace_serializable(self):
        prover = FakeLLM(scripted=["by simp"])
        reasoner = FakeLLM()
        orch = HilbertOrchestrator(small_cfg(), reasoner, prover,
                                   make_verify(["by simp"]))
        root = run(orch.prove(STMT))
        d = root.to_dict()
        json.dumps(d)            # 必须可序列化进 hilbert_trace.json
        assert d["status"] == "proved"


# ─── 配置 ───────────────────────────────────────────────────────────

class TestHilbertConfig:
    def test_from_dict_roles_and_knobs(self):
        cfg = HilbertConfig.from_dict({
            "reasoner": {"provider": "openai", "model": "gpt-x",
                         "temperature": 0.3},
            "prover": {"provider": "vllm", "api_base": "http://h:8000/v1"},
            "max_depth": 3, "prover_passes": 8,
        })
        assert cfg.reasoner.provider == "openai"
        assert cfg.reasoner.temperature == 0.3
        assert cfg.prover.api_base == "http://h:8000/v1"
        assert cfg.max_depth == 3 and cfg.prover_passes == 8
        # provider_config 与 create_async_provider 的 dict 契约一致
        pc = cfg.prover.provider_config()
        assert pc["provider"] == "vllm" and "api_base" in pc

    def test_from_yaml(self, tmp_path):
        y = tmp_path / "h.yaml"
        y.write_text("reasoner:\n  provider: mock\nmax_depth: 1\n",
                     encoding="utf-8")
        cfg = HilbertConfig.from_yaml(str(y))
        assert cfg.reasoner.provider == "mock" and cfg.max_depth == 1
