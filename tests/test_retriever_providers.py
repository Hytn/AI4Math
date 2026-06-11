"""tests/test_retriever_providers.py — 检索 provider 层测试

覆盖:
  1. 注册表/构造 (含未知名容错、env 未配置时返回空)
  2. RetrievalCache rw/ro/off 三模式
  3. MultiRetriever 串联去重 + degraded 上报
  4. LeanSearch v2 响应解析的多 schema 容错 (无网络, 纯解析)
  5. PremiseSearchTool 注入 providers 后的优先序;
     **未注入时与历史行为一致** (回归保护)
"""
from __future__ import annotations

import asyncio
import json

import pytest

from prover.premise.providers import (
    MultiRetriever, RetrievalCache, RetrievedPremise, RetrieverProvider,
    build_providers, provider_names)
from prover.premise.providers.leansearch_v2 import LeanSearchV2Provider


# ─── 测试用 stub provider ───────────────────────────────────────────

class _StubProvider(RetrieverProvider):
    name = "stub"

    def __init__(self, hits=None, fail=False, unavailable=False):
        self._hits = hits or []
        self._fail = fail
        self._unavailable = unavailable

    def available(self):
        return not self._unavailable

    def search(self, query, top_k=10, *, goal_state=""):
        if self._fail:
            raise RuntimeError("boom")
        return self._hits[:top_k]


def _hit(name, score=1.0, source="stub"):
    return RetrievedPremise(name=name, statement=f"thm {name}",
                            score=score, source=source)


# ─── 1. 注册表 ──────────────────────────────────────────────────────

class TestRegistry:
    def test_builtin_providers_registered(self):
        names = provider_names()
        for expected in ("local_tfidf", "leansearch_v2", "loogle",
                         "leanstatesearch"):
            assert expected in names

    def test_build_unknown_name_skipped_not_raised(self):
        provs = build_providers("no_such_provider,local_tfidf",
                                cache_mode="off")
        assert [p.name for p in provs] == ["local_tfidf"]

    def test_env_unset_returns_empty(self, monkeypatch):
        monkeypatch.delenv("AI4MATH_PREMISE_PROVIDERS", raising=False)
        assert build_providers() == []

    def test_env_spec(self, monkeypatch):
        monkeypatch.setenv("AI4MATH_PREMISE_PROVIDERS",
                           "loogle, local_tfidf")
        monkeypatch.setenv("AI4MATH_RETRIEVAL_CACHE_MODE", "off")
        provs = build_providers()
        assert [p.name for p in provs] == ["loogle", "local_tfidf"]


# ─── 2. 缓存 ────────────────────────────────────────────────────────

class TestRetrievalCache:
    def test_rw_roundtrip(self, tmp_path):
        p = str(tmp_path / "c.jsonl")
        c = RetrievalCache(path=p, mode="rw")
        k = RetrievalCache.make_key("x", "q", 5)
        assert c.get(k) is None
        c.put(k, "q", [{"name": "A"}])
        assert c.get(k) == [{"name": "A"}]
        # 重新加载持久化
        c2 = RetrievalCache(path=p, mode="rw")
        assert c2.get(k) == [{"name": "A"}]

    def test_ro_never_writes(self, tmp_path):
        p = str(tmp_path / "c.jsonl")
        c = RetrievalCache(path=p, mode="ro")
        k = RetrievalCache.make_key("x", "q", 5)
        c.put(k, "q", [{"name": "A"}])
        assert c.get(k) is None
        assert not (tmp_path / "c.jsonl").exists()

    def test_off_passthrough(self, tmp_path):
        c = RetrievalCache(path=str(tmp_path / "c.jsonl"), mode="off")
        k = RetrievalCache.make_key("x", "q", 5)
        c.put(k, "q", [{"name": "A"}])
        assert c.get(k) is None

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "c.jsonl"
        good = {"k": "x\tabc", "query": "q", "results": [{"name": "A"}]}
        p.write_text("not json\n" + json.dumps(good) + "\n")
        c = RetrievalCache(path=str(p), mode="rw")
        assert c.get("x\tabc") == [{"name": "A"}]

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError):
            RetrievalCache(mode="banana")


# ─── 3. MultiRetriever ─────────────────────────────────────────────

class TestMultiRetriever:
    def test_priority_and_dedup(self):
        p1 = _StubProvider(hits=[_hit("A", source="p1"),
                                 _hit("B", source="p1")])
        p1.name = "p1"
        p2 = _StubProvider(hits=[_hit("B", source="p2"),
                                 _hit("C", source="p2")])
        p2.name = "p2"
        mr = MultiRetriever(providers=[p1, p2])
        hits, degraded = mr.search("q", top_k=10)
        assert [h.name for h in hits] == ["A", "B", "C"]
        assert hits[1].source == "p1"     # 先到先得
        assert degraded == []

    def test_failed_provider_reported_not_raised(self):
        bad = _StubProvider(fail=True)
        bad.name = "bad"
        ok = _StubProvider(hits=[_hit("A")])
        ok.name = "ok"
        mr = MultiRetriever(providers=[bad, ok])
        hits, degraded = mr.search("q", top_k=5)
        assert [h.name for h in hits] == ["A"]
        assert degraded == ["bad"]

    def test_unavailable_provider_reported(self):
        off = _StubProvider(unavailable=True)
        off.name = "off"
        mr = MultiRetriever(providers=[off])
        hits, degraded = mr.search("q")
        assert hits == [] and degraded == ["off"]

    def test_top_k_respected(self):
        p = _StubProvider(hits=[_hit(f"T{i}") for i in range(20)])
        p.name = "p"
        hits, _ = MultiRetriever(providers=[p]).search("q", top_k=3)
        assert len(hits) == 3


# ─── 4. LeanSearch v2 响应解析 ──────────────────────────────────────

class TestLeanSearchParsing:
    def _provider(self, tmp_path):
        return LeanSearchV2Provider(
            cache=RetrievalCache(path=str(tmp_path / "c.jsonl"),
                                 mode="off"))

    def test_nested_batch_shape(self, tmp_path):
        raw = [[{"result": {"formal_name": "Nat.add_comm",
                            "formal_type": "∀ n m, n+m=m+n",
                            "module_name": "Mathlib.Data.Nat",
                            "informal_description": "addition commutes"},
                 "score": 0.97}]]
        out = self._provider(tmp_path)._parse(raw)
        assert out[0].name == "Nat.add_comm"
        assert out[0].score == pytest.approx(0.97)
        assert out[0].module == "Mathlib.Data.Nat"
        assert "commutes" in out[0].informal

    def test_flat_list_shape(self, tmp_path):
        raw = [{"name": "add_comm", "statement": "a+b=b+a",
                "distance": 0.25}]
        out = self._provider(tmp_path)._parse(raw)
        assert out[0].name == "add_comm"
        assert out[0].score == pytest.approx(1.0 / 1.25)

    def test_dict_wrapped_shape(self, tmp_path):
        raw = {"results": [{"name": "mul_comm"}]}
        out = self._provider(tmp_path)._parse(raw)
        assert out[0].name == "mul_comm"
        assert out[0].score > 0          # 排名衰减分

    def test_garbage_returns_empty(self, tmp_path):
        assert self._provider(tmp_path)._parse("?!") == []
        assert self._provider(tmp_path)._parse({"x": 1}) == []
        assert self._provider(tmp_path)._parse([{"no_name": 1}]) == []

    def test_cache_replay_skips_http(self, tmp_path, monkeypatch):
        cache = RetrievalCache(path=str(tmp_path / "c.jsonl"), mode="rw")
        p = LeanSearchV2Provider(cache=cache)
        key = RetrievalCache.make_key(p.name, "comm", 0, extra=p.url)
        cache.put(key, "comm", [{"name": "Nat.add_comm",
                                 "statement": "", "score": 0.9,
                                 "module": "", "informal": "", "kind": ""}])

        def _no_http(*a, **k):
            raise AssertionError("HTTP should not be called on cache hit")
        monkeypatch.setattr(p, "_http_search", _no_http)
        hits = p.search("comm", top_k=5)
        assert hits[0].name == "Nat.add_comm"

    def test_ro_miss_degrades_without_http(self, tmp_path, monkeypatch):
        cache = RetrievalCache(path=str(tmp_path / "none.jsonl"), mode="ro")
        p = LeanSearchV2Provider(cache=cache)
        monkeypatch.setattr(
            p, "_http_search",
            lambda *a, **k: pytest.fail("must not hit network in ro"))
        assert p.search("anything") == []


# ─── 5. PremiseSearchTool 注入 ──────────────────────────────────────

class TestPremiseSearchToolInjection:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro) \
            if False else asyncio.run(coro)

    def test_default_behavior_unchanged(self):
        """不注入 providers → 走历史链路 (heuristic 兜底)。"""
        from agent.tools.builtin.premise_search import PremiseSearchTool
        tool = PremiseSearchTool()
        res = self._run(tool.execute({"query": "add_comm"}, ctx=None))
        data = json.loads(res.content)
        assert data, "heuristic fallback should fire for add_comm"
        assert all(r["source"] in ("heuristic", "tfidf", "knowledge_store")
                   for r in data)
        assert "heuristic_fallback" in res.metadata.get(
            "degraded_providers", []) or any(
            r["source"] != "heuristic" for r in data)

    def test_injected_providers_take_priority(self):
        from agent.tools.builtin.premise_search import PremiseSearchTool
        stub = _StubProvider(hits=[
            _hit("LeanSearchHit.one", score=0.99, source="leansearch_v2")])
        stub.name = "leansearch_v2"
        tool = PremiseSearchTool(providers=[stub])
        res = self._run(tool.execute(
            {"query": "commutativity of addition"}, ctx=None))
        data = json.loads(res.content)
        assert data[0]["name"] == "LeanSearchHit.one"
        assert data[0]["source"] == "leansearch_v2"

    def test_degraded_provider_surfaces_in_metadata(self):
        from agent.tools.builtin.premise_search import PremiseSearchTool
        bad = _StubProvider(fail=True)
        bad.name = "leansearch_v2"
        tool = PremiseSearchTool(providers=[bad])
        res = self._run(tool.execute({"query": "add_comm"}, ctx=None))
        assert "leansearch_v2" in res.metadata.get("degraded_providers", [])

    def test_goal_state_passed_through(self):
        from agent.tools.builtin.premise_search import PremiseSearchTool
        captured = {}

        class _Cap(_StubProvider):
            name = "cap"

            def search(self, query, top_k=10, *, goal_state=""):
                captured["gs"] = goal_state
                return [_hit("X", source="cap")]

        tool = PremiseSearchTool(providers=[_Cap()])
        self._run(tool.execute(
            {"query": "q", "goal_state": "⊢ 1 + 1 = 2"}, ctx=None))
        assert captured["gs"] == "⊢ 1 + 1 = 2"
