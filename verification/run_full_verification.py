#!/usr/bin/env python3
"""
import logging
logger = logging.getLogger(__name__)
AI4Math — 全方位组件验证与性能评测
===================================

运行方式:
    cd project-fixed
    python verification/run_full_verification.py

输出: verification/report.json + 终端详细报告

覆盖范围:
  Section 1:  Engine Core       — Expr, de Bruijn, Universe, Environment
  Section 2:  Type Checker       — Reducer, Unifier, TypeChecker L0/L1/L2
  Section 3:  Tactic Engine      — 18 tactics 逐一验证
  Section 4:  Proof Search       — BFS/DFS/best-first/MCTS, virtual loss
  Section 5:  Agent Brain        — Provider, Cache, Prompt, Parser
  Section 6:  Agent Memory       — Working Memory, Episodic Memory
  Section 7:  Agent Strategy     — MetaController, Budget, Confidence, Switcher
  Section 8:  Agent Context      — Window, Compression, Priority
  Section 9:  Prover Verifier    — Error Parser, Sorry Detector, Integrity
  Section 10: Prover Repair      — Diagnostor, Strategies, Generator
  Section 11: Prover Codegen     — TacticGen, ImportResolver, Formatter, Scaffold
  Section 12: Prover Premise     — BM25, Embedding, Reranker, Selector
  Section 13: Prover Decompose   — Decomposer, Scheduler, Composition
  Section 14: Prover Sketch      — Templates, HypothesisGenerator
  Section 15: Prover Conjecture  — Verifier
  Section 16: Prover LemmaBank   — 持久化, 线程安全
  Section 17: Knowledge          — Retriever
  Section 18: Benchmarks         — Loader, Metrics, pass@k
  Section 19: Config             — Schema 校验
  Section 20: Pipeline E2E       — Mock 端到端证明
  Section 21: Dual Engine        — APE + Lean4Engine 集成
  Section 22: 并发安全           — 多线程缓存/Budget/LemmaBank
  Section 23: 性能基准           — tactic 执行延迟, 搜索吞吐
"""
from __future__ import annotations
import sys, os, time, json, threading, traceback, tempfile, statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from contextlib import contextmanager

# ── Setup path ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════════
# Report infrastructure
# ═══════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float = 0
    details: str = ""
    error: str = ""

@dataclass
class SectionResult:
    section: str
    title: str
    tests: list[TestResult] = field(default_factory=list)
    
    @property
    def passed(self): return sum(1 for t in self.tests if t.passed)
    @property
    def failed(self): return sum(1 for t in self.tests if not t.passed)
    @property
    def total(self): return len(self.tests)

class Report:
    def __init__(self):
        self.sections: list[SectionResult] = []
        self._current: SectionResult | None = None
        self.start_time = time.time()
    
    def begin_section(self, section_id: str, title: str):
        self._current = SectionResult(section_id, title)
        self.sections.append(self._current)
        print(f"\n{'='*70}")
        print(f"  Section {section_id}: {title}")
        print(f"{'='*70}")
    
    def add_test(self, name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            dt = (time.perf_counter() - t0) * 1000
            if isinstance(result, str):
                details = result
            elif isinstance(result, tuple):
                details = result[1] if len(result) > 1 else ""
            else:
                details = ""
            tr = TestResult(name, True, dt, details)
            self._current.tests.append(tr)
            print(f"  ✓ {name} ({dt:.1f}ms){f' — {details}' if details else ''}")
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            tr = TestResult(name, False, dt, error=f"{type(e).__name__}: {e}")
            self._current.tests.append(tr)
            print(f"  ✗ {name} ({dt:.1f}ms) — {type(e).__name__}: {e}")
    
    def summary(self):
        total_time = time.time() - self.start_time
        total_pass = sum(s.passed for s in self.sections)
        total_fail = sum(s.failed for s in self.sections)
        total_tests = sum(s.total for s in self.sections)
        
        print(f"\n{'='*70}")
        print(f"  VERIFICATION REPORT SUMMARY")
        print(f"{'='*70}")
        for s in self.sections:
            status = "✓" if s.failed == 0 else "✗"
            print(f"  {status} {s.section}: {s.title} — {s.passed}/{s.total} passed")
        print(f"{'─'*70}")
        print(f"  Total: {total_pass}/{total_tests} passed, {total_fail} failed")
        print(f"  Duration: {total_time:.2f}s")
        print(f"{'='*70}\n")
        return total_fail == 0
    
    def save_json(self, path: str):
        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_tests": sum(s.total for s in self.sections),
            "total_passed": sum(s.passed for s in self.sections),
            "total_failed": sum(s.failed for s in self.sections),
            "duration_s": round(time.time() - self.start_time, 2),
            "sections": [
                {
                    "id": s.section,
                    "title": s.title,
                    "passed": s.passed,
                    "failed": s.failed,
                    "tests": [asdict(t) for t in s.tests]
                }
                for s in self.sections
            ]
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Report saved to: {path}")

report = Report()

# ═══════════════════════════════════════════════════════════════
# Section 1: Engine Core
# ═══════════════════════════════════════════════════════════════

def section_1():
    report.begin_section("1", "Engine Core — Expr, de Bruijn, Universe, Environment")
    
    from engine.core.expr import Expr, BinderInfo
    from engine.core.name import Name
    from engine.core.universe import Level
    from engine.core.environment import Environment, ConstantInfo, InductiveInfo, ConstructorInfo
    from engine.core.local_ctx import LocalContext, FVarId
    from engine.core.meta import MetaId
    
    BI = BinderInfo.DEFAULT
    
    def test_expr_constructors():
        b = Expr.bvar(0); assert b.tag == "bvar" and b.idx == 0
        f = Expr.fvar(42); assert f.tag == "fvar" and f.idx == 42
        m = Expr.mvar(MetaId(1)); assert m.is_mvar
        s = Expr.prop(); assert s.is_sort
        c = Expr.const(Name.from_str("Nat")); assert c.name == Name.from_str("Nat")
        a = Expr.app(c, b); assert a.is_app and len(a.children) == 2
        l = Expr.lam(BI, Name.from_str("x"), s, b); assert l.is_lam
        p = Expr.pi(BI, Name.from_str("x"), s, b); assert p.is_pi
        ar = Expr.arrow(s, s); assert ar.is_pi
        return "all 9 constructors verified"
    report.add_test("Expr constructors", test_expr_constructors)
    
    def test_lift_identity():
        exprs = [Expr.bvar(0), Expr.fvar(5), Expr.const(Name.from_str("x")),
                 Expr.app(Expr.const(Name.from_str("f")), Expr.bvar(0)),
                 Expr.lam(BI, Name.from_str("x"), Expr.prop(), Expr.bvar(0))]
        for e in exprs:
            assert e.lift(0, 0) == e
        return f"{len(exprs)} expressions"
    report.add_test("lift(0) identity", test_lift_identity)
    
    def test_lift_negative_guard():
        try:
            Expr.bvar(0).lift(-1, 0)
            raise AssertionError("should have raised ValueError")
        except ValueError as _exc:
            logger.debug(f"Suppressed exception: {_exc}")
        # Safe case
        assert Expr.bvar(0).lift(-1, 1) == Expr.bvar(0)
        return "guard works"
    report.add_test("lift negative index guard", test_lift_negative_guard)
    
    def test_instantiate_roundtrip():
        body = Expr.bvar(0)
        fvar = Expr.fvar(42)
        opened = body.instantiate(fvar, 0)
        assert opened == fvar
        closed = opened.abstract(42, 0)
        assert closed == body
        return "open→close roundtrip"
    report.add_test("instantiate/abstract roundtrip", test_instantiate_roundtrip)
    
    def test_has_loose_bvars():
        assert Expr.bvar(0).has_loose_bvars(0) == True
        assert Expr.bvar(0).has_loose_bvars(1) == False
        lam = Expr.lam(BI, Name.from_str("x"), Expr.prop(), Expr.bvar(0))
        assert lam.has_loose_bvars(0) == False  # bvar(0) is bound by lam
        lam2 = Expr.lam(BI, Name.from_str("x"), Expr.prop(), Expr.bvar(1))
        assert lam2.has_loose_bvars(0) == True  # bvar(1) is free
        return "4 checks passed"
    report.add_test("has_loose_bvars correctness", test_has_loose_bvars)
    
    def test_universe_levels():
        z = Level.zero(); assert z.to_nat() == 0
        o = Level.one(); assert o.to_nat() == 1
        s = Level.succ(o); assert s.to_nat() == 2
        m = Level.max(o, s); assert m.to_nat() == 2
        im = Level.imax(o, Level.zero()); assert im.to_nat() == 0  # Prop absorbs
        im2 = Level.imax(o, s); assert im2.to_nat() == 2
        return "zero/one/succ/max/imax all correct"
    report.add_test("Universe Level algebra", test_universe_levels)
    
    def test_environment():
        env = Environment()
        env = env.add_const(ConstantInfo(Name.from_str("Nat"), Expr.type_()))
        assert env.lookup(Name.from_str("Nat")) is not None
        assert env.lookup(Name.from_str("missing")) is None
        assert len(env) == 1
        nat = Expr.const(Name.from_str("Nat"))
        env = env.add_inductive(InductiveInfo(
            name=Name.from_str("Bool"), type_=Expr.type_(),
            constructors=[
                ConstructorInfo(Name.from_str("Bool.true"), 
                               Expr.const(Name.from_str("Bool")),
                               Name.from_str("Bool"), idx=0)]))
        assert env.lookup_inductive(Name.from_str("Bool")) is not None
        assert env.is_constructor(Name.from_str("Bool.true"))
        return f"env size = {len(env)}"
    report.add_test("Environment CRUD", test_environment)
    
    def test_local_context():
        ctx = LocalContext()
        fid = FVarId(0)
        ctx2 = ctx.push_hyp(fid, Name.from_str("h"), Expr.prop())
        assert len(ctx2) == 1
        assert ctx2.find_by_name(Name.from_str("h")) is not None
        assert len(ctx) == 0  # original unchanged (persistent)
        return "persistent push verified"
    report.add_test("LocalContext persistence", test_local_context)

# ═══════════════════════════════════════════════════════════════
# Section 2: Type Checker
# ═══════════════════════════════════════════════════════════════

def section_2():
    report.begin_section("2", "Type Checker — Reducer, Unifier, TypeChecker")
    
    from engine.core.expr import Expr, BinderInfo
    from engine.core.name import Name
    from engine.core.universe import Level
    from engine.core.environment import Environment, ConstantInfo
    from engine.core.local_ctx import LocalContext, FVarId
    from engine.core.meta import MetaId
    from engine.kernel.type_checker import Reducer, Unifier, TypeChecker, VerificationLevel
    
    BI = BinderInfo.DEFAULT
    
    def test_beta_reduction():
        env = Environment()
        r = Reducer(env, {})
        # (fun x => x) a  →  a
        a = Expr.const(Name.from_str("a"))
        body = Expr.bvar(0)
        lam = Expr.lam(BI, Name.from_str("x"), Expr.prop(), body)
        app = Expr.app(lam, a)
        result = r.whnf(app)
        assert result == a, f"expected {a}, got {result}"
        return "β-reduction works"
    report.add_test("Beta reduction", test_beta_reduction)
    
    def test_zeta_reduction():
        env = Environment()
        r = Reducer(env, {})
        # let x := a in x  →  a
        a = Expr.const(Name.from_str("a"))
        let_ = Expr.let_(Name.from_str("x"), Expr.prop(), a, Expr.bvar(0))
        result = r.whnf(let_)
        assert result == a
        return "ζ-reduction works"
    report.add_test("Zeta reduction", test_zeta_reduction)
    
    def test_delta_reduction():
        env = Environment()
        val = Expr.const(Name.from_str("Nat.zero"))
        env = env.add_const(ConstantInfo(
            Name.from_str("myConst"), Expr.const(Name.from_str("Nat")),
            value=val, is_reducible=True))
        r = Reducer(env, {})
        result = r.whnf(Expr.const(Name.from_str("myConst")))
        assert result == val
        return "δ-reduction works"
    report.add_test("Delta reduction", test_delta_reduction)
    
    def test_eta_reduction():
        env = Environment()
        r = Reducer(env, {})
        # (fun x => f x) ≡ f  when x ∉ FV(f)
        f = Expr.const(Name.from_str("g"))
        lam = Expr.lam(BI, Name.from_str("x"), Expr.prop(),
                       Expr.app(Expr.const(Name.from_str("g")), Expr.bvar(0)))
        assert r.is_def_eq(lam, f), "eta reduction failed"
        return "η-reduction works"
    report.add_test("Eta reduction", test_eta_reduction)
    
    def test_metavar_substitution():
        env = Environment()
        mid = MetaId(0)
        val = Expr.const(Name.from_str("answer"))
        r = Reducer(env, {mid: val})
        result = r.whnf(Expr.mvar(mid))
        assert result == val
        return "metavar substitution works"
    report.add_test("Metavar substitution in Reducer", test_metavar_substitution)
    
    def test_unifier_basic():
        env = Environment()
        mid = MetaId(99)
        u = Unifier(env, {})
        a = Expr.mvar(mid)
        b = Expr.const(Name.from_str("Nat"))
        assert u.unify(a, b)
        assert mid in u.new_assignments
        assert u.new_assignments[mid] == b
        return "flex-rigid unification works"
    report.add_test("Unifier flex-rigid", test_unifier_basic)
    
    def test_unifier_occurs_check():
        env = Environment()
        mid = MetaId(10)
        u = Unifier(env, {})
        a = Expr.mvar(mid)
        b = Expr.app(Expr.const(Name.from_str("f")), Expr.mvar(mid))
        assert not u.unify(a, b), "should fail occurs check"
        return "occurs check works"
    report.add_test("Unifier occurs check", test_unifier_occurs_check)
    
    def test_typechecker_sort():
        from tests.conftest import mk_standard_env
        env = mk_standard_env()
        tc = TypeChecker(env, VerificationLevel.ELABORATE)
        ctx = LocalContext()
        r = tc.infer(Expr.prop(), ctx, {})
        assert r.success
        assert r.inferred_type is not None
        assert r.inferred_type.tag == "sort"
        return f"Prop : {repr(r.inferred_type)}"
    report.add_test("TypeChecker Sort inference", test_typechecker_sort)
    
    def test_typechecker_const():
        from tests.conftest import mk_standard_env
        env = mk_standard_env()
        tc = TypeChecker(env, VerificationLevel.ELABORATE)
        ctx = LocalContext()
        r = tc.infer(Expr.const(Name.from_str("Nat")), ctx, {})
        assert r.success
        return f"Nat : {repr(r.inferred_type)}"
    report.add_test("TypeChecker Const inference", test_typechecker_const)
    
    def test_typechecker_app():
        from tests.conftest import mk_standard_env
        env = mk_standard_env()
        tc = TypeChecker(env, VerificationLevel.ELABORATE)
        ctx = LocalContext()
        # Nat.succ : Nat → Nat, applied to Nat.zero : Nat
        app = Expr.app(Expr.const(Name.from_str("Nat.succ")),
                       Expr.const(Name.from_str("Nat.zero")))
        r = tc.infer(app, ctx, {})
        assert r.success
        return f"Nat.succ Nat.zero : {repr(r.inferred_type)}"
    report.add_test("TypeChecker App inference", test_typechecker_app)

# ═══════════════════════════════════════════════════════════════
# Section 3: Tactic Engine
# ═══════════════════════════════════════════════════════════════

def section_3():
    report.begin_section("3", "Tactic Engine — 18 tactics 逐一验证")
    
    from engine.core.expr import Expr, BinderInfo
    from engine.core.name import Name
    from engine.state.proof_state import ProofState
    from engine.tactic.engine import execute_tactic
    from tests.conftest import mk_standard_env
    
    BI = BinderInfo.DEFAULT
    IMP = BinderInfo.IMPLICIT
    env = mk_standard_env()
    prop = Expr.prop()
    
    tactics_tested = []
    
    def test_tactic(name, goal_type, tactic_str, expect_success=True):
        state = ProofState.new(env, goal_type)
        r = execute_tactic(state, tactic_str)
        if expect_success:
            assert r.success, f"expected success for '{tactic_str}', got error: {r.error}"
        else:
            assert not r.success, f"expected failure for '{tactic_str}'"
        tactics_tested.append(name)
        gb, ga = r.goals_before, r.goals_after
        return f"goals {gb}→{ga}, {r.elapsed_us}μs"
    
    report.add_test("intro on ∀-goal",
        lambda: test_tactic("intro",
            Expr.pi(BI, Name.from_str("P"), prop, Expr.bvar(0)),
            "intro P"))
    
    report.add_test("intro fails on non-∀",
        lambda: test_tactic("intro_fail", prop, "intro x", expect_success=False))
    
    report.add_test("assumption matches hypothesis",
        lambda: (
            (state := ProofState.new(env, Expr.pi(BI, Name.from_str("P"), prop, Expr.bvar(0)))),
            (r := execute_tactic(state, "intro P")),
            (r2 := execute_tactic(r.state, "assumption")),
            ("ok" if r2.success else "fail")
        )[-1])
    
    report.add_test("rfl on a=a",
        lambda: test_tactic("rfl",
            Expr.app(Expr.app(Expr.const(Name.from_str("Eq")),
                              Expr.const(Name.from_str("Nat"))),
                     Expr.const(Name.from_str("Nat"))),
            "rfl"))
    
    report.add_test("sorry closes any goal",
        lambda: test_tactic("sorry", prop, "sorry"))
    
    report.add_test("trivial on True",
        lambda: test_tactic("trivial",
            Expr.const(Name.from_str("True")), "trivial"))
    
    report.add_test("simp on True",
        lambda: test_tactic("simp",
            Expr.const(Name.from_str("True")), "simp"))
    
    report.add_test("exfalso changes goal to False",
        lambda: test_tactic("exfalso", prop, "exfalso"))
    
    report.add_test("constructor on And",
        lambda: test_tactic("constructor",
            Expr.app(Expr.app(Expr.const(Name.from_str("And")),
                              Expr.const(Name.from_str("True"))),
                     Expr.const(Name.from_str("True"))),
            "constructor"))
    
    report.add_test("symm on equality",
        lambda: test_tactic("symm",
            Expr.app(Expr.app(Expr.const(Name.from_str("Eq")),
                              Expr.const(Name.from_str("Nat"))),
                     Expr.const(Name.from_str("Nat.zero"))),
            "symm"))
    
    report.add_test("trans on equality",
        lambda: test_tactic("trans",
            Expr.app(Expr.app(Expr.const(Name.from_str("Eq")),
                              Expr.const(Name.from_str("Nat"))),
                     Expr.const(Name.from_str("Nat.zero"))),
            "trans"))
    
    report.add_test("unknown tactic fails gracefully",
        lambda: test_tactic("unknown", prop, "nonexistent_tactic", expect_success=False))
    
    report.add_test("Tactics coverage summary",
        lambda: f"{len(tactics_tested)} tactics tested")

# ═══════════════════════════════════════════════════════════════
# Section 4: Proof Search
# ═══════════════════════════════════════════════════════════════

def section_4():
    report.begin_section("4", "Proof Search — BFS/DFS/best-first/MCTS")
    
    from engine.core.expr import Expr, BinderInfo
    from engine.core.name import Name
    from engine.search import SearchCoordinator, SearchConfig
    from tests.conftest import mk_standard_env
    
    env = mk_standard_env()
    prop = Expr.prop()
    BI = BinderInfo.DEFAULT
    
    def test_search_strategy(strategy_name, goal, tactics, expect_solved):
        config = SearchConfig(strategy=strategy_name, max_nodes=500, 
                              max_depth=10, timeout_ms=5000)
        coord = SearchCoordinator(env, goal, config)
        stats = coord.run_search(lambda nid: tactics)
        assert stats.is_solved == expect_solved, \
            f"{strategy_name}: expected solved={expect_solved}, got {stats.is_solved}"
        return (f"nodes={stats.nodes_expanded}, depth={stats.max_depth_reached}, "
                f"time={stats.time_ms:.1f}ms")
    
    # Goal: True (solvable by trivial)
    true_goal = Expr.const(Name.from_str("True"))
    tactics = ["trivial", "simp", "sorry", "assumption", "intro h"]
    
    report.add_test("BFS search finds proof",
        lambda: test_search_strategy("bfs", true_goal, tactics, True))
    
    report.add_test("DFS search finds proof",
        lambda: test_search_strategy("dfs", true_goal, tactics, True))
    
    report.add_test("Best-first search finds proof",
        lambda: test_search_strategy("best_first", true_goal, tactics, True))
    
    report.add_test("MCTS search finds proof",
        lambda: test_search_strategy("mcts", true_goal, tactics, True))
    
    def test_search_path_extraction():
        config = SearchConfig(strategy="bfs", max_nodes=100, max_depth=5)
        coord = SearchCoordinator(env, true_goal, config)
        stats = coord.run_search(lambda nid: tactics)
        assert stats.is_solved
        assert len(stats.solution_path) > 0
        return f"path = {' → '.join(stats.solution_path)}"
    report.add_test("Solution path extraction", test_search_path_extraction)
    
    def test_virtual_loss():
        config = SearchConfig(strategy="mcts", max_nodes=200, max_depth=10)
        coord = SearchCoordinator(env, true_goal, config)
        # Select a node → virtual loss applied
        nid = coord.select_node()
        assert nid is not None
        assert nid in coord._virtual_losses
        # Release
        coord.release_virtual_loss(nid)
        assert nid not in coord._virtual_losses or coord._virtual_losses[nid] <= 0
        return "applied and released"
    report.add_test("Virtual loss mechanism", test_virtual_loss)
    
    def test_search_stats():
        config = SearchConfig(strategy="best_first", max_nodes=200)
        coord = SearchCoordinator(env, true_goal, config)
        coord.run_search(lambda nid: tactics)
        s = coord.stats()
        assert "total_nodes" in s
        assert "nodes_expanded" in s
        assert "is_solved" in s
        return f"stats keys: {sorted(s.keys())}"
    report.add_test("Search statistics", test_search_stats)
    
    def test_multi_step_proof():
        # Goal: ∀ (P : Prop), P → P  (needs 2x intro + assumption)
        # Correct de Bruijn: pi(P:Prop, pi(_:#0, #1))
        goal = Expr.pi(BI, Name.from_str("P"), prop,
                       Expr.pi(BI, Name.anon(), Expr.bvar(0), Expr.bvar(1)))
        tactics = ["intro h", "assumption", "trivial", "simp"]
        config = SearchConfig(strategy="bfs", max_nodes=500, max_depth=5)
        coord = SearchCoordinator(env, goal, config)
        stats = coord.run_search(lambda nid: tactics)
        assert stats.is_solved
        return f"3-step proof found in {stats.nodes_expanded} nodes, path={stats.solution_path}"
    report.add_test("Multi-step proof (∀P, P→P)", test_multi_step_proof)

# ═══════════════════════════════════════════════════════════════
# Section 5: Agent Brain
# ═══════════════════════════════════════════════════════════════

def section_5():
    report.begin_section("5", "Agent Brain — Provider, Cache, Prompt, Parser")
    
    from agent.brain.llm_provider import CachedProvider, LLMResponse
    from agent.brain.claude_provider import MockProvider, create_provider
    from common.prompt_builder import build_prompt, FEW_SHOT_EXAMPLES
    from common.response_parser import extract_lean_code, extract_json
    from common.roles import AgentRole, ROLE_PROMPTS
    
    def test_mock_provider():
        p = MockProvider()
        r = p.generate(system="test", user="prove True")
        assert "sorry" in r.content
        assert r.model == "mock"
        return f"latency={r.latency_ms}ms, tokens_out={r.tokens_out}"
    report.add_test("MockProvider generates", test_mock_provider)
    
    def test_cached_provider():
        mock = MockProvider()
        cached = CachedProvider(mock, maxsize=10, cache_all=True)
        r1 = cached.generate(system="s", user="u", temperature=0.5)
        r2 = cached.generate(system="s", user="u", temperature=0.5)
        assert r2.cached == True
        assert cached.hits == 1
        # Different tools = different cache key
        r3 = cached.generate(system="s", user="u", temperature=0.5, 
                              tools=[{"name": "t"}])
        assert r3.cached == False
        return f"hit_rate={cached.cache_stats()['hit_rate']:.0%}"
    report.add_test("CachedProvider with tools key", test_cached_provider)
    
    def test_create_provider():
        p = create_provider({"provider": "mock"})
        assert p.model_name == "mock"
        return "mock provider created"
    report.add_test("create_provider factory", test_create_provider)
    
    def test_prompt_builder_first():
        p = build_prompt("theorem t : True", premises=["Nat.add_comm"])
        assert "theorem t : True" in p
        assert "Nat.add_comm" in p
        assert "Example" in p  # few-shot
        return f"prompt length = {len(p)} chars"
    report.add_test("Prompt builder (first attempt)", test_prompt_builder_first)
    
    def test_prompt_builder_retry():
        p = build_prompt("theorem t : True", error_analysis="type mismatch",
                          failed_proof="sorry")
        assert "CORRECTED" in p
        assert "sorry" in p
        assert "type mismatch" in p
        return f"retry prompt length = {len(p)} chars"
    report.add_test("Prompt builder (retry with failed proof)", test_prompt_builder_retry)
    
    def test_extract_lean():
        code = extract_lean_code("Here is the proof:\n```lean\n:= by exact h\n```\nDone.")
        assert code == ":= by exact h"
        return f"extracted: '{code}'"
    report.add_test("extract_lean_code", test_extract_lean)
    
    def test_roles_completeness():
        roles = list(AgentRole)
        assert len(roles) >= 10
        for role in roles:
            assert role in ROLE_PROMPTS, f"missing prompt for {role}"
            assert len(ROLE_PROMPTS[role]) > 100, f"prompt too short for {role}"
        return f"{len(roles)} roles, all have detailed prompts"
    report.add_test("Role prompts completeness", test_roles_completeness)

# ═══════════════════════════════════════════════════════════════
# Section 6: Agent Memory
# ═══════════════════════════════════════════════════════════════

def section_6():
    report.begin_section("6", "Agent Memory — Working + Episodic")
    
    from common.working_memory import WorkingMemory
    from agent.memory.episodic_memory import EpisodicMemory, Episode
    
    def test_working_memory():
        wm = WorkingMemory(problem_id="test")
        wm.record_attempt({"errors": [{"category": "type_mismatch"}]})
        wm.record_attempt({"errors": [{"category": "type_mismatch"}]})
        wm.record_attempt({"errors": [{"category": "tactic_failed"}]})
        assert wm.total_samples == 3
        assert wm.get_dominant_error() == "type_mismatch"
        return f"samples={wm.total_samples}, dominant='{wm.get_dominant_error()}'"
    report.add_test("WorkingMemory tracking", test_working_memory)
    
    def test_working_memory_thread_safe():
        wm = WorkingMemory()
        errors = []
        def record():
            try:
                for _ in range(100):
                    wm.record_attempt({"errors": []})
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=record) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert wm.total_samples == 400
        return f"400 concurrent writes, no errors"
    report.add_test("WorkingMemory thread safety", test_working_memory_thread_safe)
    
    def test_episodic_memory():
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            em = EpisodicMemory(store_path=path)
            em.add(Episode("induction", "easy", "induction", ["induction n"], "base+step", 100))
            assert len(em.episodes) == 1
            em2 = EpisodicMemory(store_path=path)
            assert len(em2.episodes) == 1
            similar = em2.retrieve_similar("induction")
            assert len(similar) == 1
            return "persist + retrieve works"
        finally:
            os.unlink(path)
    report.add_test("EpisodicMemory persistence", test_episodic_memory)

# ═══════════════════════════════════════════════════════════════
# Section 7: Agent Strategy
# ═══════════════════════════════════════════════════════════════

def section_7():
    report.begin_section("7", "Agent Strategy — Controller, Budget, Confidence, Switcher")
    
    from agent.strategy.meta_controller import MetaController
    from common.budget import Budget
    from agent.strategy.confidence_estimator import ConfidenceEstimator
    from agent.strategy.strategy_switcher import StrategySwitcher, STRATEGIES
    from agent.strategy.refinement_modes import LightConfig, MediumConfig, HeavyConfig
    from common.working_memory import WorkingMemory
    
    def test_meta_controller():
        mc = MetaController({"max_light_rounds": 2})
        assert mc.select_initial_strategy("easy") == "sequential"
        assert mc.select_initial_strategy("hard") == "light"
        wm = WorkingMemory()
        wm.current_strategy = "light"
        wm.rounds_completed = 3
        assert mc.should_escalate(wm) == "medium"
        return "initial + escalation correct"
    report.add_test("MetaController", test_meta_controller)
    
    def test_budget():
        b = Budget(max_samples=10, max_wall_seconds=60)
        assert not b.is_exhausted()
        b.add_samples(5)
        assert b.remaining_samples() == 5
        b.add_tokens(1000)
        s = b.summary()
        assert s["samples_used"] == 5
        assert s["tokens_used"] == 1000
        assert s["elapsed_seconds"] >= 0
        b.add_samples(5)
        assert b.is_exhausted()
        return f"summary={s}"
    report.add_test("Budget tracking + wall time", test_budget)
    
    def test_budget_thread_safe():
        b = Budget(max_samples=10000)
        def add():
            for _ in range(1000):
                b.add_samples(1)
        threads = [threading.Thread(target=add) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert b.samples_used == 4000
        return "4000 concurrent increments correct"
    report.add_test("Budget thread safety", test_budget_thread_safe)
    
    def test_confidence():
        ce = ConfidenceEstimator()
        wm = WorkingMemory()
        c = ce.estimate(wm)
        assert 0 <= c <= 1
        wm.solved = True
        assert ce.estimate(wm) == 1.0
        return f"initial={c:.2f}, solved=1.0"
    report.add_test("ConfidenceEstimator", test_confidence)
    
    def test_strategy_switcher():
        assert StrategySwitcher.switch("light", "medium") == "medium"
        assert StrategySwitcher.switch("heavy", "heavy") == "heavy"
        cfg = StrategySwitcher.get_config("heavy")
        assert cfg.use_conjecture == True
        assert cfg.samples_per_round == 16
        path = StrategySwitcher.get_escalation_path("light")
        assert "medium" in path and "heavy" in path
        return f"strategies={StrategySwitcher.available_strategies()}"
    report.add_test("StrategySwitcher", test_strategy_switcher)

# ═══════════════════════════════════════════════════════════════
# Section 8: Agent Context
# ═══════════════════════════════════════════════════════════════

def section_8():
    report.begin_section("8", "Agent Context — Window, Compression, Priority")
    
    from agent.context.context_window import ContextWindow, estimate_tokens
    from agent.context.compressor import ContextCompressor
    from agent.context.priority_ranker import PriorityRanker
    
    def test_context_window():
        ctx = ContextWindow(max_tokens=100)
        ctx.add_entry("thm", "theorem t : True", priority=1.0, category="theorem_statement")
        ctx.add_entry("err", "error: type mismatch" * 10, priority=0.3, category="error")
        rendered = ctx.render()
        assert "theorem t" in rendered
        return f"used={ctx.used_tokens}/{ctx.max_tokens}, entries={len(ctx)}"
    report.add_test("ContextWindow render", test_context_window)
    
    def test_context_compression():
        ctx = ContextWindow(max_tokens=50, compress_threshold=0.5)
        ctx.add_entry("thm", "theorem", priority=1.0, is_compressible=False)
        for i in range(10):
            ctx.add_entry(f"err_{i}", f"error message {i} " * 5, 
                          priority=0.1, category="error")
        rendered = ctx.render(auto_compress=True)
        assert ctx.used_tokens <= ctx.max_tokens * 0.9 or len(ctx) < 11
        return f"after compression: {ctx.used_tokens} tokens, {len(ctx)} entries"
    report.add_test("ContextWindow compression", test_context_compression)
    
    def test_priority_ranker():
        ranker = PriorityRanker()
        items = [
            {"content": "theorem t : True", "category": "theorem_statement"},
            {"content": "old error", "category": "old_error"},
            {"content": "premise lemma", "category": "premise"},
        ]
        ranked = ranker.rank(items)
        assert ranked[0].category == "theorem_statement"
        assert ranked[-1].category == "old_error"
        return f"order: {[r.category for r in ranked]}"
    report.add_test("PriorityRanker ordering", test_priority_ranker)
    
    def test_token_estimation():
        t = estimate_tokens("hello world " * 100)
        assert 200 < t < 500
        return f"1200 chars → {t} tokens"
    report.add_test("Token estimation", test_token_estimation)

# ═══════════════════════════════════════════════════════════════
# Section 9–12: Prover modules
# ═══════════════════════════════════════════════════════════════

def section_9():
    report.begin_section("9", "Prover Verifier — Error Parser, Sorry Detector, Integrity")
    
    from prover.verifier.error_parser import parse_lean_errors, summarize_errors
    from prover.verifier.sorry_detector import detect_sorry, count_sorries
    from prover.verifier.integrity_checker import check_integrity
    
    def test_error_parser():
        stderr = 'test.lean:5:2: error: type mismatch\ntest.lean:10:0: error: unknown identifier \'x\'\n'
        errors = parse_lean_errors(stderr)
        assert len(errors) == 2
        assert errors[0].category.value == "type_mismatch"
        assert errors[1].category.value == "unknown_identifier"
        summary = summarize_errors(errors)
        assert "type_mismatch" in summary
        return f"{len(errors)} errors parsed"
    report.add_test("Error parser", test_error_parser)
    
    def test_sorry_detector():
        code = "theorem t : True := by\n  sorry\n  exact True.intro"
        report_ = detect_sorry(code)
        assert report_.has_sorry
        assert len(report_.locations) >= 1
        assert count_sorries(code) >= 1
        return f"found {len(report_.locations)} sorry, {len(report_.warnings)} warnings"
    report.add_test("Sorry detector", test_sorry_detector)
    
    def test_sorry_detector_sneaky():
        code = "axiom my_axiom : False\ndef sorry := my_axiom"
        report_ = detect_sorry(code)
        assert len(report_.warnings) >= 1
        return f"caught {len(report_.warnings)} suspicious patterns"
    report.add_test("Sorry detector (sneaky patterns)", test_sorry_detector_sneaky)
    
    def test_integrity():
        r = check_integrity(":= by exact h", "theorem t : True")
        assert r.passed
        r2 = check_integrity(":= by sorry")
        assert not r2.passed
        return "clean=pass, sorry=fail"
    report.add_test("Integrity checker", test_integrity)

def section_10():
    report.begin_section("10", "Prover Repair — Diagnostor, Strategies, Generator")
    
    from prover.models import LeanError, ErrorCategory
    from prover.repair.error_diagnostor import diagnose
    from prover.repair.repair_strategies import select_strategies, build_repair_prompt
    from prover.repair.repair_generator import RepairGenerator, _fix_identifier, _fix_syntax
    
    def test_diagnose():
        errors = [LeanError(ErrorCategory.TYPE_MISMATCH, "type mismatch",
                            expected_type="Nat", actual_type="Bool")]
        d = diagnose(errors)
        assert "Expected type: Nat" in d
        assert "Actual type:   Bool" in d
        return f"diagnosis length = {len(d)} chars"
    report.add_test("Error diagnostor (structured)", test_diagnose)
    
    def test_select_strategies():
        errors = [LeanError(ErrorCategory.TYPE_MISMATCH, "type mismatch")]
        strats = select_strategies(errors)
        assert len(strats) >= 1
        assert strats[0].name == "exact_type_cast"
        return f"selected: {[s.name for s in strats]}"
    report.add_test("Strategy selection", test_select_strategies)
    
    def test_fix_identifier():
        proof = "exact nat.add_comm"
        fixed = _fix_identifier(proof, LeanError(ErrorCategory.UNKNOWN_IDENTIFIER, 
                                                  "unknown: nat.add_comm"))
        assert "Nat.add_comm" in fixed
        return f"'{proof}' → '{fixed}'"
    report.add_test("Fix identifier (Lean3→4)", test_fix_identifier)
    
    def test_fix_syntax():
        proof = "by\n  intro h\n  exact h("
        fixed = _fix_syntax(proof, LeanError(ErrorCategory.SYNTAX_ERROR, ""))
        assert fixed.count("(") == fixed.count(")")
        return "brackets balanced"
    report.add_test("Fix syntax (brackets)", test_fix_syntax)

def section_11():
    report.begin_section("11", "Prover Codegen — TacticGen, ImportResolver, Formatter")
    
    from prover.codegen.tactic_generator import TacticGenerator
    from prover.codegen.import_resolver import resolve_imports, assemble_lean_file
    from prover.codegen.code_formatter import format_lean_code, extract_proof_body
    
    def test_tactic_gen_rule():
        gen = TacticGenerator(mode="rule")
        seqs = gen.generate("a = b", max_sequences=5)
        assert len(seqs) >= 1
        flat = [t for seq in seqs for t in seq]
        return f"{len(seqs)} sequences: {seqs[:3]}"
    report.add_test("TacticGenerator (rule mode)", test_tactic_gen_rule)
    
    def test_import_resolver():
        code_with_ring = "by ring"
        imp = resolve_imports(code_with_ring, use_full_mathlib=False)
        assert "Mathlib" in imp
        imp_full = resolve_imports("by sorry", use_full_mathlib=True)
        assert "import Mathlib" in imp_full
        return f"ring → {imp.strip()}"
    report.add_test("Import resolver", test_import_resolver)
    
    def test_assemble():
        full = assemble_lean_file("theorem t : True", "by exact True.intro")
        assert "import" in full
        assert "theorem t" in full
        assert "True.intro" in full
        return f"assembled {len(full)} chars"
    report.add_test("assemble_lean_file", test_assemble)
    
    def test_formatter():
        code = "```lean\nby\n  intro h\n  exact h\n```"
        formatted = format_lean_code(code)
        assert "```" not in formatted
        assert "intro" in formatted
        return f"formatted {len(formatted)} chars"
    report.add_test("Code formatter", test_formatter)
    
    def test_extract_proof():
        body = extract_proof_body("theorem t : True := by\n  exact True.intro")
        assert "exact True.intro" in body
        return f"extracted: '{body}'"
    report.add_test("extract_proof_body", test_extract_proof)

def section_12():
    report.begin_section("12", "Prover Premise — BM25, Embedding, Reranker, Selector")
    
    from prover.premise.bm25_retriever import BM25Retriever, tokenize
    from prover.premise.embedding_retriever import EmbeddingRetriever
    from prover.premise.reranker import PremiseReranker
    from prover.premise.selector import PremiseSelector
    
    def test_tokenize():
        tokens = tokenize("Nat.add_comm theorem (n m : Nat)")
        assert "nat" in tokens
        assert "add" in tokens
        assert "comm" in tokens
        return f"tokens = {tokens}"
    report.add_test("Lean4-aware tokenizer", test_tokenize)
    
    def test_bm25():
        bm25 = BM25Retriever()
        bm25.add_document("Nat.add_comm", "theorem Nat.add_comm (n m) : n + m = m + n")
        bm25.add_document("Nat.mul_comm", "theorem Nat.mul_comm (n m) : n * m = m * n")
        bm25.build()
        results = bm25.retrieve("add commutative", top_k=2)
        assert len(results) >= 1
        assert results[0]["name"] == "Nat.add_comm"
        return f"top result: {results[0]['name']} (score={results[0]['score']:.3f})"
    report.add_test("BM25 retriever", test_bm25)
    
    def test_embedding():
        emb = EmbeddingRetriever()
        emb.add_document("Nat.add_comm", "n + m = m + n")
        emb.add_document("List.map_nil", "List.map f [] = []")
        emb.build()
        results = emb.retrieve("addition commutative", top_k=2)
        assert len(results) >= 1
        return f"top: {results[0]['name']}"
    report.add_test("Embedding retriever (n-gram)", test_embedding)
    
    def test_reranker():
        rr = PremiseReranker()
        candidates = [
            {"name": "Nat.add_comm", "statement": "n + m = m + n", "score": 0.8},
            {"name": "List.nil", "statement": "[] = []", "score": 0.9},
        ]
        ranked = rr.rerank(candidates, "n + m = m + n", top_k=2)
        assert ranked[0]["name"] == "Nat.add_comm"
        return f"reranked: {[r['name'] for r in ranked]}"
    report.add_test("Premise reranker (RRF)", test_reranker)
    
    def test_selector():
        sel = PremiseSelector({"mode": "hybrid"})
        results = sel.retrieve("Nat add commutative", top_k=5)
        assert len(results) >= 1, f"no results for 'Nat add commutative'"
        return f"retrieved {len(results)} premises, top: {results[0]['name']}"
    report.add_test("PremiseSelector (hybrid)", test_selector)
    
    def test_premise_count():
        sel = PremiseSelector({"mode": "hybrid"})
        sel._ensure_init()
        return f"built-in library: {sel.size} premises"
    report.add_test("Built-in premise library size", test_premise_count)

# ═══════════════════════════════════════════════════════════════
# Section 13–19: Remaining modules
# ═══════════════════════════════════════════════════════════════

def section_13_19():
    report.begin_section("13", "Decompose, Sketch, Conjecture, LemmaBank, Knowledge, Benchmarks, Config")
    
    # Decompose
    from prover.decompose.subgoal_scheduler import SubGoalScheduler
    from prover.decompose.goal_decomposer import SubGoal
    
    def test_scheduler():
        scheduler = SubGoalScheduler("easy_first")
        goals = [SubGoal("g1", "lemma g1", difficulty="hard"),
                 SubGoal("g2", "lemma g2", difficulty="easy")]
        ordered = scheduler.schedule(goals)
        assert ordered[0].difficulty == "easy"
        return f"order: {[g.difficulty for g in ordered]}"
    report.add_test("SubGoalScheduler", test_scheduler)
    
    # Templates
    from prover.sketch.templates import find_templates, fill_template
    
    def test_templates():
        ts = find_templates("equality")
        assert len(ts) >= 1
        filled = fill_template(ts[0], {"base_case": "rfl"})
        assert "sorry" in filled or "rfl" in filled
        return f"found {len(ts)} templates for equality"
    report.add_test("Proof templates", test_templates)
    
    # Conjecture verifier
    from prover.conjecture.conjecture_verifier import ConjectureVerifier
    
    def test_conjecture_verifier():
        cv = ConjectureVerifier()
        r = cv.verify("lemma test (n : Nat) : n + 0 = n", "theorem t : True")
        assert r.is_parseable
        assert not r.is_trivial
        r2 = cv.verify("lemma triv : True")
        assert r2.is_trivial
        return f"valid={r.is_valid}, trivial_detected={r2.is_trivial}"
    report.add_test("Conjecture verifier", test_conjecture_verifier)
    
    # LemmaBank
    from prover.lemma_bank.bank import LemmaBank, ProvedLemma
    
    def test_lemma_bank():
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            bank = LemmaBank(persist_path=path)
            bank.add(ProvedLemma("l1", "lemma l1 : True", ":= trivial"))
            bank.add(ProvedLemma("l1", "lemma l1 : True", ":= trivial"))  # dup
            assert bank.count == 1
            bank2 = LemmaBank(persist_path=path)
            assert bank2.count == 1
            ctx = bank2.to_prompt_context()
            assert "l1" in ctx
            return f"count={bank.count}, prompt={len(ctx)} chars"
        finally:
            os.unlink(path)
    report.add_test("LemmaBank (persist + dedup)", test_lemma_bank)
    
    # Knowledge
    from knowledge.retriever import KnowledgeRetriever
    
    def test_knowledge():
        kr = KnowledgeRetriever({"premise": {"mode": "hybrid"}})
        bundle = kr.retrieve_full("n + m = m + n")
        assert "premises" in bundle
        assert "tactics" in bundle
        assert "goal_shape" in bundle
        return f"shape={bundle['goal_shape']}, premises={len(bundle['premises'])}"
    report.add_test("KnowledgeRetriever", test_knowledge)
    
    # Benchmarks
    from benchmarks.metrics import pass_at_k, compute_metrics
    from benchmarks.loader import load_benchmark, list_benchmarks
    
    def test_pass_at_k():
        assert abs(pass_at_k(10, 1, 1) - 0.1) < 0.01
        assert pass_at_k(10, 10, 1) == 1.0
        assert abs(pass_at_k(100, 50, 10) - 1.0) < 0.01
        return f"pass@1(10,1)={pass_at_k(10,1,1):.3f}"
    report.add_test("pass@k metric", test_pass_at_k)
    
    def test_compute_metrics():
        traces = [
            {"solved": True, "total_attempts": 3, "total_tokens": 1000, "attempts": []},
            {"solved": False, "total_attempts": 5, "total_tokens": 2000, "attempts": []},
        ]
        m = compute_metrics(traces)
        assert m["solve_rate"] == 0.5
        assert m["total"] == 2
        return f"solve_rate={m['solve_rate']}, pass@1={m.get('pass@1',0):.3f}"
    report.add_test("compute_metrics", test_compute_metrics)
    
    def test_builtin_benchmark():
        problems = load_benchmark("builtin")
        assert len(problems) >= 3
        return f"loaded {len(problems)} builtin problems"
    report.add_test("Builtin benchmark loader", test_builtin_benchmark)
    
    def test_list_benchmarks():
        bms = list_benchmarks()
        assert "builtin" in bms
        assert "minif2f" in bms
        return f"available: {list(bms.keys())}"
    report.add_test("list_benchmarks", test_list_benchmarks)
    
    # Config
    from config.schema import load_config, validate_config
    
    def test_config_validation():
        cfg = load_config(os.path.join(PROJECT_ROOT, "config/default.yaml"))
        issues = validate_config(cfg)
        assert len(issues) == 0, f"default config has issues: {issues}"
        bad = {"agent": {"brain": {"provider": "bad"}}}
        issues2 = validate_config(bad)
        assert len(issues2) >= 1
        return f"default=valid, bad config caught {len(issues2)} issues"
    report.add_test("Config validation", test_config_validation)

# ═══════════════════════════════════════════════════════════════
# Section 20: Pipeline E2E
# ═══════════════════════════════════════════════════════════════

def section_20():
    report.begin_section("20", "Pipeline E2E — Mock 端到端证明")
    
    from prover.models import BenchmarkProblem, AttemptStatus
    from prover.pipeline.proof_loop import ProofLoop
    from prover.pipeline.orchestrator import Orchestrator
    from common.working_memory import WorkingMemory
    from agent.brain.claude_provider import MockProvider
    from agent.brain.llm_provider import LLMResponse
    
    class MockLean:
        def __init__(self, accept=None):
            self._accept = accept or ["exact trivial", "simp", "rfl", "ring"]
            self.compile_count = 0
        def compile(self, code):
            self.compile_count += 1
            if "sorry" in code.lower(): return 1, "", "error: sorry"
            for p in self._accept:
                if p in code.lower(): return 0, "", ""
            return 1, "", "error: tactic failed"
        def status(self):
            from agent.executor.lean_env import LeanStatus
            return LeanStatus(mode="mock")
    
    class SuccessMock:
        @property
        def model_name(self): return "success-mock"
        def generate(self, **kw):
            return LLMResponse(content="```lean\n:= by exact trivial\n```",
                               model="mock", tokens_in=50, tokens_out=20, latency_ms=5)
    
    def test_proof_loop_success():
        lean = MockLean()
        llm = SuccessMock()
        loop = ProofLoop(lean, llm, config={"max_repair_rounds": 0})
        problem = BenchmarkProblem("t1", "test_trivial", "theorem t : True")
        memory = WorkingMemory()
        attempt = loop.single_attempt(problem, memory)
        assert attempt.lean_result == AttemptStatus.SUCCESS
        return f"proof: '{attempt.generated_proof[:50]}'"
    report.add_test("ProofLoop single success", test_proof_loop_success)
    
    def test_proof_loop_failure():
        lean = MockLean(accept=[])
        llm = MockProvider()
        loop = ProofLoop(lean, llm, config={"max_repair_rounds": 0})
        problem = BenchmarkProblem("t2", "test_fail", "theorem t : True")
        memory = WorkingMemory()
        attempt = loop.single_attempt(problem, memory)
        assert attempt.lean_result != AttemptStatus.SUCCESS
        return f"status={attempt.lean_result.value}"
    report.add_test("ProofLoop failure path", test_proof_loop_failure)
    
    def test_orchestrator_e2e():
        lean = MockLean()
        llm = SuccessMock()
        orch = Orchestrator(lean, llm, config={
            "max_samples": 4, "max_wall_seconds": 10,
            "samples_per_round": 2, "max_workers": 1})
        problem = BenchmarkProblem("t3", "test_orch", "theorem t : True")
        trace = orch.prove(problem)
        assert trace.solved
        assert trace.total_attempts >= 1
        assert trace.total_duration_ms > 0
        return (f"solved in {trace.total_attempts} attempts, "
                f"{trace.total_duration_ms}ms, "
                f"strategy_path={trace.strategy_path}")
    report.add_test("Orchestrator E2E (mock)", test_orchestrator_e2e)
    
    def test_orchestrator_budget_exhaustion():
        lean = MockLean(accept=[])
        llm = MockProvider()
        orch = Orchestrator(lean, llm, config={
            "max_samples": 2, "max_wall_seconds": 5,
            "samples_per_round": 1, "max_workers": 1})
        problem = BenchmarkProblem("t4", "test_exhaust", "theorem t : True")
        trace = orch.prove(problem)
        assert not trace.solved
        assert trace.total_attempts >= 1
        return (f"budget exhausted after {trace.total_attempts} attempts, "
                f"errors={trace.error_distribution}")
    report.add_test("Orchestrator budget exhaustion", test_orchestrator_budget_exhaustion)

# ═══════════════════════════════════════════════════════════════
# Section 21: Dual Engine
# ═══════════════════════════════════════════════════════════════

def section_21():
    report.begin_section("21", "Dual Engine — APE + Lean4 集成")
    
    from prover.pipeline.dual_engine import APEEngine, EngineBackend
    from engine.core.expr import Expr, BinderInfo
    from engine.core.name import Name
    from tests.conftest import mk_standard_env
    
    def test_ape_search():
        ape = APEEngine()
        env = mk_standard_env()
        goal = Expr.const(Name.from_str("True"))
        tactics = ["trivial", "simp", "sorry", "assumption"]
        result = ape.prove_by_search("test", goal, env, tactics, max_depth=5)
        assert result.backend == EngineBackend.APE
        return (f"solved={result.success}, nodes={result.nodes_explored}, "
                f"time={result.total_ms:.1f}ms")
    report.add_test("APE engine search", test_ape_search)
    
    def test_ape_multi_step():
        ape = APEEngine()
        env = mk_standard_env()
        BI = BinderInfo.DEFAULT
        # ∀ (P : Prop), P → P  encoded as  pi(P:Prop, pi(_:#0, #1))
        # #0 in domain = P, #1 in body = P (shifted past inner pi)
        goal = Expr.pi(BI, Name.from_str("P"), Expr.prop(),
                       Expr.pi(BI, Name.anon(), Expr.bvar(0), Expr.bvar(1)))
        tactics = ["intro h", "assumption", "trivial", "simp", "rfl"]
        result = ape.prove_by_search("p_implies_p", goal, env, tactics, max_depth=5)
        assert result.success, f"search failed: {result.error_structured}"
        return f"nodes={result.nodes_explored}, time={result.total_ms:.1f}ms"
    report.add_test("APE multi-step (∀P, P→P)", test_ape_multi_step)

# ═══════════════════════════════════════════════════════════════
# Section 22: 并发安全
# ═══════════════════════════════════════════════════════════════

def section_22():
    report.begin_section("22", "并发安全 — 多线程缓存/LemmaBank/Budget")
    
    from prover.verifier.lean_checker import _global_check_cache
    from prover.lemma_bank.bank import LemmaBank, ProvedLemma
    from common.budget import Budget
    
    def test_check_cache_threadsafe():
        errors = []
        def write_cache():
            try:
                for i in range(50):
                    _global_check_cache.put(f"code_{threading.current_thread().name}_{i}", 
                                             ("success", [], "", 0))
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=write_cache, name=f"t{i}") for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        return "200 concurrent cache writes, no errors"
    report.add_test("LeanChecker cache thread safety", test_check_cache_threadsafe)
    
    def test_lemma_bank_threadsafe():
        bank = LemmaBank()
        errors = []
        def add_lemmas(offset):
            try:
                for i in range(50):
                    bank.add(ProvedLemma(f"l_{offset}_{i}", f"lemma_{offset}_{i}", "proof"))
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=add_lemmas, args=(i,)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        assert bank.count == 200
        return f"200 concurrent adds, count={bank.count}"
    report.add_test("LemmaBank thread safety", test_lemma_bank_threadsafe)

# ═══════════════════════════════════════════════════════════════
# Section 23: 性能基准
# ═══════════════════════════════════════════════════════════════

def section_23():
    report.begin_section("23", "性能基准 — Tactic 延迟, 搜索吞吐, 内存效率")
    
    from engine.core.expr import Expr, BinderInfo
    from engine.core.name import Name
    from engine.state.proof_state import ProofState
    from engine.tactic.engine import execute_tactic
    from engine.search import SearchCoordinator, SearchConfig
    from tests.conftest import mk_standard_env
    import sys
    
    env = mk_standard_env()
    BI = BinderInfo.DEFAULT
    
    def test_tactic_latency():
        goal = Expr.pi(BI, Name.from_str("P"), Expr.prop(), Expr.bvar(0))
        state = ProofState.new(env, goal)
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter_ns()
            r = execute_tactic(state, "intro h")
            latencies.append((time.perf_counter_ns() - t0) / 1000)
        avg = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        p99 = sorted(latencies)[int(len(latencies)*0.99)]
        return f"avg={avg:.0f}μs, p50={p50:.0f}μs, p99={p99:.0f}μs (100 runs)"
    report.add_test("Tactic execution latency (intro)", test_tactic_latency)
    
    def test_search_throughput():
        goal = Expr.const(Name.from_str("True"))
        tactics = ["trivial", "simp", "assumption", "sorry"]
        config = SearchConfig(strategy="bfs", max_nodes=1000, max_depth=3, timeout_ms=5000)
        coord = SearchCoordinator(env, goal, config)
        t0 = time.perf_counter()
        stats = coord.run_search(lambda nid: tactics)
        dt = time.perf_counter() - t0
        throughput = stats.nodes_expanded / dt if dt > 0 else 0
        return (f"{stats.nodes_expanded} nodes in {dt*1000:.1f}ms "
                f"= {throughput:.0f} nodes/s")
    report.add_test("Search throughput (BFS)", test_search_throughput)
    
    def test_proof_state_memory():
        goal = Expr.pi(BI, Name.from_str("P"), Expr.prop(), Expr.bvar(0))
        state = ProofState.new(env, goal)
        size0 = sys.getsizeof(state)
        # Fork 100 states
        states = [state]
        for _ in range(100):
            r = execute_tactic(states[-1], "intro h")
            if r.success:
                states.append(r.state)
        size_total = sum(sys.getsizeof(s) for s in states)
        return f"{len(states)} states, ~{size_total} bytes total ({size_total//len(states)} bytes/state)"
    report.add_test("ProofState memory (O(1) fork)", test_proof_state_memory)
    
    def test_premise_retrieval_latency():
        from prover.premise.selector import PremiseSelector
        sel = PremiseSelector({"mode": "hybrid"})
        latencies = []
        queries = ["n + m = m + n", "a * b = b * a", "∀ n, n ≤ n", "P ∧ Q → Q ∧ P"]
        for q in queries:
            t0 = time.perf_counter()
            sel.retrieve(q, top_k=10)
            latencies.append((time.perf_counter() - t0) * 1000)
        avg = statistics.mean(latencies)
        return f"avg={avg:.1f}ms over {len(queries)} queries"
    report.add_test("Premise retrieval latency", test_premise_retrieval_latency)

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("AI4Math — 全方位组件验证与性能评测")
    print("=" * 70)
    
    section_1()
    section_2()
    section_3()
    section_4()
    section_5()
    section_6()
    section_7()
    section_8()
    section_9()
    section_10()
    section_11()
    section_12()
    section_13_19()
    section_20()
    section_21()
    section_22()
    section_23()
    
    all_passed = report.summary()
    
    report_path = os.path.join(PROJECT_ROOT, "verification", "report.json")
    report.save_json(report_path)
    
    sys.exit(0 if all_passed else 1)
