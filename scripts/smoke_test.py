#!/usr/bin/env python3
"""
scripts/smoke_test.py — 冒烟测试

不需要 Lean 环境或 LLM API key，只验证代码结构和 mock 流程能跑通。

用法: python scripts/smoke_test.py
"""

import sys
import json
import tempfile
from pathlib import Path

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed += 1


print("\n🔍 AI4Math Demo — Smoke Test\n")

# ── 1. Imports ─────────────────────────────────────────────────
print("[1/6] Module imports")


def test_imports():
    from core.models import ProofTrace, ProofAttempt, AttemptStatus
    from core.lean_checker import LeanChecker, parse_lean_errors
    from core.llm_policy import create_provider, build_prompt, extract_lean_code
    from core.error_analyzer import analyze_errors, summarize_error_history
    from core.retriever import PremiseRetriever
    from core.orchestrator import Orchestrator, OrchestratorConfig
    from benchmarks.loader import load_builtin_examples, load_benchmark
    from benchmarks.eval_runner import EvalRunner


check("all modules import", test_imports)

# ── 2. Data Models ─────────────────────────────────────────────
print("[2/6] Data models")

from core.models import *


def test_trace_roundtrip():
    trace = ProofTrace(problem_id="t", problem_name="test", theorem_statement="theorem t : True")
    a = ProofAttempt(attempt_number=1, lean_result=AttemptStatus.SUCCESS, generated_proof=":= trivial")
    trace.add_attempt(a)
    assert trace.solved
    d = trace.to_dict()
    assert d["solved"] == True
    assert d["total_attempts"] == 1
    # JSON roundtrip
    s = json.dumps(d)
    loaded = json.loads(s)
    assert loaded["solved"] == True


check("ProofTrace create + serialize", test_trace_roundtrip)


def test_trace_file_io():
    trace = ProofTrace(problem_id="io_test", problem_name="io")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.json"
        trace.save(p)
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["problem_id"] == "io_test"


check("ProofTrace save to file", test_trace_file_io)

# ── 3. Error Parsing ──────────────────────────────────────────
print("[3/6] Lean error parsing")

from core.lean_checker import parse_lean_errors


def test_error_parsing():
    stderr = "AI4MathCheck.lean:5:2: error: unknown identifier 'foo'\nAI4MathCheck.lean:8:4: error: tactic simp failed"
    errors = parse_lean_errors(stderr)
    assert len(errors) == 2
    assert errors[0].category == ErrorCategory.UNKNOWN_IDENTIFIER
    assert errors[1].category == ErrorCategory.TACTIC_FAILED
    assert errors[0].line == 5


check("parse Lean stderr", test_error_parsing)

from core.error_analyzer import analyze_errors, summarize_error_history


def test_error_analysis():
    err = LeanError(category=ErrorCategory.TYPE_MISMATCH, message="type mismatch")
    text = analyze_errors([err])
    assert "mismatch" in text.lower()
    history = [(":= by simp", [err])]
    summary = summarize_error_history(history)
    assert "Attempt" in summary


check("error analysis + history", test_error_analysis)

# ── 4. LLM Policy ─────────────────────────────────────────────
print("[4/6] LLM policy")

from core.llm_policy import create_provider, build_prompt, extract_lean_code


def test_mock_provider():
    p = create_provider({"provider": "mock"})
    r = p.generate("sys", "user")
    assert r.model == "mock"
    assert len(r.content) > 0


check("mock LLM provider", test_mock_provider)


def test_prompt_building():
    prompt = build_prompt("theorem foo : True", premises=["Nat.add_comm"])
    assert "foo" in prompt
    assert "Nat.add_comm" in prompt
    # retry prompt
    prompt2 = build_prompt("theorem foo : True", error_analysis="some error")
    assert "error" in prompt2.lower()


check("prompt building", test_prompt_building)


def test_code_extraction():
    assert "trivial" in extract_lean_code("```lean\n:= by trivial\n```")
    assert "ring" in extract_lean_code("```\n:= by ring\n```")
    assert "omega" in extract_lean_code(":= by omega")


check("code extraction", test_code_extraction)

# ── 5. Benchmark Loader ───────────────────────────────────────
print("[5/6] Benchmark loader")

from benchmarks.loader import load_builtin_examples, load_benchmark


def test_builtin_loader():
    problems = load_builtin_examples()
    assert len(problems) >= 3
    for p in problems:
        assert p.theorem_statement.startswith("theorem")
        assert p.problem_id


check("builtin examples", test_builtin_loader)


def test_load_benchmark_dispatch():
    problems = load_benchmark("builtin")
    assert len(problems) >= 3


check("load_benchmark dispatch", test_load_benchmark_dispatch)

# ── 6. Orchestrator (mock) ────────────────────────────────────
print("[6/6] Orchestrator loop")

from core.lean_checker import LeanChecker
from core.retriever import PremiseRetriever
from core.orchestrator import Orchestrator, OrchestratorConfig


class _FakeLean(LeanChecker):
    def __init__(self):
        self.n = 0

    def check(self, theorem_statement, proof, extra_imports=None):
        self.n += 1
        if self.n >= 2:
            return AttemptStatus.SUCCESS, [], "", "", 50
        err = LeanError(category=ErrorCategory.TACTIC_FAILED, message="simp failed")
        return AttemptStatus.LEAN_ERROR, [err], "", "error: simp failed", 50


class _FakeLLM:
    model_name = "fake"

    def generate(self, system, user, temperature=0.7):
        from core.llm_policy import LLMResponse
        return LLMResponse(content="```lean\n:= by ring\n```", model="fake", tokens_in=10, tokens_out=5, latency_ms=10)


def test_orchestrator():
    # Test sequential strategy
    orc = Orchestrator(
        lean_checker=_FakeLean(),
        llm_provider=_FakeLLM(),
        retriever=PremiseRetriever(),
        config=OrchestratorConfig(strategy="sequential", max_attempts=5),
    )
    problems = load_builtin_examples()
    trace = orc.prove(problems[0])
    assert trace.solved
    assert trace.total_attempts == 2

    # Test rollout strategy
    from core.rollout import RolloutConfig
    orc2 = Orchestrator(
        lean_checker=_FakeLean(),
        llm_provider=_FakeLLM(),
        retriever=PremiseRetriever(),
        config=OrchestratorConfig(strategy="rollout", samples_per_round=4, max_rounds=2, max_workers=1),
    )
    trace2 = orc2.prove(problems[0])
    assert trace2.total_attempts > 0


check("orchestrator (sequential + rollout)", test_orchestrator)

# ── Summary ────────────────────────────────────────────────────
print(f"\n{'='*50}")
if failed == 0:
    print(f"  All {passed} tests passed! ✓")
    print(f"  Your code skeleton is ready.")
else:
    print(f"  {passed} passed, {failed} failed")
print(f"{'='*50}\n")

sys.exit(1 if failed else 0)
