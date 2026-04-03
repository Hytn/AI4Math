#!/usr/bin/env python3
"""scripts/smoke_test.py — Smoke test (no Lean/API needed)"""
import sys; sys.path.insert(0, ".")
passed = failed = 0

def check(name, fn):
    global passed, failed
    try: fn(); print(f"  ✓ {name}"); passed += 1
    except Exception as e: print(f"  ✗ {name}: {e}"); failed += 1

print("\n🔍 AI4Math — Smoke Test\n")

print("[1] Module imports")
def t1():
    from prover.models import ProofTrace, ProofAttempt, AttemptStatus
    from prover.pipeline.orchestrator import Orchestrator
    from agent.brain.claude_provider import create_provider, MockProvider
    from agent.memory.working_memory import WorkingMemory
    from agent.strategy.meta_controller import MetaController
    from benchmarks.loader import load_benchmark
check("all imports", t1)

print("[2] Data models")
def t2():
    from prover.models import ProofTrace, ProofAttempt, AttemptStatus
    trace = ProofTrace(problem_id="t", theorem_statement="theorem t : True")
    a = ProofAttempt(lean_result=AttemptStatus.SUCCESS, generated_proof=":= trivial")
    trace.add_attempt(a)
    assert trace.solved
check("trace + attempt", t2)

print("[3] Mock provider")
def t3():
    from agent.brain.claude_provider import create_provider
    p = create_provider({"provider": "mock"})
    r = p.generate("sys", "user")
    assert r.model == "mock" and len(r.content) > 0
check("mock LLM", t3)

print("[4] Builtin loader")
def t4():
    from benchmarks.loader import load_benchmark
    probs = load_benchmark("builtin")
    assert len(probs) >= 3
check("builtin benchmark", t4)

print("[5] Strategy")
def t5():
    from agent.strategy.meta_controller import MetaController
    from agent.memory.working_memory import WorkingMemory
    mc = MetaController({"max_light_rounds": 2})
    mem = WorkingMemory(current_strategy="light", rounds_completed=3)
    assert mc.should_escalate(mem) == "medium"
check("strategy escalation", t5)

print("[6] Lemma bank")
def t6():
    from prover.lemma_bank.bank import LemmaBank, ProvedLemma
    bank = LemmaBank()
    bank.add(ProvedLemma("h1", "lemma h1 : True", ":= trivial", 1))
    assert bank.count == 1
    assert "Already proved" in bank.to_prompt_context()
check("lemma bank", t6)

print(f"\n{'='*50}")
print(f"  {'All ' + str(passed) + ' passed! ✓' if failed == 0 else str(passed) + ' passed, ' + str(failed) + ' failed'}")
print(f"{'='*50}\n")
sys.exit(1 if failed else 0)
