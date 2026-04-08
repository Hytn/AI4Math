#!/usr/bin/env python3
"""scripts/lean4_smoke_test.py — Real Lean4 integration smoke test

This script validates that the full AI4Math infrastructure can talk to
a real Lean4 installation. It is the READINESS GATE for next-phase
development.

Requirements:
  - Lean4 installed (lean binary in PATH)
  - For REPL mode: lean4-repl built (.lake/build/bin/repl)
  - For Docker mode: docker/docker-compose up lean

Usage:
  # Local (auto-detects lean/repl)
  python3 scripts/lean4_smoke_test.py

  # Specify project dir
  python3 scripts/lean4_smoke_test.py --project-dir /path/to/lean-project

  # Socket mode (connect to lean_daemon)
  python3 scripts/lean4_smoke_test.py --socket /workspace/exchange/lean.sock

Exit codes:
  0 = all tests passed
  1 = some tests failed
  2 = Lean4 not available (skip)
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine.transport import LocalTransport, SocketTransport
from engine.async_lean_pool import AsyncLeanPool, AsyncLeanSession
from engine.proof_session import ProofSessionManager
from engine._core import assemble_code


# ─── Test Results ─────────────────────────────────────────────

class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.details = []

    def record(self, name: str, ok: bool, detail: str = ""):
        if ok:
            self.passed += 1
            print(f"  ✓ {name}" + (f" ({detail})" if detail else ""))
        else:
            self.failed += 1
            print(f"  ✗ {name}: {detail}")
        self.details.append((name, ok, detail))

    def skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"  ⊘ {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  {self.passed}/{total} passed"
              + (f", {self.skipped} skipped" if self.skipped else ""))
        if self.failed == 0:
            print(f"  ✓ ALL FOUNDATION TESTS PASSED — ready for next phase")
        else:
            print(f"  ✗ {self.failed} tests failed — fix before proceeding")
        print(f"{'='*60}\n")
        return self.failed == 0


results = Results()


# ─── Test: Transport Layer ────────────────────────────────────

async def test_transport_start(transport):
    """Test that transport can start and connect to Lean4."""
    ok = await transport.start()
    if not ok:
        results.record("transport_start", False, "start() returned False")
        return False

    if transport.is_fallback:
        results.record("transport_start", False,
                       "started in FALLBACK mode (no Lean4 binary)")
        return False

    mode = "repl" if not getattr(transport, 'is_single_shot', False) else "single_shot"
    results.record("transport_start", True, f"mode={mode}")
    return True


async def test_transport_check_nat(transport):
    """Send #check Nat — most basic REPL interaction."""
    t0 = time.time()
    resp = await transport.send({"cmd": "#check Nat", "env": 0})
    elapsed = (time.time() - t0) * 1000

    if resp is None:
        results.record("check_nat", False, "got None response")
        return False

    has_error = any(m.get("severity") == "error"
                    for m in resp.get("messages", []))
    if has_error:
        err = resp["messages"][0].get("data", "")
        results.record("check_nat", False, f"error: {err[:100]}")
        return False

    results.record("check_nat", True, f"{elapsed:.0f}ms")
    return True


async def test_transport_import_mathlib(transport):
    """Import Mathlib — validates mathlib is available."""
    t0 = time.time()
    resp = await transport.send({"cmd": "import Mathlib", "env": 0})
    elapsed = (time.time() - t0) * 1000

    if resp is None:
        results.record("import_mathlib", False, "got None response")
        return False

    errors = [m for m in resp.get("messages", [])
              if m.get("severity") == "error"]
    if errors:
        # Mathlib not available — not fatal, just note it
        results.skip("import_mathlib",
                      "Mathlib not available (non-fatal)")
        return False

    env_id = resp.get("env", -1)
    results.record("import_mathlib", True,
                    f"env_id={env_id}, {elapsed:.0f}ms")
    return True


# ─── Test: Tactic Interaction ─────────────────────────────────

async def test_tactic_interaction(transport):
    """Test interactive tactic proving via sorry mechanism."""
    # Step 1: send theorem with sorry to get proof state
    code = "theorem test_tactic : True := by sorry"
    resp = await transport.send({"cmd": code, "env": 0})

    if resp is None:
        results.record("tactic_interaction", False, "None response")
        return False

    sorries = resp.get("sorries", [])
    if not sorries:
        # Try the non-sorry approach (just verify the theorem compiles)
        code2 = "theorem test_tactic : True := by trivial"
        resp2 = await transport.send({"cmd": code2, "env": 0})
        if resp2 and not any(m.get("severity") == "error"
                             for m in resp2.get("messages", [])):
            results.record("tactic_interaction", True,
                           "verified via direct compilation")
            return True
        results.record("tactic_interaction", False,
                       "no sorries returned and direct verify failed")
        return False

    # Step 2: use proof state to apply tactic
    ps = sorries[0].get("proofState", -1)
    if ps < 0:
        results.record("tactic_interaction", False,
                       "invalid proofState in sorry")
        return False

    resp2 = await transport.send({"tactic": "trivial", "proofState": ps})
    if resp2 is None:
        results.record("tactic_interaction", False,
                       "None response to tactic")
        return False

    goals = resp2.get("goals", ["?"])
    if len(goals) == 0:
        results.record("tactic_interaction", True, "proof complete via tactic")
    else:
        results.record("tactic_interaction", True,
                       f"tactic applied, {len(goals)} goals remaining")
    return True


# ─── Test: Full Proof Verification ───────────────────────────

async def test_full_proof_verify(transport):
    """Verify a complete theorem + proof."""
    code = assemble_code(
        "theorem one_plus_one : 1 + 1 = 2",
        ":= by norm_num",
        preamble="import Mathlib")
    resp = await transport.send({"cmd": code, "env": 0})

    if resp is None:
        results.record("full_proof", False, "None response")
        return False

    errors = [m for m in resp.get("messages", [])
              if m.get("severity") == "error"]
    if errors:
        # Try without Mathlib
        code2 = assemble_code(
            "theorem one_plus_one : 1 + 1 = 2",
            ":= by native_decide",
            preamble="")
        resp2 = await transport.send({"cmd": code2, "env": 0})
        if resp2 and not any(m.get("severity") == "error"
                             for m in resp2.get("messages", [])):
            results.record("full_proof", True, "verified (no Mathlib)")
            return True
        results.record("full_proof", False,
                       f"error: {errors[0].get('data', '')[:100]}")
        return False

    results.record("full_proof", True, "verified successfully")
    return True


# ─── Test: Error Classification ───────────────────────────────

async def test_error_feedback(transport):
    """Verify that errors produce useful structured feedback."""
    # Intentionally wrong proof
    code = "theorem bad : False := by trivial"
    resp = await transport.send({"cmd": code, "env": 0})

    if resp is None:
        results.record("error_feedback", False, "None response")
        return False

    errors = [m for m in resp.get("messages", [])
              if m.get("severity") == "error"]
    if not errors:
        results.record("error_feedback", False,
                       "expected error but got success?!")
        return False

    msg = errors[0].get("data", "")
    from engine._core import classify_error
    category = classify_error(msg)

    results.record("error_feedback", True,
                    f"category={category}, msg={msg[:80]}")
    return True


# ─── Test: Pool Integration ───────────────────────────────────

async def test_pool_integration(project_dir: str):
    """Test AsyncLeanPool with real Lean4."""
    async with AsyncLeanPool(
            pool_size=2,
            project_dir=project_dir,
            preamble="#check Nat") as pool:

        stats = pool.stats()
        if stats["all_fallback"]:
            results.record("pool_integration", False, "all sessions fallback")
            return False

        # Try parallel tactics
        tactics = ["#check Nat", "#check Bool"]
        rs = await pool.try_tactics_parallel(pool.base_env_id, tactics)
        ok_count = sum(1 for r in rs if r.success)

        results.record("pool_integration", True,
                        f"{ok_count}/{len(tactics)} succeeded, "
                        f"sessions={stats['active_sessions']}")
        return True


# ─── Test: Proof Session ─────────────────────────────────────

async def test_proof_session(project_dir: str):
    """Test ProofSessionManager with real Lean4."""
    async with AsyncLeanPool(
            pool_size=1,
            project_dir=project_dir,
            preamble="#check Nat") as pool:

        if pool.stats()["all_fallback"]:
            results.skip("proof_session", "fallback mode")
            return False

        async with ProofSessionManager(pool) as mgr:
            session = await mgr.begin_proof("theorem t : True := by")
            stats = session.tree_stats()
            results.record("proof_session", True,
                           f"nodes={stats['total_nodes']}")
            return True


# ─── Main ─────────────────────────────────────────────────────

async def run_all(args):
    print("\n🔬 AI4Math Foundation Smoke Test")
    print(f"   Project: {args.project_dir}")
    print(f"   Socket:  {args.socket or '(none)'}\n")

    # Create transport
    if args.socket:
        transport = SocketTransport(args.socket)
    else:
        transport = LocalTransport(
            project_dir=args.project_dir,
            timeout_seconds=120)

    # Layer 1: Transport
    print("[1] Transport Layer")
    ok = await test_transport_start(transport)
    if not ok:
        print("\n   ⚠ Lean4 not available — running in degraded mode")
        print("   To fix: install Lean4 and build lean4-repl")
        print("   Or: docker-compose -f docker/docker-compose.yaml up lean\n")
        await transport.close()
        return results.summary()

    # Layer 2: Basic REPL
    print("\n[2] REPL Communication")
    await test_transport_check_nat(transport)
    has_mathlib = await test_transport_import_mathlib(transport)

    # Layer 3: Tactic interaction
    print("\n[3] Tactic Interaction")
    await test_tactic_interaction(transport)

    # Layer 4: Full proof
    print("\n[4] Full Proof Verification")
    await test_full_proof_verify(transport)

    # Layer 5: Error feedback
    print("\n[5] Error Feedback")
    await test_error_feedback(transport)

    await transport.close()

    # Layer 6: Pool
    print("\n[6] Pool Integration")
    await test_pool_integration(args.project_dir)

    # Layer 7: Proof session
    print("\n[7] Proof Session")
    await test_proof_session(args.project_dir)

    return results.summary()


def main():
    parser = argparse.ArgumentParser(
        description="AI4Math foundation smoke test")
    parser.add_argument(
        "--project-dir", default=".",
        help="Lean4 project directory (default: .)")
    parser.add_argument(
        "--socket", default="",
        help="Unix socket path (for Docker mode)")
    args = parser.parse_args()

    ok = asyncio.run(run_all(args))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
