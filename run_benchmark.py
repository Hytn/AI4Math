#!/usr/bin/env python3
"""
run_benchmark.py — Run miniF2F benchmark through APE engine with LLM integration.

Measures: success rate, search efficiency, timing, L0/L1 filtering.
"""
import sys, os, time, json, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

from engine.core import Expr, Name, BinderInfo, MetaId
from engine.core.local_ctx import LocalContext
from engine.state.proof_state import ProofState
from engine.state.views import GoalView
from engine.lean_bridge.prelude import build_prelude_env
from engine.lean_bridge.minif2f_problems import MINIF2F_PROBLEMS, MiniF2FProblem
from engine.tactic.engine import execute_tactic, TacticResult
from engine.llm import LLMTacticEngine, LLMSuggestion
from engine.kernel.type_checker import TypeChecker, VerificationLevel, Reducer


@dataclass
class ProblemResult:
    problem_id: str
    name: str
    solved: bool
    proof_tactics: List[str]
    total_ms: float
    nodes_explored: int
    l0_filtered: int
    l1_filtered: int
    max_depth: int
    llm_calls: int
    llm_total_ms: float
    tactic_total_us: int
    sorry_used: bool = False


def run_proof_search(env, problem: MiniF2FProblem, llm: LLMTacticEngine,
                     max_depth: int = 15, max_nodes: int = 200) -> ProblemResult:
    """Run proof search on a single problem."""
    t0 = time.perf_counter()

    state = ProofState.new(env, problem.goal_expr)
    proof_tactics = []
    nodes_explored = 0
    l0_filtered = 0
    l1_filtered = 0
    llm_calls = 0
    llm_total_ms = 0.0
    tactic_total_us = 0

    current_state = state

    for step in range(max_depth):
        goal = current_state.main_goal()
        if goal is None:
            break  # proof complete or no goals

        if current_state.is_complete():
            break

        # Build goal view for LLM
        goal_view = {
            "target": repr(goal.target),
            "shape": "forall" if goal.target.is_pi else "other",
            "hypotheses": [
                {"name": str(d.user_name), "type": repr(d.type_)}
                for d in goal.local_ctx
            ],
            "depth": goal.depth,
        }

        # Get LLM suggestions
        suggestion = llm.suggest(goal_view, max_suggestions=8)
        llm_calls += 1
        llm_total_ms += suggestion.elapsed_ms

        # Try each suggested tactic
        found = False
        for tactic_str in suggestion.tactics:
            nodes_explored += 1
            if nodes_explored > max_nodes:
                break

            # L0 quick filter
            tac_name = tactic_str.split()[0] if tactic_str.split() else ""
            if tac_name == "intro" and not goal.target.is_pi:
                reducer = Reducer(env, dict(current_state.meta_ctx.assignments))
                target_whnf = reducer.whnf(goal.target)
                if not target_whnf.is_pi:
                    l0_filtered += 1
                    continue
            if tac_name == "assumption" and len(goal.local_ctx) == 0:
                l0_filtered += 1
                continue

            # Execute tactic (L1 check happens inside)
            result = execute_tactic(current_state, tactic_str)
            tactic_total_us += result.elapsed_us

            if result.success and result.state is not None:
                # Don't count sorry as "real" progress unless it's the only option
                if "sorry" in tactic_str and len(suggestion.tactics) > 1:
                    continue

                proof_tactics.append(tactic_str)
                current_state = result.state
                found = True
                break
            else:
                l1_filtered += 1

        if not found:
            # Use sorry as last resort to continue exploring
            result = execute_tactic(current_state, "sorry")
            if result.success and result.state is not None:
                proof_tactics.append("sorry")
                current_state = result.state

    total_ms = (time.perf_counter() - t0) * 1000
    is_complete = current_state.is_complete()
    sorry_used = any("sorry" in t for t in proof_tactics)
    real_solved = is_complete and not sorry_used

    return ProblemResult(
        problem_id=problem.id,
        name=problem.name,
        solved=real_solved,
        proof_tactics=proof_tactics,
        total_ms=total_ms,
        nodes_explored=nodes_explored,
        l0_filtered=l0_filtered,
        l1_filtered=l1_filtered,
        max_depth=len(proof_tactics),
        llm_calls=llm_calls,
        llm_total_ms=llm_total_ms,
        tactic_total_us=tactic_total_us,
        sorry_used=sorry_used,
    )


def main():
    print("=" * 72)
    print("AI4Math — miniF2F Benchmark (APE Engine + LLM)")
    print("=" * 72)
    print()

    # Build environment
    print("Building Lean4 prelude environment...", end=" ")
    env = build_prelude_env()
    print(f"done ({len(env)} declarations)")

    # Initialize LLM engine (heuristic mode — no API needed)
    llm = LLMTacticEngine(use_api=False)  # use heuristic for reproducibility
    print(f"LLM engine: heuristic mode")
    print()

    # Run benchmark
    results: List[ProblemResult] = []
    solved_count = 0
    total_nodes = 0
    total_l0 = 0
    total_l1 = 0

    for i, problem in enumerate(MINIF2F_PROBLEMS):
        print(f"[{i+1}/{len(MINIF2F_PROBLEMS)}] {problem.id}: {problem.name}")

        result = run_proof_search(env, problem, llm)
        results.append(result)

        status = "✓ SOLVED" if result.solved else ("△ sorry" if result.sorry_used else "✗ FAILED")
        print(f"  {status} | tactics: {' → '.join(result.proof_tactics)}")
        print(f"  nodes: {result.nodes_explored} | L0 filter: {result.l0_filtered} | L1 filter: {result.l1_filtered} | {result.total_ms:.2f}ms")
        print()

        if result.solved:
            solved_count += 1
        total_nodes += result.nodes_explored
        total_l0 += result.l0_filtered
        total_l1 += result.l1_filtered

    # Summary
    print("=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)
    total = len(MINIF2F_PROBLEMS)
    sorry_count = sum(1 for r in results if r.sorry_used and not r.solved)
    failed_count = total - solved_count - sorry_count
    print(f"Problems:      {total}")
    print(f"Solved (real): {solved_count}/{total} ({100*solved_count/total:.1f}%)")
    print(f"Sorry used:    {sorry_count}")
    print(f"Failed:        {failed_count}")
    print(f"Total nodes:   {total_nodes}")
    print(f"L0 filtered:   {total_l0} ({100*total_l0/max(total_nodes,1):.1f}%)")
    print(f"L1 filtered:   {total_l1} ({100*total_l1/max(total_nodes,1):.1f}%)")
    total_time = sum(r.total_ms for r in results)
    total_tactic = sum(r.tactic_total_us for r in results)
    print(f"Total time:    {total_time:.1f}ms")
    print(f"Avg tactic:    {total_tactic/max(total_nodes,1):.1f}μs/tactic")
    print()

    # Per-problem table
    print(f"{'Problem':<20} {'Status':<10} {'Tactics':>8} {'Nodes':>8} {'L0+L1':>8} {'Time':>10}")
    print("─" * 68)
    for r in results:
        status = "✓" if r.solved else ("△" if r.sorry_used else "✗")
        print(f"{r.problem_id:<20} {status:<10} {r.max_depth:>8} {r.nodes_explored:>8} "
              f"{r.l0_filtered+r.l1_filtered:>8} {r.total_ms:>8.2f}ms")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/minif2f_benchmark.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to results/minif2f_benchmark.json")

    return results


if __name__ == "__main__":
    main()
