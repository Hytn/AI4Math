"""run_single.py — 单题测试 CLI"""
import argparse, logging, sys
sys.path.insert(0, ".")
from config.schema import load_config
from agent.brain.claude_provider import create_provider
from agent.executor.lean_env import LeanEnvironment
from prover.premise.selector import PremiseSelector
from prover.pipeline.orchestrator import Orchestrator
from prover.models import BenchmarkProblem
from benchmarks.datasets.builtin.problems import BUILTIN_PROBLEMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

def main():
    parser = argparse.ArgumentParser(description="AI4Math — Single Problem Test")
    parser.add_argument("--builtin", type=str, help="Builtin problem name")
    parser.add_argument("--theorem", type=str, help="Custom theorem statement")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--provider", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.provider:
        config.setdefault("agent", {}).setdefault("brain", {})["provider"] = args.provider

    if args.builtin:
        matching = [p for p in BUILTIN_PROBLEMS if args.builtin in p.name]
        problem = matching[0] if matching else BUILTIN_PROBLEMS[0]
    elif args.theorem:
        problem = BenchmarkProblem("custom", "custom", args.theorem)
    else:
        problem = BUILTIN_PROBLEMS[0]

    print(f"\nProblem: {problem.name}\nStatement: {problem.theorem_statement}\n")

    brain_cfg = config.get("agent", {}).get("brain", {})
    llm = create_provider(brain_cfg)
    v = config.get("prover", {}).get("verifier", {})
    lean = LeanEnvironment(mode=v.get("mode", "docker"), timeout=v.get("timeout_seconds", 120))
    retriever = PremiseSelector()
    p_cfg = config.get("prover", {}).get("pipeline", {})
    orc = Orchestrator(lean, llm, retriever, p_cfg)
    trace = orc.prove(problem)
    trace.save(Path("results/traces") / f"{problem.problem_id}.json")
    print(f"\n{'✓ PROVED!' if trace.solved else '✗ FAILED'} — {trace.total_attempts} attempts, {trace.total_tokens} tokens")
    if trace.solved: print(f"Proof:\n{trace.successful_proof}")

from pathlib import Path
if __name__ == "__main__": main()
