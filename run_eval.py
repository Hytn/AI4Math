"""run_eval.py — 批量评测 CLI"""
import argparse, logging, sys
sys.path.insert(0, ".")
from config.schema import load_config
from agent.brain.claude_provider import create_provider
from agent.executor.lean_env import LeanEnvironment
from prover.premise.selector import PremiseSelector
from prover.pipeline.orchestrator import Orchestrator
from benchmarks.loader import load_benchmark
from benchmarks.eval_runner import EvalRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

def main():
    parser = argparse.ArgumentParser(description="AI4Math — Batch Evaluation")
    parser.add_argument("--benchmark", default="builtin")
    parser.add_argument("--path", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--output", default="results")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.provider:
        config.setdefault("agent", {}).setdefault("brain", {})["provider"] = args.provider

    problems = load_benchmark(args.benchmark, args.split, args.path)
    if args.limit: problems = problems[:args.limit]
    if not problems: print("No problems loaded."); return

    print(f"Loaded {len(problems)} problems from {args.benchmark}/{args.split}")

    brain_cfg = config.get("agent", {}).get("brain", {})
    llm = create_provider(brain_cfg)
    verifier_cfg = config.get("prover", {}).get("verifier", {})
    lean = LeanEnvironment(mode=verifier_cfg.get("mode", "docker"),
                           docker_image=verifier_cfg.get("docker_image", "ai4math-lean"),
                           timeout=verifier_cfg.get("timeout_seconds", 120))
    retriever = PremiseSelector(config.get("prover", {}).get("premise", {}))

    pipeline_cfg = config.get("prover", {}).get("pipeline", {})
    pipeline_cfg.update(config.get("agent", {}).get("strategy", {}))
    orc = Orchestrator(lean_env=lean, llm_provider=llm, retriever=retriever, config=pipeline_cfg)
    runner = EvalRunner(orchestrator=orc, output_dir=args.output)
    result = runner.run(problems, args.benchmark, args.split)
    print(f"\n{result.summary()}")

if __name__ == "__main__":
    main()
