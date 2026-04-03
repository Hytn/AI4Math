"""
run_eval.py — 批量评测入口

用法:
  # 内置示例 (不需要 Lean 环境即可冒烟测试流程)
  python run_eval.py --benchmark builtin --provider mock

  # miniF2F 评测
  python run_eval.py --benchmark miniF2F --path ./miniF2F --split test --max-attempts 10

  # 只跑前 N 道题 (调试用)
  python run_eval.py --benchmark miniF2F --path ./miniF2F --limit 5
"""

import argparse
import logging
import yaml
from pathlib import Path

from core.models import BenchmarkProblem, ProofTrace
from core.lean_checker import LeanChecker
from core.llm_policy import create_provider
from core.retriever import PremiseRetriever
from core.orchestrator import Orchestrator, OrchestratorConfig
from benchmarks.loader import load_benchmark
from benchmarks.eval_runner import EvalRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="AI4Math — 批量评测")
    parser.add_argument("--benchmark", type=str, default="builtin")
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 道题")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # 加载题目
    problems = load_benchmark(
        benchmark=args.benchmark,
        split=args.split,
        path=args.path or config.get("benchmark", {}).get("path", ""),
    )

    if args.limit:
        problems = problems[:args.limit]

    if not problems:
        print("No problems loaded. Check your benchmark/path settings.")
        return

    print(f"Loaded {len(problems)} problems from {args.benchmark}/{args.split}")

    # 构建组件
    llm_config = config.get("llm", {})
    if args.provider:
        llm_config["provider"] = args.provider
    llm = create_provider(llm_config)

    lean_config = config.get("lean", {})
    lean = LeanChecker(
        mode=lean_config.get("mode", "docker"),
        docker_image=lean_config.get("docker_image", "ai4math-lean"),
        docker_container=lean_config.get("docker_container", ""),
        timeout_seconds=lean_config.get("timeout_seconds", 120),
    )

    retriever = PremiseRetriever(config.get("retriever", {}))

    orch_config = config.get("orchestrator", {})
    orc = Orchestrator(
        lean_checker=lean,
        llm_provider=llm,
        retriever=retriever,
        config=OrchestratorConfig(
            max_attempts=args.max_attempts or orch_config.get("max_attempts", 10),
            temperature=orch_config.get("temperature", 0.7),
        ),
    )

    # 执行评测
    runner = EvalRunner(
        orchestrator=orc,
        output_dir=args.output,
    )

    result = runner.run(
        problems=problems,
        benchmark_name=args.benchmark,
        split=args.split,
    )

    print(f"\n{result.summary()}")


if __name__ == "__main__":
    main()
