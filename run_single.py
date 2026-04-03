"""
run_single.py — 单题测试入口

快速测试一道题的证明流程，适合开发调试。

用法:
  # 跑内置示例
  python run_single.py --builtin nat_add_comm

  # 跑自定义 theorem
  python run_single.py --theorem "theorem foo (n : Nat) : n + 0 = n"

  # 指定参数
  python run_single.py --builtin amgm_two_vars --max-attempts 5 --provider mock
"""

import argparse
import logging
import yaml
from pathlib import Path

from core.models import BenchmarkProblem, ProofAttempt, AttemptStatus
from core.lean_checker import LeanChecker
from core.llm_policy import create_provider
from core.retriever import PremiseRetriever
from core.orchestrator import Orchestrator, OrchestratorConfig
from benchmarks.loader import load_builtin_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def on_attempt_callback(attempt: ProofAttempt):
    """每次尝试完成后的回调，打印实时状态"""
    status_icon = "✓" if attempt.lean_result == AttemptStatus.SUCCESS else "✗"
    print(f"\n  {status_icon} Attempt {attempt.attempt_number}")
    print(f"    Model: {attempt.llm_model}")
    print(f"    Tokens: {attempt.llm_tokens_in + attempt.llm_tokens_out}")
    print(f"    LLM latency: {attempt.llm_latency_ms}ms")
    print(f"    Lean check: {attempt.lean_check_ms}ms")

    if attempt.generated_proof:
        preview = attempt.generated_proof[:200]
        if len(attempt.generated_proof) > 200:
            preview += "\n    ... (truncated)"
        print(f"    Proof:\n    {preview}")

    if attempt.lean_errors:
        for err in attempt.lean_errors[:3]:
            print(f"    Error: {err.to_prompt_str()}")


def main():
    parser = argparse.ArgumentParser(description="AI4Math — 单题测试")
    parser.add_argument("--builtin", type=str, help="内置示例名称")
    parser.add_argument("--theorem", type=str, help="自定义 theorem statement")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--provider", type=str, default=None, help="LLM provider override")
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # 确定题目
    if args.builtin:
        examples = load_builtin_examples()
        matching = [p for p in examples if args.builtin in p.name]
        if not matching:
            print(f"No builtin example matching '{args.builtin}'")
            print(f"Available: {[p.name for p in examples]}")
            return
        problem = matching[0]
    elif args.theorem:
        problem = BenchmarkProblem(
            problem_id="custom",
            name="custom_theorem",
            theorem_statement=args.theorem,
        )
    else:
        # 默认跑第一个内置示例
        problem = load_builtin_examples()[0]

    print(f"\n{'='*60}")
    print(f"Problem: {problem.name}")
    print(f"Statement: {problem.theorem_statement}")
    print(f"{'='*60}\n")

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
        lean_project_dir=lean_config.get("lean_project_dir", "/workspace/lean-project"),
        local_project_dir=lean_config.get("local_project_dir", ""),
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
            temperature_escalation=orch_config.get("temperature_escalation", True),
            max_error_history=orch_config.get("max_error_history", 3),
        ),
        on_attempt=on_attempt_callback,
    )

    # 执行
    trace = orc.prove(problem)

    # 保存结果
    output_path = Path(args.output) / "traces" / f"{problem.problem_id}.json"
    trace.save(output_path)
    print(f"\nTrace saved to: {output_path}")

    # 最终状态
    if trace.solved:
        print(f"\n{'='*60}")
        print("✓ PROVED!")
        print(f"  Proof:\n{trace.successful_proof}")
        print(f"  Attempts: {trace.total_attempts}")
        print(f"  Tokens: {trace.total_tokens}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"✗ FAILED after {trace.total_attempts} attempts")
        print(f"  Tokens: {trace.total_tokens}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
