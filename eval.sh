#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# eval.sh — AI4Math 统一评测入口
# ═══════════════════════════════════════════════════════════════
#
# Usage:
#   ./eval.sh builtin                      # Smoke test
#   ./eval.sh minif2f                      # miniF2F test split
#   ./eval.sh minif2f --split valid        # miniF2F valid split
#   ./eval.sh putnambench                  # PutnamBench
#   ./eval.sh fate-m / fate-h / fate-x     # FATE variants
#   ./eval.sh proofnet                     # ProofNet
#   ./eval.sh formalmath --limit 100       # FormalMATH (first 100)
#   ./eval.sh all                          # Run all benchmarks
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BENCHMARK="${1:-builtin}"
shift || true

# Default data paths
declare -A BENCH_PATHS=(
    [builtin]=""
    [minif2f]="data/miniF2F"
    [putnambench]="data/PutnamBench"
    [fate-m]="data/FATE-M"
    [fate-h]="data/FATE-H"
    [fate-x]="data/FATE-X"
    [proofnet]="data/ProofNet"
    [formalmath]="data/FormalMATH"
)

# Default config overrides
declare -A BENCH_CONFIGS=(
    [builtin]="config/default.yaml"
    [putnambench]="config/experiments/putnambench.yaml"
)

run_benchmark() {
    local bench="$1"; shift
    local path="${BENCH_PATHS[$bench]:-}"
    local config="${BENCH_CONFIGS[$bench]:-config/default.yaml}"

    echo "════════════════════════════════════════════════════"
    echo "  Evaluating: $bench"
    echo "  Config:     $config"
    echo "  Data path:  ${path:-'(builtin)'}"
    echo "════════════════════════════════════════════════════"

    python run_eval.py \
        --benchmark "$bench" \
        --config "$config" \
        ${path:+--path "$path"} \
        --output "results" \
        "$@"
}

if [ "$BENCHMARK" = "all" ]; then
    for bench in builtin minif2f putnambench fate-m fate-h fate-x proofnet formalmath; do
        run_benchmark "$bench" "$@" || echo "⚠ $bench failed, continuing..."
    done
else
    run_benchmark "$BENCHMARK" "$@"
fi
