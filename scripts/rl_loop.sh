#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  AI4Math — One-Command RL Flywheel
# ═══════════════════════════════════════════════════════════════════════
#
#  Closes the v4 RL gap. Chains:
#
#      eval (run_eval.py)            → results/rl/iter_<N>/traces/
#         ↓
#      collect (dialogs → SFT JSONL) → results/rl/iter_<N>/sft.jsonl
#         ↓
#      train_wm (sklearn classifier) → results/rl/iter_<N>/world_model.pkl
#         ↓
#      train_llm (delegated)         → results/rl/iter_<N>/model_weights/
#
#  Examples:
#    bash scripts/rl_loop.sh                    # 1 iter on builtin, mock LLM
#    bash scripts/rl_loop.sh --real --iters 3   # 3 iters, real Claude API
#    bash scripts/rl_loop.sh --benchmark minif2f --successful-only
#
#  Env:
#    ANTHROPIC_API_KEY   — required for --real mode
#    MODEL               — default: claude-sonnet-4-20250514
#    TRAIN_CMD           — shell template for stage 4 (LLM training).
#                          Use {sft_jsonl} and {model_out} placeholders.
#                          Omit to skip stage 4 (SFT JSONL still produced).
#
#  All artifacts land under  results/rl/iter_<N>/...
#
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────
ITERS=1
PROVIDER="mock"
PROFILE="whole_proof_repair"
BENCHMARK="builtin"
LIMIT=0
MAX_SAMPLES=4
SUCCESSFUL_ONLY=""
STAGES="eval,collect,train_wm,train_llm"
OUT_ROOT="results/rl"
SFT_PRESET="qwen3"
KEEP_GOING=""
VERBOSE=""
TRAIN_CMD="${TRAIN_CMD:-}"
MODEL="${MODEL:-}"

# ── Args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --iters)          ITERS="$2"; shift 2 ;;
        --real)           PROVIDER="anthropic"; shift ;;
        --mock)           PROVIDER="mock"; shift ;;
        --provider)       PROVIDER="$2"; shift 2 ;;
        --profile)        PROFILE="$2"; shift 2 ;;
        --benchmark)      BENCHMARK="$2"; shift 2 ;;
        --limit)          LIMIT="$2"; shift 2 ;;
        --samples|--max-samples)  MAX_SAMPLES="$2"; shift 2 ;;
        --model)          MODEL="$2"; shift 2 ;;
        --successful-only) SUCCESSFUL_ONLY="--successful-only"; shift ;;
        --stages)         STAGES="$2"; shift 2 ;;
        --sft-preset)     SFT_PRESET="$2"; shift 2 ;;
        --out-root)       OUT_ROOT="$2"; shift 2 ;;
        --train-cmd)      TRAIN_CMD="$2"; shift 2 ;;
        --keep-going)     KEEP_GOING="--keep-going"; shift ;;
        --verbose|-v)     VERBOSE="--verbose"; shift ;;
        --help|-h)
            sed -n '1,40p' "$0" | sed 's/^#//'
            exit 0 ;;
        *)
            echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Pre-flight ────────────────────────────────────────────────────────
if [[ "$PROVIDER" == "anthropic" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "[FAIL] --real mode but ANTHROPIC_API_KEY not set" >&2
    exit 2
fi

CMD=( "$REPO_ROOT/scripts/rl_pipeline.py" "loop"
      "--iters" "$ITERS"
      "--out-root" "$OUT_ROOT"
      "--profile" "$PROFILE"
      "--benchmark" "$BENCHMARK"
      "--provider" "$PROVIDER"
      "--max-samples" "$MAX_SAMPLES"
      "--stages" "$STAGES"
      "--sft-preset" "$SFT_PRESET" )

[[ "$LIMIT" -gt 0 ]]            && CMD+=( "--limit" "$LIMIT" )
[[ -n "$MODEL" ]]               && CMD+=( "--model" "$MODEL" )
[[ -n "$SUCCESSFUL_ONLY" ]]     && CMD+=( "$SUCCESSFUL_ONLY" )
[[ -n "$KEEP_GOING" ]]          && CMD+=( "$KEEP_GOING" )
[[ -n "$VERBOSE" ]]             && CMD+=( "$VERBOSE" )
[[ -n "$TRAIN_CMD" ]]           && CMD+=( "--train-cmd" "$TRAIN_CMD" )

echo "[INFO] AI4Math RL flywheel starting"
echo "[INFO]   iterations: $ITERS"
echo "[INFO]   profile:    $PROFILE"
echo "[INFO]   benchmark:  $BENCHMARK"
echo "[INFO]   provider:   $PROVIDER"
echo "[INFO]   stages:     $STAGES"
echo "[INFO]   out_root:   $OUT_ROOT"
[[ -n "$TRAIN_CMD" ]] && echo "[INFO]   train_cmd:  $TRAIN_CMD"

exec python3 "${CMD[@]}"
