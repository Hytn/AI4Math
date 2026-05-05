#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  reproduce_minif2f.sh — miniF2F-test 复现脚本
# ═══════════════════════════════════════════════════════════════════════
#
#  这个脚本不是魔法。它做的事:
#
#    1. 检查 Python ≥ 3.10 + pip 依赖
#    2. 拉 miniF2F-lean4 数据集 (244 题 test split) — 如果还没拉
#    3. 检查 Lean 4 / elan 是否就绪 + 本机 toolchain 是否匹配 v4.24.0
#    4. (可选) 在 data/miniF2F/ 里 lake build (首次约 30-60 分钟)
#    5. 调用 run_eval.py, 把 project_dir 正确指向 data/miniF2F
#    6. 输出 pass@k 报表 + 每题 dialog.json
#
#  ★ 这个脚本不能让 mock provider 跑出真实分数。证明能力 100% 来自后端 LLM。
#  ★ pass@k 数字只有 --lean-mode=real 时才有意义; skip 模式仅做 LLM 输出
#    存活检查 (非 sorry / 有 ```lean 块), 报表会被打上 [unverified] 标记。
#
# ─── 用法示例 ──────────────────────────────────────────────────────────
#
#  # 0. 完全冒烟 (没装 lean / 没有 API key 也能跑) — 仅检查管线
#  bash reproduce_minif2f.sh --smoke
#
#  # 1. DeepSeek-Prover-V2 671B 自托管 (vLLM): 论文里 Pass@8192 = 88.9%.
#  #    你需要一台 8×H100 节点先把模型部署起来:
#  #      python -m vllm.entrypoints.openai.api_server \
#  #          --model deepseek-ai/DeepSeek-Prover-V2-671B \
#  #          --tensor-parallel-size 8 --port 8000
#  bash reproduce_minif2f.sh \
#      --provider vllm \
#      --api-base http://localhost:8000/v1 \
#      --model deepseek-ai/DeepSeek-Prover-V2-671B \
#      --profile whole_proof \
#      --samples 32 \
#      --lean
#
#  # 2. 用 Claude Opus 4.7 + 编译反馈循环 (走 Anthropic API):
#  export ANTHROPIC_API_KEY=sk-ant-...
#  bash reproduce_minif2f.sh \
#      --provider anthropic \
#      --model claude-opus-4-5 \
#      --profile whole_proof_repair \
#      --samples 32 --lean
#
#  # 3. 用 DeepSeek API 的 deepseek-reasoner:
#  export DEEPSEEK_API_KEY=sk-...
#  bash reproduce_minif2f.sh \
#      --provider deepseek --model deepseek-reasoner \
#      --profile whole_proof_repair --samples 32 --lean
#
#  # 4. 异构并行 (4 路混搭, 基础设施开销最大但样本效率最高):
#  bash reproduce_minif2f.sh \
#      --provider anthropic --model claude-opus-4-5 \
#      --profile heterogeneous --samples 16 --lean
#
# ─── 参数 ──────────────────────────────────────────────────────────────
#
#  --smoke              纯冒烟, 走 mock provider + mock backend, 不需要任何外部资源
#  --provider NAME      anthropic / openai / deepseek / vllm / sglang / ollama / mock
#  --model NAME         模型名 (默认 claude-sonnet-4-20250514)
#  --api-base URL       OpenAI-compatible 端点 (vllm / sglang / 自托管必填)
#  --profile NAME       prover.unified profile (默认 whole_proof_repair).
#                       论文复现用 dsp_v2_cot / 工程评测用 whole_proof_repair.
#  --samples N          每题采样次数 (pass@N 的 N). 默认 32.
#                       注意: 论文里 88.9% 是 pass@8192, API 调用预算上是不可行的.
#  --limit N            只跑前 N 题 (调试用, 默认 0=全部 244 题)
#  --lean               启用真实 Lean 4 验证 (强烈推荐). 没开等于报 [unverified].
#  --skip-lean-build    跳过 lake build 步骤 (假设你已经手动构建过了)
#  --pool-size N        Lean REPL 并行池大小 (默认 4)
#  --output-dir DIR     结果根目录 (默认 results/minif2f_run_<timestamp>)
#  --temperature T      override profile 默认 temperature
#                       (DSP-V2 系列用 1.0, 通用 LLM 用 0.7-0.8)
#  --max-turns N        override profile 默认 max_turns (反馈循环深度)
#  --knowledge-db PATH  跨 eval run 共享同一份持久化 KB (A/B sweep 必用)
#  --dialog-index PATH  跨题成功 dialog 检索 (要求 profile 开启 inject_similar_dialogs)
#  --lemma-bank-db PATH 跨题引理库 (BM25 检索辅助引理)
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'
C='\033[0;36m'; W='\033[1;37m'; N='\033[0m'
info()   { echo -e "${C}[INFO]${N} $*"; }
ok()     { echo -e "${G}[ OK ]${N} $*"; }
warn()   { echo -e "${Y}[WARN]${N} $*"; }
fail()   { echo -e "${R}[FAIL]${N} $*"; exit 1; }
section(){ echo -e "\n${W}═══ $* ═══${N}"; }

# ── Defaults ──────────────────────────────────────────────────────────
SMOKE=0
PROVIDER="anthropic"
MODEL="claude-sonnet-4-20250514"
API_BASE=""
PROFILE="whole_proof_repair"
SAMPLES=32
LIMIT=0
LEAN_MODE="skip"
POOL_SIZE=4
OUTPUT_DIR="results/minif2f_run_$(date +%Y%m%d_%H%M%S)"
SKIP_LEAN_BUILD=0
TEMPERATURE=""        # default: use profile's own temperature
MAX_TURNS=""          # default: use profile's own max_turns
KNOWLEDGE_DB=""       # default: scoped inside OUTPUT_DIR
DIALOG_INDEX=""
LEMMA_BANK_DB=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)            SMOKE=1; shift ;;
        --provider)         PROVIDER="$2"; shift 2 ;;
        --model)            MODEL="$2"; shift 2 ;;
        --api-base)         API_BASE="$2"; shift 2 ;;
        --profile)          PROFILE="$2"; shift 2 ;;
        --samples)          SAMPLES="$2"; shift 2 ;;
        --limit)            LIMIT="$2"; shift 2 ;;
        --lean)             LEAN_MODE="real"; shift ;;
        --skip-lean-build)  SKIP_LEAN_BUILD=1; shift ;;
        --pool-size)        POOL_SIZE="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --temperature)      TEMPERATURE="$2"; shift 2 ;;
        --max-turns)        MAX_TURNS="$2"; shift 2 ;;
        --knowledge-db)     KNOWLEDGE_DB="$2"; shift 2 ;;
        --dialog-index)     DIALOG_INDEX="$2"; shift 2 ;;
        --lemma-bank-db)    LEMMA_BANK_DB="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# ═══/,/^# ═══/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) fail "未知参数: $1 (用 --help 看完整列表)" ;;
    esac
done

if [ "$SMOKE" -eq 1 ]; then
    info "smoke 模式: 强制 provider=mock, backend=mock, lean-mode=skip, samples=1, limit=5"
    PROVIDER="mock"
    LEAN_MODE="skip"
    SAMPLES=1
    LIMIT=5
fi

# ───── Step 1: Python + 依赖 ──────────────────────────────────────────
section "Step 1/5  Python + 依赖检查"
python3 -c 'import sys; assert sys.version_info >= (3,10), sys.version' \
    || fail "需要 Python 3.10+ (本机 $(python3 --version 2>&1))"
ok "$(python3 --version)"

if ! python3 -c "import anthropic" 2>/dev/null; then
    info "安装依赖中..."
    pip install -r requirements.txt -q --break-system-packages 2>/dev/null \
        || pip install -r requirements.txt -q
fi
ok "依赖就绪"

# ───── Step 2: 数据集 ─────────────────────────────────────────────────
section "Step 2/5  miniF2F-lean4 数据集"
MINIF2F_DIR="$SCRIPT_DIR/data/miniF2F"
if [ -d "$MINIF2F_DIR/MiniF2F" ]; then
    ok "$MINIF2F_DIR 已存在"
else
    info "拉 miniF2F-lean4..."
    rm -rf "$MINIF2F_DIR"
    git clone --depth 1 https://github.com/yangky11/miniF2F-lean4.git \
        "$MINIF2F_DIR" 2>&1 | tail -3
    [ -d "$MINIF2F_DIR/MiniF2F" ] || fail "git clone 失败"
fi

N_TEST=$(ls "$MINIF2F_DIR/MiniF2F/Test/"*.lean 2>/dev/null | wc -l)
if [ "$N_TEST" -lt 240 ]; then
    fail "miniF2F-test 只发现 $N_TEST 题, 期望 244 题"
fi
ok "miniF2F-test 已就绪: $N_TEST 道 .lean 文件"

# ───── Step 3: Lean toolchain ─────────────────────────────────────────
section "Step 3/5  Lean 4 toolchain"
PINNED=$(cat "$MINIF2F_DIR/lean-toolchain" 2>/dev/null | tr -d '[:space:]')
info "miniF2F 锁定的 toolchain: $PINNED"

if [ "$LEAN_MODE" != "real" ]; then
    warn "未传 --lean. 跳过 toolchain 检查 (报表将被标 [unverified])"
elif ! command -v lean >/dev/null 2>&1; then
    fail "未在 PATH 找到 lean 二进制。

  以下两种方式之一安装:

  a) 用 elan 装 (推荐):
       curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \\
            | sh -s -- -y --default-toolchain $PINNED

  b) 用 docker:
       docker build -t ai4math-lean -f docker/Dockerfile.lean docker/

  装好后重跑本脚本 (--lean 仍生效).

  如果你只想冒烟一遍管线, 不验证证明对错:
       bash reproduce_minif2f.sh --smoke"
else
    HAVE=$(lean --version 2>&1 | head -1 || echo unknown)
    info "lean --version: $HAVE"
    HAVE_TAG=$(echo "$HAVE" | grep -oE 'version [0-9]+\.[0-9]+(\.[0-9]+)?' | head -1 | sed 's/version /v/')
    PINNED_TAG=$(echo "$PINNED" | grep -oE 'v[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
    if [ -n "$HAVE_TAG" ] && [ -n "$PINNED_TAG" ] && [ "$HAVE_TAG" != "$PINNED_TAG" ]; then
        warn "Lean 版本错配: 锁定 $PINNED_TAG, 本机 $HAVE_TAG."
        warn "Mathlib API 在这些版本之间不稳定 — 大概率全部 verify 失败."
        warn "建议: 在 $MINIF2F_DIR 内运行 'elan override set $PINNED'"
    fi

    if [ "$SKIP_LEAN_BUILD" -eq 0 ]; then
        info "在 $MINIF2F_DIR 内 lake build (首次约 30-60 分钟)..."
        cd "$MINIF2F_DIR"
        lake exe cache get >/dev/null 2>&1 || true
        if ! lake build 2>&1 | tail -5; then
            warn "lake build 报错。如果你确认 mathlib 已在别处编译过, 可加 --skip-lean-build"
        fi
        cd "$SCRIPT_DIR"
    else
        info "--skip-lean-build 已启用, 跳过编译"
    fi
fi
ok "Lean 阶段结束"

# ───── Step 4: API key 检查 ───────────────────────────────────────────
section "Step 4/5  LLM provider"
case "$PROVIDER" in
    anthropic)
        [ -z "${ANTHROPIC_API_KEY:-}" ] && fail "未设 ANTHROPIC_API_KEY"
        ok "ANTHROPIC_API_KEY 已设"
        ;;
    deepseek)
        [ -z "${DEEPSEEK_API_KEY:-}" ] && fail "未设 DEEPSEEK_API_KEY"
        ok "DEEPSEEK_API_KEY 已设"
        ;;
    openai)
        [ -z "${OPENAI_API_KEY:-}" ] && fail "未设 OPENAI_API_KEY"
        ok "OPENAI_API_KEY 已设"
        ;;
    vllm|sglang|ollama|openai_compat)
        [ -z "$API_BASE" ] && fail "$PROVIDER 必须配 --api-base http://host:port/v1"
        info "Probing $API_BASE/models ..."
        if curl -sS --max-time 5 "$API_BASE/models" >/dev/null 2>&1; then
            ok "$API_BASE 可达"
        else
            warn "$API_BASE 探测失败 (可能 /models 没暴露). 继续, 评测时会再试."
        fi
        ;;
    mock) info "mock provider — 不调外部 LLM (smoke 模式)" ;;
    *) fail "未知 provider: $PROVIDER" ;;
esac

# ───── Step 5: 评测 ────────────────────────────────────────────────────
section "Step 5/5  跑评测"
info "Provider     : $PROVIDER"
info "Model        : $MODEL"
[ -n "$API_BASE" ] && info "API base     : $API_BASE"
info "Profile      : $PROFILE"
info "Samples (k)  : $SAMPLES"
info "Lean mode    : $LEAN_MODE"
info "Limit        : ${LIMIT:-全部 244 题}"
info "Output       : $OUTPUT_DIR"

LIMIT_ARG=""; [ "${LIMIT:-0}" -gt 0 ] 2>/dev/null && LIMIT_ARG="--limit $LIMIT"
API_BASE_ARG=""; [ -n "$API_BASE" ] && API_BASE_ARG="--api-base $API_BASE"
TEMP_ARG="";  [ -n "$TEMPERATURE"   ] && TEMP_ARG="--temperature $TEMPERATURE"
TURNS_ARG=""; [ -n "$MAX_TURNS"     ] && TURNS_ARG="--max-turns $MAX_TURNS"
KB_ARG="";    [ -n "$KNOWLEDGE_DB"  ] && KB_ARG="--knowledge-db $KNOWLEDGE_DB"
DI_ARG="";    [ -n "$DIALOG_INDEX"  ] && DI_ARG="--dialog-index $DIALOG_INDEX"
LB_ARG="";    [ -n "$LEMMA_BANK_DB" ] && LB_ARG="--lemma-bank-db $LEMMA_BANK_DB"

mkdir -p "$OUTPUT_DIR"
python3 run_eval.py \
    --benchmark    minif2f \
    --split        test \
    --provider     "$PROVIDER" \
    --model        "$MODEL" \
    --profile      "$PROFILE" \
    --max-samples  "$SAMPLES" \
    --lean-mode    "$LEAN_MODE" \
    --pool-size    "$POOL_SIZE" \
    --project-dir  "$MINIF2F_DIR" \
    --output-dir   "$OUTPUT_DIR" \
    --resume \
    $LIMIT_ARG $API_BASE_ARG $TEMP_ARG $TURNS_ARG \
    $KB_ARG $DI_ARG $LB_ARG

# ───── 报表 ───────────────────────────────────────────────────────────
section "结果"
EVAL_FILE="$OUTPUT_DIR/evals/eval_minif2f_test.json"
if [ -f "$EVAL_FILE" ]; then
    info "汇总: $EVAL_FILE"
    info "每题 dialog: $OUTPUT_DIR/traces/minif2f/<problem_id>/dialog.json"
    python3 - <<EOF
import json
with open("$EVAL_FILE") as f: e = json.load(f)
m = e["metrics"]
print()
print(f"  miniF2F-test  ({e.get('verification','?')})")
print(f"  ─────────────────────────────")
print(f"  Solved      : {m['solved']}/{m['total']}  ({m['solve_rate']*100:.1f}%)")
for k in sorted(int(k.split('@')[1]) for k in m if k.startswith('pass@')):
    print(f"  pass@{k:<6} : {m[f'pass@{k}']:.4f}")
print()
if e.get('verification') == 'unverified':
    print('  ⚠ 这次跑的是 --lean-mode=skip; 数字只表示 LLM 输出了非空、无 sorry 的代码,')
    print('    没有 Lean 4 编译器认可。要真实数字请加 --lean.')
EOF
fi
ok "完成。要在同一份 dialog 上加跑更多 samples, 重跑相同命令即可 (脚本默认 --resume)."
