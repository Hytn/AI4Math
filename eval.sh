#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  AI4Math — One-Command Evaluation
# ═══════════════════════════════════════════════════════════════════════
#
#  复现全部结果只需:
#    bash eval.sh                           # 默认: mock 冒烟测试
#    bash eval.sh --real                    # 真实 Claude API 评测
#    bash eval.sh --benchmark minif2f       # 只跑 miniF2F
#    bash eval.sh --quick                   # 快速验证 (每 benchmark 10 题)
#
#  环境变量:
#    ANTHROPIC_API_KEY  — Claude API Key (不设则 mock 模式)
#    MODEL              — 模型名 (默认 claude-sonnet-4-20250514)
#    MAX_SAMPLES        — 每题最大尝试次数 (默认 8)
#
#  v11 历史:
#    - 删除 Step 3 "APE 引擎性能基准": import 的 engine.core / engine.search
#      在 v8 已被删除, 这一步从 v8 起就会立即崩溃。
#    - 删除 --multi-role 旗标: 在 v9 从 run_eval.py argparse 移除,
#      转发给 run_eval.py 会触发 'unrecognized arguments' 错误。
#    - 新增 --profile 旗标: 与 run_unified.py 行为一致。
#
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'
C='\033[0;36m'; W='\033[1;37m'; N='\033[0m'
info()  { echo -e "${C}[INFO]${N} $*"; }
ok()    { echo -e "${G}[ OK ]${N} $*"; }
warn()  { echo -e "${Y}[WARN]${N} $*"; }
fail()  { echo -e "${R}[FAIL]${N} $*"; exit 1; }
header(){ echo -e "\n${W}═══════════════════════════════════════════════════════════${N}"; echo -e "${W}  $*${N}"; echo -e "${W}═══════════════════════════════════════════════════════════${N}\n"; }

MODE="mock"; BENCHMARK="all"; LIMIT=0; SPLIT="test"; LEAN_MODE="skip"
MAX_SAMPLES="${MAX_SAMPLES:-8}"; MODEL="${MODEL:-claude-sonnet-4-20250514}"
PROFILE="whole_proof_repair"
NO_KNOWLEDGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --real)         MODE="anthropic"; shift ;;
        --mock)         MODE="mock"; shift ;;
        --quick)        LIMIT=10; shift ;;
        --benchmark)    BENCHMARK="$2"; shift 2 ;;
        --limit)        LIMIT="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --samples)      MAX_SAMPLES="$2"; shift 2 ;;
        --max-samples)  MAX_SAMPLES="$2"; shift 2 ;;
        --split)        SPLIT="$2"; shift 2 ;;
        --lean)         LEAN_MODE="real"; shift ;;
        --profile)      PROFILE="$2"; shift 2 ;;
        --no-knowledge) NO_KNOWLEDGE="--no-knowledge"; shift ;;
        --early-stop)
            warn "--early-stop was removed from run_eval.py in v9; ignoring."
            shift ;;
        --help|-h)
            echo "Usage: bash eval.sh [OPTIONS]"
            echo "  --real               使用真实 Claude API (需 ANTHROPIC_API_KEY)"
            echo "  --mock               使用 mock 模式 (默认)"
            echo "  --quick              快速测试 (每基准 10 题)"
            echo "  --benchmark NAME     跑指定 benchmark (默认 all)"
            echo "  --limit N            每基准最多跑 N 题"
            echo "  --model NAME         模型 (默认 claude-sonnet-4-20250514)"
            echo "  --samples N          pass@k 的 k (默认 8)"
            echo "  --split test|valid   数据切分 (默认 test)"
            echo "  --lean               启用真实 Lean 4 验证"
            echo "  --profile NAME       prover.unified profile (默认 whole_proof_repair)"
            echo "  --no-knowledge       禁用知识系统"
            exit 0 ;;
        *) fail "未知参数: $1" ;;
    esac
done

# ── Step 1: 依赖 ──
header "Step 1/3 — 依赖检查"
python3 -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+'" || \
    fail "需要 Python 3.10+"
python3 -c "import anthropic" 2>/dev/null || {
    info "安装依赖..."; pip install -r requirements.txt --break-system-packages -q 2>/dev/null || pip install -r requirements.txt -q; }
ok "依赖就绪"

# ── Step 2: 数据集 ──
header "Step 2/3 — 数据集验证 (已内置 1,631 道真实题目)"
python3 -c "
import sys; sys.path.insert(0,'.')
from benchmarks.loader import load_benchmark
total=0
for n,p,d in [('builtin','','内置冒烟'),('minif2f','data/miniF2F','miniF2F'),('putnambench','data/PutnamBench','PutnamBench'),('proofnet','data/ProofNet','ProofNet'),('fate-m','data/FATE-M','FATE-M 代数'),('fate-h','data/FATE-H','FATE-H 代数'),('fate-x','data/FATE-X','FATE-X 代数')]:
    try:
        ps=load_benchmark(n,'test',path=p) if p else load_benchmark(n)
        total+=len(ps); print(f'  \033[0;32m✓\033[0m {n:<15} {len(ps):>5} 题  {d}')
    except Exception as e: print(f'  \033[0;31m✗\033[0m {n:<15} 加载失败: {e}')
print(f'\n  合计: {total} 道形式化数学题')
"

# ── Step 3: 评测 ──
header "Step 3/3 — Benchmark 评测"
info "Provider=$MODE  Model=$MODEL  Profile=$PROFILE  Samples=$MAX_SAMPLES  Limit=${LIMIT:-全部}"
LIMIT_ARG=""; [ "${LIMIT:-0}" -gt 0 ] 2>/dev/null && LIMIT_ARG="--limit $LIMIT"
[ "$BENCHMARK" = "all" ] && BL="builtin minif2f putnambench proofnet fate-m fate-h fate-x" || BL="$BENCHMARK"

for bench in $BL; do
    echo -e "\n${C}──── 评测: $bench ────${N}"
    python3 run_eval.py --benchmark "$bench" --provider "$MODE" --model "$MODEL" \
        --max-samples "$MAX_SAMPLES" --lean-mode "$LEAN_MODE" --split "$SPLIT" \
        --profile "$PROFILE" $LIMIT_ARG $NO_KNOWLEDGE 2>&1 \
        || warn "$bench 出现错误"
done

header "评测完成"
info "结果: results/evals/ (汇总) | results/traces/ (每题详情)"
[ "$MODE" = "mock" ] && echo -e "${Y}提示: Mock 模式. 真实评测:${N}\n  export ANTHROPIC_API_KEY=\"sk-...\"\n  bash eval.sh --real --model claude-opus-4-6"
echo ""
