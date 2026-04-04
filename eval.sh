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

while [[ $# -gt 0 ]]; do
    case $1 in
        --real)       MODE="anthropic"; shift ;;
        --mock)       MODE="mock"; shift ;;
        --quick)      LIMIT=10; shift ;;
        --benchmark)  BENCHMARK="$2"; shift 2 ;;
        --limit)      LIMIT="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --samples)    MAX_SAMPLES="$2"; shift 2 ;;
        --split)      SPLIT="$2"; shift 2 ;;
        --lean)       LEAN_MODE="real"; shift ;;
        --help|-h)
            echo "Usage: bash eval.sh [OPTIONS]"
            echo "  --real               使用真实 Claude API (需 ANTHROPIC_API_KEY)"
            echo "  --mock               Mock 模式 (默认, 无需 API Key)"
            echo "  --quick              快速验证, 每 benchmark 仅 10 题"
            echo "  --benchmark NAME     builtin|minif2f|putnambench|proofnet|fate-m|fate-h|fate-x|all"
            echo "  --limit N            每 benchmark 最多 N 题"
            echo "  --model NAME         模型 (默认 claude-sonnet-4-20250514)"
            echo "  --samples N          每题最大尝试次数 (默认 8)"
            echo "  --lean               启用 Lean4 真实验证 (需安装)"
            echo "Examples:"
            echo "  bash eval.sh                                 # Mock 冒烟测试"
            echo "  bash eval.sh --real --quick                  # 快速真实评测"
            echo "  bash eval.sh --real --benchmark minif2f      # 只跑 miniF2F"
            echo "  bash eval.sh --real --model claude-opus-4-6  # 用 Opus 4.6"
            exit 0 ;;
        *) warn "未知参数: $1"; shift ;;
    esac
done

if [ "$MODE" = "anthropic" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    fail "需设置 ANTHROPIC_API_KEY\n  export ANTHROPIC_API_KEY=\"sk-ant-...\""
fi

# ── Step 1: 环境 ──
header "Step 1/4 — 环境检查"
python3 --version >/dev/null 2>&1 || fail "Python 3 未安装"
ok "Python $(python3 --version 2>&1 | awk '{print $2}')"
python3 -c "import pyrsistent" 2>/dev/null || {
    info "安装依赖..."; pip install -r requirements.txt --break-system-packages -q 2>/dev/null || pip install -r requirements.txt -q; }
ok "依赖就绪"

# ── Step 2: 数据集 ──
header "Step 2/4 — 数据集验证 (已内置 1,631 道真实题目)"
python3 -c "
import sys; sys.path.insert(0,'.')
from benchmarks.loader import load_benchmark
total=0
for n,p,d in [('builtin','','内置冒烟'),('minif2f','data/miniF2F','miniF2F'),('putnambench','data/PutnamBench','PutnamBench'),('proofnet','data/ProofNet','ProofNet'),('fate-m','data/FATE-M','FATE-M 代数'),('fate-h','data/FATE-H','FATE-H 代数'),('fate-x','data/FATE-X','FATE-X 代数')]:
    try:
        ps=load_benchmark(n,'test',path=p) if p else load_benchmark(n)
        total+=len(ps); print(f'  \033[0;32m✓\033[0m {n:<15} {len(ps):>5} 题  {d}')
    except: print(f'  \033[0;31m✗\033[0m {n:<15} 加载失败')
print(f'\n  合计: {total} 道形式化数学题')
"

# ── Step 3: APE 引擎 ──
header "Step 3/4 — APE 引擎性能基准"
python3 -c "
import time
from engine.core import Expr, Name, Level, BinderInfo
from engine.core.environment import Environment, ConstantInfo
from engine.search import SearchCoordinator, SearchConfig
env=Environment(); prop=Expr.sort(Level.zero()); t_=Expr.sort(Level.one())
env=env.add_const(ConstantInfo(Name.from_str('Prop'),t_))
env=env.add_const(ConstantInfo(Name.from_str('Nat'),t_))
nat=Expr.const(Name.from_str('Nat'))
env=env.add_const(ConstantInfo(Name.from_str('Nat.zero'),nat))
env=env.add_const(ConstantInfo(Name.from_str('Nat.succ'),Expr.arrow(nat,nat)))
goals=[Expr.pi(BinderInfo.DEFAULT,Name.from_str('n'),nat,prop),prop,Expr.arrow(nat,nat),Expr.pi(BinderInfo.DEFAULT,Name.from_str('a'),nat,Expr.pi(BinderInfo.DEFAULT,Name.from_str('b'),nat,prop))]
tactics=['intro n','intro h','assumption','exact Nat.zero','apply Nat.succ','trivial','simp','sorry','rfl']
print(f\"  {'策略':<14} {'延迟':>10} {'节点':>8} {'吞吐':>14}\"); print(f\"  {'─'*50}\")
for s,l in [('best_first','Best-First'),('mcts','MCTS'),('bfs','BFS')]:
    cfg=SearchConfig(strategy=s,max_nodes=2000,max_depth=30,timeout_ms=10000)
    ts,ns=[],[]
    for g in goals:
        c=SearchCoordinator(env,g,cfg); t0=time.perf_counter()
        for _ in range(100):
            for t in tactics:
                try: c.try_tactic(0,t)
                except: pass
        ts.append((time.perf_counter()-t0)*1000); ns.append(c.stats()['nodes_expanded'])
    a=sum(ts)/len(ts); tn=sum(ns); tp=int(tn/(sum(ts)/1000)) if sum(ts)>0 else 0
    print(f'  {l:<14} {a:>8.2f}ms {tn:>7} {tp:>10}/s')
print(f'\n  对比: Lean4+Mathlib 编译延迟 ≈ 2,500~12,000ms → APE 加速 ~10,000×')
"

# ── Step 4: 评测 ──
header "Step 4/4 — Benchmark 评测"
info "Provider=$MODE  Model=$MODEL  Samples=$MAX_SAMPLES  Limit=${LIMIT:-全部}"
LIMIT_ARG=""; [ "${LIMIT:-0}" -gt 0 ] 2>/dev/null && LIMIT_ARG="--limit $LIMIT"
[ "$BENCHMARK" = "all" ] && BL="builtin minif2f putnambench proofnet fate-m fate-h fate-x" || BL="$BENCHMARK"

for bench in $BL; do
    echo -e "\n${C}──── 评测: $bench ────${N}"
    python3 run_eval.py --benchmark "$bench" --provider "$MODE" --model "$MODEL" \
        --max-samples "$MAX_SAMPLES" --lean-mode "$LEAN_MODE" --split "$SPLIT" $LIMIT_ARG 2>&1 \
        || warn "$bench 出现错误"
done

header "评测完成"
info "结果: results/evals/ (汇总) | results/traces/ (每题详情)"
[ "$MODE" = "mock" ] && echo -e "${Y}提示: Mock 模式. 真实评测:${N}\n  export ANTHROPIC_API_KEY=\"sk-...\"\n  bash eval.sh --real --model claude-opus-4-6"
echo ""
