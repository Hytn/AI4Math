#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# setup_and_eval.sh — AI4Math 基准数据集下载 & 评测一键脚本
# ═══════════════════════════════════════════════════════════════════════
#
# 用法:
#   bash setup_and_eval.sh              # 下载全部数据集 + 运行评测
#   bash setup_and_eval.sh --download   # 仅下载
#   bash setup_and_eval.sh --eval       # 仅评测 (需已下载)
#   bash setup_and_eval.sh --info       # 仅统计数据集信息
#
# 环境变量:
#   ANTHROPIC_API_KEY   — Claude API Key (不设则用 mock 模式)
#   PROVIDER            — LLM provider: anthropic / mock (默认 mock)
#   MODEL               — 模型名 (默认 claude-sonnet-4-20250514)
#   MAX_SAMPLES          — 每题最大尝试次数 (默认 8)
#   LIMIT               — 每个 benchmark 最多测几题 (默认 0=全部)
#   BENCHMARKS           — 要跑的 benchmark, 逗号分隔 (默认 all)
#   LEAN_MODE            — lean4验证模式: real / skip (默认 skip)
#
# 支持的真实基准:
#   miniF2F      — 488 道奥赛+高中+本科数学 (Lean 4, yangky11 版)
#   PutnamBench  — 672 道 Putnam 竞赛 (Lean 4)
#   ProofNet     — 371 道本科数学 (Lean 4)
#   FormalMATH   — 5560 道多领域数学 (Lean 4)
#
# 注意:
#   README 中提到的 "FATE-M/H/X" 并非真实存在的公开基准。
#   已替换为上述四个领域内公认的标准基准。
#
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DATA_DIR="$SCRIPT_DIR/data"

# ── 颜色 ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ── 默认参数 ──
PROVIDER="${PROVIDER:-mock}"
MODEL="${MODEL:-claude-sonnet-4-20250514}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
LIMIT="${LIMIT:-0}"
BENCHMARKS="${BENCHMARKS:-all}"
LEAN_MODE="${LEAN_MODE:-skip}"
ACTION="${1:---all}"

# ═══════════════════════════════════════════════════════════════════════
# 第一部分: 下载真实基准数据集
# ═══════════════════════════════════════════════════════════════════════

download_benchmarks() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  下载真实基准数据集"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    mkdir -p "$DATA_DIR"

    # ── 1. miniF2F (Lean 4) ──
    # 来源: LeanDojo 维护的 Lean 4 版本
    # 488 道题 (244 test + 244 valid), 涵盖 AMC/AIME/IMO/MATH
    local minif2f_dir="$DATA_DIR/miniF2F"
    if [ -d "$minif2f_dir/.git" ]; then
        info "miniF2F 已存在, 更新中..."
        cd "$minif2f_dir" && git pull --quiet 2>/dev/null || true && cd "$SCRIPT_DIR"
    else
        info "下载 miniF2F (yangky11/miniF2F-lean4)..."
        git clone --depth 1 https://github.com/yangky11/miniF2F-lean4.git "$minif2f_dir" 2>&1 | tail -1
    fi
    ok "miniF2F → $minif2f_dir"

    # ── 2. PutnamBench ──
    # 672 道 Putnam 竞赛题 (1962-2024), Lean 4 / Isabelle / Coq
    local putnam_dir="$DATA_DIR/PutnamBench"
    if [ -d "$putnam_dir/.git" ]; then
        info "PutnamBench 已存在, 更新中..."
        cd "$putnam_dir" && git pull --quiet 2>/dev/null || true && cd "$SCRIPT_DIR"
    else
        info "下载 PutnamBench (trishullab/PutnamBench)..."
        git clone --depth 1 https://github.com/trishullab/PutnamBench.git "$putnam_dir" 2>&1 | tail -1
    fi
    ok "PutnamBench → $putnam_dir"

    # ── 3. ProofNet (Lean 4) ──
    # 371 道本科级数学 (分析, 代数, 拓扑)
    local proofnet_dir="$DATA_DIR/ProofNet"
    if [ -d "$proofnet_dir/.git" ]; then
        info "ProofNet 已存在, 更新中..."
        cd "$proofnet_dir" && git pull --quiet 2>/dev/null || true && cd "$SCRIPT_DIR"
    else
        info "下载 ProofNet (rahul3613/ProofNet-lean4)..."
        git clone --depth 1 https://github.com/rahul3613/ProofNet-lean4.git "$proofnet_dir" 2>&1 | tail -1
    fi
    ok "ProofNet → $proofnet_dir"

    # ── 4. FormalMATH ──
    # 5560 道多领域多难度 (高中竞赛 ~ 本科定理)
    local formalmath_dir="$DATA_DIR/FormalMATH"
    if [ -d "$formalmath_dir/.git" ]; then
        info "FormalMATH 已存在, 更新中..."
        cd "$formalmath_dir" && git pull --quiet 2>/dev/null || true && cd "$SCRIPT_DIR"
    else
        info "下载 FormalMATH (Sphere-AI-Lab/FormalMATH-Bench)..."
        git clone --depth 1 https://github.com/Sphere-AI-Lab/FormalMATH-Bench.git "$formalmath_dir" 2>&1 | tail -1
    fi
    ok "FormalMATH → $formalmath_dir"

    echo ""
    ok "所有数据集下载完成 → $DATA_DIR/"
}

# ═══════════════════════════════════════════════════════════════════════
# 第二部分: 统计数据集信息
# ═══════════════════════════════════════════════════════════════════════

show_info() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  数据集统计"
    echo "════════════════════════════════════════════════════════════════"
    echo ""

    python3 - << 'PYEOF'
import sys, os
sys.path.insert(0, ".")
from benchmarks.loader import load_benchmark

datasets = {
    "builtin":    {"path": "",                   "desc": "内置冒烟测试"},
    "minif2f":    {"path": "data/miniF2F",       "desc": "奥赛+高中+本科 (AMC/AIME/IMO)"},
    "putnambench":{"path": "data/PutnamBench",   "desc": "Putnam 竞赛 (1962-2024)"},
    "proofnet":   {"path": "data/ProofNet",      "desc": "本科数学 (分析/代数/拓扑)"},
    "formalmath": {"path": "data/FormalMATH",    "desc": "多领域多难度"},
}

print(f"{'基准':<15} {'题数':>6}  {'难度分布':<35} {'说明'}")
print("─" * 85)
for name, info in datasets.items():
    try:
        problems = load_benchmark(name, path=info["path"]) if info["path"] else load_benchmark(name)
        n = len(problems)
        diffs = {}
        for p in problems:
            d = p.difficulty or "unknown"
            diffs[d] = diffs.get(d, 0) + 1
        diff_str = ", ".join(f"{k}:{v}" for k, v in sorted(diffs.items()))
        print(f"{name:<15} {n:>6}  {diff_str:<35} {info['desc']}")
    except Exception as e:
        print(f"{name:<15}  {'ERR':>5}  {str(e)[:50]:<35} {info['desc']}")
print()
PYEOF
}

# ═══════════════════════════════════════════════════════════════════════
# 第三部分: 运行评测
# ═══════════════════════════════════════════════════════════════════════

run_eval() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  运行评测"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    info "配置:"
    info "  Provider:     $PROVIDER"
    info "  Model:        $MODEL"
    info "  Max samples:  $MAX_SAMPLES"
    info "  Limit:        ${LIMIT:-全部}"
    info "  Lean 验证:    $LEAN_MODE"
    info "  Benchmarks:   $BENCHMARKS"
    echo ""

    if [ "$PROVIDER" = "anthropic" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        fail "ANTHROPIC_API_KEY 未设置, 切换到 mock 模式"
        PROVIDER="mock"
    fi

    local limit_arg=""
    if [ "${LIMIT:-0}" -gt 0 ] 2>/dev/null; then
        limit_arg="--limit $LIMIT"
    fi

    local bench_list
    if [ "$BENCHMARKS" = "all" ]; then
        bench_list="builtin minif2f putnambench proofnet formalmath"
    else
        bench_list="${BENCHMARKS//,/ }"
    fi

    for bench in $bench_list; do
        echo ""
        echo "────────────────────────────────────────────────────────────"
        info "评测: $bench"
        echo "────────────────────────────────────────────────────────────"

        python3 run_eval.py \
            --benchmark "$bench" \
            --provider "$PROVIDER" \
            --model "$MODEL" \
            --max-samples "$MAX_SAMPLES" \
            --lean-mode "$LEAN_MODE" \
            $limit_arg \
            2>&1 || warn "评测 $bench 出现错误"
    done
}

# ═══════════════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          AI4Math — 真实基准评测                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

case "$ACTION" in
    --download)
        download_benchmarks
        ;;
    --eval)
        run_eval
        ;;
    --info)
        show_info
        ;;
    --all|*)
        download_benchmarks
        show_info
        run_eval
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════"
ok "完成！结果保存在 results/ 目录"
echo "════════════════════════════════════════════════════════════════"
