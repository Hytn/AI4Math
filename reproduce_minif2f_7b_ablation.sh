#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  reproduce_minif2f_7b_ablation.sh
#
#  对 DeepSeek-Prover-V2-7B 做 A/B 消融实验, 验证本框架相对论文的增量。
#
#  ─── Idea ─────────────────────────────────────────────────────────
#  论文 (arXiv:2504.21801, Table 1) 报告 7B 模型的 miniF2F-test:
#       non-CoT  pass@8192 = 75.0%
#       CoT      pass@8192 = 82.0%
#  论文用的是单 profile + 独立 i.i.d. 采样, 没有:
#     (a) verify-and-fix 反馈循环
#     (b) 跨题知识沉淀 (lemma bank, dialog index)
#     (c) 异构并行 (broadcast bus 共享发现)
#     (d) 世界模型 (Qwen / sklearn) 做 tactic gating
#     (e) MCTS / best-first 在 LLM expansion 上做的探索调度
#  这些都是本框架的"开关式"组合点。本脚本把 5 档配置都跑一遍, 对照同
#  一个 7B 模型 + 同一个采样预算, 看哪一档真的胜过论文 baseline。
#
#  ─── 跑法 ─────────────────────────────────────────────────────────
#  
#  先把 DeepSeek-Prover-V2-7B 通过 vLLM 部署起来 (1×80GB H100 即可):
#
#    python -m vllm.entrypoints.openai.api_server \
#        --model deepseek-ai/DeepSeek-Prover-V2-7B \
#        --tensor-parallel-size 1 \
#        --gpu-memory-utilization 0.92 \
#        --max-model-len 32768 \
#        --port 8000 \
#        --served-model-name DSP-V2-7B
#
#  然后:
#    bash reproduce_minif2f_7b_ablation.sh \
#        --api-base http://localhost:8000/v1 \
#        --samples 32        # 先跑 32 看趋势, 再加到 128/512
#
#  脚本会跑 5 档 profile, 每档独立 output 目录, 最后打 A/B 对比表。
#
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'
C='\033[0;36m'; W='\033[1;37m'; N='\033[0m'
section(){ echo -e "\n${W}═══ $* ═══${N}"; }
info()   { echo -e "${C}[INFO]${N} $*"; }
ok()     { echo -e "${G}[ OK ]${N} $*"; }
fail()   { echo -e "${R}[FAIL]${N} $*"; exit 1; }

# ── Defaults ──────────────────────────────────────────────────────────
API_BASE=""
MODEL="DSP-V2-7B"
SAMPLES=32
LIMIT=0
LEAN_MODE="real"
TEMPERATURE=1.0
ROOT="results/dsp_v2_7b_ablation_$(date +%Y%m%d_%H%M%S)"
SMOKE=0
KB_DIR="results/dsp_v2_7b_kb"        # 持久化的跨题知识库 (跨 profile 共享)

while [[ $# -gt 0 ]]; do
    case $1 in
        --api-base)    API_BASE="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --samples)     SAMPLES="$2"; shift 2 ;;
        --limit)       LIMIT="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --no-lean)     LEAN_MODE="skip"; shift ;;
        --smoke)       SMOKE=1; shift ;;
        --root)        ROOT="$2"; shift 2 ;;
        --kb-dir)      KB_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# ═══/,/^# ═══/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) fail "未知参数: $1" ;;
    esac
done

if [ "$SMOKE" -eq 1 ]; then
    info "smoke 模式: provider=mock, backend=mock, lean-mode=skip, samples=1, limit=3"
    API_BASE=""
    SAMPLES=1
    LIMIT=3
    LEAN_MODE="skip"
fi

# 确定 provider
if [ "$SMOKE" -eq 1 ]; then
    PROVIDER="mock"
    BACKEND_ARG="--backend mock"
elif [ -n "$API_BASE" ]; then
    PROVIDER="vllm"
    BACKEND_ARG=""
else
    fail "需要 --api-base http://host:port/v1 (vLLM 部署) 或 --smoke (烟测)"
fi

# 数据准备 (走 reproduce_minif2f.sh 的 step 1-3 即可)
if [ ! -d "data/miniF2F/MiniF2F" ]; then
    section "数据集 + Lean 准备 (走 reproduce_minif2f.sh)"
    bash reproduce_minif2f.sh --smoke 2>&1 | head -20 || true
fi

# 5 档 profile + 1 个论文 baseline 引用
declare -a PROFILES=(
    "dsp_v2_non_cot:论文非CoT baseline (Appendix A.1 prompt)"
    "dsp_v2_cot:论文CoT baseline (Appendix A.2 prompt) — 对照点"
    "dsp_v2_repair:+ verify-and-fix 反馈循环 (本框架增量 #1)"
    "dsp_v2_repair_knowledge:+ 跨题 lemma bank + dialog index (本框架增量 #2)"
    "dsp_v2_heterogeneous:+ 4 路异构并行 + broadcast bus (本框架增量 #3)"
)

LIMIT_ARG=""; [ "${LIMIT:-0}" -gt 0 ] 2>/dev/null && LIMIT_ARG="--limit $LIMIT"
API_BASE_ARG=""; [ -n "$API_BASE" ] && API_BASE_ARG="--api-base $API_BASE"

mkdir -p "$ROOT" "$KB_DIR"

section "Configuration"
info "Provider     : $PROVIDER"
info "Model        : $MODEL"
[ -n "$API_BASE" ] && info "API base     : $API_BASE"
info "Samples (k)  : $SAMPLES"
info "Temperature  : $TEMPERATURE"
info "Lean mode    : $LEAN_MODE"
info "Limit        : ${LIMIT:-全部 244 题}"
info "Output root  : $ROOT"
info "Knowledge DB : $KB_DIR (跨 profile 共享, 在最后两档生效)"

# ── 5-way sweep ──────────────────────────────────────────────────────
for entry in "${PROFILES[@]}"; do
    name="${entry%%:*}"
    desc="${entry#*:}"
    section "Profile: $name  —  $desc"
    OUT="$ROOT/$name"

    # 知识增强 + 异构 profile 共享同一份持久化知识库
    KB_ARGS=""
    if [[ "$name" == "dsp_v2_repair_knowledge" || "$name" == "dsp_v2_heterogeneous" ]]; then
        KB_ARGS="--knowledge-db $KB_DIR/main.sqlite \
                 --dialog-index $KB_DIR/main.sqlite \
                 --lemma-bank-db $KB_DIR/lemmas.sqlite"
    fi

    python3 run_eval.py \
        --benchmark    minif2f \
        --split        test \
        --provider     "$PROVIDER" \
        --model        "$MODEL" \
        --profile      "$name" \
        --max-samples  "$SAMPLES" \
        --temperature  "$TEMPERATURE" \
        --lean-mode    "$LEAN_MODE" \
        --project-dir  "data/miniF2F" \
        --output-dir   "$OUT" \
        --resume \
        $LIMIT_ARG $API_BASE_ARG $BACKEND_ARG $KB_ARGS \
        || { fail "profile=$name 跑失败 (但其他 profile 还会继续)"; }

    # 单独打这一档的 pass@k
    EVAL="$OUT/evals/eval_minif2f_test.json"
    if [ -f "$EVAL" ]; then
        python3 - "$name" "$EVAL" <<'EOF'
import json, sys
name, path = sys.argv[1], sys.argv[2]
e = json.load(open(path))
m = e["metrics"]
print()
print(f"  ── {name} ──")
print(f"  Solved: {m['solved']}/{m['total']} ({m['solve_rate']*100:.1f}%) "
      f"[verification={e.get('verification','?')}]")
ks = sorted(int(k.split('@')[1]) for k in m if isinstance(k, str) and k.startswith('pass@'))
for k in ks:
    print(f"  pass@{k:<6} {m[f'pass@{k}']:.4f}")
EOF
    fi
done

# ── 汇总对比表 ────────────────────────────────────────────────────────
section "A/B 对比 — DeepSeek-Prover-V2-7B 在 miniF2F-test"
echo
echo "  论文 baseline (arXiv:2504.21801, Table 1, 7B 行):"
echo "    non-CoT pass@1=55.5%, pass@32=68.0%, pass@1024=73.2%, pass@8192=75.0%"
echo "    CoT     pass@1=58.6%, pass@32=75.6%, pass@1024=79.9%, pass@8192=82.0%"
echo
echo "  本次 sweep 的 5 档 (--samples=$SAMPLES):"
echo

python3 - <<EOF
import json, os
ROOT = "$ROOT"
profiles = [
    ("dsp_v2_non_cot",            "论文非CoT baseline"),
    ("dsp_v2_cot",                "论文CoT baseline"),
    ("dsp_v2_repair",             "+repair (增量1)"),
    ("dsp_v2_repair_knowledge",   "+repair+knowledge (增量2)"),
    ("dsp_v2_heterogeneous",      "+heterogeneous (增量3)"),
]
header = f"  {'Profile':<28} {'Solved':>9} {'pass@1':>8} {'pass@k':>8}"
print(header); print("  " + "─"*55)
for prof, desc in profiles:
    p = os.path.join(ROOT, prof, "evals", "eval_minif2f_test.json")
    if not os.path.exists(p):
        print(f"  {prof:<28} {'(not run)':>9}")
        continue
    e = json.load(open(p))
    m = e["metrics"]
    ks = sorted(int(k.split('@')[1]) for k in m if isinstance(k,str) and k.startswith('pass@'))
    pk_max = ks[-1] if ks else 1
    print(f"  {prof:<28} {m['solved']:>4}/{m['total']:<3}  "
          f"{m.get('pass@1',0):>7.4f}  pass@{pk_max:<2}={m[f'pass@{pk_max}']:.4f}")
print()
print("  关键问题: 哪一档的 pass@$SAMPLES 已经 ≥ 论文 7B 同 budget 的 pass@$SAMPLES?")
print("    论文 7B CoT pass@32 = 75.6%   (Table 1)")
print("    本次实验的 pass@$SAMPLES 见上表 pass@k 列。")
EOF

ok "完成。每档完整 dialog 在 $ROOT/<profile>/traces/minif2f/<id>/dialog.json"
ok "想加 sample, 重跑这个脚本 (--resume 自动跳过已完成的题)"
