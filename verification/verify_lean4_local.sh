#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# AI4Math — Lean4 本地环境验证脚本
# ═══════════════════════════════════════════════════════════════
#
# 使用方法:
#   cd project-fixed
#   bash verification/verify_lean4_local.sh
#
# 前置要求:
#   - elan 已安装 (curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh)
#   - lean4 已安装 (elan default leanprover/lean4:stable)
#   - 可选: lake + Mathlib 项目已配置
#
# 输出: verification/lean4_report.json
# ═══════════════════════════════════════════════════════════════

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPORT_FILE="$SCRIPT_DIR/lean4_report.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

passed=0
failed=0
total=0
results="[]"

add_result() {
    local name="$1" status="$2" details="$3"
    total=$((total + 1))
    if [ "$status" = "pass" ]; then
        passed=$((passed + 1))
        echo -e "  ${GREEN}✓${NC} $name — $details"
    else
        failed=$((failed + 1))
        echo -e "  ${RED}✗${NC} $name — $details"
    fi
    results=$(echo "$results" | python3 -c "
import sys, json
r = json.load(sys.stdin)
r.append({'name': '$name', 'passed': '$status' == 'pass', 'details': '''$details'''})
json.dump(r, sys.stdout)
")
}

echo "═══════════════════════════════════════════════════════════════"
echo "  AI4Math — Lean4 本地环境验证"
echo "═══════════════════════════════════════════════════════════════"

# ── Section 1: 环境检测 ──
echo ""
echo "Section 1: 环境检测"
echo "───────────────────────────────────────────────────────────────"

# Check elan
if command -v elan &>/dev/null; then
    ELAN_VERSION=$(elan --version 2>/dev/null || echo "unknown")
    add_result "elan installed" "pass" "$ELAN_VERSION"
else
    add_result "elan installed" "fail" "elan not found. Install: curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh"
fi

# Check lean
if command -v lean &>/dev/null; then
    LEAN_VERSION=$(lean --version 2>/dev/null | head -1 || echo "unknown")
    add_result "lean4 installed" "pass" "$LEAN_VERSION"
else
    add_result "lean4 installed" "fail" "lean not found. Run: elan default leanprover/lean4:stable"
fi

# Check lake
if command -v lake &>/dev/null; then
    LAKE_VERSION=$(lake --version 2>/dev/null | head -1 || echo "unknown")
    add_result "lake installed" "pass" "$LAKE_VERSION"
else
    add_result "lake installed" "fail" "lake not found"
fi

# ── Section 2: 基本编译验证 ──
echo ""
echo "Section 2: 基本 Lean4 编译验证"
echo "───────────────────────────────────────────────────────────────"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Test 1: Simple theorem
cat > "$TMPDIR/test1.lean" << 'EOF'
theorem test_true : True := trivial
theorem test_rfl : 1 = 1 := rfl
theorem test_intro : ∀ (P : Prop), P → P := fun _ h => h
#check test_true
#check test_rfl
#check test_intro
EOF

if lean "$TMPDIR/test1.lean" 2>/dev/null; then
    add_result "Simple theorems compile" "pass" "True, rfl, intro all pass"
else
    add_result "Simple theorems compile" "fail" "$(lean "$TMPDIR/test1.lean" 2>&1 | head -3)"
fi

# Test 2: Nat induction
cat > "$TMPDIR/test2.lean" << 'EOF'
theorem nat_add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.succ_add, ih]
EOF

if lean "$TMPDIR/test2.lean" 2>/dev/null; then
    add_result "Nat induction compiles" "pass" "n + 0 = n by induction"
else
    add_result "Nat induction compiles" "fail" "$(lean "$TMPDIR/test2.lean" 2>&1 | head -3)"
fi

# Test 3: Sorry detection
cat > "$TMPDIR/test3.lean" << 'EOF'
theorem has_sorry : 1 + 1 = 3 := by sorry
EOF

if lean "$TMPDIR/test3.lean" 2>&1 | grep -qi "sorry"; then
    add_result "Sorry detection works" "pass" "Lean4 reports sorry usage"
else
    add_result "Sorry detection works" "fail" "sorry not detected"
fi

# Test 4: Error reporting
cat > "$TMPDIR/test4.lean" << 'EOF'
theorem type_error : Nat := "hello"
EOF

ERROR_OUTPUT=$(lean "$TMPDIR/test4.lean" 2>&1 || true)
if echo "$ERROR_OUTPUT" | grep -qi "mismatch\|error"; then
    add_result "Error reporting works" "pass" "type mismatch detected"
else
    add_result "Error reporting works" "fail" "no error reported"
fi

# Test 5: stdin compilation (used by REPL)
STDIN_RESULT=$(echo 'theorem stdin_test : True := trivial' | lean --stdin 2>&1 || true)
if ! echo "$STDIN_RESULT" | grep -qi "error"; then
    add_result "stdin compilation" "pass" "lean --stdin works"
else
    add_result "stdin compilation" "fail" "$STDIN_RESULT"
fi

# ── Section 3: Mathlib 可用性 ──
echo ""
echo "Section 3: Mathlib 可用性 (可选)"
echo "───────────────────────────────────────────────────────────────"

# Check if a Mathlib project exists
MATHLIB_PROJECT=""
for candidate in "$PROJECT_DIR/lean-project" "$HOME/.ai4math/lean-project" "$PROJECT_DIR/data/lean-project"; do
    if [ -f "$candidate/lakefile.lean" ] || [ -f "$candidate/lakefile.toml" ]; then
        MATHLIB_PROJECT="$candidate"
        break
    fi
done

if [ -n "$MATHLIB_PROJECT" ]; then
    add_result "Mathlib project found" "pass" "$MATHLIB_PROJECT"
    
    # Try compiling with Mathlib
    cat > "$TMPDIR/test_mathlib.lean" << 'EOF'
import Mathlib.Tactic

theorem mathlib_test : ∀ (n m : Nat), n + m = m + n := by
  intros; omega

theorem ring_test (a b : Int) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by
  ring
EOF

    if timeout 120 lake env lean "$TMPDIR/test_mathlib.lean" 2>/dev/null; then
        add_result "Mathlib compilation" "pass" "omega + ring work"
    else
        add_result "Mathlib compilation" "fail" "timeout or error (Mathlib may need building: lake build)"
    fi
else
    add_result "Mathlib project found" "fail" "No lakefile found. Set up with: lake init AI4Math && lake add mathlib"
    echo -e "  ${YELLOW}⚠ Skipping Mathlib-dependent tests${NC}"
fi

# ── Section 4: REPL 交互验证 ──
echo ""
echo "Section 4: REPL 交互模式"
echo "───────────────────────────────────────────────────────────────"

# Test incremental tactic verification
cat > "$TMPDIR/test_repl.lean" << 'EOF'
theorem repl_test (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q := by
  constructor
  · exact hp
  · exact hq
EOF

if lean "$TMPDIR/test_repl.lean" 2>/dev/null; then
    add_result "Multi-tactic proof" "pass" "constructor + exact"
else
    add_result "Multi-tactic proof" "fail" "$(lean "$TMPDIR/test_repl.lean" 2>&1 | head -3)"
fi

# Test tactic error feedback
cat > "$TMPDIR/test_feedback.lean" << 'EOF'
theorem feedback_test (n : Nat) : n = n + 1 := by
  omega
EOF

FEEDBACK=$(lean "$TMPDIR/test_feedback.lean" 2>&1 || true)
if echo "$FEEDBACK" | grep -qi "error\|failed\|omega"; then
    add_result "Tactic error feedback" "pass" "omega correctly fails on false goal"
else
    add_result "Tactic error feedback" "fail" "no useful feedback"
fi

# ── Section 5: 性能基准 ──
echo ""
echo "Section 5: Lean4 编译性能"
echo "───────────────────────────────────────────────────────────────"

# Measure simple compile time
cat > "$TMPDIR/perf_simple.lean" << 'EOF'
theorem perf1 : True := trivial
theorem perf2 : 1 + 1 = 2 := rfl
theorem perf3 (P : Prop) : P → P := fun h => h
EOF

COMPILE_START=$(date +%s%N)
lean "$TMPDIR/perf_simple.lean" 2>/dev/null
COMPILE_END=$(date +%s%N)
COMPILE_MS=$(( (COMPILE_END - COMPILE_START) / 1000000 ))
add_result "Simple compile latency" "pass" "${COMPILE_MS}ms for 3 theorems"

# ── Summary ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  LEAN4 VERIFICATION SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo -e "  Passed: ${GREEN}${passed}${NC}/${total}"
echo -e "  Failed: ${RED}${failed}${NC}/${total}"
echo "═══════════════════════════════════════════════════════════════"

# Save JSON report
python3 -c "
import json, sys
results = json.loads('''$results''')
report = {
    'test': 'lean4_local_verification',
    'total': $total,
    'passed': $passed,
    'failed': $failed,
    'results': results
}
with open('$REPORT_FILE', 'w') as f:
    json.dump(report, f, indent=2)
print(f'  Report saved to: $REPORT_FILE')
"

exit $failed
