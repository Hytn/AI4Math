"""tests/test_prover/test_repair.py — 修复模块测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pytest
from prover.models import LeanError, ErrorCategory
from prover.repair.repair_strategies import (
    select_strategies, build_repair_prompt, RepairStrategy, STRATEGIES)
from prover.repair.patch_applier import (
    apply_patches, Patch, create_line_replace_patch,
    create_tactic_replace_patch, create_full_replace_patch, diff_proofs)
from prover.repair.repair_generator import _fix_syntax, _fix_identifier


# ── Repair Strategies ──

class TestRepairStrategies:
    def test_select_for_type_mismatch(self):
        errors = [LeanError(ErrorCategory.TYPE_MISMATCH, "type mismatch")]
        strategies = select_strategies(errors)
        assert len(strategies) > 0
        assert any(ErrorCategory.TYPE_MISMATCH in s.applicable_errors for s in strategies)

    def test_select_for_unknown_identifier(self):
        errors = [LeanError(ErrorCategory.UNKNOWN_IDENTIFIER, "unknown 'foo'")]
        strategies = select_strategies(errors)
        assert any("fix_identifier" == s.name for s in strategies)

    def test_select_for_mixed_errors(self):
        errors = [
            LeanError(ErrorCategory.TYPE_MISMATCH, "mismatch"),
            LeanError(ErrorCategory.TACTIC_FAILED, "tactic failed"),
        ]
        strategies = select_strategies(errors, max_strategies=3)
        assert len(strategies) <= 3

    def test_empty_errors(self):
        strategies = select_strategies([])
        assert strategies == []

    def test_always_includes_fallback(self):
        errors = [LeanError(ErrorCategory.TYPE_MISMATCH, "mismatch")]
        strategies = select_strategies(errors, max_strategies=10)
        assert any(s.name == "try_alternative_approach" for s in strategies)

    def test_build_repair_prompt(self):
        errors = [LeanError(ErrorCategory.TYPE_MISMATCH, "expected Nat got Bool", line=5)]
        strategies = select_strategies(errors)
        prompt = build_repair_prompt(errors, strategies)
        assert "type_mismatch" in prompt
        assert "line 5" in prompt


# ── Patch Applier ──

class TestPatchApplier:
    def test_replace_line(self):
        original = "line1\nline2\nline3"
        patches = [create_line_replace_patch(2, "REPLACED")]
        result = apply_patches(original, patches)
        assert "REPLACED" in result
        assert "line2" not in result

    def test_replace_tactic(self):
        original = "by\n  sorry\n  exact h"
        patches = [create_tactic_replace_patch("sorry", "trivial")]
        result = apply_patches(original, patches)
        assert "trivial" in result
        assert "sorry" not in result

    def test_full_replace(self):
        result = apply_patches("old code", [create_full_replace_patch("new code")])
        assert result == "new code"

    def test_no_patches(self):
        assert apply_patches("original", []) == "original"

    def test_multiple_patches(self):
        original = "a\nb\nc"
        patches = [
            create_line_replace_patch(1, "A"),
            create_line_replace_patch(3, "C"),
        ]
        result = apply_patches(original, patches)
        assert "A" in result and "C" in result and "b" in result

    def test_diff_proofs(self):
        changes = diff_proofs("line1\nline2", "line1\nchanged")
        assert len(changes) == 1
        assert changes[0]["line"] == 2
        assert changes[0]["kind"] == "modified"


# ── Quick fix helpers ──

class TestQuickFix:
    def test_fix_unbalanced_brackets(self):
        result = _fix_syntax("exact (f (g x", LeanError(ErrorCategory.SYNTAX_ERROR, ""))
        assert result.count("(") == result.count(")")

    def test_fix_lean4_identifier(self):
        error = LeanError(ErrorCategory.UNKNOWN_IDENTIFIER, "unknown identifier 'nat.add_comm'")
        result = _fix_identifier("apply nat.add_comm", error)
        # Should try to fix casing
        assert result  # at least returns something
