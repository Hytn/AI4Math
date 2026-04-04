"""tests/test_prover/test_verifier.py — 验证器模块测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pytest
from prover.verifier.sorry_detector import detect_sorry, count_sorries, SorryReport
from prover.verifier.goal_extractor import extract_goals, ExtractedGoal
from prover.verifier.error_parser import parse_lean_errors
from prover.verifier.integrity_checker import check_integrity


# ── Sorry Detector ──

class TestSorryDetector:
    def test_clean_code(self):
        report = detect_sorry("theorem t : True := by trivial")
        assert report.is_clean

    def test_detect_sorry(self):
        report = detect_sorry("theorem t : True := by sorry")
        assert report.has_sorry
        assert len(report.locations) == 1
        assert report.locations[0]["keyword"] == "sorry"

    def test_detect_admit(self):
        report = detect_sorry("theorem t : True := by admit")
        assert report.has_sorry

    def test_detect_axiom_warning(self):
        report = detect_sorry("axiom my_axiom : False\ntheorem t : True := by trivial")
        assert not report.has_sorry  # axiom isn't sorry
        assert len(report.warnings) > 0
        assert "axiom" in report.warnings[0].lower()

    def test_detect_native_decide_warning(self):
        report = detect_sorry("theorem t : True := by native_decide")
        assert len(report.warnings) > 0

    def test_detect_unsafe_coerce(self):
        report = detect_sorry("def f := unsafeCoerce 42")
        assert len(report.warnings) > 0

    def test_detect_max_heartbeats_zero(self):
        report = detect_sorry("set_option maxHeartbeats 0\ntheorem t := by sorry")
        assert report.has_sorry
        assert any("maxHeartbeats" in w for w in report.warnings)

    def test_skip_comments(self):
        report = detect_sorry("-- sorry this is a comment\ntheorem t : True := by trivial")
        assert report.is_clean

    def test_count_sorries(self):
        assert count_sorries("sorry\nsorry\nexact h") == 2
        assert count_sorries("exact h") == 0

    def test_multiple_sorries(self):
        code = "theorem t := by\n  sorry\n  sorry\n  sorry"
        report = detect_sorry(code)
        assert len(report.locations) == 3


# ── Goal Extractor ──

class TestGoalExtractor:
    def test_simple_goal(self):
        output = "n : Nat\n⊢ n = n"
        goals = extract_goals(output)
        assert len(goals) >= 1
        assert goals[0].target == "n = n"
        assert len(goals[0].hypotheses) == 1
        assert goals[0].hypotheses[0]["name"] == "n"

    def test_case_goal(self):
        output = "case zero\n⊢ 0 = 0"
        goals = extract_goals(output)
        assert len(goals) >= 1
        assert goals[0].case_name == "zero"

    def test_unsolved_goals(self):
        output = "error: unsolved goals\nn : Nat\nm : Nat\n⊢ n + m = m + n"
        goals = extract_goals(output)
        assert len(goals) >= 1

    def test_format_for_prompt(self):
        goal = ExtractedGoal(
            index=0, target="n = n",
            hypotheses=[{"name": "n", "type": "Nat"}])
        s = goal.to_string()
        assert "n : Nat" in s
        assert "⊢ n = n" in s

    def test_no_goals(self):
        goals = extract_goals("All goals completed!")
        assert goals == []


# ── Error Parser ──

class TestErrorParser:
    def test_parse_type_mismatch(self):
        stderr = "Test.lean:5:4: error: type mismatch\n  expected Nat, got Bool"
        errors = parse_lean_errors(stderr)
        assert len(errors) == 1
        assert errors[0].category.value == "type_mismatch"
        assert errors[0].line == 5

    def test_parse_unknown_identifier(self):
        stderr = "Test.lean:3:10: error: unknown identifier 'foo'"
        errors = parse_lean_errors(stderr)
        assert len(errors) == 1
        assert errors[0].category.value == "unknown_identifier"

    def test_parse_tactic_failed(self):
        stderr = "Test.lean:7:2: error: tactic 'simp' failed"
        errors = parse_lean_errors(stderr)
        assert len(errors) == 1
        assert errors[0].category.value == "tactic_failed"

    def test_parse_multiple_errors(self):
        stderr = ("Test.lean:1:0: error: unknown identifier 'x'\n"
                  "Test.lean:3:0: error: type mismatch at h")
        errors = parse_lean_errors(stderr)
        assert len(errors) == 2


# ── Integrity Checker ──

class TestIntegrityChecker:
    def test_clean_proof(self):
        report = check_integrity("theorem t : True := by trivial")
        assert report.passed

    def test_sorry_fails(self):
        report = check_integrity("theorem t : True := by sorry")
        assert not report.passed
        assert any("sorry" in i.lower() for i in report.issues)

    def test_axiom_fails(self):
        report = check_integrity("axiom my_ax : False")
        assert not report.passed

    def test_debug_commands_warning(self):
        report = check_integrity("#check Nat\ntheorem t : True := by trivial")
        assert report.passed  # non-critical
        assert len(report.issues) > 0
