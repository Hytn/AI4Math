"""tests/test_prover/test_verifier.py — 验证器模块测试

v11: ``goal_extractor`` and ``error_parser`` were deleted (zero main-path
callers, replaced by ``engine.error_intelligence``). Their TestGoalExtractor
and TestErrorParser classes have been removed along with them.
``sorry_detector`` and ``integrity_checker`` are kept (the former is wired
into AgentLoop; the latter is wired into LeanVerifyTool as of v11).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prover.verifier.sorry_detector import detect_sorry, count_sorries
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


# ── Integrity Checker ──

class TestIntegrityChecker:
    def test_clean_proof(self):
        report = check_integrity("theorem t : True := by trivial")
        assert report.passed

    def test_sorry_fails(self):
        report = check_integrity("theorem t : True := by sorry")
        assert not report.passed
        assert any("sorry" in i.message.lower() for i in report.issues)

    def test_axiom_fails(self):
        report = check_integrity("axiom my_ax : False")
        assert not report.passed

    def test_debug_commands_warning(self):
        report = check_integrity("#check Nat\ntheorem t : True := by trivial")
        assert report.passed  # non-critical
        assert len(report.issues) > 0
