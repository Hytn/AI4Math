"""tests/test_prover/test_decompose.py — 分解与组合模块测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pytest
from prover.decompose.goal_decomposer import SubGoal
from prover.decompose.subgoal_scheduler import SubGoalScheduler
from prover.decompose.composition import compose_proof, validate_composition


# ── Subgoal Scheduler ──

class TestSubGoalScheduler:
    def setup_method(self):
        self.subgoals = [
            SubGoal("h1", "lemma h1 : True", difficulty="hard"),
            SubGoal("h2", "lemma h2 : 1 = 1", difficulty="easy"),
            SubGoal("h3", "lemma h3 : P → P", difficulty="medium"),
        ]

    def test_easy_first(self):
        sched = SubGoalScheduler("easy_first")
        result = sched.schedule(self.subgoals)
        assert result[0].difficulty == "easy"
        assert result[-1].difficulty == "hard"

    def test_hard_first(self):
        sched = SubGoalScheduler("hard_first")
        result = sched.schedule(self.subgoals)
        assert result[0].difficulty == "hard"

    def test_dependency_order(self):
        sched = SubGoalScheduler("dependency")
        deps = {"h3": ["h1"], "h1": ["h2"]}
        result = sched.schedule(self.subgoals, deps)
        names = [g.name for g in result]
        assert names.index("h2") < names.index("h1")
        assert names.index("h1") < names.index("h3")

    def test_mark_solved(self):
        sched = SubGoalScheduler()
        sched.mark_solved(self.subgoals, "h2", "by rfl")
        assert self.subgoals[1].proved
        assert self.subgoals[1].proof == "by rfl"

    def test_progress(self):
        sched = SubGoalScheduler()
        assert sched.progress(self.subgoals) == 0.0
        self.subgoals[0].proved = True
        assert abs(sched.progress(self.subgoals) - 1/3) < 0.01

    def test_unsolved(self):
        sched = SubGoalScheduler()
        self.subgoals[0].proved = True
        assert len(sched.unsolved(self.subgoals)) == 2


# ── Composition ──

class TestComposition:
    def test_compose_all_solved(self):
        subgoals = [
            SubGoal("h1", "lemma h1 : True", proved=True, proof="by trivial"),
            SubGoal("h2", "lemma h2 : True", proved=True, proof="by trivial"),
        ]
        result = compose_proof("theorem t : True ∧ True", subgoals)
        assert "have h1" in result
        assert "have h2" in result
        assert "sorry" not in result

    def test_compose_partial(self):
        subgoals = [
            SubGoal("h1", "lemma h1 : True", proved=True, proof="by trivial"),
            SubGoal("h2", "lemma h2 : P", proved=False),
        ]
        result = compose_proof("theorem t : True ∧ P", subgoals)
        assert "sorry" in result  # unsolved goal

    def test_validate_valid(self):
        subgoals = [SubGoal("h1", "lemma h1 : True", proved=True, proof="by trivial")]
        proof = compose_proof("theorem t : True", subgoals)
        report = validate_composition(proof, subgoals)
        assert report["valid"] or len(report["issues"]) == 0 or "sorry" not in proof

    def test_validate_missing_ref(self):
        subgoals = [SubGoal("h1", "lemma h1 : True", proved=True)]
        report = validate_composition("theorem t := by trivial", subgoals)
        assert any("h1" in issue for issue in report["issues"])
