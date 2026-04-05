"""tests/test_p1_fixes.py — Tests for P1-level improvements

Covers:
  1. Enhanced IntegrityChecker (native_decide, maxHeartbeats, unsafe, etc.)
  2. Expanded premise knowledge base (external JSONL loading)
  3. Thread safety in SearchCoordinator._backpropagate
  4. Exception logging in Orchestrator (no silent swallowing)
"""
import sys
import os
import json
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


# ═══════════════════════════════════════════════════════════════
# 1. Enhanced IntegrityChecker
# ═══════════════════════════════════════════════════════════════

class TestIntegrityCheckerV2:
    def test_sorry_critical(self):
        from prover.verifier.integrity_checker import check_integrity, Severity
        report = check_integrity("theorem t : True := by sorry")
        assert not report.passed
        assert any(i.severity == Severity.CRITICAL for i in report.issues)

    def test_admit_critical(self):
        from prover.verifier.integrity_checker import check_integrity
        report = check_integrity("theorem t : True := by admit")
        assert not report.passed

    def test_native_decide_critical(self):
        from prover.verifier.integrity_checker import check_integrity, Severity
        code = "theorem t : 2 + 2 = 4 := by native_decide"
        report = check_integrity(code)
        assert not report.passed
        assert any("native_decide" in i.message for i in report.issues)
        assert any(i.severity == Severity.CRITICAL for i in report.issues)

    def test_max_heartbeats_zero_critical(self):
        from prover.verifier.integrity_checker import check_integrity
        code = "set_option maxHeartbeats 0\ntheorem t : True := by trivial"
        report = check_integrity(code)
        assert not report.passed
        assert any("heartbeat" in i.message.lower() for i in report.issues)

    def test_unsafe_def_critical(self):
        from prover.verifier.integrity_checker import check_integrity
        code = "unsafe def cheat : False := sorry\ntheorem t : True := by trivial"
        report = check_integrity(code)
        assert not report.passed

    def test_implemented_by_critical(self):
        from prover.verifier.integrity_checker import check_integrity
        code = '@[implemented_by cheat_impl]\ndef f := 42'
        report = check_integrity(code)
        assert not report.passed

    def test_extern_critical(self):
        from prover.verifier.integrity_checker import check_integrity
        code = '@[extern "lean_io_hack"]\ndef f := 42'
        report = check_integrity(code)
        assert not report.passed

    def test_decreasing_by_sorry_critical(self):
        from prover.verifier.integrity_checker import check_integrity
        code = "def f (n : Nat) := f (n - 1)\ndecreasing_by sorry"
        report = check_integrity(code)
        assert not report.passed

    def test_import_lean_warning(self):
        from prover.verifier.integrity_checker import check_integrity, Severity
        code = "import Lean\nimport Mathlib\ntheorem t : True := by trivial"
        report = check_integrity(code)
        # Should pass (it's a warning, not critical)
        assert report.passed
        assert any(i.severity == Severity.WARNING for i in report.issues)
        assert any("Lean" in i.message for i in report.issues)

    def test_run_tac_warning(self):
        from prover.verifier.integrity_checker import check_integrity, Severity
        code = "theorem t : True := by run_tac do pure ()"
        report = check_integrity(code)
        # run_tac is a warning, not critical
        assert any(i.severity == Severity.WARNING for i in report.issues)

    def test_debug_commands_info(self):
        from prover.verifier.integrity_checker import check_integrity, Severity
        code = "#check Nat\n#eval 42\ntheorem t : True := by trivial"
        report = check_integrity(code)
        assert report.passed
        assert any(i.severity == Severity.INFO for i in report.issues)

    def test_clean_proof_passes(self):
        from prover.verifier.integrity_checker import check_integrity
        code = "import Mathlib\n\ntheorem t (n : Nat) : n + 0 = n := by simp"
        report = check_integrity(code)
        assert report.passed
        assert len(report.critical_issues) == 0

    def test_sorry_in_comment_ok(self):
        from prover.verifier.integrity_checker import check_integrity
        code = "-- sorry is just a comment\ntheorem t : True := by trivial"
        report = check_integrity(code)
        assert report.passed

    def test_sorry_in_block_comment_ok(self):
        from prover.verifier.integrity_checker import check_integrity
        code = "/- sorry in block comment -/\ntheorem t : True := by trivial"
        report = check_integrity(code)
        assert report.passed

    def test_statement_modification_warning(self):
        from prover.verifier.integrity_checker import check_integrity
        code = "theorem t : False := by trivial"
        report = check_integrity(code, original_statement="theorem t : True")
        assert any("modified" in i.message.lower() for i in report.issues)

    def test_summary(self):
        from prover.verifier.integrity_checker import check_integrity
        r1 = check_integrity("theorem t : True := by trivial")
        assert "PASSED" in r1.summary()

        r2 = check_integrity("theorem t : True := by sorry")
        assert "FAILED" in r2.summary()

    def test_large_heartbeats_warning(self):
        from prover.verifier.integrity_checker import check_integrity, Severity
        code = "set_option maxHeartbeats 99999999\ntheorem t : True := by trivial"
        report = check_integrity(code)
        # Very large but not zero → warning, not critical
        # (maxHeartbeats 0 is critical, large value is warning)
        has_heartbeat_issue = any("heartbeat" in i.message.lower() for i in report.issues)
        assert has_heartbeat_issue


# ═══════════════════════════════════════════════════════════════
# 2. Expanded premise knowledge base
# ═══════════════════════════════════════════════════════════════

class TestExpandedPremises:
    def test_external_premises_loaded(self):
        """PremiseSelector should load from data/premises/*.jsonl."""
        from prover.premise.selector import PremiseSelector

        selector = PremiseSelector({"mode": "hybrid"})
        # The 334 external premises should be loaded
        assert selector.size > 100  # Was 71, now should be 300+

    def test_retrieve_nat_subtraction(self):
        """Should retrieve ℕ subtraction lemmas for subtraction queries."""
        from prover.premise.selector import PremiseSelector

        selector = PremiseSelector({"mode": "bm25"})
        results = selector.retrieve(
            "theorem t (n m : Nat) : n - m + m = n", top_k=10)
        names = [r["name"] for r in results]
        # Should find the critical subtraction lemmas
        assert any("sub" in n.lower() for n in names)

    def test_retrieve_group_lemmas(self):
        """Should retrieve group theory lemmas for group queries."""
        from prover.premise.selector import PremiseSelector

        selector = PremiseSelector({"mode": "bm25"})
        results = selector.retrieve(
            "theorem t [Group G] (a : G) : a * a⁻¹ = 1", top_k=10)
        names = [r["name"] for r in results]
        assert any("inv" in n.lower() or "cancel" in n.lower() for n in names)

    def test_external_jsonl_format(self):
        """External JSONL should have name, statement, domain fields."""
        import json
        jsonl_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "premises", "mathlib_core.jsonl")
        if not os.path.exists(jsonl_path):
            pytest.skip("mathlib_core.jsonl not found")

        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                assert "name" in entry, f"Line {i}: missing 'name'"
                assert "statement" in entry, f"Line {i}: missing 'statement'"
                assert "domain" in entry, f"Line {i}: missing 'domain'"
                if i > 20:
                    break  # Spot check first 20

    def test_custom_premise_dir(self):
        """PremiseSelector should support custom premise_dirs."""
        from prover.premise.selector import PremiseSelector

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a custom premise file
            custom_path = os.path.join(tmpdir, "custom.jsonl")
            with open(custom_path, "w") as f:
                f.write(json.dumps({
                    "name": "custom_lemma_xyz",
                    "statement": "∀ (x : ℕ), x + x = 2 * x",
                    "domain": "nat",
                }) + "\n")

            selector = PremiseSelector({
                "mode": "bm25",
                "premise_dirs": [tmpdir],
            })
            # Query should match the premise's statement content
            results = selector.retrieve("x + x = 2 * x", top_k=50)
            names = [r["name"] for r in results]
            assert "custom_lemma_xyz" in names

    def test_domain_coverage(self):
        """Premises should cover all major mathematical domains."""
        import json
        jsonl_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "premises", "mathlib_core.jsonl")
        if not os.path.exists(jsonl_path):
            pytest.skip("mathlib_core.jsonl not found")

        domains = set()
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                domains.add(entry.get("domain", ""))

        # Should cover at least these critical domains
        for required in ["nat", "int", "real", "logic", "algebra",
                         "finset", "set", "topology", "analysis"]:
            assert required in domains, f"Missing domain: {required}"


# ═══════════════════════════════════════════════════════════════
# 3. Thread safety in SearchCoordinator._backpropagate
# ═══════════════════════════════════════════════════════════════

class TestBackpropagateThreadSafety:
    def test_concurrent_backpropagation(self):
        """Multiple threads calling _backpropagate should not lose updates."""
        from engine.core import Expr, Name, BinderInfo
        from engine.core.universe import Level
        from engine.core.environment import Environment, ConstantInfo
        from engine.search import SearchCoordinator, SearchConfig

        env = Environment()
        prop = Expr.sort(Level.zero())
        type_ = Expr.sort(Level.one())
        env = env.add_const(ConstantInfo(Name.from_str("Prop"), type_))
        env = env.add_const(ConstantInfo(Name.from_str("Nat"), type_))
        nat = Expr.const(Name.from_str("Nat"))
        env = env.add_const(ConstantInfo(Name.from_str("Nat.zero"), nat))
        env = env.add_const(ConstantInfo(
            Name.from_str("Nat.succ"), Expr.arrow(nat, nat)))

        goal = Expr.pi(BinderInfo.DEFAULT, Name.from_str("n"), nat, prop)
        config = SearchConfig(strategy="best_first", max_nodes=1000)
        coordinator = SearchCoordinator(env, goal, config)

        # Expand root with multiple tactics to create child nodes
        coordinator.try_tactic(0, "intro n")
        coordinator.try_tactic(0, "sorry")

        # Now backpropagate from multiple threads concurrently
        errors = []
        n_threads = 10

        def backprop_worker(success):
            try:
                coordinator._backpropagate(0, success)
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=backprop_worker, args=(i % 2 == 0,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5)

        # No errors should have occurred
        assert len(errors) == 0, f"Thread errors: {errors}"

        # Root node should have been visited n_threads times
        from engine.state.search_tree import NodeId
        root = coordinator._tree.get(NodeId(0))
        assert root is not None
        # Due to locking, visit count should be exactly n_threads
        # (if no lock, it could be less due to lost updates)
        assert root.visit_count >= n_threads


# ═══════════════════════════════════════════════════════════════
# 4. Exception logging (no silent except:pass)
# ═══════════════════════════════════════════════════════════════

class TestExceptionLogging:
    def test_orchestrator_has_no_bare_except_pass(self):
        """Orchestrator should not have bare 'except: pass' anymore."""
        import ast
        orch_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prover", "pipeline", "orchestrator.py")
        with open(orch_path) as f:
            source = f.read()

        # Parse the AST and check for bare except handlers
        tree = ast.parse(source)
        bare_passes = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                # Check if the handler body is just `pass`
                if (len(node.body) == 1 and
                        isinstance(node.body[0], ast.Pass) and
                        node.type is None):
                    bare_passes += 1

        assert bare_passes == 0, (
            f"Found {bare_passes} bare 'except: pass' in orchestrator.py. "
            "All exceptions should be logged.")

    def test_orchestrator_exceptions_have_names(self):
        """All except handlers should capture the exception variable."""
        import ast
        orch_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prover", "pipeline", "orchestrator.py")
        with open(orch_path) as f:
            source = f.read()

        tree = ast.parse(source)
        unnamed = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is not None and node.name is None:
                    # Has a type (e.g., Exception) but no name (no `as e`)
                    unnamed += 1

        assert unnamed == 0, (
            f"Found {unnamed} exception handlers without a name variable. "
            "Use 'except Exception as e:' to enable logging.")
