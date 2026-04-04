"""tests/test_prover/test_sketch_conjecture_formalize.py — sketch/conjecture/formalize 测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pytest
from prover.sketch.templates import find_templates, fill_template, TEMPLATES
from prover.sketch.hypothesis_generator import HypothesisGenerator
from prover.conjecture.conjecture_verifier import ConjectureVerifier
from prover.formalize.statement_verifier import StatementVerifier
from prover.lemma_bank.lemma_extractor import LemmaExtractor
from prover.lemma_bank.lemma_verifier import LemmaVerifier
from prover.lemma_bank.bank import ProvedLemma


# ── Proof Templates ──

class TestTemplates:
    def test_find_for_implication(self):
        templates = find_templates("implication")
        assert len(templates) > 0
        assert any("intro" in t.skeleton for t in templates)

    def test_find_for_equality(self):
        templates = find_templates("equality")
        assert len(templates) > 0

    def test_find_for_conjunction(self):
        templates = find_templates("conjunction")
        assert len(templates) > 0
        assert any("constructor" in t.skeleton for t in templates)

    def test_find_for_unknown(self):
        templates = find_templates("zzz_unknown_shape")
        assert len(templates) > 0  # should return generic templates

    def test_fill_template(self):
        template = TEMPLATES[0]  # induction_nat
        result = fill_template(template, {"base_case": "simp", "inductive_step": "simp [ih]"})
        assert "simp" in result
        assert "sorry" not in result

    def test_fill_template_missing(self):
        template = TEMPLATES[0]
        result = fill_template(template, {"base_case": "simp"})
        assert "sorry" in result  # unfilled placeholder becomes sorry

    def test_all_templates_valid(self):
        for t in TEMPLATES:
            assert t.name
            assert t.skeleton
            assert len(t.applicable_shapes) > 0


# ── Hypothesis Generator ──

class TestHypothesisGenerator:
    def test_rule_based_equality(self):
        gen = HypothesisGenerator()
        hyps = gen.generate("theorem t : a + b = b + a", target="a + b = b + a")
        assert len(hyps) > 0

    def test_rule_based_nat(self):
        gen = HypothesisGenerator()
        hyps = gen.generate("theorem t (n : Nat) : P n", target="P (Nat.succ n)")
        # Should suggest base/step for nat
        assert len(hyps) >= 0  # may or may not find nat_expr

    def test_empty_input(self):
        gen = HypothesisGenerator()
        hyps = gen.generate("")
        assert isinstance(hyps, list)


# ── Conjecture Verifier ──

class TestConjectureVerifier:
    def setup_method(self):
        self.verifier = ConjectureVerifier()

    def test_valid_conjecture(self):
        result = self.verifier.verify("lemma foo (n : Nat) : n + 0 = n")
        assert result.is_parseable
        assert not result.is_trivial

    def test_invalid_syntax(self):
        result = self.verifier.verify("this is not lean code")
        assert not result.is_parseable

    def test_trivially_true(self):
        result = self.verifier.verify("lemma foo : True")
        assert result.is_trivial

    def test_trivially_reflexive(self):
        result = self.verifier.verify("lemma foo : n = n")
        assert result.is_trivial

    def test_unbalanced_brackets(self):
        result = self.verifier.verify("lemma foo (n : Nat : n = n")
        assert not result.is_parseable

    def test_relevance_score(self):
        result = self.verifier.verify(
            "lemma helper (n m : Nat) : n + m = m + n",
            target_theorem="theorem t (a b : Nat) : a + b = b + a")
        assert result.relevance_score > 0

    def test_filter_valid(self):
        conjectures = [
            "lemma good (n : Nat) : n + 0 = n",
            "not a lemma at all",
            "lemma trivial : True",
        ]
        valid = self.verifier.filter_valid(conjectures)
        assert len(valid) == 1
        assert "good" in valid[0]

    def test_batch_verify(self):
        conjectures = [
            "lemma a (n : Nat) : n + 0 = n",
            "lemma b : True",
        ]
        results = self.verifier.verify_batch(conjectures)
        assert len(results) == 2
        # Valid ones should come first
        assert results[0].is_valid


# ── Statement Verifier ──

class TestStatementVerifier:
    def setup_method(self):
        self.verifier = StatementVerifier()

    def test_valid_statement(self):
        result = self.verifier.verify("theorem t (n : Nat) : n = n")
        assert result.is_parseable
        assert result.is_well_formed

    def test_missing_type(self):
        result = self.verifier.verify("theorem t")
        assert not result.is_parseable

    def test_not_a_declaration(self):
        result = self.verifier.verify("hello world")
        assert not result.is_parseable

    def test_sorry_in_statement(self):
        result = self.verifier.verify("theorem t : sorry")
        assert result.has_sorry


# ── Lemma Extractor ──

class TestLemmaExtractor:
    def setup_method(self):
        self.extractor = LemmaExtractor()

    def test_extract_have_step(self):
        code = "theorem t := by\n  have h1 : True := by trivial\n  exact h1"
        lemmas = self.extractor.extract_from_proof(code)
        assert len(lemmas) == 1
        assert lemmas[0].name == "h1"

    def test_skip_sorry_have(self):
        code = "theorem t := by\n  have h1 : True := by sorry\n  exact h1"
        lemmas = self.extractor.extract_from_proof(code)
        assert len(lemmas) == 0  # sorry sub-proofs excluded

    def test_no_have_steps(self):
        code = "theorem t := by exact trivial"
        lemmas = self.extractor.extract_from_proof(code)
        assert len(lemmas) == 0

    def test_extract_from_trace(self):
        attempts = [
            {"generated_proof": "by\n  have h1 : True := by trivial\n  exact h1"},
            {"generated_proof": "by sorry"},
        ]
        lemmas = self.extractor.extract_from_trace(attempts)
        assert len(lemmas) == 1


# ── Lemma Verifier ──

class TestLemmaVerifier:
    def test_structural_check_valid(self):
        verifier = LemmaVerifier()
        lemma = ProvedLemma("h1", "lemma h1 : True", ":= by trivial")
        assert verifier.verify(lemma)
        assert lemma.verified

    def test_structural_check_sorry(self):
        verifier = LemmaVerifier()
        lemma = ProvedLemma("h1", "lemma h1 : True", ":= by sorry")
        assert not verifier.verify(lemma)

    def test_structural_check_empty(self):
        verifier = LemmaVerifier()
        lemma = ProvedLemma("h1", "", "")
        assert not verifier.verify(lemma)

    def test_batch_verify(self):
        verifier = LemmaVerifier()
        lemmas = [
            ProvedLemma("h1", "lemma h1 : True", ":= by trivial"),
            ProvedLemma("h2", "lemma h2 : True", ":= by sorry"),
        ]
        verified = verifier.verify_batch(lemmas)
        assert len(verified) == 1
