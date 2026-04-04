"""tests/test_prover/test_codegen.py — 代码生成模块测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pytest
from prover.codegen.code_formatter import (
    format_lean_code, extract_proof_body, wrap_proof, _strip_markdown,
    _normalize_unicode, _collapse_blank_lines)
from prover.codegen.tactic_generator import TacticGenerator


# ── Code Formatter ──

class TestCodeFormatter:
    def test_strip_markdown(self):
        code = "```lean\ntheorem t := by sorry\n```"
        assert "```" not in _strip_markdown(code)
        assert "theorem" in _strip_markdown(code)

    def test_normalize_unicode(self):
        result = _normalize_unicode("fun x -> x")
        assert "→" in result

    def test_collapse_blank_lines(self):
        code = "a\n\n\n\n\nb"
        result = _collapse_blank_lines(code)
        assert result.count("\n") <= 3

    def test_format_full(self):
        raw = "```lean\ntheorem t : True := by\n  trivial\n```"
        result = format_lean_code(raw)
        assert "```" not in result
        assert "theorem" in result
        assert result.endswith("\n")

    def test_extract_proof_body(self):
        full = "theorem t : True := by\n  exact trivial"
        body = extract_proof_body(full)
        assert body == "exact trivial"

    def test_extract_proof_body_term(self):
        full = "theorem t : True := trivial"
        body = extract_proof_body(full)
        assert body == "trivial"

    def test_wrap_proof_tactic(self):
        result = wrap_proof("theorem t : True", "trivial")
        assert ":= by" in result
        assert "trivial" in result

    def test_wrap_proof_by(self):
        result = wrap_proof("theorem t : True", "by exact trivial")
        assert ":= by" in result

    def test_wrap_proof_term(self):
        result = wrap_proof("theorem t : True", ":= trivial")
        assert ":= trivial" in result


# ── Tactic Generator ──

class TestTacticGenerator:
    def test_rule_mode_implication(self):
        gen = TacticGenerator(mode="rule")
        seqs = gen.generate("P → Q")
        assert len(seqs) > 0
        # Should contain a sequence with intro
        flat = [t for seq in seqs for t in seq]
        assert any("intro" in t for t in flat)

    def test_rule_mode_equality(self):
        gen = TacticGenerator(mode="rule")
        seqs = gen.generate("n + m = m + n")
        assert len(seqs) > 0
        flat = [t for seq in seqs for t in seq]
        assert any(t in flat for t in ["ring", "omega", "simp", "rfl"])

    def test_rule_mode_nat(self):
        gen = TacticGenerator(mode="rule")
        seqs = gen.generate("∀ (n : Nat), P n")
        flat = [t for seq in seqs for t in seq]
        assert any("intro" in t for t in flat)

    def test_max_sequences(self):
        gen = TacticGenerator(mode="rule")
        seqs = gen.generate("P → Q", max_sequences=2)
        assert len(seqs) <= 2

    def test_deduplicate(self):
        gen = TacticGenerator(mode="rule")
        seqs = gen.generate("P → Q", max_sequences=10)
        unique = set(tuple(s) for s in seqs)
        assert len(unique) == len(seqs)
