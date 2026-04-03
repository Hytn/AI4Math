"""benchmarks/datasets/builtin/problems.py — 内置冒烟测试题"""
from prover.models import BenchmarkProblem

BUILTIN_PROBLEMS = [
    BenchmarkProblem("builtin_nat_add_comm", "nat_add_comm",
                     "theorem nat_add_comm (a b : Nat) : a + b = b + a",
                     difficulty="easy", source="builtin", natural_language="Addition is commutative."),
    BenchmarkProblem("builtin_int_mul_comm", "int_mul_comm",
                     "theorem int_mul_comm (a b : Int) : a * b = b * a",
                     difficulty="easy", source="builtin"),
    BenchmarkProblem("builtin_abs_nonneg", "abs_nonneg",
                     "theorem abs_nonneg_example (a : Int) : 0 ≤ |a|",
                     difficulty="easy", source="builtin"),
    BenchmarkProblem("builtin_sum_first_n", "sum_first_n",
                     "theorem sum_first_n (n : Nat) : 2 * (Finset.range (n + 1)).sum id = n * (n + 1)",
                     difficulty="medium", source="builtin"),
    BenchmarkProblem("builtin_amgm_two", "amgm_two",
                     "theorem amgm_two (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : a * b ≤ (a + b) ^ 2 / 4",
                     difficulty="medium", source="builtin"),
]
