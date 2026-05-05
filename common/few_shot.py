"""common/few_shot.py — Lean 4 few-shot examples used at prompt-build time.


``FEW_SHOT_EXAMPLES`` 字符串常量 —— ``build_prompt`` / ``FIRST_ATTEMPT``
/ ``RETRY`` 三个 helper 全部只在测试里调过。提取为单文件常量, 删除 builder。
"""
FEW_SHOT_EXAMPLES = """\
## Example proofs for reference

Example 1 (automation — try simple tactics first):
```lean
theorem add_comm_nat (n m : Nat) : n + m = m + n := by
  omega
```

Example 2 (induction with structured cases):
```lean
theorem sum_range_id (n : Nat) : 2 * (Finset.range n).sum id = n * (n - 1) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ]
    simp [Nat.mul_add, Nat.add_mul]
    omega
```

Example 3 (have steps for intermediate results):
```lean
theorem sq_nonneg_sum (a b : ℝ) : 0 ≤ a^2 + b^2 := by
  have ha := sq_nonneg a
  have hb := sq_nonneg b
  linarith
```

Example 4 (rewriting with specific lemmas):
```lean
theorem dvd_mul_of_dvd_left {a b : Nat} (h : a ∣ b) (c : Nat) : a ∣ b * c := by
  rcases h with ⟨k, hk⟩
  exact ⟨k * c, by rw [hk]; ring⟩
```

Example 5 (cases and contradiction):
```lean
theorem not_prime_one : ¬ Nat.Prime 1 := by
  intro h
  exact Nat.Prime.one_lt h |>.false
```
"""
