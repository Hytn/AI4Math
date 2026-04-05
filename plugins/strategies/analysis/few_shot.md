## Analysis proof examples

Example 1 (inequality with positivity):
```lean
theorem sq_add_sq_nonneg (a b : ℝ) : 0 ≤ a ^ 2 + b ^ 2 := by
  positivity
```

Example 2 (using field_simp and ring):
```lean
theorem div_add_div (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) :
    a / b + c / d = (a * d + b * c) / (b * d) := by
  field_simp
  ring
```

Example 3 (continuity composition):
```lean
theorem continuous_sq : Continuous (fun x : ℝ => x ^ 2) := by
  exact continuous_pow 2
```
