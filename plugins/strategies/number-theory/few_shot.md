## Number theory proof patterns for Lean 4

### Pattern 1: Natural number subtraction (CRITICAL)
In Lean4, ℕ subtraction truncates: if a < b, then a - b = 0.
Always prove `b ≤ a` BEFORE using `a - b` in equalities.

```lean
-- WRONG: ring fails on ℕ subtraction
theorem bad_example (n : ℕ) : 2^n - 1 + 1 = 2^n := by ring  -- FAILS

-- CORRECT: establish ≤ first, then use Nat.sub_add_cancel
theorem good_example (n : ℕ) : 2^n - 1 + 1 = 2^n := by
  have h : 1 ≤ 2^n := Nat.one_le_two_pow
  exact Nat.sub_add_cancel h
```

### Pattern 2: Induction on ℕ with recursive sequences
```lean
theorem seq_example (n : ℕ) (u : ℕ → ℕ) (h₀ : u 0 = 0)
  (h₁ : ∀ n, u (n + 1) = 2 * u n + 1) : u n = 2^n - 1 := by
  induction n with
  | zero => simp [h₀]
  | succ n ih =>
    rw [h₁, ih]
    have h : n + 1 ≤ 2^n := -- prove by separate induction or Nat.lt_two_pow
    omega  -- or use Nat.sub_add_cancel + arithmetic
```

### Pattern 3: Divisibility proofs
```lean
-- Use omega for simple divisibility
theorem div_example (n : ℕ) : 2 ∣ n * (n + 1) := by omega

-- For more complex cases, use Nat.dvd_trans
theorem div_chain (a b c : ℕ) (h1 : a ∣ b) (h2 : b ∣ c) : a ∣ c :=
  Nat.dvd_trans h1 h2
```

### Pattern 4: Modular arithmetic
```lean
-- norm_num handles concrete modular computations
theorem mod_example : 17 % 5 = 2 := by norm_num

-- For symbolic, use Int.emod_emod_of_dvd or unfold definitions
```
