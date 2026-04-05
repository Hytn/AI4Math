## Algebra proof examples

Example 1 (group identity):
```lean
theorem mul_left_cancel {G : Type*} [Group G] (a b c : G) (h : a * b = a * c) : b = c := by
  have := congr_arg (a⁻¹ * ·) h
  simp [← mul_assoc, inv_mul_cancel] at this
  exact this
```

Example 2 (ring homomorphism):
```lean
theorem map_neg_one {R S : Type*} [Ring R] [Ring S] (f : R →+* S) : f (-1) = -1 := by
  simp [map_neg, map_one]
```

Example 3 (subgroup closure):
```lean
theorem Subgroup.closure_le {G : Type*} [Group G] (H : Subgroup G) (s : Set G)
    (hs : s ⊆ H) : Subgroup.closure s ≤ H := by
  exact Subgroup.closure_le.mpr hs
```
