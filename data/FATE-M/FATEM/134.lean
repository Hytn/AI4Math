import Mathlib

/--
Show that if $H$ is a normal subgroup of $G$ and $H$ is a $p$-group, then $H$ is contained in
every Sylow $p$-subgroup of $G$.
-/
theorem Sylow.le_of_normal {G : Type*} [Group G] [Finite G] (H : Subgroup G) [H.Normal] (p : ℕ)
    (hp : p.Prime) (hH : IsPGroup p H) (P : Sylow p G) : H ≤ P := by
  sorry
