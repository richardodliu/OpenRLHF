import KLChain

namespace REINFORCEProMax

/- Equality case for the strictly-positive finite-cell log-sum inequality.
   This is the exact one-cell building block of the KL coarse-graining equality
   condition; zero-mass cells are intentionally excluded. -/
theorem finite_log_sum_equality_iff
    {n : ℕ} (p q : Fin n → ℝ) (hn : 0 < n)
    (hp : ∀ y, 0 < p y) (hq : ∀ y, 0 < q y) :
    (∑ y : Fin n, p y) * Real.log ((∑ y : Fin n, q y) / (∑ y : Fin n, p y)) =
      ∑ y : Fin n, p y * Real.log (q y / p y) ↔
      ∀ y z : Fin n, q y / p y = q z / p z := by
  let P : ℝ := ∑ y : Fin n, p y
  have hP : 0 < P := by
    dsimp [P]
    have hne : (Finset.univ : Finset (Fin n)).Nonempty := by
      refine ⟨⟨0, hn⟩, ?_⟩
      simp
    exact Finset.sum_pos (fun y hy => hp y) hne
  have hweights : ∀ y : Fin n, 0 ≤ p y / P := by
    intro y
    exact le_of_lt (div_pos (hp y) hP)
  have hweights_pos : ∀ y : Fin n, 0 < p y / P := by
    intro y
    exact div_pos (hp y) hP
  have hweights_sum : (∑ y : Fin n, p y / P) = 1 := by
    calc
      (∑ y : Fin n, p y / P) = (∑ y : Fin n, p y) / P := by
        rw [Finset.sum_div]
      _ = 1 := by
        change P / P = 1
        field_simp [ne_of_gt hP]
  have hratio : ∀ y : Fin n, q y / p y ∈ Set.Ioi (0 : ℝ) := by
    intro y
    exact Set.mem_Ioi.mpr (div_pos (hq y) (hp y))
  have hleft :
      (∑ y : Fin n, (p y / P) * Real.log (q y / p y)) =
        (∑ y : Fin n, p y * Real.log (q y / p y)) / P := by
    calc
      (∑ y : Fin n, (p y / P) * Real.log (q y / p y)) =
          ∑ y : Fin n, (p y * Real.log (q y / p y)) / P := by
            apply Finset.sum_congr rfl
            intro y hy
            ring
      _ = (∑ y : Fin n, p y * Real.log (q y / p y)) / P := by
            rw [Finset.sum_div]
  have harg :
      (∑ y : Fin n, (p y / P) * (q y / p y)) =
        (∑ y : Fin n, q y) / P := by
    calc
      (∑ y : Fin n, (p y / P) * (q y / p y)) =
          ∑ y : Fin n, q y / P := by
            apply Finset.sum_congr rfl
            intro y hy
            field_simp [ne_of_gt (hp y), ne_of_gt hP]
            ring
      _ = (∑ y : Fin n, q y) / P := by rw [Finset.sum_div]
  constructor
  · intro heq
    have hscaled :
        Real.log ((∑ y : Fin n, q y) / P) =
          (∑ y : Fin n, p y * Real.log (q y / p y)) / P := by
      apply (eq_div_iff (ne_of_gt hP)).2
      rw [mul_comm]
      exact heq
    have hj_eq :
        Real.log (∑ y : Fin n, (p y / P) * (q y / p y)) =
          ∑ y : Fin n, (p y / P) * Real.log (q y / p y) := by
      rw [harg, hleft]
      exact hscaled
    have heq_points :=
      (strictConcaveOn_log_Ioi.map_sum_eq_iff
        (t := (Finset.univ : Finset (Fin n)))
        (w := fun y : Fin n => p y / P)
        (p := fun y : Fin n => q y / p y)
        (fun y hy => hweights_pos y)
        hweights_sum
        (by intro y hy; exact hratio y)).mp hj_eq
    intro y z
    have hy := heq_points y (by simp)
    have hz := heq_points z (by simp)
    rw [hy, hz]
  · intro hconst
    obtain ⟨y₀, hy₀⟩ : ∃ y₀ : Fin n, True := ⟨⟨0, hn⟩, trivial⟩
    let r : ℝ := q y₀ / p y₀
    have hr : 0 < r := by
      dsimp [r]
      exact div_pos (hq y₀) (hp y₀)
    have hratio_const : ∀ y, q y / p y = r := by
      intro y
      dsimp [r]
      exact hconst y y₀
    have hq_point : ∀ y, q y = r * p y := by
      intro y
      have hy := hratio_const y
      field_simp [ne_of_gt (hp y)] at hy
      linarith
    have hq_sum : (∑ y : Fin n, q y) = r * P := by
      calc
        (∑ y : Fin n, q y) = ∑ y : Fin n, r * p y := by
          apply Finset.sum_congr rfl
          intro y hy
          exact hq_point y
        _ = r * P := by
          dsimp [P]
          rw [Finset.mul_sum]
    have hlogarg : (∑ y : Fin n, q y) / P = r := by
      rw [hq_sum]
      field_simp [ne_of_gt hP]
    have hlogratio : ∀ y, Real.log (q y / p y) = Real.log r := by
      intro y
      rw [hratio_const y]
    rw [hlogarg]
    simp_rw [hlogratio]
    rw [Finset.sum_mul]

/- The commonly quoted equality condition is false under one-sided absolute
   continuity if it only quantifies over tokens with positive mass under the
   first distribution.  A q-only token inside the cell contributes to the
   coarse-graining gap.  This concrete two-token counterexample is the exact
   failure mode: the positive-p support condition is vacuous on token 1, yet
   the full reverse KL is strictly larger than the one-cell KL. -/
theorem coarse_grain_kl_positive_support_condition_counterexample :
    let p : Fin 2 → ℝ := ![1, 0]
    let q : Fin 2 → ℝ := ![(1 / 2 : ℝ), (1 / 2 : ℝ)]
    (∑ y : Fin 2, p y = 1) ∧
      (∑ y : Fin 2, q y = 1) ∧
      (∀ y, p y > 0 → q y > 0) ∧
      (∀ y z, p y > 0 → p z > 0 → p y / q y = p z / q z) ∧
      0 < (∑ y : Fin 2, p y * Real.log (p y / q y)) := by
  dsimp
  have hlog2 : 0 < Real.log (2 : ℝ) := Real.log_pos (by norm_num)
  constructor
  · norm_num
  constructor
  · norm_num
  constructor
  · intro y hy
    fin_cases y <;> norm_num at hy ⊢
  constructor
  · intro y z hy hz
    fin_cases y <;> fin_cases z <;> norm_num at hy hz ⊢
  · norm_num [Fin.sum_univ_two, Real.log_div (by norm_num : (2 : ℝ) ≠ 0)]
    simpa [Real.log_inv] using hlog2

end REINFORCEProMax
