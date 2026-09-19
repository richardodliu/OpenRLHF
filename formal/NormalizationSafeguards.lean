import ProMax

/-!
Exact-real analysis of the implemented additive stabilizer, product cap and
positive factor clamps. The raw moments are computed over nonzero active
tokens. In particular, a positive additive stabilizer changes the second
moment even when no clamp activates; the ideal unit-moment theorem must not
be applied to that implementation branch without a residual term.
-/
namespace REINFORCEProMax.NormalizationSafeguards

open scoped BigOperators

noncomputable def signScale (α β x : ℝ) : ℝ := if 0 < x then α * x else if x < 0 then β * x else 0

theorem signScale_split (α β x : ℝ) :
    signScale α β x = α * (if 0 < x then x else 0) + β * (if x < 0 then x else 0) := by
  rcases lt_trichotomy x 0 with hneg | hzero | hpos
  · simp [signScale, hneg, not_lt_of_gt hneg, not_lt_of_ge (le_of_lt hneg)]
  · subst x; simp [signScale]
  · simp [signScale, hpos, not_lt_of_gt hpos, not_lt_of_ge (le_of_lt hpos)]

theorem signScale_square_split (α β x : ℝ) :
    (signScale α β x) ^ 2 = α ^ 2 * (if 0 < x then x ^ 2 else 0) +
      β ^ 2 * (if x < 0 then x ^ 2 else 0) := by
  rcases lt_trichotomy x 0 with hneg | hzero | hpos
  · simp [signScale, hneg, not_lt_of_gt hneg, not_lt_of_ge (le_of_lt hneg), mul_pow]
  · subst x; simp [signScale]
  · simp [signScale, hpos, not_lt_of_gt hpos, not_lt_of_ge (le_of_lt hpos), mul_pow]

/-- Actual finite positive/negative active index sets generate the sufficient statistics. -/
theorem finite_scaled_moments {ι : Type*} (S : Finset ι) (A : ι → ℝ) (α β : ℝ) :
    (∑ i ∈ S, signScale α β (A i)) =
      α * (∑ i ∈ S.filter (fun i => 0 < A i), A i) +
      β * (∑ i ∈ S.filter (fun i => A i < 0), A i) ∧
    (∑ i ∈ S, (signScale α β (A i)) ^ 2) =
      α ^ 2 * (∑ i ∈ S.filter (fun i => 0 < A i), (A i) ^ 2) +
      β ^ 2 * (∑ i ∈ S.filter (fun i => A i < 0), (A i) ^ 2) := by
  classical
  constructor
  · simp_rw [signScale_split]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    simp only [Finset.sum_filter]
  · simp_rw [signScale_square_split]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    simp only [Finset.sum_filter]

theorem signScale_preserves_sign (α β x : ℝ) (hα : 0 < α) (hβ : 0 < β) :
    (0 < signScale α β x ↔ 0 < x) ∧ (signScale α β x < 0 ↔ x < 0) ∧
      (signScale α β x = 0 ↔ x = 0) := by
  rcases lt_trichotomy x 0 with h | rfl | h
  · have hp := mul_neg_of_pos_of_neg hβ h
    simp [signScale, h, not_lt_of_gt h, hp, not_lt_of_gt hp, ne_of_lt hp, ne_of_lt h]
  · simp [signScale]
  · have hp := mul_pos hα h
    simp [signScale, h, not_lt_of_gt h, hp, not_lt_of_gt hp, ne_of_gt hp, ne_of_gt h]

noncomputable def positiveClamp (ε K x : ℝ) : ℝ := min (max x ε) K

theorem positiveClamp_pos (ε K x : ℝ) (hε : 0 < ε) (hK : 0 < K) :
    0 < positiveClamp ε K x := by
  exact lt_min (lt_of_lt_of_le hε (le_max_right _ _)) hK

theorem positiveClamp_mem (ε K x : ℝ) (hεK : ε ≤ K) :
    ε ≤ positiveClamp ε K x ∧ positiveClamp ε K x ≤ K := by
  exact ⟨le_min (le_max_right _ _) hεK, min_le_right _ _⟩

/-- With the product cap inactive, epsilon preserves centering but changes variance. -/
theorem stabilized_closed_form (P N Qp Qn m ε : ℝ)
    (hP : 0 < P) (hN : 0 < N) (hQp : 0 < Qp) (hQn : 0 ≤ Qn)
    (hm : 0 < m) (hε : 0 ≤ ε) :
    let κ := P / N
    let D := Qp + κ ^ 2 * Qn
    let α := Real.sqrt (m / (D + ε))
    let β := α * κ
    0 < α ∧ 0 < β ∧ α * P - β * N = 0 ∧
      (α ^ 2 * Qp + β ^ 2 * Qn) / m = D / (D + ε) := by
  dsimp
  have hD : 0 < Qp + (P / N) ^ 2 * Qn := by positivity
  have hden : 0 < Qp + (P / N) ^ 2 * Qn + ε := by positivity
  have hα : 0 < Real.sqrt (m / (Qp + (P / N) ^ 2 * Qn + ε)) :=
    Real.sqrt_pos.mpr (div_pos hm hden)
  have hs := Real.sq_sqrt (le_of_lt (div_pos hm hden))
  refine ⟨hα, mul_pos hα (div_pos hP hN), ?_, ?_⟩
  · field_simp [ne_of_gt hN]
    <;> ring
  · rw [mul_pow, hs]
    field_simp [ne_of_gt hden, ne_of_gt hm]
    <;> ring

theorem stabilizer_second_moment_lt_one (D ε : ℝ) (hD : 0 < D) (hε : 0 < ε) :
    0 < D / (D + ε) ∧ D / (D + ε) < 1 := by
  exact ⟨div_pos hD (by positivity), (div_lt_one (by positivity)).mpr (by linarith)⟩

/-- An explicit relative-variance deficit, small whenever epsilon is small relative to D. -/
theorem stabilizer_deficit_bound (D ε : ℝ) (hD : 0 < D) (hε : 0 ≤ ε) :
    1 - D / (D + ε) = ε / (D + ε) ∧
      0 ≤ 1 - D / (D + ε) ∧ 1 - D / (D + ε) ≤ ε / D := by
  have hden : 0 < D + ε := by positivity
  have heq : 1 - D / (D + ε) = ε / (D + ε) := by
    field_simp [ne_of_gt hden]
    <;> ring
  refine ⟨heq, ?_, ?_⟩
  · rw [heq]; positivity
  · rw [heq]
    exact div_le_div_of_nonneg_left hε hD (by linarith)

/-- Product capping can increase or decrease the second moment relative to one. -/
theorem capped_stabilized_moment (Qp Qn κ m C ε : ℝ)
    (hQp : 0 < Qp) (hQn : 0 ≤ Qn) (hm : 0 < m) (hC : 0 ≤ C) (hε : 0 ≤ ε) :
    let D := Qp + κ ^ 2 * Qn
    let Dc := Qp + min (κ ^ 2 * Qn) C + ε
    let α := Real.sqrt (m / Dc)
    let β := α * κ
    (α ^ 2 * Qp + β ^ 2 * Qn) / m = D / Dc ∧
      D / Dc - 1 = (κ ^ 2 * Qn - min (κ ^ 2 * Qn) C - ε) / Dc := by
  dsimp
  have hcap : 0 ≤ min (κ ^ 2 * Qn) C := le_min (by positivity) hC
  have hden : 0 < Qp + min (κ ^ 2 * Qn) C + ε := by positivity
  have hs := Real.sq_sqrt (le_of_lt (div_pos hm hden))
  constructor
  · rw [mul_pow, hs]
    field_simp [ne_of_gt hden, ne_of_gt hm]
    <;> ring
  · field_simp [ne_of_gt hden]
    <;> ring

theorem factor_mean_residual (P N α β a b : ℝ) (hcenter : α * P - β * N = 0) :
    a * P - b * N = (a - α) * P - (b - β) * N := by nlinarith

theorem factor_mean_residual_bound (P N α β a b : ℝ) (hP : 0 ≤ P) (hN : 0 ≤ N)
    (hcenter : α * P - β * N = 0) :
    |a * P - b * N| ≤ |a - α| * P + |b - β| * N := by
  rw [factor_mean_residual P N α β a b hcenter]
  calc
    |(a - α) * P - (b - β) * N| ≤ |(a - α) * P| + |(b - β) * N| := abs_sub _ _
    _ = |a - α| * P + |b - β| * N := by rw [abs_mul, abs_mul, abs_of_nonneg hP, abs_of_nonneg hN]

theorem factor_second_moment_residual_bound (Qp Qn α β a b : ℝ) (hQp : 0 ≤ Qp) (hQn : 0 ≤ Qn) :
    |(a ^ 2 * Qp + b ^ 2 * Qn) - (α ^ 2 * Qp + β ^ 2 * Qn)| ≤
      |a - α| * |a + α| * Qp + |b - β| * |b + β| * Qn := by
  have heq : (a ^ 2 * Qp + b ^ 2 * Qn) - (α ^ 2 * Qp + β ^ 2 * Qn) =
      (a - α) * (a + α) * Qp + (b - β) * (b + β) * Qn := by ring
  rw [heq]
  calc
    _ ≤ |(a - α) * (a + α) * Qp| + |(b - β) * (b + β) * Qn| := abs_add _ _
    _ = _ := by simp only [abs_mul, abs_of_nonneg hQp, abs_of_nonneg hQn]

end REINFORCEProMax.NormalizationSafeguards
