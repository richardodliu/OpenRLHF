import Mathlib.Tactic

/-!
Exact finite-batch decomposition of the implemented ProMax objective. The
finite index runs over batch/action positions, and `a` includes the action
mask and the selected reduction weights. `r` is current/old, `w` is
old/rollout, `M` is the cached prefix gate, and `H` is normalized advantage.
`c` is the clipped current/old ratio; the identities hold for any `c`, so no
particular numerical implementation of clipping is hidden as an assumption.
-/
namespace REINFORCEProMax.Objective

open scoped BigOperators
variable {I : Type*} [Fintype I]

/-- The optional dual-clip branch is outside the standard PPO penalty theorem.
For advantage -1, ratio 4, PPO upper clip 6/5, and dual clip 3,
the maximization objective rises from -4 to -3. Its deduction from the
unclipped objective is therefore negative, even with unit gate/IS weights. -/
theorem dual_clip_negative_penalty_counterexample :
    min ((4 : ℝ) * (-1)) ((6 / 5) * (-1)) = -4 ∧
    max (min ((4 : ℝ) * (-1)) ((6 / 5) * (-1))) (3 * (-1)) = -3 ∧
    (4 : ℝ) * (-1) -
      max (min ((4 : ℝ) * (-1)) ((6 / 5) * (-1))) (3 * (-1)) < 0 := by
  norm_num

/-- Positive sign-dependent scales preserve each signed contribution, but
need not preserve their vector sum's direction: (1,-1) becomes (2,-1). -/
theorem asymmetric_scaling_direction_counterexample :
    ¬ ∃ s : ℝ, (2 : ℝ) = s * 1 ∧ (-1 : ℝ) = s * (-1) := by
  rintro ⟨s, h₁, h₂⟩
  linarith

/-- The failure persists for scales satisfying ProMax's actual constraints:
one +1 token and four -1 tokens normalize with alpha=2, beta=1/2.
Their normalized mean is zero and their second moment is one. If the raw
positive/negative gradient sums are (1,0) and (0,-1), the total direction
changes from (1,-1) to (2,-1/2). -/
theorem normalized_scaling_direction_counterexample :
    (2 : ℝ) + 4 * (-(1 / 2)) = 0 ∧
    ((2 : ℝ)^2 + 4 * (-(1 / 2))^2) / 5 = 1 ∧
    ¬ ∃ s : ℝ, (2 : ℝ) = s * 1 ∧ (-(1 / 2) : ℝ) = s * (-1) := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  rintro ⟨s, h₁, h₂⟩
  linarith

noncomputable def objective (a M w r c H : I → ℝ) : ℝ :=
  ∑ i, a i * M i * w i * min (r i * H i) (c i * H i)

noncomputable def rawSurrogate (a w r A : I → ℝ) : ℝ :=
  ∑ i, a i * (r i * w i) * A i

noncomputable def gateCorrection (a M w r A : I → ℝ) : ℝ :=
  ∑ i, a i * (r i * w i) * (1 - M i) * A i

noncomputable def normalizationCorrection (a M w r A H : I → ℝ) : ℝ :=
  ∑ i, a i * (r i * w i) * M i * (H i - A i)

noncomputable def clippingPenalty (a M w r c H : I → ℝ) : ℝ :=
  ∑ i, a i * M i * w i * (r i * H i - min (r i * H i) (c i * H i))

/-- Exact decomposition for every finite batch, with no assumption that the
prefix gate is independent of the current sampled action or of the reward. -/
theorem finite_promax_objective_decomposition (a M w r c A H : I → ℝ) :
    objective a M w r c H = rawSurrogate a w r A - gateCorrection a M w r A +
      normalizationCorrection a M w r A H - clippingPenalty a M w r c H := by
  unfold objective rawSurrogate gateCorrection normalizationCorrection clippingPenalty
  rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Correcting the observed change at two policies recovers the raw
surrogate change, with the same frozen batch, weights, gate, and advantages. -/
theorem corrected_objective_delta (a M w r r₀ c c₀ A H : I → ℝ) :
    (objective a M w r c H - objective a M w r₀ c₀ H) +
      (gateCorrection a M w r A - gateCorrection a M w r₀ A) -
      (normalizationCorrection a M w r A H - normalizationCorrection a M w r₀ A H) +
      (clippingPenalty a M w r c H - clippingPenalty a M w r₀ c₀ H) =
        rawSurrogate a w r A - rawSurrogate a w r₀ A := by
  rw [finite_promax_objective_decomposition, finite_promax_objective_decomposition]
  ring

/-- PPO clipping is a nonnegative deduction from the gated, normalized
unclipped objective. The gate and normalization corrections remain signed. -/
theorem finite_clipping_penalty_nonnegative (a M w r c H : I → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hM : ∀ i, 0 ≤ M i) (hw : ∀ i, 0 ≤ w i) :
    0 ≤ clippingPenalty a M w r c H := by
  apply Finset.sum_nonneg
  intro i _
  exact mul_nonneg (mul_nonneg (mul_nonneg (ha i) (hM i)) (hw i))
    (sub_nonneg.mpr (min_le_left _ _))

theorem finite_clipped_le_unclipped (a M w r c A H : I → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hM : ∀ i, 0 ≤ M i) (hw : ∀ i, 0 ≤ w i) :
    objective a M w r c H ≤ rawSurrogate a w r A - gateCorrection a M w r A +
      normalizationCorrection a M w r A H := by
  rw [finite_promax_objective_decomposition]
  exact sub_le_self _ (finite_clipping_penalty_nonnegative a M w r c H ha hM hw)

/-- A computable batch discrepancy bound, retaining the gate, asymmetric
normalization, both importance ratios, and PPO clipping exactly. -/
theorem finite_promax_objective_error_bound (a M w r c A H : I → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hM0 : ∀ i, 0 ≤ M i) (hM1 : ∀ i, M i ≤ 1)
    (hw : ∀ i, 0 ≤ w i) (hr : ∀ i, 0 ≤ r i) :
    |objective a M w r c H - rawSurrogate a w r A| ≤
      (∑ i, a i * (r i * w i) * (1 - M i) * |A i|) +
      (∑ i, a i * (r i * w i) * M i * |H i - A i|) +
        clippingPenalty a M w r c H := by
  have hg : |gateCorrection a M w r A| ≤
      ∑ i, a i * (r i * w i) * (1 - M i) * |A i| := by
    unfold gateCorrection
    calc
      _ ≤ ∑ i, |a i * (r i * w i) * (1 - M i) * A i| :=
        Finset.abs_sum_le_sum_abs _ _
      _ = _ := by
        apply Finset.sum_congr rfl
        intro i _
        rw [abs_mul, abs_of_nonneg
          (mul_nonneg (mul_nonneg (ha i) (mul_nonneg (hr i) (hw i)))
            (sub_nonneg.mpr (hM1 i))) ]
  have hn : |normalizationCorrection a M w r A H| ≤
      ∑ i, a i * (r i * w i) * M i * |H i - A i| := by
    unfold normalizationCorrection
    calc
      _ ≤ ∑ i, |a i * (r i * w i) * M i * (H i - A i)| :=
        Finset.abs_sum_le_sum_abs _ _
      _ = _ := by
        apply Finset.sum_congr rfl
        intro i _
        rw [abs_mul, abs_of_nonneg
          (mul_nonneg (mul_nonneg (ha i) (mul_nonneg (hr i) (hw i))) (hM0 i))]
  have hc := finite_clipping_penalty_nonnegative a M w r c H ha hM0 hw
  rw [finite_promax_objective_decomposition]
  calc
    _ = |(-gateCorrection a M w r A + normalizationCorrection a M w r A H) -
        clippingPenalty a M w r c H| := by congr 1; ring
    _ ≤ |(-gateCorrection a M w r A + normalizationCorrection a M w r A H)| +
        |clippingPenalty a M w r c H| := by
      simpa only [sub_zero, zero_sub, abs_neg] using
        (abs_sub_le (-gateCorrection a M w r A + normalizationCorrection a M w r A H)
          0 (clippingPenalty a M w r c H))
    _ ≤ (|gateCorrection a M w r A| + |normalizationCorrection a M w r A H|) +
        clippingPenalty a M w r c H := by
      rw [abs_of_nonneg hc]
      have habs := abs_add_le (-gateCorrection a M w r A) (normalizationCorrection a M w r A H)
      rw [abs_neg] at habs
      exact add_le_add_right habs _
    _ ≤ _ := add_le_add_right (add_le_add hg hn) _

end REINFORCEProMax.Objective
