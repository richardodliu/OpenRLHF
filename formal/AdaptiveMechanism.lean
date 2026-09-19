import NormalizationSafeguards

/-!
Mechanism of ideal asymmetric normalization for actual two-level RLOO rewards.
The positive and negative RLOO amplitudes are generally different. Token
counts determine the normalized amplitudes, whereas mean response lengths
determine the ratio of the two normalization factors.
-/
namespace REINFORCEProMax.AdaptiveMechanism

open scoped BigOperators
open NormalizationSafeguards

noncomputable def idealAlpha (P N Qp Qn mass : ℝ) : ℝ :=
  Real.sqrt (mass / (Qp + (P / N) ^ 2 * Qn))

noncomputable def idealBeta (P N Qp Qn mass : ℝ) : ℝ :=
  idealAlpha P N Qp Qn mass * (P / N)

/-- Exact amplitudes for arbitrary positive two-level input magnitudes. -/
theorem two_level_normalized_amplitudes (P N a b : ℝ)
    (hP : 0 < P) (hN : 0 < N) (ha : 0 < a) (hb : 0 < b) :
    idealAlpha (P * a) (N * b) (P * a ^ 2) (N * b ^ 2) (P + N) * a =
      Real.sqrt (N / P) ∧
    idealBeta (P * a) (N * b) (P * a ^ 2) (N * b ^ 2) (P + N) * b =
      Real.sqrt (P / N) := by
  have hden : 0 < P * a ^ 2 + (P * a / (N * b)) ^ 2 * (N * b ^ 2) := by positivity
  have hratio : (P + N) / (P * a ^ 2 + (P * a / (N * b)) ^ 2 * (N * b ^ 2)) =
      (N / P) / a ^ 2 := by
    field_simp
    <;> ring
  have hα : idealAlpha (P * a) (N * b) (P * a ^ 2) (N * b ^ 2) (P + N) =
      Real.sqrt (N / P) / a := by
    unfold idealAlpha
    rw [hratio, Real.sqrt_div (by positivity), Real.sqrt_sq (le_of_lt ha)]
  constructor
  · rw [hα, div_mul_cancel₀ _ (ne_of_gt ha)]
  · unfold idealBeta
    rw [hα]
    have hs : (Real.sqrt (N / P)) ^ 2 = N / P := Real.sq_sqrt (by positivity)
    have hval : Real.sqrt (N / P) / a * (P * a / (N * b)) * b =
        Real.sqrt (N / P) * (P / N) := by
      field_simp
      <;> ring
    rw [hval]
    apply (sq_eq_sq₀ (by positivity) (Real.sqrt_nonneg _)).mp
    rw [mul_pow, hs, Real.sq_sqrt (by positivity)]
    field_simp
    <;> ring

noncomputable def rlooSignal {ι : Type*} [Fintype ι] (r : ι → ℝ) (i : ι) : ℝ :=
  r i - ((∑ j, r j) - r i) / ((Fintype.card ι : ℝ) - 1)

def twoLevelReward (k m : ℕ) (hi lo : ℝ) : Sum (Fin k) (Fin m) → ℝ :=
  Sum.elim (fun _ => hi) (fun _ => lo)

theorem two_level_reward_sum (k m : ℕ) (hi lo : ℝ) :
    (∑ i, twoLevelReward k m hi lo i) = (k : ℝ) * hi + (m : ℝ) * lo := by
  simp [twoLevelReward, Fintype.sum_sum_type]

theorem rloo_success_amplitude (k m : ℕ) (hi lo : ℝ)
    (hn : (k : ℝ) + m - 1 ≠ 0) (i : Fin k) :
    rlooSignal (twoLevelReward k m hi lo) (Sum.inl i) =
      (m : ℝ) * (hi - lo) / ((k : ℝ) + m - 1) := by
  rw [rlooSignal, two_level_reward_sum]
  simp only [twoLevelReward, Sum.elim_inl, Fintype.card_sum, Fintype.card_fin, Nat.cast_add]
  field_simp [hn]
  <;> ring

theorem rloo_failure_amplitude (k m : ℕ) (hi lo : ℝ)
    (hn : (k : ℝ) + m - 1 ≠ 0) (i : Fin m) :
    rlooSignal (twoLevelReward k m hi lo) (Sum.inr i) =
      -((k : ℝ) * (hi - lo) / ((k : ℝ) + m - 1)) := by
  rw [rlooSignal, two_level_reward_sum]
  simp only [twoLevelReward, Sum.elim_inr, Fintype.card_sum, Fintype.card_fin, Nat.cast_add]
  field_simp [hn]
  <;> ring

/-- The RLOO amplitude ratio cancels success frequency in the scale ratio. -/
theorem rloo_scale_ratio_mean_lengths (k m P N gap : ℝ)
    (hk : 0 < k) (hm : 0 < m) (hP : 0 < P) (hN : 0 < N)
    (hgap : 0 < gap) (hn : k + m - 1 ≠ 0) :
    (P * (m * gap / (k + m - 1))) / (N * (k * gap / (k + m - 1))) =
      (P / k) / (N / m) := by
  field_simp
  <;> ring

/-- Aggregate raw positive/negative token mass is imbalanced exactly by mean lengths. -/
theorem raw_mass_ratio_mean_lengths (k m Lp Ln gap : ℝ)
    (hk : 0 < k) (hm : 0 < m) (hLn : 0 < Ln) (hgap : 0 < gap)
    (hn : k + m - 1 ≠ 0) :
    ((k * Lp) * (m * gap / (k + m - 1))) /
      ((m * Ln) * (k * gap / (k + m - 1))) = Lp / Ln := by
  field_simp
  <;> ring

/-- The normalized positive and negative coefficient masses are equal. -/
theorem normalized_mass_balance (P N : ℝ) (hP : 0 < P) (hN : 0 < N) :
    P * Real.sqrt (N / P) = N * Real.sqrt (P / N) := by
  apply (sq_eq_sq₀ (by positivity) (by positivity)).mp
  rw [mul_pow, mul_pow, Real.sq_sqrt (by positivity), Real.sq_sqrt (by positivity)]
  field_simp
  <;> ring

/-- For exactly two reward levels, ordinary token-weighted centering and
population-standard-deviation scaling give the same two coefficients. -/
theorem two_level_token_zscore (P N a b : ℝ)
    (hP : 0 < P) (hN : 0 < N) (ha : 0 < a) (hb : 0 < b) :
    let μ := (P * a - N * b) / (P + N)
    let v := (P * (a - μ) ^ 2 + N * (-b - μ) ^ 2) / (P + N)
    (a - μ) / Real.sqrt v = Real.sqrt (N / P) ∧
      (-b - μ) / Real.sqrt v = -Real.sqrt (P / N) := by
  dsimp only
  let μ := (P * a - N * b) / (P + N)
  let v := (P * (a - μ) ^ 2 + N * (-b - μ) ^ 2) / (P + N)
  have hpos : a - μ = N * (a + b) / (P + N) := by
    dsimp [μ]
    field_simp
    ring
  have hneg : b + μ = P * (a + b) / (P + N) := by
    dsimp [μ]
    field_simp
    ring
  have hv : v = P * N * (a + b) ^ 2 / (P + N) ^ 2 := by
    dsimp [v, μ]
    field_simp
    ring
  have hvpos : 0 < v := by rw [hv]; positivity
  have hpospos : 0 < a - μ := by rw [hpos]; positivity
  have hnegpos : 0 < b + μ := by rw [hneg]; positivity
  have hspos : 0 < Real.sqrt v := Real.sqrt_pos.mpr hvpos
  have hp : (a - μ) / Real.sqrt v = Real.sqrt (N / P) := by
    apply (sq_eq_sq₀ (le_of_lt (div_pos hpospos hspos)) (Real.sqrt_nonneg _)).mp
    rw [div_pow, Real.sq_sqrt (le_of_lt hvpos), Real.sq_sqrt (by positivity), hv, hpos]
    field_simp
    ring
  have hn : (b + μ) / Real.sqrt v = Real.sqrt (P / N) := by
    apply (sq_eq_sq₀ (le_of_lt (div_pos hnegpos hspos)) (Real.sqrt_nonneg _)).mp
    rw [div_pow, Real.sq_sqrt (le_of_lt hvpos), Real.sq_sqrt (by positivity), hv, hneg]
    field_simp
    ring
  refine ⟨hp, ?_⟩
  change (-b - μ) / Real.sqrt v = -Real.sqrt (P / N)
  calc
    (-b - μ) / Real.sqrt v = -((b + μ) / Real.sqrt v) := by ring
    _ = -Real.sqrt (P / N) := by rw [hn]

/-- With one success and nine equally long failures, both factors equal three. -/
theorem rare_success_does_not_imply_negative_attenuation :
    idealAlpha 1 1 1 (1 / 9) 10 = 3 ∧
    idealBeta 1 1 1 (1 / 9) 10 = 3 := by
  norm_num [idealAlpha, idealBeta]

/-- With the product and factor clamps inactive, epsilon gives one common shrinkage. -/
theorem stabilizer_common_shrinkage (mass D ε : ℝ)
    (hmass : 0 < mass) (hD : 0 < D) (hε : 0 ≤ ε) :
    Real.sqrt (mass / (D + ε)) =
      Real.sqrt (D / (D + ε)) * Real.sqrt (mass / D) := by
  rw [← Real.sqrt_mul (by positivity)]
  congr 1
  field_simp
  <;> ring

theorem paired_factor_ratio (α κ : ℝ) (hα : 0 < α) : (α * κ) / α = κ := by
  exact mul_div_cancel_left₀ κ (ne_of_gt hα)

/-- Common positive rescaling divides the ideal factors by the same scale. -/
theorem ideal_factors_rescale (P N Qp Qn mass c : ℝ) (hc : 0 < c) :
    idealAlpha (c * P) (c * N) (c ^ 2 * Qp) (c ^ 2 * Qn) mass =
      idealAlpha P N Qp Qn mass / c ∧
    idealBeta (c * P) (c * N) (c ^ 2 * Qp) (c ^ 2 * Qn) mass =
      idealBeta P N Qp Qn mass / c := by
  have hr : c * P / (c * N) = P / N := mul_div_mul_left P N (ne_of_gt hc)
  have hd : c ^ 2 * Qp + (P / N) ^ 2 * (c ^ 2 * Qn) =
      (Qp + (P / N) ^ 2 * Qn) * c ^ 2 := by ring
  have hα : idealAlpha (c * P) (c * N) (c ^ 2 * Qp) (c ^ 2 * Qn) mass =
      idealAlpha P N Qp Qn mass / c := by
    unfold idealAlpha
    rw [hr, hd, ← div_div, Real.sqrt_div' _ (sq_nonneg c), Real.sqrt_sq (le_of_lt hc)]
  refine ⟨hα, ?_⟩
  simp only [idealBeta, hα, hr]
  ring

theorem sign_scale_rescale (α β x c : ℝ) (hc : 0 < c) :
    signScale (α / c) (β / c) (c * x) = signScale α β x := by
  have hneg : c * x < 0 ↔ x < 0 := by
    simpa using (mul_lt_mul_left hc : c * x < c * 0 ↔ x < 0)
  simp only [signScale, mul_pos_iff_of_pos_left hc, hneg]
  split_ifs <;> field_simp <;> ring

noncomputable def positiveTotal {ι : Type*} [Fintype ι] (w A : ι → ℝ) : ℝ :=
  ∑ i, w i * (if 0 < A i then A i else 0)

noncomputable def negativeTotal {ι : Type*} [Fintype ι] (w A : ι → ℝ) : ℝ :=
  ∑ i, w i * (if A i < 0 then -A i else 0)

noncomputable def positiveSquare {ι : Type*} [Fintype ι] (w A : ι → ℝ) : ℝ :=
  ∑ i, w i * (if 0 < A i then (A i) ^ 2 else 0)

noncomputable def negativeSquare {ι : Type*} [Fintype ι] (w A : ι → ℝ) : ℝ :=
  ∑ i, w i * (if A i < 0 then (A i) ^ 2 else 0)

noncomputable def nonzeroMass {ι : Type*} [Fintype ι] (w A : ι → ℝ) : ℝ :=
  ∑ i, w i * (if A i = 0 then 0 else 1)

/-- Finite sums include arbitrary response-length or action-mask weights. -/
theorem weighted_statistics_rescale {ι : Type*} [Fintype ι]
    (w A : ι → ℝ) (c : ℝ) (hc : 0 < c) :
    positiveTotal w (fun i => c * A i) = c * positiveTotal w A ∧
    negativeTotal w (fun i => c * A i) = c * negativeTotal w A ∧
    positiveSquare w (fun i => c * A i) = c ^ 2 * positiveSquare w A ∧
    negativeSquare w (fun i => c * A i) = c ^ 2 * negativeSquare w A ∧
    nonzeroMass w (fun i => c * A i) = nonzeroMass w A := by
  have hneg : ∀ x : ℝ, c * x < 0 ↔ x < 0 := by
    intro x
    simpa using (mul_lt_mul_left hc : c * x < c * 0 ↔ x < 0)
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp only [positiveTotal, mul_pos_iff_of_pos_left hc, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    split_ifs <;> ring
  · simp only [negativeTotal, hneg, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    split_ifs <;> ring
  · simp only [positiveSquare, mul_pos_iff_of_pos_left hc, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    split_ifs <;> ring
  · simp only [negativeSquare, hneg, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    split_ifs <;> ring
  · simp [nonzeroMass, ne_of_gt hc]

noncomputable def weightedNormalize {ι : Type*} [Fintype ι] (w A : ι → ℝ) (i : ι) : ℝ :=
  signScale
    (idealAlpha (positiveTotal w A) (negativeTotal w A)
      (positiveSquare w A) (negativeSquare w A) (nonzeroMass w A))
    (idealBeta (positiveTotal w A) (negativeTotal w A)
      (positiveSquare w A) (negativeSquare w A) (nonzeroMass w A)) (A i)

/-- Exact scale invariance of the finite ideal normalization map. -/
theorem weighted_normalization_scale_invariant {ι : Type*} [Fintype ι]
    (w A : ι → ℝ) (c : ℝ) (hc : 0 < c) (i : ι) :
    weightedNormalize w (fun j => c * A j) i = weightedNormalize w A i := by
  obtain ⟨hP, hN, hQp, hQn, hm⟩ := weighted_statistics_rescale w A c hc
  unfold weightedNormalize
  rw [hP, hN, hQp, hQn, hm]
  obtain ⟨hα, hβ⟩ := ideal_factors_rescale (positiveTotal w A) (negativeTotal w A)
    (positiveSquare w A) (negativeSquare w A) (nonzeroMass w A) c hc
  rw [hα, hβ, sign_scale_rescale _ _ _ _ hc]

theorem mean_baseline_is_scaled_rloo {ι : Type*} [Fintype ι]
    (r : ι → ℝ) (hn : 1 < (Fintype.card ι : ℝ)) (i : ι) :
    r i - (∑ j, r j) / (Fintype.card ι : ℝ) =
      (((Fintype.card ι : ℝ) - 1) / (Fintype.card ι : ℝ)) * rlooSignal r i := by
  unfold rlooSignal
  field_simp [ne_of_gt (lt_trans zero_lt_one hn), ne_of_gt (sub_pos.mpr hn)]
  <;> ring

/-- Identical ideal normalized coefficients for mean and RLOO baselines. -/
theorem mean_and_rloo_normalization_equal {ι : Type*} [Fintype ι]
    (w r : ι → ℝ) (hn : 1 < (Fintype.card ι : ℝ)) (i : ι) :
    weightedNormalize w (fun j => r j - (∑ k, r k) / (Fintype.card ι : ℝ)) i =
      weightedNormalize w (rlooSignal r) i := by
  have hc : 0 < ((Fintype.card ι : ℝ) - 1) / (Fintype.card ι : ℝ) :=
    div_pos (by linarith) (by linarith)
  have hfun : (fun j => r j - (∑ k, r k) / (Fintype.card ι : ℝ)) =
      (fun j => (((Fintype.card ι : ℝ) - 1) / (Fintype.card ι : ℝ)) * rlooSignal r j) := by
    funext j
    exact mean_baseline_is_scaled_rloo r hn j
  rw [hfun]
  exact weighted_normalization_scale_invariant w (rlooSignal r) _ hc i

/-- The moments computed from a finite two-level response group, with length weights. -/
theorem two_level_weighted_statistics (k m : ℕ) (wp : Fin k → ℝ) (wn : Fin m → ℝ)
    (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    let w := Sum.elim wp wn
    let A := twoLevelReward k m a (-b)
    positiveTotal w A = (∑ i, wp i) * a ∧
    negativeTotal w A = (∑ i, wn i) * b ∧
    positiveSquare w A = (∑ i, wp i) * a ^ 2 ∧
    negativeSquare w A = (∑ i, wn i) * b ^ 2 ∧
    nonzeroMass w A = (∑ i, wp i) + (∑ i, wn i) := by
  have ha0 := ne_of_gt ha
  have hb0 := ne_of_gt hb
  have hna : ¬ a < 0 := not_lt_of_ge (le_of_lt ha)
  have hnb : ¬ 0 < -b := by linarith
  have hbneg : -b < 0 := by linarith
  simp [positiveTotal, negativeTotal, positiveSquare, negativeSquare, nonzeroMass,
    twoLevelReward, Fintype.sum_sum_type, ha, hna, hnb, hbneg, ha0, hb0,
    ← Finset.sum_mul]

theorem finite_two_level_normalization (k m : ℕ) (wp : Fin k → ℝ) (wn : Fin m → ℝ)
    (a b : ℝ) (hP : 0 < ∑ i, wp i) (hN : 0 < ∑ i, wn i) (ha : 0 < a) (hb : 0 < b) :
    (∀ i : Fin k, weightedNormalize (Sum.elim wp wn) (twoLevelReward k m a (-b))
      (Sum.inl i) = Real.sqrt ((∑ j, wn j) / (∑ j, wp j))) ∧
    (∀ i : Fin m, weightedNormalize (Sum.elim wp wn) (twoLevelReward k m a (-b))
      (Sum.inr i) = -Real.sqrt ((∑ j, wp j) / (∑ j, wn j))) := by
  obtain ⟨hP', hN', hQp, hQn, hmass⟩ := two_level_weighted_statistics k m wp wn a b ha hb
  obtain ⟨hpos, hneg⟩ := two_level_normalized_amplitudes _ _ a b hP hN ha hb
  have hbneg : -b < 0 := by linarith
  have hnb : ¬ 0 < -b := by linarith
  constructor
  · intro i
    unfold weightedNormalize
    rw [hP', hN', hQp, hQn, hmass]
    simpa [signScale, twoLevelReward, ha] using hpos
  · intro i
    unfold weightedNormalize
    rw [hP', hN', hQp, hQn, hmass]
    simpa [signScale, twoLevelReward, hnb, hbneg] using congrArg Neg.neg hneg

/-- Full composition: actual finite RLOO rewards, response lengths, and normalization. -/
theorem finite_binary_rloo_normalization (k m : ℕ) (hk : 0 < k) (hm : 0 < m)
    (hi lo : ℝ) (hgap : lo < hi) (wp : Fin k → ℝ) (wn : Fin m → ℝ)
    (hP : 0 < ∑ i, wp i) (hN : 0 < ∑ i, wn i) :
    (∀ i : Fin k, weightedNormalize (Sum.elim wp wn)
      (rlooSignal (twoLevelReward k m hi lo)) (Sum.inl i) =
        Real.sqrt ((∑ j, wn j) / (∑ j, wp j))) ∧
    (∀ i : Fin m, weightedNormalize (Sum.elim wp wn)
      (rlooSignal (twoLevelReward k m hi lo)) (Sum.inr i) =
        -Real.sqrt ((∑ j, wp j) / (∑ j, wn j))) := by
  have hk' : (1 : ℝ) ≤ k := by exact_mod_cast hk
  have hm' : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hd : 0 < (k : ℝ) + m - 1 := by linarith
  let a := (m : ℝ) * (hi - lo) / ((k : ℝ) + m - 1)
  let b := (k : ℝ) * (hi - lo) / ((k : ℝ) + m - 1)
  have ha : 0 < a := div_pos (mul_pos (by linarith) (sub_pos.mpr hgap)) hd
  have hb : 0 < b := div_pos (mul_pos (by linarith) (sub_pos.mpr hgap)) hd
  have hfun : rlooSignal (twoLevelReward k m hi lo) = twoLevelReward k m a (-b) := by
    funext i
    cases i with
    | inl i => exact rloo_success_amplitude k m hi lo (ne_of_gt hd) i
    | inr i => exact rloo_failure_amplitude k m hi lo (ne_of_gt hd) i
  rw [hfun]
  exact finite_two_level_normalization k m wp wn a b hP hN ha hb

noncomputable def multivaluedExampleRewards : Fin 3 → ℝ := ![0, 2 / 3, 1]
def multivaluedExampleLengths : Fin 3 → ℝ := ![1, 1, 4]

/-- A bounded three-level RLOO group: token centering flips a positive signal. -/
theorem multivalued_rloo_centering_counterexample :
    let A := rlooSignal multivaluedExampleRewards
    let w := multivaluedExampleLengths
    let μ := (∑ i, w i * A i) / (∑ i, w i)
    let v := (∑ i, w i * (A i - μ) ^ 2) / (∑ i, w i)
    A 0 = -5 / 6 ∧ A 1 = 1 / 6 ∧ A 2 = 2 / 3 ∧
      μ = 1 / 3 ∧ v = 11 / 36 ∧ A 1 - μ = -1 / 6 := by
  norm_num [rlooSignal, multivaluedExampleRewards, multivaluedExampleLengths, Fin.sum_univ_succ]
  change (1 : ℝ) - (5 / 3 - 1) / 2 = 2 / 3
  norm_num

theorem multivalued_rloo_zscore_negative :
    let A := rlooSignal multivaluedExampleRewards
    let w := multivaluedExampleLengths
    let μ := (∑ i, w i * A i) / (∑ i, w i)
    let v := (∑ i, w i * (A i - μ) ^ 2) / (∑ i, w i)
    (A 1 - μ) / Real.sqrt v < 0 := by
  obtain ⟨hA0, hA1, hA2, hμ, hv, hcenter⟩ := multivalued_rloo_centering_counterexample
  dsimp only at *
  rw [hcenter, hv]
  exact div_neg_of_neg_of_pos (by norm_num) (Real.sqrt_pos.mpr (by norm_num))

theorem multivalued_rloo_adaptive_positive :
    0 < weightedNormalize multivaluedExampleLengths (rlooSignal multivaluedExampleRewards) 1 := by
  norm_num [weightedNormalize, positiveTotal, negativeTotal, positiveSquare, negativeSquare,
    nonzeroMass, idealAlpha, idealBeta, signScale, rlooSignal, multivaluedExampleRewards,
    multivaluedExampleLengths, Fin.sum_univ_succ]

end REINFORCEProMax.AdaptiveMechanism
