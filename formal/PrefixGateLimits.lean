import TreePrefixGate

/-!
The implemented gate tests the current prefix average, rather than permanent
survival of all earlier tests. On an actual normalized two-action tree, a
reaccepted token can carry an arbitrarily large old/rollout coefficient.
-/
namespace REINFORCEProMax.AutoregressiveTree

noncomputable def binaryOddsPolicy (K : ℝ) (hK : 0 < K)
    (favored : List Bool → Bool) : Policy Bool where
  prob h a := if a = favored h then K / (K + 1) else 1 / (K + 1)
  nonneg := by
    intro h a
    split_ifs <;> positivity
  mass := by
    intro h
    rw [Fintype.sum_bool]
    cases hf : favored h <;> simp [hf] <;> field_simp <;> ring

theorem binaryOddsPolicy_positive (K : ℝ) (hK : 0 < K)
    (favored : List Bool → Bool) (h : List Bool) (a : Bool) :
    0 < (binaryOddsPolicy K hK favored).prob h a := by
  simp only [binaryOddsPolicy]
  split_ifs <;> positivity

noncomputable def cancellationRollout (K : ℝ) (hK : 0 < K) : Policy Bool :=
  binaryOddsPolicy K hK (fun h => h.isEmpty)

noncomputable def cancellationSnapshot (K : ℝ) (hK : 0 < K) : Policy Bool :=
  binaryOddsPolicy K hK (fun h => !h.isEmpty)

theorem cancellation_mutual_support (K : ℝ) (hK : 0 < K) :
    ∀ h a, (cancellationRollout K hK).prob h a = 0 ↔
      (cancellationSnapshot K hK).prob h a = 0 := by
  intro h a
  have hp := ne_of_gt (binaryOddsPolicy_positive K hK (fun h => h.isEmpty) h a)
  have hq := ne_of_gt (binaryOddsPolicy_positive K hK (fun h => !h.isEmpty) h a)
  exact iff_of_false hp hq

theorem cancellation_localTV (K : ℝ) (hK : 0 < K) (hK1 : 1 ≤ K) :
    localTV (cancellationRollout K hK) (cancellationSnapshot K hK) [] =
      (K - 1) / (K + 1) := by
  have hd : 0 < K + 1 := by linarith
  have hp : 0 ≤ K / (K + 1) - 1 / (K + 1) := by
    exact sub_nonneg.mpr ((div_le_div_iff_of_pos_right hd).mpr hK1)
  have hn : 1 / (K + 1) - K / (K + 1) ≤ 0 := by linarith
  simp only [localTV, Fintype.sum_bool, cancellationRollout,
    cancellationSnapshot, binaryOddsPolicy, List.isEmpty_nil, Bool.not_true,
    if_true, Bool.true_eq_false, Bool.false_eq_true, if_false]
  rw [abs_of_nonpos hn, abs_of_nonneg hp]
  ring

theorem cancellation_token_ratios (K : ℝ) (hK : 0 < K) :
    tokenRatio (cancellationRollout K hK) (cancellationSnapshot K hK) [] true = K⁻¹ ∧
    tokenRatio (cancellationRollout K hK) (cancellationSnapshot K hK) [true] true = K := by
  have hK0 := ne_of_gt hK
  have hd : K + 1 ≠ 0 := by positivity
  simp [tokenRatio, cancellationRollout, cancellationSnapshot, binaryOddsPolicy]
  constructor <;> field_simp

theorem cancellation_prefix_log_zero (K : ℝ) (hK : 0 < K) :
    logRatioTrace (cancellationRollout K hK) (cancellationSnapshot K hK)
      [true, true] = 0 := by
  obtain ⟨hfirst, hsecond⟩ := cancellation_token_ratios K hK
  have hs := logRatioTrace_snoc (cancellationRollout K hK)
    (cancellationSnapshot K hK) [true] true
  have hf := logRatioTrace_snoc (cancellationRollout K hK)
    (cancellationSnapshot K hK) [] true
  simp only [List.nil_append] at hf
  rw [List.cons_append, List.nil_append] at hs
  rw [hs, hf, hfirst, hsecond, Real.log_inv]
  simp [logRatioTrace, ratioTrace]

/-- No bound depending only on the prefix thresholds and a two-token horizon
can bound the retained token coefficient. All policies are normalized and
strictly positive, and the displayed accepted path has positive rollout mass. -/
theorem accepted_token_weight_unbounded (B τp τn : ℝ)
    (hτp : 0 ≤ τp) (hτn : 0 ≤ τn) :
    ∃ p q : Policy Bool,
      (∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) ∧
      0 < pathMass p [] [true, true] ∧
      prefixLogGate p q τp τn [true, true] = 1 ∧
      B < tokenRatio p q [true] true := by
  let K := max B 0 + 1
  have hK : 0 < K := by dsimp [K]; linarith [le_max_right B 0]
  refine ⟨cancellationRollout K hK, cancellationSnapshot K hK,
    cancellation_mutual_support K hK, ?_, ?_, ?_⟩
  · simp only [pathMass]
    have h₁ := binaryOddsPolicy_positive K hK (fun h => h.isEmpty) [] true
    have h₂ := binaryOddsPolicy_positive K hK (fun h => h.isEmpty) [true] true
    simpa [cancellationRollout] using mul_pos h₁ h₂
  · unfold prefixLogGate
    rw [cancellation_prefix_log_zero]
    simp [hτp, hτn]
  · rw [(cancellation_token_ratios K hK).2]
    dsimp [K]
    linarith [le_max_left B 0]

theorem cancellation_path_mass (K : ℝ) (hK : 0 < K) :
    pathMass (cancellationRollout K hK) [] [true, true] = K / (K + 1)^2 := by
  simp [pathMass, cancellationRollout, binaryOddsPolicy]
  field_simp
  simp [pow_two]

theorem cancellation_gate_accepts (K τp τn : ℝ) (hK : 0 < K)
    (hτp : 0 ≤ τp) (hτn : 0 ≤ τn) :
    prefixLogGate (cancellationRollout K hK) (cancellationSnapshot K hK)
      τp τn [true, true] = 1 := by
  unfold prefixLogGate
  rw [cancellation_prefix_log_zero]
  simp [hτp, hτn]

/-- The large coefficient is not made harmless by the rarity of this path:
its single nonnegative contribution to the retained-weight second moment
already diverges with K. -/
theorem cancellation_second_moment_contribution (K τp τn : ℝ) (hK : 0 < K)
    (hτp : 0 ≤ τp) (hτn : 0 ≤ τn) :
    pathMass (cancellationRollout K hK) [] [true, true] *
      (prefixLogGate (cancellationRollout K hK) (cancellationSnapshot K hK)
        τp τn [true, true] *
        tokenRatio (cancellationRollout K hK) (cancellationSnapshot K hK) [true] true)^2 =
      K^3 / (K + 1)^2 := by
  rw [cancellation_path_mass, cancellation_gate_accepts K τp τn hK hτp hτn,
    (cancellation_token_ratios K hK).2]
  ring

theorem cancellation_second_moment_lower_bound (K : ℝ) (hK : 1 ≤ K) :
    K / 4 ≤ K^3 / (K + 1)^2 := by
  have hd : 0 < (K + 1)^2 := sq_pos_of_pos (by linarith)
  apply (le_div_iff₀ hd).mpr
  have hsq : (K + 1)^2 ≤ 4 * K^2 := by nlinarith [sq_nonneg (K - 1)]
  have hprod := mul_le_mul_of_nonneg_left hsq (show 0 ≤ K by linarith)
  nlinarith

/-- Actual second moment under the complete two-step rollout distribution,
not a conditional-on-acceptance or conditional-on-one-path statistic. -/
noncomputable def retainedTokenSecondMoment (p q : Policy Bool) (τp τn : ℝ) : ℝ :=
  ∑ y : Fin 2 → Bool, pathMass p [] (List.ofFn y) *
    (prefixLogGate p q τp τn (List.ofFn y) * tokenRatio p q [y 0] (y 1))^2

theorem cancellation_retained_second_moment (K τp τn : ℝ) (hK : 0 < K)
    (hτp : 0 ≤ τp) (hτn : 0 ≤ τn) :
    K^3 / (K + 1)^2 ≤
      retainedTokenSecondMoment (cancellationRollout K hK)
        (cancellationSnapshot K hK) τp τn := by
  rw [← cancellation_second_moment_contribution K τp τn hK hτp hτn]
  have hs := Finset.single_le_sum
    (s := (Finset.univ : Finset (Fin 2 → Bool)))
    (f := fun y => pathMass (cancellationRollout K hK) [] (List.ofFn y) *
      (prefixLogGate (cancellationRollout K hK) (cancellationSnapshot K hK)
        τp τn (List.ofFn y) *
        tokenRatio (cancellationRollout K hK) (cancellationSnapshot K hK) [y 0] (y 1))^2)
    (fun y _ => mul_nonneg (pathMass_nonneg _ _ _) (sq_nonneg _))
    (Finset.mem_univ (fun _ => true))
  simpa [retainedTokenSecondMoment, List.ofFn_succ] using hs

theorem retained_second_moment_unbounded (B τp τn : ℝ)
    (hτp : 0 ≤ τp) (hτn : 0 ≤ τn) :
    ∃ p q : Policy Bool,
      (∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) ∧
      B < retainedTokenSecondMoment p q τp τn := by
  let K := max (4 * B) 1 + 1
  have hK1 : 1 ≤ K := by dsimp [K]; linarith [le_max_right (4 * B) 1]
  have hK : 0 < K := by linarith
  refine ⟨cancellationRollout K hK, cancellationSnapshot K hK,
    cancellation_mutual_support K hK, ?_⟩
  have hfirst : B < K / 4 := by dsimp [K]; linarith [le_max_left (4 * B) 1]
  exact lt_of_lt_of_le hfirst ((cancellation_second_moment_lower_bound K hK1).trans
    (cancellation_retained_second_moment K τp τn hK hτp hτn))

end REINFORCEProMax.AutoregressiveTree
