import TreeTV

/-!
Finite-history-tree version of the cumulative log-ratio / conditional-KL
identity. The recursion makes the prefix visitation weights implicit in the
normalized rollout tree. `chainKL` is the cumulative conditional reverse-KL
cost along that tree. The joint prefix KL is independently defined from the
two path-mass functions, and identified with this recursive conditional cost.
-/
namespace REINFORCEProMax.AutoregressiveTree

open scoped BigOperators
variable {A : Type*} [Fintype A]

noncomputable def localReverseKL (p q : Policy A) (h : List A) : ℝ :=
  ∑ a, p.prob h a * Real.log (tokenRatio q p h a)

noncomputable def chainLogRatio (p q : Policy A) : ℕ → List A → ℝ
  | 0, _ => 0
  | n + 1, h =>
      logRatioMean p q h + ∑ a, p.prob h a * chainLogRatio p q n (h ++ [a])

noncomputable def chainKL (p q : Policy A) : ℕ → List A → ℝ
  | 0, _ => 0
  | n + 1, h =>
      localReverseKL p q h + ∑ a, p.prob h a * chainKL p q n (h ++ [a])

theorem local_log_ratio_eq_neg_reverseKL
    (p q : Policy A) (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (h : List A) :
    logRatioMean p q h = - localReverseKL p q h := by
  dsimp [logRatioMean, localReverseKL]
  rw [← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl
  intro a ha
  by_cases hp : p.prob h a = 0
  · have hq : q.prob h a = 0 := (hs h a).mp hp
    simp [logRatioMean, localReverseKL, tokenRatio, hp, hq]
  · have hppos : 0 < p.prob h a := lt_of_le_of_ne (p.nonneg h a) (Ne.symm hp)
    have hqne : q.prob h a ≠ 0 := by
      intro hq
      exact hp ((hs h a).mpr hq)
    have hqpos : 0 < q.prob h a := lt_of_le_of_ne (q.nonneg h a) (Ne.symm hqne)
    have hpne : p.prob h a ≠ 0 := ne_of_gt hppos
    have hlog : Real.log (tokenRatio p q h a) =
        - Real.log (tokenRatio q p h a) := by
      dsimp [tokenRatio]
      rw [Real.log_div hqne hpne, Real.log_div hpne hqne]
      ring
    rw [hlog]
    ring

theorem chain_log_ratio_eq_neg_chain_kl
    (p q : Policy A) (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (n : ℕ) (h : List A) :
    chainLogRatio p q n h = - chainKL p q n h := by
  induction n generalizing h with
  | zero => simp [chainLogRatio, chainKL]
  | succ n ih =>
      simp only [chainLogRatio, chainKL]
      rw [local_log_ratio_eq_neg_reverseKL p q hs h]
      simp_rw [ih]
      simp_rw [mul_neg]
      rw [Finset.sum_neg_distrib]
      ring

/-- Cumulative sampled log-ratio recorded along a complete prefix. -/
noncomputable def logRatioTrace (p q : Policy A) (h : List A) : ℝ :=
  ((ratioTrace p q h).map Real.log).sum

theorem logRatioTrace_snoc (p q : Policy A) (h : List A) (a : A) :
    logRatioTrace p q (h ++ [a]) = logRatioTrace p q h + Real.log (tokenRatio p q h a) := by
  simp [logRatioTrace, ratioTrace_snoc]

/-- The recursion is the expectation of the actual trajectory log-ratio sum,
with the already observed prefix contributing a constant offset. -/
theorem trajectory_log_ratio_expectation (p q : Policy A) (n : ℕ) (h : List A) :
    value p (logRatioTrace p q) n h = logRatioTrace p q h + chainLogRatio p q n h := by
  induction n generalizing h with
  | zero => simp [value, chainLogRatio]
  | succ n ih =>
      simp only [value, ih, logRatioTrace_snoc, chainLogRatio, mul_add, Finset.sum_add_distrib]
      rw [← Finset.sum_mul, p.mass, one_mul]
      simp only [logRatioMean, add_assoc]

theorem pathMass_mutual_support (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (h y : List A) :
    pathMass p h y = 0 ↔ pathMass q h y = 0 := by
  induction y generalizing h with
  | nil => simp [pathMass]
  | cons a y ih => simp only [pathMass, mul_eq_zero, hs h a, ih]

/-- On positive rollout paths, the joint log density ratio is the sum of
the token log density ratios. Zero-mass paths are dealt with by weighting. -/
theorem pathMass_log_ratio (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (y : List A)
    (hp : pathMass p [] y ≠ 0) :
    Real.log (pathMass q [] y / pathMass p [] y) = logRatioTrace p q y := by
  induction y using List.reverseRecOn with
  | nil => simp [pathMass, logRatioTrace, ratioTrace]
  | append_singleton y a ih =>
      have hp' := (mul_ne_zero_iff.mp (show pathMass p [] y * p.prob y a ≠ 0 by
        simpa only [pathMass_snoc] using hp))
      have hq' : pathMass q [] y ≠ 0 := mt (pathMass_mutual_support p q hs [] y).mpr hp'.1
      have hqa : q.prob y a ≠ 0 := mt (hs y a).mpr hp'.2
      rw [pathMass_snoc, pathMass_snoc, mul_div_mul_comm, Real.log_mul
        (div_ne_zero hq' hp'.1) (div_ne_zero hqa hp'.2), ih hp'.1, logRatioTrace_snoc]
      rfl

/-- Joint-prefix reverse KL, defined from the finite probability masses,
using the usual zero-weight convention (Lean's real logarithm at zero is 0). -/
noncomputable def jointPrefixKL (p q : Policy A) (n : ℕ) : ℝ :=
  pathSum (fun y => pathMass p [] y * Real.log (pathMass p [] y / pathMass q [] y)) n []

theorem jointPrefixKL_eq_neg_expected_log_ratio (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (n : ℕ) :
    jointPrefixKL p q n = - value p (logRatioTrace p q) n [] := by
  have hpoint : (fun y => pathMass p [] y * Real.log (pathMass p [] y / pathMass q [] y)) =
      (fun y => (-1 : ℝ) * (pathMass p [] y * logRatioTrace p q y)) := by
    funext y
    by_cases hp : pathMass p [] y = 0
    · simp [hp]
    · have hq : pathMass q [] y ≠ 0 := mt (pathMass_mutual_support p q hs [] y).mpr hp
      have hl := pathMass_log_ratio p q hs y hp
      rw [Real.log_div hq hp] at hl
      rw [Real.log_div hp hq]
      rw [← hl]
      ring
  unfold jointPrefixKL
  rw [hpoint, pathSum_const_mul, pathSum_mass_value]
  simp [pathMass]

/-- Full finite autoregressive KL chain rule for the independently constructed
joint distribution. No visitation consistency or chain rule is assumed. -/
theorem jointPrefixKL_eq_chainKL (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (n : ℕ) :
    jointPrefixKL p q n = chainKL p q n [] := by
  rw [jointPrefixKL_eq_neg_expected_log_ratio p q hs,
    trajectory_log_ratio_expectation, chain_log_ratio_eq_neg_chain_kl p q hs]
  simp [logRatioTrace, ratioTrace]

theorem expected_log_ratio_eq_neg_jointPrefixKL (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (n : ℕ) :
    value p (logRatioTrace p q) n [] = -jointPrefixKL p q n := by
  rw [jointPrefixKL_eq_neg_expected_log_ratio p q hs, neg_neg]

end REINFORCEProMax.AutoregressiveTree
