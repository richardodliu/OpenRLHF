import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic

/-!
  Small, self-contained checks for the mathematical claims in the
  REINFORCE Pro Max manuscript.  The file deliberately avoids modelling a
  neural network or an optimiser: each theorem below formalises only an
  algebraic/analytic statement made in the paper.
-/

namespace REINFORCEProMax

section RLOO

/- The RLOO/mean-baseline relation, written with `s` for the sum of the
   rewards other than the current reward.  The side conditions express
   `n ≥ 2` without introducing a finite-sum API into this first check. -/
theorem rloo_scaling (n r s : ℝ) (hn0 : n ≠ 0) (hn1 : n ≠ 1) :
    r - (s + r) / n = ((n - 1) / n) * (r - s / (n - 1)) := by
  field_simp [hn0, sub_ne_zero.mpr hn1]
  ring

/- The leave-one-out shaped rewards have zero sum within a prompt group.  We
   write the sum over the other samples as `sum r - r i`; this is equal to the
   usual `erase i` sum by `Finset.sum_erase_eq_sub`. -/
theorem rloo_shaped_sum_zero {n : ℕ} (r : Fin n → ℝ) (hn : 2 ≤ n) :
    (∑ i : Fin n, (r i - ((∑ j : Fin n, r j) - r i) / ((n : ℝ) - 1))) = 0 := by
  have hn' : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hn1 : (n : ℝ) - 1 ≠ 0 := by linarith
  rw [Finset.sum_sub_distrib, ← Finset.sum_div, Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp [hn1]
  ring

theorem rloo_shaped_sum_zero_erase {n : ℕ} (r : Fin n → ℝ) (hn : 2 ≤ n) :
    (∑ i : Fin n,
      (r i - (∑ j ∈ (Finset.univ.erase i), r j) / ((n : ℝ) - 1))) = 0 := by
  have hrewrite : ∀ i : Fin n,
      (∑ j ∈ (Finset.univ.erase i), r j) = (∑ j : Fin n, r j) - r i := by
    intro i
    exact Finset.sum_erase_eq_sub (Finset.mem_univ i)
  simp_rw [hrewrite]
  exact rloo_shaped_sum_zero r hn

theorem all_nonpos_zero_of_sum_zero {n : ℕ} (a : Fin n → ℝ)
    (hsum : (∑ i : Fin n, a i) = 0) (hnonpos : ∀ i, a i ≤ 0) :
    ∀ i, a i = 0 := by
  have hnonneg : ∀ i, 0 ≤ -a i := by
    intro i
    linarith [hnonpos i]
  have hsumneg : (∑ i : Fin n, (-a i)) = 0 := by
    calc
      (∑ i : Fin n, (-a i)) = -(∑ i : Fin n, a i) := by simp
      _ = 0 := by rw [hsum]; simp
  have hallfun : (fun i => -a i) = 0 :=
    (Fintype.sum_eq_zero_iff_of_nonneg hnonneg).mp hsumneg
  intro i
  simpa using (show -a i = 0 by simpa using congrFun hallfun i)

theorem all_nonneg_zero_of_sum_zero {n : ℕ} (a : Fin n → ℝ)
    (hsum : (∑ i : Fin n, a i) = 0) (hnonneg : ∀ i, 0 ≤ a i) :
    ∀ i, a i = 0 := by
  have hallfun : (fun i => a i) = 0 :=
    (Fintype.sum_eq_zero_iff_of_nonneg hnonneg).mp hsum
  intro i
  exact congrFun hallfun i

/- Consequently, if all leave-one-out advantages have one sign, all are
   identically zero (for exact arithmetic). This is the degeneracy handled by
   the optional uniform-scale mode in the manuscript. -/
theorem rloo_zero_if_nonpositive {n : ℕ} (r : Fin n → ℝ) (hn : 2 ≤ n)
    (h : ∀ i : Fin n,
      r i - (∑ j : Fin n, r j - r i) / ((n : ℝ) - 1) ≤ 0) :
    ∀ i : Fin n,
      r i - (∑ j : Fin n, r j - r i) / ((n : ℝ) - 1) = 0 := by
  let a : Fin n → ℝ := fun i =>
    r i - (∑ j : Fin n, r j - r i) / ((n : ℝ) - 1)
  have hs : (∑ i : Fin n, a i) = 0 := by
    dsimp [a]
    exact rloo_shaped_sum_zero r hn
  have hz := all_nonpos_zero_of_sum_zero a hs (by intro i; exact h i)
  intro i
  exact hz i

theorem rloo_zero_if_nonnegative {n : ℕ} (r : Fin n → ℝ) (hn : 2 ≤ n)
    (h : ∀ i : Fin n, 0 ≤
      r i - (∑ j : Fin n, r j - r i) / ((n : ℝ) - 1)) :
    ∀ i : Fin n,
      r i - (∑ j : Fin n, r j - r i) / ((n : ℝ) - 1) = 0 := by
  let a : Fin n → ℝ := fun i =>
    r i - (∑ j : Fin n, r j - r i) / ((n : ℝ) - 1)
  have hs : (∑ i : Fin n, a i) = 0 := by
    dsimp [a]
    exact rloo_shaped_sum_zero r hn
  have hz := all_nonneg_zero_of_sum_zero a hs (by intro i; exact h i)
  intro i
  exact hz i

/- If every response contributes at least one active token, emptiness of one
   token-sign group is equivalent to all represented RLOO advantages having
   one weak sign; the response-level zero-sum identity then forces every
   shaped reward to vanish. -/
theorem rloo_token_sign_degeneracy {n : ℕ} {ι : Type*} [Fintype ι] [DecidableEq ι]
    (r : Fin n → ℝ) (active : Fin n → Finset ι) (hn : 2 ≤ n)
    (hactive : ∀ i, (active i).Nonempty) :
    ((∀ i t, t ∈ active i →
        r i - (∑ j : Fin n, r j - r i) / ((n : ℝ) - 1) ≤ 0) →
      ∀ i, r i - (∑ j : Fin n, r j - r i) / ((n : ℝ) - 1) = 0) ∧
    ((∀ i t, t ∈ active i →
        0 ≤ r i - (∑ j : Fin n, r j - r i) / ((n : ℝ) - 1)) →
      ∀ i, r i - (∑ j : Fin n, r j - r i) / ((n : ℝ) - 1) = 0) := by
  constructor
  · intro h
    apply rloo_zero_if_nonpositive r hn
    intro i
    obtain ⟨t, ht⟩ := hactive i
    exact h i t ht
  · intro h
    apply rloo_zero_if_nonnegative r hn
    intro i
    obtain ⟨t, ht⟩ := hactive i
    exact h i t ht

end RLOO

section AdaptiveNormalization

theorem adaptive_closed_form
    (P N Qp Qn m : ℝ)
    (hP : 0 < P) (hN : 0 < N) (hQp : 0 < Qp) (hQn : 0 < Qn) (hm : 0 < m) :
    let κ := P / N
    let α := Real.sqrt (m / (Qp + κ ^ 2 * Qn))
    let β := α * κ
    α > 0 ∧ β > 0 ∧ α * P - β * N = 0 ∧
      α ^ 2 * Qp + β ^ 2 * Qn = m := by
  dsimp
  have hN0 : N ≠ 0 := ne_of_gt hN
  have hden : 0 < Qp + (P / N) ^ 2 * Qn := by positivity
  have harg : 0 < m / (Qp + (P / N) ^ 2 * Qn) := div_pos hm hden
  have hsqrt : (Real.sqrt (m / (Qp + (P / N) ^ 2 * Qn))) ^ 2 =
      m / (Qp + (P / N) ^ 2 * Qn) := by
    exact Real.sq_sqrt (le_of_lt harg)
  have hα : 0 < Real.sqrt (m / (Qp + (P / N) ^ 2 * Qn)) := Real.sqrt_pos.2 harg
  have hκ : 0 < P / N := div_pos hP hN
  refine ⟨hα, mul_pos hα hκ, ?_, ?_⟩
  · field_simp [hN0]
    ring
  · calc
      (Real.sqrt (m / (Qp + (P / N) ^ 2 * Qn))) ^ 2 * Qp +
          (Real.sqrt (m / (Qp + (P / N) ^ 2 * Qn)) * (P / N)) ^ 2 * Qn =
          (Real.sqrt (m / (Qp + (P / N) ^ 2 * Qn))) ^ 2 *
            (Qp + (P / N) ^ 2 * Qn) := by ring
      _ = (m / (Qp + (P / N) ^ 2 * Qn)) * (Qp + (P / N) ^ 2 * Qn) := by rw [hsqrt]
      _ = m := by field_simp [ne_of_gt hden]

/- In the idealized ±1 reward case, the asymmetric scales have an exact
   length-mass form: their squared magnitudes are the opposite-sign token
   mass ratios. -/
theorem binary_reward_scaling_sq
    (P N : ℝ) (hP : 0 < P) (hN : 0 < N) :
    let κ := P / N
    let α := Real.sqrt ((P + N) / (P + κ ^ 2 * N))
    let β := α * κ
    α ^ 2 = N / P ∧ β ^ 2 = P / N := by
  dsimp
  have hP0 : P ≠ 0 := ne_of_gt hP
  have hN0 : N ≠ 0 := ne_of_gt hN
  have hden : 0 < P + (P / N) ^ 2 * N := by positivity
  have hsqrt : (Real.sqrt ((P + N) / (P + (P / N) ^ 2 * N))) ^ 2 =
      (P + N) / (P + (P / N) ^ 2 * N) := by
    exact Real.sq_sqrt (by positivity)
  have halpha : (Real.sqrt ((P + N) / (P + (P / N) ^ 2 * N))) ^ 2 = N / P := by
    rw [hsqrt]
    field_simp [hP0, hN0]
    ring
  constructor
  · exact halpha
  · rw [mul_pow, halpha]
    field_simp [hP0, hN0]
    ring

theorem positive_scaling_preserves_sign (a x : ℝ) (ha : 0 < a) :
    (a * x > 0 ↔ x > 0) ∧ (a * x < 0 ↔ x < 0) := by
  constructor
  · constructor
    · intro h
      have h' : a * 0 < a * x := by simpa using h
      exact (mul_lt_mul_left ha).mp h'
    · intro h
      exact mul_pos ha h
  · constructor
    · intro h
      have h' : a * x < a * 0 := by simpa using h
      exact (mul_lt_mul_left ha).mp h'
    · intro h
      exact mul_neg_of_pos_of_neg ha h

theorem positive_scaling_preserves_order (a x y : ℝ) (ha : 0 < a) :
    (a * x ≤ a * y ↔ x ≤ y) ∧ (a * x < a * y ↔ x < y) := by
  constructor
  · exact mul_le_mul_left ha
  · exact mul_lt_mul_left ha

/- The optional uniform-scale branch multiplies a sampled score sum by
   `r / n`.  For a positive group size this coefficient preserves the sign
   of the scalar reward, including the zero-reward case common with 0/1
   correctness rewards. -/
theorem uniform_scale_coefficient_sign (r n : ℝ) (hn : 0 < n) :
    (r / n > 0 ↔ r > 0) ∧ (r / n < 0 ↔ r < 0) ∧ (r / n = 0 ↔ r = 0) := by
  have hn0 : n ≠ 0 := ne_of_gt hn
  refine ⟨?_, ?_, ?_⟩
  · constructor
    · intro h
      rcases div_pos_iff.mp h with hpos | hneg
      · exact hpos.1
      · exact False.elim ((not_lt_of_ge (le_of_lt hn)) hneg.2)
    · intro hr
      exact div_pos hr hn
  · constructor
    · intro h
      rcases div_neg_iff.mp h with hpos | hneg
      · exact False.elim ((not_lt_of_ge (le_of_lt hn)) hpos.2)
      · exact hneg.1
    · intro hr
      exact div_neg_of_neg_of_pos hr hn
  · rw [div_eq_zero_iff, or_iff_left hn0]

end AdaptiveNormalization

section PrefixStability

theorem convex_combo_interval
    (lo hi a b θ : ℝ)
    (hlo : lo ≤ a) (ha : a ≤ hi) (hlb : lo ≤ b) (hb : b ≤ hi)
    (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) :
    lo ≤ θ * a + (1 - θ) * b ∧ θ * a + (1 - θ) * b ≤ hi := by
  constructor
  · nlinarith [mul_nonneg hθ0 (sub_nonneg.mpr hlo),
      mul_nonneg (sub_nonneg.mpr hθ1) (sub_nonneg.mpr hlb)]
  · nlinarith [mul_nonneg hθ0 (sub_nonneg.mpr ha),
      mul_nonneg (sub_nonneg.mpr hθ1) (sub_nonneg.mpr hb)]

/- This is the log-space core of Lemma `lem:prefix-stability`: the new
   prefix log-average is a convex combination of the old average and the
   next log-ratio. -/
theorem prefix_log_average_stability
    (lo hi old next θ : ℝ)
    (hold_lo : lo ≤ old) (hold_hi : old ≤ hi)
    (hnext_lo : lo ≤ next) (hnext_hi : next ≤ hi)
    (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) :
    lo ≤ θ * old + (1 - θ) * next ∧ θ * old + (1 - θ) * next ≤ hi := by
  exact convex_combo_interval lo hi old next θ hold_lo hold_hi hnext_lo hnext_hi hθ0 hθ1

theorem prefix_geomean_stability
    (lo hi old next θ : ℝ)
    (hlo : 0 < lo) (hold_lo : lo ≤ old) (hold_hi : old ≤ hi)
    (hnext_lo : lo ≤ next) (hnext_hi : next ≤ hi)
    (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) :
    lo ≤ Real.exp (θ * Real.log old + (1 - θ) * Real.log next) ∧
      Real.exp (θ * Real.log old + (1 - θ) * Real.log next) ≤ hi := by
  have hold_pos : 0 < old := lt_of_lt_of_le hlo hold_lo
  have hnext_pos : 0 < next := lt_of_lt_of_le hlo hnext_lo
  have hlog_lo_old : Real.log lo ≤ Real.log old :=
    (Real.log_le_log_iff hlo hold_pos).2 hold_lo
  have hlog_old_hi : Real.log old ≤ Real.log hi :=
    (Real.log_le_log_iff hold_pos (lt_of_lt_of_le hold_pos hold_hi)).2 hold_hi
  have hlog_lo_next : Real.log lo ≤ Real.log next :=
    (Real.log_le_log_iff hlo hnext_pos).2 hnext_lo
  have hlog_next_hi : Real.log next ≤ Real.log hi :=
    (Real.log_le_log_iff hnext_pos (lt_of_lt_of_le hnext_pos hnext_hi)).2 hnext_hi
  have hcombo := prefix_log_average_stability (Real.log lo) (Real.log hi)
    (Real.log old) (Real.log next) θ hlog_lo_old hlog_old_hi hlog_lo_next hlog_next_hi hθ0 hθ1
  constructor
  · calc
      lo = Real.exp (Real.log lo) := (Real.exp_log hlo).symm
      _ ≤ Real.exp (θ * Real.log old + (1 - θ) * Real.log next) :=
        (Real.exp_le_exp).2 hcombo.1
  · calc
      Real.exp (θ * Real.log old + (1 - θ) * Real.log next) ≤ Real.exp (Real.log hi) :=
        (Real.exp_le_exp).2 hcombo.2
      _ = hi := Real.exp_log (lt_of_lt_of_le hold_pos hold_hi)

/- If an accepted prefix has log-density-ratio average `s` in
   `[-τn, τp]`, its density-ratio residual is bounded by the larger of the
   two endpoint residuals.  This is the finite real/log-space core of the
   accepted-prefix bound in `thm:pre-causal-mechanism`; it does not model
   the surrounding probability space or the prefix mask itself. -/
theorem accepted_prefix_product_deviation_bound
    (s τp τn : ℝ)
    (hs_lo : -τn ≤ s) (hs_hi : s ≤ τp)
    (hτp : 0 ≤ τp) (hτn : 0 ≤ τn) :
    |Real.exp s - 1| ≤
      max (Real.exp τp - 1) (1 - Real.exp (-τn)) := by
  have hupper_nonneg : 0 ≤ Real.exp τp - 1 := by
    have hmon : Real.exp 0 ≤ Real.exp τp := Real.exp_le_exp.mpr hτp
    simpa using sub_nonneg.mpr hmon
  have hlower_nonneg : 0 ≤ 1 - Real.exp (-τn) := by
    have hneg : -τn ≤ 0 := by linarith
    have hmon : Real.exp (-τn) ≤ Real.exp 0 := Real.exp_le_exp.mpr hneg
    simpa using sub_nonneg.mpr hmon
  by_cases hs_nonneg : 0 ≤ s
  · have hexp_nonneg : 0 ≤ Real.exp s - 1 := by
      have hmon : Real.exp 0 ≤ Real.exp s := Real.exp_le_exp.mpr hs_nonneg
      simpa using sub_nonneg.mpr hmon
    rw [abs_of_nonneg hexp_nonneg]
    have hmon : Real.exp s ≤ Real.exp τp := Real.exp_le_exp.mpr hs_hi
    exact le_max_of_le_left (by linarith [hupper_nonneg])
  · have hs_nonpos : s ≤ 0 := le_of_not_ge hs_nonneg
    have hexp_le_one : Real.exp s ≤ 1 := by
      have hmon : Real.exp s ≤ Real.exp 0 := Real.exp_le_exp.mpr hs_nonpos
      simpa using hmon
    rw [abs_of_nonpos (sub_nonpos.mpr hexp_le_one)]
    have hmon : Real.exp (-τn) ≤ Real.exp s :=
      Real.exp_le_exp.mpr hs_lo
    exact le_max_of_le_right (by linarith [hlower_nonneg])

/- A deterministic bridge for the implementation's two-ratio form.  If `z` is
   the rollout-to-old log-ratio prefix and `u` is the old-to-current log-ratio,
   a uniform per-token bound on `u` moves every prefix arithmetic mean by at
   most that bound.  This is the finite-sequence statement needed to transfer
   a prefix gate computed from cached old/rollout log-probabilities to the
   current-policy ratio, provided the PPO update itself stays in a trust
   region. -/
theorem prefix_average_additive_perturbation
    {n : ℕ} (hn : 0 < n) (z u : Fin n → ℝ) (η : ℝ)
    (hu : ∀ i, |u i| ≤ η) :
    |((∑ i : Fin n, (z i + u i)) / (n : ℝ)) -
      ((∑ i : Fin n, z i) / (n : ℝ))| ≤ η := by
  have hn' : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hlow : -(n : ℝ) * η ≤ ∑ i : Fin n, u i := by
    have hsum : ∑ i : Fin n, (-η) ≤ ∑ i : Fin n, u i :=
      Finset.sum_le_sum (fun i hi => (abs_le.mp (hu i)).1)
    simpa [nsmul_eq_mul] using hsum
  have hupp : ∑ i : Fin n, u i ≤ (n : ℝ) * η := by
    have hsum : ∑ i : Fin n, u i ≤ ∑ i : Fin n, η :=
      Finset.sum_le_sum (fun i hi => (abs_le.mp (hu i)).2)
    simpa [nsmul_eq_mul] using hsum
  have hdiff :
      ((∑ i : Fin n, (z i + u i)) / (n : ℝ)) -
        ((∑ i : Fin n, z i) / (n : ℝ)) =
        (∑ i : Fin n, u i) / (n : ℝ) := by
    rw [Finset.sum_add_distrib]
    ring
  rw [hdiff]
  rw [abs_le]
  constructor
  · apply (le_div_iff₀ hn').2
    nlinarith
  · apply (div_le_iff₀ hn').2
    nlinarith

/- The same perturbation estimate written for an arbitrary finite active set.
   This is the form used when EOS padding is present: the active positions are
   selected by a finite set, while padding positions are absent from both the
   numerator and denominator. -/
theorem finset_average_additive_perturbation
    {ι : Type*} [DecidableEq ι] (s : Finset ι) (hs : s.Nonempty)
    (z u : ι → ℝ) (η : ℝ) (hu : ∀ i ∈ s, |u i| ≤ η) :
    |((∑ i ∈ s, (z i + u i)) / (s.card : ℝ)) -
      ((∑ i ∈ s, z i) / (s.card : ℝ))| ≤ η := by
  have hcard : 0 < (s.card : ℝ) := by
    exact_mod_cast hs.card_pos
  have hlow : -(s.card : ℝ) * η ≤ ∑ i ∈ s, u i := by
    have hsum : ∑ i ∈ s, (-η) ≤ ∑ i ∈ s, u i := by
      apply Finset.sum_le_sum
      intro i hi
      exact (abs_le.mp (hu i hi)).1
    simpa [nsmul_eq_mul] using hsum
  have hupp : ∑ i ∈ s, u i ≤ (s.card : ℝ) * η := by
    have hsum : ∑ i ∈ s, u i ≤ ∑ i ∈ s, η := by
      apply Finset.sum_le_sum
      intro i hi
      exact (abs_le.mp (hu i hi)).2
    simpa [nsmul_eq_mul] using hsum
  have hdiff :
      ((∑ i ∈ s, (z i + u i)) / (s.card : ℝ)) -
        ((∑ i ∈ s, z i) / (s.card : ℝ)) =
        (∑ i ∈ s, u i) / (s.card : ℝ) := by
    rw [Finset.sum_add_distrib]
    ring
  rw [hdiff, abs_le]
  constructor
  · apply (le_div_iff₀ hcard).2
    nlinarith
  · apply (div_le_iff₀ hcard).2
    nlinarith

/- Exact log-space composition of the PPO current/old ratio and the
   old/rollout correction ratio. -/
theorem log_ratio_composition
    (r w ρ : ℝ) (hr : 0 < r) (hw : 0 < w) (hρ : ρ = r * w) :
    Real.log ρ = Real.log r + Real.log w := by
  rw [hρ, Real.log_mul (ne_of_gt hr) (ne_of_gt hw)]

theorem single_spike_dilution
    {t T : ℕ} (ht : 0 < t) (hle : t ≤ T) {x : ℝ} (hx : 0 ≤ x) :
    x / (T : ℝ) ≤ x / (t : ℝ) := by
  have ht' : (0 : ℝ) < (t : ℝ) := by exact_mod_cast ht
  have hT' : (0 : ℝ) < (T : ℝ) := lt_of_lt_of_le ht' (by exact_mod_cast hle)
  rw [div_le_div_iff₀ hT' ht']
  nlinarith [hx, (show (t : ℝ) ≤ T from by exact_mod_cast hle)]

/- A finite-support representation of the prefix statistic.  `prefixSum z j`
   sums positions at most `j`, and `prefixAverage` divides by the number of
   positions in that prefix.  The definitions are intentionally finite: they
   expose exactly the arithmetic used by the prefix mask without modelling a
   policy or a probability space. -/
def prefixSum {n : ℕ} (z : Fin n → ℝ) (j : Fin n) : ℝ :=
  ∑ i ∈ (Finset.univ.filter (fun i : Fin n => i ≤ j)), z i

noncomputable def prefixAverage {n : ℕ} (z : Fin n → ℝ) (j : Fin n) : ℝ :=
  prefixSum z j / ((j.val + 1 : ℕ) : ℝ)

/- Finite predicates matching the three mask granularities in the manuscript.
   They are stated in log space, so exponentiating the interval gives the
   implementation's positive ratio interval. -/
def intervalAccepted (lo hi x : ℝ) : Prop := lo ≤ x ∧ x ≤ hi

def tokenAccepted {n : ℕ} (z : Fin n → ℝ) (lo hi : ℝ) (i : Fin n) : Prop :=
  intervalAccepted lo hi (z i)

def prefixAccepted {n : ℕ} (z : Fin n → ℝ) (lo hi : ℝ) (i : Fin n) : Prop :=
  intervalAccepted lo hi (prefixAverage z i)

def sequenceAccepted {n : ℕ} (z : Fin n → ℝ) (lo hi : ℝ) : Prop :=
  intervalAccepted lo hi ((∑ i : Fin n, z i) / (n : ℝ))

/- One isolated log-ratio spike has prefix average `x/(k+1)` at its
   location and sequence average `x/n`.  This is the exact finite statement
   behind the dilution comparison in `thm:prefix-tighter` (Part (b)). -/
def singleSpike {n : ℕ} (k : Fin n) (x : ℝ) (i : Fin n) : ℝ :=
  if i = k then x else 0

theorem single_spike_prefix_formulas
    {n : ℕ} (k : Fin n) (x : ℝ) :
    prefixAverage (singleSpike k x) k = x / ((k.val + 1 : ℕ) : ℝ) ∧
      (∑ i : Fin n, singleSpike k x i) / (n : ℝ) = x / (n : ℝ) := by
  constructor
  · rw [prefixAverage]
    congr 1
    classical
    simp [prefixSum, singleSpike, Finset.sum_ite_irrel, Finset.sum_ite_eq',
      Finset.mem_filter]
  · congr 1
    classical
    simp [singleSpike]

/- If the positive spike is outside the prefix threshold at `k` but its
   sequence-level geometric-mean proxy is in-bound, prefix masking rejects
   the spike while sequence masking accepts the whole sequence.  The
   hypotheses are expressed in log space, where the mask is an interval test. -/
theorem single_spike_prefix_rejects_sequence_accepts
    {n : ℕ} (hn : 0 < n) (k : Fin n) (x lo hi : ℝ)
    (hx : 0 < x) (hlo : lo ≤ 0)
    (hpref : hi * ((k.val + 1 : ℕ) : ℝ) < x)
    (hseq : x / (n : ℝ) ≤ hi) :
    hi < prefixAverage (singleSpike k x) k ∧
      lo ≤ (∑ i : Fin n, singleSpike k x i) / (n : ℝ) ∧
      (∑ i : Fin n, singleSpike k x i) / (n : ℝ) ≤ hi := by
  have hform := single_spike_prefix_formulas k x
  rw [hform.1, hform.2]
  have hden : 0 < ((k.val + 1 : ℕ) : ℝ) := by positivity
  have hn' : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  exact ⟨(lt_div_iff₀ hden).2 hpref, le_trans hlo (le_of_lt (div_pos hx hn')), hseq⟩

/- In the same positive-spike regime, the token-level interval test rejects
   exactly the spike and accepts every other (zero) log-ratio token. -/
theorem single_spike_token_interval_pattern
    {n : ℕ} (k : Fin n) (x lo hi : ℝ)
    (hlo : lo ≤ 0) (hhi : 0 ≤ hi) (hx : hi < x) :
    ¬ (lo ≤ singleSpike k x k ∧ singleSpike k x k ≤ hi) ∧
      ∀ i : Fin n, i ≠ k →
        lo ≤ singleSpike k x i ∧ singleSpike k x i ≤ hi := by
  constructor
  · simp [singleSpike]
    intro _
    exact hx
  · intro i hik
    simp [singleSpike, hik, hlo, hhi]

/- Exponentiating the log-space separation gives the corresponding ratio-space
   statement used by the actual prefix and sequence masks. -/
theorem single_spike_prefix_geomean_separation
    {n : ℕ} (hn : 0 < n) (k : Fin n) (x lo hi : ℝ)
    (hx : 0 < x) (hlo : lo ≤ 0)
    (hpref : hi * ((k.val + 1 : ℕ) : ℝ) < x)
    (hseq : x / (n : ℝ) ≤ hi) :
    Real.exp hi < Real.exp (prefixAverage (singleSpike k x) k) ∧
      Real.exp lo ≤ Real.exp ((∑ i : Fin n, singleSpike k x i) / (n : ℝ)) ∧
      Real.exp ((∑ i : Fin n, singleSpike k x i) / (n : ℝ)) ≤ Real.exp hi := by
  have h := single_spike_prefix_rejects_sequence_accepts hn k x lo hi hx hlo hpref hseq
  exact ⟨(Real.exp_lt_exp).2 h.1, (Real.exp_le_exp).2 h.2.1, (Real.exp_le_exp).2 h.2.2⟩

/- The integrated finite single-spike separation used in the late-deviation
   discussion: the token test rejects exactly the spike, the prefix test
   rejects its position, and the sequence average remains accepted. -/
theorem finite_single_spike_mask_separation
    {n : ℕ} (hn : 0 < n) (k : Fin n) (x lo hi : ℝ)
    (hlo : lo ≤ 0) (hhi : 0 ≤ hi) (hx : hi < x)
    (hpref : hi * ((k.val + 1 : ℕ) : ℝ) < x)
    (hseq : x / (n : ℝ) ≤ hi) :
    (¬ (lo ≤ singleSpike k x k ∧ singleSpike k x k ≤ hi)) ∧
      (∀ i : Fin n, i ≠ k → lo ≤ singleSpike k x i ∧ singleSpike k x i ≤ hi) ∧
      hi < prefixAverage (singleSpike k x) k ∧
      lo ≤ (∑ i : Fin n, singleSpike k x i) / (n : ℝ) ∧
      (∑ i : Fin n, singleSpike k x i) / (n : ℝ) ≤ hi := by
  have htok := single_spike_token_interval_pattern k x lo hi hlo hhi hx
  have hsep := single_spike_prefix_rejects_sequence_accepts hn k x lo hi
    (by linarith) hlo hpref hseq
  exact ⟨htok.1, htok.2, hsep.1, hsep.2.1, hsep.2.2⟩

/- The separation regime is nonempty whenever the spike prefix is shorter
   than the full sequence.  For any positive upper log-threshold `τ`, choosing
   `x = τ (k+1+n)/2` makes the prefix average exceed `τ` while the sequence
   average remains at most `τ`.  This turns the qualitative ``there exist
   regimes'' statement in Part (b) into an explicit finite witness. -/
theorem exists_single_spike_prefix_sequence_separation
    {n : ℕ} (hn : 0 < n) (k : Fin n) (hk : k.val + 1 < n)
    (τ : ℝ) (hτ : 0 < τ) :
    ∃ x : ℝ,
      τ < prefixAverage (singleSpike k x) k ∧
      0 ≤ (∑ i : Fin n, singleSpike k x i) / (n : ℝ) ∧
      (∑ i : Fin n, singleSpike k x i) / (n : ℝ) ≤ τ := by
  let d : ℝ := ((k.val + 1 : ℕ) : ℝ)
  let N : ℝ := (n : ℝ)
  let x : ℝ := τ * (d + N) / 2
  have hd : 0 < d := by dsimp [d]; positivity
  have hN : 0 < N := by dsimp [N]; exact_mod_cast hn
  have hdk : d < N := by
    dsimp [d, N]
    exact_mod_cast hk
  have hx : 0 < x := by
    dsimp [x]
    positivity
  refine ⟨x, ?_, ?_, ?_⟩
  · rw [(single_spike_prefix_formulas k x).1]
    apply (lt_div_iff₀ hd).2
    dsimp [x]
    nlinarith
  · rw [(single_spike_prefix_formulas k x).2]
    positivity
  · rw [(single_spike_prefix_formulas k x).2]
    apply (div_le_iff₀ hN).2
    dsimp [x]
    nlinarith

/- Every position strictly before a single spike has zero prefix average.
   Consequently, whenever zero lies in the acceptance interval, all prefixes
   before a late deviation are accepted, exactly as claimed in Part (b). -/
theorem single_spike_late_prefix_zero
    {n : ℕ} {j k : Fin n} (x : ℝ) (hjk : j < k) :
    prefixAverage (singleSpike k x) j = 0 := by
  rw [prefixAverage]
  have hk : k ∉ (Finset.univ.filter (fun i : Fin n => i ≤ j)) := by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact not_le_of_gt hjk
  simp [prefixSum, singleSpike, hk]

theorem single_spike_late_prefix_accepted
    {n : ℕ} {j k : Fin n} (x lo hi : ℝ) (hjk : j < k)
    (hlo : lo ≤ 0) (hhi : 0 ≤ hi) :
    lo ≤ prefixAverage (singleSpike k x) j ∧
      prefixAverage (singleSpike k x) j ≤ hi := by
  rw [single_spike_late_prefix_zero x hjk]
  exact ⟨hlo, hhi⟩

/- Early deviation with bounded subsequent noise.  Writing `m+1` for
   the prefix length avoids partial indexing: position `0` is the initial
   deviation and every `i.succ` is a later position.  These are the exact
   sufficient rejection windows of Part (a), with both signs included. -/
theorem early_deviation_upper_window
    {m : ℕ} (z : Fin (m + 1) → ℝ) (ε τ : ℝ)
    (hnoise : ∀ i : Fin m, -ε ≤ z i.succ)
    (hden : 0 < τ + ε)
    (hwindow : ((m + 1 : ℕ) : ℝ) < (z 0 + ε) / (τ + ε)) :
    τ < (∑ i : Fin (m + 1), z i) / ((m + 1 : ℕ) : ℝ) := by
  have hsum : (m : ℝ) * (-ε) ≤ ∑ i : Fin m, z i.succ := by
    calc
      (m : ℝ) * (-ε) = ∑ i : Fin m, (-ε) := by simp [nsmul_eq_mul]
      _ ≤ ∑ i : Fin m, z i.succ := Finset.sum_le_sum (fun i hi => hnoise i)
  have hwindow' := (lt_div_iff₀ hden).1 hwindow
  have hm : (0 : ℝ) < ((m + 1 : ℕ) : ℝ) := by positivity
  rw [Fin.sum_univ_succ]
  apply (lt_div_iff₀ hm).2
  push_cast at hwindow' ⊢
  nlinarith

theorem early_deviation_lower_window
    {m : ℕ} (z : Fin (m + 1) → ℝ) (ε τ : ℝ)
    (hnoise : ∀ i : Fin m, z i.succ ≤ ε)
    (hden : 0 < τ + ε)
    (hwindow : ((m + 1 : ℕ) : ℝ) < (-z 0 + ε) / (τ + ε)) :
    (∑ i : Fin (m + 1), z i) / ((m + 1 : ℕ) : ℝ) < -τ := by
  have hsum : (∑ i : Fin m, z i.succ) ≤ (m : ℝ) * ε := by
    calc
      (∑ i : Fin m, z i.succ) ≤ ∑ i : Fin m, ε :=
        Finset.sum_le_sum (fun i hi => hnoise i)
      _ = (m : ℝ) * ε := by simp [nsmul_eq_mul]
  have hwindow' := (lt_div_iff₀ hden).1 hwindow
  have hm : (0 : ℝ) < ((m + 1 : ℕ) : ℝ) := by positivity
  rw [Fin.sum_univ_succ]
  apply (div_lt_iff₀ hm).2
  push_cast at hwindow' ⊢
  nlinarith

/- Any nonempty finite prefix whose entries all lie in an interval has its
   arithmetic mean in the same interval.  This is the list form of the
   repeated convex-combination argument used for late deviations in Part
   (b), and is useful when a prefix is represented independently of its
   ambient trajectory. -/
theorem list_average_interval
    (xs : List ℝ) (lo hi : ℝ)
    (hlo : ∀ x ∈ xs, lo ≤ x) (hhi : ∀ x ∈ xs, x ≤ hi)
    (hn : xs ≠ []) :
    lo ≤ xs.sum / (xs.length : ℝ) ∧ xs.sum / (xs.length : ℝ) ≤ hi := by
  have hlow : (xs.length : ℝ) * lo ≤ xs.sum := by
    have h := List.sum_le_sum (l := xs) (f := fun _ : ℝ => lo) (g := fun x => x) hlo
    simpa [List.map_const, nsmul_eq_mul] using h
  have hhigh : xs.sum ≤ (xs.length : ℝ) * hi := by
    have h := List.sum_le_sum (l := xs) (f := fun x => x) (g := fun _ : ℝ => hi) hhi
    simpa [List.map_const, nsmul_eq_mul] using h
  have hlen : (0 : ℝ) < (xs.length : ℝ) := by
    exact_mod_cast (List.length_pos_iff.mpr hn)
  constructor
  · apply (le_div_iff₀ hlen).2
    nlinarith [hlow]
  · apply (div_le_iff₀ hlen).2
    nlinarith [hhigh]

/- Monotone violation: if every token log-ratio lies strictly above the
   upper threshold, every finite-horizon prefix/sequence average also lies
   above it.  The lower version is stated separately. -/
theorem monotone_violation_average_gt
    {n : ℕ} (hn : 0 < n) (z : Fin n → ℝ) (τ : ℝ)
    (h : ∀ i, τ < z i) :
    τ < (∑ i : Fin n, z i) / (n : ℝ) := by
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := by
    refine ⟨⟨0, hn⟩, ?_⟩
    simp
  have hs : (∑ i : Fin n, τ) < ∑ i : Fin n, z i := by
    apply Finset.sum_lt_sum_of_nonempty hne
    intro i hi
    exact h i
  have hn' : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hc : (∑ i : Fin n, τ) = (n : ℝ) * τ := by simp [nsmul_eq_mul]
  rw [hc] at hs
  exact (lt_div_iff₀ hn').2 (by nlinarith)

theorem monotone_violation_average_lt
    {n : ℕ} (hn : 0 < n) (z : Fin n → ℝ) (τ : ℝ)
    (h : ∀ i, z i < -τ) :
    (∑ i : Fin n, z i) / (n : ℝ) < -τ := by
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := by
    refine ⟨⟨0, hn⟩, ?_⟩
    simp
  have hs : (∑ i : Fin n, z i) < ∑ i : Fin n, (-τ) := by
    apply Finset.sum_lt_sum_of_nonempty hne
    intro i hi
    exact h i
  have hn' : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hc : (∑ i : Fin n, (-τ)) = (n : ℝ) * (-τ) := by simp [nsmul_eq_mul]
  rw [hc] at hs
  exact (div_lt_iff₀ hn').2 (by nlinarith)

/- Finite-measure step used in the accepted-prefix drift bound.  The mask is
   allowed to be any nonnegative weight (binary masks are the special case
   `mask i ∈ {0,1}`), and the pointwise residual bound is required only where
   the mask has positive mass. -/
theorem finite_masked_expectation_bound
    {n : ℕ} (weight mask residual : Fin n → ℝ) (B : ℝ)
    (hweight : ∀ i, 0 ≤ weight i)
    (hmask : ∀ i, 0 ≤ mask i)
    (hpoint : ∀ i, 0 < mask i → residual i ≤ B) :
    (∑ i : Fin n, weight i * mask i * residual i) ≤
      B * (∑ i : Fin n, weight i * mask i) := by
  have hterm : ∀ i : Fin n,
      weight i * mask i * residual i ≤ weight i * mask i * B := by
    intro i
    by_cases hm : 0 < mask i
    · have hmul : 0 ≤ weight i * mask i := mul_nonneg (hweight i) (hmask i)
      exact mul_le_mul_of_nonneg_left (hpoint i hm) hmul
    · have hm0 : mask i = 0 := le_antisymm (not_lt.mp hm) (hmask i)
      simp [hm0]
  calc
    (∑ i : Fin n, weight i * mask i * residual i) ≤
        ∑ i : Fin n, weight i * mask i * B :=
      Finset.sum_le_sum (fun i hi => hterm i)
    _ = B * (∑ i : Fin n, weight i * mask i) := by
      rw [← Finset.sum_mul]
      ring

end PrefixStability

section Counterexamples

/- The appendix proof invokes `exp u ≥ 1 + u + u²/2` for every real `u`.
   This theorem records the exact counterexample `u = -2`. -/
theorem exp_quadratic_claim_is_false :
    ¬ (∀ u : ℝ, Real.exp u ≥ 1 + u + u ^ 2 / 2) := by
  intro h
  have hbad := h (-2 : ℝ)
  have hexp : Real.exp (-2 : ℝ) < 1 := (Real.exp_lt_one_iff).2 (by norm_num)
  norm_num at hbad

/- A finite two-point instance of the support assumption in the paper:
   p=(1,0), q=(1/2,1/2).  The displayed ratio identity gives 1/4 while
   total variation is 1/2, because q has mass outside supp(p). -/
theorem tv_ratio_identity_counterexample :
    let tv : ℝ := (1 / 2 : ℝ) * (|1 - (1 / 2 : ℝ)| + |0 - (1 / 2 : ℝ)|)
    let ratio_expectation : ℝ := (1 / 2 : ℝ) *
      (1 * |(1 / 2 : ℝ) / 1 - 1| + 0 * |(1 / 2 : ℝ) / 0 - 1|)
    tv = 1 / 2 ∧ ratio_expectation = 1 / 4 ∧ tv ≠ ratio_expectation := by
  norm_num [abs_of_nonneg, abs_of_nonpos]

noncomputable def counterexampleP : Fin 2 → ℝ := fun y => if y = 0 then 1 else 0

noncomputable def counterexampleQ : Fin 2 → ℝ := fun _ => 1 / 2

noncomputable def counterexampleRatio (y : Fin 2) : ℝ :=
  if counterexampleP y = 0 then 0 else counterexampleQ y / counterexampleP y

theorem tv_ratio_counterexample_has_paper_support :
    (∀ y, 0 ≤ counterexampleP y) ∧
    (∀ y, 0 ≤ counterexampleQ y) ∧
    (∀ y, counterexampleP y > 0 → counterexampleQ y > 0) ∧
    (∑ y : Fin 2, counterexampleP y) = 1 ∧
    (∑ y : Fin 2, counterexampleQ y) = 1 := by
  constructor
  · intro y
    fin_cases y <;> norm_num [counterexampleP]
  constructor
  · intro y
    norm_num [counterexampleQ]
  constructor
  · intro y hy
    fin_cases y <;> norm_num [counterexampleP, counterexampleQ] at hy ⊢
  constructor <;> norm_num [counterexampleP, counterexampleQ, Fin.sum_univ_two]

theorem tv_ratio_counterexample_explicit :
    (1 / 2 : ℝ) * (∑ y : Fin 2, |counterexampleP y - counterexampleQ y|) = 1 / 2 ∧
    (1 / 2 : ℝ) * (∑ y : Fin 2,
      counterexampleP y * |counterexampleRatio y - 1|) = 1 / 4 := by
  constructor <;> norm_num [counterexampleP, counterexampleQ, counterexampleRatio,
    Fin.sum_univ_two, abs_of_nonneg, abs_of_nonpos]

theorem counterexample_prefix_ratio_mass_is_not_one :
    (∑ y : Fin 2, counterexampleP y * counterexampleRatio y) = 1 / 2 := by
  norm_num [counterexampleP, counterexampleQ, counterexampleRatio, Fin.sum_univ_two]

/- The paper's `k₃` proxy calculation uses the same missing identity
   `E_p[ρ] = 1`.  On the same two-point distributions its expectation is
   `log 2 - 1/2`, whereas `KL(p || q) = log 2`. -/
theorem k3_proxy_counterexample :
    let lhs : ℝ := ∑ y : Fin 2, counterexampleP y *
      (counterexampleRatio y - 1 - Real.log (counterexampleRatio y))
    let rhs : ℝ := ∑ y : Fin 2, counterexampleP y *
      Real.log (counterexampleP y / counterexampleQ y)
    lhs ≠ rhs := by
  norm_num [counterexampleP, counterexampleQ, counterexampleRatio, Fin.sum_univ_two]
  have hlog : Real.log (1 / 2 : ℝ) = -Real.log 2 := by
    rw [show (1 / 2 : ℝ) = (2 : ℝ)⁻¹ by norm_num, Real.log_inv]
  rw [hlog]
  linarith

/- A one-step (T=1) finite-horizon instance.  The rollout policy p puts all
   mass on a zero-reward trajectory, while q assigns additional mass to a
   reward-one trajectory.  Any rollout expectation of the ratio terms misses
   this q-only trajectory, so the claimed exact surrogate identity fails. -/
noncomputable def counterexampleReward : Fin 2 → ℝ := fun y => if y = 1 then 1 else 0

theorem finite_horizon_surrogate_counterexample :
    let jp : ℝ := ∑ y : Fin 2, counterexampleP y * counterexampleReward y
    let jq : ℝ := ∑ y : Fin 2, counterexampleQ y * counterexampleReward y
    let rollout_surrogate : ℝ :=
      ∑ y : Fin 2, counterexampleP y * counterexampleReward y *
        (counterexampleRatio y - 1)
    let remainder : ℝ := 0
    jq - jp ≠ rollout_surrogate - remainder := by
  norm_num [counterexampleP, counterexampleQ, counterexampleReward,
    counterexampleRatio, Fin.sum_univ_two]

/- The ratio form of TV is valid after adding the missing reverse absolute
   continuity condition `q > 0 -> p > 0`.  This finite-support theorem is the
   corrected version of the manuscript's one-line identity. -/
theorem tv_ratio_identity_of_reverse_support
    {n : ℕ} (p q : Fin n → ℝ)
    (hp : ∀ y, 0 ≤ p y) (hq : ∀ y, 0 ≤ q y)
    (hqp : ∀ y, q y > 0 → p y > 0) :
    (1 / 2 : ℝ) * (∑ y : Fin n, |p y - q y|) =
      (1 / 2 : ℝ) * (∑ y : Fin n, p y * |(if p y = 0 then 0 else q y / p y) - 1|) := by
  have hpoint : ∀ y : Fin n,
      |p y - q y| = p y * |(if p y = 0 then 0 else q y / p y) - 1| := by
    intro y
    by_cases hpy : p y = 0
    · have hqy : q y = 0 := by
        by_contra hqyn
        have hqyp : q y > 0 := lt_of_le_of_ne (hq y) (Ne.symm hqyn)
        exact (ne_of_gt (hqp y hqyp)) hpy
      simp [hpy, hqy]
    · have hpypos : 0 < p y := lt_of_le_of_ne (hp y) (Ne.symm hpy)
      have hrewrite : q y / p y - 1 = (q y - p y) / p y := by
        field_simp [hpy]
      simp [hpy]
      rw [hrewrite, abs_div, abs_of_pos hpypos]
      have habs : |p y - q y| = |q y - p y| := by
        rw [show p y - q y = -(q y - p y) by ring, abs_neg]
      rw [habs]
      field_simp [hpy]
  rw [show (∑ y : Fin n, |p y - q y|) =
      ∑ y : Fin n, p y * |(if p y = 0 then 0 else q y / p y) - 1| by
        apply Finset.sum_congr rfl
        intro y hy
        exact hpoint y]

theorem ratio_mass_eq_one_of_reverse_support
    {n : ℕ} (p q : Fin n → ℝ)
    (hq : ∀ y, 0 ≤ q y)
    (hqp : ∀ y, q y > 0 → p y > 0)
    (hsumq : ∑ y : Fin n, q y = 1) :
    ∑ y : Fin n, p y * (if p y = 0 then 0 else q y / p y) = 1 := by
  have hpoint : ∀ y : Fin n,
      p y * (if p y = 0 then 0 else q y / p y) = q y := by
    intro y
    by_cases hpy : p y = 0
    · have hqy : q y = 0 := by
        by_contra hqyn
        have hqyp : q y > 0 := lt_of_le_of_ne (hq y) (Ne.symm hqyn)
        exact (ne_of_gt (hqp y hqyp)) hpy
      simp [hpy, hqy]
    · field_simp [hpy]
  calc
    ∑ y : Fin n, p y * (if p y = 0 then 0 else q y / p y) = ∑ y : Fin n, q y := by
      apply Finset.sum_congr rfl
      intro y hy
      exact hpoint y
    _ = 1 := hsumq

/- A finite-support change-of-measure identity.  The reverse support
   condition is exactly what is needed when the ratio is written as `q / p`:
   a zero-mass point of `p` must also have zero mass under `q`. -/
theorem finite_change_of_measure
    {n : ℕ} (p q R : Fin n → ℝ)
    (hp : ∀ y, 0 ≤ p y) (hq : ∀ y, 0 ≤ q y)
    (hqp : ∀ y, q y > 0 → p y > 0) :
    ∑ y : Fin n, p y * R y * (if p y = 0 then 0 else q y / p y) =
      ∑ y : Fin n, q y * R y := by
  have hpoint : ∀ y : Fin n,
      p y * R y * (if p y = 0 then 0 else q y / p y) = q y * R y := by
    intro y
    by_cases hpy : p y = 0
    · have hqy : q y = 0 := by
        by_contra hqyn
        have hqyp : q y > 0 := lt_of_le_of_ne (hq y) (Ne.symm hqyn)
        exact (ne_of_gt (hqp y hqyp)) hpy
      simp [hpy, hqy]
    · have hpypos : 0 < p y := lt_of_le_of_ne (hp y) (Ne.symm hpy)
      simp only [if_neg hpy]
      calc
        p y * R y * (q y / p y) = R y * q y := by
          field_simp [hpy]
          ring
        _ = q y * R y := by ring
  apply Finset.sum_congr rfl
  intro y hy
  exact hpoint y

/- The corresponding one-step surrogate identity, written as an expectation
   of the ratio residual.  This is the finite-support algebra used at each
   horizon telescoping step. -/
theorem finite_surrogate_difference_identity
    {n : ℕ} (p q R : Fin n → ℝ)
    (hp : ∀ y, 0 ≤ p y) (hq : ∀ y, 0 ≤ q y)
    (hqp : ∀ y, q y > 0 → p y > 0) :
    ∑ y : Fin n, p y * R y * ((if p y = 0 then 0 else q y / p y) - 1) =
      (∑ y : Fin n, q y * R y) - ∑ y : Fin n, p y * R y := by
  have hcm := finite_change_of_measure p q R hp hq hqp
  have hrewrite :
      (∑ y : Fin n, p y * R y * ((if p y = 0 then 0 else q y / p y) - 1)) =
        (∑ y : Fin n, p y * R y * (if p y = 0 then 0 else q y / p y)) -
          ∑ y : Fin n, p y * R y := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro y hy
    ring
  rw [hrewrite, hcm]

/- A finite conditional KL identity.  With mutual support, the expected
   log-ratio under `p` is the negative of the reverse log-ratio expression;
   this is the local algebraic step used in the autoregressive KL chain rule. -/
noncomputable def ratioOrOne {n : ℕ} (p q : Fin n → ℝ) (y : Fin n) : ℝ :=
  if p y = 0 then 1 else q y / p y

theorem finite_kl_log_ratio
    {n : ℕ} (p q : Fin n → ℝ)
    (hp : ∀ y, 0 ≤ p y) (hq : ∀ y, 0 ≤ q y)
    (hpq : ∀ y, p y > 0 → q y > 0)
    (hqp : ∀ y, q y > 0 → p y > 0) :
    (∑ y : Fin n, p y * Real.log (ratioOrOne p q y)) =
      - (∑ y : Fin n, p y * Real.log (ratioOrOne q p y)) := by
  rw [← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl
  intro y hy
  by_cases hpy : p y = 0
  · have hqy : q y = 0 := by
      by_contra hqyn
      have hqyp : q y > 0 := lt_of_le_of_ne (hq y) (Ne.symm hqyn)
      exact (ne_of_gt (hqp y hqyp)) hpy
    simp [ratioOrOne, hpy, hqy]
  · have hppos : 0 < p y := lt_of_le_of_ne (hp y) (Ne.symm hpy)
    have hqpos : 0 < q y := hpq y hppos
    have hqne : q y ≠ 0 := ne_of_gt hqpos
    have hlog : Real.log (q y / p y) = -Real.log (p y / q y) := by
      rw [Real.log_div hqne (ne_of_gt hppos), Real.log_div (ne_of_gt hppos) hqne]
      ring
    simp only [ratioOrOne, if_neg hpy, if_neg (ne_of_gt hqpos), hlog]
    ring

/- Under mutual support and normalized finite distributions, the sample-based
   `k₃` proxy used in the appendix has the expected reverse-KL form.  We state
   the conclusion directly in terms of the log-ratio, which is the algebraic
   identity used by the manuscript:

       E_p[q/p - 1 - log(q/p)] = - E_p[log(q/p)].

   The normalization step is essential: without `Σ q = 1`, the first two
   terms do not cancel. -/
theorem finite_k3_proxy_eq_reverse_kl
    {n : ℕ} (p q : Fin n → ℝ)
    (hp : ∀ y, 0 ≤ p y) (hq : ∀ y, 0 ≤ q y)
    (hpq : ∀ y, p y > 0 → q y > 0)
    (hqp : ∀ y, q y > 0 → p y > 0)
    (hsp : ∑ y : Fin n, p y = 1)
    (hsq : ∑ y : Fin n, q y = 1) :
    (∑ y : Fin n,
      p y * (ratioOrOne p q y - 1 - Real.log (ratioOrOne p q y))) =
      - (∑ y : Fin n, p y * Real.log (ratioOrOne p q y)) := by
  have hmass : (∑ y : Fin n, p y * ratioOrOne p q y) = 1 := by
    have hpoint : ∀ y : Fin n, p y * ratioOrOne p q y = q y := by
      intro y
      by_cases hpy : p y = 0
      · have hqy : q y = 0 := by
          by_contra hqyn
          have hqyp : q y > 0 := lt_of_le_of_ne (hq y) (Ne.symm hqyn)
          exact (ne_of_gt (hqp y hqyp)) hpy
        simp [ratioOrOne, hpy, hqy]
      · have hppos : 0 < p y := lt_of_le_of_ne (hp y) (Ne.symm hpy)
        have hqpos : 0 < q y := hpq y hppos
        have hqne : q y ≠ 0 := ne_of_gt hqpos
        simp only [ratioOrOne, if_neg hpy]
        field_simp [ne_of_gt hppos, hqne]
    calc
      (∑ y : Fin n, p y * ratioOrOne p q y) = ∑ y : Fin n, q y := by
        apply Finset.sum_congr rfl
        intro y hy
        exact hpoint y
      _ = 1 := hsq
  have hpoint : ∀ y : Fin n,
      p y * (ratioOrOne p q y - 1 - Real.log (ratioOrOne p q y)) =
        p y * ratioOrOne p q y - p y - p y * Real.log (ratioOrOne p q y) := by
    intro y
    ring
  calc
    (∑ y : Fin n,
        p y * (ratioOrOne p q y - 1 - Real.log (ratioOrOne p q y))) =
        ∑ y : Fin n,
          (p y * ratioOrOne p q y - p y - p y * Real.log (ratioOrOne p q y)) := by
      apply Finset.sum_congr rfl
      intro y hy
      exact hpoint y
    _ = (∑ y : Fin n, p y * ratioOrOne p q y) -
          (∑ y : Fin n, p y) -
          (∑ y : Fin n, p y * Real.log (ratioOrOne p q y)) := by
      rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib]
    _ = - (∑ y : Fin n, p y * Real.log (ratioOrOne p q y)) := by
      rw [hmass, hsp]
      ring

/- Finite one-step performance difference: under reverse support, the change
   in expected reward is exactly the rollout expectation of the ratio
   residual.  This is the finite-state analogue of the surrogate identity
   used before the horizon telescoping argument. -/
theorem finite_performance_difference
    {n : ℕ} (p q R : Fin n → ℝ)
    (hp : ∀ y, 0 ≤ p y) (hq : ∀ y, 0 ≤ q y)
    (hqp : ∀ y, q y > 0 → p y > 0) :
    (∑ y : Fin n, q y * R y) - ∑ y : Fin n, p y * R y =
      ∑ y : Fin n, p y * R y * ((if p y = 0 then 0 else q y / p y) - 1) := by
  symm
  exact finite_surrogate_difference_identity p q R hp hq hqp

/- The product version of the finite-horizon telescoping identity.  The
   recursively defined right-hand side is
   `Σ_t (ρ_t - 1) * Π_{j>t} ρ_j`; exposing it as a list keeps this check
   independent of any probability or measurability API. -/
def telescopingRhs : List ℝ → ℝ
  | [] => 0
  | x :: xs => (x - 1) * xs.prod + telescopingRhs xs

def surrogateRhs : List ℝ → ℝ
  | [] => 0
  | x :: xs => (x - 1) + surrogateRhs xs

def remainderRhs : List ℝ → ℝ
  | [] => 0
  | x :: xs => (x - 1) * (1 - xs.prod) + remainderRhs xs

theorem surrogate_minus_remainder (ρ : List ℝ) :
    surrogateRhs ρ - remainderRhs ρ = telescopingRhs ρ := by
  induction ρ with
  | nil => simp [surrogateRhs, remainderRhs, telescopingRhs]
  | cons x xs ih =>
      simp only [surrogateRhs, remainderRhs, telescopingRhs]
      calc
        x - 1 + surrogateRhs xs - ((x - 1) * (1 - xs.prod) + remainderRhs xs) =
            (x - 1) * xs.prod + (surrogateRhs xs - remainderRhs xs) := by ring
        _ = (x - 1) * xs.prod + telescopingRhs xs := by rw [ih]

/- A direct finite-trajectory version of the paper's finite-horizon identity.
   `ratios y` contains the token ratios on trajectory `y`; `hchange` states
   that their product is the trajectory density ratio.  No stochastic API is
   needed for this algebraic core. -/
theorem finite_horizon_surrogate_identity
    {n : ℕ} (p q R : Fin n → ℝ) (ratios : Fin n → List ℝ)
    (hchange : ∀ y, q y = p y * (ratios y).prod) :
    (∑ y : Fin n, q y * R y) - ∑ y : Fin n, p y * R y =
      (∑ y : Fin n, p y * R y * surrogateRhs (ratios y)) -
        (∑ y : Fin n, p y * R y * remainderRhs (ratios y)) := by
  have htel : ∀ xs : List ℝ, xs.prod - 1 = telescopingRhs xs := by
    intro xs
    induction xs with
    | nil => simp [telescopingRhs]
    | cons x xs ih =>
        simp only [List.prod_cons, telescopingRhs]
        rw [show x * xs.prod - 1 = (x - 1) * xs.prod + (xs.prod - 1) by ring]
        rw [ih]
  have hpoint : ∀ y : Fin n,
      q y * R y - p y * R y =
        p y * R y * surrogateRhs (ratios y) -
          p y * R y * remainderRhs (ratios y) := by
    intro y
    calc
      q y * R y - p y * R y = p y * (ratios y).prod * R y - p y * R y := by rw [hchange y]
      _ = p y * R y * ((ratios y).prod - 1) := by ring
      _ = p y * R y * telescopingRhs (ratios y) := by
        rw [htel]
      _ = p y * R y * surrogateRhs (ratios y) -
          p y * R y * remainderRhs (ratios y) := by
        rw [← surrogate_minus_remainder]
        ring
  calc
    (∑ y : Fin n, q y * R y) - ∑ y : Fin n, p y * R y =
        ∑ y : Fin n, (q y * R y - p y * R y) := by rw [Finset.sum_sub_distrib]
    _ = ∑ y : Fin n,
        (p y * R y * surrogateRhs (ratios y) -
          p y * R y * remainderRhs (ratios y)) := by
      apply Finset.sum_congr rfl
      intro y hy
      exact hpoint y
    _ = (∑ y : Fin n, p y * R y * surrogateRhs (ratios y)) -
        (∑ y : Fin n, p y * R y * remainderRhs (ratios y)) := by
      rw [Finset.sum_sub_distrib]

theorem finite_horizon_product_telescoping (ρ : List ℝ) :
    ρ.prod - 1 = telescopingRhs ρ := by
  induction ρ with
  | nil => simp [telescopingRhs]
  | cons x xs ih =>
      simp only [List.prod_cons, telescopingRhs]
      rw [show x * xs.prod - 1 = (x - 1) * xs.prod + (xs.prod - 1) by ring]
      rw [ih]

/- Corrected abnormal-token prefix bound.  The inlier bound is explicit:
   no invalid global quadratic lower bound for `exp` is used.  `B` marks
   abnormal positions, and `hcount` controls their empirical fraction. -/
theorem abnormal_prefix_average_bound
    {n : ℕ} (z : Fin n → ℝ) (B : Fin n → Prop) [DecidablePred B]
    (τ₀ Λp Λn ξ : ℝ)
    (hn : 0 < n) (hτ : 0 ≤ τ₀) (hξ : 0 ≤ ξ)
    (hΛp : τ₀ ≤ Λp) (hΛn : τ₀ ≤ Λn)
    (hpoint : ∀ i, -Λn ≤ z i ∧ z i ≤ Λp)
    (hgood : ∀ i, ¬ B i → -τ₀ ≤ z i ∧ z i ≤ τ₀)
    (hcount : ((Finset.univ.filter B).card : ℝ) / (n : ℝ) ≤ ξ) :
    -τ₀ - ξ * (Λn - τ₀) ≤ (∑ i : Fin n, z i) / (n : ℝ) ∧
      (∑ i : Fin n, z i) / (n : ℝ) ≤ τ₀ + ξ * (Λp - τ₀) := by
  let bad : Finset (Fin n) := Finset.univ.filter B
  have hn' : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hcard_nonneg : 0 ≤ (bad.card : ℝ) := by positivity
  have hcount' : (bad.card : ℝ) ≤ ξ * (n : ℝ) := by
    have := hcount
    dsimp [bad] at this ⊢
    apply (div_le_iff₀ hn').mp at this
    nlinarith
  have hsum_ind : (∑ i : Fin n, (if B i then (1 : ℝ) else 0)) = (bad.card : ℝ) := by
    simp [bad]
  have hΛpd : 0 ≤ Λp - τ₀ := sub_nonneg.mpr hΛp
  have hΛnd : 0 ≤ Λn - τ₀ := sub_nonneg.mpr hΛn
  have hdomain : 0 ≤ τ₀ * ξ := mul_nonneg hτ hξ
  have hup : ∀ i : Fin n, z i ≤ τ₀ + (if B i then (1 : ℝ) else 0) * (Λp - τ₀) := by
    intro i
    by_cases hi : B i
    · simp [hi]
      nlinarith [(hpoint i).2]
    · simp [hi]
      exact (hgood i hi).2
  have hlo : ∀ i : Fin n, -τ₀ - (if B i then (1 : ℝ) else 0) * (Λn - τ₀) ≤ z i := by
    intro i
    by_cases hi : B i
    · simp [hi]
      nlinarith [(hpoint i).1]
    · simp [hi]
      exact (hgood i hi).1
  have hsum_up : (∑ i : Fin n, z i) ≤
      (n : ℝ) * τ₀ + (bad.card : ℝ) * (Λp - τ₀) := by
    have h : (∑ i : Fin n, z i) ≤
        ∑ i : Fin n, (τ₀ + (if B i then (1 : ℝ) else 0) * (Λp - τ₀)) := by
      exact Finset.sum_le_sum (fun i hi => hup i)
    calc
      (∑ i : Fin n, z i) ≤ ∑ i : Fin n,
          (τ₀ + (if B i then (1 : ℝ) else 0) * (Λp - τ₀)) := h
      _ = (n : ℝ) * τ₀ + (bad.card : ℝ) * (Λp - τ₀) := by
        rw [Finset.sum_add_distrib]
        have hterm : (∑ i : Fin n,
            (if B i then (1 : ℝ) else 0) * (Λp - τ₀)) =
            (bad.card : ℝ) * (Λp - τ₀) := by
          rw [← Finset.sum_mul, hsum_ind]
        rw [hterm]
        simp [nsmul_eq_mul]
  have hsum_lo : (n : ℝ) * (-τ₀) - (bad.card : ℝ) * (Λn - τ₀) ≤
      ∑ i : Fin n, z i := by
    have h : (∑ i : Fin n, (-τ₀ - (if B i then (1 : ℝ) else 0) * (Λn - τ₀))) ≤
        (∑ i : Fin n, z i) := by
      exact Finset.sum_le_sum (fun i hi => hlo i)
    calc
      (n : ℝ) * (-τ₀) - (bad.card : ℝ) * (Λn - τ₀) =
          ∑ i : Fin n, (-τ₀ - (if B i then (1 : ℝ) else 0) * (Λn - τ₀)) := by
            rw [Finset.sum_sub_distrib]
            have hterm : (∑ i : Fin n,
                (if B i then (1 : ℝ) else 0) * (Λn - τ₀)) =
                (bad.card : ℝ) * (Λn - τ₀) := by
              rw [← Finset.sum_mul, hsum_ind]
            rw [hterm]
            simp [nsmul_eq_mul]
      _ ≤ ∑ i : Fin n, z i := h
  constructor
  · apply (le_div_iff₀ hn').2
    nlinarith [hcount', hΛnd, hdomain]
  · apply (div_le_iff₀ hn').2
    nlinarith [hcount', hΛpd, hdomain]

/- Closed finite-sequence form of the late single-spike case in
   `thm:prefix-tighter` (Part (b)). -/
theorem prefix_tighter_single_spike_closed_loop
    {n : ℕ} (hn : 0 < n) (k : Fin n) (x lo hi : ℝ)
    (hx : 0 < x) (hlo : lo ≤ 0) (hhi : 0 ≤ hi)
    (hpref : hi * ((k.val + 1 : ℕ) : ℝ) < x)
    (hseq : x / (n : ℝ) ≤ hi) :
    hi < prefixAverage (singleSpike k x) k ∧
      lo ≤ (∑ i : Fin n, singleSpike k x i) / (n : ℝ) ∧
      (∑ i : Fin n, singleSpike k x i) / (n : ℝ) ≤ hi ∧
      (¬ (lo ≤ singleSpike k x k ∧ singleSpike k x k ≤ hi)) ∧
      (∀ i : Fin n, i ≠ k →
        lo ≤ singleSpike k x i ∧ singleSpike k x i ≤ hi) ∧
      (∀ j : Fin n, j < k → prefixAverage (singleSpike k x) j = 0) := by
  have hsep := single_spike_prefix_rejects_sequence_accepts hn k x lo hi hx hlo hpref hseq
  have hk1 : (1 : ℝ) ≤ ((k.val + 1 : ℕ) : ℝ) := by
    have : (1 : ℕ) ≤ k.val + 1 := Nat.succ_le_succ (Nat.zero_le k.val)
    exact_mod_cast this
  have hspike : hi < x := by nlinarith [hpref, hk1, hhi]
  have htok := single_spike_token_interval_pattern k x lo hi hlo hhi hspike
  have hlate : ∀ j : Fin n, j < k → prefixAverage (singleSpike k x) j = 0 := by
    intro j hj
    exact single_spike_late_prefix_zero x hj
  exact ⟨hsep.1, hsep.2.1, hsep.2.2, htok.1, htok.2, hlate⟩

end Counterexamples

end REINFORCEProMax
