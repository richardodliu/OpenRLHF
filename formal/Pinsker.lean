import CoarseKL
import Mathlib.Analysis.Calculus.Deriv.MeanValue

namespace REINFORCEProMax

noncomputable def bernoulliPinskerGap (p q : ℝ) : ℝ :=
  p * (Real.log p - Real.log q) +
    (1 - p) * (Real.log (1 - p) - Real.log (1 - q)) - 2 * (p - q) ^ 2

theorem bernoulliPinskerGap_deriv (p q : ℝ) (hq0 : 0 < q) (hq1 : q < 1) :
    HasDerivAt (bernoulliPinskerGap p)
      ((q - p) * ((1 - 2 * q) ^ 2 / (q * (1 - q)))) q := by
  have hq : q ≠ 0 := ne_of_gt hq0
  have hqc : 1 - q ≠ 0 := ne_of_gt (sub_pos.mpr hq1)
  have hlog := Real.hasDerivAt_log hq
  have hlogc := ((hasDerivAt_const q (1 : ℝ)).sub (hasDerivAt_id q)).log hqc
  have h := (((hasDerivAt_const q (Real.log p)).sub hlog).const_mul p).add
    (((hasDerivAt_const q (Real.log (1 - p))).sub hlogc).const_mul (1 - p))
  have hf := h.sub ((((hasDerivAt_const q p).sub (hasDerivAt_id q)).pow 2).const_mul 2)
  convert hf using 1 <;> try rfl
  field_simp
  <;> ring

theorem bernoulli_pinsker (p q : ℝ) (hp0 : 0 < p) (hp1 : p < 1)
    (hq0 : 0 < q) (hq1 : q < 1) :
    2 * (p - q) ^ 2 ≤ p * Real.log (p / q) + (1 - p) * Real.log ((1 - p) / (1 - q)) := by
  have hself : bernoulliPinskerGap p p = 0 := by simp [bernoulliPinskerGap]
  have hnonneg : 0 ≤ bernoulliPinskerGap p q := by
    rcases le_total p q with hpq | hqp
    · have hderiv : ∀ x ∈ Set.Icc p q, HasDerivAt (bernoulliPinskerGap p)
          ((x - p) * ((1 - 2 * x) ^ 2 / (x * (1 - x)))) x := by
        intro x hx
        exact bernoulliPinskerGap_deriv p x (lt_of_lt_of_le hp0 hx.1) (lt_of_le_of_lt hx.2 hq1)
      have hmono := monotoneOn_of_deriv_nonneg (convex_Icc p q)
        (fun x hx => (hderiv x hx).continuousAt.continuousWithinAt)
        (fun x hx => (hderiv x (interior_subset hx)).differentiableAt.differentiableWithinAt)
        (by
          intro x hx
          have hx' := interior_subset hx
          rw [(hderiv x hx').deriv]
          exact mul_nonneg (sub_nonneg.mpr hx'.1)
            (div_nonneg (sq_nonneg _) (mul_nonneg (le_of_lt (lt_of_lt_of_le hp0 hx'.1))
              (sub_nonneg.mpr (le_trans hx'.2 (le_of_lt hq1))))))
      have := hmono ⟨le_rfl, hpq⟩ ⟨hpq, le_rfl⟩ hpq
      rwa [hself] at this
    · have hderiv : ∀ x ∈ Set.Icc q p, HasDerivAt (bernoulliPinskerGap p)
          ((x - p) * ((1 - 2 * x) ^ 2 / (x * (1 - x)))) x := by
        intro x hx
        exact bernoulliPinskerGap_deriv p x (lt_of_lt_of_le hq0 hx.1) (lt_of_le_of_lt hx.2 hp1)
      have hmono := antitoneOn_of_deriv_nonpos (convex_Icc q p)
        (fun x hx => (hderiv x hx).continuousAt.continuousWithinAt)
        (fun x hx => (hderiv x (interior_subset hx)).differentiableAt.differentiableWithinAt)
        (by
          intro x hx
          have hx' := interior_subset hx
          rw [(hderiv x hx').deriv]
          exact mul_nonpos_of_nonpos_of_nonneg (sub_nonpos.mpr hx'.2)
            (div_nonneg (sq_nonneg _) (mul_nonneg (le_of_lt (lt_of_lt_of_le hq0 hx'.1))
              (sub_nonneg.mpr (le_trans hx'.2 (le_of_lt hp1))))))
      have := hmono ⟨le_rfl, hqp⟩ ⟨hqp, le_rfl⟩ hqp
      rwa [hself] at this
  rw [Real.log_div (ne_of_gt hp0) (ne_of_gt hq0),
    Real.log_div (ne_of_gt (sub_pos.mpr hp1)) (ne_of_gt (sub_pos.mpr hq1))]
  exact sub_nonneg.mp hnonneg

/-- The positive-difference event attains TV for normalized finite masses. -/
theorem finite_tv_positive_event {I : Type*} [Fintype I] (p q : I → ℝ)
    (hp : ∑ i, p i = 1) (hq : ∑ i, q i = 1) :
    (1 / 2 : ℝ) * ∑ i, |p i - q i| =
      ∑ i ∈ Finset.univ.filter (fun i => q i < p i), (p i - q i) := by
  classical
  let S := Finset.univ.filter (fun i => q i < p i)
  have hzero : (∑ i, (p i - q i)) = 0 := by rw [Finset.sum_sub_distrib, hp, hq]; ring
  have hsplit := Finset.sum_add_sum_compl S (fun i => p i - q i)
  have habsS : (∑ i ∈ S, |p i - q i|) = ∑ i ∈ S, (p i - q i) := by
    apply Finset.sum_congr rfl
    intro i hi
    exact abs_of_pos (sub_pos.mpr (Finset.mem_filter.mp hi).2)
  have habsC : (∑ i ∈ Sᶜ, |p i - q i|) = -(∑ i ∈ Sᶜ, (p i - q i)) := by
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro i hi
    have hnot : ¬ q i < p i := by simpa [S] using (Finset.mem_compl.mp hi)
    exact abs_of_nonpos (sub_nonpos.mpr (le_of_not_gt hnot))
  have habssplit := Finset.sum_add_sum_compl S (fun i => |p i - q i|)
  rw [habsS, habsC] at habssplit
  rw [hzero] at hsplit
  change _ = ∑ i ∈ S, (p i - q i)
  linarith

/-- Pinsker with the sharp constant for arbitrary finite mutually supported
probability masses. Zero probabilities are allowed. -/
theorem finite_pinsker_squared {I : Type*} [Fintype I] (p q : I → ℝ)
    (hp0 : ∀ i, 0 ≤ p i) (hq0 : ∀ i, 0 ≤ q i)
    (hp : ∑ i, p i = 1) (hq : ∑ i, q i = 1)
    (hs : ∀ i, p i = 0 ↔ q i = 0) :
    2 * ((1 / 2 : ℝ) * ∑ i, |p i - q i|) ^ 2 ≤
      ∑ i, p i * Real.log (p i / q i) := by
  classical
  let S := Finset.univ.filter (fun i => q i < p i)
  let P := ∑ i ∈ S, p i
  let Q := ∑ i ∈ S, q i
  have hsupport : ∀ i, 0 < p i → 0 < q i := by
    intro i hi
    exact lt_of_le_of_ne (hq0 i) (Ne.symm (mt (hs i).mpr (ne_of_gt hi)))
  have hreverse : ∀ i, 0 < q i → 0 < p i := by
    intro i hi
    exact lt_of_le_of_ne (hp0 i) (Ne.symm (mt (hs i).mp (ne_of_gt hi)))
  have hP0 : 0 ≤ P := Finset.sum_nonneg fun i _ => hp0 i
  have hQ0 : 0 ≤ Q := Finset.sum_nonneg fun i _ => hq0 i
  have hPc : (∑ i ∈ Sᶜ, p i) = 1 - P := by
    have := Finset.sum_add_sum_compl S p
    rw [hp] at this
    change P + _ = 1 at this
    linarith
  have hQc : (∑ i ∈ Sᶜ, q i) = 1 - Q := by
    have := Finset.sum_add_sum_compl S q
    rw [hq] at this
    change Q + _ = 1 at this
    linarith
  have hP1 : P ≤ 1 := by
    have := Finset.sum_nonneg (s := Sᶜ) (fun i _ => hp0 i)
    rw [hPc] at this
    linarith
  have hTV : (1 / 2 : ℝ) * ∑ i, |p i - q i| = P - Q := by
    rw [finite_tv_positive_event p q hp hq]
    exact Finset.sum_sub_distrib
  have hPQ : Q ≤ P := by
    have ht : 0 ≤ (1 / 2 : ℝ) * ∑ i, |p i - q i| :=
      mul_nonneg (by norm_num) (Finset.sum_nonneg fun i _ => abs_nonneg _)
    rw [hTV] at ht
    linarith
  rw [hTV]
  by_cases heq : P = Q
  · have hkl := finite_cell_kl_contraction Finset.univ p q
      (fun i _ => hp0 i) (fun i _ => hq0 i) (fun i _ => hsupport i)
    simpa [hp, hq, heq] using hkl
  have hPQ' : Q < P := lt_of_le_of_ne hPQ (Ne.symm heq)
  have hPpos : 0 < P := lt_of_le_of_lt hQ0 hPQ'
  have hQpos : 0 < Q := cell_mass_support S p q
    (fun i _ => hp0 i) (fun i _ => hq0 i) (fun i _ => hsupport i) hPpos
  have hQlt : Q < 1 := lt_of_lt_of_le hPQ' hP1
  have hPcomp : 0 < ∑ i ∈ Sᶜ, p i := cell_mass_support Sᶜ q p
    (fun i _ => hq0 i) (fun i _ => hp0 i) (fun i _ => hreverse i)
    (by rw [hQc]; linarith)
  have hPlt : P < 1 := by rw [hPc] at hPcomp; linarith
  have hbinary := bernoulli_pinsker P Q hPpos hPlt hQpos hQlt
  have hS := finite_cell_kl_contraction S p q
    (fun i _ => hp0 i) (fun i _ => hq0 i) (fun i _ => hsupport i)
  have hC := finite_cell_kl_contraction Sᶜ p q
    (fun i _ => hp0 i) (fun i _ => hq0 i) (fun i _ => hsupport i)
  change P * Real.log (P / Q) ≤ _ at hS
  rw [hPc, hQc] at hC
  have hsplit := Finset.sum_add_sum_compl S (fun i => p i * Real.log (p i / q i))
  linarith

theorem finite_pinsker {I : Type*} [Fintype I] (p q : I → ℝ)
    (hp0 : ∀ i, 0 ≤ p i) (hq0 : ∀ i, 0 ≤ q i)
    (hp : ∑ i, p i = 1) (hq : ∑ i, q i = 1)
    (hs : ∀ i, p i = 0 ↔ q i = 0) :
    (1 / 2 : ℝ) * ∑ i, |p i - q i| ≤
      Real.sqrt ((∑ i, p i * Real.log (p i / q i)) / 2) := by
  have hb := finite_pinsker_squared p q hp0 hq0 hp hq hs
  have hnonneg : 0 ≤ (∑ i, p i * Real.log (p i / q i)) / 2 := by nlinarith [sq_nonneg ((1 / 2 : ℝ) * ∑ i, |p i - q i|)]
  have hroot := Real.sq_sqrt hnonneg
  have hpos := Real.sqrt_nonneg ((∑ i, p i * Real.log (p i / q i)) / 2)
  nlinarith

end REINFORCEProMax
