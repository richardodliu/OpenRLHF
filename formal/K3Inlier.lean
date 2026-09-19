import ProMax

/-! Exact inverse certificates for the sample k3 proxy. These do not replace
sample proxy thresholding with a population conditional KL threshold. -/
namespace REINFORCEProMax

noncomputable def logK3 (z : ℝ) : ℝ := Real.exp z - 1 - z

theorem logK3_nonneg (z : ℝ) : 0 ≤ logK3 z := by
  have := Real.add_one_le_exp z
  dsimp [logK3]
  linarith

theorem logK3_strictMono_nonneg {x y : ℝ} (hx : 0 ≤ x) (hxy : x < y) :
    logK3 x < logK3 y := by
  have he := Real.add_one_lt_exp (ne_of_gt (sub_pos.mpr hxy))
  have hxexp : 1 ≤ Real.exp x := Real.one_le_exp_iff.mpr hx
  have hpos := Real.exp_pos x
  have hmul := mul_lt_mul_of_pos_left he hpos
  have hprod : Real.exp x * Real.exp (y - x) = Real.exp y := by
    rw [← Real.exp_add]
    congr 1
    ring
  rw [hprod] at hmul
  have hc := mul_nonneg (sub_nonneg.mpr hxexp) (le_of_lt (sub_pos.mpr hxy))
  dsimp [logK3]
  nlinarith

theorem logK3_strictAnti_nonpos {x y : ℝ} (hy : y ≤ 0) (hxy : x < y) :
    logK3 y < logK3 x := by
  have he := Real.add_one_lt_exp (ne_of_lt (sub_neg.mpr hxy))
  have hyexp : Real.exp y ≤ 1 := Real.exp_le_one_iff.mpr hy
  have hpos := Real.exp_pos y
  have hmul := mul_lt_mul_of_pos_left he hpos
  have hprod : Real.exp y * Real.exp (x - y) = Real.exp x := by
    rw [← Real.exp_add]
    congr 1
    ring
  rw [hprod] at hmul
  have hc := mul_nonneg (sub_nonneg.mpr hyexp) (le_of_lt (sub_pos.mpr hxy))
  dsimp [logK3]
  nlinarith

theorem logK3_interval_certificate (z δ lower upper : ℝ)
    (hl : 0 ≤ lower) (hu : 0 ≤ upper)
    (hleft : δ ≤ logK3 (-lower)) (hright : δ ≤ logK3 upper)
    (hinlier : logK3 z ≤ δ) : -lower ≤ z ∧ z ≤ upper := by
  constructor
  · by_contra h
    have := logK3_strictAnti_nonpos (neg_nonpos.mpr hl) (lt_of_not_ge h)
    linarith
  · by_contra h
    have := logK3_strictMono_nonneg hu (lt_of_not_ge h)
    linarith

theorem k3_log_inlier_bound (rho δ τ : ℝ) (hr : 0 < rho) (ht : 0 ≤ τ)
    (hδ : δ ≤ min (logK3 (-τ)) (logK3 τ))
    (hinlier : rho - 1 - Real.log rho ≤ δ) : |Real.log rho| ≤ τ := by
  have hc := logK3_interval_certificate (Real.log rho) δ τ τ ht ht
    (le_trans hδ (min_le_left _ _)) (le_trans hδ (min_le_right _ _))
    (by simpa [logK3, Real.exp_log hr] using hinlier)
  exact abs_le.mpr hc

theorem k3_symmetric_threshold_positive {τ : ℝ} (ht : 0 < τ) :
    0 < min (logK3 (-τ)) (logK3 τ) := by
  have hn := logK3_strictAnti_nonpos (le_refl (0 : ℝ)) (neg_neg_of_pos ht)
  have hp := logK3_strictMono_nonneg (le_refl (0 : ℝ)) ht
  simp only [logK3, Real.exp_zero, sub_self, sub_zero] at hn hp
  exact lt_min hn hp

theorem k3_abnormal_prefix_average_bound {n : ℕ} (z : Fin n → ℝ)
    (δ τ lower upper ξ : ℝ) (hn : 0 < n) (ht : 0 ≤ τ) (hξ : 0 ≤ ξ)
    (hl : τ ≤ lower) (hu : τ ≤ upper)
    (hδ : δ ≤ min (logK3 (-τ)) (logK3 τ))
    (hpoint : ∀ i, -lower ≤ z i ∧ z i ≤ upper)
    (hcount : ((Finset.univ.filter (fun i => δ < logK3 (z i))).card : ℝ) / (n : ℝ) ≤ ξ) :
    -τ - ξ * (lower - τ) ≤ (∑ i : Fin n, z i) / (n : ℝ) ∧
      (∑ i : Fin n, z i) / (n : ℝ) ≤ τ + ξ * (upper - τ) := by
  apply abnormal_prefix_average_bound z (fun i => δ < logK3 (z i))
    τ upper lower ξ hn ht hξ hu hl hpoint _ hcount
  intro i hi
  exact logK3_interval_certificate (z i) δ τ τ ht ht
    (le_trans hδ (min_le_left _ _)) (le_trans hδ (min_le_right _ _)) (le_of_not_gt hi)

end REINFORCEProMax
