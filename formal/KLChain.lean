import ProMax

/-!
  A finite context/token version of the conditional KL chain-rule algebra used
  by the manuscript.  The context weights `d` stand for the rollout
  visitation distribution; consistency of these weights with an
  autoregressive policy is intentionally an explicit modelling obligation
  outside this finite-sum identity.
-/
namespace REINFORCEProMax

/- The finite log-sum inequality is the analytic building block behind the
   KL coarse-graining remark.  We state the strictly-positive version first;
   zero-mass cells can be added by the usual support decomposition. -/
theorem finite_log_sum_inequality
    {n : ℕ} (p q : Fin n → ℝ) (hn : 0 < n)
    (hp : ∀ y, 0 < p y) (hq : ∀ y, 0 < q y) :
    (∑ y : Fin n, p y) * Real.log ((∑ y : Fin n, q y) / (∑ y : Fin n, p y)) ≥
      ∑ y : Fin n, p y * Real.log (q y / p y) := by
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
  have hjensen := strictConcaveOn_log_Ioi.concaveOn.le_map_sum
    (t := (Finset.univ : Finset (Fin n)))
    (w := fun y : Fin n => p y / P)
    (p := fun y : Fin n => q y / p y)
    (by intro y hy; exact hweights y) hweights_sum (by
      intro y hy
      exact hratio y)
  simp only [smul_eq_mul] at hjensen
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
  rw [hleft, harg] at hjensen
  have hmul := mul_le_mul_of_nonneg_left hjensen (le_of_lt hP)
  field_simp [ne_of_gt hP] at hmul
  nlinarith

/- The finite normalized-policy specialization gives the sign of the
   rollout expectation of a log likelihood ratio.  This is the missing
   non-positive mean used by the local sub-Gaussian second-moment bridge. -/
theorem finite_log_ratio_mean_nonpos
    {n : ℕ} (hn : 0 < n) (p q : Fin n → ℝ)
    (hp : ∀ y, 0 < p y) (hq : ∀ y, 0 < q y)
    (hsp : ∑ y : Fin n, p y = 1)
    (hsq : ∑ y : Fin n, q y = 1) :
    ∑ y : Fin n, p y * Real.log (q y / p y) ≤ 0 := by
  have hls := finite_log_sum_inequality p q hn hp hq
  rw [hsp, hsq] at hls
  simpa using hls

theorem finite_conditional_kl_chain
    {T C V : ℕ}
    (d : Fin T → Fin C → ℝ)
    (p q : Fin T → Fin C → Fin V → ℝ)
    (hp : ∀ t c v, 0 ≤ p t c v)
    (hq : ∀ t c v, 0 ≤ q t c v)
    (hpq : ∀ t c v, p t c v > 0 → q t c v > 0)
    (hqp : ∀ t c v, q t c v > 0 → p t c v > 0) :
    (∑ t : Fin T, ∑ c : Fin C, ∑ v : Fin V,
      d t c * p t c v * Real.log (ratioOrOne (p t c) (q t c) v)) =
      - (∑ t : Fin T, ∑ c : Fin C,
        d t c * (∑ v : Fin V,
          p t c v * Real.log (ratioOrOne (q t c) (p t c) v))) := by
  have hlocal : ∀ (t : Fin T) (c : Fin C),
      (∑ v : Fin V,
        p t c v * Real.log (ratioOrOne (p t c) (q t c) v)) =
        - (∑ v : Fin V,
          p t c v * Real.log (ratioOrOne (q t c) (p t c) v)) := by
    intro t c
    exact finite_kl_log_ratio (p t c) (q t c)
      (by intro v; exact hp t c v)
      (by intro v; exact hq t c v)
      (by intro v hv; exact hpq t c v hv)
      (by intro v hv; exact hqp t c v hv)
  calc
    (∑ t : Fin T, ∑ c : Fin C, ∑ v : Fin V,
        d t c * p t c v * Real.log (ratioOrOne (p t c) (q t c) v))
        = ∑ t : Fin T, ∑ c : Fin C,
          d t c * (∑ v : Fin V,
            p t c v * Real.log (ratioOrOne (p t c) (q t c) v)) := by
      apply Finset.sum_congr rfl
      intro t ht
      apply Finset.sum_congr rfl
      intro c hc
      calc
        (∑ v : Fin V,
            d t c * p t c v * Real.log (ratioOrOne (p t c) (q t c) v)) =
            ∑ v : Fin V,
              d t c * (p t c v * Real.log (ratioOrOne (p t c) (q t c) v)) := by
                apply Finset.sum_congr rfl
                intro v hv
                ring
        _ = d t c * (∑ v : Fin V,
              p t c v * Real.log (ratioOrOne (p t c) (q t c) v)) := by
                rw [Finset.mul_sum]
    _ = ∑ t : Fin T, ∑ c : Fin C,
          d t c * (-(∑ v : Fin V,
            p t c v * Real.log (ratioOrOne (q t c) (p t c) v))) := by
      apply Finset.sum_congr rfl
      intro t ht
      apply Finset.sum_congr rfl
      intro c hc
      rw [hlocal t c]
    _ = - (∑ t : Fin T, ∑ c : Fin C,
          d t c * (∑ v : Fin V,
            p t c v * Real.log (ratioOrOne (q t c) (p t c) v))) := by
      calc
        (∑ t : Fin T, ∑ c : Fin C,
            d t c * (-(∑ v : Fin V,
              p t c v * Real.log (ratioOrOne (q t c) (p t c) v)))) =
            ∑ t : Fin T, ∑ c : Fin C,
              -(d t c * (∑ v : Fin V,
                p t c v * Real.log (ratioOrOne (q t c) (p t c) v))) := by
                  apply Finset.sum_congr rfl
                  intro t ht
                  apply Finset.sum_congr rfl
                  intro c hc
                  ring
        _ = - (∑ t : Fin T, ∑ c : Fin C,
              d t c * (∑ v : Fin V,
                p t c v * Real.log (ratioOrOne (q t c) (p t c) v))) := by
                  simp only [Finset.sum_neg_distrib]

end REINFORCEProMax
