import TreePinsker

/-!
The paper's concrete log-average prefix gate on the same normalized history
tree as its TV and concentration theorems. Acceptance, positive support,
the density-ratio product, and the rollout expectation are all constructed.
-/
namespace REINFORCEProMax.AutoregressiveTree

open scoped BigOperators
variable {A : Type*} [Fintype A]

noncomputable def prefixLogGate (p q : Policy A) (τp τn : ℝ) (y : List A) : ℝ :=
  if -τn ≤ logRatioTrace p q y / (y.length : ℝ) ∧
      logRatioTrace p q y / (y.length : ℝ) ≤ τp then 1 else 0

theorem prefixLogGate_binary (p q : Policy A) (τp τn : ℝ) (y : List A) :
    prefixLogGate p q τp τn y = 0 ∨ prefixLogGate p q τp τn y = 1 := by
  unfold prefixLogGate
  split_ifs <;> simp

theorem prefixLogGate_nonneg (p q : Policy A) (τp τn : ℝ) (y : List A) :
    0 ≤ prefixLogGate p q τp τn y := by
  rcases prefixLogGate_binary p q τp τn y with hh | hh <;> simp [hh]

theorem prefixLogGate_le_one (p q : Policy A) (τp τn : ℝ) (y : List A) :
    prefixLogGate p q τp τn y ≤ 1 := by
  rcases prefixLogGate_binary p q τp τn y with hh | hh <;> simp [hh]

theorem ratioTrace_prod_eq_exp_on_support (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (y : List A)
    (hp : 0 < pathMass p [] y) :
    (ratioTrace p q y).prod = Real.exp (logRatioTrace p q y) := by
  have hq : 0 < pathMass q [] y := lt_of_le_of_ne (pathMass_nonneg q [] y)
    (Ne.symm (mt (pathMass_mutual_support p q hs [] y).mpr (ne_of_gt hp)))
  rw [← pathMass_log_ratio p q hs y (ne_of_gt hp), Real.exp_log (div_pos hq hp)]
  apply (eq_div_iff (ne_of_gt hp)).mpr
  simpa only [mul_comm] using (pathMass_ratio p q (fun h a => (hs h a).mp) y).symm

noncomputable def prefixEnvelope (τp τn : ℝ) (n : ℕ) : ℝ :=
  max (Real.exp ((n : ℝ) * τp) - 1) (1 - Real.exp (-((n : ℝ) * τn)))

theorem prefixEnvelope_nonneg (τp τn : ℝ) (hτp : 0 ≤ τp) (n : ℕ) :
    0 ≤ prefixEnvelope τp τn n := by
  apply le_max_of_le_left
  have hb := Real.exp_le_exp.mpr (show (0 : ℝ) ≤ (n : ℝ) * τp by positivity)
  simpa using sub_nonneg.mpr hb

/-- The exact envelope on every accepted, positive-mass prefix. -/
theorem accepted_prefix_density_envelope (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (τp τn : ℝ) (hτp : 0 ≤ τp) (hτn : 0 ≤ τn)
    (y : List A) (hy : 0 < y.length) (hp : 0 < pathMass p [] y)
    (hgate : prefixLogGate p q τp τn y = 1) :
    (Real.exp (-((y.length : ℝ) * τn)) ≤ (ratioTrace p q y).prod ∧
      (ratioTrace p q y).prod ≤ Real.exp ((y.length : ℝ) * τp)) ∧
      |(ratioTrace p q y).prod - 1| ≤ prefixEnvelope τp τn y.length := by
  have ha : -τn ≤ logRatioTrace p q y / (y.length : ℝ) ∧
      logRatioTrace p q y / (y.length : ℝ) ≤ τp := by
    unfold prefixLogGate at hgate
    split_ifs at hgate with hh
    · exact hh
    · norm_num at hgate
  have hlen : (0 : ℝ) < (y.length : ℝ) := by exact_mod_cast hy
  have hlo : -((y.length : ℝ) * τn) ≤ logRatioTrace p q y := by
    have hb := (le_div_iff₀ hlen).mp ha.1
    nlinarith only [hb]
  have hhi : logRatioTrace p q y ≤ (y.length : ℝ) * τp := by
    simpa only [mul_comm] using (div_le_iff₀ hlen).mp ha.2
  rw [ratioTrace_prod_eq_exp_on_support p q hs y hp]
  exact ⟨⟨Real.exp_le_exp.mpr hlo, Real.exp_le_exp.mpr hhi⟩,
    accepted_prefix_product_deviation_bound _ _ _ hlo hhi (by positivity) (by positivity)⟩

theorem value_eq_sum_pathMass (p : Policy A) (f : List A → ℝ) (n : ℕ) :
    value p f n [] = ∑ y : Fin n → A, pathMass p [] (List.ofFn y) * f (List.ofFn y) := by
  have hb := pathSum_mass_value p f n []
  simpa only [pathSum_eq_sum_ofFn, List.nil_append, pathMass, one_mul] using hb.symm

theorem value_mono_on_support_length (p : Policy A) (f g : List A → ℝ) (n : ℕ)
    (hfg : ∀ y, y.length = n → 0 < pathMass p [] y → f y ≤ g y) :
    value p f n [] ≤ value p g n [] := by
  rw [value_eq_sum_pathMass, value_eq_sum_pathMass]
  apply Finset.sum_le_sum
  intro y _
  by_cases hz : pathMass p [] (List.ofFn y) = 0
  · simp [hz]
  · have hp := lt_of_le_of_ne (pathMass_nonneg p [] (List.ofFn y)) (Ne.symm hz)
    exact mul_le_mul_of_nonneg_left (hfg _ (by simp) hp) hp.le

/-- Full accepted-prefix expectation theorem, including acceptance mass.
No independent-gate assumption and no strictly positive vocabulary premise. -/
theorem prefix_gate_residual_bound (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (τp τn : ℝ) (hτp : 0 ≤ τp) (hτn : 0 ≤ τn) (n : ℕ) (hn : 0 < n) :
    value p (fun y => prefixLogGate p q τp τn y * |(ratioTrace p q y).prod - 1|) n [] ≤
      prefixEnvelope τp τn n * value p (prefixLogGate p q τp τn) n [] := by
  rw [← value_const_mul]
  apply value_mono_on_support_length
  intro y hlen hp
  rcases prefixLogGate_binary p q τp τn y with hzero | hone
  · simp [hzero]
  · simp only [hone, one_mul, mul_one]
    have hb := (accepted_prefix_density_envelope p q hs τp τn hτp hτn y
      (by omega) hp hone).2
    simpa only [hlen] using hb

theorem prefix_gate_residual_le_envelope (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (τp τn : ℝ) (hτp : 0 ≤ τp) (hτn : 0 ≤ τn) (n : ℕ) (hn : 0 < n) :
    value p (fun y => prefixLogGate p q τp τn y * |(ratioTrace p q y).prod - 1|) n [] ≤
      prefixEnvelope τp τn n := by
  have hm := value_mono p (prefixLogGate p q τp τn) (fun _ => 1)
    (prefixLogGate_le_one p q τp τn) n []
  rw [value_const] at hm
  exact (prefix_gate_residual_bound p q hs τp τn hτp hτn n hn).trans
    (mul_le_of_le_one_right (prefixEnvelope_nonneg τp τn hτp n) hm)

theorem prefix_gate_tv_bound (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (τp τn : ℝ) (hτp : 0 ≤ τp) (hτn : 0 ≤ τn) (n : ℕ) (hn : 0 < n) :
    acceptedPrefixTV p q (prefixLogGate p q τp τn) n ≤
      (1 / 2 : ℝ) * prefixEnvelope τp τn n * value p (prefixLogGate p q τp τn) n [] := by
  rw [acceptedPrefixTV_identity p q (fun h a => (hs h a).mp) _
    (prefixLogGate_nonneg p q τp τn)]
  simpa only [mul_assoc] using mul_le_mul_of_nonneg_left
    (prefix_gate_residual_bound p q hs τp τn hτp hτn n hn) (by norm_num : (0 : ℝ) ≤ 1 / 2)

end REINFORCEProMax.AutoregressiveTree
