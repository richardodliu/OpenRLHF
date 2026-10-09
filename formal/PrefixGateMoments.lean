import TreePrefixGate

namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators
variable {A : Type*} [Fintype A]

theorem prefix_log_bounds_of_gate (p q : Policy A) (τp τn : ℝ) (h : List A)
    (hg : prefixLogGate p q τp τn h = 1) :
    -((h.length : ℝ) * τn) ≤ logRatioTrace p q h ∧
      logRatioTrace p q h ≤ (h.length : ℝ) * τp := by
  by_cases hz : h = []
  · subst h
    simp [logRatioTrace, ratioTrace]
  have hl : 0 < (h.length : ℝ) := by exact_mod_cast List.length_pos_iff.mpr hz
  have hg' : -τn ≤ logRatioTrace p q h / (h.length : ℝ) ∧
      logRatioTrace p q h / (h.length : ℝ) ≤ τp := by
    unfold prefixLogGate at hg
    split_ifs at hg with hc
    · exact hc
    · norm_num at hg
  constructor
  · have hb := (le_div_iff₀ hl).mp hg'.1
    nlinarith
  · have hb := (div_le_iff₀ hl).mp hg'.2
    nlinarith

noncomputable def consecutiveEnvelope (τp τn : ℝ) (n : ℕ) : ℝ :=
  Real.exp (((n : ℝ) + 1) * τp + (n : ℝ) * τn)

/-- Two consecutive accepted prefixes bound the current token coefficient.
This includes the first token: the empty prefix has log-ratio zero. -/
theorem consecutive_acceptance_token_bound (p q : Policy A) (τp τn : ℝ)
    (h : List A) (a : A)
    (hprev : prefixLogGate p q τp τn h = 1)
    (hnext : prefixLogGate p q τp τn (h ++ [a]) = 1) :
    tokenRatio p q h a ≤ consecutiveEnvelope τp τn h.length := by
  have hw : 0 ≤ tokenRatio p q h a := div_nonneg (q.nonneg h a) (p.nonneg h a)
  by_cases hz : tokenRatio p q h a = 0
  · rw [hz]; exact (Real.exp_pos _).le
  have hp := (prefix_log_bounds_of_gate p q τp τn h hprev).1
  have hn := (prefix_log_bounds_of_gate p q τp τn (h ++ [a]) hnext).2
  rw [logRatioTrace_snoc] at hn
  simp only [List.length_append, List.length_singleton, Nat.cast_add, Nat.cast_one] at hn
  have hb : Real.log (tokenRatio p q h a) ≤
      ((h.length : ℝ) + 1) * τp + (h.length : ℝ) * τn := by linarith
  have he := Real.exp_le_exp.mpr hb
  rwa [Real.exp_log (lt_of_le_of_ne hw (Ne.symm hz))] at he

noncomputable def localRetainedMoment (p q : Policy A) (τp τn : ℝ) (h : List A) : ℝ :=
  ∑ a, p.prob h a * prefixLogGate p q τp τn (h ++ [a]) * (tokenRatio p q h a)^2

/-- An accepted token after an upper-threshold excursion has a bounded
coefficient, even though its preceding prefix was rejected. -/
theorem upper_reentry_token_bound (p q : Policy A) (τp τn : ℝ)
    (h : List A) (a : A)
    (hupper : (h.length : ℝ) * τp ≤ logRatioTrace p q h)
    (hnext : prefixLogGate p q τp τn (h ++ [a]) = 1) :
    tokenRatio p q h a ≤ Real.exp τp := by
  have hw : 0 ≤ tokenRatio p q h a := div_nonneg (q.nonneg h a) (p.nonneg h a)
  by_cases hz : tokenRatio p q h a = 0
  · rw [hz]; exact (Real.exp_pos _).le
  have hn := (prefix_log_bounds_of_gate p q τp τn (h ++ [a]) hnext).2
  rw [logRatioTrace_snoc] at hn
  simp only [List.length_append, List.length_singleton, Nat.cast_add, Nat.cast_one] at hn
  have hb : Real.log (tokenRatio p q h a) ≤ τp := by linarith
  have he := Real.exp_le_exp.mpr hb
  rwa [Real.exp_log (lt_of_le_of_ne hw (Ne.symm hz))] at he

/-- The retained first moment keeps the actual accepted importance mass. -/
noncomputable def localRetainedMass (p q : Policy A) (τp τn : ℝ) (h : List A) : ℝ :=
  ∑ a, p.prob h a * prefixLogGate p q τp τn (h ++ [a]) * tokenRatio p q h a

theorem local_retained_mass_nonneg (p q : Policy A) (τp τn : ℝ) (h : List A) :
    0 ≤ localRetainedMass p q τp τn h := by
  apply Finset.sum_nonneg
  intro a _
  exact mul_nonneg (mul_nonneg (p.nonneg h a) (prefixLogGate_nonneg p q τp τn _))
    (div_nonneg (q.nonneg h a) (p.nonneg h a))

theorem local_retained_mass_le_one (p q : Policy A) (τp τn : ℝ) (h : List A) :
    localRetainedMass p q τp τn h ≤ 1 := by
  calc
    _ ≤ ∑ a, q.prob h a := by
      apply Finset.sum_le_sum
      intro a _
      rcases prefixLogGate_binary p q τp τn (h ++ [a]) with hz | ho
      · simpa only [hz, mul_zero, zero_mul] using q.nonneg h a
      · simp only [ho, mul_one]
        by_cases hp : p.prob h a = 0
        · simpa only [hp, zero_mul] using q.nonneg h a
        · apply le_of_eq
          unfold tokenRatio
          field_simp
    _ = 1 := q.mass h

/-- Keep the accepted first moment in the envelope bound before using
normalization. This records the mass removed by the actual prefix mask. -/
theorem local_retained_moment_of_envelope_mass (p q : Policy A) (τp τn C : ℝ)
    (h : List A)
    (hb : ∀ a, prefixLogGate p q τp τn (h ++ [a]) = 1 → tokenRatio p q h a ≤ C) :
    localRetainedMoment p q τp τn h ≤ C * localRetainedMass p q τp τn h := by
  unfold localRetainedMoment localRetainedMass
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro a _
  rcases prefixLogGate_binary p q τp τn (h ++ [a]) with hz | ho
  · simp [hz]
  · simp only [ho, mul_one]
    have hw : 0 ≤ p.prob h a * tokenRatio p q h a :=
      mul_nonneg (p.nonneg h a) (div_nonneg (q.nonneg h a) (p.nonneg h a))
    calc
      _ = (p.prob h a * tokenRatio p q h a) * tokenRatio p q h a := by ring
      _ ≤ (p.prob h a * tokenRatio p q h a) * C := mul_le_mul_of_nonneg_left (hb a ho) hw
      _ = _ := by ring

/-- The unit-mass consequence of the sharper retained-mass envelope. -/
theorem local_retained_moment_of_envelope (p q : Policy A) (τp τn C : ℝ)
    (h : List A) (hC : 0 ≤ C)
    (hb : ∀ a, prefixLogGate p q τp τn (h ++ [a]) = 1 → tokenRatio p q h a ≤ C) :
    localRetainedMoment p q τp τn h ≤ C := by
  exact (local_retained_moment_of_envelope_mass p q τp τn C h hb).trans
    (by simpa using mul_le_mul_of_nonneg_left (local_retained_mass_le_one p q τp τn h) hC)

theorem upper_reentry_moment_bound (p q : Policy A) (τp τn : ℝ) (h : List A)
    (hupper : (h.length : ℝ) * τp ≤ logRatioTrace p q h) :
    localRetainedMoment p q τp τn h ≤ Real.exp τp := by
  exact local_retained_moment_of_envelope p q τp τn _ h (Real.exp_pos _).le
    (fun a ha => upper_reentry_token_bound p q τp τn h a hupper ha)

/-- The aggregate contribution from upper-threshold rejected prefixes is
controlled by their probability, without an extra moment assumption. -/
theorem upper_reentry_contribution_bound (p q : Policy A) (τp τn : ℝ) (n : ℕ) :
    value p (fun h => if (h.length : ℝ) * τp < logRatioTrace p q h
      then localRetainedMoment p q τp τn h else 0) n [] ≤
    Real.exp τp * value p (fun h =>
      if (h.length : ℝ) * τp < logRatioTrace p q h then 1 else 0) n [] := by
  rw [← value_const_mul]
  apply value_mono
  intro h
  by_cases hu : (h.length : ℝ) * τp < logRatioTrace p q h
  · simp only [hu, if_pos, mul_one]
    exact upper_reentry_moment_bound p q τp τn h hu.le
  · simp [hu]

/-- Normalization improves the naive squared-envelope estimate to C, since
the first moment of a supported likelihood ratio is at most one. -/
theorem consecutive_acceptance_moment_bound (p q : Policy A) (τp τn : ℝ) (h : List A)
    (hprev : prefixLogGate p q τp τn h = 1) :
    localRetainedMoment p q τp τn h ≤ consecutiveEnvelope τp τn h.length := by
  exact local_retained_moment_of_envelope p q τp τn _ h (Real.exp_pos _).le
    (fun a ha => consecutive_acceptance_token_bound p q τp τn h a hprev ha)

theorem local_reentry_moment_bound (p q : Policy A) (τp τn : ℝ) (h : List A) :
    localRetainedMoment p q τp τn h ≤
      consecutiveEnvelope τp τn h.length * prefixLogGate p q τp τn h +
      (1 - prefixLogGate p q τp τn h) * localRetainedMoment p q τp τn h := by
  rcases prefixLogGate_binary p q τp τn h with hz | ho
  · simp [hz]
  · simpa [ho] using consecutive_acceptance_moment_bound p q τp τn h ho

/-- At depth n+1 all uncontrolled second-moment contribution is explicitly
assigned to transitions from a rejected prefix into an accepted one. -/
theorem retained_moment_reentry_bound (p q : Policy A) (τp τn : ℝ) (n : ℕ) :
    value p (localRetainedMoment p q τp τn) n [] ≤
      consecutiveEnvelope τp τn n * value p (prefixLogGate p q τp τn) n [] +
      value p (fun h => (1 - prefixLogGate p q τp τn h) *
        localRetainedMoment p q τp τn h) n [] := by
  rw [← value_const_mul, ← value_add]
  apply value_mono_on_support_length
  intro h hlen _
  simpa [hlen] using local_reentry_moment_bound p q τp τn h

/-- An explicit horizon scaling: T*tau_+ <= a and T*tau_- <= b
keep the consecutive-acceptance envelope below exp(a+b) at all t<=T. -/
theorem consecutive_envelope_horizon_bound (τp τn a b : ℝ) (n T : ℕ)
    (hτp : 0 ≤ τp) (hτn : 0 ≤ τn) (hn : n + 1 ≤ T)
    (hp : (T : ℝ) * τp ≤ a) (hm : (T : ℝ) * τn ≤ b) :
    consecutiveEnvelope τp τn n ≤ Real.exp (a + b) := by
  have hn' : (n : ℝ) + 1 ≤ T := by exact_mod_cast hn
  have h₁ := mul_le_mul_of_nonneg_right hn' hτp
  have h₂ := mul_le_mul_of_nonneg_right (show (n : ℝ) ≤ T by linarith) hτn
  apply Real.exp_le_exp.mpr
  linarith

/-- The previous prefix need only satisfy the lower threshold. A preceding
upper-threshold rejection does not destroy the retained-token envelope. -/
theorem lower_safe_retained_moment_bound (p q : Policy A) (τp τn : ℝ) (h : List A)
    (hlower : -((h.length : ℝ) * τn) ≤ logRatioTrace p q h) :
    localRetainedMoment p q τp τn h ≤
      consecutiveEnvelope τp τn h.length * localRetainedMass p q τp τn h := by
  apply local_retained_moment_of_envelope_mass p q τp τn _ h
  intro a ha
  have hw : 0 ≤ tokenRatio p q h a := div_nonneg (q.nonneg h a) (p.nonneg h a)
  by_cases hz : tokenRatio p q h a = 0
  · rw [hz]
    exact (Real.exp_pos _).le
  have hn := (prefix_log_bounds_of_gate p q τp τn (h ++ [a]) ha).2
  rw [logRatioTrace_snoc] at hn
  simp only [List.length_append, List.length_singleton, Nat.cast_add, Nat.cast_one] at hn
  have hb : Real.log (tokenRatio p q h a) ≤
      ((h.length : ℝ) + 1) * τp + (h.length : ℝ) * τn := by linarith
  have he := Real.exp_le_exp.mpr hb
  rwa [Real.exp_log (lt_of_le_of_ne hw (Ne.symm hz))] at he

/-- Retain the accepted first-moment factor on controlled histories;
lower-threshold excursions retain their exact second-moment contribution. -/
theorem retained_moment_lower_reentry_mass_bound (p q : Policy A) (τp τn : ℝ) (n : ℕ) :
    value p (localRetainedMoment p q τp τn) n [] ≤
      consecutiveEnvelope τp τn n * value p (localRetainedMass p q τp τn) n [] +
      value p (fun h => if logRatioTrace p q h < -((h.length : ℝ) * τn)
        then localRetainedMoment p q τp τn h else 0) n [] := by
  rw [← value_const_mul, ← value_add]
  apply value_mono_on_support_length
  intro h hlen _
  by_cases hl : logRatioTrace p q h < -((h.length : ℝ) * τn)
  · simp only [hl, if_pos]
    exact le_add_of_nonneg_left
      (mul_nonneg (Real.exp_pos _).le (local_retained_mass_nonneg p q τp τn h))
  · rw [if_neg hl, add_zero]
    simpa only [hlen] using lower_safe_retained_moment_bound p q τp τn h (le_of_not_gt hl)

/-- The previous uniform bound follows by replacing retained mass by one. -/
theorem retained_moment_lower_reentry_bound (p q : Policy A) (τp τn : ℝ) (n : ℕ) :
    value p (localRetainedMoment p q τp τn) n [] ≤
      consecutiveEnvelope τp τn n +
      value p (fun h => if logRatioTrace p q h < -((h.length : ℝ) * τn)
        then localRetainedMoment p q τp τn h else 0) n [] := by
  have hm : value p (localRetainedMass p q τp τn) n [] ≤ 1 := by
    calc
      _ ≤ value p (fun _ => 1) n [] := value_mono p _ _
        (fun h => local_retained_mass_le_one p q τp τn h) n []
      _ = 1 := value_const p 1 n []
  have hb := mul_le_mul_of_nonneg_left hm (Real.exp_pos
    (((n : ℝ) + 1) * τp + (n : ℝ) * τn)).le
  simpa only [mul_one] using (retained_moment_lower_reentry_mass_bound p q τp τn n).trans
    (add_le_add_right hb _)


theorem retained_moment_bound_of_reentry (p q : Policy A) (τp τn R : ℝ) (n : ℕ)
    (hR : value p (fun h => if logRatioTrace p q h < -((h.length : ℝ) * τn)
      then localRetainedMoment p q τp τn h else 0) n [] ≤ R) :
    value p (localRetainedMoment p q τp τn) n [] ≤ consecutiveEnvelope τp τn n + R := by
  exact (retained_moment_lower_reentry_bound p q τp τn n).trans (add_le_add_left hR _)

theorem retained_moment_horizon_bound (p q : Policy A) (τp τn a b R : ℝ) (n T : ℕ)
    (hτp : 0 ≤ τp) (hτn : 0 ≤ τn) (hn : n + 1 ≤ T)
    (hp : (T : ℝ) * τp ≤ a) (hm : (T : ℝ) * τn ≤ b)
    (hR : value p (fun h => if logRatioTrace p q h < -((h.length : ℝ) * τn)
      then localRetainedMoment p q τp τn h else 0) n [] ≤ R) :
    value p (localRetainedMoment p q τp τn) n [] ≤ Real.exp (a + b) + R := by
  exact (retained_moment_bound_of_reentry p q τp τn R n hR).trans
    (add_le_add_right (consecutive_envelope_horizon_bound τp τn a b n T hτp hτn hn hp hm) R)

/-- The new residual is a subset of the former rejection-to-acceptance
contribution, so its uniform bound is no larger than the former C + R. -/
theorem lower_reentry_le_all_reentry (p q : Policy A) (τp τn : ℝ) (n : ℕ) :
    value p (fun h => if logRatioTrace p q h < -((h.length : ℝ) * τn)
      then localRetainedMoment p q τp τn h else 0) n [] ≤
    value p (fun h => (1 - prefixLogGate p q τp τn h) *
      localRetainedMoment p q τp τn h) n [] := by
  apply value_mono
  intro h
  by_cases hl : logRatioTrace p q h < -((h.length : ℝ) * τn)
  · have hg : prefixLogGate p q τp τn h = 0 := by
      rcases prefixLogGate_binary p q τp τn h with hz | ho
      · exact hz
      · exact False.elim ((not_lt_of_ge (prefix_log_bounds_of_gate p q τp τn h ho).1) hl)
    simp [hl, hg]
  · rw [if_neg hl]
    apply mul_nonneg (sub_nonneg.mpr (prefixLogGate_le_one p q τp τn h))
    apply Finset.sum_nonneg
    intro a _
    exact mul_nonneg (mul_nonneg (p.nonneg h a)
      (prefixLogGate_nonneg p q τp τn _)) (sq_nonneg _)

end REINFORCEProMax.AutoregressiveTree
