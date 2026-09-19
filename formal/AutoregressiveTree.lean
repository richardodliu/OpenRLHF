import Mathlib.Tactic

/-!
A finite-vocabulary autoregressive model. Policies may depend on the entire
history; no independence across positions or on-policy restriction is imposed.
`value` sums terminal rewards over the full continuation tree. `additive`
sums a local signal over policy-generated prefixes, with its first argument
equal to the number of actions remaining after the current action.
-/
namespace REINFORCEProMax.AutoregressiveTree

open scoped BigOperators

variable {A : Type*} [Fintype A]

structure Policy (A : Type*) [Fintype A] where
  prob : List A → A → ℝ
  nonneg : ∀ h a, 0 ≤ prob h a
  mass : ∀ h, ∑ a, prob h a = 1

noncomputable def value (p : Policy A) (R : List A → ℝ) : ℕ → List A → ℝ
  | 0, h => R h
  | n + 1, h => ∑ a, p.prob h a * value p R n (h ++ [a])

noncomputable def additive (p : Policy A) (f : ℕ → List A → ℝ) : ℕ → List A → ℝ
  | 0, _ => 0
  | n + 1, h => f n h + ∑ a, p.prob h a * additive p f n (h ++ [a])

noncomputable def localTV (p q : Policy A) (h : List A) : ℝ :=
  (1 / 2 : ℝ) * ∑ a, |q.prob h a - p.prob h a|

noncomputable def gain (p q : Policy A) (R : List A → ℝ)
    (n : ℕ) (h : List A) : ℝ :=
  ∑ a, (q.prob h a - p.prob h a) * value p R n (h ++ [a])

noncomputable def surrogate (p q : Policy A) (R : List A → ℝ)
    (n : ℕ) (h : List A) : ℝ := additive p (gain p q R) n h

noncomputable def maskedGain (p q : Policy A) (R : List A → ℝ)
    (m : List A → ℝ) (n : ℕ) (h : List A) : ℝ := m h * gain p q R n h

noncomputable def maskedSurrogate (p q : Policy A) (R : List A → ℝ)
    (m : List A → ℝ) (n : ℕ) (h : List A) : ℝ := additive p (maskedGain p q R m) n h

noncomputable def driftCost (p q : Policy A) (n : ℕ) (h : List A) : ℝ :=
  additive p (fun _ h => localTV p q h) n h

/- A finite-tree second moment for a multiplicative prefix weight.  This is
   deliberately stated for an arbitrary nonnegative local weight `rho`; a
   likelihood-ratio interpretation is obtained by instantiating `rho` with
   the current/rollout token ratio. -/
noncomputable def prefixSecondMoment (p : Policy A) (rho : List A → A → ℝ)
    : ℕ → List A → ℝ
  | 0, _ => 1
  | n + 1, h =>
      ∑ a, p.prob h a * (rho h a) ^ 2 * prefixSecondMoment p rho n (h ++ [a])

theorem prefix_second_moment_bound
    (p : Policy A) (rho : List A → A → ℝ) (C : ℝ)
    (hC : 0 ≤ C)
    (hlocal : ∀ h, ∑ a, p.prob h a * (rho h a) ^ 2 ≤ C)
    (n : ℕ) (h : List A) :
    prefixSecondMoment p rho n h ≤ C ^ n := by
  induction n generalizing h with
  | zero => simp [prefixSecondMoment]
  | succ n ih =>
      have hcoeff : ∀ a, 0 ≤ p.prob h a * (rho h a) ^ 2 := by
        intro a
        exact mul_nonneg (p.nonneg h a) (sq_nonneg _)
      have hsum :
          (∑ a, p.prob h a * (rho h a) ^ 2 * prefixSecondMoment p rho n (h ++ [a])) ≤
            ∑ a, p.prob h a * (rho h a) ^ 2 * C ^ n := by
        apply Finset.sum_le_sum
        intro a _
        exact mul_le_mul_of_nonneg_left (ih (h ++ [a])) (hcoeff a)
      rw [prefixSecondMoment]
      calc
        (∑ a, p.prob h a * (rho h a) ^ 2 * prefixSecondMoment p rho n (h ++ [a])) ≤
            ∑ a, p.prob h a * (rho h a) ^ 2 * C ^ n := hsum
        _ = (∑ a, p.prob h a * (rho h a) ^ 2) * C ^ n := by
          rw [Finset.sum_mul]
        _ ≤ C * C ^ n := by
          exact mul_le_mul_of_nonneg_right (hlocal h) (pow_nonneg hC n)
        _ = C ^ (n + 1) := by ring

/- A finite local bridge from a centered log-ratio MGF bound to the
   second-moment condition used by `prefix_second_moment_bound`.  The
   non-positive conditional mean is explicit here: for a likelihood ratio it
   is the finite KL sign inequality, while keeping it as a hypothesis makes
   this lemma independent of a particular policy representation. -/
theorem local_second_moment_of_centered_log_mgf
    (p : Policy A) (z : List A → A → ℝ) (μ σ : List A → ℝ)
    (hμ : ∀ h, μ h ≤ 0)
    (hmgf : ∀ h,
      ∑ a, p.prob h a * Real.exp (2 * (z h a - μ h)) ≤
        Real.exp (2 * (σ h) ^ 2)) :
    ∀ h, ∑ a, p.prob h a * (Real.exp (z h a)) ^ 2 ≤
      Real.exp (2 * (σ h) ^ 2) := by
  intro h
  have hfactor :
      (∑ a, p.prob h a * (Real.exp (z h a)) ^ 2) =
        Real.exp (2 * μ h) *
          (∑ a, p.prob h a * Real.exp (2 * (z h a - μ h))) := by
    calc
      (∑ a, p.prob h a * (Real.exp (z h a)) ^ 2) =
          ∑ a, p.prob h a * Real.exp (2 * z h a) := by
            apply Finset.sum_congr rfl
            intro a ha
            rw [pow_two, ← Real.exp_add]
            congr 1
            ring
      _ = ∑ a, p.prob h a *
          (Real.exp (2 * μ h) * Real.exp (2 * (z h a - μ h))) := by
            apply Finset.sum_congr rfl
            intro a ha
            rw [← Real.exp_add]
            congr 1
            ring
      _ = Real.exp (2 * μ h) *
          (∑ a, p.prob h a * Real.exp (2 * (z h a - μ h))) := by
            symm
            calc
              Real.exp (2 * μ h) *
                  (∑ a, p.prob h a * Real.exp (2 * (z h a - μ h))) =
                  ∑ a, Real.exp (2 * μ h) *
                    (p.prob h a * Real.exp (2 * (z h a - μ h))) := by
                      exact Finset.mul_sum _ _ _
              _ = ∑ a, p.prob h a *
                  (Real.exp (2 * μ h) * Real.exp (2 * (z h a - μ h))) := by
                    apply Finset.sum_congr rfl
                    intro a ha
                    ring
  have hμexp : Real.exp (2 * μ h) ≤ 1 := by
    rw [← Real.exp_zero]
    apply Real.exp_le_exp.mpr
    linarith [hμ h]
  calc
    (∑ a, p.prob h a * (Real.exp (z h a)) ^ 2) =
        Real.exp (2 * μ h) *
          (∑ a, p.prob h a * Real.exp (2 * (z h a - μ h))) := hfactor
    _ ≤ Real.exp (2 * μ h) * Real.exp (2 * (σ h) ^ 2) := by
      exact mul_le_mul_of_nonneg_left (hmgf h) (Real.exp_nonneg _)
    _ ≤ 1 * Real.exp (2 * (σ h) ^ 2) := by
      exact mul_le_mul_of_nonneg_right hμexp (Real.exp_nonneg _)
    _ = Real.exp (2 * (σ h) ^ 2) := by ring

theorem abs_weighted_sum_le (w f : A → ℝ) (B : ℝ)
    (hw : ∀ a, 0 ≤ w a) (hf : ∀ a, |f a| ≤ B) :
    |∑ a, w a * f a| ≤ (∑ a, w a) * B := by
  calc
    _ ≤ ∑ a, |w a * f a| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ a, w a * |f a| := by simp_rw [abs_mul, abs_of_nonneg (hw _)]
    _ ≤ ∑ a, w a * B := Finset.sum_le_sum fun a _ =>
      mul_le_mul_of_nonneg_left (hf a) (hw a)
    _ = _ := (Finset.sum_mul _ _ _).symm

theorem abs_signed_sum_le (w f : A → ℝ) (B : ℝ) (hf : ∀ a, |f a| ≤ B) :
    |∑ a, w a * f a| ≤ (∑ a, |w a|) * B := by
  calc
    _ ≤ ∑ a, |w a * f a| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ a, |w a| * |f a| := by simp_rw [abs_mul]
    _ ≤ ∑ a, |w a| * B := Finset.sum_le_sum fun a _ =>
      mul_le_mul_of_nonneg_left (hf a) (abs_nonneg _)
    _ = _ := (Finset.sum_mul _ _ _).symm

theorem value_abs_le (p : Policy A) (R : List A → ℝ) (B : ℝ)
    (hR : ∀ h, |R h| ≤ B) (n : ℕ) (h : List A) : |value p R n h| ≤ B := by
  induction n generalizing h with
  | zero => exact hR h
  | succ n ih =>
      simpa only [value, p.mass, one_mul] using
        abs_weighted_sum_le (p.prob h) (fun a => value p R n (h ++ [a])) B
          (p.nonneg h) (fun a => ih (h ++ [a]))

/-- The local gain is the current-policy expectation of the rollout
advantage. The baseline value disappears by policy normalization. -/
theorem gain_eq_advantage (p q : Policy A) (R : List A → ℝ)
    (n : ℕ) (h : List A) :
    gain p q R n h = ∑ a, q.prob h a *
      (value p R n (h ++ [a]) - value p R (n + 1) h) := by
  simp only [gain, mul_sub, sub_mul, Finset.sum_sub_distrib]
  rw [← Finset.sum_mul, q.mass, one_mul, value]

theorem masked_surrogate_eq_surrogate
    (p q : Policy A) (R : List A → ℝ) (m : List A → ℝ)
    (hm : ∀ h, m h = 0 ∨ m h = 1)
    (hinactive : ∀ h, m h = 0 → ∀ a, q.prob h a = p.prob h a)
    (n : ℕ) (h : List A) :
    maskedSurrogate p q R m n h = surrogate p q R n h := by
  have hgain : ∀ k h, maskedGain p q R m k h = gain p q R k h := by
    intro k h
    rcases hm h with hz | ho
    · rw [maskedGain, hz]
      unfold gain
      have hzero : (∑ a, (q.prob h a - p.prob h a) * value p R k (h ++ [a])) = 0 := by
        apply Finset.sum_eq_zero
        intro a _
        rw [hinactive h hz a]
        ring
      rw [hzero]
      ring
    · rw [maskedGain, ho, one_mul]
  unfold maskedSurrogate surrogate
  have hfun : maskedGain p q R m = gain p q R := by
    funext k h'
    exact hgain k h'
  rw [hfun]

/-- A genuine performance-difference identity on the complete history tree:
the current-policy occupancy, rather than rollout occupancy, evaluates gain. -/
theorem performance_difference (p q : Policy A) (R : List A → ℝ)
    (n : ℕ) (h : List A) :
    value q R n h - value p R n h = additive q (gain p q R) n h := by
  induction n generalizing h with
  | zero => simp [value, additive]
  | succ n ih =>
      simp only [value, additive, gain]
      simp_rw [← ih, mul_sub, sub_mul]
      simp only [Finset.sum_sub_distrib]
      ring

theorem gain_abs_le (p q : Policy A) (R : List A → ℝ) (B : ℝ)
    (hR : ∀ h, |R h| ≤ B) (n : ℕ) (h : List A) :
    |gain p q R n h| ≤ 2 * B * localTV p q h := by
  have hb := abs_signed_sum_le (fun a => q.prob h a - p.prob h a)
    (fun a => value p R n (h ++ [a])) B (fun a => value_abs_le p R B hR n _)
  dsimp [gain, localTV] at *
  nlinarith

theorem additive_abs_le (p : Policy A) (f : ℕ → List A → ℝ) (B : ℝ)
    (hf : ∀ n h, |f n h| ≤ B) (n : ℕ) (h : List A) :
    |additive p f n h| ≤ (n : ℝ) * B := by
  induction n generalizing h with
  | zero => simp [additive]
  | succ n ih =>
      have hs := abs_weighted_sum_le (p.prob h)
        (fun a => additive p f n (h ++ [a])) ((n : ℝ) * B)
        (p.nonneg h) (fun a => ih _)
      rw [p.mass, one_mul] at hs
      have ha := abs_add (f n h) (∑ a, p.prob h a * additive p f n (h ++ [a]))
      simp only [additive, Nat.cast_add, Nat.cast_one]
      linarith [hf n h]

/-- Changing the prefix-generating policy has quadratic-in-horizon effect
on any bounded common local signal. This is proved from the tree recursion. -/
theorem additive_difference_abs_le (p q : Policy A) (f : ℕ → List A → ℝ)
    (B ε : ℝ) (hB : 0 ≤ B)
    (hf : ∀ n h, |f n h| ≤ B) (hTV : ∀ h, localTV p q h ≤ ε)
    (n : ℕ) (h : List A) :
    |additive q f n h - additive p f n h| ≤
      B * ε * (n : ℝ) * ((n : ℝ) - 1) := by
  induction n generalizing h with
  | zero => simp [additive]
  | succ n ih =>
      have heq : additive q f (n + 1) h - additive p f (n + 1) h =
          (∑ a, q.prob h a * (additive q f n (h ++ [a]) - additive p f n (h ++ [a]))) +
          ∑ a, (q.prob h a - p.prob h a) * additive p f n (h ++ [a]) := by
        simp only [additive, mul_sub, sub_mul, Finset.sum_sub_distrib]
        ring
      have hfirst := abs_weighted_sum_le (q.prob h)
        (fun a => additive q f n (h ++ [a]) - additive p f n (h ++ [a]))
        (B * ε * (n : ℝ) * ((n : ℝ) - 1)) (q.nonneg h) (fun a => ih _)
      rw [q.mass, one_mul] at hfirst
      have hsecond := abs_signed_sum_le (fun a => q.prob h a - p.prob h a)
        (fun a => additive p f n (h ++ [a])) ((n : ℝ) * B)
        (fun a => additive_abs_le p f B hf n _)
      have hl1 : (∑ a, |q.prob h a - p.prob h a|) ≤ 2 * ε := by
        have := hTV h
        dsimp [localTV] at this
        linarith
      have hscale := mul_le_mul_of_nonneg_right hl1
        (show 0 ≤ (n : ℝ) * B by positivity)
      rw [heq]
      have ha := abs_add
        (∑ a, q.prob h a * (additive q f n (h ++ [a]) - additive p f n (h ++ [a])))
        (∑ a, (q.prob h a - p.prob h a) * additive p f n (h ++ [a]))
      push_cast
      nlinarith

/-- The finite-horizon quadratic surrogate error bound, with no separate
ratio/coupling or imported performance-difference assumption. -/
theorem surrogate_error_quadratic (p q : Policy A) (R : List A → ℝ)
    (B ε : ℝ) (hB : 0 ≤ B) (hε : 0 ≤ ε)
    (hR : ∀ h, |R h| ≤ B) (hTV : ∀ h, localTV p q h ≤ ε)
    (n : ℕ) (h : List A) :
    |value q R n h - value p R n h - surrogate p q R n h| ≤
      2 * B * (n : ℝ) * ((n : ℝ) - 1) * ε ^ 2 := by
  rw [performance_difference, surrogate]
  have hg : ∀ k h, |gain p q R k h| ≤ 2 * B * ε := by
    intro k h
    exact (gain_abs_le p q R B hR k h).trans
      (mul_le_mul_of_nonneg_left (hTV h) (by positivity))
  have hb := additive_difference_abs_le p q (gain p q R) (2 * B * ε) ε
    (by positivity) hg hTV n h
  nlinarith

theorem abs_weighted_sum_le_sum (w f F : A → ℝ)
    (hw : ∀ a, 0 ≤ w a) (hf : ∀ a, |f a| ≤ F a) :
    |∑ a, w a * f a| ≤ ∑ a, w a * F a := by
  calc
    _ ≤ ∑ a, |w a * f a| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ a, w a * |f a| := by simp_rw [abs_mul, abs_of_nonneg (hw _)]
    _ ≤ _ := Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (hf a) (hw a)

/-- Expected-sum-TV control of the true reward change. The occupancy cost is
explicitly generated by the rollout policy, even though the new policy is q. -/
theorem value_difference_abs_le_drift (p q : Policy A) (R : List A → ℝ)
    (B : ℝ) (hR : ∀ h, |R h| ≤ B) (n : ℕ) (h : List A) :
    |value q R n h - value p R n h| ≤ 2 * B * driftCost p q n h := by
  induction n generalizing h with
  | zero => simp [value, driftCost, additive]
  | succ n ih =>
      have heq : value q R (n + 1) h - value p R (n + 1) h =
          (∑ a, p.prob h a * (value q R n (h ++ [a]) - value p R n (h ++ [a]))) +
          ∑ a, (q.prob h a - p.prob h a) * value q R n (h ++ [a]) := by
        simp only [value, mul_sub, sub_mul, Finset.sum_sub_distrib]
        ring
      have hfirst := abs_weighted_sum_le_sum (p.prob h)
        (fun a => value q R n (h ++ [a]) - value p R n (h ++ [a]))
        (fun a => 2 * B * driftCost p q n (h ++ [a])) (p.nonneg h) (fun a => ih _)
      have hsecond := abs_signed_sum_le (fun a => q.prob h a - p.prob h a)
        (fun a => value q R n (h ++ [a])) B (fun a => value_abs_le q R B hR n _)
      have hsum : (∑ a, p.prob h a * (2 * B * driftCost p q n (h ++ [a]))) =
          2 * B * ∑ a, p.prob h a * driftCost p q n (h ++ [a]) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro a _
        ring
      rw [hsum] at hfirst
      rw [heq]
      have ha := abs_add
        (∑ a, p.prob h a * (value q R n (h ++ [a]) - value p R n (h ++ [a])))
        (∑ a, (q.prob h a - p.prob h a) * value q R n (h ++ [a]))
      change _ ≤ 2 * B * (localTV p q h + ∑ a, p.prob h a * driftCost p q n (h ++ [a]))
      dsimp [localTV]
      nlinarith

theorem surrogate_abs_le_drift (p q : Policy A) (R : List A → ℝ)
    (B : ℝ) (hR : ∀ h, |R h| ≤ B) (n : ℕ) (h : List A) :
    |surrogate p q R n h| ≤ 2 * B * driftCost p q n h := by
  induction n generalizing h with
  | zero => simp [surrogate, driftCost, additive]
  | succ n ih =>
      have hfirst := gain_abs_le p q R B hR n h
      have hsecond := abs_weighted_sum_le_sum (p.prob h)
        (fun a => surrogate p q R n (h ++ [a]))
        (fun a => 2 * B * driftCost p q n (h ++ [a])) (p.nonneg h) (fun a => ih _)
      have hsum : (∑ a, p.prob h a * (2 * B * driftCost p q n (h ++ [a]))) =
          2 * B * ∑ a, p.prob h a * driftCost p q n (h ++ [a]) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro a _
        ring
      rw [hsum] at hsecond
      have ha := abs_add (gain p q R n h)
        (∑ a, p.prob h a * surrogate p q R n (h ++ [a]))
      change |gain p q R n h + ∑ a, p.prob h a * surrogate p q R n (h ++ [a])| ≤
        2 * B * (localTV p q h + ∑ a, p.prob h a * driftCost p q n (h ++ [a]))
      linarith

/-- The second remainder bound also follows from the normalized tree alone:
no extra coupling hypothesis and no bound on individual importance ratios. -/
theorem surrogate_error_linear (p q : Policy A) (R : List A → ℝ)
    (B : ℝ) (hR : ∀ h, |R h| ≤ B) (n : ℕ) (h : List A) :
    |value q R n h - value p R n h - surrogate p q R n h| ≤
      4 * B * driftCost p q n h := by
  have h1 := value_difference_abs_le_drift p q R B hR n h
  have h2 := surrogate_abs_le_drift p q R B hR n h
  have h3 := abs_sub (value q R n h - value p R n h) (surrogate p q R n h)
  linarith

theorem performance_lower_bound (p q : Policy A) (R : List A → ℝ)
    (B ε : ℝ) (hB : 0 ≤ B) (hε : 0 ≤ ε)
    (hR : ∀ h, |R h| ≤ B) (hTV : ∀ h, localTV p q h ≤ ε)
    (n : ℕ) (h : List A) :
    surrogate p q R n h - min (2 * B * (n : ℝ) * ((n : ℝ) - 1) * ε ^ 2)
      (4 * B * driftCost p q n h) ≤ value q R n h - value p R n h := by
  have h1 := surrogate_error_quadratic p q R B ε hB hε hR hTV n h
  have h2 := surrogate_error_linear p q R B hR n h
  have hmin := le_min h1 h2
  have hl := (abs_le.mp hmin).1
  linarith

end REINFORCEProMax.AutoregressiveTree
