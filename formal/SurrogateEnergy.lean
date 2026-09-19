import RLOOSurrogate

/-!
The additive likelihood-ratio residual is a martingale sum on the actual
history tree. Its second moment is the sum of conditional chi-square costs,
without independent tokens or a product-ratio second-moment bound.
-/
namespace REINFORCEProMax.AutoregressiveTree

open scoped BigOperators
noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

def localRatioEnergy (p q : Policy A) (h : List A) : ℝ :=
  ∑ a, p.prob h a * (tokenRatio p q h a - 1) ^ 2

def ratioEnergy (p q : Policy A) (T : ℕ) : ℝ :=
  additive p (fun _ => localRatioEnergy p q) T []

theorem local_ratio_residual_mean_zero (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (h : List A) :
    (∑ a, p.prob h a * (tokenRatio p q h a - 1)) = 0 := by
  simp_rw [mul_sub, prob_mul_ratio p q hs, mul_one]
  rw [Finset.sum_sub_distrib, p.mass, q.mass]
  ring

theorem local_ratio_energy_nonneg (p q : Policy A) (h : List A) :
    0 ≤ localRatioEnergy p q h :=
  Finset.sum_nonneg fun a _ => mul_nonneg (p.nonneg h a) (sq_nonneg _)

/-- In particular, shared absorbing EOS kernels contribute zero energy. -/
theorem local_ratio_energy_zero_of_equal (p q : Policy A) (h : List A)
    (heq : ∀ a, q.prob h a = p.prob h a) : localRatioEnergy p q h = 0 := by
  apply Finset.sum_eq_zero
  intro a _
  by_cases hz : p.prob h a = 0
  · simp [hz]
  · simp [tokenRatio, heq a, div_self hz]

theorem residual_square_step (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (h : List A) (c : ℝ) :
    (∑ a, p.prob h a * (c + (tokenRatio p q h a - 1)) ^ 2) =
      c ^ 2 + localRatioEnergy p q h := by
  have heq : ∀ a, p.prob h a * (c + (tokenRatio p q h a - 1)) ^ 2 =
      p.prob h a * c ^ 2 + 2 * c * (p.prob h a * (tokenRatio p q h a - 1)) +
        p.prob h a * (tokenRatio p q h a - 1) ^ 2 := by intro a; ring
  simp_rw [heq, Finset.sum_add_distrib]
  rw [← Finset.sum_mul, p.mass, one_mul, ← Finset.mul_sum,
    local_ratio_residual_mean_zero p q hs h]
  simp [localRatioEnergy]

theorem trajectory_residual_second_moment (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) (h : List A) :
    value p (fun y => (surrogateRhs (ratioTrace p q y)) ^ 2) n h =
      (surrogateRhs (ratioTrace p q h)) ^ 2 +
        additive p (fun _ => localRatioEnergy p q) n h := by
  induction n generalizing h with
  | zero => simp [value, additive]
  | succ n ih =>
      simp only [value, ih, ratioTrace_snoc, surrogateRhs_snoc, mul_add,
        Finset.sum_add_distrib]
      rw [residual_square_step p q hs]
      simp only [additive]
      ring

theorem response_residual_energy (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (T : ℕ) :
    BatchReduction.expectation (rolloutMass p T)
      (fun y => (responseRatioResidual p q T y) ^ 2) = ratioEnergy p q T := by
  unfold BatchReduction.expectation responseRatioResidual
  rw [rolloutMass_expectation p (fun y => (surrogateRhs (ratioTrace p q y)) ^ 2) T,
    trajectory_residual_second_moment p q hs]
  simp [ratioTrace, surrogateRhs, ratioEnergy]

theorem ratio_energy_eq_prefix_sum (p q : Policy A) (T : ℕ) :
    ratioEnergy p q T = ∑ t ∈ Finset.range T, value p (localRatioEnergy p q) t [] := by
  exact additive_eq_prefix_sum p (fun _ => localRatioEnergy p q) T []

theorem ratio_energy_nonneg (p q : Policy A) (T : ℕ) : 0 ≤ ratioEnergy p q T := by
  rw [ratio_energy_eq_prefix_sum]
  exact Finset.sum_nonneg fun t _ => value_nonneg p _ (local_ratio_energy_nonneg p q) t []

/-- Only positive rollout actions need a ratio bound; shared zero-probability
actions, including absorbing EOS padding, impose no artificial lower bound. -/
theorem local_ratio_energy_of_bound (p q : Policy A) (d : ℝ) (hd : 0 ≤ d)
    (h : List A) (hr : ∀ a, 0 < p.prob h a → |tokenRatio p q h a - 1| ≤ d) :
    localRatioEnergy p q h ≤ d ^ 2 := by
  calc
    _ ≤ ∑ a, p.prob h a * d ^ 2 := by
      apply Finset.sum_le_sum
      intro a _
      by_cases hz : p.prob h a = 0
      · simp [hz]
      · have hb := (sq_le_sq₀ (abs_nonneg _) hd).mpr
          (hr a (lt_of_le_of_ne (p.nonneg h a) (Ne.symm hz)))
        rw [sq_abs] at hb
        exact mul_le_mul_of_nonneg_left hb (p.nonneg h a)
    _ = _ := by rw [← Finset.sum_mul, p.mass, one_mul]

theorem additive_ratio_energy_bound (p q : Policy A) (D : ℝ) (N : ℕ)
    (hlocal : ∀ h, h.length < N → localRatioEnergy p q h ≤ D)
    (n : ℕ) (h : List A) (hh : h.length + n ≤ N) :
    additive p (fun _ => localRatioEnergy p q) n h ≤ (n : ℝ) * D := by
  induction n generalizing h with
  | zero => simp [additive]
  | succ n ih =>
      have hb : (∑ a, p.prob h a * additive p (fun _ => localRatioEnergy p q) n (h ++ [a])) ≤
          (n : ℝ) * D := by
        calc
          _ ≤ ∑ a, p.prob h a * ((n : ℝ) * D) := by
            apply Finset.sum_le_sum
            intro a _
            have hh' : (h ++ [a]).length + n ≤ N := by
              simp only [List.length_append, List.length_singleton]
              omega
            exact mul_le_mul_of_nonneg_left (ih _ hh') (p.nonneg h a)
          _ = _ := by rw [← Finset.sum_mul, p.mass, one_mul]
      have hl := hlocal h (by omega)
      simp only [additive, Nat.cast_add, Nat.cast_one]
      linarith

theorem ratio_energy_horizon_bound (p q : Policy A) (D : ℝ) (T : ℕ)
    (hlocal : ∀ h, h.length < T → localRatioEnergy p q h ≤ D) :
    ratioEnergy p q T ≤ (T : ℝ) * D :=
  additive_ratio_energy_bound p q D T hlocal T [] (by simp)

end
end REINFORCEProMax.AutoregressiveTree
