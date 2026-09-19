import TreePinsker
import RLOOMoments
import BatchReduction

namespace REINFORCEProMax.AutoregressiveTree

open scoped BigOperators

noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

def rolloutMass (p : Policy A) (n : ℕ) (y : Fin n → A) : ℝ :=
  pathMass p [] (List.ofFn y)

def responseRatioResidual (p q : Policy A) (n : ℕ) (y : Fin n → A) : ℝ :=
  surrogateRhs (ratioTrace p q (List.ofFn y))

theorem rolloutMass_normalized (p : Policy A) (n : ℕ) :
    (∑ y : Fin n → A, rolloutMass p n y) = 1 := by
  simpa [rolloutMass, pathSum_eq_sum_ofFn] using pathMass_normalized p n

theorem rolloutMass_expectation (p : Policy A) (f : List A → ℝ) (n : ℕ) :
    (∑ y : Fin n → A, rolloutMass p n y * f (List.ofFn y)) = value p f n [] := by
  simpa [rolloutMass, pathSum_eq_sum_ofFn, pathMass] using pathSum_mass_value p f n []

theorem surrogate_constant_reward_zero (p q : Policy A) (c : ℝ) (n : ℕ) (h : List A) :
    surrogate p q (fun _ => c) n h = 0 := by
  have hg : ∀ k h, gain p q (fun _ => c) k h = 0 := by
    intro k h
    simp only [gain, value_const, ← Finset.sum_mul, Finset.sum_sub_distrib, p.mass, q.mass]
    ring
  unfold surrogate
  induction n generalizing h with
  | zero => simp [additive, hg]
  | succ n ih => simp [additive, hg, ih]

/-- The trajectory ratio-sum residual has zero rollout mean, allowing the
actual RLOO baseline to cancel without assuming token independence. -/
theorem response_ratio_residual_mean_zero (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) :
    (∑ y : Fin n → A, rolloutMass p n y * responseRatioResidual p q n y) = 0 := by
  rw [show (fun y : Fin n → A => rolloutMass p n y * responseRatioResidual p q n y) =
      (fun y => rolloutMass p n y * surrogateRhs (ratioTrace p q (List.ofFn y))) from rfl,
    rolloutMass_expectation p (fun y => surrogateRhs (ratioTrace p q y)) n]
  have h := trajectory_surrogate_bridge p q (fun _ => 1) hs n []
  simpa [surrogate_constant_reward_zero, ratioTrace, surrogateRhs] using h

/-- The raw group-average RLOO surrogate increment has expectation equal
to the actual history-tree population surrogate for an off-policy candidate. -/
theorem iid_rloo_surrogate_increment_expectation {m : ℕ} (hm : m ≠ 0)
    (p q : Policy A) (R : List A → ℝ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) :
    (∑ s : Fin (m + 1) → (Fin n → A), iidMass (rolloutMass p n) s *
      iidRlooEstimator (rolloutMass p n) (fun y => R (List.ofFn y))
        (responseRatioResidual p q n) s) = surrogate p q R n [] := by
  rw [finite_iid_rloo_unbiased hm _ _ _ (rolloutMass_normalized p n)
    (response_ratio_residual_mean_zero p q hs n)]
  have hprod : (fun y : Fin n → A => rolloutMass p n y * R (List.ofFn y) *
      responseRatioResidual p q n y) =
      (fun y => rolloutMass p n y * (R (List.ofFn y) * surrogateRhs (ratioTrace p q (List.ofFn y)))) := by
    funext y
    simp only [responseRatioResidual, mul_assoc]
  rw [hprod, rolloutMass_expectation p (fun y => R y * surrogateRhs (ratioTrace p q y)) n]
  have h := trajectory_surrogate_bridge p q R hs n []
  simpa [ratioTrace, surrogateRhs] using h

def groupSurrogateIncrement {m : ℕ} (p q : Policy A) (R : List A → ℝ)
    (n : ℕ) (s : Fin (m + 1) → (Fin n → A)) : ℝ :=
  (m + 1 : ℝ) * iidRlooEstimator (rolloutMass p n) (fun y => R (List.ofFn y))
    (responseRatioResidual p q n) s

def groupActiveLength {m : ℕ} (length : List A → ℝ) {n : ℕ}
    (s : Fin (m + 1) → (Fin n → A)) : ℝ := ∑ i, length (List.ofFn (s i))

theorem group_surrogate_increment_expectation {m : ℕ} (hm : m ≠ 0)
    (p q : Policy A) (R : List A → ℝ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) :
    BatchReduction.expectation (fun s : Fin (m + 1) → (Fin n → A) => iidMass (rolloutMass p n) s)
      (groupSurrogateIncrement p q R n) = (m + 1 : ℝ) * surrogate p q R n [] := by
  unfold BatchReduction.expectation groupSurrogateIncrement
  simp_rw [mul_left_comm _ (m + 1 : ℝ), ← Finset.mul_sum]
  rw [iid_rloo_surrogate_increment_expectation hm p q R hs n]

theorem group_active_length_expectation {m : ℕ} (p : Policy A)
    (length : List A → ℝ) (n : ℕ) :
    BatchReduction.expectation (fun s : Fin (m + 1) → (Fin n → A) => iidMass (rolloutMass p n) s)
      (groupActiveLength length) = (m + 1 : ℝ) * value p length n [] := by
  have hm : (m + 1 : ℝ) ≠ 0 := by positivity
  have hpoint : ∀ s : Fin (m + 1) → (Fin n → A), groupActiveLength length s =
      (m + 1 : ℝ) * BatchReduction.sampleMean (fun y => length (List.ofFn y)) s := by
    intro s
    simp only [groupActiveLength, BatchReduction.sampleMean, Nat.cast_add, Nat.cast_one]
    field_simp
  unfold BatchReduction.expectation
  simp_rw [hpoint, mul_left_comm _ (m + 1 : ℝ), ← Finset.mul_sum]
  rw [BatchReduction.iid_sampleMean_expectation (by omega) _ _ (rolloutMass_normalized p n)]
  change (m + 1 : ℝ) * (∑ y : Fin n → A, rolloutMass p n y * length (List.ofFn y)) = _
  rw [rolloutMass_expectation p length n]

/-- The token-normalized population target is exactly the raw surrogate
divided by expected response length; the fixed group size cancels. -/
theorem group_surrogate_ratio {m : ℕ} (hm : m ≠ 0) (p q : Policy A)
    (R length : List A → ℝ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) :
    BatchReduction.expectation (fun s : Fin (m + 1) → (Fin n → A) => iidMass (rolloutMass p n) s)
        (groupSurrogateIncrement p q R n) /
      BatchReduction.expectation (fun s : Fin (m + 1) → (Fin n → A) => iidMass (rolloutMass p n) s)
        (groupActiveLength length) = surrogate p q R n [] / value p length n [] := by
  rw [group_surrogate_increment_expectation hm p q R hs n, group_active_length_expectation p length n]
  have hm' : (m + 1 : ℝ) ≠ 0 := by positivity
  by_cases hlen : value p length n [] = 0
  · simp [hlen]
  · field_simp
    ring

/-- The same ratio identity with the positivity condition needed for its
statistical interpretation as a token-normalized target. The unrestricted
identity above is still useful algebraically because Lean's real division is
totalized at a zero denominator; this version rules out that degenerate case. -/
theorem group_surrogate_ratio_of_positive_length {m : ℕ} (hm : m ≠ 0) (p q : Policy A)
    (R length : List A → ℝ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ)
    (hpos : 0 < value p length n []) :
    BatchReduction.expectation (fun s : Fin (m + 1) → (Fin n → A) => iidMass (rolloutMass p n) s)
        (groupSurrogateIncrement p q R n) /
      BatchReduction.expectation (fun s : Fin (m + 1) → (Fin n → A) => iidMass (rolloutMass p n) s)
        (groupActiveLength length) = surrogate p q R n [] /
      value p length n [] := by
  rw [group_surrogate_increment_expectation hm p q R hs n,
    group_active_length_expectation p length n]
  have hm' : (m + 1 : ℝ) ≠ 0 := by positivity
  have hlen : value p length n [] ≠ 0 := ne_of_gt hpos
  field_simp [hm', hlen]
  ring

end
end REINFORCEProMax.AutoregressiveTree
