import SparseReturn

namespace REINFORCEProMax
noncomputable section

/-- Two independent Bernoulli tokens; reward is revealed only after the second. -/
def bernoulliTokenMass (θ : ℝ) (a : Bool) : ℝ := if a then θ else 1 - θ
def bernoulliTokenScore (θ : ℝ) (a : Bool) : ℝ := if a then 1 / θ else -(1 / (1 - θ))
def twoTokenReward (a b : Bool) : ℝ := (if a then 1 else 0) + (if b then 1 else 0)
def twoTokenObjective (θ φ : ℝ) : ℝ :=
  ∑ a : Bool, ∑ b : Bool,
    bernoulliTokenMass θ a * bernoulliTokenMass φ b * twoTokenReward a b

theorem bernoulli_token_valid (θ : ℝ) (h0 : 0 < θ) (h1 : θ < 1) :
    (∀ a, 0 < bernoulliTokenMass θ a) ∧ (∑ a, bernoulliTokenMass θ a) = 1 := by
  constructor
  · intro a; cases a <;> simp [bernoulliTokenMass] <;> linarith
  · simp [bernoulliTokenMass, Fintype.sum_bool]

/-- The score is the derivative of log probability on the interior. -/
theorem bernoulli_token_log_derivative (θ : ℝ) (h0 : 0 < θ) (h1 : θ < 1) (a : Bool) :
    HasDerivAt (fun t => Real.log (bernoulliTokenMass t a)) (bernoulliTokenScore θ a) θ := by
  cases a
  · convert ((hasDerivAt_const θ (1 : ℝ)).sub (hasDerivAt_id θ)).log
      (show 1 - θ ≠ 0 by linarith) using 1 <;>
      simp [bernoulliTokenMass, bernoulliTokenScore, div_eq_mul_inv]
  · simpa [bernoulliTokenMass, bernoulliTokenScore] using
      (hasDerivAt_id θ).log (ne_of_gt h0)

theorem two_token_objective (θ φ : ℝ) : twoTokenObjective θ φ = θ + φ := by
  simp [twoTokenObjective, bernoulliTokenMass, twoTokenReward, Fintype.sum_bool]
  <;> ring

theorem two_token_objective_derivatives (θ φ : ℝ) :
    HasDerivAt (fun t => twoTokenObjective t φ) 1 θ ∧
    HasDerivAt (fun t => twoTokenObjective θ t) 1 φ := by
  simp_rw [two_token_objective]
  exact ⟨(hasDerivAt_id θ).add_const φ, (hasDerivAt_id φ).const_add θ⟩

/-- Using reward-to-go without outer time weights gives (gamma,1), not (1,1). -/
theorem discounted_two_token_score (θ φ gamma : ℝ)
    (hθ0 : 0 < θ) (hθ1 : θ < 1) (hφ0 : 0 < φ) (hφ1 : φ < 1) :
    (∑ a : Bool, ∑ b : Bool, bernoulliTokenMass θ a * bernoulliTokenMass φ b *
      (discountedReturn gamma [0, twoTokenReward a b] * bernoulliTokenScore θ a)) = gamma ∧
    (∑ a : Bool, ∑ b : Bool, bernoulliTokenMass θ a * bernoulliTokenMass φ b *
      (discountedReturn gamma [twoTokenReward a b] * bernoulliTokenScore φ b)) = 1 := by
  have hθ : θ ≠ 0 := ne_of_gt hθ0
  have hφ : φ ≠ 0 := ne_of_gt hφ0
  have hθ' : 1 - θ ≠ 0 := by linarith
  have hφ' : 1 - φ ≠ 0 := by linarith
  constructor <;>
    simp [Fintype.sum_bool, bernoulliTokenMass, bernoulliTokenScore, twoTokenReward,
      discountedReturn] <;> field_simp [hθ, hφ, hθ', hφ'] <;> ring

/-- Even a single learning-rate rescaling cannot fix the direction when gamma != 1. -/
theorem discounted_direction_not_rescalable (gamma : ℝ) (h : gamma ≠ 1) :
    ¬ ∃ c : ℝ, c * gamma = 1 ∧ c * 1 = 1 := by
  rintro ⟨c, hfirst, hsecond⟩
  have hc : c = 1 := by simpa using hsecond
  subst c
  exact h (by simpa using hfirst)

end
end REINFORCEProMax
