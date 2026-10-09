import TreePolicyGradient

/-! The derivative of an actual autoregressive path mass is its mass times
its sum of conditional token scores. Positive sampled paths suffice for the
score interpretation; null paths have zero differential by nonnegativity. -/
namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators
noncomputable section
variable {A E : Type*} [Fintype A] [DecidableEq A]
  [NormedAddCommGroup E] [NormedSpace ℝ E]

def tokenScoreSum (P : E → Policy A) (θ : E) : List A → List A → E →L[ℝ] ℝ
  | _, [] => 0
  | h, a :: y =>
      ((P θ).prob h a)⁻¹ • fderiv ℝ (fun η => (P η).prob h a) θ +
        tokenScoreSum P θ (h ++ [a]) y

theorem pathMass_hasFDerivAt_tokenScoreSum (P : E → Policy A) (θ : E)
    (hd : ∀ h a, DifferentiableAt ℝ (fun η => (P η).prob h a) θ)
    (h y : List A) (hz : pathMass (P θ) h y ≠ 0) :
    HasFDerivAt (fun η => pathMass (P η) h y)
      (pathMass (P θ) h y • tokenScoreSum P θ h y) θ := by
  induction y generalizing h with
  | nil => simpa [pathMass, tokenScoreSum] using (hasFDerivAt_const (𝕜 := ℝ) (1 : ℝ) θ)
  | cons a y ih =>
      have hn := mul_ne_zero_iff.mp hz
      have hb := (hd h a).hasFDerivAt.mul (ih (h ++ [a]) hn.2)
      convert hb using 1
      simp only [pathMass, tokenScoreSum, smul_add, smul_smul]
      rw [add_comm]
      congr 1
      congr 1
      field_simp [hn.1]

theorem responseScore_eq_tokenScoreSum (P : E → Policy A) (n : ℕ) (θ : E)
    (hd : ∀ h a, DifferentiableAt ℝ (fun η => (P η).prob h a) θ)
    (y : Fin n → A) (hz : responseMass P n θ y ≠ 0) :
    responseScore P n θ y = tokenScoreSum P θ [] (List.ofFn y) := by
  have hb := pathMass_hasFDerivAt_tokenScoreSum P θ hd [] (List.ofFn y) hz
  unfold responseScore policyScore
  change (responseMass P n θ y)⁻¹ •
      fderiv ℝ (fun η => pathMass (P η) [] (List.ofFn y)) θ = _
  rw [hb.fderiv]
  change (responseMass P n θ y)⁻¹ •
      (responseMass P n θ y • tokenScoreSum P θ [] (List.ofFn y)) = _
  simp [smul_smul, hz]

/-- Equality under the sampling mass also includes null responses. -/
theorem weighted_responseScore_eq_tokenScoreSum (P : E → Policy A) (n : ℕ) (θ : E)
    (hd : ∀ h a, DifferentiableAt ℝ (fun η => (P η).prob h a) θ)
    (y : Fin n → A) :
    responseMass P n θ y • responseScore P n θ y =
      responseMass P n θ y • tokenScoreSum P θ [] (List.ofFn y) := by
  by_cases hz : responseMass P n θ y = 0
  · simp [hz]
  · rw [responseScore_eq_tokenScoreSum P n θ hd y hz]

/-- Each summand is the differential of the corresponding token log-probability. -/
theorem token_log_probability_hasFDerivAt (P : E → Policy A) (θ : E)
    (h : List A) (a : A)
    (hd : DifferentiableAt ℝ (fun η => (P η).prob h a) θ)
    (hz : (P θ).prob h a ≠ 0) :
    HasFDerivAt (fun η => Real.log ((P η).prob h a))
      (((P θ).prob h a)⁻¹ • fderiv ℝ (fun η => (P η).prob h a) θ) θ :=
  hd.hasFDerivAt.log hz

end
end REINFORCEProMax.AutoregressiveTree
