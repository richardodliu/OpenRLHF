import TreePinsker
import PolicyGradient

namespace REINFORCEProMax.AutoregressiveTree

open scoped BigOperators Topology

noncomputable section
variable {A E : Type*} [Fintype A] [DecidableEq A]
  [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem pathMass_differentiable (P : E → Policy A) (θ : E)
    (hd : ∀ h a, DifferentiableAt ℝ (fun η => (P η).prob h a) θ)
    (h y : List A) : DifferentiableAt ℝ (fun η => pathMass (P η) h y) θ := by
  induction y generalizing h with
  | nil => exact differentiableAt_const (1 : ℝ)
  | cons a y ih => exact (hd h a).mul (ih (h ++ [a]))

def responseMass (P : E → Policy A) (n : ℕ) (η : E) (y : Fin n → A) : ℝ :=
  pathMass (P η) [] (List.ofFn y)

theorem responseMass_normalized (P : E → Policy A) (n : ℕ) (η : E) :
    (∑ y : Fin n → A, responseMass P n η y) = 1 := by
  simpa [responseMass, pathSum_eq_sum_ofFn] using pathMass_normalized (P η) n

theorem value_eq_response_expectation (P : E → Policy A) (R : List A → ℝ)
    (n : ℕ) (η : E) :
    value (P η) R n [] = ∑ y : Fin n → A, responseMass P n η y * R (List.ofFn y) := by
  simpa [responseMass, pathSum_eq_sum_ofFn, pathMass] using (pathSum_mass_value (P η) R n []).symm

/-- The score of the full autoregressive response, zero on null responses. -/
def responseScore (P : E → Policy A) (n : ℕ) (θ : E) (y : Fin n → A) : E →L[ℝ] ℝ :=
  policyScore (responseMass P n) θ
    (fun z => fderiv ℝ (fun η => responseMass P n η z) θ) y

/-- RLOO differentiates the same recursively constructed terminal-reward
objective used in the performance-difference theorems. The only smoothness
assumption concerns conditional token probabilities, not a postulated score. -/
theorem tree_iid_rloo_policy_gradient {m : ℕ} (hm : m ≠ 0)
    (P : E → Policy A) (R : List A → ℝ) (n : ℕ) (θ : E)
    (hd : ∀ h a, DifferentiableAt ℝ (fun η => (P η).prob h a) θ) :
    HasFDerivAt (fun η => value (P η) R n [])
      (∑ s : Fin (m + 1) → (Fin n → A), iidMass (responseMass P n θ) s •
        iidRlooDifferential (fun y => R (List.ofFn y)) (responseScore P n θ) s) θ := by
  have hdpath : ∀ y : Fin n → A, HasFDerivAt (fun η => responseMass P n η y)
      (fderiv ℝ (fun η => responseMass P n η y) θ) θ := by
    intro y
    exact (pathMass_differentiable P θ hd [] (List.ofFn y)).hasFDerivAt
  have hp : ∀ᶠ η in 𝓝 θ, ∀ y : Fin n → A, 0 ≤ responseMass P n η y :=
    Filter.Eventually.of_forall (fun η y => pathMass_nonneg (P η) [] (List.ofFn y))
  have hn : ∀ᶠ η in 𝓝 θ, ∑ y : Fin n → A, responseMass P n η y = 1 :=
    Filter.Eventually.of_forall (responseMass_normalized P n)
  have hg := finite_iid_rloo_policy_gradient hm (responseMass P n)
    (fun y => R (List.ofFn y)) θ _ hdpath hp hn
  have hfun : (fun η => value (P η) R n []) =
      (fun η => ∑ y : Fin n → A, responseMass P n η y * R (List.ofFn y)) :=
    funext (value_eq_response_expectation P R n)
  rw [hfun]
  exact hg

end
end REINFORCEProMax.AutoregressiveTree
