import PromptGradient
import Mathlib.Analysis.Calculus.ParametricIntegral

/-!
An arbitrary-prompt-measure extension of the finite RLOO policy-gradient
certificate.  The finite-response calculation is unchanged; the only new
ingredient is an explicit dominated differentiation-under-the-integral
condition for the outer prompt expectation.
-/
namespace REINFORCEProMax

open scoped BigOperators Topology
open TopologicalSpace MeasureTheory Filter Metric
open PromptGradient

noncomputable section

variable {X A E : Type*} [MeasurableSpace X] [Fintype A] [DecidableEq A]
  [NormedAddCommGroup E] [NormedSpace ℝ E]

def promptRlooDifferential {m : ℕ} (p : X → E → A → ℝ) (R : X → A → ℝ)
    (θ : E) (D : X → A → E →L[ℝ] ℝ) (x : X) : E →L[ℝ] ℝ :=
  ∑ y : Fin (m + 1) → A,
    iidMass (p x θ) y • iidRlooDifferential (R x) (policyScore (p x) θ (D x)) y

/-- Pointwise finite-response RLOO unbiasedness holds for almost every prompt
when the policy regularity assumptions hold almost everywhere. -/
theorem ae_prompt_rloo_policy_gradient {m : ℕ} (hm : m ≠ 0)
    (μ : Measure X) (p : X → E → A → ℝ) (R : X → A → ℝ) (θ : E)
    (D : X → A → E →L[ℝ] ℝ)
    (hD : ∀ᵐ x ∂μ, ∀ a, HasFDerivAt (fun η => p x η a) (D x a) θ)
    (hpos : ∀ᵐ x ∂μ, ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p x η a)
    (hnorm : ∀ᵐ x ∂μ, ∀ᶠ η in 𝓝 θ, ∑ a, p x η a = 1) :
    ∀ᵐ x ∂μ, HasFDerivAt (fun η => promptReward p R x η)
      (promptRlooDifferential (m := m) p R θ D x) θ := by
  filter_upwards [hD, hpos, hnorm] with x hxD hxpos hxnorm
  exact finite_iid_rloo_policy_gradient hm (p x) (R x) θ (D x)
    (fun a => hxD a) hxpos hxnorm

/-- General-prompt RLOO gradient theorem.  The outer prompt expectation is
represented by a Bochner integral.  The displayed domination assumptions are
exactly those needed to exchange the parameter derivative and this integral;
they are not hidden modelling assumptions. -/
theorem measure_prompt_rloo_policy_gradient {m : ℕ} (hm : m ≠ 0)
    (μ : Measure X) (p : X → E → A → ℝ) (R : X → A → ℝ) (θ : E)
    (D : X → A → E →L[ℝ] ℝ)
    (ε : ℝ) (bound : X → ℝ)
    (hε : 0 < ε)
    (hD : ∀ᵐ x ∂μ, ∀ a, HasFDerivAt (fun η => p x η a) (D x a) θ)
    (hpos : ∀ᵐ x ∂μ, ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p x η a)
    (hnorm : ∀ᵐ x ∂μ, ∀ᶠ η in 𝓝 θ, ∑ a, p x η a = 1)
    (h_meas : ∀ᶠ η in 𝓝 θ,
      AEStronglyMeasurable (fun x => promptReward p R x η) μ)
    (h_int : Integrable (fun x => promptReward p R x θ) μ)
    (h_deriv_meas : AEStronglyMeasurable (promptRlooDifferential (m := m) p R θ D) μ)
    (h_lip : ∀ᵐ x ∂μ,
      LipschitzOnWith (Real.nnabs (bound x))
        (fun η => promptReward p R x η) (ball θ ε))
    (h_bound_integrable : Integrable bound μ) :
    HasFDerivAt (fun η => ∫ x, promptReward p R x η ∂μ)
      (∫ x, promptRlooDifferential (m := m) p R θ D x ∂μ) θ := by
  exact (hasFDerivAt_integral_of_dominated_loc_of_lip hε h_meas h_int h_deriv_meas
    h_lip h_bound_integrable
    (ae_prompt_rloo_policy_gradient hm μ p R θ D hD hpos hnorm)).2

end
end REINFORCEProMax
