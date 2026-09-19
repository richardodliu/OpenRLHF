import RLOOMoments
import Mathlib.Analysis.Calculus.LocalExtr.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

namespace REINFORCEProMax

open scoped BigOperators Topology

noncomputable section
variable {A E : Type*} [Fintype A] [DecidableEq A]
  [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- A zero-probability outcome of a locally nonnegative differentiable
policy has zero derivative, so it contributes nothing to the score identity. -/
theorem policy_mass_derivative_zero (p : E → A → ℝ) (θ : E)
    (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a) (a : A) (hz : p θ a = 0) :
    D a = 0 := by
  have hmin : IsLocalMin (fun η => p η a) θ := by
    filter_upwards [hpos] with η hη
    simpa [hz] using hη a
  exact hmin.hasFDerivAt_eq_zero (hd a)

/-- The score differential is extended by zero at probability-zero outcomes. -/
def policyScore (p : E → A → ℝ) (θ : E) (D : A → E →L[ℝ] ℝ)
    (a : A) : E →L[ℝ] ℝ := (p θ a)⁻¹ • D a

theorem policyScore_is_log_derivative (p : E → A → ℝ) (θ : E)
    (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (a : A) (ha : p θ a ≠ 0) :
    HasFDerivAt (fun η => Real.log (p η a)) (policyScore p θ D a) θ :=
  (hd a).log ha

theorem policy_mass_smul_score (p : E → A → ℝ) (θ : E)
    (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a) (a : A) :
    p θ a • policyScore p θ D a = D a := by
  by_cases hz : p θ a = 0
  · simp [policyScore, hz, policy_mass_derivative_zero p θ D hd hpos a hz]
  · simp [policyScore, smul_smul, hz]

/-- The zero-mean score is derived from local normalization, not assumed. -/
theorem finite_policy_score_mean_zero (p : E → A → ℝ) (θ : E)
    (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a)
    (hnorm : ∀ᶠ η in 𝓝 θ, ∑ a, p η a = 1) :
    (∑ a, p θ a • policyScore p θ D a) = 0 := by
  simp_rw [policy_mass_smul_score p θ D hd hpos]
  have hsum : HasFDerivAt (fun η => ∑ a, p η a) (∑ a, D a) θ :=
    HasFDerivAt.sum (fun a _ => hd a)
  have hc : HasFDerivAt (fun _ : E => (1 : ℝ)) (∑ a, D a) θ :=
    hsum.congr_of_eventuallyEq (hnorm.mono (fun _ h => h.symm))
  exact hc.unique (hasFDerivAt_const (1 : ℝ) θ)

/-- On finite response support, the reward derivative equals the score
expectation in any real normed parameter space, including zero-mass outcomes. -/
theorem finite_policy_gradient (p : E → A → ℝ) (r : A → ℝ) (θ : E)
    (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a) :
    HasFDerivAt (fun η => ∑ a, p η a * r a)
      (∑ a, (p θ a * r a) • policyScore p θ D a) θ := by
  have hterm : ∀ a, (p θ a * r a) • policyScore p θ D a = r a • D a := by
    intro a
    rw [mul_comm, mul_smul, policy_mass_smul_score p θ D hd hpos]
  simp_rw [hterm]
  simpa only [mul_comm] using
    (HasFDerivAt.sum (fun a (_ : a ∈ (Finset.univ : Finset A)) => (hd a).const_mul (r a)))

/-- The actual iid leave-one-out estimator valued in parameter differentials. -/
def iidRlooDifferential {m : ℕ} (r : A → ℝ) (f : A → E →L[ℝ] ℝ)
    (s : Fin (m + 1) → A) : E →L[ℝ] ℝ :=
  (m + 1 : ℝ)⁻¹ • ∑ i : Fin (m + 1),
    (r (s i) - iidLooBaseline r (fun j => s (i.succAbove j))) • f (s i)

theorem iidRlooDifferential_apply {m : ℕ} (p r : A → ℝ) (f : A → E →L[ℝ] ℝ)
    (s : Fin (m + 1) → A) (v : E) :
    iidRlooDifferential r f s v = iidRlooEstimator p r (fun a => f a v) s := by
  simp [iidRlooDifferential, iidRlooEstimator, iidLooTerm,
    ContinuousLinearMap.sum_apply, div_eq_mul_inv, mul_comm]

theorem finite_iid_rloo_unbiased_differential {m : ℕ} (hm : m ≠ 0)
    (p r : A → ℝ) (f : A → E →L[ℝ] ℝ)
    (hp : ∑ a, p a = 1) (hscore : ∑ a, p a • f a = 0) :
    (∑ s : Fin (m + 1) → A, iidMass p s • iidRlooDifferential r f s) =
      ∑ a, (p a * r a) • f a := by
  ext v
  have hz : (∑ a, p a * f a v) = 0 := by
    simpa using congrArg (fun L : E →L[ℝ] ℝ => L v) hscore
  simpa only [ContinuousLinearMap.sum_apply, ContinuousLinearMap.smul_apply,
    smul_eq_mul, iidRlooDifferential_apply p r f] using
    finite_iid_rloo_unbiased hm p r (fun a => f a v) hp hz

/-- Full finite-support policy-gradient certificate: the expected concrete
RLOO estimator is the derivative of expected terminal reward. Zero-mean
scores and finite differentiation through expectation are both proved. -/
theorem finite_iid_rloo_policy_gradient {m : ℕ} (hm : m ≠ 0)
    (p : E → A → ℝ) (r : A → ℝ) (θ : E) (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a)
    (hnorm : ∀ᶠ η in 𝓝 θ, ∑ a, p η a = 1) :
    HasFDerivAt (fun η => ∑ a, p η a * r a)
      (∑ s : Fin (m + 1) → A,
        iidMass (p θ) s • iidRlooDifferential r (policyScore p θ D) s) θ := by
  have hp : (∑ a, p θ a) = 1 := @mem_of_mem_nhds E _ θ
    {η | ∑ a, p η a = 1} hnorm
  rw [finite_iid_rloo_unbiased_differential hm (p θ) r (policyScore p θ D)
    hp (finite_policy_score_mean_zero p θ D hd hpos hnorm)]
  exact finite_policy_gradient p r θ D hd hpos

end
end REINFORCEProMax
