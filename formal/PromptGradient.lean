import PolicyGradient

/-!
The conditional RLOO gradient is the gradient of the prompt-specific reward.
A fixed finite prompt law turns it into the gradient of the global reward;
conditioning on a single prompt does not permit that last identification.
-/
namespace REINFORCEProMax.PromptGradient

open scoped BigOperators Topology
noncomputable section
variable {X A E : Type*} [Fintype X] [Fintype A] [DecidableEq A]
  [NormedAddCommGroup E] [NormedSpace ℝ E]

def promptReward (p : X → E → A → ℝ) (R : X → A → ℝ) (x : X) (θ : E) : ℝ :=
  ∑ a, p x θ a * R x a

def globalReward (μ : X → ℝ) (p : X → E → A → ℝ) (R : X → A → ℝ) (θ : E) : ℝ :=
  ∑ x, μ x * promptReward p R x θ

omit [DecidableEq A] [NormedAddCommGroup E] [NormedSpace ℝ E] in
/-- This is a joint sampling law: first draw a prompt, then its iid responses. -/
theorem joint_prompt_response_mass_normalized {m : ℕ} (μ : X → ℝ) (p : X → A → ℝ)
    (hμ : ∑ x, μ x = 1) (hp : ∀ x, ∑ a, p x a = 1) :
    (∑ x, ∑ y : Fin (m + 1) → A, μ x * iidMass (p x) y) = 1 := by
  classical
  simp_rw [← Finset.mul_sum, iid_mass_normalized _ (hp _) (m + 1), mul_one]
  exact hμ

omit [Fintype X] [Fintype A] [DecidableEq A] [NormedAddCommGroup E] [NormedSpace ℝ E] in
theorem joint_prompt_response_mass_nonneg {m : ℕ} (μ : X → ℝ) (p : X → A → ℝ)
    (hμ : ∀ x, 0 ≤ μ x) (hp : ∀ x a, 0 ≤ p x a)
    (x : X) (y : Fin (m + 1) → A) : 0 ≤ μ x * iidMass (p x) y := by
  exact mul_nonneg (hμ x) (Finset.prod_nonneg (fun i _ => hp x (y i)))

/-- Fixed prompt weights may be pulled through the finite derivative.
No prompt-independent conditional reward gradient is assumed. -/
theorem finite_prompt_rloo_policy_gradient {m : ℕ} (hm : m ≠ 0)
    (μ : X → ℝ) (p : X → E → A → ℝ) (R : X → A → ℝ) (θ : E)
    (D : X → A → E →L[ℝ] ℝ)
    (hd : ∀ x a, HasFDerivAt (fun η => p x η a) (D x a) θ)
    (hpos : ∀ x, ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p x η a)
    (hnorm : ∀ x, ∀ᶠ η in 𝓝 θ, ∑ a, p x η a = 1) :
    HasFDerivAt (globalReward μ p R)
      (∑ x, ∑ y : Fin (m + 1) → A,
        (μ x * iidMass (p x θ) y) •
          iidRlooDifferential (R x) (policyScore (p x) θ (D x)) y) θ := by
  have hx := fun x => (finite_iid_rloo_policy_gradient hm (p x) (R x) θ (D x)
    (hd x) (hpos x) (hnorm x)).const_mul (μ x)
  simpa [globalReward, promptReward, Finset.smul_sum, smul_smul] using
    HasFDerivAt.sum (fun x (_ : x ∈ (Finset.univ : Finset X)) => hx x)

/-- Two valid Bernoulli response policies on the open unit interval.
For equally likely prompts their average reward is constant, although
the reward for the first prompt has derivative one. -/
def twoPromptSuccess (x : Bool) (θ : ℝ) : ℝ := if x then θ else 1 - θ

def twoPromptPolicy (x : Bool) (θ : ℝ) (success : Bool) : ℝ :=
  if success then twoPromptSuccess x θ else 1 - twoPromptSuccess x θ

theorem two_prompt_reward (x : Bool) (θ : ℝ) :
    (∑ a : Bool, twoPromptPolicy x θ a * (if a then (1 : ℝ) else 0)) =
      twoPromptSuccess x θ := by
  simp [twoPromptPolicy, Fintype.sum_bool]

theorem two_prompt_policy_valid (θ : ℝ) (h0 : 0 < θ) (h1 : θ < 1) (x : Bool) :
    (∀ a, 0 < twoPromptPolicy x θ a) ∧ ∑ a, twoPromptPolicy x θ a = 1 := by
  constructor
  · intro a
    cases x <;> cases a <;> simp [twoPromptPolicy, twoPromptSuccess] <;> linarith
  · cases x <;> simp [twoPromptPolicy, twoPromptSuccess, Fintype.sum_bool]

theorem fixed_prompt_gradient_ne_global (θ : ℝ) :
    HasDerivAt (twoPromptSuccess true) 1 θ ∧
    HasDerivAt (fun η => (twoPromptSuccess true η + twoPromptSuccess false η) / 2) 0 θ ∧
    (1 : ℝ) ≠ 0 := by
  constructor
  · simpa [twoPromptSuccess] using hasDerivAt_id θ
  constructor
  · convert hasDerivAt_const θ (1 / 2 : ℝ) using 1
    simp [twoPromptSuccess]
  · norm_num

end
end REINFORCEProMax.PromptGradient
