import Mathlib.Tactic
import Mathlib.Analysis.Calculus.FDeriv.Add

/-!
Finite conditional-distribution bridge between sequence-reward and advantage
surrogates. `p` is the fixed rollout action law, `q` is the current action law,
and `u a` is the rollout continuation law after action `a`. No on-policy
restriction is imposed. Zero-probability actions are permitted, provided `q`
has no mass outside the support of `p`.
-/
namespace REINFORCEProMax.SurrogateBridge

open scoped BigOperators

variable {A Z C : Type*} [Fintype A] [Fintype Z] [Fintype C]

omit [Fintype A] in
lemma ratio_cancel (p q : A → ℝ) (hsupport : ∀ a, p a = 0 → q a = 0) (a : A) :
    p a * (q a / p a) = q a := by
  by_cases h : p a = 0
  · simp [h, hsupport a h]
  · field_simp

lemma continuation_baseline (u R : Z → ℝ) (b : ℝ) (hu : ∑ z, u z = 1) :
    (∑ z, u z * (R z - b)) = (∑ z, u z * R z) - b := by
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hu, one_mul]

/-- An independently sampled baseline is integrated using its own fixed
finite distribution; the resulting scalar baseline is its mean. -/
theorem finite_independent_baseline_average
    {B : Type*} [Fintype B] (p q : A → ℝ) (u R : A → Z → ℝ)
    (β b : B → ℝ) (hβ : ∑ k, β k = 1) :
    (∑ a, ∑ z, ∑ k, p a * u a z * β k * (R a z - b k) * (q a / p a)) =
      ∑ a, ∑ z, p a * u a z * (R a z - ∑ k, β k * b k) * (q a / p a) := by
  apply Finset.sum_congr rfl
  intro a _
  apply Finset.sum_congr rfl
  intro z _
  have hb : (∑ k, β k * (R a z - b k)) = R a z - ∑ k, β k * b k := by
    simp_rw [mul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hβ, one_mul]
  calc
    _ = (p a * u a z * (q a / p a)) * ∑ k, β k * (R a z - b k) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro k _
      ring
    _ = _ := by rw [hb]; ring

/-- Compute the trajectory-reward surrogate by explicitly integrating the
rollout continuation, including an arbitrary action-dependent gate. -/
theorem finite_gated_continuation
    (p q : A → ℝ) (u R : A → Z → ℝ) (b : ℝ) (M : A → ℝ)
    (hu : ∀ a, ∑ z, u a z = 1)
    (hsupport : ∀ a, p a = 0 → q a = 0) :
    (∑ a, ∑ z, p a * u a z * (R a z - b) * (q a / p a) * M a) =
      ∑ a, q a * M a * ((∑ z, u a z * R a z) - b) := by
  apply Finset.sum_congr rfl
  intro a _
  calc
    (∑ z, p a * u a z * (R a z - b) * (q a / p a) * M a) =
        ∑ z, (p a * (q a / p a)) * M a * (u a z * (R a z - b)) := by
          apply Finset.sum_congr rfl
          intro z _
          ring
    _ = q a * M a * (∑ z, u a z * (R a z - b)) := by
          rw [ratio_cancel p q hsupport a, Finset.mul_sum]
    _ = q a * M a * ((∑ z, u a z * R a z) - b) := by
          rw [continuation_baseline (u a) (R a) b (hu a)]

/-- An action-dependent gate introduces a current-policy acceptance-mass
term. It need not be constant during a policy update even for a frozen gate. -/
theorem finite_masked_surrogate_identity
    (p q : A → ℝ) (u R : A → Z → ℝ) (b V : ℝ) (M : A → ℝ)
    (hu : ∀ a, ∑ z, u a z = 1)
    (hsupport : ∀ a, p a = 0 → q a = 0) :
    (∑ a, ∑ z, p a * u a z * (R a z - b) * (q a / p a) * M a) =
      (∑ a, q a * M a * ((∑ z, u a z * R a z) - V)) +
        (V - b) * (∑ a, q a * M a) := by
  rw [finite_gated_continuation p q u R b M hu hsupport, Finset.mul_sum,
    ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro a _
  ring

/-- Without the gate, normalization of the current policy makes the baseline
offset constant. `V` can in particular be the rollout value `sum p Q`. -/
theorem finite_surrogate_baseline_offset
    (p q : A → ℝ) (u R : A → Z → ℝ) (b V : ℝ)
    (hq : ∑ a, q a = 1) (hu : ∀ a, ∑ z, u a z = 1)
    (hsupport : ∀ a, p a = 0 → q a = 0) :
    (∑ a, ∑ z, p a * u a z * (R a z - b) * (q a / p a)) =
      (∑ a, q a * ((∑ z, u a z * R a z) - V)) + (V - b) := by
  simpa [hq] using finite_masked_surrogate_identity p q u R b V (fun _ => 1) hu hsupport

/-- Subtracting the rollout value of the trajectory surrogate recovers the
standard advantage surrogate exactly, also for off-policy `q`. -/
theorem finite_surrogate_equivalence
    (p q : A → ℝ) (u R : A → Z → ℝ) (b : ℝ)
    (hp : ∑ a, p a = 1) (hq : ∑ a, q a = 1)
    (hu : ∀ a, ∑ z, u a z = 1)
    (hsupport : ∀ a, p a = 0 → q a = 0) :
    (∑ a, ∑ z, p a * u a z * (R a z - b) * (q a / p a)) -
      (∑ a, ∑ z, p a * u a z * (R a z - b)) =
        ∑ a, q a * ((∑ z, u a z * R a z) -
          ∑ a', p a' * (∑ z, u a' z * R a' z)) := by
  let V := ∑ a, p a * (∑ z, u a z * R a z)
  have hb : (∑ a, ∑ z, p a * u a z * (R a z - b)) = V - b := by
    calc
      _ = ∑ a, p a * ((∑ z, u a z * R a z) - b) := by
        apply Finset.sum_congr rfl
        intro a _
        simp_rw [mul_assoc]
        rw [← Finset.mul_sum, continuation_baseline (u a) (R a) b (hu a)]
      _ = V - b := by
        simp_rw [mul_sub]
        rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hp, one_mul]
  rw [finite_surrogate_baseline_offset p q u R b V hq hu hsupport, hb]
  ring

/-- The sequence-reward ratio-residual surrogate is exactly the per-action
advantage surrogate after explicitly summing the rollout continuation. -/
theorem finite_reward_ratio_surrogate_equivalence
    (p q : A → ℝ) (u R : A → Z → ℝ)
    (hp : ∑ a, p a = 1) (hq : ∑ a, q a = 1)
    (hu : ∀ a, ∑ z, u a z = 1)
    (hsupport : ∀ a, p a = 0 → q a = 0) :
    (∑ a, ∑ z, p a * u a z * R a z * (q a / p a - 1)) =
      ∑ a, q a * ((∑ z, u a z * R a z) -
        ∑ a', p a' * (∑ z, u a' z * R a' z)) := by
  rw [← finite_surrogate_equivalence p q u R 0 hp hq hu hsupport]
  simp only [sub_zero]
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro a _
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro z _
  ring

/-- The bridge is preserved by fixed context weights, including predictable
EOS-padding masks, and by finite sums over contexts and time steps. -/
theorem finite_context_surrogate_equivalence
    (p q : C → A → ℝ) (u R : C → A → Z → ℝ) (b w : C → ℝ)
    (hp : ∀ c, ∑ a, p c a = 1) (hq : ∀ c, ∑ a, q c a = 1)
    (hu : ∀ c a, ∑ z, u c a z = 1)
    (hsupport : ∀ c a, p c a = 0 → q c a = 0) :
    (∑ c, w c * (∑ a, ∑ z, p c a * u c a z *
      (R c a z - b c) * (q c a / p c a))) -
    (∑ c, w c * (∑ a, ∑ z, p c a * u c a z * (R c a z - b c))) =
    ∑ c, w c * (∑ a, q c a * ((∑ z, u c a z * R c a z) -
      ∑ a', p c a' * (∑ z, u c a' z * R c a' z))) := by
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro c _
  rw [← mul_sub, finite_surrogate_equivalence (p c) (q c) (u c) (R c)
    (b c) (hp c) (hq c) (hu c) (hsupport c)]

/-- Exact derivative transfer in an arbitrary real normed parameter space:
the rollout law, continuation law, baseline, and centering value are fixed,
while every current policy is normalized and supported by the rollout law. -/
theorem finite_surrogate_derivative_equivalence
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (p : A → ℝ) (q : E → A → ℝ) (u R : A → Z → ℝ) (b V : ℝ)
    (hq : ∀ θ, ∑ a, q θ a = 1) (hu : ∀ a, ∑ z, u a z = 1)
    (hsupport : ∀ θ a, p a = 0 → q θ a = 0)
    (θ : E) (D : E →L[ℝ] ℝ)
    (hD : HasFDerivAt (fun θ => ∑ a, q θ a * ((∑ z, u a z * R a z) - V)) D θ) :
    HasFDerivAt
      (fun θ => ∑ a, ∑ z, p a * u a z * (R a z - b) * (q θ a / p a)) D θ := by
  have hfun :
      (fun θ => ∑ a, ∑ z, p a * u a z * (R a z - b) * (q θ a / p a)) =
        (fun θ => (∑ a, q θ a * ((∑ z, u a z * R a z) - V)) + (V - b)) := by
    funext θ
    exact finite_surrogate_baseline_offset p (q θ) u R b V (hq θ) hu (hsupport θ)
  rw [hfun]
  exact hD.add_const (V - b)

end REINFORCEProMax.SurrogateBridge
