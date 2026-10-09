import PrefixConcentration
import PromptMixture

/-! Finite prompt aggregation for the concentration certificate. Zero-mass
prompts require no conditional concentration bound. No independence of tokens
or automatic control by the cached training gate is assumed. -/
namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators
variable {X A : Type*} [Fintype X] [Fintype A]

theorem finite_probability_average_le (μ f : X → ℝ) (C : ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (hf : ∀ x, 0 < μ x → f x ≤ C) :
    ∑ x, μ x * f x ≤ C := by
  calc
    _ ≤ ∑ x, μ x * C := Finset.sum_le_sum fun x _ => by
      by_cases hz : μ x = 0
      · simp [hz]
      · exact mul_le_mul_of_nonneg_left
          (hf x (lt_of_le_of_ne (hμ0 x) (Ne.symm hz))) (hμ0 x)
    _ = C := by rw [← Finset.sum_mul, hμ, one_mul]

/-- The paper's all-lambda condition implies the single MGF input used below. -/
theorem centered_subgaussian_at_two (p q : Policy A) (σ : ℝ) (h : List A)
    (hmgf : ∀ lam : ℝ, ∑ a, p.prob h a *
      Real.exp (lam * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (lam ^ 2 * σ ^ 2 / 2)) :
    ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2) := by
  convert hmgf 2 using 1 <;> congr 1 <;> ring

theorem family_surrogate_error_of_mgf (μ : X → ℝ) (p q : X → Policy A)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (σ : ℝ) (N : ℕ)
    (hs : ∀ x h a, (p x).prob h a = 0 ↔ (q x).prob h a = 0)
    (hmgf : ∀ x, 0 < μ x → ∀ h, h.length < N → 0 < pathMass (p x) [] h →
      ∑ a, (p x).prob h a * Real.exp
        (2 * (Real.log (tokenRatio (p x) (q x) h a) - logRatioMean (p x) (q x) h)) ≤
          Real.exp (2 * σ ^ 2))
    (R : X → List A → ℝ) (B : ℝ) (hB : 0 ≤ B)
    (hR : ∀ x y, |R x y| ≤ B) :
    |familyValue μ q R N - familyValue μ p R N - familySurrogate μ p q R N| ≤
      2 * B * ∑ k ∈ Finset.range N, Real.sqrt (Real.exp (2 * σ ^ 2 * (k : ℝ)) - 1) := by
  unfold familyValue familySurrogate
  rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
  have hid : ∀ x, μ x * value (q x) (R x) N [] - μ x * value (p x) (R x) N [] -
      μ x * surrogate (p x) (q x) (R x) N [] =
      μ x * (value (q x) (R x) N [] - value (p x) (R x) N [] -
        surrogate (p x) (q x) (R x) N []) := by intro x; ring
  simp_rw [hid]
  calc
    _ ≤ ∑ x, |μ x * (value (q x) (R x) N [] - value (p x) (R x) N [] -
        surrogate (p x) (q x) (R x) N [])| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ x, μ x * |value (q x) (R x) N [] - value (p x) (R x) N [] -
        surrogate (p x) (q x) (R x) N []| := by
      apply Finset.sum_congr rfl
      intro x _
      rw [abs_mul, abs_of_nonneg (hμ0 x)]
    _ ≤ _ := finite_probability_average_le μ _ _ hμ0 hμ fun x hx =>
      surrogate_error_of_mgf (p x) (q x) σ (hs x) N (hmgf x hx) (R x) B hB (hR x)

end REINFORCEProMax.AutoregressiveTree
