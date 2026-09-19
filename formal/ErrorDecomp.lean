import Mathlib.Tactic
namespace REINFORCEProMax
theorem finite_tv_expectation_bound
    {n : ℕ} (p q g : Fin n → ℝ) (G : ℝ)
    (hbound : ∀ i, |g i| ≤ G) :
    |(∑ i : Fin n, q i * g i) - ∑ i : Fin n, p i * g i| ≤
      2 * G * ((1 / 2 : ℝ) * ∑ i : Fin n, |q i - p i|) := by
  have hrewrite :
      (∑ i : Fin n, q i * g i) - ∑ i : Fin n, p i * g i =
        ∑ i : Fin n, (q i - p i) * g i := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i hi
    ring
  rw [hrewrite]
  calc
    |∑ i : Fin n, (q i - p i) * g i| ≤
        ∑ i : Fin n, |(q i - p i) * g i| := by
      exact Finset.abs_sum_le_sum_abs (s := Finset.univ) (fun i : Fin n => (q i - p i) * g i)
    _ = ∑ i : Fin n, |q i - p i| * |g i| := by
      apply Finset.sum_congr rfl
      intro i hi
      rw [abs_mul]
    _ ≤ ∑ i : Fin n, |q i - p i| * G := by
      apply Finset.sum_le_sum
      intro i hi
      exact mul_le_mul_of_nonneg_left (hbound i) (abs_nonneg _)
    _ = G * ∑ i : Fin n, |q i - p i| := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i hi
      ring
    _ = 2 * G * ((1 / 2 : ℝ) * ∑ i : Fin n, |q i - p i|) := by ring

/- A finite-horizon composition of the one-step TV estimate.  The hypothesis
   `herror` is the finite-state performance-difference decomposition supplied
   by the probabilistic/autoregressive part of the model.  This theorem checks
   the remaining deterministic step: summing the context-shift terms over a
   finite horizon. -/
theorem finite_error_decomposition_bound
    {T n : ℕ} (p q g : Fin T → Fin n → ℝ) (G : Fin T → ℝ)
    (E : ℝ)
    (hbound : ∀ t i, |g t i| ≤ G t)
    (herror : E = ∑ t : Fin T,
      ((∑ i : Fin n, q t i * g t i) - ∑ i : Fin n, p t i * g t i)) :
    |E| ≤ ∑ t : Fin T,
      2 * G t * ((1 / 2 : ℝ) * ∑ i : Fin n, |q t i - p t i|) := by
  rw [herror]
  calc
    |∑ t : Fin T,
        ((∑ i : Fin n, q t i * g t i) - ∑ i : Fin n, p t i * g t i)| ≤
        ∑ t : Fin T,
          |(∑ i : Fin n, q t i * g t i) - ∑ i : Fin n, p t i * g t i| := by
      exact Finset.abs_sum_le_sum_abs (s := Finset.univ)
        (fun t : Fin T =>
          (∑ i : Fin n, q t i * g t i) - ∑ i : Fin n, p t i * g t i)
    _ ≤ ∑ t : Fin T,
          2 * G t * ((1 / 2 : ℝ) * ∑ i : Fin n, |q t i - p t i|) := by
      apply Finset.sum_le_sum
      intro t ht
      exact finite_tv_expectation_bound (p t) (q t) (g t) (G t)
        (by intro i; exact hbound t i)
end REINFORCEProMax
