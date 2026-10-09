import ProMax

namespace REINFORCEProMax

noncomputable section

/-- A coordinate of the raw group estimator, before normalization or clipping. -/
def rawLooEstimator {n : ℕ} (r f : Fin n → ℝ) : ℝ :=
  (∑ i, (r i - ((∑ j, r j) - r i) / ((n : ℝ) - 1)) * f i) / n

def rawMeanEstimator {n : ℕ} (r f : Fin n → ℝ) : ℝ :=
  (∑ i, (r i - (∑ j, r j) / (n : ℝ)) * f i) / n

/-- Deterministic proportionality, with no independence assumption on the scores. -/
theorem raw_mean_estimator_scaling {n : ℕ} (hn : 2 ≤ n) (r f : Fin n → ℝ) :
    rawMeanEstimator r f = ((n : ℝ) - 1) / n * rawLooEstimator r f := by
  have hn' : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hpoint : ∀ i, (r i - (∑ j, r j) / (n : ℝ)) * f i =
      (((n : ℝ) - 1) / n) *
        ((r i - ((∑ j, r j) - r i) / ((n : ℝ) - 1)) * f i) := by
    intro i
    have h := rloo_scaling n (r i) ((∑ j, r j) - r i) (by linarith) (by linarith)
    simp only [sub_add_cancel] at h
    rw [h]
    ring
  unfold rawMeanEstimator rawLooEstimator
  simp_rw [hpoint]
  rw [← Finset.mul_sum]
  ring

/-- Finite weighted covariance; for a probability mass this is the usual covariance. -/
def finiteCovariance {Ω : Type*} [Fintype Ω] (p X Y : Ω → ℝ) : ℝ :=
  ∑ w, p w * (X w - ∑ v, p v * X v) * (Y w - ∑ v, p v * Y v)

theorem finite_covariance_common_scaling {Ω : Type*} [Fintype Ω]
    (p X Y : Ω → ℝ) (c : ℝ) :
    finiteCovariance p (fun w => c * X w) (fun w => c * Y w) =
      c ^ 2 * finiteCovariance p X Y := by
  have hx : (∑ v, p v * (c * X v)) = c * ∑ v, p v * X v := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro v _
    ring
  have hy : (∑ v, p v * (c * Y v)) = c * ∑ v, p v * Y v := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro v _
    ring
  unfold finiteCovariance
  rw [hx, hy]
  conv_rhs => rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro w _
  ring

/-- Entrywise covariance of the complete group estimator, retaining sibling dependence. -/
theorem raw_mean_estimator_covariance {Ω : Type*} [Fintype Ω]
    {n : ℕ} (hn : 2 ≤ n) (p : Ω → ℝ) (r f g : Ω → Fin n → ℝ) :
    finiteCovariance p (fun w => rawMeanEstimator (r w) (f w))
        (fun w => rawMeanEstimator (r w) (g w)) =
      (((n : ℝ) - 1) / n) ^ 2 *
        finiteCovariance p (fun w => rawLooEstimator (r w) (f w))
          (fun w => rawLooEstimator (r w) (g w)) := by
  simp_rw [raw_mean_estimator_scaling hn]
  exact finite_covariance_common_scaling _ _ _ _

/-- Rescaling the learning rate compensates the raw estimator only for this plain update. -/
theorem raw_mean_plain_update {n : ℕ} (hn : 2 ≤ n) (r f : Fin n → ℝ)
    (θ η : ℝ) :
    θ + (η * (n : ℝ) / ((n : ℝ) - 1)) * rawMeanEstimator r f =
      θ + η * rawLooEstimator r f := by
  have hn' : (2 : ℝ) ≤ n := by exact_mod_cast hn
  rw [raw_mean_estimator_scaling hn]
  have hn0 : (n : ℝ) ≠ 0 := by linarith
  have hn1 : (n : ℝ) - 1 ≠ 0 := by linarith
  field_simp [hn0, hn1]
  <;> ring

end
end REINFORCEProMax
