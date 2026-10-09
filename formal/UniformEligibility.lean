import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic

/-! Exact-real semantics of the sample-standard-deviation uniformity test.
The denominator is n-1, matching the sample standard deviation used by the
implementation. The result bridges a tolerance test to exact homogeneous
binary groups; it does not analyze floating-point rounding. -/
namespace REINFORCEProMax.UniformEligibility
open scoped BigOperators

noncomputable def successCount {n : ℕ} (z : Fin n → Bool) : ℝ :=
  ∑ i, if z i then 1 else 0

def binaryReward {n : ℕ} (c d : ℝ) (z : Fin n → Bool) (i : Fin n) : ℝ :=
  if z i then c+d else c

noncomputable def sampleVariance {n : ℕ} (r : Fin n → ℝ) : ℝ :=
  (∑ i, (r i - (∑ j, r j)/(n : ℝ))^2)/((n : ℝ)-1)

theorem sum_two_values {n : ℕ} (z : Fin n → Bool) (a b : ℝ) :
    (∑ i, if z i then a else b) =
      successCount z * a + ((n : ℝ)-successCount z)*b := by
  have hh : ∀ i, (if z i then a else b) = b + (a-b)*(if z i then 1 else 0) := by
    intro i
    cases z i <;> simp
  simp_rw [hh]
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  change (n : ℝ)*b + (a-b)*successCount z = _
  ring

theorem sum_binary_reward {n : ℕ} (c d : ℝ) (z : Fin n → Bool) :
    (∑ i, binaryReward c d z i) = (n : ℝ)*c+d*successCount z := by
  change (∑ i, if z i then c+d else c) = _
  rw [sum_two_values]
  ring

theorem centered_binary_sum {n : ℕ} (c d a : ℝ) (z : Fin n → Bool) :
    (∑ i, (binaryReward c d z i-a)^2) =
      successCount z*(c+d-a)^2 + ((n : ℝ)-successCount z)*(c-a)^2 := by
  have hh : ∀ i, (binaryReward c d z i-a)^2 =
      if z i then (c+d-a)^2 else (c-a)^2 := by
    intro i
    cases hz : z i <;> simp [binaryReward, hz]
  simp_rw [hh]
  exact sum_two_values z _ _

theorem binary_sample_variance {n : ℕ} (hn : 2 ≤ n) (c d : ℝ) (z : Fin n → Bool) :
    sampleVariance (binaryReward c d z) =
      d^2 * successCount z * ((n : ℝ)-successCount z) / ((n : ℝ)*((n : ℝ)-1)) := by
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hn0 : (n : ℝ) ≠ 0 := by linarith
  have hn1 : (n : ℝ)-1 ≠ 0 := by linarith
  unfold sampleVariance
  rw [sum_binary_reward, centered_binary_sum]
  field_simp [hn0, hn1]
  ring

theorem mixed_count_bounds {n : ℕ} (z : Fin n → Bool)
    (ht : ∃ i, z i = true) (hf : ∃ i, z i = false) :
    1 ≤ successCount z ∧ 1 ≤ (n : ℝ)-successCount z := by
  obtain ⟨i,hi⟩ := ht
  obtain ⟨j,hj⟩ := hf
  have hnonneg : ∀ k : Fin n, 0 ≤ (if z k then (1 : ℝ) else 0) := by
    intro k
    cases z k <;> norm_num
  have hnonneg' : ∀ k : Fin n, 0 ≤ (if z k then (0 : ℝ) else 1) := by
    intro k
    cases z k <;> norm_num
  have hp := Finset.single_le_sum (fun k (_ : k ∈ Finset.univ) => hnonneg k)
    (Finset.mem_univ i)
  have hm := Finset.single_le_sum (fun k (_ : k ∈ Finset.univ) => hnonneg' k)
    (Finset.mem_univ j)
  constructor
  · simpa [hi, successCount] using hp
  · rw [sum_two_values] at hm
    simpa [hj] using hm

theorem mixed_binary_variance_lower_bound {n : ℕ} (hn : 2 ≤ n)
    (c d : ℝ) (z : Fin n → Bool) (ht : ∃ i, z i = true) (hf : ∃ i, z i = false) :
    d^2/(n : ℝ) ≤ sampleVariance (binaryReward c d z) := by
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hn0 : (n : ℝ) ≠ 0 := by linarith
  have hn1 : (n : ℝ)-1 ≠ 0 := by linarith
  have hc := mixed_count_bounds z ht hf
  have hb : (n : ℝ)-1 ≤ successCount z*((n : ℝ)-successCount z) := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hc.1) (sub_nonneg.mpr hc.2)]
  rw [binary_sample_variance hn]
  calc
    d^2/(n : ℝ) = d^2*((n : ℝ)-1)/((n : ℝ)*((n : ℝ)-1)) := by
      field_simp [hn0,hn1]
      ring
    _ ≤ _ := by
      apply div_le_div_of_nonneg_right _ (mul_nonneg (by linarith) (by linarith))
      nlinarith [mul_le_mul_of_nonneg_left hb (sq_nonneg d)]

/-- A separation condition makes the strict tolerance test select exactly
the homogeneous binary groups. Equality in n*tolerance² ≤ gap² is allowed. -/
theorem uniform_test_iff_homogeneous {n : ℕ} (hn : 2 ≤ n)
    (c d τ : ℝ) (hτ : 0 < τ) (hgap : (n : ℝ)*τ^2 ≤ d^2) (z : Fin n → Bool) :
    Real.sqrt (sampleVariance (binaryReward c d z)) < τ ↔
      (∀ i, z i = true) ∨ (∀ i, z i = false) := by
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  by_cases ht : ∀ i, z i = true
  · simp [binary_sample_variance hn, successCount, ht, hτ]
  by_cases hf : ∀ i, z i = false
  · simp [binary_sample_variance hn, successCount, hf, hτ]
  have hmt : ∃ i, z i = true := by
    obtain ⟨i,hi⟩ := not_forall.mp hf
    exact ⟨i, by cases hz : z i <;> simp_all⟩
  have hmf : ∃ i, z i = false := by
    obtain ⟨i,hi⟩ := not_forall.mp ht
    exact ⟨i, by cases hz : z i <;> simp_all⟩
  have hb := mixed_binary_variance_lower_bound hn c d z hmt hmf
  have hτv : τ^2 ≤ sampleVariance (binaryReward c d z) := by
    apply le_trans _ hb
    apply (le_div_iff₀ (by linarith : (0 : ℝ) < n)).mpr
    nlinarith [hgap]
  have hv0 := (sq_nonneg τ).trans hτv
  have hs := Real.sq_sqrt hv0
  have hsn := Real.sqrt_nonneg (sampleVariance (binaryReward c d z))
  have hnot : ¬ Real.sqrt (sampleVariance (binaryReward c d z)) < τ := by
    intro hh
    nlinarith
  simp [ht,hf,hnot]

theorem zero_one_implementation_tolerance {n : ℕ} (hn : 2 ≤ n)
    (hmax : n ≤ 10000000000000000) (z : Fin n → Bool) :
    Real.sqrt (sampleVariance (binaryReward 0 1 z)) < (1/100000000 : ℝ) ↔
      (∀ i, z i = true) ∨ (∀ i, z i = false) := by
  apply uniform_test_iff_homogeneous hn 0 1 _ (by norm_num) _ z
  have hh : (n : ℝ) ≤ 10000000000000000 := by exact_mod_cast hmax
  norm_num
  nlinarith

def oneSuccess (m : ℕ) : Fin (m+2) → Bool := fun i => decide (i = 0)

theorem one_success_variance (m : ℕ) (c d : ℝ) :
    sampleVariance (binaryReward c d (oneSuccess m)) = d^2/(m+2 : ℝ) := by
  rw [binary_sample_variance (by omega)]
  have hc : successCount (oneSuccess m) = 1 := by
    simp [successCount, oneSuccess]
  rw [hc]
  have hn0 : (m+2 : ℝ) ≠ 0 := by positivity
  have hn1 : (m+2 : ℝ)-1 ≠ 0 := by have hm := Nat.cast_nonneg (α := ℝ) m; linarith
  push_cast
  field_simp [hn0, hn1]
  ring

theorem one_success_mixed (m : ℕ) :
    ¬ ((∀ i, oneSuccess m i = true) ∨ (∀ i, oneSuccess m i = false)) := by
  rintro (h | h)
  · have hh := h (1 : Fin (m+2))
    norm_num [oneSuccess, Fin.ext_iff] at hh
  · have hh := h (0 : Fin (m+2))
    simp [oneSuccess] at hh

/-- If the separation condition fails, the minimum-variance mixed group
is classified as uniform. Hence the sufficient condition is sharp over
all binary groups of a given size. -/
theorem one_success_passes_small_gap (m : ℕ) (c d τ : ℝ)
    (hτ : 0 < τ) (hgap : d^2 < (m+2 : ℝ)*τ^2) :
    Real.sqrt (sampleVariance (binaryReward c d (oneSuccess m))) < τ := by
  rw [one_success_variance]
  have hv0 : 0 ≤ d^2/(m+2 : ℝ) := by positivity
  have hv : d^2/(m+2 : ℝ) < τ^2 :=
    (div_lt_iff₀ (by positivity : (0 : ℝ) < m+2)).mpr (by nlinarith [hgap])
  have hs := Real.sq_sqrt hv0
  have hsn := Real.sqrt_nonneg (d^2/(m+2 : ℝ))
  nlinarith

end REINFORCEProMax.UniformEligibility
