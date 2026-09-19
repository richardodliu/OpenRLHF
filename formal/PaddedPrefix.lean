import PrefixMasking
import Mathlib.Data.Finset.Sort

/-!
Exact-real semantics of the cumulative binary action-mask implementation.
An arbitrary finite subset of padded positions is enumerated in increasing
order. At every active position the masked cumulative sum / clamped count
equals the active-index prefix average. The subset need not be contiguous.
This is an index and arithmetic bridge, not a floating-point verification.
-/
namespace REINFORCEProMax.PaddedPrefix

open scoped BigOperators
open PrefixMasking

noncomputable def cumulativeSum {N : ℕ} (S : Finset (Fin N)) (z : Fin N → ℝ)
    (j : Fin N) : ℝ := ∑ i ∈ S.filter (fun i => i ≤ j), z i

def cumulativeCount {N : ℕ} (S : Finset (Fin N)) (j : Fin N) : ℕ :=
  (S.filter (fun i => i ≤ j)).card

noncomputable def implementationAverage {N : ℕ} (S : Finset (Fin N))
    (z : Fin N → ℝ) (j : Fin N) : ℝ :=
  cumulativeSum S z j / max (cumulativeCount S j : ℝ) 1

noncomputable def implementationMask {N : ℕ} (S : Finset (Fin N))
    (z : Fin N → ℝ) (lo hi : ℝ) (j : Fin N) : ℕ :=
  binaryMask (intervalAccepted (Real.exp lo) (Real.exp hi) (Real.exp (implementationAverage S z j)))

theorem cumulativeSum_eq_masked_cumsum {N : ℕ} (S : Finset (Fin N))
    (z : Fin N → ℝ) (j : Fin N) :
    cumulativeSum S z j = ∑ i ∈ Finset.Iic j, z i * (if i ∈ S then 1 else 0) := by
  classical
  simp only [mul_ite, mul_one, mul_zero]
  rw [← Finset.sum_filter]
  unfold cumulativeSum
  congr 1
  ext i
  simp [and_comm]

theorem cumulativeCount_eq_masked_cumsum {N : ℕ} (S : Finset (Fin N)) (j : Fin N) :
    cumulativeCount S j = ∑ i ∈ Finset.Iic j, if i ∈ S then 1 else 0 := by
  classical
  rw [← Finset.sum_filter]
  simp only [Finset.sum_const, smul_eq_mul, mul_one]
  unfold cumulativeCount
  congr 1
  ext i
  simp [and_comm]

theorem active_prefix_image {N n : ℕ} (S : Finset (Fin N)) (hcard : S.card = n)
    (j : Fin n) :
    (Finset.Iic j).image (S.orderEmbOfFin hcard) =
      S.filter (fun i => i ≤ S.orderEmbOfFin hcard j) := by
  classical
  ext a
  simp only [Finset.mem_image, Finset.mem_Iic, Finset.mem_filter]
  constructor
  · rintro ⟨i, hij, rfl⟩
    exact ⟨S.orderEmbOfFin_mem hcard i, (S.orderEmbOfFin hcard).monotone hij⟩
  · rintro ⟨ha, haj⟩
    have hr : a ∈ Set.range (S.orderEmbOfFin hcard) := by
      rw [S.range_orderEmbOfFin hcard]
      exact ha
    rcases hr with ⟨i, rfl⟩
    exact ⟨i, (S.orderEmbOfFin hcard).le_iff_le.mp haj, rfl⟩

theorem cumulativeSum_reindex {N n : ℕ} (S : Finset (Fin N)) (hcard : S.card = n)
    (z : Fin N → ℝ) (j : Fin n) :
    cumulativeSum S z (S.orderEmbOfFin hcard j) =
      prefixSum (fun i => z (S.orderEmbOfFin hcard i)) j := by
  classical
  rw [cumulativeSum, ← active_prefix_image S hcard j, Finset.sum_image]
  · exact (prefixSum_Iic _ _).symm
  · intro i _ k _ hik
    exact (S.orderEmbOfFin hcard).injective hik

theorem cumulativeCount_reindex {N n : ℕ} (S : Finset (Fin N)) (hcard : S.card = n)
    (j : Fin n) : cumulativeCount S (S.orderEmbOfFin hcard j) = j.val + 1 := by
  rw [cumulativeCount, ← active_prefix_image S hcard j,
    Finset.card_image_of_injective _ (S.orderEmbOfFin hcard).injective, Fin.card_Iic]

/-- The clamp is exactly inactive at every active position, including the first. -/
theorem implementationAverage_reindex {N n : ℕ} (S : Finset (Fin N)) (hcard : S.card = n)
    (z : Fin N → ℝ) (j : Fin n) :
    implementationAverage S z (S.orderEmbOfFin hcard j) =
      prefixAverage (fun i => z (S.orderEmbOfFin hcard i)) j := by
  rw [implementationAverage, cumulativeSum_reindex S hcard z j, cumulativeCount_reindex S hcard j]
  have hd : (1 : ℝ) ≤ ((j.val + 1 : ℕ) : ℝ) := by
    exact_mod_cast Nat.succ_le_succ (Nat.zero_le j.val)
  rw [max_eq_left hd]
  rfl

/-- The implemented geometric-mean gate is the same binary mask used by the main theorem. -/
theorem implementationMask_reindex {N n : ℕ} (S : Finset (Fin N)) (hcard : S.card = n)
    (z : Fin N → ℝ) (lo hi : ℝ) (j : Fin n) :
    implementationMask S z lo hi (S.orderEmbOfFin hcard j) =
      prefixMask (fun i => z (S.orderEmbOfFin hcard i)) lo hi j := by
  rw [implementationMask, implementationAverage_reindex S hcard z j, exp_interval_mask]
  rfl

theorem active_filter_image {N n : ℕ} (S : Finset (Fin N)) (hcard : S.card = n)
    (P : Fin N → Prop) [DecidablePred P] :
    (Finset.univ.filter (fun i => P (S.orderEmbOfFin hcard i))).image (S.orderEmbOfFin hcard) =
      S.filter P := by
  classical
  ext a
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨i, hi, rfl⟩
    exact ⟨S.orderEmbOfFin_mem hcard i, hi⟩
  · rintro ⟨ha, hP⟩
    have hr : a ∈ Set.range (S.orderEmbOfFin hcard) := by
      rw [S.range_orderEmbOfFin hcard]
      exact ha
    rcases hr with ⟨i, rfl⟩
    exact ⟨i, hP, rfl⟩

/-- Padding contributes no rejection count; the active count is preserved exactly. -/
theorem implementation_rejection_count {N n : ℕ} (S : Finset (Fin N)) (hcard : S.card = n)
    (z : Fin N → ℝ) (lo hi : ℝ) :
    (S.filter (fun j => implementationMask S z lo hi j = 0)).card =
      rejectionCount (prefixMask (fun i => z (S.orderEmbOfFin hcard i)) lo hi) := by
  classical
  rw [← active_filter_image S hcard (fun j => implementationMask S z lo hi j = 0),
    Finset.card_image_of_injective _ (S.orderEmbOfFin hcard).injective]
  simp only [implementationMask_reindex]
  rfl

end REINFORCEProMax.PaddedPrefix
