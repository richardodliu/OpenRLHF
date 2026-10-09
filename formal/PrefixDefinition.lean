import PaddedPrefix

namespace REINFORCEProMax
open PrefixMasking
noncomputable section

/-- Positive ratios make the log-average exactly the geometric mean. -/
theorem prefix_geometric_mean {n : ℕ} (r : Fin n → ℝ) (j : Fin n)
    (hr : ∀ i, i ≤ j → 0 < r i) :
    Real.exp (prefixAverage (fun i => Real.log (r i)) j) =
      (∏ i ∈ Finset.Iic j, r i) ^ (1 / ((j.val + 1 : ℕ) : ℝ)) := by
  rw [Real.rpow_def_of_pos (Finset.prod_pos (fun i hi => hr i (Finset.mem_Iic.mp hi))),
    Real.log_prod _ _ (fun i hi => ne_of_gt (hr i (Finset.mem_Iic.mp hi)))]
  rw [prefixAverage, prefixSum_Iic]
  congr 1
  ring

def positiveReentry : Fin 3 → ℝ := ![3, 1 / 2, 19 / 10]
def prefixRejectsMore : Fin 3 → ℝ := ![3, 3 / 2, 1 / 10]
def tokenRejectsMore : Fin 3 → ℝ := ![1 / 10, 3, 3]

theorem same_sign_examples_positive :
    (∀ i, 0 < positiveReentry i) ∧ (∀ i, 0 < prefixRejectsMore i) ∧
      (∀ i, 0 < tokenRejectsMore i) := by
  refine ⟨?_, ?_, ?_⟩ <;> intro i <;> fin_cases i <;>
    norm_num [positiveReentry, prefixRejectsMore, tokenRejectsMore]

/-- Positive increments need not give a monotone average, and rejection is not absorbing. -/
theorem positive_reentry_pattern :
    prefixAverage positiveReentry 0 = 3 ∧
    prefixAverage positiveReentry 1 = 7 / 4 ∧
    prefixAverage positiveReentry 2 = 9 / 5 ∧
    prefixMask positiveReentry (-2) 2 0 = 0 ∧
    prefixMask positiveReentry (-2) 2 1 = 1 ∧
    prefixMask positiveReentry (-2) 2 2 = 1 := by
  norm_num [prefixAverage, prefixSum, Finset.sum_filter, Fin.sum_univ_succ, Fin.le_iff_val_le_val,
    positiveReentry, prefixMask, binaryMask, prefixAccepted, intervalAccepted]

theorem same_sign_mask_patterns :
    (∀ i, prefixMask prefixRejectsMore (-2) 2 i = (![0, 0, 1] : Fin 3 → ℕ) i) ∧
    (∀ i, tokenMask prefixRejectsMore (-2) 2 i = (![0, 1, 1] : Fin 3 → ℕ) i) ∧
    (∀ i, prefixMask tokenRejectsMore (-2) 2 i = (![1, 1, 0] : Fin 3 → ℕ) i) ∧
    (∀ i, tokenMask tokenRejectsMore (-2) 2 i = (![1, 0, 0] : Fin 3 → ℕ) i) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> intro i <;> fin_cases i <;>
    norm_num [prefixAverage, prefixSum, Finset.sum_filter, Fin.sum_univ_succ, Fin.le_iff_val_le_val,
      prefixRejectsMore, tokenRejectsMore, prefixMask, tokenMask, binaryMask,
      prefixAccepted, tokenAccepted, intervalAccepted]

theorem same_sign_no_rejection_count_order :
    rejectionCount (prefixMask prefixRejectsMore (-2) 2) = 2 ∧
    rejectionCount (tokenMask prefixRejectsMore (-2) 2) = 1 ∧
    rejectionCount (prefixMask tokenRejectsMore (-2) 2) = 1 ∧
    rejectionCount (tokenMask tokenRejectsMore (-2) 2) = 2 := by
  rcases same_sign_mask_patterns with ⟨h1,h2,h3,h4⟩
  simp_rw [rejectionCount, h1, h2, h3, h4]
  decide

end
end REINFORCEProMax
