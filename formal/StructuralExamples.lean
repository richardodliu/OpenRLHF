import PrefixMasking

/-! Exact real-arithmetic certificates for the synthetic masking table.
`hi` is the positive log threshold, and `lo ≤ 0` includes log(0.9).
These statements concern mathematical gates, not floating-point kernels. -/
namespace REINFORCEProMax.PrefixMasking
open scoped BigOperators

noncomputable def spike32 (k : Fin 32) (amplitude : ℝ) (i : Fin 32) : ℝ :=
  if i = k then amplitude else 0

theorem spike32_prefix (k j : Fin 32) (amplitude : ℝ) :
    prefixAverage (spike32 k amplitude) j =
      (if k ≤ j then amplitude else 0) / (j.val + 1 : ℝ) := by
  classical
  simp [prefixAverage, prefixSum, spike32]

theorem spike32_sequence (k : Fin 32) (amplitude : ℝ) :
    (∑ i, spike32 k amplitude i) / (32 : ℝ) = amplitude / 32 := by
  classical
  simp [spike32]

theorem early_table_prefix (lo hi : ℝ) (hlo : lo ≤ 0) (hhi : 0 < hi) (j : Fin 32) :
    prefixMask (spike32 0 (8 * hi)) lo hi j = 0 ↔ j.val < 7 := by
  rw [prefixMask, binaryMask_zero]
  change ¬intervalAccepted lo hi (prefixAverage _ j) ↔ _
  rw [spike32_prefix]
  simp only [Fin.zero_le, ite_true, intervalAccepted]
  have hd : (0 : ℝ) < j.val + 1 := by positivity
  have hn : lo ≤ 8 * hi / (j.val + 1) := le_trans hlo (by positivity)
  simp only [hn, true_and, not_le]
  rw [lt_div_iff₀ hd]
  constructor
  · intro h
    have : (j.val : ℝ) < 7 := by nlinarith
    exact_mod_cast this
  · intro h
    have : (j.val : ℝ) < 7 := by exact_mod_cast h
    nlinarith

theorem late_table_prefix (lo hi : ℝ) (hlo : lo ≤ 0) (hhi : 0 < hi) (j : Fin 32) :
    prefixMask (spike32 23 (32 * hi)) lo hi j = 0 ↔ 23 ≤ j.val ∧ j.val < 31 := by
  rw [prefixMask, binaryMask_zero]
  change ¬intervalAccepted lo hi (prefixAverage _ j) ↔ _
  rw [spike32_prefix]
  by_cases hk : (23 : Fin 32) ≤ j
  · simp only [hk, ite_true, intervalAccepted]
    have hd : (0 : ℝ) < j.val + 1 := by positivity
    have hn : lo ≤ 32 * hi / (j.val + 1) := le_trans hlo (by positivity)
    have hk' : 23 ≤ j.val := hk
    simp only [hn, true_and, not_le, hk']
    rw [lt_div_iff₀ hd]
    constructor
    · intro h
      have : (j.val : ℝ) < 31 := by nlinarith
      exact_mod_cast this
    · intro h
      have : (j.val : ℝ) < 31 := by exact_mod_cast h
      nlinarith
  · have hk' : ¬23 ≤ j.val := hk
    simp [hk, intervalAccepted, hlo, le_of_lt hhi, hk']

theorem spike_table_token (k : Fin 32) (lo hi amplitude : ℝ)
    (hlo : lo ≤ 0) (hhi : 0 < hi) (ha : hi < amplitude) (j : Fin 32) :
    tokenMask (spike32 k amplitude) lo hi j = 0 ↔ j = k := by
  by_cases h : j = k
  · subst j
    simp [tokenMask, binaryMask, tokenAccepted, intervalAccepted, spike32, not_le.mpr ha]
  · simp [tokenMask, binaryMask, tokenAccepted, intervalAccepted, spike32, h, hlo, le_of_lt hhi]

theorem spike_table_sequence (k : Fin 32) (lo hi amplitude : ℝ)
    (hlo : lo ≤ 0) (ha : 0 ≤ amplitude) (hb : amplitude ≤ 32 * hi) :
    rejectionCount (sequenceMask (spike32 k amplitude) lo hi) = 0 := by
  classical
  have h : sequenceAccepted (spike32 k amplitude) lo hi := by
    change intervalAccepted lo hi ((∑ i, spike32 k amplitude i) / (32 : ℝ))
    rw [spike32_sequence]
    constructor
    · exact le_trans hlo (div_nonneg ha (by norm_num))
    · apply (div_le_iff₀ (by norm_num : (0 : ℝ) < 32)).mpr
      nlinarith
  simp [rejectionCount, sequenceMask, binaryMask, h]

theorem early_table_counts (lo hi : ℝ) (hlo : lo ≤ 0) (hhi : 0 < hi) :
    rejectionCount (tokenMask (spike32 0 (8 * hi)) lo hi) = 1 ∧
    rejectionCount (sequenceMask (spike32 0 (8 * hi)) lo hi) = 0 ∧
    rejectionCount (prefixMask (spike32 0 (8 * hi)) lo hi) = 7 := by
  refine ⟨rejectionCount_singleton _ 0
    (spike_table_token 0 lo hi (8 * hi) hlo hhi (by linarith)),
    spike_table_sequence 0 lo hi (8 * hi) hlo (by positivity) (by linarith), ?_⟩
  unfold rejectionCount
  simp_rw [early_table_prefix lo hi hlo hhi]
  decide

theorem late_table_counts (lo hi : ℝ) (hlo : lo ≤ 0) (hhi : 0 < hi) :
    rejectionCount (tokenMask (spike32 23 (32 * hi)) lo hi) = 1 ∧
    rejectionCount (sequenceMask (spike32 23 (32 * hi)) lo hi) = 0 ∧
    rejectionCount (prefixMask (spike32 23 (32 * hi)) lo hi) = 8 := by
  refine ⟨rejectionCount_singleton _ 23
    (spike_table_token 23 lo hi (32 * hi) hlo hhi (by linarith)),
    spike_table_sequence 23 lo hi (32 * hi) hlo (by positivity) (le_refl _), ?_⟩
  unfold rejectionCount
  simp_rw [late_table_prefix lo hi hlo hhi]
  decide

theorem drift_table_counts (lo hi : ℝ) (hhi : 0 < hi) :
    rejectionCount (tokenMask (fun _ : Fin 32 => 2 * hi) lo hi) = 32 ∧
    rejectionCount (sequenceMask (fun _ : Fin 32 => 2 * hi) lo hi) = 32 ∧
    rejectionCount (prefixMask (fun _ : Fin 32 => 2 * hi) lo hi) = 32 := by
  have h := sustained_violation_complete (fun _ : Fin 32 => 2 * hi) lo hi
    (Or.inl (by intro i; dsimp; linarith))
  exact ⟨h.token_count, h.sequence_count, h.prefix_count⟩

end REINFORCEProMax.PrefixMasking
