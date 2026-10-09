import PrefixMasking
import TreeTV

namespace REINFORCEProMax
open PrefixMasking
noncomputable section

/-- The token mask only rejects the specified outlier; no background-noise restriction. -/
theorem token_mask_single_outlier {n : ℕ} (z : Fin n → ℝ) (lo hi : ℝ) (k : Fin n)
    (hk : ¬tokenAccepted z lo hi k)
    (hrest : ∀ i, i ≠ k → tokenAccepted z lo hi i) :
    (∀ i, tokenMask z lo hi i = 0 ↔ i = k) ∧
    rejectionCount (tokenMask z lo hi) = 1 := by
  have hp : ∀ i, tokenMask z lo hi i = 0 ↔ i = k := by
    intro i
    by_cases he : i = k
    · subst i; simp [tokenMask, hk]
    · have ha := hrest i he
      simp [tokenMask, ha, he]
  exact ⟨hp, rejectionCount_singleton _ k hp⟩

end
end REINFORCEProMax

namespace REINFORCEProMax.AutoregressiveTree
variable {A : Type*} [Fintype A]

/-- A changed sampled first-token probability makes every later full-prefix law differ. -/
theorem first_token_ratio_context_shift (p q : Policy A) (a : A)
    (hp : 0 < p.prob [] a) (hr : tokenRatio p q [] a ≠ 1)
    (t : ℕ) (ht : 1 ≤ t) :
    0 < acceptedPrefixTV p q (fun _ => 1) t := by
  have hne : q.prob [] a ≠ p.prob [] a := by
    intro he
    apply hr
    simp [tokenRatio, he, ne_of_gt hp]
  have hterm : 0 < |q.prob [] a - p.prob [] a| := abs_pos.mpr (sub_ne_zero.mpr hne)
  have hb := Finset.single_le_sum
    (fun (b : A) (_ : b ∈ Finset.univ) => abs_nonneg (q.prob [] b - p.prob [] b))
    (Finset.mem_univ a)
  have htv : 0 < localTV p q [] :=
    mul_pos (by norm_num) (hterm.trans_le hb)
  exact first_disagreement_propagates p q 0 (by intro h hh; simp at hh)
    [] rfl (by simp [pathMass]) htv t ht

end REINFORCEProMax.AutoregressiveTree
