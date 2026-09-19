import TreeKL
import Pinsker

namespace REINFORCEProMax.AutoregressiveTree

variable {A : Type*} [Fintype A]

/-- The recursive tree sum is an ordinary finite sum over length-n strings. -/
theorem pathSum_eq_sum_ofFn (f : List A → ℝ) (n : ℕ) (h : List A) :
    pathSum f n h = ∑ y : Fin n → A, f (h ++ List.ofFn y) := by
  induction n generalizing h with
  | zero => simp [pathSum]
  | succ n ih =>
      rw [pathSum]
      simp_rw [ih]
      rw [← Fintype.sum_prod_type']
      apply Fintype.sum_equiv (Fin.consEquiv (fun _ : Fin (n + 1) => A))
      intro y
      simp [Fin.consEquiv, List.ofFn_succ, List.append_assoc]

/-- Pinsker for the actual joint prefix masses, with the manuscript's KL
orientation and exact 1/2 constant. -/
theorem prefixTV_le_sqrt_jointPrefixKL (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (n : ℕ) :
    acceptedPrefixTV p q (fun _ => 1) n ≤ Real.sqrt (jointPrefixKL p q n / 2) := by
  have hpmass : (∑ y : Fin n → A, pathMass p [] (List.ofFn y)) = 1 := by
    simpa [pathSum_eq_sum_ofFn] using pathMass_normalized p n
  have hqmass : (∑ y : Fin n → A, pathMass q [] (List.ofFn y)) = 1 := by
    simpa [pathSum_eq_sum_ofFn] using pathMass_normalized q n
  have hb := finite_pinsker
    (fun y : Fin n → A => pathMass p [] (List.ofFn y))
    (fun y : Fin n → A => pathMass q [] (List.ofFn y))
    (fun y => pathMass_nonneg p [] _) (fun y => pathMass_nonneg q [] _)
    hpmass hqmass (fun y => pathMass_mutual_support p q hs [] _)
  simpa [acceptedPrefixTV, jointPrefixKL, pathSum_eq_sum_ofFn, abs_sub_comm] using hb

theorem prefixTV_le_sqrt_chainKL (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (n : ℕ) :
    acceptedPrefixTV p q (fun _ => 1) n ≤ Real.sqrt (chainKL p q n [] / 2) := by
  rw [← jointPrefixKL_eq_chainKL p q hs]
  exact prefixTV_le_sqrt_jointPrefixKL p q hs n

end REINFORCEProMax.AutoregressiveTree
