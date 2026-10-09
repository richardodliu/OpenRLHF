import PrefixConcentration
import TreePrefixGate

/-! The cached gate uses rollout/snapshot probabilities, while drift may
compare rollout with a different evaluated policy. Concentration premises
belong to the compared pair, not to the pair used to cache the mask. -/
namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators
variable {A : Type*} [Fintype A]

theorem cached_gate_drift_identity (p old q : Policy A) (τp τn : ℝ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) :
    acceptedPrefixTV p q (prefixLogGate p old τp τn) n =
      (1 / 2 : ℝ) * value p (fun y =>
        prefixLogGate p old τp τn y * |(ratioTrace p q y).prod - 1|) n [] :=
  acceptedPrefixTV_identity p q hs _ (prefixLogGate_nonneg p old τp τn) n

theorem cached_gate_drift_of_mgf (p old q : Policy A) (τp τn σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (n : ℕ)
    (hmgf : ∀ h, h.length < n → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2)) :
    acceptedPrefixTV p q (prefixLogGate p old τp τn) n ≤
      (1 / 2 : ℝ) * Real.sqrt (value p (prefixLogGate p old τp τn) n []) *
        Real.sqrt (Real.exp (2 * σ ^ 2 * (n : ℝ)) - 1) :=
  acceptedPrefixTV_of_mgf p q σ hs n hmgf _
    (prefixLogGate_nonneg p old τp τn) (prefixLogGate_le_one p old τp τn)

/-- A fixed common mask gives a triangle inequality even when its two
accepted masses differ. This is half L1, with no conditioning on acceptance. -/
theorem acceptedPrefixTV_triangle (p old q : Policy A) (M : List A → ℝ) (n : ℕ) :
    acceptedPrefixTV p q M n ≤
      acceptedPrefixTV p old M n + acceptedPrefixTV old q M n := by
  unfold acceptedPrefixTV
  simp only [pathSum_eq_sum_ofFn, List.nil_append]
  rw [← mul_add, ← Finset.sum_add_distrib]
  apply mul_le_mul_of_nonneg_left _ (by norm_num)
  apply Finset.sum_le_sum
  intro y _
  have hb := abs_sub_le (M (List.ofFn y) * pathMass q [] (List.ofFn y))
    (M (List.ofFn y) * pathMass old [] (List.ofFn y))
    (M (List.ofFn y) * pathMass p [] (List.ofFn y))
  simpa only [add_comm] using hb

end REINFORCEProMax.AutoregressiveTree
