import RLOOSurrogate
import ProMaxObjective

/-! Exact conversion from the actual corrected ProMax objective to the
RLOO group statistic, with arbitrary batch-dependent gates and normalization.
The comparison policy is the rollout policy, evaluated using the same frozen
snapshot and advantages. Shared absorbing padding has ratio residual zero.
-/
namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators
noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

def responseTokenRatio (p q : Policy A) {T : ℕ} (y : Fin T → A) (t : Fin T) : ℝ :=
  tokenRatio p q ((List.ofFn y).take t.val) (y t)

theorem response_residual_eq_token_sum (p q : Policy A) {T : ℕ} (y : Fin T → A) :
    responseRatioResidual p q T y = ∑ t, (responseTokenRatio p q y t - 1) := by
  unfold responseRatioResidual
  induction T with
  | zero => simp [List.ofFn_zero, ratioTrace, surrogateRhs]
  | succ T ih =>
    have hy := List.ofFn_succ' y
    rw [List.concat_eq_append] at hy
    rw [hy, ratioTrace_snoc, surrogateRhs_snoc, ih, Fin.sum_univ_castSucc]
    congr 1
    · apply Finset.sum_congr rfl
      intro t _
      unfold responseTokenRatio
      rw [hy, List.take_append_of_le_length (by simp)]
      rfl
    · unfold responseTokenRatio
      rw [hy, List.take_append_of_le_length (by simp), List.take_of_length_le (by simp)]

def groupAdvantage (R : List A → ℝ) {m T : ℕ}
    (s : Fin (m + 1) → (Fin T → A)) (i : Fin (m + 1)) : ℝ :=
  R (List.ofFn (s i)) - iidLooBaseline (fun y => R (List.ofFn y))
    (fun j => s (i.succAbove j))

theorem group_increment_eq_token_sum (p q : Policy A) (R : List A → ℝ) {m T : ℕ}
    (s : Fin (m + 1) → (Fin T → A)) :
    groupSurrogateIncrement p q R T s =
      ∑ i, ∑ t, groupAdvantage R s i * (responseTokenRatio p q (s i) t - 1) := by
  unfold groupSurrogateIncrement iidRlooEstimator
  rw [mul_div_cancel₀ _ (by positivity : (m + 1 : ℝ) ≠ 0)]
  apply Finset.sum_congr rfl
  intro i _
  simp only [iidLooTerm, groupAdvantage, response_residual_eq_token_sum, Finset.mul_sum]

def correctedBatchChange {I : Type*} [Fintype I]
    (a M w r r₀ c c₀ V H : I → ℝ) : ℝ :=
  (Objective.objective a M w r c H - Objective.objective a M w r₀ c₀ H) +
  (Objective.gateCorrection a M w r V - Objective.gateCorrection a M w r₀ V) -
  (Objective.normalizationCorrection a M w r V H - Objective.normalizationCorrection a M w r₀ V H) +
  (Objective.clippingPenalty a M w r c H - Objective.clippingPenalty a M w r₀ c₀ H)

/-- On a realized batch the actual gate, normalized advantages and clipping
need not obey independence assumptions. Both current/old and rollout/old
ratios are evaluated with the same detached old/rollout coefficients. -/
theorem corrected_batch_change_eq_rloo_ratio {n m T : ℕ}
    (p q : Fin n → Policy A) (R L : Fin n → List A → ℝ)
    (s : Fin n → Fin (m + 1) → Fin T → A)
    (mask M w r r₀ c c₀ H : (Fin n × Fin (m + 1) × Fin T) → ℝ)
    (hw : ∀ k, r k * w k = responseTokenRatio (p k.1) (q k.1) (s k.1 k.2.1) k.2.2)
    (hw₀ : ∀ k, r₀ k * w k = 1)
    (hpad : ∀ k, mask k *
      (responseTokenRatio (p k.1) (q k.1) (s k.1 k.2.1) k.2.2 - 1) =
      responseTokenRatio (p k.1) (q k.1) (s k.1 k.2.1) k.2.2 - 1) :
    let D := ∑ b, groupActiveLength (L b) (s b)
    correctedBatchChange (fun k => mask k / D) M w r r₀ c c₀
        (fun k => groupAdvantage (R k.1) (s k.1) k.2.1) H =
      (∑ b, groupSurrogateIncrement (p b) (q b) (R b) T (s b)) / D := by
  dsimp only
  unfold correctedBatchChange
  rw [Objective.corrected_objective_delta]
  unfold Objective.rawSurrogate
  rw [← Finset.sum_sub_distrib]
  simp_rw [hw, hw₀]
  have hterm : ∀ k : Fin n × Fin (m + 1) × Fin T,
      (mask k / (∑ b, groupActiveLength (L b) (s b))) *
          responseTokenRatio (p k.1) (q k.1) (s k.1 k.2.1) k.2.2 *
          groupAdvantage (R k.1) (s k.1) k.2.1 -
      (mask k / (∑ b, groupActiveLength (L b) (s b))) * 1 *
          groupAdvantage (R k.1) (s k.1) k.2.1 =
      (groupAdvantage (R k.1) (s k.1) k.2.1 *
          (responseTokenRatio (p k.1) (q k.1) (s k.1 k.2.1) k.2.2 - 1)) /
          (∑ b, groupActiveLength (L b) (s b)) := by
    intro k
    calc
      _ = groupAdvantage (R k.1) (s k.1) k.2.1 *
        (mask k * (responseTokenRatio (p k.1) (q k.1) (s k.1 k.2.1) k.2.2 - 1)) /
        (∑ b, groupActiveLength (L b) (s b)) := by ring
      _ = _ := by rw [hpad]
  simp_rw [hterm, group_increment_eq_token_sum]
  rw [← Finset.sum_div, Fintype.sum_prod_type]
  congr 1
  apply Finset.sum_congr rfl
  intro b _
  exact Fintype.sum_prod_type _

end
end REINFORCEProMax.AutoregressiveTree
