import FiniteSelection
import RLOOSurrogate
import AdaptiveBound
import ProMaxBatch

/-!
Finite-sample performance certificates on actual iid RLOO groups. A block
contains a prompt and all sibling responses, so the proof never treats
dependent leave-one-out contributions as independent samples. Prompt laws,
response lengths, policy values, surrogate means and Adaptive costs are
constructed from the same normalized autoregressive model.
-/
namespace REINFORCEProMax.AutoregressiveTree

open scoped BigOperators
open BatchReduction
noncomputable section

variable {X A : Type*} [Fintype X] [DecidableEq X] [Fintype A] [DecidableEq A]

abbrev RolloutBlock (X A : Type*) (m T : ℕ) := X × (Fin (m + 1) → (Fin T → A))

def blockMass (μ : X → ℝ) (p : X → Policy A) (m T : ℕ)
    (b : RolloutBlock X A m T) : ℝ := μ b.1 * iidMass (rolloutMass (p b.1) T) b.2

def blockIncrement (p q : X → Policy A) (R : X → List A → ℝ) {m T : ℕ}
    (b : RolloutBlock X A m T) : ℝ := groupSurrogateIncrement (p b.1) (q b.1) (R b.1) T b.2

def blockLength (L : X → List A → ℝ) {m T : ℕ}
    (b : RolloutBlock X A m T) : ℝ := groupActiveLength (L b.1) b.2

theorem block_mass_nonneg (μ : X → ℝ) (p : X → Policy A)
    (hμ : ∀ x, 0 ≤ μ x) (m T : ℕ) (b : RolloutBlock X A m T) :
    0 ≤ blockMass μ p m T b :=
  mul_nonneg (hμ b.1) (iid_mass_nonneg _ (fun y => pathMass_nonneg _ _ _) b.2)

theorem block_mass_normalized (μ : X → ℝ) (p : X → Policy A)
    (hμ : ∑ x, μ x = 1) (m T : ℕ) :
    (∑ b : RolloutBlock X A m T, blockMass μ p m T b) = 1 := by
  unfold blockMass
  rw [Fintype.sum_prod_type]
  simp_rw [← Finset.mul_sum, iid_mass_normalized _ (rolloutMass_normalized _ T)]
  simpa using hμ

theorem block_increment_expectation {m T : ℕ} (hm : m ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0) :
    expectation (blockMass μ p m T) (blockIncrement p q R) =
      (m + 1 : ℝ) * familySurrogate μ p q R T := by
  unfold expectation blockMass blockIncrement
  rw [Fintype.sum_prod_type]
  simp_rw [mul_assoc, ← Finset.mul_sum]
  change (∑ x, μ x * expectation (iidMass (rolloutMass (p x) T))
    (groupSurrogateIncrement (p x) (q x) (R x) T)) = _
  simp_rw [group_surrogate_increment_expectation hm _ _ _ (hs _) T]
  unfold familySurrogate
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro x _
  ring

theorem block_length_expectation (μ : X → ℝ) (p : X → Policy A)
    (L : X → List A → ℝ) (m T : ℕ) :
    expectation (blockMass μ p m T) (blockLength L) =
      (m + 1 : ℝ) * familyValue μ p L T := by
  unfold expectation blockMass blockLength
  rw [Fintype.sum_prod_type]
  simp_rw [mul_assoc, ← Finset.mul_sum]
  change (∑ x, μ x * expectation (iidMass (rolloutMass (p x) T))
    (groupActiveLength (L x))) = _
  simp_rw [group_active_length_expectation]
  unfold familyValue
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro x _
  ring

theorem block_length_lower_bound (L : X → List A → ℝ) {m T : ℕ} (l : ℝ)
    (hL : ∀ x (y : Fin T → A), l ≤ L x (List.ofFn y))
    (b : RolloutBlock X A m T) : (m + 1 : ℝ) * l ≤ blockLength L b := by
  unfold blockLength groupActiveLength
  calc
    (m + 1 : ℝ) * l = ∑ _i : Fin (m + 1), l := by simp
    _ ≤ _ := Finset.sum_le_sum fun i _ => hL b.1 (b.2 i)

theorem family_length_positive (μ : X → ℝ) (p : X → Policy A)
    (L : X → List A → ℝ) (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (T : ℕ) (l : ℝ) (hl : 0 < l)
    (hL : ∀ x (y : Fin T → A), l ≤ L x (List.ofFn y)) :
    0 < familyValue μ p L T := by
  have hb := expectation_lower_bound (blockMass μ p 0 T) (blockLength L)
    (block_mass_nonneg μ p hμ0 0 T) (block_mass_normalized μ p hμ 0 T)
    ((0 + 1 : ℝ) * l) (fun b => by simpa using block_length_lower_bound L l hL b)
  rw [block_length_expectation] at hb
  norm_num at hb
  exact hl.trans_le hb

/-- Sibling count cancels from the population token target; random lengths
remain through the expected response length under the original prompt law. -/
theorem block_ratio_target {m T : ℕ} (hm : m ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R L : X → List A → ℝ)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0) :
    expectation (blockMass μ p m T) (blockIncrement p q R) /
      expectation (blockMass μ p m T) (blockLength L) =
      familySurrogate μ p q R T / familyValue μ p L T := by
  rw [block_increment_expectation hm μ p q R hs, block_length_expectation]
  exact mul_div_mul_left _ _ (by positivity : (m + 1 : ℝ) ≠ 0)

def familyAdaptiveCost (μ : X → ℝ) (p q : X → Policy A)
    (B ε δ : ℝ) (T : ℕ) : ℝ :=
  4 * B * ∑ x, μ x * adaptiveCost (p x) (q x) ε δ T

/-- The algebraic objective identity instantiated on the exact prompt/group
sample space of the statistical theorem. Normalization and gates may use
the entire realized batch. The weights use its actual active-token count. -/
theorem corrected_block_objective_eq {n m T : ℕ} (hn : n ≠ 0)
    (p q : X → Policy A) (R L : X → List A → ℝ)
    (s : Fin n → RolloutBlock X A m T)
    (mask M w r r₀ c c₀ H : (Fin n × Fin (m + 1) × Fin T) → ℝ)
    (hw : ∀ k, r k * w k = responseTokenRatio (p (s k.1).1) (q (s k.1).1)
      ((s k.1).2 k.2.1) k.2.2)
    (hw₀ : ∀ k, r₀ k * w k = 1)
    (hpad : ∀ k, mask k *
      (responseTokenRatio (p (s k.1).1) (q (s k.1).1) ((s k.1).2 k.2.1) k.2.2 - 1) =
      responseTokenRatio (p (s k.1).1) (q (s k.1).1) ((s k.1).2 k.2.1) k.2.2 - 1) :
    correctedBatchChange (fun k => mask k / ∑ b, blockLength L (s b)) M w r r₀ c c₀
      (fun k => groupAdvantage (R (s k.1).1) (s k.1).2 k.2.1) H =
      sampleMean (blockIncrement p q R) s / sampleMean (blockLength L) s := by
  rw [← token_reduction_eq_mean_ratio hn]
  exact corrected_batch_change_eq_rloo_ratio (fun b => p (s b).1) (fun b => q (s b).1)
    (fun b => R (s b).1) (fun b => L (s b).1) (fun b => (s b).2)
    mask M w r r₀ c c₀ H hw hw₀ hpad

theorem rloo_uniform_estimation_bound {m T n : ℕ} (hm : m ≠ 0) (hn : n ≠ 0)
    {J : Type*} [Fintype J]
    (μ : X → ℝ) (p : X → Policy A) (q : J → X → Policy A)
    (R L : X → List A → ℝ) (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (l : ℝ) (η : J → ℝ) (hl : 0 < l) (hη : ∀ j, 0 < η j)
    (hL : ∀ x (y : Fin T → A), l ≤ L x (List.ofFn y))
    (hs : ∀ j x h a, (p x).prob h a = 0 → (q j x).prob h a = 0) :
    let law := blockMass μ p m T
    let Z := fun j => blockIncrement (m := m) (T := T) p (q j) R
    let length := blockLength (m := m) (T := T) L
    let κ := fun j => familySurrogate μ p (q j) R T / familyValue μ p L T
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T => ∃ j,
      η j ≤ |sampleMean (Z j) s / sampleMean length s - κ j|), iidMass law s) ≤
      ∑ j, expectation law (fun b => (Z j b - κ j * length b) ^ 2) /
        ((n : ℝ) * ((m + 1 : ℝ) * l) ^ 2 * (η j) ^ 2) := by
  classical
  dsimp only
  apply iid_token_mean_uniform_tail hn _ _ _ (block_mass_nonneg μ p hμ0 m T)
    (block_mass_normalized μ p hμ m T) _ _ _ (mul_pos (by positivity) hl) hη
    (block_length_lower_bound L l hL)
  intro j
  have hLen := family_length_positive μ p L hμ0 hμ T l hl hL
  have hEL : expectation (blockMass μ p m T) (blockLength L) ≠ 0 := by
    rw [block_length_expectation]
    exact ne_of_gt (mul_pos (by positivity) hLen)
  have hc := ratio_target_centered (blockMass μ p m T)
    (blockIncrement p (q j) R) (blockLength L) hEL
  rwa [block_ratio_target hm μ p (q j) R L (hs j)] at hc

theorem family_performance_from_block_target {m T : ℕ} (hm : m ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R L : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hLen : 0 < familyValue μ p L T)
    (B ε δ : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 ↔ (q x).prob h a = 0)
    (hTV : ∀ x h, h.length < T → localTV (p x) (q x) h ≤ ε)
    (hKL : ∀ x h, h.length < T → localReverseKL (p x) (q x) h ≤ δ) :
    familyValue μ p L T *
      (expectation (blockMass μ p m T) (blockIncrement p q R) /
        expectation (blockMass μ p m T) (blockLength L)) -
      familyAdaptiveCost μ p q B ε δ T ≤ familyValue μ q R T - familyValue μ p R T := by
  rw [block_ratio_target hm μ p q R L (fun x h a => (hs x h a).mp),
    mul_div_cancel₀ _ (ne_of_gt hLen)]
  unfold familyAdaptiveCost
  rw [family_adaptive_cost]
  exact family_performance_lower_bound_adaptive μ p q R hμ0 B ε δ hB hR T hs hTV hKL

/-- On the simultaneous estimation event, this applies to every candidate
and therefore to any selection from the fixed family. `estimate` can be
rewritten to the full corrected objective by corrected_block_objective_eq. -/
theorem rloo_performance_from_estimate {m T : ℕ} (hm : m ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R L : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hLen : 0 < familyValue μ p L T)
    (B ε δ η estimate : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 ↔ (q x).prob h a = 0)
    (hTV : ∀ x h, h.length < T → localTV (p x) (q x) h ≤ ε)
    (hKL : ∀ x h, h.length < T → localReverseKL (p x) (q x) h ≤ δ)
    (hest : |estimate - familySurrogate μ p q R T / familyValue μ p L T| ≤ η) :
    familyValue μ p L T * (estimate - η) - familyAdaptiveCost μ p q B ε δ T ≤
      familyValue μ q R T - familyValue μ p R T := by
  have hb := family_performance_from_block_target hm μ p q R L hμ0 hLen
    B ε δ hB hR hs hTV hKL
  rw [block_ratio_target hm μ p q R L (fun x h a => (hs x h a).mp)] at hb
  exact performance_lower_bound_from_estimate _ _ _ _ _ _ hLen.le hb hest

/-- The paper's corrected-objective performance inequality, composed inside
Lean rather than leaving the objective/statistic substitution to prose.
This is conditional on the estimation event; its probability is controlled
separately by `rloo_uniform_estimation_bound`. -/
theorem rloo_performance_from_corrected_objective {m T n : ℕ}
    (hm : m ≠ 0) (hn : n ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R L : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hLen : 0 < familyValue μ p L T)
    (B ε δ η : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 ↔ (q x).prob h a = 0)
    (hTV : ∀ x h, h.length < T → localTV (p x) (q x) h ≤ ε)
    (hKL : ∀ x h, h.length < T → localReverseKL (p x) (q x) h ≤ δ)
    (s : Fin n → RolloutBlock X A m T)
    (mask M w r r₀ c c₀ H : (Fin n × Fin (m + 1) × Fin T) → ℝ)
    (hw : ∀ k, r k * w k = responseTokenRatio (p (s k.1).1) (q (s k.1).1)
      ((s k.1).2 k.2.1) k.2.2)
    (hw₀ : ∀ k, r₀ k * w k = 1)
    (hpad : ∀ k, mask k *
      (responseTokenRatio (p (s k.1).1) (q (s k.1).1) ((s k.1).2 k.2.1) k.2.2 - 1) =
      responseTokenRatio (p (s k.1).1) (q (s k.1).1) ((s k.1).2 k.2.1) k.2.2 - 1)
    (hest : |sampleMean (blockIncrement p q R) s / sampleMean (blockLength L) s -
      familySurrogate μ p q R T / familyValue μ p L T| ≤ η) :
    familyValue μ p L T *
      (correctedBatchChange (fun k => mask k / ∑ b, blockLength L (s b)) M w r r₀ c c₀
        (fun k => groupAdvantage (R (s k.1).1) (s k.1).2 k.2.1) H - η) -
      familyAdaptiveCost μ p q B ε δ T ≤ familyValue μ q R T - familyValue μ p R T := by
  rw [corrected_block_objective_eq hn p q R L s mask M w r r₀ c c₀ H hw hw₀ hpad]
  exact rloo_performance_from_estimate hm μ p q R L hμ0 hLen B ε δ η
    (sampleMean (blockIncrement p q R) s / sampleMean (blockLength L) s)
    hB hR hs hTV hKL hest

/-- A finite family may be selected using all evaluation blocks. This
composes actual RLOO, random-length reduction and the proved Adaptive bound;
the population mean and performance bound are not independent premises. -/
theorem rloo_selected_false_improvement_bound {m T n : ℕ} (hm : m ≠ 0) (hn : n ≠ 0)
    {J : Type*} [Fintype J]
    (μ : X → ℝ) (p : X → Policy A) (q : J → X → Policy A)
    (R L : X → List A → ℝ) (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (l B : ℝ) (ε δ η : J → ℝ) (hl : 0 < l) (hB : 0 ≤ B) (hη : ∀ j, 0 < η j)
    (hL : ∀ x (y : Fin T → A), l ≤ L x (List.ofFn y))
    (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ j x h a, (p x).prob h a = 0 ↔ (q j x).prob h a = 0)
    (hTV : ∀ j x h, h.length < T → localTV (p x) (q j x) h ≤ ε j)
    (hKL : ∀ j x h, h.length < T → localReverseKL (p x) (q j x) h ≤ δ j)
    (select : (Fin n → RolloutBlock X A m T) → J) :
    let law := blockMass μ p m T
    let Z := fun j => blockIncrement (m := m) (T := T) p (q j) R
    let length := blockLength (m := m) (T := T) L
    let κ := fun j => familySurrogate μ p (q j) R T / familyValue μ p L T
    let cost := fun j => familyAdaptiveCost μ p (q j) B (ε j) (δ j) T
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T =>
      familyValue μ (q (select s)) R T - familyValue μ p R T ≤ 0 ∧
      cost (select s) / familyValue μ p L T + η (select s) <
        sampleMean (Z (select s)) s / sampleMean length s), iidMass law s) ≤
      ∑ j, expectation law (fun b => (Z j b - κ j * length b) ^ 2) /
        ((n : ℝ) * ((m + 1 : ℝ) * l) ^ 2 * (η j) ^ 2) := by
  classical
  dsimp only
  have hLen := family_length_positive μ p L hμ0 hμ T l hl hL
  have hEL : expectation (blockMass μ p m T) (blockLength L) ≠ 0 := by
    rw [block_length_expectation]
    exact ne_of_gt (mul_pos (by positivity) hLen)
  have hcenter : ∀ j, expectation (blockMass μ p m T) (fun b =>
      blockIncrement p (q j) R b -
        (familySurrogate μ p (q j) R T / familyValue μ p L T) * blockLength L b) = 0 := by
    intro j
    have hc := ratio_target_centered (blockMass μ p m T)
      (blockIncrement p (q j) R) (blockLength L) hEL
    rwa [block_ratio_target hm μ p (q j) R L (fun x h a => (hs j x h a).mp)] at hc
  apply selected_false_improvement_bound hn
    (blockMass μ p m T) (blockLength L) (fun j => blockIncrement p (q j) R)
    (block_mass_nonneg μ p hμ0 m T) (block_mass_normalized μ p hμ m T)
    ((m + 1 : ℝ) * l) (familyValue μ p L T)
    (fun j => familySurrogate μ p (q j) R T / familyValue μ p L T)
    (fun j => familyAdaptiveCost μ p (q j) B (ε j) (δ j) T) η
    (fun j => familyValue μ (q j) R T - familyValue μ p R T)
    (mul_pos (by positivity) hl) hLen hη (block_length_lower_bound L l hL) hcenter
    _ select
  intro j
  have hb := family_performance_from_block_target hm μ p (q j) R L hμ0 hLen
    B (ε j) (δ j) hB hR (hs j) (hTV j) (hKL j)
  rwa [block_ratio_target hm μ p (q j) R L (fun x h a => (hs j x h a).mp)] at hb

end
end REINFORCEProMax.AutoregressiveTree
