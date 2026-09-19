import EmpiricalCertificate
import RLOOCovariance

/-!
Exact within-prompt RLOO variance and the between-prompt variance component
for the actual autoregressive rollout blocks. These sharpen the existing
full-objective certificate without changing gates, advantages or reduction.
-/
namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators
open BatchReduction
noncomputable section
variable {X A : Type*} [Fintype X] [DecidableEq X] [Fintype A] [DecidableEq A]

def responseCenteredReward (p : Policy A) (R : List A → ℝ) (T : ℕ) (y : Fin T → A) : ℝ :=
  R (List.ofFn y) - expectation (rolloutMass p T) (fun z => R (List.ofFn z))

theorem response_centered_covariance (p q : Policy A) (R : List A → ℝ) (T : ℕ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) :
    expectation (rolloutMass p T) (fun y =>
      responseCenteredReward p R T y * responseRatioResidual p q T y) = surrogate p q R T [] := by
  unfold responseCenteredReward
  simp only [sub_mul, expectation_sub, expectation_const_mul]
  have hz : expectation (rolloutMass p T) (responseRatioResidual p q T) = 0 :=
    response_ratio_residual_mean_zero p q hs T
  rw [hz, mul_zero, sub_zero]
  unfold expectation responseRatioResidual
  rw [rolloutMass_expectation p (fun y => R y * surrogateRhs (ratioTrace p q y)) T]
  exact trajectory_surrogate_eq p q R hs T

def responseRlooVariance (p q : Policy A) (R : List A → ℝ) (m T : ℕ) : ℝ :=
  (expectation (rolloutMass p T) (fun y =>
      (responseCenteredReward p R T y) ^ 2 * (responseRatioResidual p q T y) ^ 2) -
      (surrogate p q R T []) ^ 2) / (m + 1 : ℝ) +
    (expectation (rolloutMass p T) (fun y => (responseCenteredReward p R T y) ^ 2) *
      ratioEnergy p q T + (surrogate p q R T []) ^ 2) / ((m + 1 : ℝ) * (m : ℝ))

theorem response_rloo_variance_exact {m : ℕ} (hm : m ≠ 0)
    (p q : Policy A) (R : List A → ℝ) (T : ℕ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) :
    expectation (fun s : Fin (m + 1) → (Fin T → A) => iidMass (rolloutMass p T) s)
      (fun s => (iidRlooEstimator (rolloutMass p T) (fun y => R (List.ofFn y))
        (responseRatioResidual p q T) s - surrogate p q R T []) ^ 2) =
      responseRlooVariance p q R m T := by
  have hb := actual_rloo_variance hm (rolloutMass p T) (fun y => R (List.ofFn y))
    (responseRatioResidual p q T) (rolloutMass_normalized p T)
    (response_ratio_residual_mean_zero p q hs T)
  dsimp only at hb
  change expectation _ (fun s : Fin (m + 1) → (Fin T → A) =>
    (iidRlooEstimator _ _ _ s - expectation (rolloutMass p T)
      (fun y => responseCenteredReward p R T y * responseRatioResidual p q T y)) ^ 2) = _ at hb
  rw [response_centered_covariance p q R T hs] at hb
  change _ = (expectation (rolloutMass p T) (fun y =>
      (responseCenteredReward p R T y) ^ 2 * (responseRatioResidual p q T y) ^ 2) - _) / _ +
      (expectation (rolloutMass p T) (fun y => (responseCenteredReward p R T y) ^ 2) *
        expectation (rolloutMass p T) (fun y => (responseRatioResidual p q T y) ^ 2) + _) / _ at hb
  rw [response_residual_energy p q hs T] at hb
  have hθ := response_centered_covariance p q R T hs
  unfold responseCenteredReward at hθ
  rw [hθ] at hb
  exact hb

theorem response_rloo_variance_bound {m : ℕ} (hm : m ≠ 0)
    (p q : Policy A) (R : List A → ℝ) (T : ℕ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0)
    (B : ℝ) (hB : 0 ≤ B) (hR : ∀ y, |R y| ≤ B) :
    responseRlooVariance p q R m T ≤ rlooVarianceFactor m * (B ^ 2 * ratioEnergy p q T) := by
  have hb := actual_rloo_variance_bound hm (rolloutMass p T) (fun y => R (List.ofFn y))
    (responseRatioResidual p q T) (fun y => pathMass_nonneg _ _ _)
    (rolloutMass_normalized p T) (response_ratio_residual_mean_zero p q hs T) B hB (fun y => hR _)
  change expectation _ (fun s : Fin (m + 1) → (Fin T → A) =>
    (iidRlooEstimator _ _ _ s - expectation (rolloutMass p T)
      (fun y => responseCenteredReward p R T y * responseRatioResidual p q T y)) ^ 2) ≤ _ at hb
  rw [response_centered_covariance p q R T hs,
    response_rloo_variance_exact hm p q R T hs, response_residual_energy p q hs T] at hb
  exact hb

def familyPromptVariance (μ : X → ℝ) (p q : X → Policy A)
    (R : X → List A → ℝ) (T : ℕ) : ℝ :=
  ∑ x, μ x * (surrogate (p x) (q x) (R x) T [] - familySurrogate μ p q R T) ^ 2

def familyRlooVariance (μ : X → ℝ) (p q : X → Policy A)
    (R : X → List A → ℝ) (m T : ℕ) : ℝ :=
  familyPromptVariance μ p q R T + ∑ x, μ x * responseRlooVariance (p x) (q x) (R x) m T

theorem block_surrogate_variance_exact {m T : ℕ} (hm : m ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0) :
    expectation (blockMass μ p m T)
      (fun b => (blockSurrogate p q R b - familySurrogate μ p q R T) ^ 2) =
      familyRlooVariance μ p q R m T := by
  unfold expectation blockMass blockSurrogate blockIncrement groupSurrogateIncrement
  simp only [mul_div_cancel_left₀ _ (by positivity : (m + 1 : ℝ) ≠ 0)]
  rw [Fintype.sum_prod_type]
  simp_rw [mul_assoc, ← Finset.mul_sum]
  change (∑ x, μ x * expectation (fun s : Fin (m + 1) → (Fin T → A) =>
    iidMass (rolloutMass (p x) T) s) (fun s =>
      (iidRlooEstimator (rolloutMass (p x) T) (fun y => R x (List.ofFn y))
        (responseRatioResidual (p x) (q x) T) s - familySurrogate μ p q R T) ^ 2)) = _
  have hlocal : ∀ x, expectation (fun s : Fin (m + 1) → (Fin T → A) =>
      iidMass (rolloutMass (p x) T) s) (fun s =>
        (iidRlooEstimator (rolloutMass (p x) T) (fun y => R x (List.ofFn y))
          (responseRatioResidual (p x) (q x) T) s - familySurrogate μ p q R T) ^ 2) =
      responseRlooVariance (p x) (q x) (R x) m T +
        (surrogate (p x) (q x) (R x) T [] - familySurrogate μ p q R T) ^ 2 := by
    intro x
    rw [expectation_square_shift _ _
      (iid_mass_normalized _ (rolloutMass_normalized _ T) (m + 1))
      (surrogate (p x) (q x) (R x) T []) _
      (iid_rloo_surrogate_increment_expectation hm (p x) (q x) (R x) (hs x) T),
      response_rloo_variance_exact hm (p x) (q x) (R x) T (hs x)]
  simp_rw [hlocal, mul_add, Finset.sum_add_distrib]
  unfold familyRlooVariance familyPromptVariance
  ring

theorem family_rloo_variance_bound {m T : ℕ} (hm : m ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (B : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0) :
    familyRlooVariance μ p q R m T ≤ familyPromptVariance μ p q R T +
      rlooVarianceFactor m * (B ^ 2 * familyRatioEnergy μ p q T) := by
  unfold familyRlooVariance
  apply add_le_add_left
  calc
    _ ≤ ∑ x, μ x * (rlooVarianceFactor m * (B ^ 2 * ratioEnergy (p x) (q x) T)) :=
      Finset.sum_le_sum fun x _ => mul_le_mul_of_nonneg_left
        (response_rloo_variance_bound hm (p x) (q x) (R x) T (hs x) B hB (hR x)) (hμ0 x)
    _ = _ := by
      unfold familyRatioEnergy
      simp only [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro x _
      ring

theorem group_surrogate_mse_exact {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ) (hμ : ∑ x, μ x = 1)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0) :
    expectation (fun s : Fin n → RolloutBlock X A m T => iidMass (blockMass μ p m T) s)
      (fun s => (sampleMean (blockSurrogate p q R) s - familySurrogate μ p q R T) ^ 2) =
      familyRlooVariance μ p q R m T / (n : ℝ) := by
  have hb := finite_iid_baseline_variance hn (blockMass μ p m T) (blockSurrogate p q R)
    (block_mass_normalized μ p hμ m T) (familySurrogate μ p q R T)
    (block_surrogate_expectation hm μ p q R hs)
  change expectation _ (fun s : Fin n → RolloutBlock X A m T =>
    (sampleMean (blockSurrogate p q R) s - familySurrogate μ p q R T) ^ 2) =
      expectation (blockMass μ p m T) (fun b =>
        (blockSurrogate p q R b - familySurrogate μ p q R T) ^ 2) / (n : ℝ) at hb
  rwa [block_surrogate_variance_exact hm μ p q R hs] at hb

theorem family_rloo_variance_combined_bound {m T : ℕ} (hm : m ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0) :
    familyRlooVariance μ p q R m T ≤
      min (4 * B ^ 2 * familyRatioEnergy μ p q T)
        (familyPromptVariance μ p q R T +
          rlooVarianceFactor m * (B ^ 2 * familyRatioEnergy μ p q T)) := by
  apply le_min
  · have hb := variance_le_second_moment (blockMass μ p m T) (blockSurrogate p q R)
      (block_mass_normalized μ p hμ m T)
    rw [block_surrogate_expectation hm μ p q R hs,
      block_surrogate_variance_exact hm μ p q R hs] at hb
    exact hb.trans (block_surrogate_second_moment hm μ p q R hμ0 B hB hR hs)
  · exact family_rloo_variance_bound hm μ p q R hμ0 B hB hR hs

theorem group_surrogate_uniform_tail {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
    {J : Type*} [Fintype J]
    (μ : X → ℝ) (p : X → Policy A) (q : J → X → Policy A)
    (R : X → List A → ℝ) (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (η : J → ℝ) (hη : ∀ j, 0 < η j)
    (hs : ∀ j x h a, (p x).prob h a = 0 → (q j x).prob h a = 0) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T => ∃ j,
      η j ≤ |sampleMean (blockSurrogate p (q j) R) s - familySurrogate μ p (q j) R T|),
        iidMass (blockMass μ p m T) s) ≤
      ∑ j, familyRlooVariance μ p (q j) R m T / ((n : ℝ) * (η j) ^ 2) := by
  classical
  apply (finite_event_union_bound (iidMass (blockMass μ p m T))
    (iid_mass_nonneg _ (block_mass_nonneg μ p hμ0 m T)) _).trans
  apply Finset.sum_le_sum
  intro j _
  have hv : expectation (blockMass μ p m T) (fun b =>
      (blockSurrogate p (q j) R b - expectation (blockMass μ p m T) (blockSurrogate p (q j) R)) ^ 2) ≤
      familyRlooVariance μ p (q j) R m T := by
    rw [block_surrogate_expectation hm μ p (q j) R (hs j),
      block_surrogate_variance_exact hm μ p (q j) R (hs j)]
  have hb := iid_sample_mean_tail_of_variance hn (blockMass μ p m T) (blockSurrogate p (q j) R)
    (block_mass_nonneg μ p hμ0 m T) (block_mass_normalized μ p hμ m T)
    (familyRlooVariance μ p (q j) R m T) (η j) (hη j) hv
  rwa [block_surrogate_expectation hm μ p (q j) R (hs j)] at hb

theorem group_selected_false_improvement_bound {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
    {J : Type*} [Fintype J]
    (μ : X → ℝ) (p : X → Policy A) (q : J → X → Policy A)
    (R : X → List A → ℝ) (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B : ℝ) (ε δ η : J → ℝ) (hB : 0 ≤ B) (hη : ∀ j, 0 < η j)
    (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ j x h a, (p x).prob h a = 0 ↔ (q j x).prob h a = 0)
    (hTV : ∀ j x h, h.length < T → localTV (p x) (q j x) h ≤ ε j)
    (hKL : ∀ j x h, h.length < T → localReverseKL (p x) (q j x) h ≤ δ j)
    (select : (Fin n → RolloutBlock X A m T) → J) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T =>
      familyValue μ (q (select s)) R T - familyValue μ p R T ≤ 0 ∧
      familyAdaptiveCost μ p (q (select s)) B (ε (select s)) (δ (select s)) T +
        η (select s) < sampleMean (blockSurrogate p (q (select s)) R) s),
      iidMass (blockMass μ p m T) s) ≤
      ∑ j, familyRlooVariance μ p (q j) R m T / ((n : ℝ) * (η j) ^ 2) := by
  classical
  let E := fun s : Fin n → RolloutBlock X A m T => ∃ j,
    η j ≤ |sampleMean (blockSurrogate p (q j) R) s - familySurrogate μ p (q j) R T|
  have hsub : Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T =>
      familyValue μ (q (select s)) R T - familyValue μ p R T ≤ 0 ∧
      familyAdaptiveCost μ p (q (select s)) B (ε (select s)) (δ (select s)) T +
        η (select s) < sampleMean (blockSurrogate p (q (select s)) R) s) ⊆
      Finset.univ.filter E := by
    intro s hsamp
    obtain ⟨hbad, hcert⟩ := (Finset.mem_filter.mp hsamp).2
    apply Finset.mem_filter.mpr
    refine ⟨Finset.mem_univ _, select s, ?_⟩
    by_contra hsmall
    have hb := rescaled_performance_from_estimate μ p (q (select s)) R T hμ0
      B (ε (select s)) (δ (select s)) (η (select s))
      (sampleMean (blockSurrogate p (q (select s)) R) s) hB hR
      (hs (select s)) (hTV (select s)) (hKL (select s)) (le_of_lt (lt_of_not_ge hsmall))
    linarith
  exact (Finset.sum_le_sum_of_subset_of_nonneg hsub
    (fun s _ _ => iid_mass_nonneg _ (block_mass_nonneg μ p hμ0 m T) s)).trans
    (group_surrogate_uniform_tail hn hm μ p q R hμ0 hμ η hη
      (fun j x h a => (hs j x h a).mp))

end
end REINFORCEProMax.AutoregressiveTree
