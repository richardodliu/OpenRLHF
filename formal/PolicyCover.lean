import GroupCertificate

/-!
Uniform certificates for policies fitted on the evaluation data. A fixed
finite cover of a pre-specified policy class pays an explicit approximation
margin; the selected policy need not be a cover center. The metric controls
additive response ratio residuals; a token-ratio cover is sufficient. The
proof retains the actual RLOO statistic.
-/
namespace REINFORCEProMax.BatchReduction
open scoped BigOperators
noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

theorem sample_mean_abs_bound {n : ℕ} (hn : n ≠ 0) (f : A → ℝ)
    (C : ℝ) (hf : ∀ a, |f a| ≤ C) (s : Fin n → A) : |sampleMean f s| ≤ C := by
  have hnpos : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero hn
  calc
    _ = |∑ i, f (s i)| / (n : ℝ) := by simp [sampleMean, abs_div, abs_of_pos hnpos]
    _ ≤ (∑ i, |f (s i)|) / (n : ℝ) :=
      div_le_div_of_nonneg_right (Finset.abs_sum_le_sum_abs _ _) hnpos.le
    _ ≤ (∑ _i : Fin n, C) / (n : ℝ) :=
      div_le_div_of_nonneg_right (Finset.sum_le_sum fun i _ => hf (s i)) hnpos.le
    _ = C := by simp [ne_of_gt hnpos]

theorem sample_mean_difference_bound {n : ℕ} (hn : n ≠ 0) (f g : A → ℝ)
    (C : ℝ) (hfg : ∀ a, |f a - g a| ≤ C) (s : Fin n → A) :
    |sampleMean f s - sampleMean g s| ≤ C := by
  have heq : sampleMean f s - sampleMean g s = sampleMean (fun a => f a - g a) s := by
    simp only [sampleMean, Finset.sum_sub_distrib, sub_div]
  rw [heq]
  exact sample_mean_abs_bound hn _ C hfg s

theorem rloo_estimator_difference_bound {m : ℕ} (hm : m ≠ 0)
    (p r f g : A → ℝ) (B e : ℝ) (hB : 0 ≤ B)
    (hr : ∀ a, |r a| ≤ B) (hfg : ∀ a, |f a - g a| ≤ e)
    (s : Fin (m + 1) → A) :
    |iidRlooEstimator p r f s - iidRlooEstimator p r g s| ≤ 2 * B * e := by
  have hterm : ∀ i : Fin (m + 1), |iidLooTerm p r f i s - iidLooTerm p r g i s| ≤ 2 * B * e := by
    intro i
    have ha : |r (s i) - iidLooBaseline r (fun j => s (i.succAbove j))| ≤ 2 * B :=
      (abs_sub _ _).trans (by linarith [hr (s i),
        loo_baseline_abs_bound hm r B hr (fun j => s (i.succAbove j))])
    simp only [iidLooTerm, ← mul_sub, abs_mul]
    exact mul_le_mul ha (hfg (s i)) (abs_nonneg _) (by positivity)
  simpa only [sampleMean, iidRlooEstimator, Nat.cast_add, Nat.cast_one] using
    sample_mean_difference_bound (by omega : m + 1 ≠ 0)
      (fun i : Fin (m + 1) => iidLooTerm p r f i s)
      (fun i : Fin (m + 1) => iidLooTerm p r g i s) (2 * B * e) hterm id

end
end REINFORCEProMax.BatchReduction

namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators
open BatchReduction
noncomputable section
attribute [local instance] Classical.propDecidable
variable {X A : Type*} [Fintype X] [DecidableEq X] [Fintype A] [DecidableEq A]

theorem response_residual_difference_of_token_cover (p q q' : Policy A) (T : ℕ) (d : ℝ)
    (hc : ∀ h a, h.length < T → |tokenRatio p q h a - tokenRatio p q' h a| ≤ d)
    (y : Fin T → A) :
    |responseRatioResidual p q T y - responseRatioResidual p q' T y| ≤ (T : ℝ) * d := by
  rw [response_residual_eq_token_sum, response_residual_eq_token_sum, ← Finset.sum_sub_distrib]
  calc
    _ ≤ ∑ t, |(responseTokenRatio p q y t - 1) - (responseTokenRatio p q' y t - 1)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _t : Fin T, d := by
      apply Finset.sum_le_sum
      intro t _
      simp only [sub_sub_sub_cancel_right, responseTokenRatio]
      exact hc _ _ ((List.length_take_le _ _).trans_lt t.isLt)
    _ = _ := by simp

theorem response_surrogate_expectation (p q : Policy A) (R : List A → ℝ) (T : ℕ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) :
    expectation (rolloutMass p T) (fun y => R (List.ofFn y) * responseRatioResidual p q T y) =
      surrogate p q R T [] := by
  unfold expectation responseRatioResidual
  rw [rolloutMass_expectation p (fun y => R y * surrogateRhs (ratioTrace p q y)) T]
  exact trajectory_surrogate_eq p q R hs T

theorem family_surrogate_difference_of_residual_cover (μ : X → ℝ) (p q q' : X → Policy A)
    (R : X → List A → ℝ) (T : ℕ) (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B e : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0)
    (hs' : ∀ x h a, (p x).prob h a = 0 → (q' x).prob h a = 0)
    (hc : ∀ x y, |responseRatioResidual (p x) (q x) T y -
      responseRatioResidual (p x) (q' x) T y| ≤ e) :
    |familySurrogate μ p q R T - familySurrogate μ p q' R T| ≤ B * e := by
  have hlocal : ∀ x, |surrogate (p x) (q x) (R x) T [] -
      surrogate (p x) (q' x) (R x) T []| ≤ B * e := by
    intro x
    rw [← response_surrogate_expectation (p x) (q x) (R x) T (hs x),
      ← response_surrogate_expectation (p x) (q' x) (R x) T (hs' x), ← expectation_sub]
    apply expectation_abs_bound _ _ (fun y => pathMass_nonneg _ _ _)
      (rolloutMass_normalized _ T) (B * e)
    intro y
    rw [← mul_sub, abs_mul]
    exact mul_le_mul (hR x _) (hc x y) (abs_nonneg _) hB
  change |expectation μ (fun x => surrogate (p x) (q x) (R x) T []) -
    expectation μ (fun x => surrogate (p x) (q' x) (R x) T [])| ≤ _
  rw [← expectation_sub]
  exact expectation_abs_bound μ _ hμ0 hμ (B * e) hlocal

theorem block_surrogate_difference_of_residual_cover {m T : ℕ} (hm : m ≠ 0)
    (p q q' : X → Policy A) (R : X → List A → ℝ) (B e : ℝ) (hB : 0 ≤ B)
    (hR : ∀ x y, |R x y| ≤ B)
    (hc : ∀ x y, |responseRatioResidual (p x) (q x) T y -
      responseRatioResidual (p x) (q' x) T y| ≤ e)
    (b : RolloutBlock X A m T) :
    |blockSurrogate p q R b - blockSurrogate p q' R b| ≤ 2 * B * e := by
  unfold blockSurrogate blockIncrement groupSurrogateIncrement
  simp only [mul_div_cancel_left₀ _ (by positivity : (m + 1 : ℝ) ≠ 0)]
  exact rloo_estimator_difference_bound hm _ _ _ _ B e hB (fun y => hR b.1 _) (hc b.1) b.2

theorem empirical_surrogate_difference_of_residual_cover {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
    (p q q' : X → Policy A) (R : X → List A → ℝ) (B e : ℝ) (hB : 0 ≤ B)
    (hR : ∀ x y, |R x y| ≤ B)
    (hc : ∀ x y, |responseRatioResidual (p x) (q x) T y -
      responseRatioResidual (p x) (q' x) T y| ≤ e)
    (s : Fin n → RolloutBlock X A m T) :
    |sampleMean (blockSurrogate p q R) s - sampleMean (blockSurrogate p q' R) s| ≤ 2 * B * e :=
  sample_mean_difference_bound hn _ _ (2 * B * e)
    (block_surrogate_difference_of_residual_cover hm p q q' R B e hB hR hc) s

theorem surrogate_deviation_transfer {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
    (μ : X → ℝ) (p q q' : X → Policy A) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B e : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0)
    (hs' : ∀ x h a, (p x).prob h a = 0 → (q' x).prob h a = 0)
    (hc : ∀ x y, |responseRatioResidual (p x) (q x) T y -
      responseRatioResidual (p x) (q' x) T y| ≤ e)
    (s : Fin n → RolloutBlock X A m T) :
    |sampleMean (blockSurrogate p q R) s - familySurrogate μ p q R T| ≤
      |sampleMean (blockSurrogate p q' R) s - familySurrogate μ p q' R T| + 3 * B * e := by
  have he := empirical_surrogate_difference_of_residual_cover hn hm p q q' R B e hB hR hc s
  have hp := family_surrogate_difference_of_residual_cover μ p q q' R T hμ0 hμ B e hB hR hs hs' hc
  have htri := abs_sub_le (sampleMean (blockSurrogate p q R) s)
    (sampleMean (blockSurrogate p q' R) s) (familySurrogate μ p q R T)
  have htri' := abs_sub_le (sampleMean (blockSurrogate p q' R) s)
    (familySurrogate μ p q' R T) (familySurrogate μ p q R T)
  rw [abs_sub_comm (familySurrogate μ p q' R T)] at htri'
  linarith

theorem covered_performance_lower_bound {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
    (μ : X → ℝ) (p q q' : X → Policy A) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B e ε δ η : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 ↔ (q x).prob h a = 0)
    (hs' : ∀ x h a, (p x).prob h a = 0 → (q' x).prob h a = 0)
    (hc : ∀ x y, |responseRatioResidual (p x) (q x) T y -
      responseRatioResidual (p x) (q' x) T y| ≤ e)
    (hTV : ∀ x h, h.length < T → localTV (p x) (q x) h ≤ ε)
    (hKL : ∀ x h, h.length < T → localReverseKL (p x) (q x) h ≤ δ)
    (s : Fin n → RolloutBlock X A m T)
    (hcenter : |sampleMean (blockSurrogate p q' R) s - familySurrogate μ p q' R T| ≤ η) :
    sampleMean (blockSurrogate p q R) s - familyAdaptiveCost μ p q B ε δ T - η - 3 * B * e ≤
      familyValue μ q R T - familyValue μ p R T := by
  have htransfer := surrogate_deviation_transfer hn hm μ p q q' R hμ0 hμ B e hB hR
    (fun x h a => (hs x h a).mp) hs' hc s
  have hb := rescaled_performance_from_estimate μ p q R T hμ0 B ε δ (η + 3 * B * e)
    (sampleMean (blockSurrogate p q R) s) hB hR hs hTV hKL (by linarith)
  linarith

/-- The class index type is arbitrary, and can be a continuous parameter
space. Only the fixed cover centers carry a finite union cost. -/
theorem covered_surrogate_uniform_tail {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
    {Θ J : Type*} [Fintype J]
    (μ : X → ℝ) (p : X → Policy A) (q : Θ → X → Policy A)
    (center : J → Θ) (net : Θ → J) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B e : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (η : J → ℝ) (hη : ∀ j, 0 < η j)
    (hs : ∀ θ x h a, (p x).prob h a = 0 → (q θ x).prob h a = 0)
    (hc : ∀ θ x y, |responseRatioResidual (p x) (q θ x) T y -
      responseRatioResidual (p x) (q (center (net θ)) x) T y| ≤ e) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T => ∃ θ,
      η (net θ) + 3 * B * e ≤
        |sampleMean (blockSurrogate p (q θ) R) s - familySurrogate μ p (q θ) R T|),
        iidMass (blockMass μ p m T) s) ≤
      ∑ j, familyRlooVariance μ p (q (center j)) R m T / ((n : ℝ) * (η j) ^ 2) := by
  classical
  have hsub : Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T => ∃ θ,
      η (net θ) + 3 * B * e ≤
        |sampleMean (blockSurrogate p (q θ) R) s - familySurrogate μ p (q θ) R T|) ⊆
      Finset.univ.filter (fun s => ∃ j,
        η j ≤ |sampleMean (blockSurrogate p (q (center j)) R) s -
          familySurrogate μ p (q (center j)) R T|) := by
    intro s hsamp
    obtain ⟨θ, hlarge⟩ := (Finset.mem_filter.mp hsamp).2
    refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, net θ, ?_⟩
    have htransfer := surrogate_deviation_transfer hn hm μ p (q θ) (q (center (net θ))) R
      hμ0 hμ B e hB hR (hs θ) (hs (center (net θ))) (hc θ) s
    linarith
  exact (Finset.sum_le_sum_of_subset_of_nonneg hsub
    (fun s _ _ => iid_mass_nonneg _ (block_mass_nonneg μ p hμ0 m T) s)).trans
    (group_surrogate_uniform_tail hn hm μ p (fun j => q (center j)) R hμ0 hμ η hη
      (fun j => hs (center j)))

/-- An arbitrary evaluation-data fit within the covered class obeys the
same reward certificate, with its own Adaptive cost and a cover margin.
It is not required to equal one of the fixed centers. -/
theorem covered_selected_false_improvement_bound {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
    {Θ J : Type*} [Fintype J]
    (μ : X → ℝ) (p : X → Policy A) (q : Θ → X → Policy A)
    (center : J → Θ) (net : Θ → J) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B e : ℝ) (ε δ : Θ → ℝ) (η : J → ℝ) (hB : 0 ≤ B) (hη : ∀ j, 0 < η j)
    (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ θ x h a, (p x).prob h a = 0 ↔ (q θ x).prob h a = 0)
    (hc : ∀ θ x y, |responseRatioResidual (p x) (q θ x) T y -
      responseRatioResidual (p x) (q (center (net θ)) x) T y| ≤ e)
    (hTV : ∀ θ x h, h.length < T → localTV (p x) (q θ x) h ≤ ε θ)
    (hKL : ∀ θ x h, h.length < T → localReverseKL (p x) (q θ x) h ≤ δ θ)
    (select : (Fin n → RolloutBlock X A m T) → Θ) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T =>
      familyValue μ (q (select s)) R T - familyValue μ p R T ≤ 0 ∧
      familyAdaptiveCost μ p (q (select s)) B (ε (select s)) (δ (select s)) T +
        η (net (select s)) + 3 * B * e < sampleMean (blockSurrogate p (q (select s)) R) s),
      iidMass (blockMass μ p m T) s) ≤
      ∑ j, familyRlooVariance μ p (q (center j)) R m T / ((n : ℝ) * (η j) ^ 2) := by
  classical
  let E := fun s : Fin n → RolloutBlock X A m T => ∃ θ,
    η (net θ) + 3 * B * e ≤
      |sampleMean (blockSurrogate p (q θ) R) s - familySurrogate μ p (q θ) R T|
  have hsub : Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T =>
      familyValue μ (q (select s)) R T - familyValue μ p R T ≤ 0 ∧
      familyAdaptiveCost μ p (q (select s)) B (ε (select s)) (δ (select s)) T +
        η (net (select s)) + 3 * B * e < sampleMean (blockSurrogate p (q (select s)) R) s) ⊆
      Finset.univ.filter E := by
    intro s hsamp
    obtain ⟨hbad, hcert⟩ := (Finset.mem_filter.mp hsamp).2
    apply Finset.mem_filter.mpr
    refine ⟨Finset.mem_univ _, select s, ?_⟩
    by_contra hsmall
    have hb := rescaled_performance_from_estimate μ p (q (select s)) R T hμ0
      B (ε (select s)) (δ (select s)) (η (net (select s)) + 3 * B * e)
      (sampleMean (blockSurrogate p (q (select s)) R) s) hB hR
      (hs (select s)) (hTV (select s)) (hKL (select s)) (le_of_lt (lt_of_not_ge hsmall))
    linarith
  exact (Finset.sum_le_sum_of_subset_of_nonneg hsub
    (fun s _ _ => iid_mass_nonneg _ (block_mass_nonneg μ p hμ0 m T) s)).trans
    (covered_surrogate_uniform_tail hn hm μ p q center net R hμ0 hμ B e hB hR η hη
      (fun θ x h a => (hs θ x h a).mp) hc)

end
end REINFORCEProMax.AutoregressiveTree
