import ProMaxCertificate
import SurrogateEnergy
import Mathlib.Algebra.Order.Chebyshev

/-!
Observed-token rescaling removes the unknown population response length.
The sampling margin uses the additive ratio energy on actual RLOO blocks.
The bound is conservative in sibling count: no independence is imposed on
the shared-baseline contributions, and no extra 1/n variance rate is claimed.
-/
namespace REINFORCEProMax.BatchReduction

open scoped BigOperators
noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

theorem loo_baseline_abs_bound {m : ℕ} (hm : m ≠ 0) (r : A → ℝ)
    (B : ℝ) (hr : ∀ a, |r a| ≤ B) (z : Fin m → A) : |iidLooBaseline r z| ≤ B := by
  have hmpos : (0 : ℝ) < m := by exact_mod_cast Nat.pos_of_ne_zero hm
  calc
    |iidLooBaseline r z| = |∑ i, r (z i)| / (m : ℝ) := by
      simp [iidLooBaseline, abs_div, abs_of_pos hmpos]
    _ ≤ (∑ i, |r (z i)|) / (m : ℝ) :=
      div_le_div_of_nonneg_right (Finset.abs_sum_le_sum_abs _ _) hmpos.le
    _ ≤ (∑ _i : Fin m, B) / (m : ℝ) :=
      div_le_div_of_nonneg_right (Finset.sum_le_sum fun i _ => hr (z i)) hmpos.le
    _ = B := by simp [ne_of_gt hmpos]

theorem rloo_estimator_square_bound {m : ℕ} (hm : m ≠ 0) (p r f : A → ℝ)
    (B : ℝ) (hB : 0 ≤ B) (hr : ∀ a, |r a| ≤ B) (s : Fin (m + 1) → A) :
    (iidRlooEstimator p r f s) ^ 2 ≤ 4 * B ^ 2 * sampleMean (fun a => (f a) ^ 2) s := by
  have hterm : ∀ i : Fin (m + 1), (iidLooTerm p r f i s) ^ 2 ≤ 4 * B ^ 2 * (f (s i)) ^ 2 := by
    intro i
    have ha : |r (s i) - iidLooBaseline r (fun j => s (i.succAbove j))| ≤ 2 * B :=
      (abs_sub _ _).trans (by linarith [hr (s i),
        loo_baseline_abs_bound hm r B hr (fun j => s (i.succAbove j))])
    have hsq := (sq_le_sq₀ (abs_nonneg _) (by positivity : 0 ≤ 2 * B)).mpr ha
    rw [sq_abs] at hsq
    unfold iidLooTerm
    rw [mul_pow]
    nlinarith [mul_le_mul_of_nonneg_right hsq (sq_nonneg (f (s i)))]
  calc
    _ ≤ (∑ i, (iidLooTerm p r f i s) ^ 2) / (m + 1 : ℝ) := by
      simpa [iidRlooEstimator] using
        (sum_div_card_sq_le_sum_sq_div_card
          (s := Finset.univ) (f := fun i : Fin (m + 1) => iidLooTerm p r f i s))
    _ ≤ (∑ i, 4 * B ^ 2 * (f (s i)) ^ 2) / (m + 1 : ℝ) :=
      div_le_div_of_nonneg_right (Finset.sum_le_sum fun i _ => hterm i) (by positivity)
    _ = _ := by simp only [sampleMean, ← Finset.mul_sum, Nat.cast_add, Nat.cast_one]; ring

theorem rloo_estimator_second_moment_bound {m : ℕ} (hm : m ≠ 0) (p r f : A → ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (B : ℝ) (hB : 0 ≤ B) (hr : ∀ a, |r a| ≤ B) :
    expectation (fun s : Fin (m + 1) → A => iidMass p s)
      (fun s => (iidRlooEstimator p r f s) ^ 2) ≤
      4 * B ^ 2 * expectation p (fun a => (f a) ^ 2) := by
  calc
    _ ≤ ∑ s : Fin (m + 1) → A, iidMass p s *
        (4 * B ^ 2 * sampleMean (fun a => (f a) ^ 2) s) :=
      Finset.sum_le_sum fun s _ => mul_le_mul_of_nonneg_left
        (rloo_estimator_square_bound hm p r f B hB hr s) (iid_mass_nonneg p hp0 s)
    _ = 4 * B ^ 2 * ∑ s : Fin (m + 1) → A,
        iidMass p s * sampleMean (fun a => (f a) ^ 2) s := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro s _
      ring
    _ = _ := by rw [iid_sampleMean_expectation (by omega) p _ hp]

theorem variance_le_second_moment (p f : A → ℝ) (hp : ∑ a, p a = 1) :
    expectation p (fun a => (f a - expectation p f) ^ 2) ≤
      expectation p (fun a => (f a) ^ 2) := by
  have heq : expectation p (fun a => (f a - expectation p f) ^ 2) =
      expectation p (fun a => (f a) ^ 2) - (expectation p f) ^ 2 := by
    let c := expectation p f
    have hpoint : ∀ a, p a * (f a - c) ^ 2 =
        p a * (f a) ^ 2 - 2 * c * (p a * f a) + p a * c ^ 2 := by intro a; ring
    change (∑ a, p a * (f a - c) ^ 2) = _
    simp_rw [hpoint, Finset.sum_add_distrib, Finset.sum_sub_distrib]
    rw [← Finset.mul_sum, ← Finset.sum_mul, hp]
    change expectation p (fun a => (f a) ^ 2) - 2 * c * c + 1 * c ^ 2 = _
    dsimp [c]
    ring
  rw [heq]
  linarith [sq_nonneg (expectation p f)]

theorem iid_sample_mean_tail_of_second_moment {n : ℕ} (hn : n ≠ 0) (p f : A → ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (V η : ℝ) (hη : 0 < η) (hV : expectation p (fun a => (f a) ^ 2) ≤ V) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → A =>
      η ≤ |sampleMean f s - expectation p f|), iidMass p s) ≤ V / ((n : ℝ) * η ^ 2) := by
  have hvar := finite_iid_baseline_variance hn p f hp (expectation p f) rfl
  change expectation (iidMass p) (fun s : Fin n → A =>
    (sampleMean f s - expectation p f) ^ 2) = _ at hvar
  have hmse : expectation (iidMass p) (fun s : Fin n → A =>
      (sampleMean f s - expectation p f) ^ 2) ≤ V / (n : ℝ) := by
    rw [hvar]
    exact div_le_div_of_nonneg_right ((variance_le_second_moment p f hp).trans hV)
      (Nat.cast_nonneg n)
  simpa only [div_div] using finite_squared_tail_bound (iidMass p)
    (fun s : Fin n → A => sampleMean f s - expectation p f) (iid_mass_nonneg p hp0)
    η (V / (n : ℝ)) hη hmse

end
end REINFORCEProMax.BatchReduction

namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators
open BatchReduction
noncomputable section
variable {X A : Type*} [Fintype X] [DecidableEq X] [Fintype A] [DecidableEq A]

def blockSurrogate (p q : X → Policy A) (R : X → List A → ℝ) {m T : ℕ}
    (b : RolloutBlock X A m T) : ℝ := blockIncrement p q R b / (m + 1 : ℝ)

def familyRatioEnergy (μ : X → ℝ) (p q : X → Policy A) (T : ℕ) : ℝ :=
  ∑ x, μ x * ratioEnergy (p x) (q x) T

theorem block_surrogate_expectation {m T : ℕ} (hm : m ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0) :
    expectation (blockMass μ p m T) (blockSurrogate p q R) = familySurrogate μ p q R T := by
  unfold blockSurrogate expectation
  simp only [← mul_div_assoc, ← Finset.sum_div]
  change expectation (blockMass μ p m T) (blockIncrement p q R) / (m + 1 : ℝ) = _
  rw [block_increment_expectation hm μ p q R hs]
  exact mul_div_cancel_left₀ _ (by positivity)

theorem block_surrogate_second_moment {m T : ℕ} (hm : m ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (B : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0) :
    expectation (blockMass μ p m T) (fun b => (blockSurrogate p q R b) ^ 2) ≤
      4 * B ^ 2 * familyRatioEnergy μ p q T := by
  unfold expectation blockMass blockSurrogate blockIncrement groupSurrogateIncrement
  simp only [mul_div_cancel_left₀ _ (by positivity : (m + 1 : ℝ) ≠ 0)]
  rw [Fintype.sum_prod_type]
  simp_rw [mul_assoc, ← Finset.mul_sum]
  change (∑ x, μ x * expectation (fun s : Fin (m + 1) → (Fin T → A) =>
    iidMass (rolloutMass (p x) T) s) (fun s =>
      (iidRlooEstimator (rolloutMass (p x) T) (fun y => R x (List.ofFn y))
        (responseRatioResidual (p x) (q x) T) s) ^ 2)) ≤ _
  calc
    _ ≤ ∑ x, μ x * (4 * B ^ 2 * ratioEnergy (p x) (q x) T) := by
      apply Finset.sum_le_sum
      intro x _
      apply mul_le_mul_of_nonneg_left _ (hμ0 x)
      have hb := rloo_estimator_second_moment_bound hm (rolloutMass (p x) T)
        (fun y => R x (List.ofFn y)) (responseRatioResidual (p x) (q x) T)
        (fun y => pathMass_nonneg _ _ _) (rolloutMass_normalized _ T) B hB (fun y => hR x _)
      rwa [response_residual_energy _ _ (hs x)] at hb
    _ = _ := by
      unfold familyRatioEnergy
      simp only [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro x _
      ring

/-- The observed mean token count rescales the token objective to the mean
per-response RLOO contribution, exactly even for random response lengths. -/
theorem observed_token_rescaling {n m T : ℕ}
    (p q : X → Policy A) (R L : X → List A → ℝ)
    (s : Fin n → RolloutBlock X A m T)
    (hL : sampleMean (blockLength L) s ≠ 0) :
    (sampleMean (blockLength L) s / (m + 1 : ℝ)) *
      (sampleMean (blockIncrement p q R) s / sampleMean (blockLength L) s) =
        sampleMean (blockSurrogate p q R) s := by
  have hmean : sampleMean (blockSurrogate p q R) s =
      sampleMean (blockIncrement p q R) s / (m + 1 : ℝ) := by
    simp only [sampleMean, blockSurrogate, ← Finset.sum_div]
    ring
  rw [hmean]
  field_simp
  ring

theorem corrected_objective_rescaling {n m T : ℕ} (hn : n ≠ 0)
    (p q : X → Policy A) (R L : X → List A → ℝ)
    (s : Fin n → RolloutBlock X A m T)
    (hL : sampleMean (blockLength L) s ≠ 0)
    (mask M w r r₀ c c₀ H : (Fin n × Fin (m + 1) × Fin T) → ℝ)
    (hw : ∀ k, r k * w k = responseTokenRatio (p (s k.1).1) (q (s k.1).1)
      ((s k.1).2 k.2.1) k.2.2)
    (hw₀ : ∀ k, r₀ k * w k = 1)
    (hpad : ∀ k, mask k *
      (responseTokenRatio (p (s k.1).1) (q (s k.1).1) ((s k.1).2 k.2.1) k.2.2 - 1) =
      responseTokenRatio (p (s k.1).1) (q (s k.1).1) ((s k.1).2 k.2.1) k.2.2 - 1) :
    (sampleMean (blockLength L) s / (m + 1 : ℝ)) *
      correctedBatchChange (fun k => mask k / ∑ b, blockLength L (s b)) M w r r₀ c c₀
        (fun k => groupAdvantage (R (s k.1).1) (s k.1).2 k.2.1) H =
      sampleMean (blockSurrogate p q R) s := by
  rw [corrected_block_objective_eq hn p q R L s mask M w r r₀ c c₀ H hw hw₀ hpad]
  exact observed_token_rescaling p q R L s hL

theorem rescaled_surrogate_unbiased {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hμ : ∑ x, μ x = 1)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0) :
    expectation (fun s : Fin n → RolloutBlock X A m T => iidMass (blockMass μ p m T) s)
      (sampleMean (blockSurrogate p q R)) = familySurrogate μ p q R T := by
  unfold expectation
  rw [iid_sampleMean_expectation hn _ _ (block_mass_normalized μ p hμ m T),
    block_surrogate_expectation hm μ p q R hs]

theorem family_ratio_energy_horizon_bound (μ : X → ℝ) (p q : X → Policy A)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1) (D : ℝ) (T : ℕ)
    (hlocal : ∀ x h, h.length < T → localRatioEnergy (p x) (q x) h ≤ D) :
    familyRatioEnergy μ p q T ≤ (T : ℝ) * D := by
  calc
    _ ≤ ∑ x, μ x * ((T : ℝ) * D) := Finset.sum_le_sum fun x _ =>
      mul_le_mul_of_nonneg_left (ratio_energy_horizon_bound _ _ D T (hlocal x)) (hμ0 x)
    _ = _ := by rw [← Finset.sum_mul, hμ, one_mul]

theorem rescaled_surrogate_uniform_tail {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
    {J : Type*} [Fintype J]
    (μ : X → ℝ) (p : X → Policy A) (q : J → X → Policy A)
    (R : X → List A → ℝ) (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B : ℝ) (η : J → ℝ) (hB : 0 ≤ B) (hη : ∀ j, 0 < η j)
    (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ j x h a, (p x).prob h a = 0 → (q j x).prob h a = 0) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T => ∃ j,
      η j ≤ |sampleMean (blockSurrogate p (q j) R) s - familySurrogate μ p (q j) R T|),
        iidMass (blockMass μ p m T) s) ≤
      ∑ j, (4 * B ^ 2 * familyRatioEnergy μ p (q j) T) / ((n : ℝ) * (η j) ^ 2) := by
  classical
  apply (finite_event_union_bound (iidMass (blockMass μ p m T))
    (iid_mass_nonneg _ (block_mass_nonneg μ p hμ0 m T)) _).trans
  apply Finset.sum_le_sum
  intro j _
  have hb := iid_sample_mean_tail_of_second_moment hn (blockMass μ p m T)
    (blockSurrogate p (q j) R) (block_mass_nonneg μ p hμ0 m T)
    (block_mass_normalized μ p hμ m T) (4 * B ^ 2 * familyRatioEnergy μ p (q j) T)
    (η j) (hη j) (block_surrogate_second_moment hm μ p (q j) R hμ0 B hB hR (hs j))
  rwa [block_surrogate_expectation hm μ p (q j) R (hs j)] at hb

theorem family_ratio_energy_of_ratio_bound (μ : X → ℝ) (p q : X → Policy A)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1) (d : ℝ) (hd : 0 ≤ d) (T : ℕ)
    (hr : ∀ x h, h.length < T → ∀ a, 0 < (p x).prob h a →
      |tokenRatio (p x) (q x) h a - 1| ≤ d) :
    familyRatioEnergy μ p q T ≤ (T : ℝ) * d ^ 2 := by
  apply family_ratio_energy_horizon_bound μ p q hμ0 hμ (d ^ 2) T
  intro x h hh
  exact local_ratio_energy_of_bound (p x) (q x) d hd h (hr x h hh)

theorem rescaled_performance_from_estimate (μ : X → ℝ) (p q : X → Policy A)
    (R : X → List A → ℝ) (T : ℕ) (hμ0 : ∀ x, 0 ≤ μ x)
    (B ε δ η estimate : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 ↔ (q x).prob h a = 0)
    (hTV : ∀ x h, h.length < T → localTV (p x) (q x) h ≤ ε)
    (hKL : ∀ x h, h.length < T → localReverseKL (p x) (q x) h ≤ δ)
    (hest : |estimate - familySurrogate μ p q R T| ≤ η) :
    estimate - η - familyAdaptiveCost μ p q B ε δ T ≤
      familyValue μ q R T - familyValue μ p R T := by
  have hb := family_performance_lower_bound_adaptive μ p q R hμ0 B ε δ hB hR T hs hTV hKL
  rw [← family_adaptive_cost] at hb
  change familySurrogate μ p q R T - familyAdaptiveCost μ p q B ε δ T ≤ _ at hb
  have he := (abs_le.mp hest).2
  linarith

/-- Selection may depend on all evaluation data, within the family fixed
before evaluation. The failure probability uses actual RLOO block moments. -/
theorem rescaled_selected_false_improvement_bound {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
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
      ∑ j, (4 * B ^ 2 * familyRatioEnergy μ p (q j) T) / ((n : ℝ) * (η j) ^ 2) := by
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
    (rescaled_surrogate_uniform_tail hn hm μ p q R hμ0 hμ B η hB hη hR
      (fun j x h a => (hs j x h a).mp))

end
end REINFORCEProMax.AutoregressiveTree
