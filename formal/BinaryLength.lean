import BinaryUpdate

/-!
Arbitrary response lengths in the on-policy binary update. Conditioning on
the complete sibling responses gives a positive reward-gradient component
and an explicit within-reward-class coefficient/score covariance. Length,
normalization, and the random token denominator all remain in the coefficient.
-/
namespace REINFORCEProMax.BinaryLength
open scoped BigOperators Topology
open BatchReduction UniformScale BinaryUpdate NormalizationSafeguards
noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

def classWeight (p : A → ℝ) (b : A → Bool) (q : Bool) (a : A) : ℝ :=
  if b a = q then p a else 0

def classMass (p : A → ℝ) (b : A → Bool) (q : Bool) : ℝ :=
  ∑ a, classWeight p b q a

def classMean (p : A → ℝ) (b : A → Bool) (q : Bool) (f : A → ℝ) : ℝ :=
  (∑ a, classWeight p b q a * f a) / classMass p b q

def center (p : A → ℝ) (b : A → Bool) (f : A → ℝ) (a : A) : ℝ :=
  f a - classMean p b (b a) f

theorem class_weight_nonnegative (p : A → ℝ) (b : A → Bool)
    (hp : ∀ a, 0 ≤ p a) (q : Bool) (a : A) : 0 ≤ classWeight p b q a := by
  unfold classWeight
  split_ifs
  · exact hp a
  · exact le_rfl

theorem class_mean_centered (p f : A → ℝ) (b : A → Bool) (q : Bool)
    (hq : classMass p b q ≠ 0) :
    (∑ a, classWeight p b q a * (f a - classMean p b q f)) = 0 := by
  simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul]
  change (∑ a, classWeight p b q a * f a) - classMass p b q *
    ((∑ a, classWeight p b q a * f a) / classMass p b q) = 0
  field_simp

theorem class_center_sum_zero (p f : A → ℝ) (b : A → Bool) (q : Bool)
    (hq : classMass p b q ≠ 0) :
    (∑ a, classWeight p b q a * center p b f a) = 0 := by
  have heq : ∀ a, classWeight p b q a * center p b f a =
      classWeight p b q a * (f a - classMean p b q f) := by
    intro a
    by_cases h : b a = q
    · simp [center, h]
    · simp [classWeight, h]
  simp_rw [heq]
  exact class_mean_centered p f b q hq

theorem split_by_class (p f : A → ℝ) (b : A → Bool) :
    (∑ a, p a * f a) =
      (∑ a, classWeight p b true a * f a) + (∑ a, classWeight p b false a * f a) := by
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro a _
  cases h : b a <;> simp [classWeight, h]

theorem center_times_class_mean_zero (p c f : A → ℝ) (b : A → Bool)
    (hmass : ∀ q, classMass p b q ≠ 0) :
    (∑ a, p a * center p b c a * classMean p b (b a) f) = 0 := by
  have heq : ∀ a, p a * center p b c a * classMean p b (b a) f =
      classWeight p b true a * center p b c a * classMean p b true f +
      classWeight p b false a * center p b c a * classMean p b false f := by
    intro a
    cases h : b a <;> simp [classWeight, h]
  simp_rw [heq]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul, ← Finset.sum_mul,
    class_center_sum_zero p c b true (hmass true), class_center_sum_zero p c b false (hmass false)]
  ring

/-- This covariance identity retains all within-class dependence. -/
theorem class_covariance_decomposition (p c f : A → ℝ) (b : A → Bool)
    (hmass : ∀ q, classMass p b q ≠ 0) (hf : ∑ a, p a * f a = 0) :
    (∑ a, p a * c a * f a) =
      (classMean p b true c - classMean p b false c) * successScore p f b +
        (∑ a, p a * center p b c a * center p b f a) := by
  have hid : ∀ a, p a * c a * f a =
      p a * (if b a then classMean p b true c else classMean p b false c) * f a +
      p a * center p b c a * center p b f a +
      p a * center p b c a * classMean p b (b a) f := by
    intro a
    cases h : b a <;> simp only [h, Bool.false_eq_true, ↓reduceIte, center] <;> ring
  simp_rw [hid]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib,
    binary_score_pair p f b hf, center_times_class_mean_zero p c f b hmass, add_zero]

def coefficientSlice {m : ℕ}
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ)
    (i : Fin (m + 1)) (z : Fin m → A) (a : A) : ℝ := C (Fin.insertNth i a z) i

def classMultiplier {m : ℕ} (p : A → ℝ) (b : A → Bool)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) : ℝ :=
  ∑ i, ∑ z : Fin m → A, iidMass p z *
    (classMean p b true (coefficientSlice C i z) - classMean p b false (coefficientSlice C i z))

def covarianceError {m : ℕ} (p f : A → ℝ) (b : A → Bool)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) : ℝ :=
  ∑ i, ∑ z : Fin m → A, iidMass p z *
    (∑ a, p a * center p b (coefficientSlice C i z) a * center p b f a)

theorem general_coordinate_expectation {m : ℕ} (p f : A → ℝ) (b : A → Bool)
    (hmass : ∀ q, classMass p b q ≠ 0) (hf : ∑ a, p a * f a = 0)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) (i : Fin (m + 1)) :
    (∑ y : Fin (m + 1) → A, iidMass p y * (C y i * f (y i))) =
      (∑ z : Fin m → A, iidMass p z *
        (classMean p b true (coefficientSlice C i z) - classMean p b false (coefficientSlice C i z))) * successScore p f b +
      (∑ z : Fin m → A, iidMass p z *
        (∑ a, p a * center p b (coefficientSlice C i z) a * center p b f a)) := by
  rw [iid_expect_insertNth p _ i, Finset.sum_comm]
  simp only [Fin.insertNth_apply_same]
  have hlocal : ∀ z : Fin m → A, (∑ a, p a * iidMass p z * (C (Fin.insertNth i a z) i * f a)) =
      iidMass p z * (∑ a, p a * coefficientSlice C i z a * f a) := by
    intro z
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro a _
    unfold coefficientSlice
    ring
  simp_rw [hlocal, class_covariance_decomposition p _ f b hmass hf, mul_add, ← mul_assoc]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul]

theorem general_update_decomposition {m : ℕ} (p f : A → ℝ) (b : A → Bool)
    (hmass : ∀ q, classMass p b q ≠ 0) (hf : ∑ a, p a * f a = 0)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) :
    (∑ y : Fin (m + 1) → A, iidMass p y * (∑ i, C y i * f (y i))) =
      classMultiplier p b C * successScore p f b + covarianceError p f b C := by
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  simp_rw [general_coordinate_expectation p f b hmass hf C]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul]
  rfl

def coefficientVariance {m : ℕ} (p : A → ℝ) (b : A → Bool)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) : ℝ :=
  ∑ i, ∑ z : Fin m → A, iidMass p z *
    (∑ a, p a * (center p b (coefficientSlice C i z) a) ^ 2)

def scoreVariance (p f : A → ℝ) (b : A → Bool) : ℝ :=
  ∑ a, p a * (center p b f a) ^ 2

theorem covariance_error_square_bound {m : ℕ} (p f : A → ℝ) (b : A → Bool)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) :
    (covarianceError p f b C) ^ 2 ≤
      coefficientVariance p b C * ((m + 1 : ℝ) * scoreVariance p f b) := by
  let w := fun t : Fin (m + 1) × ((Fin m → A) × A) => iidMass p t.2.1 * p t.2.2
  let u := fun t : Fin (m + 1) × ((Fin m → A) × A) => center p b (coefficientSlice C t.1 t.2.1) t.2.2
  let v := fun t : Fin (m + 1) × ((Fin m → A) × A) => center p b f t.2.2
  have hw : ∀ t, 0 ≤ w t := fun t => mul_nonneg (iid_mass_nonneg p hp0 t.2.1) (hp0 t.2.2)
  have hcs := Finset.sum_sq_le_sum_mul_sum_of_sq_eq_mul
    (s := Finset.univ) (r := fun t => w t * u t * v t)
    (f := fun t => w t * (u t) ^ 2) (g := fun t => w t * (v t) ^ 2)
    (fun t _ => mul_nonneg (hw t) (sq_nonneg _))
    (fun t _ => mul_nonneg (hw t) (sq_nonneg _)) (fun t _ => by ring)
  have hscore : (∑ t, w t * (v t) ^ 2) = (m + 1 : ℝ) * scoreVariance p f b := by
    simp only [w, v, Fintype.sum_prod_type, mul_assoc, ← Finset.mul_sum]
    rw [← Finset.sum_mul, iid_mass_normalized p hp m, one_mul]
    simp [scoreVariance]
  rw [hscore] at hcs
  simpa only [w, u, v, Fintype.sum_prod_type, covarianceError, coefficientVariance,
    mul_assoc, ← Finset.mul_sum] using hcs

theorem covariance_error_abs_bound {m : ℕ} (p f : A → ℝ) (b : A → Bool)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) :
    |covarianceError p f b C| ≤
      Real.sqrt (coefficientVariance p b C * ((m + 1 : ℝ) * scoreVariance p f b)) := by
  have h := covariance_error_square_bound p f b hp0 hp C
  have hn := (sq_nonneg (covarianceError p f b C)).trans h
  have hs := Real.sq_sqrt hn
  have ha := Real.sqrt_nonneg (coefficientVariance p b C * ((m + 1 : ℝ) * scoreVariance p f b))
  rw [abs_le]
  constructor <;> nlinarith

theorem class_mean_lower_bound (p c : A → ℝ) (b : A → Bool)
    (hp0 : ∀ a, 0 ≤ p a) (q : Bool) (hq : 0 < classMass p b q) (d : ℝ)
    (hc : ∀ a, b a = q → d ≤ c a) : d ≤ classMean p b q c := by
  apply (le_div_iff₀ hq).mpr
  calc
    d * classMass p b q = ∑ a, classWeight p b q a * d := by
      unfold classMass
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro a _
      ring
    _ ≤ _ := by
      apply Finset.sum_le_sum
      intro a _
      by_cases h : b a = q
      · exact mul_le_mul_of_nonneg_left (hc a h) (class_weight_nonnegative p b hp0 q a)
      · simp [classWeight, h]

theorem class_mean_upper_bound (p c : A → ℝ) (b : A → Bool)
    (hp0 : ∀ a, 0 ≤ p a) (q : Bool) (hq : 0 < classMass p b q) (d : ℝ)
    (hc : ∀ a, b a = q → c a ≤ d) : classMean p b q c ≤ d := by
  apply (div_le_iff₀ hq).mpr
  calc
    _ ≤ ∑ a, classWeight p b q a * d := by
      apply Finset.sum_le_sum
      intro a _
      by_cases h : b a = q
      · exact mul_le_mul_of_nonneg_left (hc a h) (class_weight_nonnegative p b hp0 q a)
      · simp [classWeight, h]
    _ = d * classMass p b q := by
      unfold classMass
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro a _
      ring

theorem class_multiplier_lower_bound {m : ℕ} (p : A → ℝ) (b : A → Bool)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) (d : ℝ)
    (hc : ∀ i z, d ≤ classMean p b true (coefficientSlice C i z) -
      classMean p b false (coefficientSlice C i z)) :
    (m + 1 : ℝ) * d ≤ classMultiplier p b C := by
  have hi : ∀ i : Fin (m + 1), d ≤ ∑ z : Fin m → A, iidMass p z *
      (classMean p b true (coefficientSlice C i z) - classMean p b false (coefficientSlice C i z)) := by
    intro i
    calc
      d = ∑ z : Fin m → A, iidMass p z * d := by rw [← Finset.sum_mul, iid_mass_normalized p hp]; ring
      _ ≤ _ := Finset.sum_le_sum fun z _ => mul_le_mul_of_nonneg_left (hc i z) (iid_mass_nonneg p hp0 z)
  simpa [classMultiplier] using Finset.sum_le_sum (s := Finset.univ) (fun i _ => hi i)

def responseFactors {m : ℕ} (b : A → Bool) (length : A → ℝ)
    (ε cap maxScale : ℝ) (y : Fin (m + 1) → A) : ℝ × ℝ :=
  let w := fun i => length (y i)
  let a := binaryRloo (fun j => b (y j))
  let P := AdaptiveMechanism.positiveTotal w a
  let N := AdaptiveMechanism.negativeTotal w a
  safeguardedFactors (decide (P < ε ∨ N < ε)) P N
    (AdaptiveMechanism.positiveSquare w a) (AdaptiveMechanism.negativeSquare w a)
    (AdaptiveMechanism.nonzeroMass w a) ε cap maxScale

def responseDenominator {m : ℕ} (length : A → ℝ) (y : Fin (m + 1) → A) : ℝ :=
  ∑ i, length (y i)

def responseCoefficient {m : ℕ} (uniform : Bool) (b : A → Bool) (length : A → ℝ)
    (ε cap maxScale : ℝ) (y : Fin (m + 1) → A) (i : Fin (m + 1)) : ℝ :=
  let factors := responseFactors b length ε cap maxScale y
  signScale factors.1 factors.2 (binaryRloo (fun j => b (y j)) i) / responseDenominator length y +
    if uniform = true ∧ ∀ j, b (y j) = true then 1 / ((m + 1 : ℝ) * responseDenominator length y) else 0

theorem response_denominator_bounds {m : ℕ} (length : A → ℝ) (Lmax : ℝ)
    (hpos : ∀ a, 0 < length a) (hmax : ∀ a, length a ≤ Lmax) (y : Fin (m + 1) → A) :
    0 < responseDenominator length y ∧ responseDenominator length y ≤ (m + 1 : ℝ) * Lmax := by
  constructor
  · exact Finset.sum_pos (fun i _ => hpos (y i)) Finset.univ_nonempty
  · simpa [responseDenominator] using Finset.sum_le_sum (s := Finset.univ) (fun i _ => hmax (y i))

/-- Bounds on the actual two counterfactual coefficients require no
class-dependent-length assumption. -/
theorem response_slice_bounds {m : ℕ} (hm : m ≠ 0)
    (uniform : Bool) (b : A → Bool) (length : A → ℝ) (ε cap maxScale Lmax : ℝ)
    (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale)
    (hpos : ∀ a, 0 < length a) (hmax : ∀ a, length a ≤ Lmax)
    (i : Fin (m + 1)) (z : Fin m → A) (a : A) :
    (b a = true → ε * (1 - iidLooBaseline bit (fun j => b (z j))) / ((m + 1 : ℝ) * Lmax) ≤
      coefficientSlice (responseCoefficient (m := m) uniform b length ε cap maxScale) i z a) ∧
    (b a = false → coefficientSlice (responseCoefficient (m := m) uniform b length ε cap maxScale) i z a ≤
      -(ε * iidLooBaseline bit (fun j => b (z j)) / ((m + 1 : ℝ) * Lmax))) := by
  let y : Fin (m + 1) → A := Fin.insertNth i a z
  let w := fun j : Fin (m + 1) => length (y j)
  let aa := fun j : Fin (m + 1) => binaryRloo (fun k => b (y k)) j
  let PP := AdaptiveMechanism.positiveTotal w aa
  let NN := AdaptiveMechanism.negativeTotal w aa
  let QQp := AdaptiveMechanism.positiveSquare w aa
  let QQn := AdaptiveMechanism.negativeSquare w aa
  let MM := AdaptiveMechanism.nonzeroMass w aa
  have hfac := safeguarded_factors_lower_bound (decide (PP < ε ∨ NN < ε))
    PP NN QQp QQn MM ε cap maxScale hε1 hεK
  change ε ≤ (responseFactors b length ε cap maxScale y).1 ∧
    ε ≤ (responseFactors b length ε cap maxScale y).2 at hfac
  obtain ⟨hD, hDmax⟩ := response_denominator_bounds length Lmax hpos hmax y
  obtain ⟨hb0, hb1⟩ := sibling_mean_mem hm (fun j => b (z j))
  have hraw : binaryRloo (fun j => b (y j)) i = bit (b a) - iidLooBaseline bit (fun j => b (z j)) := by
    dsimp [y]
    simp [binaryRloo]
  have hboost : 0 ≤ if uniform = true ∧ ∀ j, b (y j) = true then
      1 / ((m + 1 : ℝ) * responseDenominator length y) else 0 := by
    split_ifs <;> positivity
  constructor
  · intro ha
    have hscale := positive_scaled_lower_bound _ ε _ _ ((m + 1 : ℝ) * Lmax)
      hε hfac.1 (sub_nonneg.mpr hb1) hD hDmax
    change _ ≤ signScale _ _ (binaryRloo (fun j => b (y j)) i) / _ + _
    rw [hraw, ha]
    simp only [bit, ↓reduceIte]
    rw [signScale_of_nonneg _ _ _ (sub_nonneg.mpr hb1)]
    exact hscale.trans (le_add_of_nonneg_right hboost)
  · intro ha
    have hfalse : ¬ (uniform = true ∧ ∀ j, b (y j) = true) := by
      intro h
      have hi := h.2 i
      simpa [y, ha] using hi
    have hscale := positive_scaled_lower_bound _ ε _ _ ((m + 1 : ℝ) * Lmax)
      hε hfac.2 hb0 hD hDmax
    change signScale _ _ (binaryRloo (fun j => b (y j)) i) / _ + _ ≤ _
    rw [hraw, ha, if_neg hfalse]
    simp only [bit, Bool.false_eq_true, ↓reduceIte, zero_sub, add_zero]
    rw [signScale_of_nonpos _ _ _ (neg_nonpos.mpr hb0)]
    convert neg_le_neg hscale using 1 <;> ring

theorem response_class_contrast_lower_bound {m : ℕ} (hm : m ≠ 0)
    (p : A → ℝ) (b : A → Bool) (uniform : Bool) (length : A → ℝ)
    (ε cap maxScale Lmax : ℝ) (hp0 : ∀ a, 0 ≤ p a)
    (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale)
    (hpos : ∀ a, 0 < length a) (hmax : ∀ a, length a ≤ Lmax)
    (hmass : ∀ q, 0 < classMass p b q)
    (i : Fin (m + 1)) (z : Fin m → A) :
    ε / ((m + 1 : ℝ) * Lmax) ≤
      classMean p b true (coefficientSlice (responseCoefficient (m := m) uniform b length ε cap maxScale) i z) -
      classMean p b false (coefficientSlice (responseCoefficient (m := m) uniform b length ε cap maxScale) i z) := by
  let u := iidLooBaseline bit (fun j => b (z j))
  let d := ε / ((m + 1 : ℝ) * Lmax)
  have hu0 : 0 ≤ u := (sibling_mean_mem hm (fun j => b (z j))).1
  have hu1 : u ≤ 1 := (sibling_mean_mem hm (fun j => b (z j))).2
  have htrue : d * (1 - u) ≤
      classMean p b true (coefficientSlice (responseCoefficient (m := m) uniform b length ε cap maxScale) i z) := by
    apply class_mean_lower_bound p _ b hp0 true (hmass true) (d * (1 - u))
    intro a ha
    convert (response_slice_bounds (m := m) (hm := hm) (uniform := uniform) (b := b)
      (length := length) (ε := ε) (cap := cap) (maxScale := maxScale) (Lmax := Lmax)
      (hε := hε) (hε1 := hε1) (hεK := hεK) (hpos := hpos) (hmax := hmax)
      (i := i) (z := z) (a := a)).1 ha using 1 <;> dsimp [d, u] <;> ring
  have hfalse :
      classMean p b false (coefficientSlice (responseCoefficient (m := m) uniform b length ε cap maxScale) i z) ≤ -d * u := by
    apply class_mean_upper_bound p _ b hp0 false (hmass false) (-d * u)
    intro a ha
    convert (response_slice_bounds (m := m) (hm := hm) (uniform := uniform) (b := b)
      (length := length) (ε := ε) (cap := cap) (maxScale := maxScale) (Lmax := Lmax)
      (hε := hε) (hε1 := hε1) (hεK := hεK) (hpos := hpos) (hmax := hmax)
      (i := i) (z := z) (a := a)).2 ha using 1 <;> dsimp [d, u] <;> ring
  dsimp [d, u] at htrue hfalse ⊢
  linarith

theorem response_class_multiplier_lower_bound {m : ℕ} (hm : m ≠ 0)
    (p : A → ℝ) (b : A → Bool) (uniform : Bool) (length : A → ℝ)
    (ε cap maxScale Lmax : ℝ) (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale)
    (hpos : ∀ a, 0 < length a) (hmax : ∀ a, length a ≤ Lmax)
    (hmass : ∀ q, 0 < classMass p b q) (hL : 0 < Lmax) :
    ε / Lmax ≤ classMultiplier p b (responseCoefficient (m := m) uniform b length ε cap maxScale) := by
  have h := class_multiplier_lower_bound p b hp0 hp
    (responseCoefficient (m := m) uniform b length ε cap maxScale) (ε / ((m + 1 : ℝ) * Lmax))
    (fun i z => response_class_contrast_lower_bound (m := m) (hm := hm) (p := p) (b := b)
      (uniform := uniform) (length := length) (ε := ε) (cap := cap) (maxScale := maxScale)
      (hp0 := hp0) (hε := hε) (hε1 := hε1) (hεK := hεK) (hpos := hpos) (hmax := hmax)
      (hmass := hmass) (i := i) (z := z))
  have hn : (m + 1 : ℝ) ≠ 0 := by positivity
  have heq : (m + 1 : ℝ) * (ε / ((m + 1 : ℝ) * Lmax)) = ε / Lmax := by
    field_simp [hn, ne_of_gt hL]
    ring
  rw [heq] at h
  exact h

theorem response_update_decomposition {m : ℕ} (p f : A → ℝ) (b : A → Bool)
    (hmass : ∀ q, classMass p b q ≠ 0) (hf : ∑ a, p a * f a = 0)
    (uniform : Bool) (length : A → ℝ) (ε cap maxScale : ℝ) :
    (∑ y : Fin (m + 1) → A, iidMass p y *
      (∑ i, responseCoefficient (m := m) uniform b length ε cap maxScale y i * f (y i))) =
      classMultiplier p b (responseCoefficient (m := m) uniform b length ε cap maxScale) * successScore p f b +
      covarianceError p f b (responseCoefficient (m := m) uniform b length ε cap maxScale) :=
  general_update_decomposition (m := m) p f b hmass hf
    (responseCoefficient (m := m) uniform b length ε cap maxScale)

end
end REINFORCEProMax.BinaryLength
