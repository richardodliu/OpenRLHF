import UniformScale
import NormalizationSafeguards
import AdaptiveMechanism

/-!
Population direction of a complete binary-reward update with label-dependent
coefficients. Coefficients include the random token denominator and positive
normalization safeguards. No independence between coefficients and scores is
assumed: each coordinate is integrated together with its own reward label.
-/
namespace REINFORCEProMax.BinaryUpdate
open scoped BigOperators Topology
open BatchReduction UniformScale NormalizationSafeguards
noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

theorem map_insert {m : ℕ} (b : A → Bool) (i : Fin (m + 1))
    (a : A) (z : Fin m → A) :
    (fun j : Fin (m + 1) => b ((Fin.insertNth i a z : Fin (m + 1) → A) j)) =
      Fin.insertNth i (b a) (fun j => b (z j)) := by
  apply Fin.eq_insertNth_iff.mpr
  constructor
  · simp
  · ext j
    simp [Fin.removeNth]

theorem binary_score_pair (p f : A → ℝ) (b : A → Bool)
    (hf : ∑ a, p a * f a = 0) (hi lo : ℝ) :
    (∑ a, p a * (if b a then hi else lo) * f a) =
      (hi - lo) * successScore p f b := by
  have hpoint : ∀ a, p a * (if b a then hi else lo) * f a =
      hi * (restrictedMass p b a * f a) +
      lo * (restrictedMass p (fun x => !(b x)) a * f a) := by
    intro a
    cases hb : b a <;> simp [restrictedMass, hb] <;> ring
  simp_rw [hpoint]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  change hi * successScore p f b + lo * successScore p f (fun x => !(b x)) = _
  rw [complement_score p f b hf]
  ring

def contrast {m : ℕ} (H : (Fin (m + 1) → Bool) → Fin (m + 1) → ℝ)
    (i : Fin (m + 1)) (z : Fin m → Bool) : ℝ :=
  H (Fin.insertNth i true z) i - H (Fin.insertNth i false z) i

def multiplier {m : ℕ} (p : A → ℝ) (b : A → Bool)
    (H : (Fin (m + 1) → Bool) → Fin (m + 1) → ℝ) : ℝ :=
  ∑ i, ∑ z : Fin m → A, iidMass p z * contrast H i (fun j => b (z j))

theorem coordinate_expectation {m : ℕ} (p f : A → ℝ) (b : A → Bool)
    (hf : ∑ a, p a * f a = 0)
    (H : (Fin (m + 1) → Bool) → Fin (m + 1) → ℝ) (i : Fin (m + 1)) :
    (∑ y : Fin (m + 1) → A, iidMass p y * (H (fun j => b (y j)) i * f (y i))) =
      (∑ z : Fin m → A, iidMass p z * contrast H i (fun j => b (z j))) *
        successScore p f b := by
  rw [iid_expect_insertNth p _ i, Finset.sum_comm]
  simp only [Fin.insertNth_apply_same, map_insert]
  have hterm : ∀ z : Fin m → A,
      (∑ a, p a * iidMass p z *
        (H (Fin.insertNth i (b a) (fun j => b (z j))) i * f a)) =
      iidMass p z * contrast H i (fun j => b (z j)) * successScore p f b := by
    intro z
    have heq : ∀ a, H (Fin.insertNth i (b a) (fun j => b (z j))) i =
        if b a then H (Fin.insertNth i true (fun j => b (z j))) i
        else H (Fin.insertNth i false (fun j => b (z j))) i := by
      intro a; cases hb : b a <;> simp [hb]
    simp_rw [heq]
    calc
      _ = iidMass p z * (∑ a, p a *
          (if b a then H (Fin.insertNth i true (fun j => b (z j))) i
          else H (Fin.insertNth i false (fun j => b (z j))) i) * f a) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro a _; ring
      _ = _ := by rw [binary_score_pair p f b hf]; simp only [contrast]; ring
  simp_rw [hterm]
  rw [Finset.sum_mul]

/-- The complete group expectation, including any label-dependent reduction. -/
theorem complete_update_expectation {m : ℕ} (p f : A → ℝ) (b : A → Bool)
    (hf : ∑ a, p a * f a = 0)
    (H : (Fin (m + 1) → Bool) → Fin (m + 1) → ℝ) :
    (∑ y : Fin (m + 1) → A,
      iidMass p y * (∑ i, H (fun j => b (y j)) i * f (y i))) =
      multiplier p b H * successScore p f b := by
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  simp_rw [coordinate_expectation p f b hf H]
  exact (Finset.sum_mul ..).symm

theorem multiplier_lower_bound {m : ℕ} (p : A → ℝ) (b : A → Bool)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (H : (Fin (m + 1) → Bool) → Fin (m + 1) → ℝ) (d : ℝ)
    (hH : ∀ i z, d ≤ contrast H i z) :
    (m + 1 : ℝ) * d ≤ multiplier p b H := by
  have hlocal : ∀ i : Fin (m + 1), d ≤
      ∑ z : Fin m → A, iidMass p z * contrast H i (fun j => b (z j)) := by
    intro i
    calc
      d = ∑ z : Fin m → A, iidMass p z * d := by rw [← Finset.sum_mul, iid_mass_normalized p hp]; ring
      _ ≤ _ := Finset.sum_le_sum fun z _ => mul_le_mul_of_nonneg_left (hH i _) (iid_mass_nonneg p hp0 z)
  have hsum := Finset.sum_le_sum (s := Finset.univ) (fun i _ => hlocal i)
  simpa [multiplier] using hsum

theorem signed_coefficient_contrast {m : ℕ}
    (H : (Fin (m + 1) → Bool) → Fin (m + 1) → ℝ)
    (hhi : ∀ z i, z i = true → 0 ≤ H z i)
    (hlo : ∀ z i, z i = false → H z i ≤ 0) (i : Fin (m + 1)) (z : Fin m → Bool) :
    0 ≤ contrast H i z := by
  exact sub_nonneg.mpr ((hlo _ i (by simp)).trans (hhi _ i (by simp)))

def bit (z : Bool) : ℝ := if z then 1 else 0

theorem bit_mem (z : Bool) : 0 ≤ bit z ∧ bit z ≤ 1 := by cases z <;> norm_num [bit]

theorem sibling_mean_mem {m : ℕ} (hm : m ≠ 0) (z : Fin m → Bool) :
    0 ≤ iidLooBaseline bit z ∧ iidLooBaseline bit z ≤ 1 := by
  have hmpos : (0 : ℝ) < m := by exact_mod_cast Nat.pos_of_ne_zero hm
  constructor
  · exact div_nonneg (Finset.sum_nonneg fun i _ => (bit_mem (z i)).1) (le_of_lt hmpos)
  · apply (div_le_iff₀ hmpos).mpr
    simpa using Finset.sum_le_sum (s := Finset.univ) (fun i _ => (bit_mem (z i)).2)

def binaryRloo {m : ℕ} (z : Fin (m + 1) → Bool) (i : Fin (m + 1)) : ℝ :=
  bit (z i) - iidLooBaseline bit (fun j => z (i.succAbove j))

theorem binary_rloo_insert {m : ℕ} (i : Fin (m + 1)) (q : Bool) (z : Fin m → Bool) :
    binaryRloo (Fin.insertNth i q z) i = bit q - iidLooBaseline bit z := by
  simp [binaryRloo]

theorem signScale_of_nonneg (α β x : ℝ) (hx : 0 ≤ x) : signScale α β x = α * x := by
  rcases eq_or_lt_of_le hx with h | h
  · rw [← h]; simp [signScale]
  · simp [signScale, h]

theorem signScale_of_nonpos (α β x : ℝ) (hx : x ≤ 0) : signScale α β x = β * x := by
  rcases lt_or_eq_of_le hx with h | h
  · simp [signScale, h, not_lt_of_ge hx]
  · rw [h]; simp [signScale]

theorem positive_scaled_lower_bound (a ε x D Dmax : ℝ)
    (hε : 0 ≤ ε) (ha : ε ≤ a) (hx : 0 ≤ x)
    (hD : 0 < D) (hDmax : D ≤ Dmax) : ε * x / Dmax ≤ a * x / D := by
  exact (div_le_div_of_nonneg_left (mul_nonneg hε hx) hD hDmax).trans
    (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right ha hx) (le_of_lt hD))

def normalizedCoefficient {m : ℕ}
    (α β D : (Fin (m + 1) → Bool) → ℝ)
    (z : Fin (m + 1) → Bool) (i : Fin (m + 1)) : ℝ :=
  signScale (α z) (β z) (binaryRloo z i) / D z

/-- Separate denominators for the two counterfactual labels are retained. -/
theorem normalized_contrast_lower_bound {m : ℕ} (hm : m ≠ 0)
    (α β D : (Fin (m + 1) → Bool) → ℝ) (ε Dmax : ℝ)
    (hε : 0 ≤ ε) (hα : ∀ z, ε ≤ α z) (hβ : ∀ z, ε ≤ β z)
    (hD : ∀ z, 0 < D z) (hDmax : ∀ z, D z ≤ Dmax)
    (i : Fin (m + 1)) (z : Fin m → Bool) :
    ε / Dmax ≤ contrast (normalizedCoefficient α β D) i z := by
  obtain ⟨hl, hu⟩ := sibling_mean_mem hm z
  have hpos := positive_scaled_lower_bound _ ε (1 - iidLooBaseline bit z) _ Dmax
    hε (hα (Fin.insertNth i true z)) (sub_nonneg.mpr hu)
    (hD (Fin.insertNth i true z)) (hDmax (Fin.insertNth i true z))
  have hneg := positive_scaled_lower_bound _ ε (iidLooBaseline bit z) _ Dmax
    hε (hβ (Fin.insertNth i false z)) hl
    (hD (Fin.insertNth i false z)) (hDmax (Fin.insertNth i false z))
  unfold contrast normalizedCoefficient
  rw [binary_rloo_insert, binary_rloo_insert]
  simp only [bit, Bool.false_eq_true, ↓reduceIte, zero_sub]
  rw [signScale_of_nonneg _ _ _ (sub_nonneg.mpr hu), signScale_of_nonpos _ _ _ (neg_nonpos.mpr hl)]
  have heq : ε * (1 - iidLooBaseline bit z) / Dmax +
      ε * iidLooBaseline bit z / Dmax = ε / Dmax := by ring
  have hsum := add_le_add hpos hneg
  rw [heq] at hsum
  convert hsum using 1 <;> ring

theorem normalized_multiplier_lower_bound {m : ℕ} (hm : m ≠ 0)
    (p : A → ℝ) (b : A → Bool) (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (α β D : (Fin (m + 1) → Bool) → ℝ) (ε Dmax : ℝ)
    (hε : 0 ≤ ε) (hα : ∀ z, ε ≤ α z) (hβ : ∀ z, ε ≤ β z)
    (hD : ∀ z, 0 < D z) (hDmax : ∀ z, D z ≤ Dmax) :
    (m + 1 : ℝ) * (ε / Dmax) ≤ multiplier p b (normalizedCoefficient α β D) :=
  multiplier_lower_bound p b hp0 hp _ _
    (normalized_contrast_lower_bound hm α β D ε Dmax hε hα hβ hD hDmax)

/-- Every real-arithmetic normalization path has this form: identity fallback
or the actual capped/stabilized formula followed by positive factor clamps. -/
def safeguardedFactors (fallback : Bool) (P N Qp Qn mass ε cap maxScale : ℝ) : ℝ × ℝ :=
  if fallback then (1, 1) else
    let κ := P / N
    let a := Real.sqrt (mass / (Qp + min (κ ^ 2 * Qn) cap + ε))
    (positiveClamp ε maxScale a, positiveClamp ε maxScale (a * κ))

theorem safeguarded_factors_lower_bound (fallback : Bool)
    (P N Qp Qn mass ε cap maxScale : ℝ) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale) :
    ε ≤ (safeguardedFactors fallback P N Qp Qn mass ε cap maxScale).1 ∧
    ε ≤ (safeguardedFactors fallback P N Qp Qn mass ε cap maxScale).2 := by
  cases fallback
  · exact ⟨(positiveClamp_mem ε maxScale _ hεK).1, (positiveClamp_mem ε maxScale _ hεK).1⟩
  · exact ⟨hε1, hε1⟩

def tokenDenominator {m : ℕ} (length : Bool → ℝ) (z : Fin (m + 1) → Bool) : ℝ :=
  ∑ i, length (z i)

theorem token_denominator_bounds {m : ℕ} (length : Bool → ℝ) (Lmax : ℝ)
    (hpos : ∀ b, 0 < length b) (hmax : ∀ b, length b ≤ Lmax)
    (z : Fin (m + 1) → Bool) :
    0 < tokenDenominator length z ∧ tokenDenominator length z ≤ (m + 1 : ℝ) * Lmax := by
  constructor
  · apply Finset.sum_pos (fun i _ => hpos (z i))
    exact Finset.univ_nonempty
  · have hsum := Finset.sum_le_sum (s := Finset.univ) (fun i _ => hmax (z i))
    simpa [tokenDenominator] using hsum

theorem token_normalized_multiplier_lower_bound {m : ℕ} (hm : m ≠ 0)
    (p : A → ℝ) (b : A → Bool) (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (α β : (Fin (m + 1) → Bool) → ℝ) (length : Bool → ℝ) (ε Lmax : ℝ)
    (hε : 0 ≤ ε) (hα : ∀ z, ε ≤ α z) (hβ : ∀ z, ε ≤ β z)
    (hpos : ∀ q, 0 < length q) (hmax : ∀ q, length q ≤ Lmax) :
    ε / Lmax ≤ multiplier p b (normalizedCoefficient α β (tokenDenominator length)) := by
  have h := normalized_multiplier_lower_bound hm p b hp0 hp α β (tokenDenominator length)
    ε ((m + 1 : ℝ) * Lmax) hε hα hβ
    (fun z => (token_denominator_bounds length Lmax hpos hmax z).1)
    (fun z => (token_denominator_bounds length Lmax hpos hmax z).2)
  have hn : (m + 1 : ℝ) ≠ 0 := by positivity
  have heq : (m + 1 : ℝ) * (ε / ((m + 1 : ℝ) * Lmax)) = ε / Lmax := by
    have hL : 0 < Lmax := (hpos false).trans_le (hmax false)
    field_simp [hn, ne_of_gt hL]
    ring
  rwa [heq] at h

def uniformBoost {m : ℕ} (D : (Fin (m + 1) → Bool) → ℝ)
    (z : Fin (m + 1) → Bool) (_i : Fin (m + 1)) : ℝ :=
  if ∀ j, z j = true then 1 / ((m + 1 : ℝ) * D z) else 0

theorem uniform_boost_contrast_nonnegative {m : ℕ}
    (D : (Fin (m + 1) → Bool) → ℝ) (hD : ∀ z, 0 < D z)
    (i : Fin (m + 1)) (z : Fin m → Bool) : 0 ≤ contrast (uniformBoost D) i z := by
  have hfalse : ¬ ∀ j, (Fin.insertNth i false z : Fin (m + 1) → Bool) j = true := by
    intro h
    have := h i
    simpa using this
  simp only [contrast, uniformBoost, if_neg hfalse, sub_zero]
  split_ifs
  · exact le_of_lt (div_pos one_pos (mul_pos (by positivity) (hD _)))
  · exact le_rfl

theorem adding_uniform_boost_preserves_lower_bound {m : ℕ}
    (H : (Fin (m + 1) → Bool) → Fin (m + 1) → ℝ)
    (D : (Fin (m + 1) → Bool) → ℝ) (hD : ∀ z, 0 < D z) (d : ℝ)
    (hH : ∀ i z, d ≤ contrast H i z) (i : Fin (m + 1)) (z : Fin m → Bool) :
    d ≤ contrast (fun z i => H z i + uniformBoost D z i) i z := by
  have heq : contrast (fun z i => H z i + uniformBoost D z i) i z =
      contrast H i z + contrast (uniformBoost D) i z := by unfold contrast; ring
  rw [heq]
  exact (hH i z).trans (le_add_of_nonneg_right (uniform_boost_contrast_nonnegative D hD i z))

theorem homogeneous_binary_rloo_zero {m : ℕ} (hm : m ≠ 0) (q : Bool) (i : Fin (m + 1)) :
    binaryRloo (fun _ => q) i = 0 := by
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  simp [binaryRloo, iidLooBaseline, hm']

theorem uniform_boost_matches_branch {m : ℕ} (hm : m ≠ 0)
    (α β D : (Fin (m + 1) → Bool) → ℝ) (q : Bool) (i : Fin (m + 1)) :
    normalizedCoefficient α β D (fun _ => q) i + uniformBoost D (fun _ => q) i =
      bit q / ((m + 1 : ℝ) * D (fun _ => q)) := by
  have hfalse : ¬ ∀ _j : Fin (m + 1), false = true := by simp
  simp only [normalizedCoefficient, homogeneous_binary_rloo_zero hm, signScale,
    lt_self_iff_false, ↓reduceIte, div_zero, zero_div, zero_add]
  cases q <;> simp [uniformBoost, bit, hfalse]

/-- These are the actual five token-weighted statistics and the small-sum
fallback, with the implementation's additive epsilon, product cap and clamps. -/
def implementationFactors {m : ℕ} (length : Bool → ℝ)
    (ε cap maxScale : ℝ) (z : Fin (m + 1) → Bool) : ℝ × ℝ :=
  let w := fun i => length (z i)
  let a := binaryRloo z
  let P := AdaptiveMechanism.positiveTotal w a
  let N := AdaptiveMechanism.negativeTotal w a
  safeguardedFactors (decide (P < ε ∨ N < ε)) P N
    (AdaptiveMechanism.positiveSquare w a) (AdaptiveMechanism.negativeSquare w a)
    (AdaptiveMechanism.nonzeroMass w a) ε cap maxScale

theorem implementation_factors_lower_bound {m : ℕ} (length : Bool → ℝ)
    (ε cap maxScale : ℝ) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale)
    (z : Fin (m + 1) → Bool) :
    ε ≤ (implementationFactors length ε cap maxScale z).1 ∧
    ε ≤ (implementationFactors length ε cap maxScale z).2 :=
  safeguarded_factors_lower_bound _ _ _ _ _ _ _ _ _ hε1 hεK

def implementationCoefficient {m : ℕ} (uniform : Bool) (length : Bool → ℝ)
    (ε cap maxScale : ℝ) (z : Fin (m + 1) → Bool) (i : Fin (m + 1)) : ℝ :=
  normalizedCoefficient
    (fun z => (implementationFactors length ε cap maxScale z).1)
    (fun z => (implementationFactors length ε cap maxScale z).2)
    (tokenDenominator length) z i +
    if uniform then uniformBoost (tokenDenominator length) z i else 0

theorem implementation_contrast_lower_bound {m : ℕ} (hm : m ≠ 0)
    (uniform : Bool) (length : Bool → ℝ) (ε cap maxScale Lmax : ℝ)
    (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale)
    (hpos : ∀ q, 0 < length q) (hmax : ∀ q, length q ≤ Lmax)
    (i : Fin (m + 1)) (z : Fin m → Bool) :
    ε / ((m + 1 : ℝ) * Lmax) ≤
      contrast (implementationCoefficient uniform length ε cap maxScale) i z := by
  have hα := fun z : Fin (m + 1) → Bool => (implementation_factors_lower_bound length ε cap maxScale hε1 hεK z).1
  have hβ := fun z : Fin (m + 1) → Bool => (implementation_factors_lower_bound length ε cap maxScale hε1 hεK z).2
  have hD := fun z : Fin (m + 1) → Bool => (token_denominator_bounds length Lmax hpos hmax z).1
  have hDmax := fun z : Fin (m + 1) → Bool => (token_denominator_bounds length Lmax hpos hmax z).2
  have hc := normalized_contrast_lower_bound hm _ _ _ ε _ hε hα hβ hD hDmax
  cases uniform
  · simpa only [contrast, implementationCoefficient, Bool.false_eq_true, ↓reduceIte, add_zero] using hc i z
  · simpa only [contrast, implementationCoefficient, ↓reduceIte] using
      adding_uniform_boost_preserves_lower_bound _ _ hD _ hc i z

theorem implementation_multiplier_lower_bound {m : ℕ} (hm : m ≠ 0)
    (p : A → ℝ) (b : A → Bool) (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (uniform : Bool) (length : Bool → ℝ) (ε cap maxScale Lmax : ℝ)
    (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale)
    (hpos : ∀ q, 0 < length q) (hmax : ∀ q, length q ≤ Lmax) :
    ε / Lmax ≤ multiplier p b (implementationCoefficient (m := m) uniform length ε cap maxScale) := by
  have h := multiplier_lower_bound p b hp0 hp _ _
    (implementation_contrast_lower_bound hm uniform length ε cap maxScale Lmax hε hε1 hεK hpos hmax)
  have hn : (m + 1 : ℝ) ≠ 0 := by positivity
  have hL : 0 < Lmax := (hpos false).trans_le (hmax false)
  have heq : (m + 1 : ℝ) * (ε / ((m + 1 : ℝ) * Lmax)) = ε / Lmax := by
    field_simp [hn, ne_of_gt hL]
    ring
  rwa [heq] at h

section Differential
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The multiplier is shared by every parameter direction, and the score
identities are derived from the actual differentiable policy. -/
theorem complete_update_policy_derivative {m : ℕ} (p : E → A → ℝ)
    (b : A → Bool) (θ : E) (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a)
    (hnorm : ∀ᶠ η in 𝓝 θ, ∑ a, p η a = 1)
    (H : (Fin (m + 1) → Bool) → Fin (m + 1) → ℝ) :
    let G := ∑ a, restrictedMass (p θ) b a • policyScore p θ D a
    HasFDerivAt (fun η => successMass (p η) b) G θ ∧
      ∀ v : E, (∑ y : Fin (m + 1) → A, iidMass (p θ) y *
        (∑ i, H (fun j => b (y j)) i * policyScore p θ D (y i) v)) =
        multiplier (p θ) b H * G v := by
  dsimp only
  constructor
  · exact (finite_policy_uniform_correction p b θ D 0 1 0 hd hpos hnorm).1
  · intro v
    have hf : (∑ a, p θ a * policyScore p θ D a v) = 0 := by
      simpa only [ContinuousLinearMap.sum_apply, ContinuousLinearMap.smul_apply,
        smul_eq_mul, ContinuousLinearMap.zero_apply] using
        congrArg (fun L : E →L[ℝ] ℝ => L v) (finite_policy_score_mean_zero p θ D hd hpos hnorm)
    simpa only [successScore, ContinuousLinearMap.sum_apply, ContinuousLinearMap.smul_apply,
      smul_eq_mul] using complete_update_expectation (p θ) (fun a => policyScore p θ D a v) b hf H

/-- A complete positive population multiplier for the real-arithmetic update,
including optional uniform scale and the random class-length denominator. -/
theorem implementation_policy_direction {m : ℕ} (hm : m ≠ 0)
    (p : E → A → ℝ) (b : A → Bool) (θ : E) (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a)
    (hnorm : ∀ᶠ η in 𝓝 θ, ∑ a, p η a = 1)
    (uniform : Bool) (length : Bool → ℝ) (ε cap maxScale Lmax : ℝ)
    (hε : 0 < ε) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale)
    (hlength : ∀ q, 0 < length q) (hmax : ∀ q, length q ≤ Lmax) :
    let H := implementationCoefficient (m := m) uniform length ε cap maxScale
    let G := ∑ a, restrictedMass (p θ) b a • policyScore p θ D a
    HasFDerivAt (fun η => successMass (p η) b) G θ ∧
      0 < multiplier (p θ) b H ∧ ε / Lmax ≤ multiplier (p θ) b H ∧
      ∀ v : E, (∑ y : Fin (m + 1) → A, iidMass (p θ) y *
        (∑ i, H (fun j => b (y j)) i * policyScore p θ D (y i) v)) =
        multiplier (p θ) b H * G v := by
  dsimp only
  have hp0 : ∀ a, 0 ≤ p θ a := @mem_of_mem_nhds E _ θ {η | ∀ a, 0 ≤ p η a} hpos
  have hp : (∑ a, p θ a) = 1 := @mem_of_mem_nhds E _ θ {η | ∑ a, p η a = 1} hnorm
  have hbound := implementation_multiplier_lower_bound hm (p θ) b hp0 hp uniform
    length ε cap maxScale Lmax (le_of_lt hε) hε1 hεK hlength hmax
  have hL : 0 < Lmax := (hlength false).trans_le (hmax false)
  have hderiv := complete_update_policy_derivative p b θ D hd hpos hnorm
    (implementationCoefficient (m := m) uniform length ε cap maxScale)
  exact ⟨hderiv.1, (div_pos hε hL).trans_le hbound, hbound, hderiv.2⟩

end Differential

end
end REINFORCEProMax.BinaryUpdate
