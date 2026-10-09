import BinaryLength

/-! Exact marginalization of a shared random token denominator over iid
prompt groups. A group may contain dependent RLOO responses. -/
namespace REINFORCEProMax.BatchDenominator
open scoped BigOperators
open BatchReduction
noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

def totalLength {n : ℕ} (L : A → ℝ) (s : Fin n → A) : ℝ := ∑ i, L (s i)

def inverseWeight (m : ℕ) (p L : A → ℝ) (a : A) : ℝ :=
  (m+1 : ℝ) * ∑ z : Fin m → A, iidMass p z / (L a + totalLength L z)

theorem total_length_insert {m : ℕ} (L : A → ℝ) (i : Fin (m+1))
    (a : A) (z : Fin m → A) :
    totalLength L (Fin.insertNth i a z) = L a + totalLength L z := by
  unfold totalLength
  rw [Fin.sum_univ_succAbove _ i]
  simp

theorem coordinate_expectation {m : ℕ} (p L F : A → ℝ) (i : Fin (m+1)) :
    (∑ s : Fin (m+1) → A, iidMass p s * (F (s i) / totalLength L s)) =
      ∑ a, p a * F a * ∑ z : Fin m → A, iidMass p z / (L a + totalLength L z) := by
  rw [iid_expect_insertNth p _ i]
  simp only [Fin.insertNth_apply_same, total_length_insert]
  apply Finset.sum_congr rfl
  intro a _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro z _
  ring

theorem batch_expectation {m : ℕ} (p L F : A → ℝ) :
    (∑ s : Fin (m+1) → A, iidMass p s * ((∑ i, F (s i)) / totalLength L s)) =
      ∑ a, p a * inverseWeight m p L a * F a := by
  simp_rw [Finset.sum_div, Finset.mul_sum]
  rw [Finset.sum_comm]
  simp_rw [coordinate_expectation p L F]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a _
  unfold inverseWeight
  push_cast
  ring

theorem total_length_bounds {n : ℕ} (L : A → ℝ) (lo hi : ℝ)
    (hlo : ∀ a, lo ≤ L a) (hhi : ∀ a, L a ≤ hi) (s : Fin n → A) :
    (n : ℝ)*lo ≤ totalLength L s ∧ totalLength L s ≤ (n : ℝ)*hi := by
  constructor
  · simpa [totalLength] using Finset.sum_le_sum (s := Finset.univ) (fun i _ => hlo (s i))
  · simpa [totalLength] using Finset.sum_le_sum (s := Finset.univ) (fun i _ => hhi (s i))

theorem inverse_weight_bounds (m : ℕ) (p L : A → ℝ) (lo hi : ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1) (hl : 0 < lo)
    (hlo : ∀ a, lo ≤ L a) (hhi : ∀ a, L a ≤ hi) (a : A) :
    1/hi ≤ inverseWeight m p L a ∧ inverseWeight m p L a ≤ 1/lo := by
  have hk : (0 : ℝ) < m+1 := by positivity
  have hh : 0 < hi := hl.trans_le ((hlo a).trans (hhi a))
  have hb : ∀ z : Fin m → A,
      1/((m+1 : ℝ)*hi) ≤ 1/(L a+totalLength L z) ∧
      1/(L a+totalLength L z) ≤ 1/((m+1 : ℝ)*lo) := by
    intro z
    obtain ⟨h1,h2⟩ := total_length_bounds L lo hi hlo hhi z
    have hd0 : (m+1 : ℝ)*lo ≤ L a+totalLength L z := by nlinarith [hlo a]
    have hd1 : L a+totalLength L z ≤ (m+1 : ℝ)*hi := by nlinarith [hhi a]
    have hd : 0 < L a+totalLength L z := (mul_pos hk hl).trans_le hd0
    exact ⟨one_div_le_one_div_of_le hd hd1, one_div_le_one_div_of_le (mul_pos hk hl) hd0⟩
  have hlow := Finset.sum_le_sum (s := Finset.univ) (fun z _ =>
    mul_le_mul_of_nonneg_left (hb z).1 (iid_mass_nonneg p hp0 z))
  have hupp := Finset.sum_le_sum (s := Finset.univ) (fun z _ =>
    mul_le_mul_of_nonneg_left (hb z).2 (iid_mass_nonneg p hp0 z))
  simp only [mul_one_div, ← Finset.sum_div, iid_mass_normalized p hp] at hlow hupp
  have hl' := mul_le_mul_of_nonneg_left hlow hk.le
  have hu' := mul_le_mul_of_nonneg_left hupp hk.le
  have hc : ∀ x : ℝ, (m+1 : ℝ)*(1/((m+1 : ℝ)*x)) = 1/x := by
    intro x
    have hk0 := ne_of_gt hk
    by_cases hx : x = 0
    · simp [hx]
    · field_simp [hk0, hx]
  simpa only [hc, inverseWeight] using And.intro hl' hu'

/-- For one group the effective coefficient is exactly the original inverse
length. For larger batches it is a conditional inverse of the shared sum. -/
theorem one_group_weight (p L : A → ℝ) (a : A) :
    inverseWeight 0 p L a = 1/L a := by
  simp [inverseWeight, iidMass, totalLength]

/-- Replace the inverse group length inside each raw response coefficient,
without assuming it is independent of the group signal. -/
theorem coefficient_bridge {I : Type*} [Fintype I] (m : ℕ) (p L : A → ℝ)
    (H f : A → I → ℝ) :
    (∑ s : Fin (m+1) → A, iidMass p s *
      ((∑ i, ∑ j, H (s i) j * f (s i) j) / totalLength L s)) =
      ∑ a, p a * ∑ j, (inverseWeight m p L a * H a j) * f a j := by
  change (∑ s : Fin (m+1) → A, iidMass p s *
    ((∑ i, (fun a => ∑ j, H a j * f a j) (s i)) / totalLength L s)) = _
  rw [batch_expectation p L (fun a => ∑ j, H a j * f a j)]
  apply Finset.sum_congr rfl
  intro a _
  rw [mul_assoc, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Token-proportional microbatch weights recover the global token mean.
Weights based on response counts need not satisfy this identity. -/
theorem token_weighted_partition {I : Type*} [Fintype I] (Z T : I → ℝ)
    (hT : ∀ i, T i ≠ 0) :
    (∑ i, (T i / ∑ j, T j) * (Z i / T i)) = (∑ i, Z i) / ∑ i, T i := by
  have ht : ∀ i, (T i / ∑ j, T j) * (Z i / T i) = Z i / ∑ j, T j := by
    intro i
    by_cases hs : (∑ j, T j) = 0
    · simp [hs]
    · field_simp [hT i, hs]
      ring
  simp_rw [ht]
  exact (Finset.sum_div ..).symm

/-- Unreduced binary response signal, using the existing normalization
statistics and safeguards. The optional boost applies only to all-success groups. -/
def rawResponseSignal {n : ℕ} (uniform : Bool) (b : A → Bool) (length : A → ℝ)
    (ε cap maxScale : ℝ) (y : Fin (n+1) → A) (i : Fin (n+1)) : ℝ :=
  let factors := BinaryLength.responseFactors b length ε cap maxScale y
  NormalizationSafeguards.signScale factors.1 factors.2
    (BinaryUpdate.binaryRloo (fun j => b (y j)) i) +
    if uniform = true ∧ ∀ j, b (y j) = true then 1/(n+1 : ℝ) else 0

/-- Removing the group-token reduction recovers exactly the raw signal. -/
theorem raw_signal_eq_group_scaled {n : ℕ} (uniform : Bool) (b : A → Bool)
    (length : A → ℝ) (ε cap maxScale : ℝ) (y : Fin (n+1) → A) (i : Fin (n+1))
    (hD : BinaryLength.responseDenominator length y ≠ 0) :
    rawResponseSignal uniform b length ε cap maxScale y i =
      BinaryLength.responseDenominator length y *
        BinaryLength.responseCoefficient uniform b length ε cap maxScale y i := by
  unfold rawResponseSignal BinaryLength.responseCoefficient
  split_ifs <;> dsimp only <;> field_simp <;> ring

/-- A positive lower bound on an arbitrary group weight survives the actual
normalizer. No independence from response content or length is used. -/
theorem weighted_raw_slice_bounds {n : ℕ} (hn : n ≠ 0)
    (uniform : Bool) (b : A → Bool) (length : A → ℝ) (ε cap maxScale c : ℝ)
    (W : (Fin (n+1) → A) → ℝ) (hε : 0 ≤ ε) (hε1 : ε ≤ 1)
    (hεK : ε ≤ maxScale) (hc : 0 ≤ c) (hW : ∀ y, c ≤ W y)
    (i : Fin (n+1)) (z : Fin n → A) (a : A) :
    (b a = true → c*ε*(1-iidLooBaseline BinaryUpdate.bit (fun j => b (z j))) ≤
      W (Fin.insertNth i a z) * rawResponseSignal uniform b length ε cap maxScale
        (Fin.insertNth i a z) i) ∧
    (b a = false → W (Fin.insertNth i a z) *
      rawResponseSignal uniform b length ε cap maxScale (Fin.insertNth i a z) i ≤
      -(c*ε*iidLooBaseline BinaryUpdate.bit (fun j => b (z j)))) := by
  let y : Fin (n+1) → A := Fin.insertNth i a z
  let fac := BinaryLength.responseFactors b length ε cap maxScale y
  have hf : ε ≤ fac.1 ∧ ε ≤ fac.2 :=
    BinaryUpdate.safeguarded_factors_lower_bound _ _ _ _ _ _ _ _ _ hε1 hεK
  have hw0 : 0 ≤ W y := hc.trans (hW y)
  have hwa : c*ε ≤ W y * fac.1 := mul_le_mul (hW y) hf.1 hε hw0
  have hwb : c*ε ≤ W y * fac.2 := mul_le_mul (hW y) hf.2 hε hw0
  let u := iidLooBaseline BinaryUpdate.bit (fun j => b (z j))
  obtain ⟨hu0,hu1⟩ := BinaryUpdate.sibling_mean_mem hn (fun j => b (z j))
  have hr : BinaryUpdate.binaryRloo (fun j => b (y j)) i = BinaryUpdate.bit (b a)-u := by
    simp [y, BinaryUpdate.binaryRloo, u]
  have hb0 : 0 ≤ if uniform = true ∧ ∀ j, b (y j) = true then 1/(n+1 : ℝ) else 0 := by
    split_ifs <;> positivity
  constructor
  · intro ha
    change _ ≤ W y * (NormalizationSafeguards.signScale fac.1 fac.2
      (BinaryUpdate.binaryRloo (fun j => b (y j)) i) + _)
    rw [hr, ha]
    simp only [BinaryUpdate.bit, ↓reduceIte]
    rw [BinaryUpdate.signScale_of_nonneg _ _ _ (sub_nonneg.mpr hu1)]
    have h := mul_le_mul_of_nonneg_right hwa (sub_nonneg.mpr hu1)
    have hb := mul_nonneg hw0 hb0
    dsimp [u] at *
    nlinarith
  · intro ha
    have hfalse : ¬ (uniform = true ∧ ∀ j, b (y j) = true) := by
      intro hh
      have hi := hh.2 i
      simpa [y, ha] using hi
    change W y * (NormalizationSafeguards.signScale fac.1 fac.2
      (BinaryUpdate.binaryRloo (fun j => b (y j)) i) + _) ≤ _
    rw [hr, ha, if_neg hfalse]
    simp only [BinaryUpdate.bit, Bool.false_eq_true, ↓reduceIte, zero_sub, add_zero]
    rw [BinaryUpdate.signScale_of_nonpos _ _ _ (neg_nonpos.mpr hu0)]
    have h := mul_le_mul_of_nonneg_right hwb hu0
    dsimp [u] at *
    nlinarith

theorem weighted_raw_multiplier_lower_bound {n : ℕ} (hn : n ≠ 0)
    (p : A → ℝ) (b : A → Bool) (uniform : Bool) (length : A → ℝ)
    (ε cap maxScale c : ℝ) (W : (Fin (n+1) → A) → ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (hmass : ∀ q, 0 < BinaryLength.classMass p b q)
    (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale)
    (hc : 0 ≤ c) (hW : ∀ y, c ≤ W y) :
    (n+1 : ℝ)*c*ε ≤ BinaryLength.classMultiplier p b
      (fun y i => W y * rawResponseSignal uniform b length ε cap maxScale y i) := by
  have h := BinaryLength.class_multiplier_lower_bound p b hp0 hp
    (fun y i => W y * rawResponseSignal uniform b length ε cap maxScale y i) (c*ε) ?_
  · simpa only [mul_assoc] using h
  intro i z
  let u := iidLooBaseline BinaryUpdate.bit (fun j => b (z j))
  have ht := BinaryLength.class_mean_lower_bound p _ b hp0 true (hmass true) (c*ε*(1-u))
    (fun a ha => (weighted_raw_slice_bounds hn uniform b length ε cap maxScale c W
      hε hε1 hεK hc hW i z a).1 ha)
  have hf := BinaryLength.class_mean_upper_bound p _ b hp0 false (hmass false) (-(c*ε*u))
    (fun a ha => (weighted_raw_slice_bounds hn uniform b length ε cap maxScale c W
      hε hε1 hεK hc hW i z a).2 ha)
  change c*ε ≤ BinaryLength.classMean p b true
    (fun a => W (Fin.insertNth i a z) * rawResponseSignal uniform b length ε cap maxScale (Fin.insertNth i a z) i) -
    BinaryLength.classMean p b false
    (fun a => W (Fin.insertNth i a z) * rawResponseSignal uniform b length ε cap maxScale (Fin.insertNth i a z) i)
  linarith

section PromptGroups
variable {X R : Type*} [Fintype X] [DecidableEq X] [Fintype R] [DecidableEq R]

def jointMass {n : ℕ} (μ : X → ℝ) (p : X → R → ℝ)
    (b : X × (Fin (n+1) → R)) : ℝ := μ b.1 * iidMass (p b.1) b.2

theorem joint_mass_normalized {n : ℕ} (μ : X → ℝ) (p : X → R → ℝ)
    (hμ : ∑ x, μ x = 1) (hp : ∀ x, ∑ a, p x a = 1) :
    (∑ b : X × (Fin (n+1) → R), jointMass μ p b) = 1 := by
  simp [jointMass, Fintype.sum_prod_type, ← Finset.mul_sum, iid_mass_normalized, hp, hμ]

theorem joint_mass_nonnegative {n : ℕ} (μ : X → ℝ) (p : X → R → ℝ)
    (hμ : ∀ x, 0 ≤ μ x) (hp : ∀ x a, 0 ≤ p x a)
    (b : X × (Fin (n+1) → R)) : 0 ≤ jointMass μ p b :=
  mul_nonneg (hμ b.1) (iid_mass_nonneg (p b.1) (hp b.1) b.2)

def effectiveCoefficient {n : ℕ} (m : ℕ) (μ : X → ℝ) (p : X → R → ℝ)
    (L : X × (Fin (n+1) → R) → ℝ)
    (H : X → (Fin (n+1) → R) → Fin (n+1) → ℝ)
    (x : X) (y : Fin (n+1) → R) (i : Fin (n+1)) : ℝ :=
  inverseWeight m (jointMass μ p) L (x,y) * H x y i

/-- The actual shared-denominator expectation decomposes using the effective
coefficient, not the separately length-normalized group coefficient. -/
theorem shared_batch_decomposition {n : ℕ} (m : ℕ) (μ : X → ℝ)
    (p f : X → R → ℝ) (b : X → R → Bool)
    (L : X × (Fin (n+1) → R) → ℝ)
    (H : X → (Fin (n+1) → R) → Fin (n+1) → ℝ)
    (hmass : ∀ x q, BinaryLength.classMass (p x) (b x) q ≠ 0)
    (hf : ∀ x, ∑ a, p x a * f x a = 0) :
    (∑ s : Fin (m+1) → X × (Fin (n+1) → R), iidMass (jointMass μ p) s *
      ((∑ k, ∑ i, H (s k).1 (s k).2 i * f (s k).1 ((s k).2 i)) / totalLength L s)) =
      ∑ x, μ x * (BinaryLength.classMultiplier (p x) (b x) (effectiveCoefficient m μ p L H x) *
        UniformScale.successScore (p x) (f x) (b x) +
        BinaryLength.covarianceError (p x) (f x) (b x) (effectiveCoefficient m μ p L H x)) := by
  have hb := coefficient_bridge m (jointMass μ p) L
    (fun a i => H a.1 a.2 i) (fun a i => f a.1 (a.2 i))
  rw [hb, Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro x _
  calc
    _ = μ x * ∑ y : Fin (n+1) → R, iidMass (p x) y *
        ∑ i, effectiveCoefficient m μ p L H x y i * f x (y i) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro y _
      simp only [jointMass, effectiveCoefficient, mul_assoc]
    _ = _ := by rw [BinaryLength.general_update_decomposition (p x) (f x) (b x) (hmass x) (hf x)]

/-- The actual shared-denominator coefficient retains a positive class
contrast. Content-dependent lengths still contribute the separate covariance
residual in `shared_batch_decomposition`. -/
theorem shared_batch_multiplier_lower_bound {n : ℕ} (hn : n ≠ 0) (m : ℕ)
    (μ : X → ℝ) (p : X → R → ℝ) (b : X → R → Bool)
    (length : X → R → ℝ) (uniform : Bool) (ε cap maxScale lo hi : ℝ)
    (L : X × (Fin (n+1) → R) → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (hp0 : ∀ x a, 0 ≤ p x a) (hp : ∀ x, ∑ a, p x a = 1)
    (hlo0 : 0 < lo) (hhi0 : 0 < hi) (hlo : ∀ a, lo ≤ L a) (hhi : ∀ a, L a ≤ hi)
    (hε : 0 ≤ ε) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale)
    (x : X) (hmass : ∀ q, 0 < BinaryLength.classMass (p x) (b x) q) :
    (n+1 : ℝ)*ε/hi ≤ BinaryLength.classMultiplier (p x) (b x)
      (effectiveCoefficient m μ p L
        (fun x => rawResponseSignal uniform (b x) (length x) ε cap maxScale) x) := by
  have hw : ∀ y : Fin (n+1) → R,
      1/hi ≤ inverseWeight m (jointMass μ p) L (x,y) := by
    intro y
    exact (inverse_weight_bounds m (jointMass μ p) L lo hi
      (joint_mass_nonnegative μ p hμ0 hp0) (joint_mass_normalized μ p hμ hp)
      hlo0 hlo hhi (x,y)).1
  have hb := weighted_raw_multiplier_lower_bound hn (p x) (b x) uniform (length x)
    ε cap maxScale (1/hi) (fun y => inverseWeight m (jointMass μ p) L (x,y))
    (hp0 x) (hp x) hmass hε hε1 hεK (one_div_nonneg.mpr hhi0.le) hw
  convert hb using 1 <;> ring

end PromptGroups

end
end REINFORCEProMax.BatchDenominator
