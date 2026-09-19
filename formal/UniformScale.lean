import PolicyGradient
import RLOOCovariance

/-!
The exact population contribution added by uniform-scale on homogeneous
binary-reward groups. This is an on-policy score calculation before random
token reduction; it retains the actual event shared by all responses.
-/
namespace REINFORCEProMax.UniformScale
open scoped BigOperators Topology
open BatchReduction
noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

theorem iid_mass_total (p : A → ℝ) (n : ℕ) :
    (∑ s : Fin n → A, iidMass p s) = (∑ a, p a) ^ n := by
  unfold iidMass
  calc
    _ = ∑ s ∈ Fintype.piFinset (fun _ : Fin n => (Finset.univ : Finset A)),
        ∏ i : Fin n, p (s i) := by simp
    _ = ∏ _i : Fin n, ∑ a ∈ (Finset.univ : Finset A), p a :=
      Finset.sum_prod_piFinset (s := Finset.univ) (g := fun _ a => p a)
    _ = _ := by simp

theorem iid_coordinate_unnormalized (p f : A → ℝ) (m : ℕ) (i : Fin (m + 1)) :
    (∑ s : Fin (m + 1) → A, iidMass p s * f (s i)) =
      (∑ a, p a) ^ m * (∑ a, p a * f a) := by
  rw [iid_expect_insertNth p _ i]
  simp only [Fin.insertNth_apply_same]
  calc
    _ = ∑ a, (p a * f a) * (∑ s : Fin m → A, iidMass p s) := by
      apply Finset.sum_congr rfl
      intro a _
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro s _
      ring
    _ = _ := by rw [iid_mass_total, ← Finset.sum_mul]; ring

theorem iid_sum_unnormalized (p f : A → ℝ) (m : ℕ) :
    (∑ s : Fin (m + 1) → A, iidMass p s * (∑ i, f (s i))) =
      (m + 1 : ℝ) * (∑ a, p a) ^ m * (∑ a, p a * f a) := by
  conv_lhs => rw [show (∑ s : Fin (m + 1) → A, iidMass p s * (∑ i, f (s i))) =
      ∑ s : Fin (m + 1) → A, ∑ i, iidMass p s * f (s i) by
        simp only [Finset.mul_sum]]
  rw [Finset.sum_comm]
  simp_rw [iid_coordinate_unnormalized]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    Nat.cast_add, Nat.cast_one]
  ring

def restrictedMass (p : A → ℝ) (b : A → Bool) (a : A) : ℝ := if b a then p a else 0
def successMass (p : A → ℝ) (b : A → Bool) : ℝ := ∑ a, restrictedMass p b a
def successScore (p f : A → ℝ) (b : A → Bool) : ℝ := ∑ a, restrictedMass p b a * f a

def homogeneousContribution {n : ℕ} (b : A → Bool) (c : ℝ) (f : A → ℝ)
    (s : Fin n → A) : ℝ :=
  if ∀ i, b (s i) = true then c / (n : ℝ) ^ 2 * ∑ i, f (s i) else 0

theorem restricted_iid_mass (p : A → ℝ) (b : A → Bool) {n : ℕ} (s : Fin n → A) :
    iidMass (restrictedMass p b) s = if ∀ i, b (s i) = true then iidMass p s else 0 := by
  classical
  by_cases h : ∀ i, b (s i) = true
  · simp only [if_pos h, iidMass]
    apply Finset.prod_congr rfl
    intro i _
    simp [restrictedMass, h i]
  · rw [if_neg h]
    push_neg at h
    obtain ⟨i, hi⟩ := h
    apply Finset.prod_eq_zero (Finset.mem_univ i)
    simp [restrictedMass, hi]

theorem homogeneous_expectation (p f : A → ℝ) (b : A → Bool) (c : ℝ) (m : ℕ) :
    (∑ s : Fin (m + 1) → A, iidMass p s * homogeneousContribution b c f s) =
      c / (m + 1 : ℝ) * successMass p b ^ m * successScore p f b := by
  have hterm : ∀ s : Fin (m + 1) → A,
      iidMass p s * homogeneousContribution b c f s =
      c / (m + 1 : ℝ) ^ 2 * (iidMass (restrictedMass p b) s * ∑ i, f (s i)) := by
    intro s
    rw [restricted_iid_mass]
    unfold homogeneousContribution
    split_ifs <;> push_cast <;> ring
  simp_rw [hterm]
  rw [← Finset.mul_sum, iid_sum_unnormalized]
  unfold successMass successScore
  have hn : (m + 1 : ℝ) ≠ 0 := by positivity
  field_simp
  <;> ring

theorem restricted_complement (p : A → ℝ) (b : A → Bool) (a : A) :
    restrictedMass p (fun x => !(b x)) a = p a - restrictedMass p b a := by
  cases hb : b a <;> simp [restrictedMass, hb]

theorem complement_mass (p : A → ℝ) (b : A → Bool) (hp : ∑ a, p a = 1) :
    successMass p (fun a => !(b a)) = 1 - successMass p b := by
  simp only [successMass, restricted_complement, Finset.sum_sub_distrib, hp]

theorem complement_score (p f : A → ℝ) (b : A → Bool)
    (hf : ∑ a, p a * f a = 0) :
    successScore p f (fun a => !(b a)) = -successScore p f b := by
  simp only [successScore, restricted_complement, sub_mul, Finset.sum_sub_distrib, hf, zero_sub]

def binaryCorrection {n : ℕ} (b : A → Bool) (lo hi : ℝ) (f : A → ℝ)
    (s : Fin n → A) : ℝ :=
  homogeneousContribution b hi f s + homogeneousContribution (fun a => !(b a)) lo f s

def correctionFactor (s lo hi : ℝ) (m : ℕ) : ℝ :=
  (hi * s ^ m - lo * (1 - s) ^ m) / (m + 1 : ℝ)

theorem binary_correction_expectation (p f : A → ℝ) (b : A → Bool) (lo hi : ℝ)
    (hp : ∑ a, p a = 1) (hf : ∑ a, p a * f a = 0) (m : ℕ) :
    (∑ s : Fin (m + 1) → A, iidMass p s * binaryCorrection b lo hi f s) =
      correctionFactor (successMass p b) lo hi m * successScore p f b := by
  simp only [binaryCorrection, mul_add, Finset.sum_add_distrib,
    homogeneous_expectation, complement_mass p b hp, complement_score p f b hf]
  unfold correctionFactor
  ring

theorem correction_factor_nonnegative (s lo hi : ℝ) (m : ℕ)
    (h : lo * (1 - s) ^ m ≤ hi * s ^ m) : 0 ≤ correctionFactor s lo hi m := by
  exact div_nonneg (sub_nonneg.mpr h) (by positivity)

theorem correction_factor_nonnegative_of_signed_rewards (s lo hi : ℝ) (m : ℕ)
    (hs : 0 ≤ s) (hs1 : s ≤ 1) (hl : lo ≤ 0) (hh : 0 ≤ hi) :
    0 ≤ correctionFactor s lo hi m := by
  apply correction_factor_nonnegative
  exact (mul_nonpos_of_nonpos_of_nonneg hl (pow_nonneg (sub_nonneg.mpr hs1) _)).trans
    (mul_nonneg hh (pow_nonneg hs _))

theorem zero_one_correction_factor (s : ℝ) (m : ℕ) :
    correctionFactor s 0 1 m = s ^ m / (m + 1 : ℝ) := by
  simp [correctionFactor]

theorem zero_one_correction_positive (s : ℝ) (hs : 0 < s) (m : ℕ) :
    0 < correctionFactor s 0 1 m := by rw [zero_one_correction_factor]; positivity

theorem correction_factor_reward_shift (s lo hi c : ℝ) (m : ℕ) :
    correctionFactor s (lo + c) (hi + c) m - correctionFactor s lo hi m =
      c / (m + 1 : ℝ) * (s ^ m - (1 - s) ^ m) := by
  unfold correctionFactor
  ring

def uniformPatch {n : ℕ} (b : A → Bool) (lo hi : ℝ) (f : A → ℝ)
    (mixed : (Fin n → A) → ℝ) (s : Fin n → A) : ℝ :=
  if ∀ i, b (s i) = true then hi / (n : ℝ) ^ 2 * ∑ i, f (s i)
  else if ∀ i, b (s i) = false then lo / (n : ℝ) ^ 2 * ∑ i, f (s i)
  else mixed s

theorem uniform_patch_difference {m : ℕ} (b : A → Bool) (lo hi : ℝ) (f : A → ℝ)
    (mixed : (Fin (m + 1) → A) → ℝ) (s : Fin (m + 1) → A)
    (hgood : (∀ i, b (s i) = true) → mixed s = 0)
    (hbad : (∀ i, b (s i) = false) → mixed s = 0) :
    uniformPatch b lo hi f mixed s - mixed s = binaryCorrection b lo hi f s := by
  have hn : (∀ i, (!(b (s i))) = true) ↔ ∀ i, b (s i) = false := by simp
  by_cases hg : ∀ i, b (s i) = true
  · have hb : ¬ ∀ i, b (s i) = false := by
      intro h
      have := h 0
      rw [hg 0] at this
      contradiction
    simp [uniformPatch, binaryCorrection, homogeneousContribution, hg, hb, hn, hgood hg]
  · by_cases hb : ∀ i, b (s i) = false
    · simp [uniformPatch, binaryCorrection, homogeneousContribution, hg, hb, hn, hbad hb]
    · simp [uniformPatch, binaryCorrection, homogeneousContribution, hg, hb, hn]

theorem uniform_patch_expectation (p f : A → ℝ) (b : A → Bool) (lo hi : ℝ)
    (hp : ∑ a, p a = 1) (hf : ∑ a, p a * f a = 0) (m : ℕ)
    (mixed : (Fin (m + 1) → A) → ℝ)
    (hgood : ∀ s, (∀ i, b (s i) = true) → mixed s = 0)
    (hbad : ∀ s, (∀ i, b (s i) = false) → mixed s = 0) :
    (∑ s, iidMass p s * uniformPatch b lo hi f mixed s) - (∑ s, iidMass p s * mixed s) =
      correctionFactor (successMass p b) lo hi m * successScore p f b := by
  rw [← Finset.sum_sub_distrib]
  simp_rw [← mul_sub, uniform_patch_difference b lo hi f mixed _ (hgood _) (hbad _)]
  exact binary_correction_expectation p f b lo hi hp hf m

theorem homogeneous_zero_reward {n : ℕ} (b : A → Bool) (f : A → ℝ) (s : Fin n → A) :
    homogeneousContribution b 0 f s = 0 := by simp [homogeneousContribution]

theorem constant_reward_mean_zero (p f : A → ℝ) (c : ℝ) (n : ℕ)
    (hp : ∑ a, p a = 1) (hf : ∑ a, p a * f a = 0) :
    (∑ s : Fin n → A, iidMass p s * (c / (n : ℝ) ^ 2 * ∑ i, f (s i))) = 0 := by
  have hc := iid_centered_sum_mean p f hp hf n
  calc
    _ = c / (n : ℝ) ^ 2 * (∑ s : Fin n → A, iidMass p s * ∑ i, f (s i)) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro s _
      ring
    _ = 0 := by rw [hc, mul_zero]

theorem constant_reward_variance (p f : A → ℝ) (c : ℝ) (m : ℕ)
    (hp : ∑ a, p a = 1) (hf : ∑ a, p a * f a = 0) :
    (∑ s : Fin (m + 1) → A,
      iidMass p s * (c / (m + 1 : ℝ) ^ 2 * ∑ i, f (s i)) ^ 2) =
      c ^ 2 / (m + 1 : ℝ) ^ 3 * (∑ a, p a * (f a) ^ 2) := by
  have hc := iid_centered_sum_second_moment p f hp hf (m + 1)
  have hn : (m + 1 : ℝ) ≠ 0 := by positivity
  calc
    _ = (c / (m + 1 : ℝ) ^ 2) ^ 2 *
        (∑ s : Fin (m + 1) → A, iidMass p s * (∑ i, f (s i)) ^ 2) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro s _
      push_cast
      ring
    _ = _ := by
      rw [hc, Nat.cast_add, Nat.cast_one]
      field_simp
      <;> ring

theorem rloo_homogeneous_zero {m : ℕ} (hm : m ≠ 0) (p r f : A → ℝ)
    (s : Fin (m + 1) → A) (c : ℝ) (hr : ∀ i, r (s i) = c) :
    iidRlooEstimator p r f s = 0 := by
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  simp [iidRlooEstimator, iidLooTerm, iidLooBaseline, hr, hm']

section Differential
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Success-rate differentiation and the exact correction share one
finite differentiable policy model; zero-mean scores are derived. -/
theorem finite_policy_uniform_correction (p : E → A → ℝ) (b : A → Bool)
    (θ : E) (D : A → E →L[ℝ] ℝ) (lo hi : ℝ) (m : ℕ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a)
    (hnorm : ∀ᶠ η in 𝓝 θ, ∑ a, p η a = 1) :
    let G := ∑ a, restrictedMass (p θ) b a • policyScore p θ D a
    HasFDerivAt (fun η => successMass (p η) b) G θ ∧
      ∀ v : E, (∑ s : Fin (m + 1) → A,
        iidMass (p θ) s * binaryCorrection b lo hi (fun a => policyScore p θ D a v) s) =
        correctionFactor (successMass (p θ) b) lo hi m * G v := by
  dsimp only
  have hp : (∑ a, p θ a) = 1 := @mem_of_mem_nhds E _ θ {η | ∑ a, p η a = 1} hnorm
  have hmass : ∀ η a, p η a * (if b a then (1 : ℝ) else 0) = restrictedMass (p η) b a := by
    intro η a
    cases h : b a <;> simp [restrictedMass, h]
  constructor
  · simpa only [hmass, successMass] using
      finite_policy_gradient p (fun a => if b a then 1 else 0) θ D hd hpos
  · intro v
    have hf : (∑ a, p θ a * policyScore p θ D a v) = 0 := by
      simpa only [ContinuousLinearMap.sum_apply, ContinuousLinearMap.smul_apply,
        smul_eq_mul, ContinuousLinearMap.zero_apply] using
        congrArg (fun L : E →L[ℝ] ℝ => L v) (finite_policy_score_mean_zero p θ D hd hpos hnorm)
    simpa only [successScore, ContinuousLinearMap.sum_apply,
      ContinuousLinearMap.smul_apply, smul_eq_mul] using
      binary_correction_expectation (p θ) (fun a => policyScore p θ D a v) b lo hi hp hf m

end Differential

def bernoulliMass (s : ℝ) (a : Bool) : ℝ := if a then s else 1 - s
def shiftedReward (c : ℝ) (a : Bool) : ℝ := if a then c + 1 else c
def bernoulliScore (s : ℝ) (a : Bool) : ℝ := if a then 1 / s else -1 / (1 - s)

theorem bernoulli_mass_normalized (s : ℝ) : (∑ a, bernoulliMass s a) = 1 := by
  simp [bernoulliMass, Fintype.sum_bool]

theorem bernoulli_mass_nonnegative (s : ℝ) (hs : 0 ≤ s) (hs1 : s ≤ 1) :
    ∀ a, 0 ≤ bernoulliMass s a := by
  intro a
  cases a <;> simp [bernoulliMass, hs, sub_nonneg.mpr hs1]

theorem bernoulli_score_derivative (s : ℝ) (hs : s ≠ 0) (hs1 : 1 - s ≠ 0) (a : Bool) :
    HasDerivAt (fun t => Real.log (bernoulliMass t a)) (bernoulliScore s a) s := by
  cases a
  · simpa only [bernoulliMass, Bool.false_eq_true, ↓reduceIte, bernoulliScore] using
      ((hasDerivAt_id s).const_sub 1).log hs1
  · simpa only [bernoulliMass, ↓reduceIte, bernoulliScore] using (hasDerivAt_id s).log hs

theorem bernoulli_score_zero (s : ℝ) (hs : s ≠ 0) (hs1 : 1 - s ≠ 0) :
    (∑ a, bernoulliMass s a * bernoulliScore s a) = 0 := by
  simp [bernoulliMass, bernoulliScore, Fintype.sum_bool]
  field_simp
  <;> ring

theorem bernoulli_success_score (s : ℝ) (hs : s ≠ 0) :
    successScore (bernoulliMass s) (bernoulliScore s) id = 1 := by
  simp [successScore, restrictedMass, bernoulliMass, bernoulliScore, Fintype.sum_bool, hs]

theorem shifted_reward_score (s c : ℝ) (hs : s ≠ 0) (hs1 : 1 - s ≠ 0) :
    (∑ a, bernoulliMass s a * shiftedReward c a * bernoulliScore s a) = 1 := by
  simp [bernoulliMass, shiftedReward, bernoulliScore, Fintype.sum_bool]
  field_simp
  <;> ring

theorem shifted_reward_derivative (s c : ℝ) :
    HasDerivAt (fun t => ∑ a, bernoulliMass t a * shiftedReward c a) 1 s := by
  have heq : (fun t => ∑ a, bernoulliMass t a * shiftedReward c a) = fun t => t + c := by
    funext t
    simp [bernoulliMass, shiftedReward, Fintype.sum_bool]
    ring
  rw [heq]
  simpa using (hasDerivAt_id s).add_const c

/-- Two responses, each one token. Mixed RLOO signals are +1 and -1,
so their ideal asymmetric normalization is identical. A common stabilizer
shrinkage is represented by a, with 0 < a ≤ 1 in the implementation. -/
theorem two_response_uniform_expectation (s c a : ℝ) (hs : s ≠ 0) (hs1 : 1 - s ≠ 0) :
    let p := bernoulliMass s
    let r := shiftedReward c
    let f := bernoulliScore s
    let mixed := fun z : Fin 2 → Bool => a * iidRlooEstimator p r f z
    (∑ z : Fin 2 → Bool, iidMass p z * uniformPatch id c (c + 1) f mixed z) =
      a + ((c + 1) * s - c * (1 - s)) / 2 := by
  dsimp only
  let p := bernoulliMass s
  let r := shiftedReward c
  let f := bernoulliScore s
  let mixed := fun z : Fin 2 → Bool => a * iidRlooEstimator p r f z
  have hg : ∀ z : Fin 2 → Bool, (∀ i, id (z i) = true) → mixed z = 0 := by
    intro z hz
    have hr : ∀ i, r (z i) = c + 1 := by intro i; simp [r, shiftedReward, show z i = true from hz i]
    simp only [mixed, rloo_homogeneous_zero (by omega : 1 ≠ 0) p r f z (c + 1) hr, mul_zero]
  have hb : ∀ z : Fin 2 → Bool, (∀ i, id (z i) = false) → mixed z = 0 := by
    intro z hz
    have hr : ∀ i, r (z i) = c := by intro i; simp [r, shiftedReward, show z i = false from hz i]
    simp only [mixed, rloo_homogeneous_zero (by omega : 1 ≠ 0) p r f z c hr, mul_zero]
  have hd := uniform_patch_expectation p f id c (c + 1) (bernoulli_mass_normalized s)
    (bernoulli_score_zero s hs hs1) 1 mixed hg hb
  have hm : (∑ z : Fin 2 → Bool, iidMass p z * mixed z) = a := by
    calc
      _ = a * (∑ z : Fin 2 → Bool, iidMass p z * iidRlooEstimator p r f z) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro z _
        dsimp [mixed]
        ring
      _ = a := by
        rw [finite_iid_rloo_unbiased (by omega : 1 ≠ 0) p r f
          (bernoulli_mass_normalized s) (bernoulli_score_zero s hs hs1)]
        rw [shifted_reward_score s c hs hs1, mul_one]
  have hprob : successMass p id = s := by simp [successMass, restrictedMass, p, bernoulliMass, Fintype.sum_bool]
  rw [hm, hprob, bernoulli_success_score s hs] at hd
  simp only [correctionFactor, pow_one] at hd
  norm_num at hd
  linarith

theorem shifted_uniform_reverses_update (a : ℝ) (ha : a ≤ 1) :
    (∑ z : Fin 2 → Bool, iidMass (bernoulliMass (1 / 10)) z *
      uniformPatch id 3 4 (bernoulliScore (1 / 10))
        (fun z => a * iidRlooEstimator (bernoulliMass (1 / 10)) (shiftedReward 3)
          (bernoulliScore (1 / 10)) z) z) ≤ -(3 / 20 : ℝ) := by
  have h := two_response_uniform_expectation (1 / 10) 3 a (by norm_num) (by norm_num)
  norm_num at h
  linarith

theorem zero_one_two_response_update (s : ℝ) (hs : 0 < s) (hs1 : s < 1)
    (a : ℝ) (ha : 0 < a) :
    0 < (∑ z : Fin 2 → Bool, iidMass (bernoulliMass s) z *
      uniformPatch id 0 1 (bernoulliScore s)
        (fun z => a * iidRlooEstimator (bernoulliMass s) (shiftedReward 0)
          (bernoulliScore s) z) z) := by
  have h := two_response_uniform_expectation s 0 a (ne_of_gt hs) (ne_of_gt (sub_pos.mpr hs1))
  simp only [zero_add, zero_mul, sub_zero, one_mul] at h
  linarith

/-- The implemented real-arithmetic scale on a mixed two-response group
with one active token each and reward gap one. The product cap is inactive;
the additive stabilizer and factor clamp are retained. -/
def pairScale (ε : ℝ) : ℝ := min (max (Real.sqrt (2 / (2 + ε))) ε) 10

theorem mixed_pair_sample_variance (c : ℝ) :
    (c - (c + (c + 1)) / 2) ^ 2 + ((c + 1) - (c + (c + 1)) / 2) ^ 2 = (1 / 2 : ℝ) := by
  ring

theorem mixed_pair_above_uniform_tolerance :
    (1 / 100000000 : ℝ) < Real.sqrt (1 / 2) := by
  have hs := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 1 / 2)
  have hn := Real.sqrt_nonneg (1 / 2)
  nlinarith

theorem pair_scale_bounds (ε : ℝ) (hε : 0 < ε) (hε1 : ε ≤ 1) :
    0 < pairScale ε ∧ pairScale ε ≤ 1 := by
  have hsqrt : Real.sqrt (2 / (2 + ε)) ≤ 1 := by
    apply (Real.sqrt_le_left (by norm_num)).mpr
    apply (div_le_iff₀ (by linarith : 0 < 2 + ε)).mpr
    nlinarith
  constructor
  · unfold pairScale
    exact lt_min (lt_of_lt_of_le hε (le_max_right _ _)) (by norm_num)
  · unfold pairScale
    exact (min_le_left _ _).trans (max_le hsqrt hε1)

theorem shifted_uniform_reverses_with_safeguards (ε : ℝ) (hε : 0 < ε) (hε1 : ε ≤ 1) :
    (∑ z : Fin 2 → Bool, iidMass (bernoulliMass (1 / 10)) z *
      uniformPatch id 3 4 (bernoulliScore (1 / 10))
        (fun z => pairScale ε * iidRlooEstimator (bernoulliMass (1 / 10)) (shiftedReward 3)
          (bernoulliScore (1 / 10)) z) z) ≤ -(3 / 20 : ℝ) :=
  shifted_uniform_reverses_update _ (pair_scale_bounds ε hε hε1).2

theorem zero_one_two_response_with_safeguards (s : ℝ) (hs : 0 < s) (hs1 : s < 1)
    (ε : ℝ) (hε : 0 < ε) (hε1 : ε ≤ 1) :
    0 < (∑ z : Fin 2 → Bool, iidMass (bernoulliMass s) z *
      uniformPatch id 0 1 (bernoulliScore s)
        (fun z => pairScale ε * iidRlooEstimator (bernoulliMass s) (shiftedReward 0)
          (bernoulliScore s) z) z) :=
  zero_one_two_response_update s hs hs1 _ (pair_scale_bounds ε hε hε1).1

end
end REINFORCEProMax.UniformScale
