import BatchReduction

/-!
Exact variance of the actual shared-baseline RLOO group average. We derive
the required mixed moments from the finite iid law by induction, so neither
independence of sibling contributions nor their covariance is a premise.
-/
namespace REINFORCEProMax.BatchReduction
open scoped BigOperators
noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

theorem expectation_add (p f g : A → ℝ) :
    expectation p (fun a => f a + g a) = expectation p f + expectation p g := by
  simp [expectation, mul_add, Finset.sum_add_distrib]

theorem expectation_sub (p f g : A → ℝ) :
    expectation p (fun a => f a - g a) = expectation p f - expectation p g := by
  simp [expectation, mul_sub, Finset.sum_sub_distrib]

theorem expectation_mul_const (p f : A → ℝ) (c : ℝ) :
    expectation p (fun a => f a * c) = expectation p f * c := by
  simp [expectation, ← mul_assoc, Finset.sum_mul]

theorem expectation_const_mul (p f : A → ℝ) (c : ℝ) :
    expectation p (fun a => c * f a) = c * expectation p f := by
  simp only [expectation, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a _
  ring

theorem expectation_const (p : A → ℝ) (hp : ∑ a, p a = 1) (c : ℝ) :
    expectation p (fun _ => c) = c := by
  simp [expectation, ← Finset.sum_mul, hp]

theorem expectation_div_const (p f : A → ℝ) (c : ℝ) :
    expectation p (fun a => f a / c) = expectation p f / c := by
  simp [expectation, ← mul_div_assoc, Finset.sum_div]

def sampleSum {n : ℕ} (f : A → ℝ) (s : Fin n → A) : ℝ := ∑ i, f (s i)

theorem sampleSum_cons {n : ℕ} (f : A → ℝ) (a : A) (s : Fin n → A) :
    sampleSum f (Fin.cons a s) = f a + sampleSum f s := by
  simp [sampleSum, Fin.sum_univ_succ]

theorem iid_expect_cons {n : ℕ} (p : A → ℝ) (F : (Fin (n + 1) → A) → ℝ) :
    expectation (iidMass p) F = expectation (fun s : Fin n → A => iidMass p s)
      (fun s => expectation p (fun a => F (Fin.cons a s))) := by
  unfold expectation
  rw [iid_expect_insertNth p F 0, Finset.sum_comm]
  simp only [Fin.insertNth_zero', Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro s _
  apply Finset.sum_congr rfl
  intro a _
  ring

theorem iid_sample_sum_mean (p u : A → ℝ) (hp : ∑ a, p a = 1) (n : ℕ) :
    expectation (fun s : Fin n → A => iidMass p s) (sampleSum u) =
      (n : ℝ) * expectation p u := by
  induction n with
  | zero => simp [expectation, sampleSum]
  | succ n ih =>
      rw [iid_expect_cons]
      simp only [sampleSum_cons, expectation_add, expectation_const p hp,
        expectation_const _ (iid_mass_normalized p hp n), ih, Nat.cast_add, Nat.cast_one]
      ring

theorem iid_sample_sum_centered (p u : A → ℝ) (hp : ∑ a, p a = 1)
    (hu : expectation p u = 0) (n : ℕ) :
    expectation (fun s : Fin n → A => iidMass p s) (sampleSum u) = 0 := by
  rw [iid_sample_sum_mean p u hp n, hu, mul_zero]

theorem iid_sample_sum_square (p u : A → ℝ) (hp : ∑ a, p a = 1) (n : ℕ) :
    expectation (fun s : Fin n → A => iidMass p s) (fun s => (sampleSum u s) ^ 2) =
      (n : ℝ) * expectation p (fun a => (u a) ^ 2) +
        (n : ℝ) * ((n : ℝ) - 1) * (expectation p u) ^ 2 := by
  induction n with
  | zero => simp [expectation, sampleSum]
  | succ n ih =>
      rw [iid_expect_cons]
      simp only [sampleSum_cons, add_sq, expectation_add, expectation_mul_const,
        expectation_const_mul, expectation_const p hp]
      simp only [expectation_add, expectation_const_mul,
        expectation_const _ (iid_mass_normalized p hp n), iid_sample_sum_mean p u hp n, ih]
      push_cast
      ring

theorem iid_centered_sums_cross (p u v : A → ℝ) (hp : ∑ a, p a = 1)
    (hu : expectation p u = 0) (hv : expectation p v = 0) (n : ℕ) :
    expectation (fun s : Fin n → A => iidMass p s)
      (fun s => sampleSum u s * sampleSum v s) =
      (n : ℝ) * expectation p (fun a => u a * v a) := by
  induction n with
  | zero => simp [expectation, sampleSum]
  | succ n ih =>
      rw [iid_expect_cons]
      simp only [sampleSum_cons, add_mul, mul_add, expectation_add,
        expectation_mul_const, expectation_const_mul, expectation_const p hp, hu, hv,
        zero_mul, mul_zero, add_zero, zero_add]
      rw [expectation_const _ (iid_mass_normalized p hp n), ih]
      push_cast
      ring

theorem local_mixed_square (p u v : A → ℝ) (hp : ∑ a, p a = 1)
    (hu : expectation p u = 0) (hv : expectation p v = 0) (U V : ℝ) :
    expectation p (fun a => (u a + U) ^ 2 * (v a + V) ^ 2) =
      expectation p (fun a => (u a) ^ 2 * (v a) ^ 2) +
      expectation p (fun a => (u a) ^ 2) * V ^ 2 +
      expectation p (fun a => (v a) ^ 2) * U ^ 2 +
      4 * expectation p (fun a => u a * v a) * U * V +
      2 * expectation p (fun a => (u a) ^ 2 * v a) * V +
      2 * expectation p (fun a => u a * (v a) ^ 2) * U + U ^ 2 * V ^ 2 := by
  have heq : ∀ a, (u a + U) ^ 2 * (v a + V) ^ 2 =
      (u a) ^ 2 * (v a) ^ 2 + (u a) ^ 2 * V ^ 2 + (v a) ^ 2 * U ^ 2 +
      4 * (u a * v a) * U * V + 2 * ((u a) ^ 2 * v a) * V +
      2 * (u a * (v a) ^ 2) * U + (2 * U ^ 2 * V) * v a +
      (2 * V ^ 2 * U) * u a + U ^ 2 * V ^ 2 := by intro a; ring
  simp_rw [heq]
  simp only [expectation_add, expectation_mul_const, expectation_const_mul,
    expectation_const p hp, hu, hv, mul_zero, add_zero]

theorem iid_centered_sums_mixed_square (p u v : A → ℝ) (hp : ∑ a, p a = 1)
    (hu : expectation p u = 0) (hv : expectation p v = 0) (n : ℕ) :
    expectation (fun s : Fin n → A => iidMass p s)
      (fun s => (sampleSum u s) ^ 2 * (sampleSum v s) ^ 2) =
      (n : ℝ) * expectation p (fun a => (u a) ^ 2 * (v a) ^ 2) +
      (n : ℝ) * ((n : ℝ) - 1) *
        (expectation p (fun a => (u a) ^ 2) * expectation p (fun a => (v a) ^ 2) +
          2 * (expectation p (fun a => u a * v a)) ^ 2) := by
  induction n with
  | zero => simp [expectation, sampleSum]
  | succ n ih =>
      rw [iid_expect_cons]
      simp only [sampleSum_cons, local_mixed_square p u v hp hu hv]
      have hfactor : ∀ s : Fin n → A,
          4 * expectation p (fun a => u a * v a) * sampleSum u s * sampleSum v s =
          (4 * expectation p (fun a => u a * v a)) * (sampleSum u s * sampleSum v s) := by
        intro s; ring
      simp_rw [hfactor]
      simp only [expectation_add, expectation_const_mul,
        expectation_const _ (iid_mass_normalized p hp n),
        iid_sample_sum_square p u hp n, iid_sample_sum_square p v hp n,
        iid_centered_sums_cross p u v hp hu hv n,
        iid_sample_sum_centered p u hp hu n, iid_sample_sum_centered p v hp hv n,
        hu, hv, ih]
      push_cast
      ring

theorem local_sum_product_cross (p u v : A → ℝ) (hp : ∑ a, p a = 1)
    (hu : expectation p u = 0) (hv : expectation p v = 0) (U V W : ℝ) :
    expectation p (fun a => (u a * v a + W) * (u a + U) * (v a + V)) =
      expectation p (fun a => (u a) ^ 2 * (v a) ^ 2) +
      expectation p (fun a => (u a) ^ 2 * v a) * V +
      expectation p (fun a => u a * (v a) ^ 2) * U +
      expectation p (fun a => u a * v a) * (U * V + W) + W * U * V := by
  have heq : ∀ a, (u a * v a + W) * (u a + U) * (v a + V) =
      (u a) ^ 2 * (v a) ^ 2 + ((u a) ^ 2 * v a) * V + (u a * (v a) ^ 2) * U +
      (u a * v a) * (U * V + W) + (W * V) * u a + (W * U) * v a + W * U * V := by
    intro a; ring
  simp_rw [heq]
  simp only [expectation_add, expectation_mul_const, expectation_const_mul,
    expectation_const p hp, hu, hv, mul_zero, add_zero]

theorem iid_sum_product_cross (p u v : A → ℝ) (hp : ∑ a, p a = 1)
    (hu : expectation p u = 0) (hv : expectation p v = 0) (n : ℕ) :
    expectation (fun s : Fin n → A => iidMass p s)
      (fun s => sampleSum (fun a => u a * v a) s * sampleSum u s * sampleSum v s) =
      (n : ℝ) * expectation p (fun a => (u a) ^ 2 * (v a) ^ 2) +
        (n : ℝ) * ((n : ℝ) - 1) * (expectation p (fun a => u a * v a)) ^ 2 := by
  induction n with
  | zero => simp [expectation, sampleSum]
  | succ n ih =>
      rw [iid_expect_cons]
      simp only [sampleSum_cons, local_sum_product_cross p u v hp hu hv]
      simp only [expectation_add, expectation_const_mul,
        expectation_const _ (iid_mass_normalized p hp n),
        iid_sample_sum_centered p u hp hu n, iid_sample_sum_centered p v hp hv n,
        iid_centered_sums_cross p u v hp hu hv n,
        iid_sample_sum_mean p (fun a => u a * v a) hp n, ih]
      push_cast
      ring

theorem rloo_eq_sample_covariance {m : ℕ} (hm : m ≠ 0) (p r f : A → ℝ)
    (s : Fin (m + 1) → A) :
    iidRlooEstimator p r f s =
      ((m + 1 : ℝ) * sampleSum (fun a => r a * f a) s - sampleSum r s * sampleSum f s) /
        ((m : ℝ) * (m + 1 : ℝ)) := by
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  have hbase : ∀ i : Fin (m + 1), iidLooBaseline r (fun j => s (i.succAbove j)) =
      (sampleSum r s - r (s i)) / (m : ℝ) := by
    intro i
    unfold iidLooBaseline sampleSum
    congr 1
    have hb := Fin.sum_univ_succAbove (fun j => r (s j)) i
    linarith
  have hterm : ∀ i : Fin (m + 1), iidLooTerm p r f i s =
      ((m + 1 : ℝ) * (r (s i) * f (s i)) - sampleSum r s * f (s i)) / (m : ℝ) := by
    intro i
    unfold iidLooTerm
    rw [hbase]
    field_simp
    ring
  unfold iidRlooEstimator
  simp_rw [hterm]
  rw [← Finset.sum_div, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  change ((_ * sampleSum (fun a => r a * f a) s - _ * sampleSum f s) / _) / _ = _
  rw [div_div]

theorem rloo_reward_translation {m : ℕ} (hm : m ≠ 0) (p r f : A → ℝ) (c : ℝ)
    (s : Fin (m + 1) → A) :
    iidRlooEstimator p (fun a => r a - c) f s = iidRlooEstimator p r f s := by
  rw [rloo_eq_sample_covariance hm, rloo_eq_sample_covariance hm]
  have hsum : sampleSum (fun a => r a - c) s = sampleSum r s - (m + 1 : ℝ) * c := by
    simp [sampleSum, Finset.sum_sub_distrib]
  have hprod : sampleSum (fun a => (r a - c) * f a) s =
      sampleSum (fun a => r a * f a) s - c * sampleSum f s := by
    simp [sampleSum, sub_mul, Finset.sum_sub_distrib, Finset.mul_sum]
  rw [hsum, hprod]
  ring

theorem expectation_centered_square (p f : A → ℝ) (hp : ∑ a, p a = 1)
    (c : ℝ) (hc : expectation p f = c) :
    expectation p (fun a => (f a - c) ^ 2) = expectation p (fun a => (f a) ^ 2) - c ^ 2 := by
  simp only [sub_sq, expectation_add, expectation_sub, expectation_mul_const,
    expectation_const_mul, expectation_const p hp, hc]
  ring

/-- Exact finite-sample covariance variance, with n = m + 1 >= 2.
The second summand is the contribution of the shared leave-one-out baseline. -/
theorem centered_rloo_variance {m : ℕ} (hm : m ≠ 0) (p u v : A → ℝ)
    (hp : ∑ a, p a = 1) (hu : expectation p u = 0) (hv : expectation p v = 0) :
    expectation (fun s : Fin (m + 1) → A => iidMass p s)
      (fun s => (iidRlooEstimator p u v s - expectation p (fun a => u a * v a)) ^ 2) =
      (expectation p (fun a => (u a) ^ 2 * (v a) ^ 2) -
        (expectation p (fun a => u a * v a)) ^ 2) / (m + 1 : ℝ) +
      (expectation p (fun a => (u a) ^ 2) * expectation p (fun a => (v a) ^ 2) +
        (expectation p (fun a => u a * v a)) ^ 2) / ((m + 1 : ℝ) * (m : ℝ)) := by
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  have hn' : (m + 1 : ℝ) ≠ 0 := by positivity
  have hmean : expectation (fun s : Fin (m + 1) → A => iidMass p s)
      (iidRlooEstimator p u v) = expectation p (fun a => u a * v a) := by
    have hb := finite_iid_rloo_unbiased hm p u v hp hv
    simpa [expectation, mul_assoc] using hb
  rw [expectation_centered_square _ _ (iid_mass_normalized p hp (m + 1)) _ hmean]
  have hpoint : ∀ s : Fin (m + 1) → A, (iidRlooEstimator p u v s) ^ 2 =
      ((m + 1 : ℝ) ^ 2 * (sampleSum (fun a => u a * v a) s) ^ 2 -
        (2 * (m + 1 : ℝ)) *
          (sampleSum (fun a => u a * v a) s * sampleSum u s * sampleSum v s) +
        (sampleSum u s) ^ 2 * (sampleSum v s) ^ 2) / ((m : ℝ) * (m + 1 : ℝ)) ^ 2 := by
    intro s
    rw [rloo_eq_sample_covariance hm, div_pow]
    ring
  simp_rw [hpoint]
  rw [expectation_div_const, expectation_add, expectation_sub,
    expectation_const_mul, expectation_const_mul,
    iid_sample_sum_square p (fun a => u a * v a) hp (m + 1),
    iid_sum_product_cross p u v hp hu hv (m + 1),
    iid_centered_sums_mixed_square p u v hp hu hv (m + 1)]
  simp only [mul_pow, Nat.cast_add, Nat.cast_one]
  field_simp
  ring

theorem actual_rloo_variance {m : ℕ} (hm : m ≠ 0) (p r v : A → ℝ)
    (hp : ∑ a, p a = 1) (hv : expectation p v = 0) :
    let u := fun a => r a - expectation p r
    let θ := expectation p (fun a => u a * v a)
    expectation (fun s : Fin (m + 1) → A => iidMass p s)
      (fun s => (iidRlooEstimator p r v s - θ) ^ 2) =
      (expectation p (fun a => (u a) ^ 2 * (v a) ^ 2) - θ ^ 2) / (m + 1 : ℝ) +
      (expectation p (fun a => (u a) ^ 2) * expectation p (fun a => (v a) ^ 2) + θ ^ 2) /
        ((m + 1 : ℝ) * (m : ℝ)) := by
  dsimp only
  have hu : expectation p (fun a => r a - expectation p r) = 0 := by
    rw [expectation_sub, expectation_const p hp]
    ring
  have hb := centered_rloo_variance hm p (fun a => r a - expectation p r) v hp hu hv
  simpa only [rloo_reward_translation hm] using hb

theorem expectation_nonneg (p f : A → ℝ) (hp : ∀ a, 0 ≤ p a) (hf : ∀ a, 0 ≤ f a) :
    0 ≤ expectation p f := Finset.sum_nonneg fun a _ => mul_nonneg (hp a) (hf a)

theorem expectation_mono (p f g : A → ℝ) (hp : ∀ a, 0 ≤ p a) (hfg : ∀ a, f a ≤ g a) :
    expectation p f ≤ expectation p g :=
  Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (hfg a) (hp a)

theorem covariance_variance_coefficients_nonneg (p u v : A → ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1) :
    0 ≤ expectation p (fun a => (u a) ^ 2 * (v a) ^ 2) -
      (expectation p (fun a => u a * v a)) ^ 2 ∧
    0 ≤ expectation p (fun a => (u a) ^ 2) * expectation p (fun a => (v a) ^ 2) +
      (expectation p (fun a => u a * v a)) ^ 2 := by
  constructor
  · have hb := expectation_nonneg p
      (fun a => (u a * v a - expectation p (fun a => u a * v a)) ^ 2) hp0 (fun a => sq_nonneg _)
    rw [expectation_centered_square p (fun a => u a * v a) hp _ rfl] at hb
    simpa only [mul_pow] using hb
  · exact add_nonneg (mul_nonneg (expectation_nonneg p _ hp0 fun a => sq_nonneg (u a))
      (expectation_nonneg p _ hp0 fun a => sq_nonneg (v a))) (sq_nonneg _)

theorem expectation_abs_bound (p r : A → ℝ) (hp0 : ∀ a, 0 ≤ p a)
    (hp : ∑ a, p a = 1) (B : ℝ) (hr : ∀ a, |r a| ≤ B) : |expectation p r| ≤ B := by
  calc
    _ ≤ ∑ a, |p a * r a| := Finset.abs_sum_le_sum_abs _ _
    _ = expectation p (fun a => |r a|) := by simp [expectation, abs_mul, abs_of_nonneg (hp0 _)]
    _ ≤ expectation p (fun _ => B) := expectation_mono p _ _ hp0 hr
    _ = B := expectation_const p hp B

theorem bounded_reward_variance (p r : A → ℝ) (hp0 : ∀ a, 0 ≤ p a)
    (hp : ∑ a, p a = 1) (B : ℝ) (hB : 0 ≤ B) (hr : ∀ a, |r a| ≤ B) :
    expectation p (fun a => (r a - expectation p r) ^ 2) ≤ B ^ 2 := by
  rw [expectation_centered_square p r hp _ rfl]
  have hb : expectation p (fun a => (r a) ^ 2) ≤ B ^ 2 := by
    rw [← expectation_const p hp (B ^ 2)]
    apply expectation_mono p _ _ hp0
    intro a
    have h := (sq_le_sq₀ (abs_nonneg _) hB).mpr (hr a)
    simpa only [sq_abs] using h
  linarith [sq_nonneg (expectation p r)]

theorem bounded_reward_cross_moment (p r v : A → ℝ) (hp0 : ∀ a, 0 ≤ p a)
    (hp : ∑ a, p a = 1) (B : ℝ) (hB : 0 ≤ B) (hr : ∀ a, |r a| ≤ B) :
    expectation p (fun a => (r a - expectation p r) ^ 2 * (v a) ^ 2) ≤
      4 * B ^ 2 * expectation p (fun a => (v a) ^ 2) := by
  rw [← expectation_const_mul]
  apply expectation_mono p _ _ hp0
  intro a
  have ha : |r a - expectation p r| ≤ 2 * B :=
    (abs_sub _ _).trans (by linarith [hr a, expectation_abs_bound p r hp0 hp B hr])
  have hsq := (sq_le_sq₀ (abs_nonneg _) (by positivity : 0 ≤ 2 * B)).mpr ha
  rw [sq_abs] at hsq
  have hh : (r a - expectation p r) ^ 2 ≤ 4 * B ^ 2 := by nlinarith
  exact mul_le_mul_of_nonneg_right hh (sq_nonneg _)

def rlooVarianceFactor (m : ℕ) : ℝ := 4 / (m + 1 : ℝ) + 1 / ((m + 1 : ℝ) * (m : ℝ))

theorem rloo_variance_factor_rate {m : ℕ} (hm : m ≠ 0) :
    rlooVarianceFactor m ≤ 5 / (m + 1 : ℝ) := by
  have hmone : (1 : ℝ) ≤ m := by exact_mod_cast Nat.one_le_iff_ne_zero.mpr hm
  have hn : (0 : ℝ) < m + 1 := by positivity
  have hden : (m + 1 : ℝ) ≤ (m + 1 : ℝ) * (m : ℝ) := by nlinarith
  have hb := one_div_le_one_div_of_le hn hden
  unfold rlooVarianceFactor
  calc
    _ ≤ 4 / (m + 1 : ℝ) + 1 / (m + 1 : ℝ) := add_le_add_left hb _
    _ = _ := by ring

/-- The variance decays with sibling count, without pretending that the
leave-one-out contributions are independent. -/
theorem actual_rloo_variance_bound {m : ℕ} (hm : m ≠ 0) (p r v : A → ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1) (hv : expectation p v = 0)
    (B : ℝ) (hB : 0 ≤ B) (hr : ∀ a, |r a| ≤ B) :
    expectation (fun s : Fin (m + 1) → A => iidMass p s)
      (fun s => (iidRlooEstimator p r v s -
        expectation p (fun a => (r a - expectation p r) * v a)) ^ 2) ≤
      rlooVarianceFactor m * (B ^ 2 * expectation p (fun a => (v a) ^ 2)) := by
  rw [actual_rloo_variance hm p r v hp hv]
  have hmpos : (0 : ℝ) < m := by exact_mod_cast Nat.pos_of_ne_zero hm
  have hmone : (1 : ℝ) ≤ m := by exact_mod_cast Nat.one_le_iff_ne_zero.mpr hm
  have hnpos : (0 : ℝ) < m + 1 := by positivity
  have hden : (0 : ℝ) < (m + 1 : ℝ) * (m : ℝ) := mul_pos hnpos hmpos
  let θ := expectation p (fun a => (r a - expectation p r) * v a)
  let C := expectation p (fun a => (r a - expectation p r) ^ 2 * (v a) ^ 2)
  let V := expectation p (fun a => (r a - expectation p r) ^ 2)
  let E := expectation p (fun a => (v a) ^ 2)
  change (C - θ ^ 2) / (m + 1 : ℝ) + (V * E + θ ^ 2) / ((m + 1 : ℝ) * (m : ℝ)) ≤ _
  have hid : (C - θ ^ 2) / (m + 1 : ℝ) + (V * E + θ ^ 2) / ((m + 1 : ℝ) * (m : ℝ)) =
      C / (m + 1 : ℝ) + V * E / ((m + 1 : ℝ) * (m : ℝ)) -
        ((m : ℝ) - 1) * θ ^ 2 / ((m + 1 : ℝ) * (m : ℝ)) := by
    field_simp
    ring
  rw [hid]
  have hdrop : 0 ≤ ((m : ℝ) - 1) * θ ^ 2 / ((m + 1 : ℝ) * (m : ℝ)) :=
    div_nonneg (mul_nonneg (sub_nonneg.mpr hmone) (sq_nonneg _)) hden.le
  have hC := div_le_div_of_nonneg_right (bounded_reward_cross_moment p r v hp0 hp B hB hr) hnpos.le
  have hV := div_le_div_of_nonneg_right
    (mul_le_mul_of_nonneg_right (bounded_reward_variance p r hp0 hp B hB hr)
      (expectation_nonneg p _ hp0 fun a => sq_nonneg (v a))) hden.le
  change C / (m + 1 : ℝ) ≤ (4 * B ^ 2 * E) / (m + 1 : ℝ) at hC
  change V * E / ((m + 1 : ℝ) * (m : ℝ)) ≤ B ^ 2 * E / ((m + 1 : ℝ) * (m : ℝ)) at hV
  have hscale : rlooVarianceFactor m * (B ^ 2 * E) =
      (4 * B ^ 2 * E) / (m + 1 : ℝ) + B ^ 2 * E / ((m + 1 : ℝ) * (m : ℝ)) := by
    unfold rlooVarianceFactor
    ring
  change _ ≤ rlooVarianceFactor m * (B ^ 2 * E)
  rw [hscale]
  linarith

theorem expectation_square_shift (p f : A → ℝ) (hp : ∑ a, p a = 1)
    (c d : ℝ) (hc : expectation p f = c) :
    expectation p (fun a => (f a - d) ^ 2) =
      expectation p (fun a => (f a - c) ^ 2) + (c - d) ^ 2 := by
  rw [expectation_centered_square p f hp c hc]
  simp only [sub_sq, expectation_add, expectation_sub, expectation_mul_const,
    expectation_const_mul, expectation_const p hp, hc]
  ring

theorem iid_sample_mean_tail_of_variance {n : ℕ} (hn : n ≠ 0) (p f : A → ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1) (V η : ℝ) (hη : 0 < η)
    (hV : expectation p (fun a => (f a - expectation p f) ^ 2) ≤ V) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → A =>
      η ≤ |sampleMean f s - expectation p f|), iidMass p s) ≤ V / ((n : ℝ) * η ^ 2) := by
  have hvar := finite_iid_baseline_variance hn p f hp (expectation p f) rfl
  change expectation (iidMass p) (fun s : Fin n → A =>
    (sampleMean f s - expectation p f) ^ 2) = _ at hvar
  have hmse : expectation (iidMass p) (fun s : Fin n → A =>
      (sampleMean f s - expectation p f) ^ 2) ≤ V / (n : ℝ) := by
    rw [hvar]
    exact div_le_div_of_nonneg_right hV (Nat.cast_nonneg n)
  simpa only [div_div] using finite_squared_tail_bound (iidMass p)
    (fun s : Fin n → A => sampleMean f s - expectation p f) (iid_mass_nonneg p hp0)
    η (V / (n : ℝ)) hη hmse

end
end REINFORCEProMax.BatchReduction
