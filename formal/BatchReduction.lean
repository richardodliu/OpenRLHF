import RLOOMoments

namespace REINFORCEProMax.BatchReduction

open scoped BigOperators

noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

def expectation (p f : A → ℝ) : ℝ := ∑ a, p a * f a

def sampleMean {n : ℕ} (f : A → ℝ) (s : Fin n → A) : ℝ :=
  (∑ i, f (s i)) / (n : ℝ)

/-- A full group is the independent unit for RLOO, but changing the unit
does not turn the mean of response means into the ratio of group sums. -/
theorem group_ratio_ne_response_mean :
    (((1 : ℝ) / 1 + (-1) / 3) / 2 = 1 / 3) ∧
    (((1 : ℝ) + (-1)) / (1 + 3) = 0) ∧ (1 / 3 : ℝ) ≠ 0 := by
  norm_num

/-- Any finite joint batch law is allowed; no independence between length
and contribution, or among samples inside a batch, is used here. -/
theorem ratio_expectation_gap (p Z L : A → ℝ)
    (hL : ∀ a, L a ≠ 0) (hμ : expectation p L ≠ 0) :
    expectation p (fun a => Z a / L a) - expectation p Z / expectation p L =
      -expectation p (fun a => (L a - expectation p L) * (Z a / L a)) /
        expectation p L := by
  have hcov : expectation p (fun a => (L a - expectation p L) * (Z a / L a)) =
      expectation p Z - expectation p L * expectation p (fun a => Z a / L a) := by
    unfold expectation
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro a _
    field_simp [hL a]
    <;> ring
  rw [hcov]
  field_simp
  <;> ring

/-- A response mean estimates the expected per-response mean. An element
of A may itself contain a full dependent RLOO group. -/
theorem iid_sampleMean_expectation {n : ℕ} (hn : n ≠ 0) (p f : A → ℝ)
    (hp : ∑ a, p a = 1) :
    (∑ s : Fin n → A, iidMass p s * sampleMean f s) = expectation p f :=
  finite_iid_baseline_mean hn p f hp (expectation p f) rfl

theorem sampleMean_lower_bound {n : ℕ} (hn : n ≠ 0) (L : A → ℝ)
    (l : ℝ) (hL : ∀ a, l ≤ L a) (s : Fin n → A) : l ≤ sampleMean L s := by
  have hnpos : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero hn
  have hsum := Finset.sum_le_sum (s := (Finset.univ : Finset (Fin n)))
    (fun i _ => hL (s i))
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
  exact (le_div_iff₀ hnpos).mpr (by simpa [mul_comm] using hsum)

theorem sampleMean_residual {n : ℕ} (Z L : A → ℝ) (κ : ℝ) (s : Fin n → A) :
    sampleMean (fun a => Z a - κ * L a) s = sampleMean Z s - κ * sampleMean L s := by
  unfold sampleMean
  simp_rw [Finset.sum_sub_distrib, ← Finset.mul_sum]
  ring

theorem iid_mass_nonneg (p : A → ℝ) (hp : ∀ a, 0 ≤ p a) {n : ℕ} (s : Fin n → A) :
    0 ≤ iidMass p s := Finset.prod_nonneg fun i _ => hp (s i)

/-- Independent blocks, not independent tokens or sibling responses.
For block contribution Z, length L >= l > 0, and kappa = E Z / E L,
the self-normalized token mean has O(1/n) mean squared error. -/
theorem iid_token_mean_mse {n : ℕ} (hn : n ≠ 0) (p Z L : A → ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (l κ : ℝ) (hl : 0 < l) (hL : ∀ a, l ≤ L a)
    (hcenter : expectation p (fun a => Z a - κ * L a) = 0) :
    (∑ s : Fin n → A, iidMass p s * (sampleMean Z s / sampleMean L s - κ) ^ 2) ≤
      expectation p (fun a => (Z a - κ * L a) ^ 2) / ((n : ℝ) * l ^ 2) := by
  have hpoint : ∀ s : Fin n → A, (sampleMean Z s / sampleMean L s - κ) ^ 2 ≤
      (sampleMean (fun a => Z a - κ * L a) s) ^ 2 / l ^ 2 := by
    intro s
    have hb := sampleMean_lower_bound hn L l hL s
    have hpos : 0 < sampleMean L s := lt_of_lt_of_le hl hb
    have heq : sampleMean Z s / sampleMean L s - κ =
        sampleMean (fun a => Z a - κ * L a) s / sampleMean L s := by
      rw [sampleMean_residual]
      field_simp
      <;> ring
    rw [heq, div_pow]
    exact div_le_div_of_nonneg_left (sq_nonneg _) (sq_pos_of_pos hl)
      (sq_le_sq₀ hl.le hpos.le |>.mpr hb)
  calc
    _ ≤ ∑ s : Fin n → A, iidMass p s *
        ((sampleMean (fun a => Z a - κ * L a) s) ^ 2 / l ^ 2) :=
      Finset.sum_le_sum fun s _ => mul_le_mul_of_nonneg_left (hpoint s) (iid_mass_nonneg p hp0 s)
    _ = (∑ s : Fin n → A,
        iidMass p s * (sampleMean (fun a => Z a - κ * L a) s) ^ 2) / l ^ 2 := by
      simp only [← mul_div_assoc, Finset.sum_div]
    _ = _ := by
      have hv := finite_iid_baseline_variance hn p (fun a => Z a - κ * L a) hp 0 hcenter
      simp only [sub_zero] at hv
      change (∑ s : Fin n → A,
        iidMass p s * (iidLooBaseline (fun a => Z a - κ * L a) s) ^ 2) / l ^ 2 = _
      rw [hv]
      unfold expectation
      ring

/-- Finite Markov/Chebyshev step, retaining the actual finite event mass. -/
theorem finite_squared_tail_bound (p f : A → ℝ) (hp : ∀ a, 0 ≤ p a)
    (ε V : ℝ) (hε : 0 < ε) (hV : expectation p (fun a => (f a) ^ 2) ≤ V) :
    (∑ a ∈ Finset.univ.filter (fun a => ε ≤ |f a|), p a) ≤ V / ε ^ 2 := by
  have hevent : (∑ a ∈ Finset.univ.filter (fun a => ε ≤ |f a|), p a) * ε ^ 2 ≤
      expectation p (fun a => (f a) ^ 2) := by
    rw [Finset.sum_mul]
    calc
      _ ≤ ∑ a ∈ Finset.univ.filter (fun a => ε ≤ |f a|), p a * (f a) ^ 2 := by
        apply Finset.sum_le_sum
        intro a ha
        have h := (Finset.mem_filter.mp ha).2
        apply mul_le_mul_of_nonneg_left _ (hp a)
        nlinarith [sq_abs (f a)]
      _ ≤ _ := Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
        (fun a _ _ => mul_nonneg (hp a) (sq_nonneg _))
  exact (le_div_iff₀ (sq_pos_of_pos hε)).mpr (hevent.trans hV)

theorem iid_token_mean_tail {n : ℕ} (hn : n ≠ 0) (p Z L : A → ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (l κ ε : ℝ) (hl : 0 < l) (hε : 0 < ε) (hL : ∀ a, l ≤ L a)
    (hcenter : expectation p (fun a => Z a - κ * L a) = 0) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → A =>
      ε ≤ |sampleMean Z s / sampleMean L s - κ|), iidMass p s) ≤
      expectation p (fun a => (Z a - κ * L a) ^ 2) / ((n : ℝ) * l ^ 2 * ε ^ 2) := by
  have h := finite_squared_tail_bound (iidMass p)
    (fun s : Fin n → A => sampleMean Z s / sampleMean L s - κ)
    (iid_mass_nonneg p hp0) ε _ hε (iid_token_mean_mse hn p Z L hp0 hp l κ hl hL hcenter)
  simpa only [div_div] using h

/-- The residual centering required above follows from the ratio target. -/
theorem ratio_target_centered (p Z L : A → ℝ) (hμ : expectation p L ≠ 0) :
    expectation p (fun a => Z a - (expectation p Z / expectation p L) * L a) = 0 := by
  unfold expectation
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib]
  have hfactor : (∑ a, p a * ((∑ b, p b * Z b) / (∑ b, p b * L b) * L a)) =
      ((∑ b, p b * Z b) / (∑ b, p b * L b)) * (∑ a, p a * L a) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro a _
    ring
  rw [hfactor]
  change expectation p Z - (expectation p Z / expectation p L) * expectation p L = 0
  field_simp

theorem token_reduction_eq_mean_ratio {n : ℕ} (hn : n ≠ 0)
    (Z L : A → ℝ) (s : Fin n → A) :
    (∑ i, Z (s i)) / (∑ i, L (s i)) = sampleMean Z s / sampleMean L s := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  simp [sampleMean, div_div_div_cancel_right₀ hn']

theorem expectation_lower_bound (p L : A → ℝ) (hp0 : ∀ a, 0 ≤ p a)
    (hp : ∑ a, p a = 1) (l : ℝ) (hL : ∀ a, l ≤ L a) : l ≤ expectation p L := by
  calc
    l = ∑ a, p a * l := by rw [← Finset.sum_mul, hp, one_mul]
    _ ≤ _ := Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (hL a) (hp0 a)

/-- The MSE certificate with its population target constructed, rather
than supplied through a centering hypothesis. -/
theorem iid_token_mean_mse_of_ratio {n : ℕ} (hn : n ≠ 0) (p Z L : A → ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (l : ℝ) (hl : 0 < l) (hL : ∀ a, l ≤ L a) :
    let κ := expectation p Z / expectation p L
    (∑ s : Fin n → A, iidMass p s * (sampleMean Z s / sampleMean L s - κ) ^ 2) ≤
      expectation p (fun a => (Z a - κ * L a) ^ 2) / ((n : ℝ) * l ^ 2) := by
  have hμ : expectation p L ≠ 0 :=
    ne_of_gt (lt_of_lt_of_le hl (expectation_lower_bound p L hp0 hp l hL))
  exact iid_token_mean_mse hn p Z L hp0 hp l _ hl hL (ratio_target_centered p Z L hμ)

theorem microbatch_token_gap (b Z L : A → ℝ) (hL : ∀ a, L a ≠ 0)
    (hLtotal : (∑ a, L a) ≠ 0) :
    (∑ a, (b a / ∑ j, b j) * (Z a / L a)) - (∑ a, Z a) / (∑ a, L a) =
      ∑ a, (b a / (∑ j, b j) - L a / (∑ j, L j)) * (Z a / L a) := by
  have hpoint : ∀ a, (b a / (∑ j, b j) - L a / (∑ j, L j)) * (Z a / L a) =
      (b a / ∑ j, b j) * (Z a / L a) - Z a / (∑ j, L j) := by
    intro a
    rw [sub_mul]
    congr 1
    field_simp [hL a]
    <;> ring
  simp_rw [hpoint]
  rw [Finset.sum_sub_distrib, ← Finset.sum_div]

theorem sample_weighted_response_means (b Z : A → ℝ) (hb : ∀ a, b a ≠ 0) :
    (∑ a, (b a / ∑ j, b j) * (Z a / b a)) = (∑ a, Z a) / (∑ a, b a) := by
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro a _
  by_cases ht : (∑ j, b j) = 0
  · simp [ht]
  · field_simp [hb a, ht]
    <;> ring

/-- A concrete positive-length law where the expectation of a token mean
differs from the ratio of population totals. -/
theorem random_length_ratio_counterexample :
    expectation (fun _ : Bool => (1 / 2 : ℝ))
        (fun a => (if a then (3 : ℝ) else 0) / (if a then (3 : ℝ) else 1)) = 1 / 2 ∧
    expectation (fun _ : Bool => (1 / 2 : ℝ)) (fun a => if a then (3 : ℝ) else 0) /
        expectation (fun _ : Bool => (1 / 2 : ℝ)) (fun a => if a then (3 : ℝ) else 1) = 3 / 4 := by
  norm_num [expectation, Fintype.sum_bool]

/-- Sample-count weighting of two token-mean microbatches is not a global
token mean, even when each microbatch contains exactly one response. -/
theorem dynamic_token_weight_counterexample :
    ((1 : ℝ) / 2) * (0 / 1) + (1 / 2) * (3 / 3) = 1 / 2 ∧
    ((0 : ℝ) + 3) / (1 + 3) = 3 / 4 ∧ (1 / 2 : ℝ) ≠ 3 / 4 := by
  norm_num

end
end REINFORCEProMax.BatchReduction
