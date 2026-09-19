import RLOOGeneral

namespace REINFORCEProMax

noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

/-- Mean centering also has zero group sum; this is not exclusive to RLOO. -/
theorem mean_baseline_shaped_sum_zero {n : ℕ} (r : Fin n → ℝ) (hn : n ≠ 0) :
    (∑ i : Fin n, (r i - (∑ j : Fin n, r j) / (n : ℝ))) = 0 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  rw [Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

theorem iid_mass_normalized (p : A → ℝ) (hp : ∑ a, p a = 1) (n : ℕ) :
    (∑ z : Fin n → A, iidMass p z) = 1 := by
  unfold iidMass
  calc
    _ = ∑ z ∈ Fintype.piFinset (fun _ : Fin n => (Finset.univ : Finset A)),
        ∏ j : Fin n, p (z j) := by simp
    _ = ∏ j : Fin n, ∑ a ∈ (Finset.univ : Finset A), p a :=
      Finset.sum_prod_piFinset (s := Finset.univ) (g := fun _ a => p a)
    _ = 1 := by simp [hp]

/-- Separate any sampled response from the other m iid draws. -/
theorem iid_expect_insertNth {m : ℕ} (p : A → ℝ)
    (F : (Fin (m + 1) → A) → ℝ) (i : Fin (m + 1)) :
    (∑ s : Fin (m + 1) → A, iidMass p s * F s) =
      ∑ a : A, ∑ z : Fin m → A, p a * iidMass p z * F (Fin.insertNth i a z) := by
  rw [← Fintype.sum_prod_type']
  symm
  apply Fintype.sum_equiv (Fin.insertNthEquiv (fun _ : Fin (m + 1) => A) i)
  intro az
  change p az.1 * iidMass p az.2 * F (Fin.insertNth i az.1 az.2) =
    iidMass p (Fin.insertNth i az.1 az.2) * F (Fin.insertNth i az.1 az.2)
  rw [iidMass_insertNth]
  rfl

theorem iid_centered_sum_mean (p u : A → ℝ) (hp : ∑ a, p a = 1)
    (hu : ∑ a, p a * u a = 0) (n : ℕ) :
    (∑ z : Fin n → A, iidMass p z * (∑ j : Fin n, u (z j))) = 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [iid_expect_insertNth p _ 0]
      simp only [Fin.insertNth_zero', Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ]
      have hpoint : ∀ a, (∑ z : Fin n → A,
          p a * iidMass p z * (u a + ∑ j : Fin n, u (z j))) = p a * u a := by
        intro a
        calc
          _ = p a * u a * (∑ z : Fin n → A, iidMass p z) +
              p a * (∑ z : Fin n → A, iidMass p z * (∑ j : Fin n, u (z j))) := by
            rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
            apply Finset.sum_congr rfl
            intro z _
            ring
          _ = _ := by rw [iid_mass_normalized p hp, ih]; ring
      simp_rw [hpoint]
      exact hu

theorem iid_centered_sum_second_moment (p u : A → ℝ) (hp : ∑ a, p a = 1)
    (hu : ∑ a, p a * u a = 0) (n : ℕ) :
    (∑ z : Fin n → A, iidMass p z * (∑ j : Fin n, u (z j)) ^ 2) =
      (n : ℝ) * ∑ a, p a * (u a) ^ 2 := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [iid_expect_insertNth p _ 0]
      simp only [Fin.insertNth_zero', Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ]
      have hpoint : ∀ a, (∑ z : Fin n → A,
          p a * iidMass p z * (u a + ∑ j : Fin n, u (z j)) ^ 2) =
          p a * (u a) ^ 2 + p a * ((n : ℝ) * ∑ b, p b * (u b) ^ 2) := by
        intro a
        calc
          _ = p a * (u a) ^ 2 * (∑ z : Fin n → A, iidMass p z) +
              (2 * p a * u a) * (∑ z : Fin n → A, iidMass p z * (∑ j : Fin n, u (z j))) +
              p a * (∑ z : Fin n → A, iidMass p z * (∑ j : Fin n, u (z j)) ^ 2) := by
            rw [Finset.mul_sum, Finset.mul_sum, Finset.mul_sum,
              ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
            apply Finset.sum_congr rfl
            intro z _
            ring
          _ = _ := by rw [iid_mass_normalized p hp, iid_centered_sum_mean p u hp hu, ih]; ring
      simp_rw [hpoint]
      rw [Finset.sum_add_distrib, ← Finset.sum_mul, hp]
      push_cast
      ring

theorem finite_iid_baseline_mean {m : ℕ} (hm : m ≠ 0) (p r : A → ℝ)
    (hp : ∑ a, p a = 1) (μ : ℝ) (hμ : ∑ a, p a * r a = μ) :
    (∑ z : Fin m → A, iidMass p z * iidLooBaseline r z) = μ := by
  have hcenter : (∑ a, p a * (r a - μ)) = 0 := by
    simp_rw [mul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hp, hμ]
    ring
  have hsum := iid_centered_sum_mean p (fun a => r a - μ) hp hcenter m
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  have hpoint : ∀ z : Fin m → A,
      iidMass p z * (∑ j : Fin m, (r (z j) - μ)) =
        (m : ℝ) * (iidMass p z * iidLooBaseline r z - iidMass p z * μ) := by
    intro z
    rw [Finset.sum_sub_distrib]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, iidLooBaseline]
    field_simp
    <;> ring
  simp_rw [hpoint] at hsum
  rw [← Finset.mul_sum, Finset.sum_sub_distrib, ← Finset.sum_mul, iid_mass_normalized p hp] at hsum
  have hz := (mul_eq_zero.mp hsum).resolve_left hm'
  linarith

theorem finite_iid_baseline_variance {m : ℕ} (hm : m ≠ 0) (p r : A → ℝ)
    (hp : ∑ a, p a = 1) (μ : ℝ) (hμ : ∑ a, p a * r a = μ) :
    (∑ z : Fin m → A, iidMass p z * (iidLooBaseline r z - μ) ^ 2) =
      (∑ a, p a * (r a - μ) ^ 2) / (m : ℝ) := by
  have hcenter : (∑ a, p a * (r a - μ)) = 0 := by
    simp_rw [mul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hp, hμ]
    ring
  have hm' : (m : ℝ) ≠ 0 := by exact_mod_cast hm
  have hpoint : ∀ z : Fin m → A, iidLooBaseline r z - μ =
      (∑ j : Fin m, (r (z j) - μ)) / (m : ℝ) := by
    intro z
    rw [Finset.sum_sub_distrib]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, iidLooBaseline]
    field_simp
  simp_rw [hpoint, div_pow, ← mul_div_assoc]
  rw [← Finset.sum_div, iid_centered_sum_second_moment p (fun a => r a - μ) hp hcenter]
  field_simp
  <;> ring

/-- The leave-one-out baseline at any sample index has exactly the law of
the mean of m independent rewards. -/
theorem finite_loo_baseline_marginal {m : ℕ} (p r : A → ℝ)
    (hp : ∑ a, p a = 1) (i : Fin (m + 1)) (F : ℝ → ℝ) :
    (∑ s : Fin (m + 1) → A,
      iidMass p s * F (iidLooBaseline r (fun j => s (i.succAbove j)))) =
      ∑ z : Fin m → A, iidMass p z * F (iidLooBaseline r z) := by
  rw [iid_expect_insertNth p _ i]
  simp only [Fin.insertNth_apply_succAbove]
  simp_rw [mul_assoc, ← Finset.mul_sum]
  rw [← Finset.sum_mul, hp, one_mul]

theorem finite_loo_baseline_variance {m : ℕ} (hm : m ≠ 0) (p r : A → ℝ)
    (hp : ∑ a, p a = 1) (μ : ℝ) (hμ : ∑ a, p a * r a = μ) (i : Fin (m + 1)) :
    (∑ s : Fin (m + 1) → A,
      iidMass p s * (iidLooBaseline r (fun j => s (i.succAbove j)) - μ) ^ 2) =
      (∑ a, p a * (r a - μ) ^ 2) / (m : ℝ) := by
  rw [finite_loo_baseline_marginal p r hp i (fun b => (b - μ) ^ 2)]
  exact finite_iid_baseline_variance hm p r hp μ hμ

theorem finite_iid_baseline_squared_deviation {m : ℕ} (hm : m ≠ 0) (p r : A → ℝ)
    (hp : ∑ a, p a = 1) (μ : ℝ) (hμ : ∑ a, p a * r a = μ) (R : ℝ) :
    (∑ z : Fin m → A, iidMass p z * (R - iidLooBaseline r z) ^ 2) =
      (R - μ) ^ 2 + (∑ a, p a * (r a - μ) ^ 2) / (m : ℝ) := by
  have hzero : (∑ z : Fin m → A, iidMass p z * (iidLooBaseline r z - μ)) = 0 := by
    simp_rw [mul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, iid_mass_normalized p hp,
      finite_iid_baseline_mean hm p r hp μ hμ]
    ring
  calc
    _ = (R - μ) ^ 2 * (∑ z : Fin m → A, iidMass p z) +
        (∑ z : Fin m → A, iidMass p z * (iidLooBaseline r z - μ) ^ 2) -
        (2 * (R - μ)) * (∑ z : Fin m → A, iidMass p z * (iidLooBaseline r z - μ)) := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro z _
      ring
    _ = _ := by rw [iid_mass_normalized p hp, finite_iid_baseline_variance hm p r hp μ hμ, hzero]; ring

/-- Every entry of the per-sample score outer-product second moment, now
with the concrete leave-one-out variance sigma^2 / m substituted. -/
theorem finite_loo_score_cross_moment {m : ℕ} (hm : m ≠ 0) (p r f g : A → ℝ)
    (hp : ∑ a, p a = 1) (μ : ℝ) (hμ : ∑ a, p a * r a = μ) (i : Fin (m + 1)) :
    (∑ s : Fin (m + 1) → A, iidMass p s *
      ((r (s i) - iidLooBaseline r (fun j => s (i.succAbove j))) ^ 2 * f (s i) * g (s i))) =
      (∑ a, p a * (r a - μ) ^ 2 * f a * g a) +
      ((∑ a, p a * (r a - μ) ^ 2) / (m : ℝ)) * (∑ a, p a * f a * g a) := by
  rw [iid_expect_insertNth p _ i]
  simp only [Fin.insertNth_apply_same, Fin.insertNth_apply_succAbove]
  have hinner : ∀ a, (∑ z : Fin m → A,
      p a * iidMass p z * ((r a - iidLooBaseline r z) ^ 2 * f a * g a)) =
      p a * f a * g a * ((r a - μ) ^ 2 + (∑ b, p b * (r b - μ) ^ 2) / (m : ℝ)) := by
    intro a
    calc
      _ = (p a * f a * g a) *
          (∑ z : Fin m → A, iidMass p z * (r a - iidLooBaseline r z) ^ 2) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro z _
        ring
      _ = _ := by rw [finite_iid_baseline_squared_deviation hm p r hp μ hμ]
  simp_rw [hinner]
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro a _
  ring

end
end REINFORCEProMax
