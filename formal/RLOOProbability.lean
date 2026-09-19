import ProMax

namespace REINFORCEProMax

/- Finite product-space expectation.  `p` and `u` are the two marginal mass
functions; the joint mass is their product, so all independence is explicit. -/
def prodExpect {n m : ℕ} (p : Fin n → ℝ) (u : Fin m → ℝ)
    (h : Fin n → Fin m → ℝ) : ℝ :=
  ∑ i : Fin n, ∑ j : Fin m, p i * u j * h i j

lemma fintype_sum_swap {n m : ℕ} (h : Fin n → Fin m → ℝ) :
    (∑ i : Fin n, ∑ j : Fin m, h i j) = ∑ j : Fin m, ∑ i : Fin n, h i j := by
  rw [← Fintype.sum_prod_type']
  rw [← Fintype.sum_prod_type']
  apply Fintype.sum_equiv (Equiv.prodComm (Fin n) (Fin m))
  intro x
  rfl

lemma prod_expect_factor
    {n m : ℕ} (p : Fin n → ℝ) (u : Fin m → ℝ)
    (a : Fin n → ℝ) (b : Fin m → ℝ) :
    prodExpect p u (fun i j => a i * b j) =
      (∑ i : Fin n, p i * a i) * (∑ j : Fin m, u j * b j) := by
  dsimp [prodExpect]
  have hpoint : ∀ i : Fin n, ∀ j : Fin m,
      p i * u j * (a i * b j) = (p i * a i) * (u j * b j) := by
    intro i j
    ring
  calc
    (∑ i : Fin n, ∑ j : Fin m, p i * u j * (a i * b j)) =
        ∑ i : Fin n, ∑ j : Fin m, (p i * a i) * (u j * b j) := by
          apply Finset.sum_congr rfl
          intro i hi
          apply Finset.sum_congr rfl
          intro j hj
          exact hpoint i j
    _ = (∑ i : Fin n, p i * a i) * (∑ j : Fin m, u j * b j) := by
          rw [Fintype.sum_mul_sum]

/-- An independent baseline does not change a score expectation when the score
has zero mean.  This is the finite product-space form of the baseline control
variate identity. -/
theorem finite_independent_baseline_score
    {n m : ℕ} (p : Fin n → ℝ) (u : Fin m → ℝ)
    (r f : Fin n → ℝ) (b : Fin m → ℝ)
    (_hp : ∑ i : Fin n, p i = 1) (hu : ∑ j : Fin m, u j = 1)
    (hscore : ∑ i : Fin n, p i * f i = 0) :
    prodExpect p u (fun i j => (r i - b j) * f i) =
      ∑ i : Fin n, p i * r i * f i := by
  have hsplit :
      prodExpect p u (fun i j => (r i - b j) * f i) =
        prodExpect p u (fun i j => r i * f i) -
          prodExpect p u (fun i j => b j * f i) := by
    dsimp [prodExpect]
    calc
      (∑ i : Fin n, ∑ j : Fin m, p i * u j * ((r i - b j) * f i)) =
          ∑ i : Fin n, ∑ j : Fin m,
            (p i * u j * (r i * f i) - p i * u j * (b j * f i)) := by
              apply Finset.sum_congr rfl
              intro i hi
              apply Finset.sum_congr rfl
              intro j hj
              ring
      _ = (∑ i : Fin n, ∑ j : Fin m, p i * u j * (r i * f i)) -
          (∑ i : Fin n, ∑ j : Fin m, p i * u j * (b j * f i)) := by
            calc
              (∑ i : Fin n, ∑ j : Fin m,
                  (p i * u j * (r i * f i) - p i * u j * (b j * f i)))
                  = ∑ i : Fin n,
                      ((∑ j : Fin m, p i * u j * (r i * f i)) -
                       (∑ j : Fin m, p i * u j * (b j * f i))) := by
                    apply Finset.sum_congr rfl
                    intro i hi
                    rw [Finset.sum_sub_distrib]
              _ = (∑ i : Fin n, ∑ j : Fin m, p i * u j * (r i * f i)) -
                    (∑ i : Fin n, ∑ j : Fin m, p i * u j * (b j * f i)) := by
                    rw [Finset.sum_sub_distrib]
  rw [hsplit]
  have hfirst : prodExpect p u (fun i j => r i * f i) =
      (∑ i : Fin n, p i * r i * f i) := by
    dsimp [prodExpect]
    calc
      (∑ i : Fin n, ∑ j : Fin m, p i * u j * (r i * f i)) =
          ∑ i : Fin n, (p i * r i * f i) * (∑ j : Fin m, u j) := by
            apply Finset.sum_congr rfl
            intro i hi
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro j hj
            ring
      _ = ∑ i : Fin n, p i * r i * f i := by simp [hu]
  have hsecond : prodExpect p u (fun i j => b j * f i) = 0 := by
    dsimp [prodExpect]
    calc
      (∑ i : Fin n, ∑ j : Fin m, p i * u j * (b j * f i)) =
          ∑ i : Fin n, (p i * f i) * (∑ j : Fin m, u j * b j) := by
            apply Finset.sum_congr rfl
            intro i hi
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro j hj
            ring
      _ = 0 := by
            rw [← Finset.sum_mul, hscore, zero_mul]
  rw [hfirst, hsecond, sub_zero]

/- The corresponding second-moment identity is also a finite algebraic
statement.  `u` is the independent baseline law; its mean is `μ`. -/
theorem finite_independent_baseline_second_moment
    {n m : ℕ} (p : Fin n → ℝ) (u : Fin m → ℝ)
    (r f : Fin n → ℝ) (b : Fin m → ℝ) (μ : ℝ)
    (hu : ∑ j : Fin m, u j = 1)
    (hmean : ∑ j : Fin m, u j * b j = μ) :
    prodExpect p u (fun i j => (r i - b j)^2 * (f i)^2) =
      (∑ i : Fin n, p i * (r i - μ)^2 * (f i)^2) +
      (∑ j : Fin m, u j * (b j - μ)^2) * (∑ i : Fin n, p i * (f i)^2) := by
  have hzero : ∑ j : Fin m, u j * (b j - μ) = 0 := by
    calc
      (∑ j : Fin m, u j * (b j - μ)) =
          ∑ j : Fin m, (u j * b j - u j * μ) := by
            apply Finset.sum_congr rfl
            intro j _
            ring
      _ = (∑ j : Fin m, u j * b j) - ∑ j : Fin m, u j * μ := by
            rw [Finset.sum_sub_distrib]
      _ = 0 := by
            rw [hmean, ← Finset.sum_mul, hu]
            ring
  have hfirst : prodExpect p u (fun i j => (r i - μ)^2 * (f i)^2) =
      ∑ i : Fin n, p i * (r i - μ)^2 * (f i)^2 := by
    dsimp [prodExpect]
    calc
      (∑ i : Fin n, ∑ j : Fin m,
          p i * u j * ((r i - μ)^2 * (f i)^2)) =
          ∑ i : Fin n, (p i * (r i - μ)^2 * (f i)^2) *
            (∑ j : Fin m, u j) := by
            apply Finset.sum_congr rfl
            intro i _
            calc
              (∑ j : Fin m, p i * u j * ((r i - μ)^2 * (f i)^2)) =
                  ∑ j : Fin m, (p i * (r i - μ)^2 * (f i)^2) * u j := by
                    apply Finset.sum_congr rfl
                    intro j _
                    ring
              _ = (p i * (r i - μ)^2 * (f i)^2) * (∑ j : Fin m, u j) := by
                    rw [Finset.mul_sum]
      _ = ∑ i : Fin n, p i * (r i - μ)^2 * (f i)^2 := by simp [hu]
  have hsecond : prodExpect p u (fun i j => (b j - μ)^2 * (f i)^2) =
      (∑ j : Fin m, u j * (b j - μ)^2) * (∑ i : Fin n, p i * (f i)^2) := by
    dsimp [prodExpect]
    rw [Finset.sum_comm]
    calc
      (∑ j : Fin m, ∑ i : Fin n,
          p i * u j * ((b j - μ)^2 * (f i)^2)) =
          ∑ j : Fin m, (u j * (b j - μ)^2) *
            (∑ i : Fin n, p i * (f i)^2) := by
            apply Finset.sum_congr rfl
            intro j _
            calc
              (∑ i : Fin n, p i * u j * ((b j - μ)^2 * (f i)^2)) =
                  ∑ i : Fin n, (u j * (b j - μ)^2) * (p i * (f i)^2) := by
                    apply Finset.sum_congr rfl
                    intro i _
                    ring
              _ = (u j * (b j - μ)^2) * (∑ i : Fin n, p i * (f i)^2) := by
                    rw [Finset.mul_sum]
      _ = _ := by rw [Finset.sum_mul]
  have hcross : prodExpect p u (fun i j =>
      2 * (r i - μ) * (b j - μ) * (f i)^2) = 0 := by
    dsimp [prodExpect]
    calc
      (∑ i : Fin n, ∑ j : Fin m,
          p i * u j * (2 * (r i - μ) * (b j - μ) * (f i)^2)) =
          ∑ i : Fin n, (p i * (2 * (r i - μ) * (f i)^2)) *
            (∑ j : Fin m, u j * (b j - μ)) := by
            apply Finset.sum_congr rfl
            intro i _
            calc
              (∑ j : Fin m, p i * u j *
                  (2 * (r i - μ) * (b j - μ) * (f i)^2)) =
                  ∑ j : Fin m, (p i * (2 * (r i - μ) * (f i)^2)) *
                    (u j * (b j - μ)) := by
                    apply Finset.sum_congr rfl
                    intro j _
                    ring
              _ = (p i * (2 * (r i - μ) * (f i)^2)) *
                    (∑ j : Fin m, u j * (b j - μ)) := by
                    rw [Finset.mul_sum]
      _ = 0 := by simp [hzero]
  have hexpand : prodExpect p u (fun i j => (r i - b j)^2 * (f i)^2) =
      prodExpect p u (fun i j => (r i - μ)^2 * (f i)^2) +
        prodExpect p u (fun i j => (b j - μ)^2 * (f i)^2) -
        prodExpect p u (fun i j => 2 * (r i - μ) * (b j - μ) * (f i)^2) := by
    dsimp [prodExpect]
    calc
      (∑ i : Fin n, ∑ j : Fin m,
          p i * u j * ((r i - b j)^2 * (f i)^2)) =
          ∑ i : Fin n, ∑ j : Fin m,
            (p i * u j * ((r i - μ)^2 * (f i)^2) +
              p i * u j * ((b j - μ)^2 * (f i)^2) -
              p i * u j * (2 * (r i - μ) * (b j - μ) * (f i)^2)) := by
            apply Finset.sum_congr rfl
            intro i _
            apply Finset.sum_congr rfl
            intro j _
            ring
      _ = (∑ i : Fin n, ∑ j : Fin m,
            p i * u j * ((r i - μ)^2 * (f i)^2)) +
          (∑ i : Fin n, ∑ j : Fin m,
            p i * u j * ((b j - μ)^2 * (f i)^2)) -
          (∑ i : Fin n, ∑ j : Fin m,
            p i * u j * (2 * (r i - μ) * (b j - μ) * (f i)^2)) := by
            simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib]
  rw [hexpand, hfirst, hsecond, hcross]
  ring

/- A bilinear version of the second-moment decomposition.  Taking `g = f`
   recovers the scalar theorem above; taking `f` and `g` to be coordinates of
   a vector score gives every entry of the corresponding outer-product
   identity. -/
theorem finite_independent_baseline_cross_moment
    {n m : ℕ} (p : Fin n → ℝ) (u : Fin m → ℝ)
    (r f g : Fin n → ℝ) (b : Fin m → ℝ) (μ : ℝ)
    (hu : ∑ j : Fin m, u j = 1)
    (hmean : ∑ j : Fin m, u j * b j = μ) :
    prodExpect p u (fun i j => (r i - b j)^2 * f i * g i) =
      (∑ i : Fin n, p i * (r i - μ)^2 * f i * g i) +
      (∑ j : Fin m, u j * (b j - μ)^2) * (∑ i : Fin n, p i * f i * g i) := by
  have hzero : ∑ j : Fin m, u j * (b j - μ) = 0 := by
    calc
      (∑ j : Fin m, u j * (b j - μ)) =
          ∑ j : Fin m, (u j * b j - u j * μ) := by
            apply Finset.sum_congr rfl
            intro j _
            ring
      _ = (∑ j : Fin m, u j * b j) - ∑ j : Fin m, u j * μ := by
            rw [Finset.sum_sub_distrib]
      _ = 0 := by
            rw [hmean, ← Finset.sum_mul, hu]
            ring
  have hfirst : prodExpect p u (fun i j => (r i - μ)^2 * f i * g i) =
      ∑ i : Fin n, p i * (r i - μ)^2 * f i * g i := by
    have h := prod_expect_factor p u
      (fun i => (r i - μ)^2 * f i * g i) (fun _ => (1 : ℝ))
    simpa [prodExpect, hu, mul_assoc] using h
  have hsecond : prodExpect p u (fun i j => (b j - μ)^2 * f i * g i) =
      (∑ j : Fin m, u j * (b j - μ)^2) * (∑ i : Fin n, p i * f i * g i) := by
    have h := prod_expect_factor p u
      (fun i => f i * g i) (fun j => (b j - μ)^2)
    simpa [prodExpect, mul_comm, mul_left_comm, mul_assoc] using h
  have hcross : prodExpect p u (fun i j =>
      2 * (r i - μ) * (b j - μ) * f i * g i) = 0 := by
    have h := prod_expect_factor p u
      (fun i => (r i - μ) * f i * g i) (fun j => 2 * (b j - μ))
    calc
      prodExpect p u (fun i j =>
          2 * (r i - μ) * (b j - μ) * f i * g i) =
          (∑ i : Fin n, p i * ((r i - μ) * f i * g i)) *
            (∑ j : Fin m, u j * (2 * (b j - μ))) := by
              simpa [prodExpect, mul_comm, mul_left_comm, mul_assoc] using h
      _ = 0 := by
            have hsum : (∑ j : Fin m, u j * (2 * (b j - μ))) = 0 := by
              calc
                (∑ j : Fin m, u j * (2 * (b j - μ))) =
                    (∑ j : Fin m, u j * (b j - μ)) * 2 := by
                      rw [Finset.sum_mul]
                      apply Finset.sum_congr rfl
                      intro j _
                      ring
                _ = 0 := by rw [hzero, zero_mul]
            rw [hsum, mul_zero]
  have hexpand : prodExpect p u (fun i j => (r i - b j)^2 * f i * g i) =
      prodExpect p u (fun i j => (r i - μ)^2 * f i * g i) +
        prodExpect p u (fun i j => (b j - μ)^2 * f i * g i) -
        prodExpect p u (fun i j => 2 * (r i - μ) * (b j - μ) * f i * g i) := by
    dsimp [prodExpect]
    calc
      (∑ i : Fin n, ∑ j : Fin m,
          p i * u j * ((r i - b j)^2 * f i * g i)) =
          ∑ i : Fin n, ((∑ j : Fin m,
            p i * u j * ((r i - μ)^2 * f i * g i)) +
            (∑ j : Fin m,
              p i * u j * ((b j - μ)^2 * f i * g i)) -
            (∑ j : Fin m,
              p i * u j * (2 * (r i - μ) * (b j - μ) * f i * g i))) := by
                apply Finset.sum_congr rfl
                intro i _
                rw [← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
                apply Finset.sum_congr rfl
                intro j _
                ring
      _ = (∑ i : Fin n, ∑ j : Fin m,
          p i * u j * ((r i - μ)^2 * f i * g i)) +
          (∑ i : Fin n, ∑ j : Fin m,
            p i * u j * ((b j - μ)^2 * f i * g i)) -
          (∑ i : Fin n, ∑ j : Fin m,
            p i * u j * (2 * (r i - μ) * (b j - μ) * f i * g i)) := by
                rw [Finset.sum_sub_distrib, Finset.sum_add_distrib]
  rw [hexpand, hfirst, hsecond, hcross]
  ring

/- A finite arbitrary-n RLOO interface.  Each `u i` is the finite law of the
   leave-one-out baseline for coordinate `i`; independence from coordinate `i`
   is represented by the product mass in `prodExpect`.  In an iid model the
   actual average of the other `n-1` rewards supplies such a law, with its
   normalization and mean proved separately from the product construction. -/
theorem finite_rloo_independent_baselines
    {n m : ℕ} (p : Fin n → ℝ) (u : Fin n → Fin m → ℝ)
    (r f : Fin n → ℝ) (b : Fin n → Fin m → ℝ)
    (hp : ∑ i : Fin n, p i = 1)
    (hu : ∀ i, ∑ j : Fin m, u i j = 1)
    (hn : n ≠ 0)
    (hscore : ∑ i : Fin n, p i * f i = 0) :
    (∑ i : Fin n, prodExpect p (u i)
      (fun y z => (r y - b i z) * f y)) / (n : ℝ) =
      ∑ y : Fin n, p y * r y * f y := by
  have hterm : ∀ i : Fin n,
      prodExpect p (u i) (fun y z => (r y - b i z) * f y) =
        ∑ y : Fin n, p y * r y * f y := by
    intro i
    exact finite_independent_baseline_score p (u i) r f (b i) hp (hu i) hscore
  simp_rw [hterm]
  simp only [Finset.sum_const, Finset.card_fin, nsmul_eq_mul]
  field_simp [hn]



/- The two-sample iid construction.  This is a complete finite product-space
   theorem: each sample uses the other sample as its leave-one-out baseline.
   It provides an explicit iid certificate while the arbitrary-n statement in
   the paper still requires the corresponding finite product construction. -/
theorem finite_iid_rloo_unbiased_two
    {n : ℕ} (p : Fin n → ℝ) (r f : Fin n → ℝ)
    (hp : ∑ i : Fin n, p i = 1)
    (hscore : ∑ i : Fin n, p i * f i = 0) :
    prodExpect p p (fun i j =>
      ((r i - r j) * f i + (r j - r i) * f j) / 2) =
      ∑ i : Fin n, p i * r i * f i := by
  have hfirst : prodExpect p p (fun i j => (r i - r j) * f i) =
      ∑ i : Fin n, p i * r i * f i := by
    exact finite_independent_baseline_score p p r f r hp hp hscore
  have hsecond : prodExpect p p (fun i j => (r j - r i) * f j) =
      ∑ i : Fin n, p i * r i * f i := by
    dsimp [prodExpect]
    rw [fintype_sum_swap]
    have hfirst' :
        (∑ i : Fin n, ∑ j : Fin n, p i * p j * ((r i - r j) * f i)) =
          ∑ i : Fin n, p i * r i * f i := by
      simpa [prodExpect] using hfirst
    simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc,
      mul_comm, mul_left_comm, mul_assoc] using hfirst'
  have hsplit : prodExpect p p (fun i j =>
      ((r i - r j) * f i + (r j - r i) * f j) / 2) =
      (prodExpect p p (fun i j => (r i - r j) * f i) +
        prodExpect p p (fun i j => (r j - r i) * f j)) / 2 := by
    dsimp [prodExpect]
    calc
      (∑ i : Fin n, ∑ j : Fin n,
          p i * p j * (((r i - r j) * f i + (r j - r i) * f j) / 2)) =
          (∑ i : Fin n, ∑ j : Fin n,
            p i * p j * ((r i - r j) * f i) / 2) +
          (∑ i : Fin n, ∑ j : Fin n,
            p i * p j * ((r j - r i) * f j) / 2) := by
              rw [← Finset.sum_add_distrib]
              apply Finset.sum_congr rfl
              intro i _
              rw [← Finset.sum_add_distrib]
              apply Finset.sum_congr rfl
              intro j _
              ring
      _ = (prodExpect p p (fun i j => (r i - r j) * f i) +
          prodExpect p p (fun i j => (r j - r i) * f j)) / 2 := by
            simp only [prodExpect, div_eq_mul_inv]
            simp_rw [← Finset.sum_mul (s := Finset.univ)]
            ring
  rw [hsplit, hfirst, hsecond]
  ring

end REINFORCEProMax
