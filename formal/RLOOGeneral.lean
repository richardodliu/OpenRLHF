import ProMax

namespace REINFORCEProMax

open scoped BigOperators

noncomputable section

variable {α : Type*} [Fintype α] [DecidableEq α]

def iidMass {N : ℕ} (p : α → ℝ) (s : Fin N → α) : ℝ :=
  ∏ k : Fin N, p (s k)

def iidLooBaseline {m : ℕ} (r : α → ℝ) (z : Fin m → α) : ℝ :=
  (∑ j : Fin m, r (z j)) / (m : ℝ)

def iidLooTerm {m : ℕ} (p r f : α → ℝ) (i : Fin (m + 1))
    (s : Fin (m + 1) → α) : ℝ :=
  (r (s i) - iidLooBaseline r (fun j : Fin m => s (i.succAbove j))) * f (s i)

lemma iidMass_insertNth {m : ℕ} (p : α → ℝ) (i : Fin (m + 1))
    (y : α) (z : Fin m → α) :
    iidMass p (@Fin.insertNth m (fun _ : Fin (m + 1) => α) i y z) = p y * (∏ j : Fin m, p (z j)) := by
  change (∏ k : Fin (m + 1), p ((@Fin.insertNth m (fun _ : Fin (m + 1) => α) i y z) k)) = _
  rw [@Fin.prod_univ_succAbove ℝ _ m (fun k => p ((@Fin.insertNth m (fun _ : Fin (m + 1) => α) i y z) k)) i]
  rw [@Fin.insertNth_apply_same m (fun _ : Fin (m + 1) => α) i y z]
  congr 1
  apply Finset.prod_congr rfl
  intro j hj
  rw [@Fin.insertNth_apply_succAbove m (fun _ : Fin (m + 1) => α) i y z j]

lemma iidLooTerm_insertNth {m : ℕ} (p r f : α → ℝ) (i : Fin (m + 1))
    (y : α) (z : Fin m → α) :
    iidLooTerm p r f i (@Fin.insertNth m (fun _ : Fin (m + 1) => α) i y z) =
      (r y - (∑ j : Fin m, r (z j)) / (m : ℝ)) * f y := by
  simp [iidLooTerm, iidLooBaseline, Fin.insertNth_apply_same,
    Fin.insertNth_apply_succAbove]

lemma iid_term_expectation {m : ℕ} (hm : m ≠ 0)
    (p r f : α → ℝ) (hp : ∑ y : α, p y = 1)
    (hscore : ∑ y : α, p y * f y = 0)
    (i : Fin (m + 1)) :
    (∑ s : Fin (m + 1) → α, iidMass p s * iidLooTerm p r f i s) =
      ∑ y : α, p y * r y * f y := by
  let e := Fin.insertNthEquiv (fun _ : Fin (m + 1) => α) i
  let g : α × (Fin m → α) → ℝ := fun yz =>
    (p yz.1 * (∏ j : Fin m, p (yz.2 j))) *
      ((r yz.1 - (∑ j : Fin m, r (yz.2 j)) / (m : ℝ)) * f yz.1)
  have heq : (∑ yz : α × (Fin m → α), g yz) =
      ∑ s : Fin (m + 1) → α, iidMass p s * iidLooTerm p r f i s := by
    apply Fintype.sum_equiv e g (fun s => iidMass p s * iidLooTerm p r f i s)
    intro yz
    dsimp [g]
    change (p yz.1 * (∏ j : Fin m, p (yz.2 j))) *
      ((r yz.1 - (∑ j : Fin m, r (yz.2 j)) / (m : ℝ)) * f yz.1) =
      iidMass p (Fin.insertNth i yz.1 yz.2) * iidLooTerm p r f i (Fin.insertNth i yz.1 yz.2)
    rw [iidMass_insertNth, iidLooTerm_insertNth]
  rw [← heq]
  rw [Fintype.sum_prod_type]
  dsimp [g]
  have hprod : (∑ z : Fin m → α, ∏ j : Fin m, p (z j)) = 1 := by
    calc
      (∑ z : Fin m → α, ∏ j : Fin m, p (z j)) =
          ∑ z ∈ Fintype.piFinset (fun _ : Fin m => (Finset.univ : Finset α)),
            ∏ j : Fin m, p (z j) := by simp
      _ = ∏ j : Fin m, ∑ a ∈ (Finset.univ : Finset α), p a := by
        exact Finset.sum_prod_piFinset (s := (Finset.univ : Finset α))
          (g := fun _ a => p a)
      _ = 1 := by simp [hp]
  have hexpand :
      (∑ x : α, ∑ z : Fin m → α,
        (p x * ∏ j : Fin m, p (z j)) *
          ((r x - (∑ j : Fin m, r (z j)) / (m : ℝ)) * f x)) =
      ∑ x : α, ∑ z : Fin m → α,
        ((p x * (∏ j : Fin m, p (z j))) * r x * f x -
          (p x * (∏ j : Fin m, p (z j))) * ((∑ j : Fin m, r (z j)) / (m : ℝ)) * f x) := by
    apply Finset.sum_congr rfl
    intro x hx
    apply Finset.sum_congr rfl
    intro z hz
    ring
  rw [hexpand]
  simp_rw [Finset.sum_sub_distrib]
  have hfirst :
      (∑ x : α, ∑ z : Fin m → α,
        (p x * (∏ j : Fin m, p (z j))) * r x * f x) =
      ∑ x : α, p x * r x * f x := by
    calc
      (∑ x : α, ∑ z : Fin m → α,
          (p x * (∏ j : Fin m, p (z j))) * r x * f x) =
          ∑ x : α, (p x * r x * f x) *
            (∑ z : Fin m → α, ∏ j : Fin m, p (z j)) := by
              apply Finset.sum_congr rfl
              intro x hx
              calc
                (∑ z : Fin m → α, (p x * (∏ j : Fin m, p (z j))) * r x * f x) =
                    ∑ z : Fin m → α, (p x * r x * f x) * (∏ j : Fin m, p (z j)) := by
                      apply Finset.sum_congr rfl
                      intro z hz
                      ring
                _ = (p x * r x * f x) * (∑ z : Fin m → α, ∏ j : Fin m, p (z j)) := by
                      rw [Finset.mul_sum]
      _ = ∑ x : α, p x * r x * f x := by simp [hprod]
  have hsecond :
      (∑ x : α, ∑ z : Fin m → α,
        (p x * (∏ j : Fin m, p (z j))) * ((∑ j : Fin m, r (z j)) / (m : ℝ)) * f x) = 0 := by
    rw [Finset.sum_comm]
    apply Finset.sum_eq_zero
    intro z hz
    calc
      (∑ x : α, (p x * (∏ j : Fin m, p (z j))) *
          ((∑ j : Fin m, r (z j)) / (m : ℝ)) * f x) =
          ((∏ j : Fin m, p (z j)) * ((∑ j : Fin m, r (z j)) / (m : ℝ))) *
            (∑ x : α, p x * f x) := by
              calc
                (∑ x : α, (p x * (∏ j : Fin m, p (z j))) *
                    ((∑ j : Fin m, r (z j)) / (m : ℝ)) * f x) =
                    ∑ x : α, (p x * f x) *
                      ((∏ j : Fin m, p (z j)) * ((∑ j : Fin m, r (z j)) / (m : ℝ))) := by
                        apply Finset.sum_congr rfl
                        intro x hx
                        ring
                _ = (∑ x : α, p x * f x) *
                      ((∏ j : Fin m, p (z j)) * ((∑ j : Fin m, r (z j)) / (m : ℝ))) := by
                        rw [Finset.sum_mul]
                _ = ((∏ j : Fin m, p (z j)) * ((∑ j : Fin m, r (z j)) / (m : ℝ))) *
                      (∑ x : α, p x * f x) := by ring
      _ = 0 := by simp [hscore]
  rw [hfirst, hsecond, sub_zero]

def iidRlooEstimator {m : ℕ} (p r f : α → ℝ)
    (s : Fin (m + 1) → α) : ℝ :=
  (∑ i : Fin (m + 1), iidLooTerm p r f i s) / (m + 1 : ℝ)

theorem finite_iid_rloo_unbiased {m : ℕ} (hm : m ≠ 0)
    (p r f : α → ℝ) (hp : ∑ y : α, p y = 1)
    (hscore : ∑ y : α, p y * f y = 0) :
    (∑ s : Fin (m + 1) → α, iidMass p s * iidRlooEstimator p r f s) =
      ∑ y : α, p y * r y * f y := by
  unfold iidRlooEstimator
  simp_rw [div_eq_mul_inv]
  simp_rw [← mul_assoc]
  simp_rw [Finset.mul_sum]
  rw [← Finset.sum_mul]
  rw [Finset.sum_comm]
  have hterm : ∀ i : Fin (m + 1),
      (∑ s : Fin (m + 1) → α, iidMass p s * iidLooTerm p r f i s) =
        ∑ y : α, p y * r y * f y := by
    intro i
    exact iid_term_expectation hm p r f hp hscore i
  simp_rw [hterm]
  rw [Finset.sum_const]
  have hcast : (m + 1 : ℝ) ≠ 0 := by
    positivity
  field_simp

/- The policy score is generally vector-valued.  The scalar theorem above
   lifts componentwise to any finite-dimensional score space; this is the
   finite-support certificate used for the vector policy-gradient estimator. -/
theorem finite_iid_rloo_unbiased_vector {m d : ℕ} (hm : m ≠ 0)
    (p r : α → ℝ) (f : α → Fin d → ℝ) (hp : ∑ y : α, p y = 1)
    (hscore : ∀ k : Fin d, ∑ y : α, p y * f y k = 0) :
    ∀ k : Fin d,
      (∑ s : Fin (m + 1) → α,
        iidMass p s * iidRlooEstimator p r (fun y => f y k) s) =
        ∑ y : α, p y * r y * f y k := by
  intro k
  exact finite_iid_rloo_unbiased hm p r (fun y => f y k) hp (hscore k)

end
end REINFORCEProMax
