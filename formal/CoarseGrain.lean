import Mathlib.Tactic

/-!
  Finite-support checks for the coarse-grained TV statements in
  `tex/main/5-theory.tex`.  The event `S` represents one cell of a binary
  partition; the normalization assumption is what supplies the factor 1/2.
-/
namespace REINFORCEProMax

theorem binary_tv_lower_bound
    {n : ℕ} (p q : Fin n → ℝ) (S : Finset (Fin n))
    (hsump : ∑ y : Fin n, p y = 1)
    (hsumq : ∑ y : Fin n, q y = 1) :
    |∑ y ∈ S, (p y - q y)| ≤
      (1 / 2 : ℝ) * ∑ y : Fin n, |p y - q y| := by
  let d : Fin n → ℝ := fun y => p y - q y
  have hsumd : (∑ y : Fin n, d y) = 0 := by
    dsimp [d]
    rw [Finset.sum_sub_distrib, hsump, hsumq]
    ring
  have hsplit := Finset.sum_add_sum_compl S d
  have hcomp : (∑ y ∈ Sᶜ, d y) = -∑ y ∈ S, d y := by
    have hzero : (∑ y : Fin n, d y) = 0 := hsumd
    linarith
  have hS : |∑ y ∈ S, d y| ≤ ∑ y ∈ S, |d y| :=
    Finset.abs_sum_le_sum_abs d S
  have hC : |∑ y ∈ Sᶜ, d y| ≤ ∑ y ∈ Sᶜ, |d y| :=
    Finset.abs_sum_le_sum_abs d Sᶜ
  have hC' : |∑ y ∈ S, d y| ≤ ∑ y ∈ Sᶜ, |d y| := by
    rw [hcomp, abs_neg] at hC
    exact hC
  have habssplit := Finset.sum_add_sum_compl S (fun y : Fin n => |d y|)
  have htwo :
      2 * |∑ y ∈ S, d y| ≤
        (∑ y ∈ S, |d y|) + (∑ y ∈ Sᶜ, |d y|) := by
    linarith
  have hfull :
      (∑ y ∈ S, |d y|) + (∑ y ∈ Sᶜ, |d y|) =
        ∑ y : Fin n, |d y| := habssplit
  dsimp [d] at htwo hfull ⊢
  rw [hfull] at htwo
  nlinarith

/- The corresponding finite arbitrary-partition statement.  `cell` maps each
   token to its partition cell; the left side is the TV distance after
   aggregating probabilities by that map. -/
theorem finite_coarse_tv_lower_bound
    {n m : ℕ} (p q : Fin n → ℝ) (cell : Fin n → Fin m) :
    (1 / 2 : ℝ) *
        ∑ j : Fin m,
          |∑ y ∈ (Finset.univ.filter (fun y : Fin n => cell y = j)),
            (p y - q y)| ≤
      (1 / 2 : ℝ) * ∑ y : Fin n, |p y - q y| := by
  let d : Fin n → ℝ := fun y => p y - q y
  have hcell : ∀ j : Fin m,
      |∑ y ∈ (Finset.univ.filter (fun y : Fin n => cell y = j)), d y| ≤
        ∑ y ∈ (Finset.univ.filter (fun y : Fin n => cell y = j)), |d y| := by
    intro j
    exact Finset.abs_sum_le_sum_abs d (Finset.univ.filter (fun y : Fin n => cell y = j))
  have hsum :
      (∑ j : Fin m,
        ∑ y ∈ (Finset.univ.filter (fun y : Fin n => cell y = j)), |d y|) =
        ∑ y : Fin n, |d y| := by
    simpa using (Finset.sum_fiberwise (s := (Finset.univ : Finset (Fin n)))
      cell (fun y : Fin n => |d y|))
  have htotal :
      (∑ j : Fin m,
        |∑ y ∈ (Finset.univ.filter (fun y : Fin n => cell y = j)), d y|) ≤
        ∑ y : Fin n, |d y| := by
    calc
      (∑ j : Fin m,
          |∑ y ∈ (Finset.univ.filter (fun y : Fin n => cell y = j)), d y|) ≤
          ∑ j : Fin m,
            ∑ y ∈ (Finset.univ.filter (fun y : Fin n => cell y = j)), |d y| := by
              exact Finset.sum_le_sum (fun j hj => hcell j)
      _ = ∑ y : Fin n, |d y| := hsum
  dsimp [d] at htotal ⊢
  nlinarith

def fiberSum {n m : ℕ} (cell : Fin n → Fin m) (f : Fin n → ℝ) (j : Fin m) : ℝ :=
  ∑ y ∈ (Finset.univ.filter (fun y : Fin n => cell y = j)), f y

theorem finite_tv_gap_slack_decomposition
    {n m : ℕ} (p q : Fin n → ℝ) (cell : Fin n → Fin m) :
    (1 / 2 : ℝ) * ∑ y : Fin n, |p y - q y| -
        (1 / 2 : ℝ) * ∑ j : Fin m, |fiberSum cell (fun y => p y - q y) j| =
      (1 / 2 : ℝ) * ∑ j : Fin m,
        (fiberSum cell (fun y => |p y - q y|) j -
          |fiberSum cell (fun y => p y - q y) j|) := by
  have hsum :
      (∑ j : Fin m, fiberSum cell (fun y => |p y - q y|) j) =
        ∑ y : Fin n, |p y - q y| := by
    simpa [fiberSum] using (Finset.sum_fiberwise
      (s := (Finset.univ : Finset (Fin n))) cell (fun y : Fin n => |p y - q y|))
  rw [← hsum]
  rw [Finset.sum_sub_distrib]
  ring

theorem finite_tv_gap_slack_nonneg
    {n m : ℕ} (p q : Fin n → ℝ) (cell : Fin n → Fin m) (j : Fin m) :
    0 ≤ fiberSum cell (fun y => |p y - q y|) j -
      |fiberSum cell (fun y => p y - q y) j| := by
  dsimp [fiberSum]
  have h := Finset.abs_sum_le_sum_abs
    (fun y : Fin n => p y - q y)
    (Finset.univ.filter (fun y : Fin n => cell y = j))
  linarith

theorem finite_tv_gap_slack_mass_bound
    {n m : ℕ} (p q : Fin n → ℝ) (cell : Fin n → Fin m)
    (hp : ∀ y, 0 ≤ p y) (hq : ∀ y, 0 ≤ q y) (j : Fin m) :
    fiberSum cell (fun y => |p y - q y|) j -
        |fiberSum cell (fun y => p y - q y) j| ≤
      fiberSum cell p j + fiberSum cell q j := by
  have habs : ∀ y : Fin n, |p y - q y| ≤ p y + q y := by
    intro y
    have h := abs_sub (p y) (q y)
    rw [abs_of_nonneg (hp y), abs_of_nonneg (hq y)] at h
    exact h
  have hsumabs :
      fiberSum cell (fun y => |p y - q y|) j ≤
        fiberSum cell (fun y => p y + q y) j := by
    dsimp [fiberSum]
    exact Finset.sum_le_sum (fun y hy => habs y)
  have hsplit : fiberSum cell (fun y => p y + q y) j =
      fiberSum cell p j + fiberSum cell q j := by
    dsimp [fiberSum]
    rw [Finset.sum_add_distrib]
  have hnonneg : 0 ≤ |fiberSum cell (fun y => p y - q y) j| := abs_nonneg _
  rw [hsplit] at hsumabs
  linarith

end REINFORCEProMax
