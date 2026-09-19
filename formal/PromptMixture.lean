import TreeSurrogate

/-! Finite prompt-mixture wrapper around the fixed-prompt tree theorems. -/
namespace REINFORCEProMax.AutoregressiveTree

open scoped BigOperators

variable {X A : Type*} [Fintype X] [Fintype A]

noncomputable def familyValue (μ : X → ℝ) (p : X → Policy A)
    (R : X → List A → ℝ) (n : ℕ) : ℝ :=
  ∑ x, μ x * value (p x) (R x) n []

noncomputable def familySurrogate (μ : X → ℝ) (p q : X → Policy A)
    (R : X → List A → ℝ) (n : ℕ) : ℝ :=
  ∑ x, μ x * surrogate (p x) (q x) (R x) n []

noncomputable def familyRemainder (μ : X → ℝ) (p q : X → Policy A)
    (R : X → List A → ℝ) (n : ℕ) : ℝ :=
  ∑ x, μ x * value (p x)
    (fun y => R x y * remainderRhs (ratioTrace (p x) (q x) y)) n []

theorem family_surrogate_remainder_identity
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0) (n : ℕ) :
    familyValue μ q R n - familyValue μ p R n =
      familySurrogate μ p q R n - familyRemainder μ p q R n := by
  unfold familyValue familySurrogate familyRemainder
  rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro x _
  have hx := trajectory_surrogate_remainder_identity (p x) (q x) (R x) (hs x) n
  have hsurr := trajectory_surrogate_eq (p x) (q x) (R x) (hs x) n
  rw [hsurr] at hx
  calc
    μ x * value (q x) (R x) n [] - μ x * value (p x) (R x) n [] =
        μ x * (value (q x) (R x) n [] - value (p x) (R x) n []) := by ring
    _ = μ x * (surrogate (p x) (q x) (R x) n [] -
        value (p x) (fun y => R x y * remainderRhs (ratioTrace (p x) (q x) y)) n []) := by rw [hx]
    _ = μ x * surrogate (p x) (q x) (R x) n [] -
        μ x * value (p x) (fun y => R x y * remainderRhs (ratioTrace (p x) (q x) y)) n [] := by ring

theorem family_remainder_quadratic
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0)
    (B ε : ℝ) (hB : 0 ≤ B) (hε : 0 ≤ ε)
    (hR : ∀ x h, |R x h| ≤ B) (hTV : ∀ x h, localTV (p x) (q x) h ≤ ε) (n : ℕ) :
    |familyRemainder μ p q R n| ≤ 2 * B * (n : ℝ) * ((n : ℝ) - 1) * ε ^ 2 := by
  unfold familyRemainder
  have hterm : ∀ x, |value (p x) (fun y => R x y * remainderRhs (ratioTrace (p x) (q x) y)) n []| ≤
      2 * B * (n : ℝ) * ((n : ℝ) - 1) * ε ^ 2 := by
    intro x
    exact (trajectory_remainder_bound (p x) (q x) (R x) (hs x) B ε hB hε
      (hR x) (hTV x) n).trans (min_le_left _ _)
  calc
    _ ≤ ∑ x, |μ x * value (p x)
        (fun y => R x y * remainderRhs (ratioTrace (p x) (q x) y)) n []| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ x, μ x * |value (p x)
        (fun y => R x y * remainderRhs (ratioTrace (p x) (q x) y)) n []| := by
      apply Finset.sum_congr rfl
      intro x _
      rw [abs_mul, abs_of_nonneg (hμ0 x)]
    _ ≤ ∑ x, μ x * (2 * B * (n : ℝ) * ((n : ℝ) - 1) * ε ^ 2) :=
      Finset.sum_le_sum fun x _ =>
        mul_le_mul_of_nonneg_left (hterm x) (hμ0 x)
    _ = 2 * B * (n : ℝ) * ((n : ℝ) - 1) * ε ^ 2 := by
      rw [← Finset.sum_mul, hμ]
      ring

theorem family_remainder_linear
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x)
    (hs : ∀ x h a, (p x).prob h a = 0 → (q x).prob h a = 0)
    (B : ℝ) (hR : ∀ x h, |R x h| ≤ B) (n : ℕ) :
    |familyRemainder μ p q R n| ≤
      4 * B * ∑ x, μ x * driftCost (p x) (q x) n [] := by
  unfold familyRemainder
  have hterm : ∀ x, |value (p x) (fun y => R x y * remainderRhs (ratioTrace (p x) (q x) y)) n []| ≤
      4 * B * driftCost (p x) (q x) n [] := by
    intro x
    have hid := trajectory_surrogate_remainder_identity (p x) (q x) (R x) (hs x) n
    have hsurr := trajectory_surrogate_eq (p x) (q x) (R x) (hs x) n
    rw [hsurr] at hid
    have hl := surrogate_error_linear (p x) (q x) (R x) B (hR x) n []
    have heq : value (q x) (R x) n [] - value (p x) (R x) n [] -
        surrogate (p x) (q x) (R x) n [] =
        -value (p x) (fun y => R x y * remainderRhs (ratioTrace (p x) (q x) y)) n [] := by
      linarith
    rw [heq, abs_neg] at hl
    exact hl
  calc
    _ ≤ ∑ x, |μ x * value (p x)
        (fun y => R x y * remainderRhs (ratioTrace (p x) (q x) y)) n []| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ x, μ x * |value (p x)
        (fun y => R x y * remainderRhs (ratioTrace (p x) (q x) y)) n []| := by
      apply Finset.sum_congr rfl
      intro x _
      rw [abs_mul, abs_of_nonneg (hμ0 x)]
    _ ≤ ∑ x, μ x * (4 * B * driftCost (p x) (q x) n []) :=
      Finset.sum_le_sum fun x _ =>
        mul_le_mul_of_nonneg_left (hterm x) (hμ0 x)
    _ = 4 * B * ∑ x, μ x * driftCost (p x) (q x) n [] := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro x _
      ring

end REINFORCEProMax.AutoregressiveTree
