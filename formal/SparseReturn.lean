import Mathlib.Tactic

namespace REINFORCEProMax

/-- Reward-to-go recurrence used before token-level advantage normalization. -/
noncomputable def discountedReturn (gamma : ℝ) : List ℝ → ℝ
  | [] => 0
  | r :: rs => r + gamma * discountedReturn gamma rs

theorem sparse_terminal_return (gamma R : ℝ) (n : ℕ) :
    discountedReturn gamma (List.replicate n 0 ++ [R]) = gamma ^ n * R := by
  induction n with
  | zero => simp [discountedReturn]
  | succ n ih =>
      simp only [List.replicate_succ, List.cons_append, discountedReturn, zero_add, ih]
      ring

theorem sparse_terminal_return_gamma_one (R : ℝ) (n : ℕ) :
    discountedReturn 1 (List.replicate n 0 ++ [R]) = R := by
  simp [sparse_terminal_return]

/-- If an earlier token and the terminal token must both receive the same
nonzero reward, gamma = 1 is necessary as well as sufficient. -/
theorem gamma_one_iff_constant_sparse_return (gamma R : ℝ) (hR : R ≠ 0) :
    (∀ n : ℕ, discountedReturn gamma (List.replicate n 0 ++ [R]) = R) ↔ gamma = 1 := by
  constructor
  · intro h
    have h1 := h 1
    rw [sparse_terminal_return] at h1
    simpa [pow_one] using (mul_right_cancel₀ hR (h1.trans (one_mul R).symm))
  · rintro rfl
    exact sparse_terminal_return_gamma_one R

end REINFORCEProMax
