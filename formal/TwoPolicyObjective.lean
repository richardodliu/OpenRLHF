import Mathlib.Tactic

/-! Two-policy, unclipped identities for the current Pro objective.
The mask may depend on the candidate policy. At the rollout reference,
the probability ratio is one and the prefix gate accepts all tokens.
These are finite algebraic components, not a formalization of training.
-/
namespace REINFORCEProMax.TwoPolicy
open scoped BigOperators

/-- The actual candidate gate is allowed to differ from the reference gate. -/
theorem candidate_gate_residual (rho M H A scale : ℝ) :
    scale * (rho - 1) * A - (M * rho * H - H) =
      (rho - 1) * (scale * A - H) + rho * (1 - M) * H := by
  ring

/-- Scaling discrepancy and rejected current-policy mass bound the residual. -/
theorem candidate_gate_residual_bound (rho M H A scale : ℝ)
    (hrho : 0 ≤ rho) (hM : M ≤ 1) :
    |scale * (rho - 1) * A - (M * rho * H - H)| ≤
      |rho - 1| * |scale * A - H| + rho * (1 - M) * |H| := by
  rw [candidate_gate_residual]
  calc
    |(rho - 1) * (scale * A - H) + rho * (1 - M) * H| ≤
        |(rho - 1) * (scale * A - H)| + |rho * (1 - M) * H| := abs_add _ _
    _ = |rho - 1| * |scale * A - H| + rho * (1 - M) * |H| := by
      rw [abs_mul, abs_mul, abs_of_nonneg (mul_nonneg hrho (sub_nonneg.mpr hM))]

/-- Pointwise objective decomposition, valid for candidate-dependent masks. -/
theorem objective_split (rho M H A : ℝ) :
    M * rho * H = rho * A - rho * (1 - M) * A + rho * M * (H - A) := by
  ring

/-- Restoring active-token counts gives the raw surrogate lower bound. -/
theorem scaled_raw_lower_bound (scale Z L increment distortion : ℝ)
    (hscale : 0 < scale)
    (hresidual : |scale * Z - L * increment| ≤ L * distortion) :
    L * (increment - distortion) / scale ≤ Z := by
  apply (div_le_iff₀ hscale).mpr
  have h := (abs_le.mp hresidual).1
  nlinarith
end REINFORCEProMax.TwoPolicy
