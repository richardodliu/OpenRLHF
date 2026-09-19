import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

/-!
Small implementation bridges for the PPO/vLLM correction described in the
manuscript.  The production code stores log-probability differences, computes
the prefix gate after exponentiating its prefix average, and multiplies the
PPO clipped term by the detached old/rollout coefficient.  These lemmas keep
the algebraic part of that correspondence in the Lean audit; detachment and
floating-point behaviour remain implementation properties outside this model.
-/
namespace REINFORCEProMax.ImplementationBridge

noncomputable def currentOldRatio (logCurrent logOld : ℝ) : ℝ := Real.exp (logCurrent - logOld)

noncomputable def oldRolloutRatio (logOld logRollout : ℝ) : ℝ := Real.exp (logOld - logRollout)

noncomputable def currentRolloutRatio (logCurrent logRollout : ℝ) : ℝ :=
  Real.exp (logCurrent - logRollout)

theorem current_rollout_ratio_factorization
    (logCurrent logOld logRollout : ℝ) :
    currentRolloutRatio logCurrent logRollout =
      currentOldRatio logCurrent logOld * oldRolloutRatio logOld logRollout := by
  unfold currentRolloutRatio currentOldRatio oldRolloutRatio
  rw [← Real.exp_add]
  congr 1
  ring

theorem prefix_exp_gate_iff_log_gate
    (x lo hi : ℝ) (hlo : 0 < lo) (hhi : 0 < hi) :
    (lo ≤ Real.exp x ∧ Real.exp x ≤ hi) ↔
      (Real.log lo ≤ x ∧ x ≤ Real.log hi) := by
  constructor
  · intro h
    constructor
    · exact (Real.log_le_iff_le_exp hlo).2 h.1
    · exact (Real.le_log_iff_exp_le hhi).2 h.2
  · intro h
    constructor
    · exact (Real.log_le_iff_le_exp hlo).1 h.1
    · exact (Real.le_log_iff_exp_le hhi).1 h.2

theorem prefix_gate_thresholds_match
    (x : ℝ) (epsLow epsHigh : ℝ)
    (hlow : 0 < 1 - epsLow) (hhigh : 0 < 1 + epsHigh) :
    ((1 - epsLow) ≤ Real.exp x ∧ Real.exp x ≤ (1 + epsHigh)) ↔
      (Real.log (1 - epsLow) ≤ x ∧ x ≤ Real.log (1 + epsHigh)) := by
  exact prefix_exp_gate_iff_log_gate x (1 - epsLow) (1 + epsHigh) hlow hhigh

end REINFORCEProMax.ImplementationBridge
