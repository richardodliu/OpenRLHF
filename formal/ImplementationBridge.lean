import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic
import BinaryUpdate

/-!
Small implementation bridges for the PPO/vLLM correction described in the
manuscript.  The production code stores log-probability differences, computes
the prefix gate after exponentiating its prefix average, and multiplies the
PPO clipped term by the detached old/rollout coefficient.  These lemmas keep
the algebraic part of that correspondence in the Lean audit. The small-signal
RLOO example also checks the implemented normalization fallback and its
noncentered token moments. Detachment and floating-point behaviour remain
implementation properties outside this model.
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


open scoped BigOperators

noncomputable def fallbackAdvantages (d : ℝ) : Fin 2 → ℝ :=
  AdaptiveMechanism.rlooSignal (![d, 0] : Fin 2 → ℝ)

def fallbackLengths : Fin 2 → ℝ := ![1, 2]

theorem fallback_rloo (d : ℝ) : fallbackAdvantages d = (![d, -d] : Fin 2 → ℝ) := by
  funext i
  fin_cases i <;> simp [fallbackAdvantages, AdaptiveMechanism.rlooSignal, Fin.sum_univ_two] <;> ring

noncomputable def fallbackFactors (d ε : ℝ) : ℝ × ℝ :=
  let P := AdaptiveMechanism.positiveTotal fallbackLengths (fallbackAdvantages d)
  let N := AdaptiveMechanism.negativeTotal fallbackLengths (fallbackAdvantages d)
  BinaryUpdate.safeguardedFactors (decide (P < ε ∨ N < ε)) P N
    (AdaptiveMechanism.positiveSquare fallbackLengths (fallbackAdvantages d))
    (AdaptiveMechanism.negativeSquare fallbackLengths (fallbackAdvantages d))
    (AdaptiveMechanism.nonzeroMass fallbackLengths (fallbackAdvantages d)) ε 100000000 10

/-- Both signs are present, but the actual positive token mass is below the
small-sum threshold, so the implementation returns the identity factors. -/
theorem small_signal_fallback (d ε : ℝ) (hd : 0 < d) (hsmall : d < ε) :
    fallbackFactors d ε = (1,1) := by
  unfold fallbackFactors
  rw [fallback_rloo]
  norm_num [AdaptiveMechanism.positiveTotal, AdaptiveMechanism.negativeTotal,
    fallbackLengths, Fin.sum_univ_succ, hd, not_lt_of_ge hd.le,
    BinaryUpdate.safeguardedFactors, hsmall]

noncomputable def fallbackSignal (d ε : ℝ) (i : Fin 2) : ℝ :=
  NormalizationSafeguards.signScale (fallbackFactors d ε).1 (fallbackFactors d ε).2
    (fallbackAdvantages d i)

theorem small_signal_output (d ε : ℝ) (hd : 0 < d) (hsmall : d < ε) :
    fallbackSignal d ε = (![d,-d] : Fin 2 → ℝ) := by
  funext i
  unfold fallbackSignal
  rw [small_signal_fallback d ε hd hsmall, fallback_rloo]
  fin_cases i <;> simp [NormalizationSafeguards.signScale, hd, not_lt_of_ge hd.le]

/-- A real RLOO pair has zero response sum but nonzero token mean after
fallback when the lengths are 1 and 2. The returned second moment is d². -/
theorem small_signal_fallback_moments (d ε : ℝ) (hd : 0 < d) (hsmall : d < ε) :
    (∑ i, fallbackAdvantages d i) = 0 ∧
    (∑ i, fallbackLengths i * fallbackSignal d ε i) / 3 = -d/3 ∧
    (∑ i, fallbackLengths i * (fallbackSignal d ε i)^2) / 3 = d^2 ∧
    (∑ i, fallbackLengths i * fallbackSignal d ε i) / 3 ≠ 0 := by
  rw [fallback_rloo, small_signal_output d ε hd hsmall]
  norm_num [fallbackLengths, Fin.sum_univ_two, ne_of_gt hd]
  constructor <;> ring_nf <;> simp [ne_of_gt hd]

end REINFORCEProMax.ImplementationBridge
