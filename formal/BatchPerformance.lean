import BatchReduction
import ProMaxObjective

namespace REINFORCEProMax.BatchReduction

open scoped BigOperators

/-- The sampling margin composes with a previously proved population
performance lower bound. The estimate can be the correction-adjusted ProMax
objective change, identified by Objective.corrected_objective_delta. -/
theorem performance_lower_bound_from_estimate (μ κ cost ε estimate gain : ℝ)
    (hμ : 0 ≤ μ) (hperf : μ * κ - cost ≤ gain)
    (hestimate : |estimate - κ| ≤ ε) :
    μ * (estimate - ε) - cost ≤ gain := by
  have h := (abs_le.mp hestimate).2
  have hm := mul_le_mul_of_nonneg_left (show estimate - ε ≤ κ by linarith) hμ
  linarith

theorem performance_positive_from_estimate (μ κ cost ε estimate gain : ℝ)
    (hμ : 0 < μ) (hperf : μ * κ - cost ≤ gain)
    (hestimate : |estimate - κ| ≤ ε) (haccept : cost / μ + ε < estimate) :
    0 < gain := by
  have hb := performance_lower_bound_from_estimate μ κ cost ε estimate gain hμ.le hperf hestimate
  have hm : cost < μ * (estimate - ε) := by
    have h : cost / μ < estimate - ε := by linarith
    have := (div_lt_iff₀ hμ).mp h
    nlinarith
  linarith

noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

/-- A finite-sample bound on falsely certifying improvement for a fixed
candidate policy, using independent evaluation blocks and its valid
population performance bound. It is not a uniform bound for a candidate
selected using these same blocks. -/
theorem false_improvement_certificate_bound {n : ℕ} (hn : n ≠ 0)
    (p Z L : A → ℝ) (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (l κ μ cost ε gain : ℝ) (hl : 0 < l) (hμ : 0 < μ) (hε : 0 < ε)
    (hL : ∀ a, l ≤ L a)
    (hcenter : expectation p (fun a => Z a - κ * L a) = 0)
    (hperf : μ * κ - cost ≤ gain) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → A => gain ≤ 0 ∧
        cost / μ + ε < sampleMean Z s / sampleMean L s), iidMass p s) ≤
      expectation p (fun a => (Z a - κ * L a) ^ 2) / ((n : ℝ) * l ^ 2 * ε ^ 2) := by
  have hsubset : Finset.univ.filter (fun s : Fin n → A => gain ≤ 0 ∧
      cost / μ + ε < sampleMean Z s / sampleMean L s) ⊆
      Finset.univ.filter (fun s : Fin n → A =>
        ε ≤ |sampleMean Z s / sampleMean L s - κ|) := by
    intro s hs
    have he := (Finset.mem_filter.mp hs).2
    have hκ : κ ≤ cost / μ := (le_div_iff₀ hμ).mpr (by nlinarith)
    apply Finset.mem_filter.mpr
    refine ⟨Finset.mem_univ _, ?_⟩
    have habs := le_abs_self (sampleMean Z s / sampleMean L s - κ)
    linarith
  exact (Finset.sum_le_sum_of_subset_of_nonneg hsubset
    (fun s _ _ => iid_mass_nonneg p hp0 s)).trans
      (iid_token_mean_tail hn p Z L hp0 hp l κ ε hl hε hL hcenter)

end
end REINFORCEProMax.BatchReduction
