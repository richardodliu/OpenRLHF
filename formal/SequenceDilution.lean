import ProMax
import Mathlib.Analysis.SpecificLimits.Basic

namespace REINFORCEProMax.SequenceDilution
open scoped BigOperators Topology
open Filter

/-- The finite-horizon error retains all bounded background log ratios. -/
theorem bounded_background {n : ℕ} (z : Fin n → ℝ) (k : Fin n) (ε : ℝ)
    (hε : 0 ≤ ε) (hz : ∀ i, i ≠ k → |z i| ≤ ε) :
    |(∑ i, z i) / (n : ℝ) - z k / (n : ℝ)| ≤
      ((n : ℝ) - 1) / (n : ℝ) * ε ∧
      ((n : ℝ) - 1) / (n : ℝ) * ε ≤ ε := by
  classical
  have hn : 0 < n := Nat.zero_lt_of_lt k.isLt
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hc : (Finset.univ.erase k).card = n - 1 := by simp
  have hs : |∑ i ∈ Finset.univ.erase k, z i| ≤ ((n : ℝ) - 1) * ε := by
    calc
      _ ≤ ∑ i ∈ Finset.univ.erase k, |z i| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i ∈ Finset.univ.erase k, ε := Finset.sum_le_sum (fun i hi => hz i (Finset.mem_erase.mp hi).1)
      _ = _ := by simp [hc, Nat.cast_sub (by omega : 1 ≤ n)]
  have he := Finset.sum_erase_add (Finset.univ : Finset (Fin n)) z (Finset.mem_univ k)
  constructor
  · have hid : (∑ i, z i) / (n : ℝ) - z k / (n : ℝ) =
        (∑ i ∈ Finset.univ.erase k, z i) / (n : ℝ) := by rw [← he]; ring
    rw [hid, abs_div, abs_of_pos hn']
    convert div_le_div_of_nonneg_right hs hn'.le using 1
    ring
  · have hratio : ((n : ℝ) - 1) / (n : ℝ) ≤ 1 := (div_le_one hn').mpr (by linarith)
    simpa using mul_le_mul_of_nonneg_right hratio hε

/-- A fixed log-ratio spike disappears in the sequence geometric mean. -/
theorem fixed_spike_limit (x : ℝ) :
    Tendsto (fun n : ℕ => Real.exp (x / (n : ℝ))) atTop (𝓝 1) := by
  simpa using (tendsto_const_div_atTop_nhds_zero_nat x).rexp

/-- A rational enclosure certifies rounding the paper example to five decimals. -/
theorem numerical_dilution_enclosure :
    (1.000725 : ℝ) < Real.exp (3 / 4096) ∧ Real.exp (3 / 4096) < 1.000735 := by
  have hlo := Real.add_one_le_exp (3 / 4096 : ℝ)
  have hhi := Real.exp_bound_div_one_sub_of_interval (x := (3 / 4096 : ℝ)) (by norm_num) (by norm_num)
  constructor <;> norm_num at * <;> linarith

end REINFORCEProMax.SequenceDilution
