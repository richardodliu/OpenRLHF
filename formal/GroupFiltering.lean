import Mathlib.Tactic

/-! Exact counterexample to reusing the iid RLOO identity after score-based
whole-group rejection. Two Bernoulli(1/2) responses have reward equal to score;
the default open interval (0,1) retains exactly the mixed pairs. -/
namespace REINFORCEProMax.GroupFiltering
open scoped BigOperators
noncomputable section

def reward (a : Bool) : ℝ := if a then 1 else 0
-- Score with respect to Bernoulli probability theta at theta=1/2.
def score (a : Bool) : ℝ := if a then 2 else -2

def accepted (a b : Bool) : Prop := 0 < (reward a + reward b) / 2 ∧
  (reward a + reward b) / 2 < 1

local instance (a b : Bool) : Decidable (accepted a b) := Classical.propDecidable _

def mass (a b : Bool) : ℝ := if accepted a b then 1 / 2 else 0

def rloo (a b : Bool) : ℝ :=
  ((reward a - reward b) * score a + (reward b - reward a) * score b) / 2

theorem default_gate_iff_mixed (a b : Bool) : accepted a b ↔ a ≠ b := by
  cases a <;> cases b <;> norm_num [accepted, reward]

theorem acceptance_probability :
    (∑ a : Bool, ∑ b : Bool, if accepted a b then (1 / 4 : ℝ) else 0) = 1 / 2 := by
  norm_num [Fintype.sum_bool, accepted, reward]

theorem retained_mass_normalized : (∑ a : Bool, ∑ b : Bool, mass a b) = 1 := by
  norm_num [Fintype.sum_bool, mass, accepted, reward]

theorem retained_mass_nonneg (a b : Bool) : 0 ≤ mass a b := by
  cases a <;> cases b <;> norm_num [mass, accepted, reward]

theorem retained_is_conditional_mass (a b : Bool) :
    mass a b = (if accepted a b then (1 / 4 : ℝ) else 0) / (1 / 2) := by
  cases a <;> cases b <;> norm_num [mass, accepted, reward]

theorem retained_baseline_dependence :
    (∑ a : Bool, ∑ b : Bool, mass a b * reward a) = 1 / 2 ∧
    (∑ a : Bool, ∑ b : Bool, mass a b * reward b) = 1 / 2 ∧
    (∑ a : Bool, ∑ b : Bool, mass a b * reward a * reward b) = 0 ∧
    (∑ a : Bool, ∑ b : Bool, mass a b * (reward a - 1 / 2) * (reward b - 1 / 2)) = -1 / 4 := by
  norm_num [Fintype.sum_bool, mass, accepted, reward]

theorem filtered_rloo_bias :
    (∑ a : Bool, ∑ b : Bool, (1 / 4 : ℝ) * rloo a b) = 1 ∧
    (∑ a : Bool, ∑ b : Bool, mass a b * rloo a b) = 2 := by
  norm_num [Fintype.sum_bool, mass, accepted, reward, score, rloo]

theorem bernoulli_reward_derivative :
    HasDerivAt (fun θ : ℝ => (1 - θ) * reward false + θ * reward true) 1 (1 / 2) := by
  simpa [reward] using (hasDerivAt_id (1 / 2 : ℝ))

end
end REINFORCEProMax.GroupFiltering
