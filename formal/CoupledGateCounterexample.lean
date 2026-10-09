import SelectiveRewardExample

/-! A one-parameter policy family whose retained-branch objective improves
while reward falls. Rejected and retained branches share the same parameter;
all current/snapshot ratios remain inside the PPO clipping interval. -/
namespace REINFORCEProMax.AutoregressiveTree.CoupledGateCounterexample
open scoped BigOperators
open BatchReduction
open SelectiveSampling (Response Block)

noncomputable def rollout : Policy (Fin 3) := selectivePolicy 2 (by norm_num) false
noncomputable def snapshot : Policy (Fin 3) := selectivePolicy 2 (by norm_num) true
noncomputable def shift (δ : ℝ) (h : List (Fin 3)) : ℝ :=
  if h = [0] then δ else if h = [1] then -3*δ else 0

noncomputable def candidate (δ : ℝ) (hδ : |δ| ≤ 1/45) : Policy (Fin 3) where
  prob h a := if h = [] then selectiveWeights 2 true a
    else (![1/3+shift δ h, 1/3-shift δ h, 1/3] : Fin 3 → ℝ) a
  nonneg := by
    intro h a
    obtain ⟨hlo,hhi⟩ := abs_le.mp hδ
    split_ifs
    · fin_cases a <;> norm_num [selectiveWeights]
    · fin_cases a <;> simp [shift] <;> split_ifs <;> linarith
  mass := by
    intro h
    split_ifs
    · norm_num [selectiveWeights, Fin.sum_univ_succ]
    · simp [Fin.sum_univ_succ]; ring

noncomputable def reward (y : Response) : ℝ := if y.2 = 0 then 1 else 0
noncomputable def treeReward (h : List (Fin 3)) : ℝ :=
  match h with
  | _ :: b :: _ => if b = 0 then 1 else 0
  | _ => 0

theorem reward_gain (δ : ℝ) (hδ : |δ| ≤ 1/45) :
    value (candidate δ hδ) treeReward 2 [] - value snapshot treeReward 2 [] = -δ/2 := by
  simp [value, candidate, snapshot, selectivePolicy, selectiveWeights, shift,
    treeReward, Fin.sum_univ_succ]
  ring

theorem root_ratio (δ : ℝ) (hδ : |δ| ≤ 1/45) (a : Fin 3) :
    tokenRatio snapshot (candidate δ hδ) [] a = 1 := by
  fin_cases a <;> norm_num [tokenRatio, snapshot, selectivePolicy, candidate, selectiveWeights]

noncomputable def responseIncrement (δ : ℝ) (y : Response) : ℝ :=
  if y.1 = 0 then (![3*δ,-3*δ,0] : Fin 3 → ℝ) y.2
  else if y.1 = 1 then (![-9*δ,9*δ,0] : Fin 3 → ℝ) y.2 else 0

theorem second_ratio (δ : ℝ) (hδ : |δ| ≤ 1/45) (y : Response) :
    tokenRatio snapshot (candidate δ hδ) [y.1] y.2 = 1+responseIncrement δ y := by
  rcases y with ⟨a,b⟩
  fin_cases a <;> fin_cases b <;>
    norm_num [tokenRatio, snapshot, selectivePolicy, candidate, shift, responseIncrement] <;> ring

theorem clip_bounds (δ : ℝ) (hδ : |δ| ≤ 1/45) (h : List (Fin 3)) (a : Fin 3) :
    4/5 ≤ tokenRatio snapshot (candidate δ hδ) h a ∧
      tokenRatio snapshot (candidate δ hδ) h a ≤ 6/5 := by
  obtain ⟨hlo,hhi⟩ := abs_le.mp hδ
  by_cases hz : h = []
  · subst h; rw [root_ratio]; norm_num
  · fin_cases a <;> simp [tokenRatio, snapshot, selectivePolicy, candidate, hz, shift] <;>
      (try split_ifs) <;> constructor <;> linarith

theorem candidate_positive (δ : ℝ) (hδ : |δ| ≤ 1/45) (h : List (Fin 3)) (a : Fin 3) :
    0 < (candidate δ hδ).prob h a := by
  by_contra hn
  have hz : (candidate δ hδ).prob h a = 0 :=
    le_antisymm (le_of_not_gt hn) ((candidate δ hδ).nonneg h a)
  have hb := (clip_bounds δ hδ h a).1
  norm_num [tokenRatio, hz] at hb

noncomputable def signal (b : Block) (i : Bool) : ℝ :=
  reward (SelectiveSampling.response b i)-reward (SelectiveSampling.response b (!i))
noncomputable def retainedIncrement (δ : ℝ) (y : Response) : ℝ :=
  if y.1 = 0 then responseIncrement δ y else 0
noncomputable def groupZ (δ : ℝ) (b : Block) : ℝ :=
  (reward b.1-reward b.2)*(retainedIncrement δ b.1-retainedIncrement δ b.2)
noncomputable def rawGroupZ (δ : ℝ) (b : Block) : ℝ :=
  (reward b.1-reward b.2)*(responseIncrement δ b.1-responseIncrement δ b.2)

theorem cached_gate (τ : ℝ) (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log 2)
    (b : Block) (it : Bool × Bool) :
    prefixLogGate rollout snapshot τ τ
      (SelectiveSampling.history b it ++ [SelectiveSampling.action b it]) =
      if (SelectiveSampling.response b it.1).1 = 0 then 1 else 0 := by
  rcases it with ⟨i,t⟩
  cases t
  · exact selective_gate 2 τ (by norm_num) hτ _ [] (by simp; linarith)
  · exact selective_gate 2 τ (by norm_num) hτ _ [_] (by norm_num; exact hmargin)

theorem objective_increment_unclipped {I : Type*} [Fintype I]
    (a M w r H : I → ℝ) (hc : ∀ i, 4/5 ≤ r i ∧ r i ≤ 6/5) :
    Objective.objective a M w r (fun i => min (max (r i) (4/5)) (6/5)) H -
      Objective.objective a M w (fun _ => 1) (fun _ => 1) H =
      ∑ i, a i * M i * w i * (r i-1) * H i := by
  have hc' : ∀ i, min (max (r i) (4/5)) (6/5) = r i := by
    intro i; rw [max_eq_left (hc i).1, min_eq_left (hc i).2]
  unfold Objective.objective
  simp_rw [hc', min_self, one_mul]
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

noncomputable def blockObjectiveGain (δ τ scale : ℝ) (hδ : |δ| ≤ 1/45) (b : Block) : ℝ :=
  let M := fun it => prefixLogGate rollout snapshot τ τ
    (SelectiveSampling.history b it ++ [SelectiveSampling.action b it])
  let w := fun it => tokenRatio rollout snapshot (SelectiveSampling.history b it) (SelectiveSampling.action b it)
  let r := fun it => tokenRatio snapshot (candidate δ hδ) (SelectiveSampling.history b it) (SelectiveSampling.action b it)
  let H := fun it : Bool × Bool => scale*signal b it.1
  Objective.objective (fun _ => 1/4) M w r (fun it => min (max (r it) (4/5)) (6/5)) H -
    Objective.objective (fun _ => 1/4) M w (fun _ => 1) (fun _ => 1) H

theorem block_objective_gain (δ τ scale : ℝ) (hδ : |δ| ≤ 1/45)
    (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log 2) (b : Block) :
    blockObjectiveGain δ τ scale hδ b = scale*groupZ δ b/4 := by
  unfold blockObjectiveGain
  rw [objective_increment_unclipped _ _ _ _ _ (fun i => clip_bounds δ hδ _ _)]
  simp_rw [cached_gate τ hτ hmargin]
  simp only [Fintype.sum_prod_type, Fintype.sum_bool, SelectiveSampling.history,
    SelectiveSampling.action, Bool.false_eq_true, ↓reduceIte, root_ratio,
    sub_self, mul_zero, zero_mul, add_zero]
  have hlater : ∀ a b, tokenRatio rollout snapshot [a] b = 1 := by
    intro a b
    exact selective_later_ratio 2 (by norm_num) [a] (by simp) b
  simp_rw [hlater, second_ratio]
  simp only [add_sub_cancel_left, mul_one]
  simp [signal, SelectiveSampling.response, groupZ, retainedIncrement]
  split_ifs <;> ring

set_option maxHeartbeats 800000 in
theorem population_increments (δ scale : ℝ) :
    expectation (SelectiveSampling.blockMass 2) (fun b => scale*groupZ δ b/4) = scale*δ/4 ∧
    expectation (SelectiveSampling.blockMass 2) (rawGroupZ δ) = 0 := by
  constructor <;>
    norm_num [expectation, SelectiveSampling.blockMass, SelectiveSampling.mass,
      selectiveWeights, groupZ, rawGroupZ, reward, retainedIncrement, responseIncrement,
      Fintype.sum_prod_type, Fin.sum_univ_succ, show (2 : Fin 3) ≠ 1 by decide] <;> ring

def outcome (y : Response) : Bool := decide (y.2 = 0)

theorem implementation_signal (ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2)
    (b : Block) (i : Bool) :
    4 * BinaryUpdate.implementationCoefficient false (fun _ => 2) ε 100000000 10
      (![outcome b.1, outcome b.2] : Fin 2 → Bool) (if i then 0 else 1) =
      SelectiveSampling.mixedScale ε * signal b i := by
  rw [SelectiveSampling.implementation_coefficient_pair ε h0 h1 (outcome b.1,outcome b.2)]
  cases i <;> simp [signal, SelectiveSampling.response, PositiveCertificateExample.reward,
    outcome, reward] <;> ring

theorem objective_up_reward_down (δ τ ε : ℝ) (hδ : 0 < δ) (hsmall : δ ≤ 1/45)
    (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log 2) (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2) :
    let hd : |δ| ≤ 1/45 := by rwa [abs_of_pos hδ]
    0 < expectation (SelectiveSampling.blockMass 2)
        (blockObjectiveGain δ τ (SelectiveSampling.mixedScale ε) hd) ∧
      value (candidate δ hd) treeReward 2 [] - value snapshot treeReward 2 [] < 0 := by
  dsimp only
  have hs := (SelectiveSampling.mixed_scale_bounds ε h0 h1).1
  constructor
  · have heq : blockObjectiveGain δ τ (SelectiveSampling.mixedScale ε)
        (show |δ| ≤ 1/45 by rwa [abs_of_pos hδ]) =
        (fun b => SelectiveSampling.mixedScale ε*groupZ δ b/4) := by
      funext b
      exact block_objective_gain δ τ _ _ hτ hmargin b
    rw [heq, (population_increments δ (SelectiveSampling.mixedScale ε)).1]
    positivity
  · rw [reward_gain]
    linarith

noncomputable def blockDiscrepancy (δ τ scale : ℝ) (hδ : |δ| ≤ 1/45) (b : Block) : ℝ :=
  let M := fun it => prefixLogGate rollout snapshot τ τ
    (SelectiveSampling.history b it ++ [SelectiveSampling.action b it])
  let w := fun it => tokenRatio rollout snapshot (SelectiveSampling.history b it) (SelectiveSampling.action b it)
  let r := fun it => tokenRatio snapshot (candidate δ hδ) (SelectiveSampling.history b it) (SelectiveSampling.action b it)
  ∑ it : Bool × Bool, (1/4 : ℝ) * |w it*(r it-1)| *
    |signal b it.1-M it*(scale*signal b it.1)|

set_option maxHeartbeats 1600000 in
theorem population_discrepancy (δ τ scale : ℝ) (hδ : |δ| ≤ 1/45) (hn : 0 ≤ δ)
    (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log 2) (hs : scale ≤ 1) :
    expectation (SelectiveSampling.blockMass 2) (blockDiscrepancy δ τ scale hδ) = (2-scale)*δ/4 := by
  unfold expectation blockDiscrepancy
  simp_rw [cached_gate τ hτ hmargin]
  simp only [Fintype.sum_prod_type, Fintype.sum_bool, SelectiveSampling.history,
    SelectiveSampling.action, Bool.false_eq_true, ↓reduceIte, root_ratio,
    sub_self, mul_zero, abs_zero, zero_mul, add_zero]
  have hlater : ∀ a b, tokenRatio rollout snapshot [a] b = 1 := by
    intro a b
    exact selective_later_ratio 2 (by norm_num) [a] (by simp) b
  simp_rw [hlater, second_ratio]
  simp only [add_sub_cancel_left, one_mul]
  have h3 : |3*δ| = 3*δ := abs_of_nonneg (by positivity)
  have h9 : |9*δ| = 9*δ := abs_of_nonneg (by positivity)
  have hs1 : |1-scale| = 1-scale := abs_of_nonneg (by linarith)
  have hs2 : |-1+scale| = 1-scale := by rw [abs_of_nonpos (by linarith)]; ring
  norm_num [SelectiveSampling.blockMass, SelectiveSampling.mass, selectiveWeights,
    signal, SelectiveSampling.response, reward, responseIncrement, Fin.sum_univ_succ,
    show (2 : Fin 3) ≠ 1 by decide, h3, h9, hs1, hs2, abs_neg]
  ring

theorem population_corrected_margin_nonpositive (δ τ ε : ℝ) (hδ : 0 < δ) (hsmall : δ ≤ 1/45)
    (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log 2) (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2) :
    let hd : |δ| ≤ 1/45 := by rwa [abs_of_pos hδ]
    expectation (SelectiveSampling.blockMass 2)
        (blockObjectiveGain δ τ (SelectiveSampling.mixedScale ε) hd) -
      expectation (SelectiveSampling.blockMass 2)
        (blockDiscrepancy δ τ (SelectiveSampling.mixedScale ε) hd) ≤ 0 := by
  dsimp only
  have hs := (SelectiveSampling.mixed_scale_bounds ε h0 h1).2.1
  have heq : blockObjectiveGain δ τ (SelectiveSampling.mixedScale ε)
        (show |δ| ≤ 1/45 by rwa [abs_of_pos hδ]) =
        (fun b => SelectiveSampling.mixedScale ε*groupZ δ b/4) := by
    funext b
    exact block_objective_gain δ τ _ _ hτ hmargin b
  rw [heq, (population_increments δ (SelectiveSampling.mixedScale ε)).1,
    population_discrepancy δ τ _ _ hδ.le hτ hmargin hs]
  nlinarith [mul_nonneg hδ.le (sub_nonneg.mpr hs)]

end REINFORCEProMax.AutoregressiveTree.CoupledGateCounterexample
