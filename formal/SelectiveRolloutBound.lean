import SelectiveRewardExample
import AdaptiveBound

/-! The positive reward bound holds uniformly over cached mismatch K>1.
The candidate moves 1/200 root probability between rejected actions, while
retaining the original second-step reward change. The actual Max signals,
Pro gate and active-token denominator are unchanged. -/
namespace REINFORCEProMax.AutoregressiveTree.SelectiveSampling.RolloutBound
open scoped BigOperators
noncomputable section

noncomputable def branchMask (b : Block) (it : Bool × Bool) : ℝ :=
  if (response b it.1).1 = 0 then 1 else 0

noncomputable def clipRatio (r : ℝ) := min (max r (4/5)) (6/5)

def candidate (K δ : ℝ) (hK : 1 < K) (hδ : 0 ≤ δ) (hd : δ ≤ 1/15) : Policy (Fin 3) where
  prob h a := if h = [] then
    (![1/2, 1/(2*(K+1)) + 1/200, K/(2*(K+1)) - 1/200] : Fin 3 → ℝ) a
    else if h = [0] then (![1/3+δ,1/3-δ,1/3] : Fin 3 → ℝ) a else 1/3
  nonneg := by
    intro h a
    have hden : 0 < 2*(K+1) := by linarith
    have hlast : 0 ≤ K/(2*(K+1))-1/200 := by
      have hh : (1/200 : ℝ) ≤ K/(2*(K+1)) := (le_div_iff₀ hden).mpr (by linarith)
      linarith
    split_ifs
    · fin_cases a <;> norm_num <;> first | positivity | linarith
    · fin_cases a <;> simp <;> linarith
    · norm_num
  mass := by
    intro h
    have hden : 2*(K+1) ≠ 0 := by linarith
    split_ifs
    · simp [Fin.sum_univ_succ]
      field_simp
      <;> ring
    · simp [Fin.sum_univ_succ]; ring
    · norm_num [Fin.sum_univ_succ]

theorem root_tv (K δ : ℝ) (hK : 1 < K) (hδ : 0 ≤ δ) (hd : δ ≤ 1/15) :
    localTV (selectivePolicy K (by linarith) false) (candidate K δ hK hδ hd) [] = 1/200 := by
  norm_num [localTV, selectivePolicy, selectiveWeights, candidate, Fin.sum_univ_succ] <;> norm_num [abs_div]

theorem local_tv_bound (K : ℝ) (hK : 1 < K) :
    ∀ h, localTV (selectivePolicy K (by linarith) false)
      (candidate K (1/30) hK (by norm_num) (by norm_num)) h ≤ 1/30 := by
  intro h
  by_cases h0 : h = []
  · subst h
    norm_num [localTV, selectivePolicy, selectiveWeights, candidate, Fin.sum_univ_succ] <;> norm_num [abs_div]
  · by_cases h1 : h = [0]
    · subst h
      norm_num [localTV, selectivePolicy, candidate, Fin.sum_univ_succ] <;> norm_num [abs_div] <;> norm_num [abs_div]
    · norm_num [localTV, selectivePolicy, candidate, h0, h1]

theorem reward_gain (K δ : ℝ) (hK : 1 < K) (hδ : 0 ≤ δ) (hd : δ ≤ 1/15) :
    value (candidate K δ hK hδ hd) selectiveReward 2 [] -
      value (selectivePolicy K (by linarith) false) selectiveReward 2 [] = δ/2 := by
  norm_num [value, candidate, selectivePolicy, selectiveWeights, selectiveReward, Fin.sum_univ_succ]
  ring

def correction (K : ℝ) (hK : 1 < K) (b : Block) (it : Bool × Bool) :=
  tokenRatio (selectivePolicy K (by linarith) false) (selectivePolicy K (by linarith) true)
    (history b it) (action b it)
def candidateRatio (K : ℝ) (hK : 1 < K) (b : Block) (it : Bool × Bool) :=
  tokenRatio (selectivePolicy K (by linarith) true) (candidate K (1/30) hK (by norm_num) (by norm_num))
    (history b it) (action b it)
def referenceRatio (K : ℝ) (hK : 1 < K) (b : Block) (it : Bool × Bool) :=
  tokenRatio (selectivePolicy K (by linarith) true) (selectivePolicy K (by linarith) false)
    (history b it) (action b it)
def objective (K scale : ℝ) (hK : 1 < K) (b : Block) (r : Bool × Bool → ℝ) :=
  Objective.objective (fun _ => 1/4) (branchMask b) (correction K hK b) r
    (fun it => clipRatio (r it)) (fun it => scale * signal b it.1)
def distortion (K scale : ℝ) (hK : 1 < K) (b : Block) :=
  Objective.incrementDistortion (fun _ => 1/4) (branchMask b) (correction K hK b)
    (candidateRatio K hK b) (referenceRatio K hK b)
    (fun it => scale * signal b it.1) (fun it => scale * signal b it.1)

set_option maxHeartbeats 3200000 in
theorem distortion_expectation (K scale : ℝ) (hK : 1 < K) (hs : 0 < scale) :
    BatchReduction.expectation (blockMass K) (distortion K scale hK) = scale/1200 := by
  have hp : 0 < K := by linarith
  have hk : K ≠ 0 := ne_of_gt hp
  have hdp : 0 < K+1 := by linarith
  have hd : K+1 ≠ 0 := ne_of_gt hdp
  have heq : ∀ b, distortion K scale hK b =
      ∑ it : Bool × Bool, (1/4 : ℝ) *
        (|(candidate K (1/30) hK (by norm_num) (by norm_num)).prob (history b it) (action b it) -
          (selectivePolicy K hp false).prob (history b it) (action b it)| /
          (selectivePolicy K hp false).prob (history b it) (action b it)) *
        |scale * signal b it.1 - branchMask b it * (scale * signal b it.1)| := by
    intro b
    unfold distortion Objective.incrementDistortion correction candidateRatio referenceRatio tokenRatio
    apply Finset.sum_congr rfl
    intro it _
    have ho := selective_policy_positive K hp true (history b it) (action b it)
    have hp' := selective_policy_positive K hp false (history b it) (action b it)
    rw [← sub_div, abs_div, abs_of_pos ho]
    field_simp
    <;> ring
  unfold BatchReduction.expectation
  simp_rw [heq]
  simp [blockMass, mass, selectiveWeights, branchMask, candidate,
    selectivePolicy, history, action, signal, response, reward,
    Fintype.sum_prod_type, Fintype.sum_bool, Fin.sum_univ_succ, abs_of_pos hs, abs_neg]
  <;> norm_num [abs_div]
  <;> field_simp
  <;> ring

theorem adaptive_cost_bound (K d : ℝ) (hK : 1 < K) :
    4 * adaptiveCost (selectivePolicy K (by linarith) false)
      (candidate K (1/30) hK (by norm_num) (by norm_num)) (1/30) d 2 ≤ 1/1500 := by
  have hroot := root_tv K (1/30) hK (by norm_num) (by norm_num)
  have hcap : adaptiveFutureCap (1/30) d 1 ≤ 1/30 := by
    exact (min_le_right _ _).trans (by simpa using min_le_left ((1:ℝ)*(1/30)) (Real.sqrt (1*d/2)))
  norm_num [adaptiveCost, Finset.sum_range_succ, value, hroot,
    show adaptiveFutureCap (1/30) d 0 = 0 by simp [adaptiveFutureCap]]
  nlinarith

theorem actual_gate (K τ : ℝ) (hK : 1 < K) (hτ : 0 ≤ τ) (hm : 2*τ < Real.log K)
    (b : Block) (it : Bool × Bool) :
    prefixLogGate (selectivePolicy K (by linarith) false)
      (selectivePolicy K (by linarith) true) τ τ (history b it ++ [action b it]) =
      if (response b it.1).1 = 0 then 1 else 0 := by
  rcases it with ⟨i,t⟩
  cases t
  · simp only [history, action, Bool.false_eq_true, ↓reduceIte, List.nil_append]
    exact selective_gate K τ (by linarith) hτ (response b i).1 []
      (by simpa using (show τ < Real.log K by linarith))
  · simp only [history, action, Bool.true_eq, ↓reduceIte, List.singleton_append]
    exact selective_gate K τ (by linarith) hτ (response b i).1 [(response b i).2]
      (by norm_num; exact hm)

theorem reference_clipping_zero (K scale : ℝ) (hK : 1 < K) (b : Block) :
    Objective.clippingPenalty (fun _ => 1/4) (branchMask b)
      (fun it => tokenRatio (selectivePolicy K (by linarith) false)
        (selectivePolicy K (by linarith) true) (history b it) (action b it))
      (fun it => tokenRatio (selectivePolicy K (by linarith) true)
        (selectivePolicy K (by linarith) false) (history b it) (action b it))
      (fun it => clipRatio (tokenRatio (selectivePolicy K (by linarith) true)
        (selectivePolicy K (by linarith) false) (history b it) (action b it)))
      (fun it => scale * signal b it.1) = 0 := by
  unfold Objective.clippingPenalty
  apply Finset.sum_eq_zero
  intro it _
  rcases it with ⟨i,t⟩
  by_cases hz : (response b i).1 = 0
  · cases t <;> simp [branchMask, history, action, hz, tokenRatio,
      selectivePolicy, selectiveWeights, clipRatio] <;> norm_num
  · simp [branchMask, hz]

private theorem positive_support (K δ : ℝ) (hK : 1 < K) (hδ : 0 ≤ δ) (hd : δ ≤ 1/15)
    (h : List (Fin 3)) (a : Fin 3) : 0 < (candidate K δ hK hδ hd).prob h a := by
  have hden : 0 < 2*(K+1) := by linarith
  have hlast : 0 < K/(2*(K+1))-1/200 := by
    have hh : (1/200 : ℝ) < K/(2*(K+1)) := (lt_div_iff₀ hden).mpr (by linarith)
    linarith
  dsimp only [candidate]
  split_ifs
  · fin_cases a <;> norm_num <;> first | positivity | linarith
  · fin_cases a <;> norm_num <;> linarith
  · norm_num

private theorem reward_bound (K : ℝ) (hK : 1 < K) :
    43/3000 ≤ value (candidate K (1/30) hK (by norm_num) (by norm_num)) selectiveReward 2 [] -
      value (selectivePolicy K (by linarith) false) selectiveReward 2 [] := by
  let p := selectivePolicy K (by linarith) false
  let q := candidate K (1/30) hK (by norm_num) (by norm_num)
  let d := |localReverseKL p q []| + |localReverseKL p q [0]| + 1
  have hsupport : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0 := by
    intro h a
    exact iff_of_false (ne_of_gt (selective_policy_positive K (by linarith) false h a))
      (ne_of_gt (positive_support K (1/30) hK (by norm_num) (by norm_num) h a))
  have hKL : ∀ h, localReverseKL p q h ≤ d := by
    intro h
    by_cases h0 : h = []
    · subst h; dsimp [d]
      linarith [le_abs_self (localReverseKL p q []), abs_nonneg (localReverseKL p q [0])]
    · by_cases h1 : h = [0]
      · subst h; dsimp [d]
        linarith [le_abs_self (localReverseKL p q [0]), abs_nonneg (localReverseKL p q [])]
      · have hz : localReverseKL p q h = 0 := by
          norm_num [localReverseKL, tokenRatio, p, q, selectivePolicy, candidate, h0, h1]
        rw [hz]; dsimp [d]; positivity
  have hR : ∀ y : List (Fin 3), |selectiveReward y| ≤ 1 := by
    intro y
    cases y with
    | nil => norm_num [selectiveReward]
    | cons a ys =>
      cases ys with
      | nil => norm_num [selectiveReward]
      | cons b ys => simp only [selectiveReward]; split_ifs <;> norm_num
  have he := surrogate_error_adaptive_root p q selectiveReward 1 (1/30) d
    (by norm_num) hR 2 hsupport (fun h _ => local_tv_bound K hK h) (fun h _ => hKL h)
  have hb := adaptive_cost_bound K d hK
  have hS : surrogate p q selectiveReward 2 [] = 1/60 := by
    norm_num [surrogate, additive, gain, value, selectiveReward, p, q,
      selectivePolicy, candidate, selectiveWeights, Fin.sum_univ_succ]
    <;> field_simp
    <;> ring
  rw [hS] at he
  have hl := (abs_le.mp he).1
  change 43/3000 ≤ value q selectiveReward 2 [] - value p selectiveReward 2 []
  dsimp [p,q] at hl
  linarith

set_option maxHeartbeats 3200000 in
theorem objective_eq_scalar (K δ τ scale : ℝ) (hK : 1 < K) (hδ : 0 ≤ δ) (hd : δ ≤ 1/15)
    (hτ : 0 ≤ τ) (hm : 2*τ < Real.log K) (b : Block) :
    let p := selectivePolicy K (by linarith) false
    let old := selectivePolicy K (by linarith) true
    let q := candidate K δ hK hδ hd
    let M := fun it => prefixLogGate p old τ τ (history b it ++ [action b it])
    let w := fun it => tokenRatio p old (history b it) (action b it)
    let r := fun it => tokenRatio old q (history b it) (action b it)
    let r₀ := fun it => tokenRatio old p (history b it) (action b it)
    let H := fun it : Bool × Bool => scale * signal b it.1
    Objective.objective (fun _ => 1/4) M w r (fun it => clipRatio (r it)) H -
      Objective.objective (fun _ => 1/4) M w r₀ (fun it => clipRatio (r₀ it)) H =
      scalarBlockObjective K τ scale (by linarith) b δ := by
  dsimp only
  unfold scalarBlockObjective Objective.objective
  simp only [actual_gate K τ hK hτ hm]
  congr 1 <;> apply Finset.sum_congr rfl <;> intro it _
  all_goals
    rcases it with ⟨i,t⟩
    by_cases hz : (response b i).1 = 0
    · cases t
      · simp [history, action, hz, tokenRatio, selectivePolicy, selectiveWeights, candidate, clipRatio]
        <;> norm_num
      · generalize ha : (response b i).2 = a
        fin_cases a <;> simp [history, action, hz, tokenRatio, selectivePolicy, selectiveWeights, candidate,
          clipRatio, increment, ha] <;> ring_nf <;> norm_num
    · simp [hz]

/-- Reuse the scalar-objective bridge and its existing group expectation. -/
theorem objective_expectation (K scale : ℝ) (hK : 1 < K) :
    BatchReduction.expectation (blockMass K)
      (fun b => objective K scale hK b (candidateRatio K hK b) -
        objective K scale hK b (referenceRatio K hK b)) = scale/120 := by
  have hp : 0 < K := by linarith
  have hm : 2*(0:ℝ) < Real.log K := by simpa using Real.log_pos hK
  have hb : ∀ b, objective K scale hK b (candidateRatio K hK b) -
      objective K scale hK b (referenceRatio K hK b) = scalarBlockObjective K 0 scale hp b (1/30) := by
    intro b
    have hM : (fun it => prefixLogGate (selectivePolicy K hp false) (selectivePolicy K hp true)
        0 0 (history b it ++ [action b it])) = branchMask b := by
      funext it
      exact actual_gate K 0 hK (by norm_num) hm b it
    have he := objective_eq_scalar K (1/30) 0 scale hK (by norm_num) (by norm_num)
      (by norm_num) hm b
    dsimp only at he
    simpa only [hM, objective, correction, candidateRatio, referenceRatio] using he
  unfold BatchReduction.expectation
  simp_rw [hb]
  change scalarPopulationObjective K 0 scale hp (1/30) = scale/120
  rw [scalar_population_objective_eq K (1/30) 0 scale hp (by norm_num) (by norm_num) (by norm_num) hm]
  ring

noncomputable def actualSignal (ε : ℝ) (b : Block) (it : Bool × Bool) : ℝ :=
  4 * BinaryUpdate.implementationCoefficient false (fun _ => 2) ε 100000000 10
    (![outcome b.1, outcome b.2] : Fin 2 → Bool) (if it.1 then 0 else 1)


def actualMask (K : ℝ) (hK : 1 < K) (τ : ℝ) (b : Block) (it : Bool × Bool) : ℝ :=
  prefixLogGate (selectivePolicy K (by linarith) false) (selectivePolicy K (by linarith) true)
    τ τ (history b it ++ [action b it])
def objectiveGain (K : ℝ) (hK : 1 < K) (ε τ : ℝ) (b : Block) : ℝ :=
  Objective.objective (fun _ => 1/4) (actualMask K hK τ b) (correction K hK b) (candidateRatio K hK b)
    (fun it => clipRatio (candidateRatio K hK b it)) (actualSignal ε b) -
  Objective.objective (fun _ => 1/4) (actualMask K hK τ b) (correction K hK b) (referenceRatio K hK b)
    (fun it => clipRatio (referenceRatio K hK b it)) (actualSignal ε b)
def actualDistortion (K : ℝ) (hK : 1 < K) (ε τ : ℝ) (b : Block) : ℝ :=
  Objective.incrementDistortion (fun _ => 1/4) (actualMask K hK τ b) (correction K hK b)
    (candidateRatio K hK b) (referenceRatio K hK b)
    (fun it => mixedScale ε * signal b it.1) (actualSignal ε b)
def referencePenalty (K : ℝ) (hK : 1 < K) (ε τ : ℝ) (b : Block) : ℝ :=
  Objective.clippingPenalty (fun _ => 1/4) (actualMask K hK τ b) (correction K hK b)
    (referenceRatio K hK b) (fun it => clipRatio (referenceRatio K hK b it)) (actualSignal ε b)
def populationMargin (K : ℝ) (hK : 1 < K) (ε τ : ℝ) : ℝ :=
  4 * BatchReduction.expectation (blockMass K)
    (fun b => objectiveGain K hK ε τ b - actualDistortion K hK ε τ b - referencePenalty K hK ε τ b) /
      (mixedScale ε * 2)

theorem actual_population_margin (K : ℝ) (hK : 1 < K) (ε τ : ℝ) (hε0 : 0 ≤ ε) (hε1 : ε ≤ 1/2)
    (hτ : 0 ≤ τ) (hm : 2*τ < Real.log K) : populationMargin K hK ε τ = 3/200 := by
  have hs : 0 < mixedScale ε := lt_of_lt_of_le (by norm_num) (mixed_scale_bounds ε hε0 hε1).1
  have hH : ∀ b it, actualSignal ε b it = mixedScale ε * signal b it.1 := by
    intro b it
    exact implementation_signal ε hε0 hε1 b it.1
  have hM : ∀ b it, actualMask K hK τ b it = branchMask b it := by
    intro b it
    exact actual_gate K τ hK hτ hm b it
  have hHfun : ∀ b, actualSignal ε b = fun it => mixedScale ε * signal b it.1 :=
    fun b => funext (hH b)
  have hMfun : ∀ b, actualMask K hK τ b = branchMask b := fun b => funext (hM b)
  have hp : ∀ b, referencePenalty K hK ε τ b = 0 := by
    intro b
    unfold referencePenalty
    rw [hHfun, hMfun]
    simpa only [correction, referenceRatio] using reference_clipping_zero K (mixedScale ε) hK b
  unfold populationMargin
  simp_rw [hp, sub_zero]
  have hg : ∀ b, objectiveGain K hK ε τ b =
      objective K (mixedScale ε) hK b (candidateRatio K hK b) -
      objective K (mixedScale ε) hK b (referenceRatio K hK b) := by
    intro b
    simp only [objectiveGain, objective, hHfun, hMfun]
  have hd : ∀ b, actualDistortion K hK ε τ b = distortion K (mixedScale ε) hK b := by
    intro b
    simp only [actualDistortion, distortion, hHfun, hMfun]
  simp_rw [hg, hd]
  rw [show BatchReduction.expectation (blockMass K)
      (fun b => (objective K (mixedScale ε) hK b (candidateRatio K hK b) -
        objective K (mixedScale ε) hK b (referenceRatio K hK b)) - distortion K (mixedScale ε) hK b) =
      BatchReduction.expectation (blockMass K)
        (fun b => objective K (mixedScale ε) hK b (candidateRatio K hK b) -
          objective K (mixedScale ε) hK b (referenceRatio K hK b)) -
      BatchReduction.expectation (blockMass K) (distortion K (mixedScale ε) hK) by
        unfold BatchReduction.expectation
        simp only [mul_sub, Finset.sum_sub_distrib]]
  rw [objective_expectation, distortion_expectation _ _ _ hs]
  field_simp
  <;> ring



/-- The actual objective margin retains the same positive Adaptive reward
lower bound for every K>1, including unbounded cached importance moments. -/
theorem positive_actual_reward_bound (K : ℝ) (hK : 1 < K) (ε τ : ℝ)
    (hε0 : 0 ≤ ε) (hε1 : ε ≤ 1/2) (hτ : 0 ≤ τ) (hm : 2*τ < Real.log K) :
    populationMargin K hK ε τ - 1/1500 = 43/3000 ∧
    populationMargin K hK ε τ - 1/1500 ≤
      value (candidate K (1/30) hK (by norm_num) (by norm_num)) selectiveReward 2 [] -
      value (selectivePolicy K (by linarith) false) selectiveReward 2 [] ∧
    0 < populationMargin K hK ε τ - 1/1500 := by
  rw [actual_population_margin K hK ε τ hε0 hε1 hτ hm]
  refine ⟨by norm_num, ?_, by norm_num⟩
  convert reward_bound K hK using 1 <;> norm_num

theorem candidate_gradient_step_reward_gain (K τ ε δ₀ η : ℝ) (hK : 1 < K)
    (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log K) (hε0 : 0 ≤ ε) (hε1 : ε ≤ 1/2)
    (hδ₀ : 0 < δ₀) (hsmall : δ₀ < 1/15) (hη : 0 < η)
    (hstep : δ₀ + η * mixedScale ε / 4 ≤ 1/15) :
    let δ₁ := δ₀ + η * deriv (scalarPopulationObjective K τ (mixedScale ε) (by linarith)) δ₀
    ∃ (hδ₁ : 0 ≤ δ₁) (hsmall₁ : δ₁ ≤ 1/15),
      value (candidate K δ₁ hK hδ₁ hsmall₁) selectiveReward 2 [] -
        value (candidate K δ₀ hK hδ₀.le hsmall.le) selectiveReward 2 [] =
          η * mixedScale ε / 8 ∧ 0 < η * mixedScale ε / 8 := by
  dsimp only
  rw [(scalar_population_objective_derivative K τ (mixedScale ε) δ₀ (by linarith) hτ hmargin hδ₀ hsmall).deriv]
  have hs : 0 < mixedScale ε := lt_of_lt_of_le (by norm_num) (mixed_scale_bounds ε hε0 hε1).1
  have hpos : 0 ≤ δ₀ + η * (mixedScale ε / 4) := by positivity
  have hbound : δ₀ + η * (mixedScale ε / 4) ≤ 1/15 := by nlinarith [hstep]
  refine ⟨hpos, hbound, ?_, by positivity⟩
  have hg1 := reward_gain K (δ₀ + η * (mixedScale ε / 4)) hK hpos hbound
  have hg0 := reward_gain K δ₀ hK hδ₀.le hsmall.le
  linarith


end
end REINFORCEProMax.AutoregressiveTree.SelectiveSampling.RolloutBound
