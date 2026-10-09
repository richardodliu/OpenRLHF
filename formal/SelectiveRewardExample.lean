import PrefixGateExample
import ProMaxObjective
import PositiveCertificateExample

/-! A reward-bearing extension of the selective-prefix family. The candidate
changes only the second action on the retained branch. This establishes
compatibility of nontrivial filtering with reward improvement, including a
population gradient step in the specified scalar family. It does not claim
an unrestricted neural-policy or stochastic-optimizer guarantee. -/
namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators

noncomputable def selectiveCandidate (K δ : ℝ) (hK : 0 < K)
    (hδ : 0 ≤ δ) (hsmall : δ ≤ 1/15) : Policy (Fin 3) where
  prob h a := if h = [] then selectiveWeights K true a
    else if h = [0] then (![1/3+δ, 1/3-δ, 1/3] : Fin 3 → ℝ) a else 1/3
  nonneg := by
    intro h a
    split_ifs
    · fin_cases a <;> simp [selectiveWeights] <;> positivity
    · fin_cases a <;> simp <;> linarith
    · norm_num
  mass := by
    intro h
    split_ifs
    · simp [selectiveWeights, Fin.sum_univ_succ]
      field_simp
      ring
    · simp [Fin.sum_univ_succ]; ring
    · norm_num [Fin.sum_univ_succ]

noncomputable def selectiveReward (h : List (Fin 3)) : ℝ :=
  match h with
  | a :: b :: _ => if a = 0 ∧ b = 0 then 1 else 0
  | _ => 0

theorem selective_reward_continuation (p : Policy (Fin 3)) (n : ℕ)
    (a b : Fin 3) (xs : List (Fin 3)) :
    value p selectiveReward n (a :: b :: xs) = if a = 0 ∧ b = 0 then 1 else 0 := by
  induction n generalizing xs with
  | zero => simp [value, selectiveReward]
  | succ n ih =>
    simp only [value, List.cons_append, ih]
    rw [← Finset.sum_mul, p.mass, one_mul]

theorem selective_snapshot_reward (K : ℝ) (hK : 0 < K) (n : ℕ) :
    value (selectivePolicy K hK true) selectiveReward (n+2) [] = 1/6 := by
  simp [value, selective_reward_continuation, selectivePolicy, selectiveWeights,
    Fin.sum_univ_succ, Nat.add_succ]
  <;> ring

theorem selective_candidate_reward (K δ : ℝ) (hK : 0 < K)
    (hδ : 0 ≤ δ) (hsmall : δ ≤ 1/15) (n : ℕ) :
    value (selectiveCandidate K δ hK hδ hsmall) selectiveReward (n+2) [] = 1/6+δ/2 := by
  simp [value, selective_reward_continuation, selectiveCandidate, selectiveWeights,
    Fin.sum_univ_succ, Nat.add_succ]
  <;> ring

theorem selective_reward_gain (K δ : ℝ) (hK : 0 < K)
    (hδ : 0 < δ) (hsmall : δ ≤ 1/15) (n : ℕ) :
    value (selectiveCandidate K δ hK hδ.le hsmall) selectiveReward (n+2) [] -
      value (selectivePolicy K hK true) selectiveReward (n+2) [] = δ/2 := by
  rw [selective_candidate_reward, selective_snapshot_reward]
  ring

theorem selective_candidate_root_ratio (K δ : ℝ) (hK : 0 < K)
    (hδ : 0 ≤ δ) (hsmall : δ ≤ 1/15) (a : Fin 3) :
    tokenRatio (selectivePolicy K hK true) (selectiveCandidate K δ hK hδ hsmall) [] a = 1 := by
  have hp := ne_of_gt (selective_policy_positive K hK true [] a)
  simpa [tokenRatio, selectivePolicy, selectiveCandidate] using div_self hp

theorem selective_candidate_active_ratio (K δ : ℝ) (hK : 0 < K)
    (hδ : 0 ≤ δ) (hsmall : δ ≤ 1/15) (a : Fin 3) :
    tokenRatio (selectivePolicy K hK true) (selectiveCandidate K δ hK hδ hsmall) [0] a =
      (![1+3*δ, 1-3*δ, 1] : Fin 3 → ℝ) a := by
  fin_cases a <;> norm_num [tokenRatio, selectivePolicy, selectiveCandidate] <;> ring

theorem selective_candidate_inactive_ratio (K δ : ℝ) (hK : 0 < K)
    (hδ : 0 ≤ δ) (hsmall : δ ≤ 1/15) (h : List (Fin 3)) (hh : h ≠ [0]) (a : Fin 3) :
    tokenRatio (selectivePolicy K hK true) (selectiveCandidate K δ hK hδ hsmall) h a = 1 := by
  by_cases hz : h = []
  · subst h; exact selective_candidate_root_ratio K δ hK hδ hsmall a
  · norm_num [tokenRatio, selectivePolicy, selectiveCandidate, hz, hh]

/-- Gate correction of the candidate increment vanishes pointwise, despite
half of rollout trajectories being rejected. This uses the actual cached
gate and actual three-policy ratios, not an independent binary mask. -/
theorem selective_gate_increment (K δ τ : ℝ) (hK : 0 < K)
    (hδ : 0 ≤ δ) (hsmall : δ ≤ 1/15) (hτ : 0 ≤ τ)
    (hmargin : 2*τ < Real.log K) (h : List (Fin 3)) (a : Fin 3) :
    (1 - prefixLogGate (selectivePolicy K hK false) (selectivePolicy K hK true)
      τ τ (h ++ [a])) *
    (tokenRatio (selectivePolicy K hK true) (selectiveCandidate K δ hK hδ hsmall) h a - 1) = 0 := by
  by_cases hh : h = [0]
  · subst h
    have hg := selective_gate K τ hK hτ 0 [a] (by norm_num; exact hmargin)
    simp only [List.cons_append, List.nil_append] at *
    rw [hg]
    simp
  · rw [selective_candidate_inactive_ratio K δ hK hδ hsmall h hh a]
    ring

theorem selective_candidate_clip_bounds (K δ : ℝ) (hK : 0 < K)
    (hδ : 0 ≤ δ) (hsmall : δ ≤ 1/15) (h : List (Fin 3)) (a : Fin 3) :
    4/5 ≤ tokenRatio (selectivePolicy K hK true) (selectiveCandidate K δ hK hδ hsmall) h a ∧
    tokenRatio (selectivePolicy K hK true) (selectiveCandidate K δ hK hδ hsmall) h a ≤ 6/5 := by
  by_cases hh : h = [0]
  · subst h
    rw [selective_candidate_active_ratio]
    fin_cases a <;> simp <;> constructor <;> linarith
  · rw [selective_candidate_inactive_ratio K δ hK hδ hsmall h hh a]
    norm_num


/-- Exact finite objective increment when clipping is inactive and every
changed coefficient is retained. No assumption is imposed on the frozen
advantages, so the identity also applies to normalized RLOO signals. -/
theorem retained_change_objective_identity {I : Type*} [Fintype I]
    (a M w r H : I → ℝ)
    (hgate : ∀ i, (1-M i)*(r i-1) = 0)
    (hclip : ∀ i, 4/5 ≤ r i ∧ r i ≤ 6/5) :
    Objective.objective a M w r (fun i => min (max (r i) (4/5)) (6/5)) H -
      Objective.objective a M w (fun _ => 1) (fun _ => 1) H =
        ∑ i, a i * w i * (r i-1) * H i := by
  have hc : ∀ i, min (max (r i) (4/5)) (6/5) = r i := by
    intro i
    rw [max_eq_left (hclip i).1, min_eq_left (hclip i).2]
  unfold Objective.objective
  simp_rw [hc, min_self, one_mul]
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  have hg := hgate i
  calc
    _ = a i * w i * (r i-1) * H i - a i * w i * H i * ((1-M i)*(r i-1)) := by ring
    _ = _ := by rw [hg]; ring

theorem selective_objective_gain {I : Type*} [Fintype I]
    (K δ τ : ℝ) (hK : 0 < K) (hδ : 0 ≤ δ) (hsmall : δ ≤ 1/15)
    (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log K)
    (a H : I → ℝ) (history : I → List (Fin 3)) (action : I → Fin 3) :
    let p := selectivePolicy K hK false
    let q := selectivePolicy K hK true
    let c := selectiveCandidate K δ hK hδ hsmall
    let M := fun i => prefixLogGate p q τ τ (history i ++ [action i])
    let w := fun i => tokenRatio p q (history i) (action i)
    let r := fun i => tokenRatio q c (history i) (action i)
    Objective.objective a M w r (fun i => min (max (r i) (4/5)) (6/5)) H -
      Objective.objective a M w (fun _ => 1) (fun _ => 1) H =
        ∑ i, a i * w i * (r i-1) * H i := by
  dsimp only
  apply retained_change_objective_identity
  · intro i
    exact selective_gate_increment K δ τ hK hδ hsmall hτ hmargin (history i) (action i)
  · intro i
    exact selective_candidate_clip_bounds K δ hK hδ hsmall (history i) (action i)


namespace SelectiveSampling
open BatchReduction
abbrev Response := Fin 3 × Fin 3
abbrev Block := Response × Response

noncomputable def mass (K : ℝ) (y : Response) : ℝ := selectiveWeights K false y.1 / 3
noncomputable def blockMass (K : ℝ) (b : Block) : ℝ := mass K b.1 * mass K b.2
noncomputable def reward (y : Response) : ℝ := if y.1 = 0 ∧ y.2 = 0 then 1 else 0
noncomputable def increment (δ : ℝ) (y : Response) : ℝ :=
  if y.1 = 0 then (![3*δ, -3*δ, 0] : Fin 3 → ℝ) y.2 else 0
noncomputable def groupZ (δ : ℝ) (b : Block) : ℝ :=
  (reward b.1-reward b.2)*(increment δ b.1-increment δ b.2)

/-- The sampling law is the actual two-step rollout, including rejected paths. -/
theorem mass_from_rollout (K : ℝ) (hK : 0 < K) (y : Response) :
    mass K y = (selectivePolicy K hK false).prob [] y.1 *
      (selectivePolicy K hK false).prob [y.1] y.2 := by
  simp [mass, selectivePolicy]; ring

theorem block_mass_nonneg (K : ℝ) (hK : 0 < K) (b : Block) : 0 ≤ blockMass K b := by
  unfold blockMass
  apply mul_nonneg <;> rw [mass_from_rollout K hK] <;>
    exact mul_nonneg ((selectivePolicy K hK false).nonneg _ _) ((selectivePolicy K hK false).nonneg _ _)

set_option maxHeartbeats 800000 in
theorem block_mass_normalized (K : ℝ) (hK : 0 < K) : ∑ b : Block, blockMass K b = 1 := by
  have hd : K+1 ≠ 0 := by positivity
  simp [blockMass, mass, selectiveWeights, Fintype.sum_prod_type, Fin.sum_univ_succ]
  field_simp
  <;> ring

set_option maxHeartbeats 800000 in
theorem group_moments (K δ : ℝ) (hK : 0 < K) :
    expectation (blockMass K) (groupZ δ) = δ ∧
    expectation (blockMass K) (fun b => groupZ δ b - δ) = 0 ∧
    expectation (blockMass K) (fun b => (groupZ δ b - δ)^2) = 3*δ^2 := by
  have hd : K+1 ≠ 0 := by positivity
  refine ⟨?_,?_,?_⟩ <;>
    simp [expectation, blockMass, mass, selectiveWeights, groupZ, reward, increment,
      Fintype.sum_prod_type, Fin.sum_univ_succ] <;>
    field_simp <;> ring

noncomputable def estimate {N : ℕ} (δ : ℝ) (s : Fin N → Block) : ℝ :=
  sampleMean (groupZ δ) s / sampleMean (fun _ => 4) s

theorem estimate_tail {N : ℕ} (hN : N ≠ 0) (K δ : ℝ) (hK : 0 < K) (hδ : 0 < δ) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin N → Block => δ/16 ≤ |estimate δ s-δ/4|),
      iidMass (blockMass K) s) ≤ 48/(N : ℝ) := by
  have hz : expectation (blockMass K) (fun b => groupZ δ b - (δ/4)*4) = 0 := by
    simpa using (group_moments K δ hK).2.1
  have hv : expectation (blockMass K) (fun b => (groupZ δ b - (δ/4)*4)^2) = 3*δ^2 := by
    simpa using (group_moments K δ hK).2.2
  have hb := iid_token_mean_tail hN (blockMass K) (groupZ δ) (fun _ => 4)
    (block_mass_nonneg K hK) (block_mass_normalized K hK)
    4 (δ/4) (δ/16) (by norm_num) (by positivity) (fun _ => le_rfl) hz
  rw [hv] at hb
  have hden : 3*δ^2 / ((N : ℝ)*4^2*(δ/16)^2) = 48/(N : ℝ) := by
    have hd := ne_of_gt hδ
    field_simp
    <;> ring
  simpa only [estimate, hden] using hb

noncomputable def mixedScale (ε : ℝ) : ℝ := Real.sqrt (4/(4+ε))

theorem mixed_scale_eq (ε : ℝ) (h0 : 0 ≤ ε) :
    mixedScale ε = PositiveCertificateExample.mixedScale (ε/2) := by
  unfold mixedScale PositiveCertificateExample.mixedScale
  congr 1
  have h4 : 4+ε ≠ 0 := by positivity
  have h2 : 2+ε/2 ≠ 0 := by positivity
  field_simp
  <;> ring

theorem mixed_scale_bounds (ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2) :
    3/4 ≤ mixedScale ε ∧ mixedScale ε ≤ 1 ∧
    NormalizationSafeguards.positiveClamp ε 10 (mixedScale ε) = mixedScale ε := by
  have hs := PositiveCertificateExample.mixed_scale_safeguards (ε/2) (by linarith) (by linarith)
  rw [← mixed_scale_eq ε h0] at hs
  refine ⟨hs.1,hs.2.1,?_⟩
  simp [NormalizationSafeguards.positiveClamp,
    max_eq_left (show ε ≤ mixedScale ε by linarith [hs.1]),
    min_eq_left (show mixedScale ε ≤ 10 by linarith [hs.2.1])]

theorem implementation_coefficient_pair (ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2)
    (b : Bool × Bool) (i : Fin 2) :
    BinaryUpdate.implementationCoefficient false (fun _ => 2) ε 100000000 10
      (![b.1, b.2] : Fin 2 → Bool) i =
      mixedScale ε * (PositiveCertificateExample.reward (![b.1, b.2] i) -
        PositiveCertificateExample.reward (![b.2, b.1] i)) / 4 := by
  have he : ¬ (2 : ℝ) < ε := by linarith
  have hs := (mixed_scale_bounds ε h0 h1).2.2
  unfold mixedScale at hs ⊢
  rw [Real.sqrt_div (by norm_num : (0 : ℝ) ≤ 4)] at hs
  norm_num at hs
  rcases b with ⟨x,y⟩
  have hr : BinaryUpdate.binaryRloo (![x,y] : Fin 2 → Bool) =
      ![BinaryUpdate.bit x - BinaryUpdate.bit y,
        BinaryUpdate.bit y - BinaryUpdate.bit x] := by
    funext j
    fin_cases j <;>
      simp [BinaryUpdate.binaryRloo, iidLooBaseline, Fin.sum_univ_one, Fin.succAbove]
  unfold BinaryUpdate.implementationCoefficient BinaryUpdate.normalizedCoefficient
    BinaryUpdate.implementationFactors
  dsimp only
  simp only [hr]
  fin_cases i <;> cases x <;> cases y <;>
    norm_num [BinaryUpdate.safeguardedFactors, BinaryUpdate.tokenDenominator,
      BinaryUpdate.bit, AdaptiveMechanism.positiveTotal, AdaptiveMechanism.negativeTotal,
      AdaptiveMechanism.positiveSquare, AdaptiveMechanism.negativeSquare,
      AdaptiveMechanism.nonzeroMass, Fin.sum_univ_succ,
      NormalizationSafeguards.signScale, PositiveCertificateExample.reward, he, h0, hs]

def response (b : Block) (i : Bool) : Response := if i then b.1 else b.2
def history (b : Block) (it : Bool × Bool) : List (Fin 3) :=
  if it.2 then [(response b it.1).1] else []
def action (b : Block) (it : Bool × Bool) : Fin 3 :=
  if it.2 then (response b it.1).2 else (response b it.1).1
noncomputable def signal (b : Block) (i : Bool) : ℝ := reward (response b i)-reward (response b (!i))

theorem second_ratio_increment (K δ : ℝ) (hK : 0 < K) (hδ : 0 ≤ δ)
    (hsmall : δ ≤ 1/15) (y : Response) :
    tokenRatio (selectivePolicy K hK true) (selectiveCandidate K δ hK hδ hsmall) [y.1] y.2 - 1 =
      increment δ y := by
  rcases y with ⟨a,b⟩
  fin_cases a <;> dsimp only
  · change tokenRatio (selectivePolicy K hK true) (selectiveCandidate K δ hK hδ hsmall) [0] b - 1 = _
    rw [selective_candidate_active_ratio]
    fin_cases b <;> simp [increment] <;> ring
  · rw [selective_candidate_inactive_ratio K δ hK hδ hsmall _ (by decide)]
    simp [increment]
  · rw [selective_candidate_inactive_ratio K δ hK hδ hsmall _ (by decide)]
    simp [increment]

theorem block_objective_gain (K δ τ scale : ℝ) (hK : 0 < K) (hδ : 0 ≤ δ)
    (hsmall : δ ≤ 1/15) (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log K) (b : Block) :
    let p := selectivePolicy K hK false
    let q := selectivePolicy K hK true
    let c := selectiveCandidate K δ hK hδ hsmall
    let M := fun it => prefixLogGate p q τ τ (history b it ++ [action b it])
    let w := fun it => tokenRatio p q (history b it) (action b it)
    let r := fun it => tokenRatio q c (history b it) (action b it)
    let H := fun it : Bool × Bool => scale * signal b it.1
    Objective.objective (fun _ => 1/4) M w r (fun it => min (max (r it) (4/5)) (6/5)) H -
      Objective.objective (fun _ => 1/4) M w (fun _ => 1) (fun _ => 1) H =
        scale * groupZ δ b / 4 := by
  dsimp only
  rw [selective_objective_gain K δ τ hK hδ hsmall hτ hmargin]
  simp only [Fintype.sum_prod_type, Fintype.sum_bool, history, action,
    Bool.false_eq_true, ↓reduceIte, selective_candidate_root_ratio, sub_self,
    mul_zero, zero_mul, add_zero]
  have hlater : ∀ a b, tokenRatio (selectivePolicy K hK false)
      (selectivePolicy K hK true) [a] b = 1 := by
    intro a b
    exact selective_later_ratio K hK [a] (by simp) b
  simp_rw [hlater, second_ratio_increment]
  simp [signal, response, groupZ]
  ring

theorem retained_change_discrepancy (M w r A scale : ℝ)
    (hgate : (1-M)*(r-1) = 0) :
    |w*(r-1)| * |A-M*(scale*A)| = |w*(r-1)| * |A-scale*A| := by
  by_cases hr : r = 1
  · simp [hr]
  · have hm : M = 1 := by
      have hz := (mul_eq_zero.mp hgate).resolve_right (sub_ne_zero.mpr hr)
      linarith
    simp [hm]

set_option maxHeartbeats 800000 in
theorem block_discrepancy (K δ τ scale : ℝ) (hK : 0 < K) (hδ : 0 ≤ δ)
    (hsmall : δ ≤ 1/15) (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log K)
    (hscale : scale ≤ 1) (b : Block) :
    let p := selectivePolicy K hK false
    let q := selectivePolicy K hK true
    let c := selectiveCandidate K δ hK hδ hsmall
    let M := fun it => prefixLogGate p q τ τ (history b it ++ [action b it])
    let w := fun it => tokenRatio p q (history b it) (action b it)
    let r := fun it => tokenRatio q c (history b it) (action b it)
    (∑ it : Bool × Bool, (1/4 : ℝ) * (|w it*(r it-1)| *
      |signal b it.1-M it*(scale*signal b it.1)|)) = (1-scale)*groupZ δ b/4 := by
  dsimp only
  have hpoint (it : Bool × Bool) := retained_change_discrepancy
    (prefixLogGate (selectivePolicy K hK false) (selectivePolicy K hK true)
      τ τ (history b it ++ [action b it]))
    (tokenRatio (selectivePolicy K hK false) (selectivePolicy K hK true) (history b it) (action b it))
    (tokenRatio (selectivePolicy K hK true) (selectiveCandidate K δ hK hδ hsmall) (history b it) (action b it))
    (signal b it.1) scale
    (selective_gate_increment K δ τ hK hδ hsmall hτ hmargin (history b it) (action b it))
  simp_rw [hpoint]
  simp only [Fintype.sum_prod_type, Fintype.sum_bool, history, action,
    Bool.false_eq_true, ↓reduceIte, selective_candidate_root_ratio, sub_self,
    mul_zero, abs_zero, zero_mul, add_zero]
  have hlater : ∀ a b, tokenRatio (selectivePolicy K hK false)
      (selectivePolicy K hK true) [a] b = 1 := by
    intro a b
    exact selective_later_ratio K hK [a] (by simp) b
  simp_rw [hlater, second_ratio_increment, one_mul]
  have habs : |3*δ| = 3*δ := abs_of_nonneg (by positivity)
  have hs : |1-scale| = 1-scale := abs_of_nonneg (by linarith)
  have hs' : |-1+scale| = 1-scale := by rw [abs_of_nonpos (by linarith)]; ring
  rcases b with ⟨⟨a,b⟩,⟨c,d⟩⟩
  fin_cases a <;> fin_cases b <;> fin_cases c <;> fin_cases d <;>
    simp [signal, response, reward, increment, groupZ, habs, hs, hs', abs_neg] <;> ring

theorem positive_margin_on_good_event {N : ℕ} (δ ε : ℝ) (hδ : 0 < δ)
    (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2) (s : Fin N → Block)
    (hgood : |estimate δ s-δ/4| < δ/16) :
    δ/32 < (2*mixedScale ε-1)*estimate δ s-δ/16 ∧
      (2*mixedScale ε-1)*estimate δ s-δ/16 < δ/2 := by
  have hs := mixed_scale_bounds ε h0 h1
  obtain ⟨hl,hu⟩ := abs_lt.mp hgood
  have hp : 0 < estimate δ s := by linarith
  constructor <;> nlinarith [mul_nonneg (by linarith [hs.1] : 0 ≤ 2*mixedScale ε-1-1/2) hp.le,
    mul_nonneg (by linarith [hs.2.1] : 0 ≤ 1-(2*mixedScale ε-1)) hp.le]

def outcome (y : Response) : Bool := decide (y.1 = 0 ∧ y.2 = 0)

theorem outcome_reward (y : Response) :
    PositiveCertificateExample.reward (outcome y) = reward y := by
  simp [PositiveCertificateExample.reward, outcome, reward]

theorem implementation_signal (ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2)
    (b : Block) (i : Bool) :
    4 * BinaryUpdate.implementationCoefficient false (fun _ => 2) ε 100000000 10
      (![outcome b.1, outcome b.2] : Fin 2 → Bool) (if i then 0 else 1) =
      mixedScale ε * signal b i := by
  rw [implementation_coefficient_pair ε h0 h1 (outcome b.1, outcome b.2)]
  cases i <;> simp [signal, response, outcome_reward] <;> ring

theorem batch_scaled_increment {N : ℕ} (hN : N ≠ 0) (δ scale : ℝ) (s : Fin N → Block) :
    sampleMean (fun b => scale*groupZ δ b/4) s = scale*estimate δ s := by
  have hn : (N : ℝ) ≠ 0 := by exact_mod_cast hN
  have hL : sampleMean (fun _ : Block => (4 : ℝ)) s = 4 := by simp [sampleMean, hn]
  unfold estimate
  rw [hL]
  unfold sampleMean
  dsimp only
  have hs : (∑ i, scale*groupZ δ (s i)/4) = (scale/4)*∑ i, groupZ δ (s i) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hs]
  ring

noncomputable def marginCertificate {N : ℕ} (δ ε : ℝ) (s : Fin N → Block) : ℝ :=
  (2*mixedScale ε-1)*estimate δ s-δ/16

theorem margin_from_block_gains {N : ℕ} (hN : N ≠ 0) (δ ε : ℝ) (s : Fin N → Block) :
    sampleMean (fun b => mixedScale ε*groupZ δ b/4) s -
      sampleMean (fun b => (1-mixedScale ε)*groupZ δ b/4) s - δ/16 =
        marginCertificate δ ε s := by
  rw [batch_scaled_increment hN, batch_scaled_increment hN]
  unfold marginCertificate
  ring

theorem margin_certificate_failure_bound {N : ℕ} (hN : N ≠ 0) (K δ ε : ℝ)
    (hK : 0 < K) (hδ : 0 < δ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin N → Block =>
      marginCertificate δ ε s ≤ δ/32 ∨ δ/2 ≤ marginCertificate δ ε s),
      iidMass (blockMass K) s) ≤ 48/(N : ℝ) := by
  have hsubset : Finset.univ.filter (fun s : Fin N → Block =>
      marginCertificate δ ε s ≤ δ/32 ∨ δ/2 ≤ marginCertificate δ ε s) ⊆
      Finset.univ.filter (fun s => δ/16 ≤ |estimate δ s-δ/4|) := by
    intro s hs
    apply Finset.mem_filter.mpr
    refine ⟨Finset.mem_univ _, ?_⟩
    by_contra hn
    have hg := positive_margin_on_good_event δ ε hδ h0 h1 s (lt_of_not_ge hn)
    rcases (Finset.mem_filter.mp hs).2 with h | h <;>
      unfold marginCertificate at h <;> linarith
  exact (Finset.sum_le_sum_of_subset_of_nonneg hsubset
    (fun s _ _ => iid_mass_nonneg (blockMass K) (block_mass_nonneg K hK) s)).trans
    (estimate_tail hN K δ hK hδ)

theorem confidence_at_960 (N : ℕ) (hN : 960 ≤ N) : (48 : ℝ)/(N : ℝ) ≤ 1/20 := by
  have hn : (960 : ℝ) ≤ N := by exact_mod_cast hN
  apply (div_le_iff₀ (by linarith : (0 : ℝ) < N)).mpr
  linarith

/-- The positive example's population margin, after removing the optional
finite-sample evaluation procedure from the manuscript. -/
theorem population_objective_margin (K δ ε : ℝ) (hK : 0 < K) (hδ : 0 < δ)
    (hε0 : 0 ≤ ε) (hε1 : ε ≤ 1/2) :
    expectation (blockMass K)
      (fun b => mixedScale ε * groupZ δ b / 4 -
        (1 - mixedScale ε) * groupZ δ b / 4) =
          (2 * mixedScale ε - 1) * δ / 4 ∧
    0 < (2 * mixedScale ε - 1) * δ / 4 := by
  constructor
  · calc
      _ = ((2 * mixedScale ε - 1) / 4) * expectation (blockMass K) (groupZ δ) := by
        unfold expectation
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro b _
        ring
      _ = _ := by rw [(group_moments K δ hK).1]; ring
  · have hs := (mixed_scale_bounds ε hε0 hε1).1
    exact div_pos (mul_pos (by linarith) hδ) (by norm_num)

/-- The actual clipped objective increment as a scalar function. Inside
the feasible interval, the affine ratios are exactly those of
`selectiveCandidate`; the rollout law, gate and signals stay frozen. -/
noncomputable def scalarBlockObjective (K τ scale : ℝ) (hK : 0 < K)
    (b : Block) (δ : ℝ) : ℝ :=
  let p := selectivePolicy K hK false
  let old := selectivePolicy K hK true
  let M := fun it => prefixLogGate p old τ τ (history b it ++ [action b it])
  let w := fun it => tokenRatio p old (history b it) (action b it)
  let r := fun it : Bool × Bool => 1 + if it.2 then increment δ (response b it.1) else 0
  let H := fun it : Bool × Bool => scale * signal b it.1
  Objective.objective (fun _ => 1/4) M w r (fun it => min (max (r it) (4/5)) (6/5)) H -
    Objective.objective (fun _ => 1/4) M w (fun _ => 1) (fun _ => 1) H

noncomputable def scalarPopulationObjective (K τ scale : ℝ) (hK : 0 < K) (δ : ℝ) : ℝ :=
  expectation (blockMass K) (fun b => scalarBlockObjective K τ scale hK b δ)

theorem scalar_population_objective_eq (K δ τ scale : ℝ) (hK : 0 < K)
    (hδ : 0 ≤ δ) (hsmall : δ ≤ 1/15) (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log K) :
    scalarPopulationObjective K τ scale hK δ = scale * δ / 4 := by
  have hb : ∀ b, scalarBlockObjective K τ scale hK b δ = scale * groupZ δ b / 4 := by
    intro b
    have hr : (fun it : Bool × Bool => 1 + if it.2 then increment δ (response b it.1) else 0) =
        (fun it => tokenRatio (selectivePolicy K hK true)
          (selectiveCandidate K δ hK hδ hsmall) (history b it) (action b it)) := by
      funext it
      rcases it with ⟨i,t⟩
      cases t
      · simp [history, action, selective_candidate_root_ratio]
      · simp only [Bool.true_eq, ↓reduceIte, history, action]
        have hi := second_ratio_increment K δ hK hδ hsmall (response b i)
        linarith
    unfold scalarBlockObjective
    rw [hr]
    exact block_objective_gain K δ τ scale hK hδ hsmall hτ hmargin b
  unfold scalarPopulationObjective expectation
  simp_rw [hb]
  calc
    _ = (scale / 4) * expectation (blockMass K) (groupZ δ) := by
      unfold expectation
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro b _
      ring
    _ = _ := by rw [(group_moments K δ hK).1]; ring

/-- A derivative of the concrete PPO objective, at an interior point of
the existing scalar policy family, not a postulated ascent direction. -/
theorem scalar_population_objective_derivative (K τ scale δ₀ : ℝ) (hK : 0 < K)
    (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log K) (hδ₀ : 0 < δ₀) (hsmall : δ₀ < 1/15) :
    HasDerivAt (scalarPopulationObjective K τ scale hK) (scale / 4) δ₀ := by
  have heq : scalarPopulationObjective K τ scale hK =ᶠ[nhds δ₀]
      (fun δ : ℝ => scale * δ / 4) := by
    filter_upwards [Ioo_mem_nhds hδ₀ hsmall] with δ hδ
    exact scalar_population_objective_eq K δ τ scale hK hδ.1.le hδ.2.le hτ hmargin
  have hd : HasDerivAt (fun δ : ℝ => scale * δ / 4) (scale / 4) δ₀ := by
    simpa using ((hasDerivAt_id δ₀).const_mul scale).div_const 4
  exact hd.congr_of_eventuallyEq heq

/-- A population gradient-ascent step in that same one-parameter family
strictly improves reward. This is not an unrestricted neural-policy or
stochastic-optimizer claim. -/
theorem scalar_gradient_step_reward_gain (K τ ε δ₀ η : ℝ) (hK : 0 < K)
    (hτ : 0 ≤ τ) (hmargin : 2*τ < Real.log K) (hε0 : 0 ≤ ε) (hε1 : ε ≤ 1/2)
    (hδ₀ : 0 < δ₀) (hsmall : δ₀ < 1/15) (hη : 0 < η)
    (hstep : δ₀ + η * mixedScale ε / 4 ≤ 1/15) :
    let δ₁ := δ₀ + η * deriv (scalarPopulationObjective K τ (mixedScale ε) hK) δ₀
    ∃ (hδ₁ : 0 ≤ δ₁) (hsmall₁ : δ₁ ≤ 1/15),
      value (selectiveCandidate K δ₁ hK hδ₁ hsmall₁) selectiveReward 2 [] -
        value (selectiveCandidate K δ₀ hK hδ₀.le hsmall.le) selectiveReward 2 [] =
          η * mixedScale ε / 8 ∧ 0 < η * mixedScale ε / 8 := by
  dsimp only
  rw [(scalar_population_objective_derivative K τ (mixedScale ε) δ₀ hK hτ hmargin hδ₀ hsmall).deriv]
  have hs : 0 < mixedScale ε := lt_of_lt_of_le (by norm_num) (mixed_scale_bounds ε hε0 hε1).1
  have hpos : 0 ≤ δ₀ + η * (mixedScale ε / 4) := by positivity
  have hbound : δ₀ + η * (mixedScale ε / 4) ≤ 1/15 := by nlinarith [hstep]
  refine ⟨hpos, hbound, ?_, by positivity⟩
  rw [selective_candidate_reward K (δ₀ + η * (mixedScale ε / 4)) hK hpos hbound 0,
    selective_candidate_reward K δ₀ hK hδ₀.le hsmall.le 0]
  ring

end SelectiveSampling
end REINFORCEProMax.AutoregressiveTree
