import BatchPerformance
import NormalizationSafeguards
import TreePrefixGate
import BinaryUpdate

/-! Exact one-token, binary-reward instance of the corrected ProMax reward
certificate. Two responses form each independent RLOO evaluation block. -/
namespace REINFORCEProMax.PositiveCertificateExample
open scoped BigOperators
open BatchReduction Objective

noncomputable def coinPolicy (θ : ℝ) (h0 : 0 ≤ θ) (h1 : θ ≤ 1) :
    AutoregressiveTree.Policy Bool where
  prob _ b := if b then θ else 1-θ
  nonneg := by intro h b; cases b <;> simp <;> linarith
  mass := by intro h; simp [Fintype.sum_bool]

def reward (b : Bool) : ℝ := if b then 1 else 0
def ratio (δ : ℝ) (b : Bool) : ℝ := if b then 1+2*δ else 1-2*δ
def response (b : Bool × Bool) (i : Bool) : Bool := if i then b.1 else b.2
def signal (b : Bool × Bool) (i : Bool) : ℝ := reward (response b i) - reward (response b (!i))
noncomputable def groupZ (δ : ℝ) (b : Bool × Bool) : ℝ :=
  ∑ i : Bool, signal b i * (ratio δ (response b i)-1)
noncomputable def blockMass (_b : Bool × Bool) : ℝ := (1/2 : ℝ)*(1/2)
noncomputable def estimate {N : ℕ} (δ : ℝ) (s : Fin N → Bool × Bool) : ℝ :=
  sampleMean (groupZ δ) s / sampleMean (fun _ => 2) s

theorem ratio_from_policies (δ : ℝ) (hδ : 0 < δ) (hsmall : δ ≤ 1/10) (b : Bool) :
    AutoregressiveTree.tokenRatio
      (coinPolicy (1/2) (by norm_num) (by norm_num))
      (coinPolicy (1/2+δ) (by linarith) (by linarith)) [] b = ratio δ b := by
  cases b <;> simp [AutoregressiveTree.tokenRatio, coinPolicy, ratio] <;> ring

theorem exact_reward_gain (δ : ℝ) (hδ : 0 < δ) (hsmall : δ ≤ 1/10) :
    AutoregressiveTree.value (coinPolicy (1/2+δ) (by linarith) (by linarith))
      (fun h => if h.head? = some true then 1 else 0) 1 [] -
    AutoregressiveTree.value (coinPolicy (1/2) (by norm_num) (by norm_num))
      (fun h => if h.head? = some true then 1 else 0) 1 [] = δ := by
  simp [AutoregressiveTree.value, coinPolicy, Fintype.sum_bool]

theorem block_mass_from_rollout (b : Bool × Bool) :
    blockMass b =
      (coinPolicy (1/2) (by norm_num) (by norm_num)).prob [] b.1 *
      (coinPolicy (1/2) (by norm_num) (by norm_num)).prob [] b.2 := by
  rcases b with ⟨x,y⟩; cases x <;> cases y <;> norm_num [blockMass, coinPolicy]

theorem block_mass_normalized : ∑ b : Bool × Bool, blockMass b = 1 := by
  norm_num [blockMass, Fintype.sum_prod_type, Fintype.sum_bool]

theorem cached_gate_and_weight (τp τn : ℝ) (hp : 0 ≤ τp) (hn : 0 ≤ τn) (b : Bool) :
    let p := coinPolicy (1/2) (by norm_num) (by norm_num)
    AutoregressiveTree.tokenRatio p p [] b = 1 ∧
      AutoregressiveTree.prefixLogGate p p τp τn [b] = 1 := by
  cases b <;>
    norm_num [AutoregressiveTree.tokenRatio, AutoregressiveTree.prefixLogGate,
      AutoregressiveTree.logRatioTrace, AutoregressiveTree.ratioTrace,
      AutoregressiveTree.traceStep, coinPolicy, hp, neg_nonpos.mpr hn]

theorem group_increment (δ : ℝ) (b : Bool × Bool) :
    groupZ δ b = if b.1 = b.2 then 0 else 4*δ := by
  rcases b with ⟨x,y⟩; cases x <;> cases y <;>
    simp [groupZ, Fintype.sum_bool, signal, response, reward, ratio] <;> ring

theorem group_moments (δ : ℝ) :
    expectation blockMass (groupZ δ) = 2*δ ∧
    expectation blockMass (fun b => groupZ δ b - δ*2) = 0 ∧
    expectation blockMass (fun b => (groupZ δ b - δ*2)^2) = 4*δ^2 := by
  refine ⟨?_,?_,?_⟩ <;>
    simp [expectation, blockMass, Fintype.sum_prod_type, Fintype.sum_bool, group_increment] <;> ring

theorem estimate_tail {N : ℕ} (hN : N ≠ 0) (δ : ℝ) (hδ : 0 < δ) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin N → Bool × Bool => δ/4 ≤ |estimate δ s-δ|),
      iidMass blockMass s) ≤ 16/(N : ℝ) := by
  have hb := iid_token_mean_tail hN blockMass (groupZ δ) (fun _ => 2)
    (fun _ => by norm_num [blockMass]) block_mass_normalized
    2 δ (δ/4) (by norm_num) (by positivity) (fun _ => le_rfl) (group_moments δ).2.1
  rw [(group_moments δ).2.2] at hb
  have hden : 4*δ^2 / ((N : ℝ)*2^2*(δ/4)^2) = 16/(N : ℝ) := by
    have hd := ne_of_gt hδ
    field_simp
    <;> ring
  simpa only [estimate, hden] using hb

noncomputable def ppoClip (x : ℝ) : ℝ := min (max x (4/5)) (6/5)
noncomputable def blockObjective (δ s : ℝ) (b : Bool × Bool) : ℝ :=
  objective (fun _ : Bool => 1/2) (fun _ => 1) (fun _ => 1)
    (fun i => ratio δ (response b i)) (fun i => ppoClip (ratio δ (response b i)))
    (fun i => s * signal b i)

theorem ratio_clip_inactive (δ : ℝ) (h0 : 0 ≤ δ) (h1 : δ ≤ 1/10) (b : Bool) :
    ppoClip (ratio δ b) = ratio δ b := by
  have hlo : 4/5 ≤ ratio δ b := by cases b <;> simp [ratio] <;> linarith
  have hhi : ratio δ b ≤ 6/5 := by cases b <;> simp [ratio] <;> linarith
  simp [ppoClip, max_eq_left hlo, min_eq_left hhi]

theorem block_objective_gain (δ s : ℝ) (h0 : 0 ≤ δ) (h1 : δ ≤ 1/10) (b : Bool × Bool) :
    blockObjective δ s b - blockObjective 0 s b = s * groupZ δ b / 2 := by
  unfold blockObjective objective
  simp_rw [ratio_clip_inactive δ h0 h1, ratio_clip_inactive 0 (by norm_num) (by norm_num), min_self]
  rcases b with ⟨x,y⟩; cases x <;> cases y <;>
    simp [Fintype.sum_bool, signal, response, reward, ratio, groupZ] <;> ring

theorem batch_objective_gain {N : ℕ} (hN : N ≠ 0) (δ scale : ℝ)
    (h0 : 0 ≤ δ) (h1 : δ ≤ 1/10) (s : Fin N → Bool × Bool) :
    sampleMean (fun b => blockObjective δ scale b-blockObjective 0 scale b) s = scale*estimate δ s := by
  have hn : (N : ℝ) ≠ 0 := by exact_mod_cast hN
  have hL : sampleMean (fun _ : Bool × Bool => (2 : ℝ)) s = 2 := by
    simp [sampleMean, hn]
  simp_rw [block_objective_gain δ scale h0 h1]
  unfold estimate
  rw [hL]
  unfold sampleMean
  dsimp only
  have hs : (∑ i, scale * groupZ δ (s i) / 2) =
      (scale / 2) * ∑ i, groupZ δ (s i) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    ring
  rw [hs]
  ring

noncomputable def mixedScale (ε : ℝ) : ℝ := Real.sqrt (2/(2+ε))

theorem mixed_scale_safeguards (ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2) :
    3/4 ≤ mixedScale ε ∧ mixedScale ε ≤ 1 ∧
      NormalizationSafeguards.positiveClamp ε 10 (mixedScale ε) = mixedScale ε := by
  have hd : 0 < 2+ε := by linarith
  have hs : (mixedScale ε)^2 * (2+ε) = 2 := by
    rw [mixedScale, Real.sq_sqrt (by positivity)]
    exact div_mul_cancel₀ _ (ne_of_gt hd)
  have hn : 0 ≤ mixedScale ε := Real.sqrt_nonneg _
  have hx := mul_nonneg h0 (sq_nonneg (mixedScale ε))
  have hy := mul_le_mul_of_nonneg_right h1 (sq_nonneg (mixedScale ε))
  have hl : 3/4 ≤ mixedScale ε := by nlinarith
  have hu : mixedScale ε ≤ 1 := by nlinarith
  refine ⟨hl,hu,?_⟩
  simp [NormalizationSafeguards.positiveClamp, max_eq_left (show ε ≤ mixedScale ε by linarith),
    min_eq_left (show mixedScale ε ≤ 10 by linarith)]

/-- The example's coefficient is obtained from the existing implementation
model, including its RLOO baseline, fallback, product cap and factor clamps.
The factor 1/2 is the two active tokens' reduction denominator. -/
theorem implementation_coefficient_pair (ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2)
    (b : Bool × Bool) (i : Fin 2) :
    BinaryUpdate.implementationCoefficient false (fun _ => 1) ε 100000000 10
      (![b.1, b.2] : Fin 2 → Bool) i =
      mixedScale ε * (reward (![b.1, b.2] i) - reward (![b.2, b.1] i)) / 2 := by
  have he : ¬ (1 : ℝ) < ε := by linarith
  have hs := (mixed_scale_safeguards ε h0 h1).2.2
  unfold mixedScale at hs ⊢
  rw [Real.sqrt_div (by norm_num : (0 : ℝ) ≤ 2)] at hs
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
      NormalizationSafeguards.signScale, reward, he, h0, hs]

/-- Conversion from the implementation model's reduced coefficient to the
unreduced frozen signal used by `blockObjective`. -/
theorem implementation_signal_pair (ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2)
    (b : Bool × Bool) (i : Bool) :
    2 * BinaryUpdate.implementationCoefficient false (fun _ => 1) ε 100000000 10
      (![b.1, b.2] : Fin 2 → Bool) (if i then 0 else 1) = mixedScale ε * signal b i := by
  rw [implementation_coefficient_pair ε h0 h1]
  cases i <;> simp [signal, response] <;> ring

theorem implementation_objective_eq (δ ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1/2)
    (b : Bool × Bool) :
    objective (fun _ : Bool => 1/2) (fun _ => 1) (fun _ => 1)
      (fun i => ratio δ (response b i)) (fun i => ppoClip (ratio δ (response b i)))
      (fun i => 2 * BinaryUpdate.implementationCoefficient false (fun _ => 1)
        ε 100000000 10 (![b.1, b.2] : Fin 2 → Bool) (if i then 0 else 1)) =
      blockObjective δ (mixedScale ε) b := by
  simp_rw [implementation_signal_pair ε h0 h1]
  rfl

/-- On the same confidence event, the corrected bound is strictly positive
and below the actual reward gain. It does not assume a positive margin. -/
theorem positive_certificate_on_good_event {N : ℕ} (δ : ℝ) (hδ : 0 < δ)
    (s : Fin N → Bool × Bool) (hgood : |estimate δ s - δ| < δ/4) :
    δ/2 < estimate δ s - δ/4 ∧ estimate δ s - δ/4 < δ ∧ 0 < estimate δ s - δ/4 := by
  obtain ⟨hl,hu⟩ := abs_lt.mp hgood
  refine ⟨?_,?_,?_⟩ <;> linarith

theorem confidence_at_320 (N : ℕ) (hN : 320 ≤ N) : (16 : ℝ)/(N : ℝ) ≤ 1/20 := by
  have hn : (320 : ℝ) ≤ N := by exact_mod_cast hN
  apply (div_le_iff₀ (by linarith : (0 : ℝ) < N)).mpr
  linarith

noncomputable def marginCertificate {N : ℕ} (δ ε : ℝ) (s : Fin N → Bool × Bool) : ℝ :=
  (2 * mixedScale ε - 1) * estimate δ s - δ/4

noncomputable def blockDiscrepancy (δ scale : ℝ) (b : Bool × Bool) : ℝ :=
  ∑ i : Bool, (1/2 : ℝ) * |ratio δ (response b i)-1| *
    |signal b i - scale*signal b i|

theorem block_discrepancy_eq (δ scale : ℝ) (hδ : 0 ≤ δ)
    (hscale : scale ≤ 1) (b : Bool × Bool) :
    blockDiscrepancy δ scale b = (1-scale)*groupZ δ b/2 := by
  have ha : |2*δ| = 2*δ := abs_of_nonneg (by positivity)
  have hb : |1-scale| = 1-scale := abs_of_nonneg (by linarith)
  have hc : |-1+scale| = 1-scale := by rw [abs_of_nonpos (by linarith)]; ring
  rcases b with ⟨x,y⟩; cases x <;> cases y <;>
    simp [blockDiscrepancy, Fintype.sum_bool, signal, response, reward, ratio,
      group_increment, ha, hb, hc, abs_neg] <;> ring_nf

theorem margin_from_objective {N : ℕ} (hN : N ≠ 0) (δ ε : ℝ)
    (hδ0 : 0 ≤ δ) (hδ1 : δ ≤ 1/10) (hε0 : 0 ≤ ε) (hε1 : ε ≤ 1/2)
    (s : Fin N → Bool × Bool) :
    sampleMean (fun b => blockObjective δ (mixedScale ε) b -
      blockObjective 0 (mixedScale ε) b) s -
    sampleMean (blockDiscrepancy δ (mixedScale ε)) s - δ/4 =
      marginCertificate δ ε s := by
  rw [batch_objective_gain hN δ (mixedScale ε) hδ0 hδ1]
  have hs := (mixed_scale_safeguards ε hε0 hε1).2.1
  have heq : sampleMean (blockDiscrepancy δ (mixedScale ε)) s =
      (1-mixedScale ε)*estimate δ s := by
    have hfun : blockDiscrepancy δ (mixedScale ε) =
        (fun b => blockObjective δ (1-mixedScale ε) b -
          blockObjective 0 (1-mixedScale ε) b) := by
      funext b
      rw [block_discrepancy_eq δ (mixedScale ε) hδ0 hs,
        block_objective_gain δ (1-mixedScale ε) hδ0 hδ1]
    rw [hfun, batch_objective_gain hN δ (1-mixedScale ε) hδ0 hδ1]
  rw [heq]
  unfold marginCertificate
  ring

/-- Even the conservative objective-gain margin is positive here: the
normalization discrepancy is retained rather than exactly cancelled. -/
theorem margin_certificate_on_good_event {N : ℕ} (δ ε : ℝ) (hδ : 0 < δ)
    (hε0 : 0 ≤ ε) (hε1 : ε ≤ 1/2) (s : Fin N → Bool × Bool)
    (hgood : |estimate δ s - δ| < δ/4) :
    δ/8 < marginCertificate δ ε s ∧ marginCertificate δ ε s < δ := by
  obtain ⟨hl,hu⟩ := abs_lt.mp hgood
  obtain ⟨hs0,hs1,_⟩ := mixed_scale_safeguards ε hε0 hε1
  have hx : 0 ≤ estimate δ s := by linarith
  have hlo := mul_le_mul_of_nonneg_right (show (1/2 : ℝ) ≤ 2*mixedScale ε-1 by linarith) hx
  have hhi := mul_le_mul_of_nonneg_right (show 2*mixedScale ε-1 ≤ 1 by linarith) hx
  unfold marginCertificate
  constructor <;> linarith

theorem margin_certificate_failure_bound {N : ℕ} (hN : N ≠ 0) (δ ε : ℝ)
    (hδ : 0 < δ) (hε0 : 0 ≤ ε) (hε1 : ε ≤ 1/2) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin N → Bool × Bool =>
      marginCertificate δ ε s ≤ δ/8 ∨ δ ≤ marginCertificate δ ε s),
      iidMass blockMass s) ≤ 16/(N : ℝ) := by
  have hsubset : Finset.univ.filter (fun s : Fin N → Bool × Bool =>
      marginCertificate δ ε s ≤ δ/8 ∨ δ ≤ marginCertificate δ ε s) ⊆
      Finset.univ.filter (fun s => δ/4 ≤ |estimate δ s-δ|) := by
    intro s hs
    apply Finset.mem_filter.mpr
    refine ⟨Finset.mem_univ _, ?_⟩
    by_contra hn
    have hg := margin_certificate_on_good_event δ ε hδ hε0 hε1 s (lt_of_not_ge hn)
    rcases (Finset.mem_filter.mp hs).2 with h | h <;> linarith
  exact (Finset.sum_le_sum_of_subset_of_nonneg hsubset
    (fun s _ _ => iid_mass_nonneg blockMass (fun _ => by norm_num [blockMass]) s)).trans
    (estimate_tail hN δ hδ)

end REINFORCEProMax.PositiveCertificateExample
