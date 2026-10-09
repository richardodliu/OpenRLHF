import ProMaxCertificate

/-! Derive the finite certificate's ratio and padding premises from actual
policy probabilities and a shared absorbing EOS convention on positive-mass
rollout paths. Zero-probability padded strings are not treated as samples. -/
namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators
variable {A : Type*} [Fintype A] [DecidableEq A]

/-- The emitted EOS token is active; only tokens after an EOS are padding. -/
def eosActiveMask (eos : A) (h : List A) : ℝ := if eos ∈ h then 0 else 1

def AbsorbingEOS (p : Policy A) (eos : A) : Prop :=
  ∀ h, eos ∈ h → ∀ a, p.prob h a = if a = eos then 1 else 0

noncomputable def eosActiveLength (eos : A) (h : List A) : ℝ :=
  ∑ t : Fin h.length, eosActiveMask eos (h.take t.val)

theorem eos_mask_nonneg (eos : A) (h : List A) : 0 ≤ eosActiveMask eos h := by
  unfold eosActiveMask
  split_ifs <;> norm_num

theorem eos_length_lower_bound (eos : A) (h : List A) (hh : 0 < h.length) :
    1 ≤ eosActiveLength eos h := by
  have hb := Finset.single_le_sum
    (fun (i : Fin h.length) (_ : i ∈ Finset.univ) => eos_mask_nonneg eos (h.take i.val))
    (Finset.mem_univ (⟨0,hh⟩ : Fin h.length))
  simpa [eosActiveLength, eosActiveMask] using hb

/-- Every sampled action has positive conditional probability if the whole
sampled trajectory has positive mass, even when other vocabulary tokens do not. -/
theorem sampled_token_positive (p : Policy A) {T : ℕ} (h : List A) (y : Fin T → A)
    (hy : 0 < pathMass p h (List.ofFn y)) (t : Fin T) :
    0 < p.prob (h ++ (List.ofFn y).take t.val) (y t) := by
  induction T generalizing h with
  | zero => exact Fin.elim0 t
  | succ T ih =>
    rw [List.ofFn_succ, pathMass] at hy
    have hp : 0 < p.prob h (y 0) :=
      pos_of_mul_pos_left hy (pathMass_nonneg p _ _)
    have ht : 0 < pathMass p (h ++ [y 0]) (List.ofFn (fun i => y i.succ)) :=
      (mul_pos_iff_of_pos_left hp).mp hy
    cases t using Fin.cases with
    | zero => simpa using hp
    | succ t =>
      simpa [List.ofFn_succ, List.take_succ_cons, List.append_assoc] using
        ih (h ++ [y 0]) (fun i => y i.succ) ht t

theorem three_policy_ratio_on_support (p old q : Policy A) (h : List A) (a : A)
    (hp : 0 < p.prob h a) (ho : 0 < old.prob h a) :
    tokenRatio old q h a * tokenRatio p old h a = tokenRatio p q h a ∧
      tokenRatio old p h a * tokenRatio p old h a = 1 := by
  unfold tokenRatio
  have hp0 := ne_of_gt hp
  have ho0 := ne_of_gt ho
  constructor <;> field_simp

/-- At a positive-probability sampled action, both absorbing kernels assign
probability one after EOS. Hence padding has ratio one, not 0/0. -/
theorem absorbing_ratio_on_support (p q : Policy A) (eos : A)
    (hpEOS : AbsorbingEOS p eos) (hqEOS : AbsorbingEOS q eos)
    (h : List A) (a : A) (hh : eos ∈ h) (hp : 0 < p.prob h a) :
    tokenRatio p q h a = 1 := by
  have ha : a = eos := by
    by_contra hn
    rw [hpEOS h hh a, if_neg hn] at hp
    norm_num at hp
  subst a
  simp [tokenRatio, hpEOS h hh eos, hqEOS h hh eos]

theorem eos_mask_preserves_residual (p q : Policy A) (eos : A)
    (hpEOS : AbsorbingEOS p eos) (hqEOS : AbsorbingEOS q eos)
    (h : List A) (a : A) (hp : 0 < p.prob h a) :
    eosActiveMask eos h * (tokenRatio p q h a-1) = tokenRatio p q h a-1 := by
  by_cases hh : eos ∈ h
  · rw [absorbing_ratio_on_support p q eos hpEOS hqEOS h a hh hp]
    ring
  · simp [eosActiveMask, hh]

/-- The denominator is the actual active-token count and all three ratio
premises of the prior algebraic certificate are now derived from policies. -/
theorem corrected_eos_batch_change {n m T : ℕ}
    (p old q : Fin n → Policy A) (R : Fin n → List A → ℝ) (eos : A)
    (s : Fin n → Fin (m+1) → Fin T → A)
    (hpEOS : ∀ b, AbsorbingEOS (p b) eos) (hqEOS : ∀ b, AbsorbingEOS (q b) eos)
    (hold : ∀ b h a, 0 < (p b).prob h a → 0 < (old b).prob h a)
    (hsample : ∀ b i, 0 < pathMass (p b) [] (List.ofFn (s b i)))
    (M c c₀ H : (Fin n × Fin (m+1) × Fin T) → ℝ) :
    let D := ∑ b, groupActiveLength (eosActiveLength eos) (s b)
    let h := fun k : Fin n × Fin (m+1) × Fin T => (List.ofFn (s k.1 k.2.1)).take k.2.2.val
    let a := fun k : Fin n × Fin (m+1) × Fin T => s k.1 k.2.1 k.2.2
    correctedBatchChange (fun k => eosActiveMask eos (h k)/D) M
      (fun k => tokenRatio (p k.1) (old k.1) (h k) (a k))
      (fun k => tokenRatio (old k.1) (q k.1) (h k) (a k))
      (fun k => tokenRatio (old k.1) (p k.1) (h k) (a k)) c c₀
      (fun k => groupAdvantage (R k.1) (s k.1) k.2.1) H =
      (∑ b, groupSurrogateIncrement (p b) (q b) (R b) T (s b))/D := by
  dsimp only
  apply corrected_batch_change_eq_rloo_ratio p q R (fun _ => eosActiveLength eos) s
  · intro k
    have hp := sampled_token_positive (p k.1) [] (s k.1 k.2.1) (hsample k.1 k.2.1) k.2.2
    simp only [List.nil_append] at hp
    exact (three_policy_ratio_on_support (p k.1) (old k.1) (q k.1) _ _ hp (hold _ _ _ hp)).1
  · intro k
    have hp := sampled_token_positive (p k.1) [] (s k.1 k.2.1) (hsample k.1 k.2.1) k.2.2
    simp only [List.nil_append] at hp
    exact (three_policy_ratio_on_support (p k.1) (old k.1) (q k.1) _ _ hp (hold _ _ _ hp)).2
  · intro k
    have hp := sampled_token_positive (p k.1) [] (s k.1 k.2.1) (hsample k.1 k.2.1) k.2.2
    simp only [List.nil_append] at hp
    exact eos_mask_preserves_residual (p k.1) (q k.1) eos (hpEOS k.1) (hqEOS k.1) _ _ hp

section SampleSupport
variable {X : Type*} [Fintype X] [DecidableEq X]

theorem iid_nonzero_term {B : Type*} [Fintype B] [DecidableEq B] {n : ℕ}
    (f : B → ℝ) (s : Fin n → B) (hs : iidMass f s ≠ 0) (i : Fin n) : f (s i) ≠ 0 := by
  exact (Finset.prod_ne_zero_iff.mp hs) i (Finset.mem_univ i)

/-- Positive mass under the actual iid prompt/group law implies every
response follows positive-probability rollout transitions. -/
theorem supported_sample_responses {n m T : ℕ}
    (μ : X → ℝ) (p : X → Policy A) (s : Fin n → RolloutBlock X A m T)
    (hs : iidMass (blockMass μ p m T) s ≠ 0) (b : Fin n) (i : Fin (m+1)) :
    0 < pathMass (p (s b).1) [] (List.ofFn ((s b).2 i)) := by
  have hb := iid_nonzero_term (blockMass μ p m T) s hs b
  change μ (s b).1 * iidMass (rolloutMass (p (s b).1) T) (s b).2 ≠ 0 at hb
  have hg := (mul_ne_zero_iff.mp hb).2
  have hi := iid_nonzero_term (rolloutMass (p (s b).1) T) (s b).2 hg i
  exact lt_of_le_of_ne (pathMass_nonneg _ _ _) (Ne.symm hi)

/-- Invalid padded strings may exist in the finite sample type, but the
entire collection of batches containing one has probability mass zero. -/
theorem unsupported_sample_mass_zero {n m T : ℕ}
    (μ : X → ℝ) (p : X → Policy A) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T =>
      ¬ ∀ b i, 0 < pathMass (p (s b).1) [] (List.ofFn ((s b).2 i))),
      iidMass (blockMass μ p m T) s) = 0 := by
  classical
  apply Finset.sum_eq_zero
  intro s hs
  by_contra hn
  exact (Finset.mem_filter.mp hs).2 (supported_sample_responses μ p s hn)

/-- The concrete guard bundle consumed by the existing reward-certificate
conversion, now derived from the normalized iid sample law and policies. -/
theorem sampled_certificate_ratio_padding {n m T : ℕ}
    (μ : X → ℝ) (p old q : X → Policy A) (eos : A)
    (hpEOS : ∀ x, AbsorbingEOS (p x) eos) (hqEOS : ∀ x, AbsorbingEOS (q x) eos)
    (hold : ∀ x h a, 0 < (p x).prob h a → 0 < (old x).prob h a)
    (s : Fin n → RolloutBlock X A m T) (hs : iidMass (blockMass μ p m T) s ≠ 0)
    (k : Fin n × Fin (m+1) × Fin T) :
    let x := (s k.1).1
    let y := (s k.1).2 k.2.1
    let h := (List.ofFn y).take k.2.2.val
    let a := y k.2.2
    (tokenRatio (old x) (q x) h a * tokenRatio (p x) (old x) h a = tokenRatio (p x) (q x) h a) ∧
    (tokenRatio (old x) (p x) h a * tokenRatio (p x) (old x) h a = 1) ∧
    (eosActiveMask eos h * (tokenRatio (p x) (q x) h a-1) = tokenRatio (p x) (q x) h a-1) := by
  dsimp only
  have hp := sampled_token_positive (p (s k.1).1) [] ((s k.1).2 k.2.1)
    (supported_sample_responses μ p s hs k.1 k.2.1) k.2.2
  simp only [List.nil_append] at hp
  have hr := three_policy_ratio_on_support (p (s k.1).1) (old (s k.1).1) (q (s k.1).1)
    _ _ hp (hold _ _ _ hp)
  exact ⟨hr.1,hr.2,eos_mask_preserves_residual _ _ eos (hpEOS _) (hqEOS _) _ _ hp⟩

end SampleSupport

section PopulationObjective
variable {X : Type*} [Fintype X] [DecidableEq X]

/-- The paper's group margin, with the actual active-token denominator.
The gate, normalized signal and two clipped ratios may depend on the whole
response group. The policies themselves are fixed across that expectation. -/
noncomputable def eosGroupObjectiveMargin {m T : ℕ}
    (p old q : Policy A) (R : List A → ℝ) (eos : A)
    (s : Fin (m + 1) → Fin T → A)
    (M c c₀ H : (Fin (m + 1) × Fin T) → ℝ) : ℝ :=
  let L := groupActiveLength (eosActiveLength eos) s
  let h := fun k : Fin (m + 1) × Fin T => (List.ofFn (s k.1)).take k.2.val
  let a := fun k => eosActiveMask eos (h k) / L
  let w := fun k => tokenRatio p old (h k) (s k.1 k.2)
  let r := fun k => tokenRatio old q (h k) (s k.1 k.2)
  let r₀ := fun k => tokenRatio old p (h k) (s k.1 k.2)
  let V := fun k : Fin (m + 1) × Fin T => groupAdvantage R s k.1
  L * (Objective.objective a M w r c H - Objective.objective a M w r₀ c₀ H -
    Objective.incrementDistortion a M w r r₀ V H - Objective.clippingPenalty a M w r₀ c₀ H)

/-- The same actual group objective compared with a positive reference
scale of the raw advantage. The original objective and clipping are kept. -/
noncomputable def eosGroupScaledObjectiveMargin {m T : ℕ}
    (scale : ℝ) (p old q : Policy A) (R : List A → ℝ) (eos : A)
    (s : Fin (m + 1) → Fin T → A)
    (M c c₀ H : (Fin (m + 1) × Fin T) → ℝ) : ℝ :=
  let L := groupActiveLength (eosActiveLength eos) s
  let h := fun k : Fin (m + 1) × Fin T => (List.ofFn (s k.1)).take k.2.val
  let a := fun k => eosActiveMask eos (h k) / L
  let w := fun k => tokenRatio p old (h k) (s k.1 k.2)
  let r := fun k => tokenRatio old q (h k) (s k.1 k.2)
  let r₀ := fun k => tokenRatio old p (h k) (s k.1 k.2)
  let V := fun k : Fin (m + 1) × Fin T => groupAdvantage R s k.1
  L * (Objective.objective a M w r c H - Objective.objective a M w r₀ c₀ H -
    Objective.incrementDistortion a M w r r₀ (fun k => scale * V k) H -
    Objective.clippingPenalty a M w r₀ c₀ H) / scale

/-- Direct population reward bound for the concrete EOS-masked ProMax
objective. This discharges the abstract pointwise-margin premise inside
Lean, including zero-mass strings and the random token denominator. -/
theorem population_performance_from_scaled_eos_objective {m T : ℕ} (hm : m ≠ 0) (hT : 0 < T)
    (scale : ℝ) (hscale : 0 < scale)
    (μ : X → ℝ) (p old q : X → Policy A) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x)
    (B ε δ : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 ↔ (q x).prob h a = 0)
    (hTV : ∀ x h, h.length < T → localTV (p x) (q x) h ≤ ε)
    (hKL : ∀ x h, h.length < T → localReverseKL (p x) (q x) h ≤ δ)
    (eos : A) (hpEOS : ∀ x, AbsorbingEOS (p x) eos)
    (hqEOS : ∀ x, AbsorbingEOS (q x) eos)
    (hold : ∀ x h a, 0 < (p x).prob h a → 0 < (old x).prob h a)
    (M c c₀ H : RolloutBlock X A m T → (Fin (m + 1) × Fin T) → ℝ)
    (hM : ∀ b k, 0 ≤ M b k) :
    BatchReduction.expectation (blockMass μ p m T)
      (fun b => eosGroupScaledObjectiveMargin scale (p b.1) (old b.1) (q b.1) (R b.1)
        eos b.2 (M b) (c b) (c₀ b) (H b)) / (m + 1 : ℝ) -
      familyRewardCost μ p q B ε δ T ≤
        familyValue μ q R T - familyValue μ p R T := by
  classical
  let F := fun b : RolloutBlock X A m T =>
    eosGroupScaledObjectiveMargin scale (p b.1) (old b.1) (q b.1) (R b.1)
      eos b.2 (M b) (c b) (c₀ b) (H b)
  have hpoint : ∀ b, blockMass μ p m T b ≠ 0 → F b ≤ blockIncrement p q R b := by
    intro b hb
    have hsample : ∀ i, 0 < pathMass (p b.1) [] (List.ofFn (b.2 i)) := by
      intro i
      have hg := (mul_ne_zero_iff.mp hb).2
      have hi := iid_nonzero_term (rolloutMass (p b.1) T) b.2 hg i
      exact lt_of_le_of_ne (pathMass_nonneg _ _ _) (Ne.symm hi)
    let L := groupActiveLength (eosActiveLength eos) b.2
    have hL : 0 < L := by
      have hl : (m + 1 : ℝ) * 1 ≤ L := by
        unfold L groupActiveLength
        calc
          (m + 1 : ℝ) * 1 = ∑ _i : Fin (m + 1), (1 : ℝ) := by simp
          _ ≤ _ := Finset.sum_le_sum fun i _ =>
            eos_length_lower_bound eos _ (by simpa using hT)
      exact lt_of_lt_of_le (by positivity) hl
    let h := fun k : Fin (m + 1) × Fin T => (List.ofFn (b.2 k.1)).take k.2.val
    let a := fun k => eosActiveMask eos (h k) / L
    let w := fun k => tokenRatio (p b.1) (old b.1) (h k) (b.2 k.1 k.2)
    let r := fun k => tokenRatio (old b.1) (q b.1) (h k) (b.2 k.1 k.2)
    let r₀ := fun k => tokenRatio (old b.1) (p b.1) (h k) (b.2 k.1 k.2)
    let V := fun k : Fin (m + 1) × Fin T => groupAdvantage (R b.1) b.2 k.1
    have hrat : ∀ k, r k * w k = responseTokenRatio (p b.1) (q b.1) (b.2 k.1) k.2 ∧
        r₀ k * w k = 1 := by
      intro k
      have ht := sampled_token_positive (p b.1) [] (b.2 k.1) (hsample k.1) k.2
      simp only [List.nil_append] at ht
      exact three_policy_ratio_on_support _ _ _ _ _ ht (hold _ _ _ ht)
    have hpad : ∀ k : Fin (m + 1) × Fin T,
        eosActiveMask eos (h k) *
          (responseTokenRatio (p b.1) (q b.1) (b.2 k.1) k.2 - 1) =
          responseTokenRatio (p b.1) (q b.1) (b.2 k.1) k.2 - 1 := by
      intro k
      have ht := sampled_token_positive (p b.1) [] (b.2 k.1) (hsample k.1) k.2
      simp only [List.nil_append] at ht
      exact eos_mask_preserves_residual _ _ eos (hpEOS _) (hqEOS _) _ _ ht
    have hraw : Objective.rawSurrogate a w r V - Objective.rawSurrogate a w r₀ V =
        blockIncrement p q R b / L := by
      unfold Objective.rawSurrogate
      rw [← Finset.sum_sub_distrib]
      simp_rw [(hrat _).1, (hrat _).2]
      have heq : ∀ k : Fin (m + 1) × Fin T,
          a k * responseTokenRatio (p b.1) (q b.1) (b.2 k.1) k.2 * V k -
          a k * 1 * V k =
          V k * (responseTokenRatio (p b.1) (q b.1) (b.2 k.1) k.2 - 1) / L := by
        intro k
        change (eosActiveMask eos (h k) / L) * _ * _ -
          (eosActiveMask eos (h k) / L) * _ * _ = _
        calc
          _ = V k * (eosActiveMask eos (h k) *
            (responseTokenRatio (p b.1) (q b.1) (b.2 k.1) k.2 - 1)) / L := by ring
          _ = _ := by rw [hpad]
      simp_rw [heq]
      rw [← Finset.sum_div, Fintype.sum_prod_type]
      congr 1
      exact (group_increment_eq_token_sum (p b.1) (q b.1) (R b.1) b.2).symm
    have hmarg := Objective.raw_increment_ge_scaled_objective_increment a (M b) w r r₀
      (c b) (c₀ b) V (H b) scale hscale
      (fun k => div_nonneg (eos_mask_nonneg eos _) hL.le) (hM b)
      (fun k => div_nonneg ((old b.1).nonneg _ _) ((p b.1).nonneg _ _))
    rw [hraw] at hmarg
    have hout := mul_le_mul_of_nonneg_left hmarg hL.le
    rw [mul_div_cancel₀ _ (ne_of_gt hL)] at hout
    change L * _ / scale ≤ _
    simpa only [mul_div_assoc] using hout
  let F' := fun b => if blockMass μ p m T b = 0 then blockIncrement p q R b else F b
  have hF' : ∀ b, F' b ≤ blockIncrement p q R b := by
    intro b
    dsimp only [F']
    split_ifs with hb
    · exact le_rfl
    · exact hpoint b hb
  have hexp : BatchReduction.expectation (blockMass μ p m T) F' =
      BatchReduction.expectation (blockMass μ p m T) F := by
    apply Finset.sum_congr rfl
    intro b _
    by_cases hb : blockMass μ p m T b = 0
    · simp [hb]
    · simp [F', hb]
  have hbound := population_performance_from_objective_margin hm μ p q R hμ0
    B ε δ hB hR hs hTV hKL F' hF'
  rw [hexp] at hbound
  exact hbound

/-- The previous margin is the reference-scale-one specialization. -/
theorem population_performance_from_eos_objective {m T : ℕ} (hm : m ≠ 0) (hT : 0 < T)
    (μ : X → ℝ) (p old q : X → Policy A) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x)
    (B ε δ : ℝ) (hB : 0 ≤ B) (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 ↔ (q x).prob h a = 0)
    (hTV : ∀ x h, h.length < T → localTV (p x) (q x) h ≤ ε)
    (hKL : ∀ x h, h.length < T → localReverseKL (p x) (q x) h ≤ δ)
    (eos : A) (hpEOS : ∀ x, AbsorbingEOS (p x) eos)
    (hqEOS : ∀ x, AbsorbingEOS (q x) eos)
    (hold : ∀ x h a, 0 < (p x).prob h a → 0 < (old x).prob h a)
    (M c c₀ H : RolloutBlock X A m T → (Fin (m + 1) × Fin T) → ℝ)
    (hM : ∀ b k, 0 ≤ M b k) :
    BatchReduction.expectation (blockMass μ p m T)
      (fun b => eosGroupObjectiveMargin (p b.1) (old b.1) (q b.1) (R b.1)
        eos b.2 (M b) (c b) (c₀ b) (H b)) / (m + 1 : ℝ) -
      familyRewardCost μ p q B ε δ T ≤
        familyValue μ q R T - familyValue μ p R T := by
  simpa only [eosGroupScaledObjectiveMargin, eosGroupObjectiveMargin, one_mul, div_one] using
    population_performance_from_scaled_eos_objective hm hT 1 (by norm_num)
      μ p old q R hμ0 B ε δ hB hR hs hTV hKL eos hpEOS hqEOS hold M c c₀ H hM

/-- Restore the shared batch denominator before taking expectation.
Here F is an unreduced group margin (eosGroupObjectiveMargin), so the
quotient is the actual token-weighted batch margin. No independence of
length and margin is used; only iid complete groups and positive lengths. -/
theorem restored_batch_margin_expectation {Y : Type*} [Fintype Y] [DecidableEq Y]
    {K : ℕ} (hK : K ≠ 0) (mass L F : Y → ℝ)
    (hmass : ∑ y, mass y = 1) (hL : ∀ y, 0 < L y) (n : ℝ) :
    BatchReduction.expectation (iidMass mass) (fun s : Fin K → Y =>
      (∑ i, L (s i)) * ((∑ i, F (s i)) / ∑ i, L (s i))) / ((K : ℝ) * n) =
      BatchReduction.expectation mass F / n := by
  have hlen : ∀ s : Fin K → Y, (∑ i, L (s i)) ≠ 0 := by
    intro s
    apply ne_of_gt
    exact Finset.sum_pos (fun i _ => hL (s i))
      (Finset.univ_nonempty_iff.mpr ⟨⟨0, Nat.pos_of_ne_zero hK⟩⟩)
  have hcancel : ∀ s : Fin K → Y,
      (∑ i, L (s i)) * ((∑ i, F (s i)) / ∑ i, L (s i)) = ∑ i, F (s i) := by
    intro s
    exact mul_div_cancel₀ _ (hlen s)
  simp_rw [hcancel]
  rw [← div_div]
  congr 1
  have hm := BatchReduction.iid_sampleMean_expectation hK mass F hmass
  simpa only [BatchReduction.sampleMean, BatchReduction.expectation,
    mul_div_assoc, Finset.sum_div] using hm

/-- The batch identity instantiated with the same EOS objective margin
and actual group lengths as the population reward theorem. -/
theorem eos_batch_margin_expectation {m T K : ℕ} (hK : K ≠ 0) (hT : 0 < T)
    (μ : X → ℝ) (p old q : X → Policy A) (R : X → List A → ℝ)
    (hμ : ∑ x, μ x = 1) (eos : A)
    (M c c₀ H : RolloutBlock X A m T → (Fin (m + 1) × Fin T) → ℝ) :
    let F := fun b : RolloutBlock X A m T =>
      eosGroupObjectiveMargin (p b.1) (old b.1) (q b.1) (R b.1)
        eos b.2 (M b) (c b) (c₀ b) (H b)
    let L := blockLength (fun _ : X => eosActiveLength eos)
    BatchReduction.expectation (iidMass (blockMass μ p m T))
      (fun s : Fin K → RolloutBlock X A m T =>
        (∑ i, L (s i)) * ((∑ i, F (s i)) / ∑ i, L (s i))) / ((K : ℝ) * (m + 1)) =
      BatchReduction.expectation (blockMass μ p m T) F / (m + 1 : ℝ) := by
  let F := fun b : RolloutBlock X A m T =>
    eosGroupObjectiveMargin (p b.1) (old b.1) (q b.1) (R b.1)
      eos b.2 (M b) (c b) (c₀ b) (H b)
  let L : RolloutBlock X A m T → ℝ := blockLength (fun _ : X => eosActiveLength eos)
  have hlen : ∀ b, 0 < L b := by
    intro b
    have hb := block_length_lower_bound (fun _ : X => eosActiveLength eos) (1 : ℝ)
      (fun _ y => eos_length_lower_bound eos (List.ofFn y) (by simpa using hT)) b
    exact lt_of_lt_of_le (by positivity) hb
  exact restored_batch_margin_expectation hK (blockMass μ p m T) L F
    (block_mass_normalized μ p hμ m T) hlen (m + 1 : ℝ)

end PopulationObjective

end REINFORCEProMax.AutoregressiveTree
