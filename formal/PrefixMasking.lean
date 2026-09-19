import ProMax

/-!
Trajectory-level version of the paper's three conditional masking cases.
Positions use zero-based `Fin n` indices; the prefix at index j contains j+1
active actions. No padding position is included in this active-index model.
Masks are the actual binary closed-interval tests, and rejection counts are
cardinalities of their zero sets. Both signs and threshold boundary cases are
handled without replacing bounded background noise by an all-zero trajectory.
-/
namespace REINFORCEProMax.PrefixMasking

open scoped BigOperators

noncomputable def binaryMask (P : Prop) : ℕ := by classical exact if P then 1 else 0
noncomputable def tokenMask {n : ℕ} (z : Fin n → ℝ) (lo hi : ℝ) (i : Fin n) : ℕ :=
  binaryMask (tokenAccepted z lo hi i)
noncomputable def prefixMask {n : ℕ} (z : Fin n → ℝ) (lo hi : ℝ) (i : Fin n) : ℕ :=
  binaryMask (prefixAccepted z lo hi i)
noncomputable def sequenceMask {n : ℕ} (z : Fin n → ℝ) (lo hi : ℝ) (_i : Fin n) : ℕ :=
  binaryMask (sequenceAccepted z lo hi)
noncomputable def rejectionCount {n : ℕ} (M : Fin n → ℕ) : ℕ := by
  classical exact (Finset.univ.filter (fun i => M i = 0)).card

@[simp] theorem binaryMask_zero (P : Prop) : binaryMask P = 0 ↔ ¬P := by
  classical simp [binaryMask]
@[simp] theorem binaryMask_one (P : Prop) : binaryMask P = 1 ↔ P := by
  classical simp [binaryMask]

/-- Log-space and positive-ratio-space gates are exactly the same binary test. -/
theorem exp_interval_mask (lo hi z : ℝ) :
    binaryMask (intervalAccepted (Real.exp lo) (Real.exp hi) (Real.exp z)) =
      binaryMask (intervalAccepted lo hi z) := by
  simp only [intervalAccepted, Real.exp_le_exp]

theorem prefixSum_Iic {n : ℕ} (z : Fin n → ℝ) (j : Fin n) :
    prefixSum z j = ∑ i ∈ Finset.Iic j, z i := by
  unfold prefixSum
  congr 1
  ext i
  simp

theorem prefix_interval {n : ℕ} (z : Fin n → ℝ) (lo hi : ℝ) (j : Fin n)
    (hz : ∀ i, i ≤ j → tokenAccepted z lo hi i) : prefixAccepted z lo hi j := by
  have hden : 0 < ((j.val + 1 : ℕ) : ℝ) := by positivity
  have hl : ((j.val + 1 : ℕ) : ℝ) * lo ≤ prefixSum z j := by
    rw [prefixSum_Iic]
    have hb := Finset.sum_le_sum (s := Finset.Iic j)
      (fun i hi => (hz i (Finset.mem_Iic.mp hi)).1)
    simpa using hb
  have hu : prefixSum z j ≤ ((j.val + 1 : ℕ) : ℝ) * hi := by
    rw [prefixSum_Iic]
    have hb := Finset.sum_le_sum (s := Finset.Iic j)
      (fun i hi => (hz i (Finset.mem_Iic.mp hi)).2)
    simpa using hb
  constructor
  · exact (le_div_iff₀ hden).mpr (by simpa [mul_comm] using hl)
  · exact (div_le_iff₀ hden).mpr (by simpa [mul_comm] using hu)

theorem prefix_strict_upper {n : ℕ} (z : Fin n → ℝ) (hi : ℝ) (j : Fin n)
    (hz : ∀ i, i ≤ j → hi < z i) : hi < prefixAverage z j := by
  have hne : (Finset.Iic j).Nonempty := ⟨j, by simp⟩
  have hb := Finset.sum_lt_sum_of_nonempty hne (fun i hi => hz i (Finset.mem_Iic.mp hi))
  have hs : ((j.val + 1 : ℕ) : ℝ) * hi < prefixSum z j := by
    rw [prefixSum_Iic]
    simpa using hb
  exact (lt_div_iff₀ (by positivity : (0 : ℝ) < ((j.val + 1 : ℕ) : ℝ))).mpr
    (by simpa [mul_comm] using hs)

theorem prefix_strict_lower {n : ℕ} (z : Fin n → ℝ) (lo : ℝ) (j : Fin n)
    (hz : ∀ i, i ≤ j → z i < lo) : prefixAverage z j < lo := by
  have hne : (Finset.Iic j).Nonempty := ⟨j, by simp⟩
  have hb := Finset.sum_lt_sum_of_nonempty hne (fun i hi => hz i (Finset.mem_Iic.mp hi))
  have hs : prefixSum z j < ((j.val + 1 : ℕ) : ℝ) * lo := by
    rw [prefixSum_Iic]
    simpa using hb
  exact (div_lt_iff₀ (by positivity : (0 : ℝ) < ((j.val + 1 : ℕ) : ℝ))).mpr
    (by simpa [mul_comm] using hs)

theorem prefixAverage_first {m : ℕ} (z : Fin (m + 1) → ℝ) : prefixAverage z 0 = z 0 := by
  rw [prefixAverage, prefixSum_Iic]
  have hs : Finset.Iic (0 : Fin (m + 1)) = {0} := by ext i; simp
  simp [hs]

theorem rejectionCount_singleton {n : ℕ} (M : Fin n → ℕ) (k : Fin n)
    (hM : ∀ i, M i = 0 ↔ i = k) : rejectionCount M = 1 := by
  classical
  have hs : Finset.univ.filter (fun i => M i = 0) = {k} := by ext i; simp [hM]
  simp [rejectionCount, hs]

theorem rejectionCount_all {n : ℕ} (M : Fin n → ℕ) (hM : ∀ i, M i = 0) :
    rejectionCount M = n := by classical simp [rejectionCount, hM]

theorem rejectionCount_pos {n : ℕ} (M : Fin n → ℕ) (k : Fin n) (hk : M k = 0) :
    1 ≤ rejectionCount M := by
  classical
  exact Finset.card_pos.mpr ⟨k, by simp [hk]⟩

theorem sequence_all_or_none {n : ℕ} (z : Fin n → ℝ) (lo hi : ℝ) :
    rejectionCount (sequenceMask z lo hi) = 0 ∨ rejectionCount (sequenceMask z lo hi) = n := by
  classical
  by_cases h : sequenceAccepted z lo hi
  · left; simp [rejectionCount, sequenceMask, binaryMask, h]
  · right; simp [rejectionCount, sequenceMask, binaryMask, h]

theorem bounded_noise_accepted {n : ℕ} (z : Fin n → ℝ) (lo hi ε : ℝ) (i : Fin n)
    (hlo : ε ≤ -lo) (hhi : ε ≤ hi) (hz : |z i| ≤ ε) : tokenAccepted z lo hi i := by
  rcases abs_le.mp hz with ⟨hl, hu⟩
  constructor <;> dsimp <;> linarith

/-- Retain the first spike exactly and bound every subsequent value by epsilon. -/
theorem early_prefix_sum_bounds {m : ℕ} (z : Fin (m + 1) → ℝ) (ε : ℝ)
    (hnoise : ∀ i, i ≠ 0 → |z i| ≤ ε) (j : Fin (m + 1)) :
    z 0 - (j.val : ℝ) * ε ≤ prefixSum z j ∧ prefixSum z j ≤ z 0 + (j.val : ℝ) * ε := by
  classical
  let tail := (Finset.Iic j).erase 0
  have hzero : (0 : Fin (m + 1)) ∈ Finset.Iic j := by simp
  have hcard : tail.card = j.val := by
    dsimp [tail]
    rw [Finset.card_erase_of_mem hzero, Fin.card_Iic]
    omega
  have hl : -(j.val : ℝ) * ε ≤ ∑ i ∈ tail, z i := by
    have hb := Finset.sum_le_sum (s := tail) (fun i hi =>
      (abs_le.mp (hnoise i (Finset.mem_erase.mp hi).1)).1)
    simpa [hcard, mul_neg, neg_mul] using hb
  have hu : (∑ i ∈ tail, z i) ≤ (j.val : ℝ) * ε := by
    have hb := Finset.sum_le_sum (s := tail) (fun i hi =>
      (abs_le.mp (hnoise i (Finset.mem_erase.mp hi).1)).2)
    simpa [hcard] using hb
  have hsum : (∑ i ∈ tail, z i) + z 0 = prefixSum z j := by
    rw [prefixSum_Iic]
    exact Finset.sum_erase_add _ _ hzero
  constructor <;> linarith

theorem early_upper_rejects {m : ℕ} (z : Fin (m + 1) → ℝ) (lo hi ε : ℝ)
    (hnoise : ∀ i, i ≠ 0 → |z i| ≤ ε) (hden : 0 < hi + ε) (j : Fin (m + 1))
    (hwindow : ((j.val + 1 : ℕ) : ℝ) < (z 0 + ε) / (hi + ε)) : prefixMask z lo hi j = 0 := by
  have hb := (early_prefix_sum_bounds z ε hnoise j).1
  have hw := (lt_div_iff₀ hden).mp hwindow
  have hp : hi < prefixAverage z j := by
    apply (lt_div_iff₀ (by positivity : (0 : ℝ) < ((j.val + 1 : ℕ) : ℝ))).mpr
    push_cast at hw ⊢
    nlinarith
  exact (binaryMask_zero _).mpr (fun h => (not_le_of_gt hp) h.2)

theorem early_lower_rejects {m : ℕ} (z : Fin (m + 1) → ℝ) (lo hi ε : ℝ)
    (hnoise : ∀ i, i ≠ 0 → |z i| ≤ ε) (hden : 0 < -lo + ε) (j : Fin (m + 1))
    (hwindow : ((j.val + 1 : ℕ) : ℝ) < (-z 0 + ε) / (-lo + ε)) : prefixMask z lo hi j = 0 := by
  have hb := (early_prefix_sum_bounds z ε hnoise j).2
  have hw := (lt_div_iff₀ hden).mp hwindow
  have hp : prefixAverage z j < lo := by
    apply (div_lt_iff₀ (by positivity : (0 : ℝ) < ((j.val + 1 : ℕ) : ℝ))).mpr
    push_cast at hw ⊢
    nlinarith
  exact (binaryMask_zero _).mpr (fun h => (not_le_of_gt hp) h.1)

theorem token_single_deviation {n : ℕ} (z : Fin n → ℝ) (lo hi ε : ℝ) (k : Fin n)
    (hlo : ε ≤ -lo) (hhi : ε ≤ hi) (hk : ¬tokenAccepted z lo hi k)
    (hnoise : ∀ i, i ≠ k → |z i| ≤ ε) :
    (∀ i, tokenMask z lo hi i = 0 ↔ i = k) ∧ rejectionCount (tokenMask z lo hi) = 1 := by
  have hpattern : ∀ i, tokenMask z lo hi i = 0 ↔ i = k := by
    intro i
    change binaryMask (tokenAccepted z lo hi i) = 0 ↔ i = k
    rw [binaryMask_zero]
    constructor
    · intro h
      by_contra hne
      exact h (bounded_noise_accepted z lo hi ε i hlo hhi (hnoise i hne))
    · rintro rfl
      exact hk
  exact ⟨hpattern, rejectionCount_singleton _ k hpattern⟩

structure EarlyConclusion {m : ℕ} (z : Fin (m + 1) → ℝ) (lo hi ε : ℝ) : Prop where
  token_pattern : ∀ i, tokenMask z lo hi i = 0 ↔ i = 0
  token_count : rejectionCount (tokenMask z lo hi) = 1
  prefix_first : prefixMask z lo hi 0 = 0
  count_dominance : rejectionCount (tokenMask z lo hi) ≤ rejectionCount (prefixMask z lo hi)
  upper_window : ∀ j, ((j.val + 1 : ℕ) : ℝ) < (z 0 + ε) / (hi + ε) →
    prefixMask z lo hi j = 0
  lower_window : ∀ j, ((j.val + 1 : ℕ) : ℝ) < (-z 0 + ε) / (-lo + ε) →
    prefixMask z lo hi j = 0

theorem early_deviation_complete {m : ℕ} (z : Fin (m + 1) → ℝ) (lo hi ε : ℝ)
    (hlo : lo < 0) (hhi : 0 < hi) (hε : 0 ≤ ε) (hεlo : ε ≤ -lo) (hεhi : ε ≤ hi)
    (hfirst : ¬tokenAccepted z lo hi 0) (hnoise : ∀ i, i ≠ 0 → |z i| ≤ ε) :
    EarlyConclusion z lo hi ε := by
  have ht := token_single_deviation z lo hi ε 0 hεlo hεhi hfirst hnoise
  have hp : prefixMask z lo hi 0 = 0 := by
    apply (binaryMask_zero _).mpr
    simpa [prefixAccepted, tokenAccepted, prefixAverage_first] using hfirst
  exact ⟨ht.1, ht.2, hp, by rw [ht.2]; exact rejectionCount_pos _ 0 hp,
    early_upper_rejects z lo hi ε hnoise (by linarith),
    early_lower_rejects z lo hi ε hnoise (by linarith)⟩

theorem late_prefix_accepted {n : ℕ} (z : Fin n → ℝ) (lo hi ε : ℝ) (k : Fin n)
    (hlo : ε ≤ -lo) (hhi : ε ≤ hi) (hnoise : ∀ i, i ≠ k → |z i| ≤ ε)
    (j : Fin n) (hj : j < k) : prefixMask z lo hi j = 1 := by
  apply (binaryMask_one _).mpr
  apply prefix_interval
  intro i hij
  exact bounded_noise_accepted z lo hi ε i hlo hhi
    (hnoise i (ne_of_lt (lt_of_le_of_lt hij hj)))

theorem late_rejection_count {n : ℕ} (M : Fin n → ℕ) (k : Fin n)
    (hbefore : ∀ j, j < k → M j = 1) : rejectionCount M ≤ n - k.val := by
  classical
  have hsub : Finset.univ.filter (fun i => M i = 0) ⊆ Finset.Ici k := by
    intro i hi
    apply Finset.mem_Ici.mpr
    by_contra h
    have hone := hbefore i (lt_of_not_ge h)
    have hzero := (Finset.mem_filter.mp hi).2
    omega
  simpa [rejectionCount] using Finset.card_le_card hsub

structure LateConclusion {n : ℕ} (z : Fin n → ℝ) (lo hi : ℝ) (k : Fin n) : Prop where
  token_pattern : ∀ i, tokenMask z lo hi i = 0 ↔ i = k
  token_count : rejectionCount (tokenMask z lo hi) = 1
  before_accepted : ∀ j, j < k → prefixMask z lo hi j = 1
  prefix_count : rejectionCount (prefixMask z lo hi) ≤ n - k.val
  sequence_count : rejectionCount (sequenceMask z lo hi) = 0 ∨
    rejectionCount (sequenceMask z lo hi) = n

theorem late_deviation_complete {n : ℕ} (z : Fin n → ℝ) (lo hi ε : ℝ) (k : Fin n)
    (hlo : ε ≤ -lo) (hhi : ε ≤ hi) (hk : ¬tokenAccepted z lo hi k)
    (hnoise : ∀ i, i ≠ k → |z i| ≤ ε) : LateConclusion z lo hi k := by
  have ht := token_single_deviation z lo hi ε k hlo hhi hk hnoise
  have hp := late_prefix_accepted z lo hi ε k hlo hhi hnoise
  exact ⟨ht.1, ht.2, hp, late_rejection_count _ k hp, sequence_all_or_none z lo hi⟩

theorem prefixAverage_last {m : ℕ} (z : Fin (m + 1) → ℝ) :
    prefixAverage z (Fin.last m) = (∑ i, z i) / ((m + 1 : ℕ) : ℝ) := by
  rw [prefixAverage, prefixSum_Iic]
  have hs : Finset.Iic (Fin.last m) = Finset.univ := by ext i; simp [Fin.le_last]
  simp [hs]

/-- At the final active token, prefix and sequence tests coincide for every trajectory. -/
theorem final_prefix_eq_sequence {m : ℕ} (z : Fin (m + 1) → ℝ) (lo hi : ℝ) :
    prefixMask z lo hi (Fin.last m) = sequenceMask z lo hi (Fin.last m) := by
  simp [prefixMask, sequenceMask, prefixAccepted, sequenceAccepted, prefixAverage_last]

structure AllRejected {n : ℕ} (z : Fin n → ℝ) (lo hi : ℝ) : Prop where
  token_zero : ∀ i, tokenMask z lo hi i = 0
  prefix_zero : ∀ i, prefixMask z lo hi i = 0
  sequence_zero : ∀ i, sequenceMask z lo hi i = 0
  token_count : rejectionCount (tokenMask z lo hi) = n
  prefix_count : rejectionCount (prefixMask z lo hi) = n
  sequence_count : rejectionCount (sequenceMask z lo hi) = n

theorem sustained_violation_complete {m : ℕ} (z : Fin (m + 1) → ℝ) (lo hi : ℝ)
    (hviolate : (∀ i, hi < z i) ∨ (∀ i, z i < lo)) : AllRejected z lo hi := by
  have ht : ∀ i, tokenMask z lo hi i = 0 := by
    intro i
    apply (binaryMask_zero _).mpr
    rcases hviolate with h | h
    · exact fun ha => (not_le_of_gt (h i)) ha.2
    · exact fun ha => (not_le_of_gt (h i)) ha.1
  have hp : ∀ i, prefixMask z lo hi i = 0 := by
    intro i
    apply (binaryMask_zero _).mpr
    rcases hviolate with h | h
    · exact fun ha => (not_le_of_gt (prefix_strict_upper z hi i (fun j _ => h j))) ha.2
    · exact fun ha => (not_le_of_gt (prefix_strict_lower z lo i (fun j _ => h j))) ha.1
  have hs : ∀ i, sequenceMask z lo hi i = 0 := by
    intro i
    have hlast := hp (Fin.last m)
    rw [final_prefix_eq_sequence] at hlast
    exact hlast
  exact ⟨ht, hp, hs, rejectionCount_all _ ht, rejectionCount_all _ hp, rejectionCount_all _ hs⟩

/-- All three conditional cases use the same binary masks and active-index convention. -/
theorem prefix_tighter_complete {m : ℕ} (z : Fin (m + 1) → ℝ) (lo hi ε : ℝ)
    (hlo : lo < 0) (hhi : 0 < hi) (hε : 0 ≤ ε) (hεlo : ε ≤ -lo) (hεhi : ε ≤ hi) :
    ((¬tokenAccepted z lo hi 0) → (∀ i, i ≠ 0 → |z i| ≤ ε) → EarlyConclusion z lo hi ε) ∧
    (∀ k, (¬tokenAccepted z lo hi k) → (∀ i, i ≠ k → |z i| ≤ ε) → LateConclusion z lo hi k) ∧
    (((∀ i, hi < z i) ∨ (∀ i, z i < lo)) → AllRejected z lo hi) := by
  exact ⟨early_deviation_complete z lo hi ε hlo hhi hε hεlo hεhi,
    fun k => late_deviation_complete z lo hi ε k hεlo hεhi,
    sustained_violation_complete z lo hi⟩

structure SpikeSeparation {n : ℕ} (k : Fin n) (x lo hi : ℝ) : Prop where
  token_pattern : ∀ i, tokenMask (singleSpike k x) lo hi i = 0 ↔ i = k
  token_count : rejectionCount (tokenMask (singleSpike k x) lo hi) = 1
  prefix_rejects : prefixMask (singleSpike k x) lo hi k = 0
  sequence_accepts : ∀ i, sequenceMask (singleSpike k x) lo hi i = 1

theorem single_spike_separation_of_bounds {n : ℕ} (k : Fin n) (x lo hi : ℝ)
    (hlo : lo ≤ 0) (hhi : 0 ≤ hi) (hx : ¬intervalAccepted lo hi x)
    (hp : ¬prefixAccepted (singleSpike k x) lo hi k)
    (hs : sequenceAccepted (singleSpike k x) lo hi) : SpikeSeparation k x lo hi := by
  have hk : ¬tokenAccepted (singleSpike k x) lo hi k := by
    simpa [tokenAccepted, singleSpike] using hx
  have hnoise : ∀ i, i ≠ k → |singleSpike k x i| ≤ (0 : ℝ) := by
    intro i hi
    simp [singleSpike, hi]
  have ht := token_single_deviation (singleSpike k x) lo hi 0 k (by linarith) hhi hk hnoise
  exact ⟨ht.1, ht.2, (binaryMask_zero _).mpr hp, fun _ => (binaryMask_one _).mpr hs⟩

/-- An explicit positive spike works at every position strictly before the last. -/
theorem upper_midpoint_spike_separation {n : ℕ} (k : Fin n) (hk : k.val + 1 < n)
    (lo hi : ℝ) (hlo : lo ≤ 0) (hhi : 0 < hi) :
    let x := hi * (((k.val + 1 : ℕ) : ℝ) + (n : ℝ)) / 2
    SpikeSeparation k x lo hi := by
  let d : ℝ := ((k.val + 1 : ℕ) : ℝ)
  let N : ℝ := (n : ℝ)
  let x := hi * (d + N) / 2
  change SpikeSeparation k x lo hi
  have hd : 0 < d := by dsimp [d]; positivity
  have hd1 : 1 ≤ d := by dsimp [d]; exact_mod_cast Nat.succ_le_succ (Nat.zero_le k.val)
  have hdN : d < N := by dsimp [d, N]; exact_mod_cast hk
  have hN : 0 < N := lt_trans hd hdN
  have hx : 0 < x := by dsimp [x]; positivity
  have hpx : hi * d < x := by dsimp [x]; nlinarith
  have htx : hi < x := by nlinarith
  have hp : hi < prefixAverage (singleSpike k x) k := by
    rw [(single_spike_prefix_formulas k x).1]
    exact (lt_div_iff₀ hd).mpr hpx
  have hs : sequenceAccepted (singleSpike k x) lo hi := by
    change lo ≤ (∑ i, singleSpike k x i) / (n : ℝ) ∧
      (∑ i, singleSpike k x i) / (n : ℝ) ≤ hi
    rw [(single_spike_prefix_formulas k x).2]
    refine ⟨le_trans hlo (le_of_lt (div_pos hx hN)), ?_⟩
    apply (div_le_iff₀ hN).mpr
    dsimp [x]
    nlinarith
  exact single_spike_separation_of_bounds k x lo hi hlo (le_of_lt hhi)
    (fun ha => (not_le_of_gt htx) ha.2) (fun ha => (not_le_of_gt hp) ha.2) hs

/-- The lower-threshold case is proved with an explicit negative spike. -/
theorem lower_midpoint_spike_separation {n : ℕ} (k : Fin n) (hk : k.val + 1 < n)
    (lo hi : ℝ) (hlo : lo < 0) (hhi : 0 ≤ hi) :
    let x := lo * (((k.val + 1 : ℕ) : ℝ) + (n : ℝ)) / 2
    SpikeSeparation k x lo hi := by
  let d : ℝ := ((k.val + 1 : ℕ) : ℝ)
  let N : ℝ := (n : ℝ)
  let x := lo * (d + N) / 2
  change SpikeSeparation k x lo hi
  have hd : 0 < d := by dsimp [d]; positivity
  have hd1 : 1 ≤ d := by dsimp [d]; exact_mod_cast Nat.succ_le_succ (Nat.zero_le k.val)
  have hdN : d < N := by dsimp [d, N]; exact_mod_cast hk
  have hN : 0 < N := lt_trans hd hdN
  have hx : x < 0 := by
    dsimp [x]
    exact div_neg_of_neg_of_pos (mul_neg_of_neg_of_pos hlo (by linarith)) (by norm_num)
  have hpx : x < lo * d := by dsimp [x]; nlinarith
  have htx : x < lo := by nlinarith
  have hp : prefixAverage (singleSpike k x) k < lo := by
    rw [(single_spike_prefix_formulas k x).1]
    exact (div_lt_iff₀ hd).mpr hpx
  have hs : sequenceAccepted (singleSpike k x) lo hi := by
    change lo ≤ (∑ i, singleSpike k x i) / (n : ℝ) ∧
      (∑ i, singleSpike k x i) / (n : ℝ) ≤ hi
    rw [(single_spike_prefix_formulas k x).2]
    refine ⟨?_, le_trans (le_of_lt (div_neg_of_neg_of_pos hx hN)) hhi⟩
    apply (le_div_iff₀ hN).mpr
    dsimp [x]
    nlinarith
  exact single_spike_separation_of_bounds k x lo hi (le_of_lt hlo) hhi
    (fun ha => (not_le_of_gt htx) ha.1) (fun ha => (not_le_of_gt hp) ha.1) hs

end REINFORCEProMax.PrefixMasking
