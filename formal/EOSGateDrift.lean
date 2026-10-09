import CachedGateDrift
import EOSCertificateBridge

/-! Action-mask semantics for the cached gate and accepted-prefix drift.
The effective gate includes the current action mask. Padding cannot count
as an accepted training action, regardless of its raw gate value. -/
namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators
variable {A : Type*} [Fintype A] [DecidableEq A]

theorem logRatioTrace_eq_token_sum (p q : Policy A) {T : ℕ} (y : Fin T → A) :
    logRatioTrace p q (List.ofFn y) =
      ∑ t, Real.log (responseTokenRatio p q y t) := by
  induction T with
  | zero => simp [List.ofFn_zero, logRatioTrace, ratioTrace]
  | succ T ih =>
    have hy := List.ofFn_succ' y
    rw [List.concat_eq_append] at hy
    rw [hy, logRatioTrace_snoc, ih, Fin.sum_univ_castSucc]
    congr 1
    · apply Finset.sum_congr rfl
      intro t _
      unfold responseTokenRatio
      rw [hy, List.take_append_of_le_length (by simp)]
      rfl
    · unfold responseTokenRatio
      rw [hy, List.take_append_of_le_length (by simp), List.take_of_length_le (by simp)]

theorem eos_prefix_masks_one (eos : A) (y : List A) (hy : eos ∉ y.dropLast)
    (t : Fin y.length) : eosActiveMask eos (y.take t.val) = 1 := by
  have ht : eos ∉ y.take t.val := by
    intro hm
    apply hy
    rw [List.dropLast_eq_take]
    exact List.take_subset_take_left y (by omega) hm
  simp [eosActiveMask, ht]

noncomputable def eosPrefixAverage (p old : Policy A) (eos : A) (y : List A) : ℝ :=
  (∑ t : Fin y.length, eosActiveMask eos (y.take t.val) *
    Real.log (tokenRatio p old (y.take t.val) (y.get t))) / max (eosActiveLength eos y) 1

/-- At an active action, all earlier actions in a single EOS-terminated
response are active. Thus masked cumsum/count is the unpadded tree average. -/
theorem eos_prefix_average_active (p old : Policy A) (eos : A) (y : List A)
    (hlen : 0 < y.length) (hy : eos ∉ y.dropLast) :
    eosPrefixAverage p old eos y = logRatioTrace p old y / (y.length : ℝ) := by
  have hm := eos_prefix_masks_one eos y hy
  have hcount : eosActiveLength eos y = (y.length : ℝ) := by
    simp [eosActiveLength, hm]
  have hsum := logRatioTrace_eq_token_sum p old y.get
  simp only [responseTokenRatio, List.ofFn_get] at hsum
  unfold eosPrefixAverage
  simp only [hm, one_mul, hcount]
  rw [max_eq_left (by exact_mod_cast hlen), ← hsum]

noncomputable def eosImplementationGate (p old : Policy A) (eos : A)
    (τp τn : ℝ) (y : List A) : ℝ :=
  if -τn ≤ eosPrefixAverage p old eos y ∧ eosPrefixAverage p old eos y ≤ τp
    then 1 else 0

noncomputable def eosEffectiveGate (p old : Policy A) (eos : A)
    (τp τn : ℝ) (y : List A) : ℝ :=
  eosActiveMask eos y.dropLast * prefixLogGate p old τp τn y

theorem eos_gate_exp_comparison (p old : Policy A) (eos : A)
    (τp τn : ℝ) (y : List A) :
    (if Real.exp (-τn) ≤ Real.exp (eosPrefixAverage p old eos y) ∧
      Real.exp (eosPrefixAverage p old eos y) ≤ Real.exp τp then (1 : ℝ) else 0) =
      eosImplementationGate p old eos τp τn y := by
  simp [eosImplementationGate]

/-- Only the effective gates are equated after EOS: raw padded-depth gates
need not equal the implementation's gate, but neither contributes to loss. -/
theorem eos_effective_gate_implementation (p old : Policy A) (eos : A)
    (τp τn : ℝ) (y : List A) (hlen : 0 < y.length) :
    eosActiveMask eos y.dropLast * eosImplementationGate p old eos τp τn y =
      eosEffectiveGate p old eos τp τn y := by
  by_cases hy : eos ∈ y.dropLast
  · simp [eosEffectiveGate, eosActiveMask, hy]
  · simp only [eosImplementationGate, eos_prefix_average_active p old eos y hlen hy,
      eosEffectiveGate, prefixLogGate]

theorem eos_effective_gate_nonneg (p old : Policy A) (eos : A)
    (τp τn : ℝ) (y : List A) : 0 ≤ eosEffectiveGate p old eos τp τn y :=
  mul_nonneg (eos_mask_nonneg _ _) (prefixLogGate_nonneg _ _ _ _ _)

theorem eos_effective_gate_le_one (p old : Policy A) (eos : A)
    (τp τn : ℝ) (y : List A) : eosEffectiveGate p old eos τp τn y ≤ 1 := by
  by_cases hy : eos ∈ y.dropLast
  · simp [eosEffectiveGate, eosActiveMask, hy]
  · simpa [eosEffectiveGate, eosActiveMask, hy] using prefixLogGate_le_one p old τp τn y

theorem eos_effective_gate_zero_after_eos (p old : Policy A) (eos : A)
    (τp τn : ℝ) (y : List A) (hy : eos ∈ y.dropLast) :
    eosEffectiveGate p old eos τp τn y = 0 := by
  simp [eosEffectiveGate, eosActiveMask, hy]

theorem eos_gate_drift_identity (p old q : Policy A) (eos : A) (τp τn : ℝ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) :
    acceptedPrefixTV p q (eosEffectiveGate p old eos τp τn) n =
      (1 / 2 : ℝ) * value p (fun y =>
        eosEffectiveGate p old eos τp τn y * |(ratioTrace p q y).prod - 1|) n [] :=
  acceptedPrefixTV_identity p q hs _ (eos_effective_gate_nonneg p old eos τp τn) n

theorem eos_gate_drift_of_mgf (p old q : Policy A) (eos : A) (τp τn σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (n : ℕ)
    (hmgf : ∀ h, h.length < n → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2)) :
    acceptedPrefixTV p q (eosEffectiveGate p old eos τp τn) n ≤
      (1 / 2 : ℝ) * Real.sqrt (value p (eosEffectiveGate p old eos τp τn) n []) *
        Real.sqrt (Real.exp (2 * σ ^ 2 * (n : ℝ)) - 1) :=
  acceptedPrefixTV_of_mgf p q σ hs n hmgf _
    (eos_effective_gate_nonneg p old eos τp τn) (eos_effective_gate_le_one p old eos τp τn)

theorem eos_effective_horizon_of_mgf (p old q : Policy A) (eos : A) (τp τn σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (N : ℕ)
    (hmgf : ∀ h, h.length < N → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2)) :
    (∑ t : Fin N, acceptedPrefixTV p q (eosEffectiveGate p old eos τp τn) (t.val + 1)) ≤
      (1 / 2 : ℝ) *
        Real.sqrt (∑ t : Fin N, value p (eosEffectiveGate p old eos τp τn) (t.val + 1) []) *
        Real.sqrt (∑ t : Fin N, (Real.exp (2 * σ ^ 2 * ((t.val + 1 : ℕ) : ℝ)) - 1)) :=
  effective_horizon_of_mgf p q σ hs N hmgf
    (fun _ => eosEffectiveGate p old eos τp τn)
    (fun _ => eos_effective_gate_nonneg p old eos τp τn)
    (fun _ => eos_effective_gate_le_one p old eos τp τn)

end REINFORCEProMax.AutoregressiveTree
