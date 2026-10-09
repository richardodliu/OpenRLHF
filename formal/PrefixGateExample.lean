import PrefixGateMoments

/-! A normalized three-action family with exactly half of rollout mass
accepted at every depth up to a prescribed horizon. Rejected branches do
not reenter, and retained token-weight second moments equal one half. -/
namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators

noncomputable def selectiveWeights (K : ℝ) (swap : Bool) : Fin 3 → ℝ :=
  ![1/2, if swap then K / (2*(K+1)) else 1 / (2*(K+1)),
    if swap then 1 / (2*(K+1)) else K / (2*(K+1))]

noncomputable def selectivePolicy (K : ℝ) (hK : 0 < K) (swap : Bool) : Policy (Fin 3) where
  prob h a := if h = [] then selectiveWeights K swap a else 1/3
  nonneg := by
    intro h a
    split_ifs
    · fin_cases a <;> cases swap <;> simp [selectiveWeights] <;> positivity
    · norm_num
  mass := by
    intro h
    by_cases hh : h = []
    · simp only [hh, if_true]
      cases swap <;> simp [selectiveWeights, Fin.sum_univ_succ] <;> field_simp <;> ring
    · norm_num [hh, Fin.sum_univ_succ]

theorem selective_policy_positive (K : ℝ) (hK : 0 < K) (swap : Bool)
    (h : List (Fin 3)) (a : Fin 3) : 0 < (selectivePolicy K hK swap).prob h a := by
  simp only [selectivePolicy]
  split_ifs
  · fin_cases a <;> cases swap <;> simp [selectiveWeights] <;> positivity
  · norm_num

theorem selective_mutual_support (K : ℝ) (hK : 0 < K) (h : List (Fin 3)) (a : Fin 3) :
    (selectivePolicy K hK false).prob h a = 0 ↔ (selectivePolicy K hK true).prob h a = 0 :=
  iff_of_false (ne_of_gt (selective_policy_positive K hK false h a))
    (ne_of_gt (selective_policy_positive K hK true h a))

theorem selective_root_ratio (K : ℝ) (hK : 0 < K) (a : Fin 3) :
    tokenRatio (selectivePolicy K hK false) (selectivePolicy K hK true) [] a =
      ![1, K, K⁻¹] a := by
  have hK0 := ne_of_gt hK
  have hd : K + 1 ≠ 0 := by positivity
  fin_cases a <;> simp [tokenRatio, selectivePolicy, selectiveWeights] <;> field_simp <;> simp [mul_comm]

theorem selective_later_ratio (K : ℝ) (hK : 0 < K) (h : List (Fin 3))
    (hh : h ≠ []) (a : Fin 3) :
    tokenRatio (selectivePolicy K hK false) (selectivePolicy K hK true) h a = 1 := by
  norm_num [tokenRatio, selectivePolicy, hh]

theorem selective_log_trace (K : ℝ) (hK : 0 < K) (a : Fin 3) (xs : List (Fin 3)) :
    logRatioTrace (selectivePolicy K hK false) (selectivePolicy K hK true) (a :: xs) =
      Real.log (![1, K, K⁻¹] a) := by
  induction xs using List.reverseRecOn with
  | nil => simp [logRatioTrace, ratioTrace, traceStep, selective_root_ratio]
  | append_singleton xs b ih =>
    change logRatioTrace _ _ ((a :: xs) ++ [b]) = _
    rw [logRatioTrace_snoc, selective_later_ratio K hK _ (by simp) b, Real.log_one, add_zero, ih]

theorem selective_gate (K τ : ℝ) (hK : 0 < K) (hτ : 0 ≤ τ)
    (a : Fin 3) (xs : List (Fin 3))
    (hmargin : ((xs.length : ℝ) + 1) * τ < Real.log K) :
    prefixLogGate (selectivePolicy K hK false) (selectivePolicy K hK true)
      τ τ (a :: xs) = if a = 0 then 1 else 0 := by
  have hd : 0 < (xs.length : ℝ) + 1 := by positivity
  have hhigh : τ < Real.log K / ((xs.length : ℝ) + 1) :=
    (lt_div_iff₀ hd).mpr (by simpa [mul_comm] using hmargin)
  unfold prefixLogGate
  rw [selective_log_trace]
  simp only [List.length_cons, Nat.cast_add, Nat.cast_one]
  fin_cases a
  · simp [hτ]
  · simp only [Matrix.cons_val_one, Matrix.cons_val_zero]
    norm_num [not_le.mpr hhigh]
  · simp only [Matrix.cons_val_two, Matrix.cons_val_zero, Real.log_inv, neg_div]
    have hn : ¬ -τ ≤ -(Real.log K / ((xs.length : ℝ) + 1)) := by linarith
    have hn' : ¬ -τ ≤ -Real.log K / ((xs.length : ℝ) + 1) := by simpa only [neg_div] using hn
    norm_num [hn']

noncomputable def neutralBranch (h : List (Fin 3)) : ℝ :=
  if h.head? = some 0 then 1 else 0

theorem neutral_branch_value (p : Policy (Fin 3)) (n : ℕ) (a : Fin 3) (xs : List (Fin 3)) :
    value p neutralBranch n (a :: xs) = if a = 0 then 1 else 0 := by
  induction n generalizing xs with
  | zero => simp [value, neutralBranch]
  | succ n ih =>
    simp only [value, List.cons_append, ih]
    rw [← Finset.sum_mul, p.mass, one_mul]

theorem selective_neutral_mass (K : ℝ) (hK : 0 < K) (n : ℕ) :
    value (selectivePolicy K hK false) neutralBranch (n+1) [] = 1/2 := by
  simp [value, neutral_branch_value, selectivePolicy, selectiveWeights, Fin.sum_univ_succ]

theorem selective_acceptance_mass (K τ : ℝ) (hK : 0 < K) (hτ : 0 ≤ τ) (n T : ℕ)
    (hn : n+1 ≤ T) (hmargin : (T : ℝ) * τ < Real.log K) :
    value (selectivePolicy K hK false)
      (prefixLogGate (selectivePolicy K hK false) (selectivePolicy K hK true) τ τ)
      (n+1) [] = 1/2 := by
  rw [← selective_neutral_mass K hK n]
  rw [value_eq_sum_pathMass, value_eq_sum_pathMass]
  apply Finset.sum_congr rfl
  intro y _
  congr 1
  have hlen : (List.ofFn y).length = n+1 := by simp
  generalize List.ofFn y = h at *
  cases h with
  | nil => simp at hlen
  | cons a xs =>
    have hnat : xs.length + 1 ≤ T := by simp only [List.length_cons] at hlen; omega
    have hb : (xs.length : ℝ) + 1 ≤ T := by exact_mod_cast hnat
    have hm := lt_of_le_of_lt (mul_le_mul_of_nonneg_right hb hτ) hmargin
    simpa [neutralBranch] using selective_gate K τ hK hτ a xs hm

theorem selective_root_moment (K τ : ℝ) (hK : 0 < K) (hτ : 0 ≤ τ)
    (hmargin : τ < Real.log K) :
    localRetainedMoment (selectivePolicy K hK false) (selectivePolicy K hK true) τ τ [] = 1/2 := by
  have hg : ∀ a, prefixLogGate (selectivePolicy K hK false) (selectivePolicy K hK true)
      τ τ [a] = if a = 0 then 1 else 0 := by
    intro a
    exact selective_gate K τ hK hτ a [] (by simpa using hmargin)
  simp only [localRetainedMoment, List.nil_append, hg, selective_root_ratio]
  norm_num [selectivePolicy, selectiveWeights, Fin.sum_univ_succ]

theorem selective_later_moment (K τ : ℝ) (hK : 0 < K) (hτ : 0 ≤ τ)
    (a : Fin 3) (xs : List (Fin 3))
    (hmargin : ((xs.length : ℝ) + 2) * τ < Real.log K) :
    localRetainedMoment (selectivePolicy K hK false) (selectivePolicy K hK true)
      τ τ (a :: xs) = if a = 0 then 1 else 0 := by
  have hg : ∀ b, prefixLogGate (selectivePolicy K hK false) (selectivePolicy K hK true)
      τ τ ((a :: xs) ++ [b]) = if a = 0 then 1 else 0 := by
    intro b
    apply selective_gate K τ hK hτ a (xs ++ [b])
    simpa [List.length_append, Nat.cast_add, add_assoc, show (1 : ℝ) + 1 = 2 by norm_num] using hmargin
  simp only [localRetainedMoment, hg, selective_later_ratio K hK (a :: xs) (by simp),
    one_pow, mul_one, ← Finset.sum_mul, (selectivePolicy K hK false).mass, one_mul]

theorem selective_retained_moment (K τ : ℝ) (hK : 0 < K) (hτ : 0 ≤ τ) (n T : ℕ)
    (hn : n+1 ≤ T) (hmargin : (T : ℝ) * τ < Real.log K) :
    value (selectivePolicy K hK false)
      (localRetainedMoment (selectivePolicy K hK false) (selectivePolicy K hK true) τ τ)
      n [] = 1/2 := by
  cases n with
  | zero =>
    have hT : (1 : ℝ) ≤ T := by exact_mod_cast hn
    have hm : τ < Real.log K := by nlinarith
    simpa [value] using selective_root_moment K τ hK hτ hm
  | succ n =>
    rw [← selective_neutral_mass K hK n, value_eq_sum_pathMass, value_eq_sum_pathMass]
    apply Finset.sum_congr rfl
    intro y _
    congr 1
    have hlen : (List.ofFn y).length = n+1 := by simp
    generalize List.ofFn y = h at *
    cases h with
    | nil => simp at hlen
    | cons a xs =>
      have hnat : xs.length + 2 ≤ T := by simp only [List.length_cons] at hlen; omega
      have hb : (xs.length : ℝ) + 2 ≤ T := by exact_mod_cast hnat
      have hm := lt_of_le_of_lt (mul_le_mul_of_nonneg_right hb hτ) hmargin
      simpa [neutralBranch] using selective_later_moment K τ hK hτ a xs hm

theorem selective_reentry_zero (K τ : ℝ) (hK : 0 < K) (hτ : 0 ≤ τ) (n T : ℕ)
    (hn : n+1 ≤ T) (hmargin : (T : ℝ) * τ < Real.log K) :
    value (selectivePolicy K hK false)
      (fun h => (1 - prefixLogGate (selectivePolicy K hK false) (selectivePolicy K hK true) τ τ h) *
        localRetainedMoment (selectivePolicy K hK false) (selectivePolicy K hK true) τ τ h)
      n [] = 0 := by
  rw [value_eq_sum_pathMass]
  apply Finset.sum_eq_zero
  intro y _
  have hlen : (List.ofFn y).length = n := by simp
  generalize List.ofFn y = h at *
  cases h with
  | nil => simp [prefixLogGate, logRatioTrace, ratioTrace, hτ]
  | cons a xs =>
    have hnat : xs.length + 2 ≤ T := by simp only [List.length_cons] at hlen; omega
    have hb : (xs.length : ℝ) + 2 ≤ T := by exact_mod_cast hnat
    have hm := lt_of_le_of_lt (mul_le_mul_of_nonneg_right hb hτ) hmargin
    have hm' : ((xs.length : ℝ) + 1) * τ < Real.log K := by nlinarith
    rw [selective_gate K τ hK hτ a xs hm', selective_later_moment K τ hK hτ a xs hm]
    split_ifs <;> ring

/-- The same half-retention example also has zero lower-side reentry. -/
theorem selective_lower_reentry_zero (K τ : ℝ) (hK : 0 < K) (hτ : 0 ≤ τ) (n T : ℕ)
    (hn : n+1 ≤ T) (hmargin : (T : ℝ) * τ < Real.log K) :
    value (selectivePolicy K hK false)
      (fun h => if logRatioTrace (selectivePolicy K hK false) (selectivePolicy K hK true) h <
        -((h.length : ℝ) * τ)
        then localRetainedMoment (selectivePolicy K hK false) (selectivePolicy K hK true) τ τ h
        else 0) n [] = 0 := by
  apply le_antisymm
  · exact (lower_reentry_le_all_reentry _ _ τ τ n).trans_eq
      (selective_reentry_zero K τ hK hτ n T hn hmargin)
  · conv_lhs => rw [← value_const (selectivePolicy K hK false) 0 n []]
    apply value_mono
    intro h
    split_ifs
    · apply Finset.sum_nonneg
      intro a _
      exact mul_nonneg (mul_nonneg ((selectivePolicy K hK false).nonneg h a)
        (prefixLogGate_nonneg _ _ τ τ _)) (sq_nonneg _)
    · exact le_rfl

theorem selective_unfiltered_root_moment (K : ℝ) (hK : 0 < K) :
    (∑ a, (selectivePolicy K hK false).prob [] a *
      (tokenRatio (selectivePolicy K hK false) (selectivePolicy K hK true) [] a)^2) =
      (K + K⁻¹) / 2 := by
  have hK0 := ne_of_gt hK
  have hd : K+1 ≠ 0 := by positivity
  simp only [selective_root_ratio]
  simp [selectivePolicy, selectiveWeights, Fin.sum_univ_succ]
  field_simp
  ring

end REINFORCEProMax.AutoregressiveTree
