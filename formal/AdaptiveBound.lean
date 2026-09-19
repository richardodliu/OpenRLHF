import TreePinsker
import PromptMixture

/-!
The full Adaptive remainder bound on the normalized autoregressive tree.
The continuation starts at an arbitrary history. Bounds on local divergences
are needed only before the chosen horizon; no token independence is used.
The performance-error recursion is proved, rather than supplied as a premise.
-/
namespace REINFORCEProMax.AutoregressiveTree

open scoped BigOperators
variable {A : Type*} [Fintype A]

/-- Re-root the same policy at an observed history. -/
def continuationPolicy (p : Policy A) (h : List A) : Policy A where
  prob y a := p.prob (h ++ y) a
  nonneg y a := p.nonneg (h ++ y) a
  mass y := p.mass (h ++ y)

theorem value_continuation (p : Policy A) (R : List A → ℝ)
    (h : List A) (n : ℕ) (y : List A) :
    value (continuationPolicy p h) (fun z => R (h ++ z)) n y =
      value p R n (h ++ y) := by
  induction n generalizing y with
  | zero => rfl
  | succ n ih =>
      simp only [value, ih]
      simp only [continuationPolicy, List.append_assoc]

theorem chainKL_continuation (p q : Policy A) (h : List A) (n : ℕ) (y : List A) :
    chainKL (continuationPolicy p h) (continuationPolicy q h) n y =
      chainKL p q n (h ++ y) := by
  induction n generalizing y with
  | zero => rfl
  | succ n ih =>
      simp only [chainKL, ih]
      simp only [localReverseKL, tokenRatio, continuationPolicy, List.append_assoc]

noncomputable def futureTV (p q : Policy A) (n : ℕ) (h : List A) : ℝ :=
  acceptedPrefixTV (continuationPolicy p h) (continuationPolicy q h) (fun _ => 1) n

theorem prefixTV_le_one (p q : Policy A) (n : ℕ) :
    acceptedPrefixTV p q (fun _ => 1) n ≤ 1 := by
  have hb := pathSum_mono
    (fun y => |pathMass q [] y - pathMass p [] y|)
    (fun y => pathMass q [] y + pathMass p [] y)
    (fun y => (abs_sub _ _).trans (by
      rw [abs_of_nonneg (pathMass_nonneg q [] y), abs_of_nonneg (pathMass_nonneg p [] y)]))
    n []
  rw [pathSum_add, pathMass_normalized, pathMass_normalized] at hb
  simp only [acceptedPrefixTV, one_mul]
  linarith

theorem value_le_of_terminal_length (p : Policy A) (R : List A → ℝ)
    (B : ℝ) (n : ℕ) (h : List A)
    (hR : ∀ y, y.length = h.length + n → R y ≤ B) :
    value p R n h ≤ B := by
  induction n generalizing h with
  | zero => exact hR h (by omega)
  | succ n ih =>
      calc
        value p R (n + 1) h ≤ ∑ a, p.prob h a * B := by
          apply Finset.sum_le_sum
          intro a _
          apply mul_le_mul_of_nonneg_left _ (p.nonneg h a)
          apply ih
          intro y hy
          apply hR y
          simp only [List.length_append, List.length_singleton] at hy
          omega
        _ = B := by rw [← Finset.sum_mul, p.mass, one_mul]

theorem chainKL_le_within (p q : Policy A) (δ : ℝ) (N : ℕ)
    (hKL : ∀ h, h.length < N → localReverseKL p q h ≤ δ)
    (n : ℕ) (h : List A) (hh : h.length + n ≤ N) :
    chainKL p q n h ≤ (n : ℝ) * δ := by
  induction n generalizing h with
  | zero => simp [chainKL]
  | succ n ih =>
      have hb : (∑ a, p.prob h a * chainKL p q n (h ++ [a])) ≤ (n : ℝ) * δ := by
        calc
          _ ≤ ∑ a, p.prob h a * ((n : ℝ) * δ) := by
            apply Finset.sum_le_sum
            intro a _
            exact mul_le_mul_of_nonneg_left
              (ih _ (by simp only [List.length_append, List.length_singleton]; omega))
              (p.nonneg h a)
          _ = _ := by rw [← Finset.sum_mul, p.mass, one_mul]
      rw [chainKL]
      push_cast
      linarith [hKL h (by omega)]

theorem futureTV_le_linear (p q : Policy A) (ε : ℝ) (N : ℕ)
    (hTV : ∀ h, h.length < N → localTV p q h ≤ ε)
    (n : ℕ) (h : List A) (hh : h.length + n ≤ N) :
    futureTV p q n h ≤ (n : ℝ) * ε := by
  have hb := sequenceTV_le_expected_tokenTV (continuationPolicy p h) (continuationPolicy q h) n
  change futureTV p q n h ≤ _ at hb
  calc
    futureTV p q n h ≤ _ := hb
    _ ≤ ∑ k ∈ Finset.range n, ε := by
      apply Finset.sum_le_sum
      intro k hk
      have heq : localTV (continuationPolicy p h) (continuationPolicy q h) =
          (fun y => localTV p q (h ++ y)) := rfl
      rw [heq, value_continuation, List.append_nil]
      exact value_le_of_terminal_length p (localTV p q) ε k h
        (fun y hy => hTV y (by have := Finset.mem_range.mp hk; omega))
    _ = (n : ℝ) * ε := by simp

theorem futureTV_le_pinsker (p q : Policy A) (δ : ℝ) (N : ℕ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (hKL : ∀ h, h.length < N → localReverseKL p q h ≤ δ)
    (n : ℕ) (h : List A) (hh : h.length + n ≤ N) :
    futureTV p q n h ≤ Real.sqrt ((n : ℝ) * δ / 2) := by
  have hb := prefixTV_le_sqrt_chainKL (continuationPolicy p h) (continuationPolicy q h)
    (fun y a => hs (h ++ y) a) n
  rw [chainKL_continuation, List.append_nil] at hb
  exact hb.trans (Real.sqrt_le_sqrt (div_le_div_of_nonneg_right
    (chainKL_le_within p q δ N hKL n h hh) (by norm_num)))

noncomputable def adaptiveFutureCap (ε δ : ℝ) (n : ℕ) : ℝ :=
  min 1 (min ((n : ℝ) * ε) (Real.sqrt ((n : ℝ) * δ / 2)))

theorem futureTV_le_adaptive (p q : Policy A) (ε δ : ℝ) (N : ℕ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (hTV : ∀ h, h.length < N → localTV p q h ≤ ε)
    (hKL : ∀ h, h.length < N → localReverseKL p q h ≤ δ)
    (n : ℕ) (h : List A) (hh : h.length + n ≤ N) :
    futureTV p q n h ≤ adaptiveFutureCap ε δ n :=
  le_min (prefixTV_le_one _ _ _) (le_min (futureTV_le_linear p q ε N hTV n h hh)
    (futureTV_le_pinsker p q δ N hs hKL n h hh))

theorem value_difference_le_prefixTV (p q : Policy A) (R : List A → ℝ)
    (B : ℝ) (hR : ∀ y, |R y| ≤ B) (n : ℕ) :
    |value q R n [] - value p R n []| ≤
      2 * B * acceptedPrefixTV p q (fun _ => 1) n := by
  have hp := pathSum_mass_value p R n []
  have hq := pathSum_mass_value q R n []
  simp only [pathMass, one_mul, pathSum_eq_sum_ofFn, List.nil_append] at hp hq
  rw [← hp, ← hq, ← Finset.sum_sub_distrib]
  have hb := abs_signed_sum_le
    (fun y : Fin n → A => pathMass q [] (List.ofFn y) - pathMass p [] (List.ofFn y))
    (fun y => R (List.ofFn y)) B (fun y => hR _)
  simp only [sub_mul] at hb
  simpa only [acceptedPrefixTV, one_mul, pathSum_eq_sum_ofFn, List.nil_append] using
    (show _ ≤ 2 * B * ((1 / 2 : ℝ) *
      ∑ y : Fin n → A, |pathMass q [] (List.ofFn y) - pathMass p [] (List.ofFn y)|) by
      nlinarith only [hb])

theorem value_difference_le_futureTV (p q : Policy A) (R : List A → ℝ)
    (B : ℝ) (hR : ∀ y, |R y| ≤ B) (n : ℕ) (h : List A) :
    |value q R n h - value p R n h| ≤ 2 * B * futureTV p q n h := by
  have hb := value_difference_le_prefixTV (continuationPolicy p h) (continuationPolicy q h)
    (fun y => R (h ++ y)) B (fun y => hR _) n
  simpa only [value_continuation, List.append_nil, futureTV] using hb

/-- The exact local interaction, using the rollout distribution for the
outer recursion and the current policy only inside the future reward. -/
noncomputable def futureInteraction (p q : Policy A) (R : List A → ℝ)
    (n : ℕ) (h : List A) : ℝ :=
  ∑ a, (q.prob h a - p.prob h a) *
    (value q R n (h ++ [a]) - value p R n (h ++ [a]))

theorem surrogate_error_future_identity (p q : Policy A) (R : List A → ℝ)
    (n : ℕ) (h : List A) :
    value q R n h - value p R n h - surrogate p q R n h =
      additive p (futureInteraction p q R) n h := by
  induction n generalizing h with
  | zero => simp [value, surrogate, additive]
  | succ n ih =>
      change _ = futureInteraction p q R n h +
        ∑ a, p.prob h a * additive p (futureInteraction p q R) n (h ++ [a])
      simp_rw [← ih]
      simp only [value, surrogate, additive, gain, futureInteraction,
        mul_sub, sub_mul, Finset.sum_sub_distrib]
      ring

theorem futureInteraction_abs_le (p q : Policy A) (R : List A → ℝ)
    (B ε δ : ℝ) (hB : 0 ≤ B) (hR : ∀ y, |R y| ≤ B) (N : ℕ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (hTV : ∀ h, h.length < N → localTV p q h ≤ ε)
    (hKL : ∀ h, h.length < N → localReverseKL p q h ≤ δ)
    (n : ℕ) (h : List A) (hh : h.length + (n + 1) ≤ N) :
    |futureInteraction p q R n h| ≤
      4 * B * (localTV p q h * adaptiveFutureCap ε δ n) := by
  have hf : ∀ a, |value q R n (h ++ [a]) - value p R n (h ++ [a])| ≤
      2 * B * adaptiveFutureCap ε δ n := by
    intro a
    exact (value_difference_le_futureTV p q R B hR n _).trans
      (mul_le_mul_of_nonneg_left
        (futureTV_le_adaptive p q ε δ N hs hTV hKL n _
          (by simp only [List.length_append, List.length_singleton]; omega))
        (by positivity))
  have hb := abs_signed_sum_le (fun a => q.prob h a - p.prob h a)
    (fun a => value q R n (h ++ [a]) - value p R n (h ++ [a]))
    (2 * B * adaptiveFutureCap ε δ n) hf
  unfold futureInteraction localTV
  nlinarith only [hb]

/-- Propagate local absolute bounds only over the specified finite horizon. -/
theorem additive_abs_le_within (p : Policy A) (f F : ℕ → List A → ℝ) (N : ℕ)
    (hf : ∀ k h, h.length + (k + 1) ≤ N → |f k h| ≤ F k h)
    (n : ℕ) (h : List A) (hh : h.length + n ≤ N) :
    |additive p f n h| ≤ additive p F n h := by
  induction n generalizing h with
  | zero => simp [additive]
  | succ n ih =>
      have hsum := abs_weighted_sum_le_sum (p.prob h)
        (fun a => additive p f n (h ++ [a]))
        (fun a => additive p F n (h ++ [a])) (p.nonneg h)
        (fun a => ih _ (by simp only [List.length_append, List.length_singleton]; omega))
      exact (abs_add _ _).trans (add_le_add (hf n h hh) hsum)

theorem additive_const_mul (p : Policy A) (c : ℝ) (f : ℕ → List A → ℝ)
    (n : ℕ) (h : List A) :
    additive p (fun k y => c * f k y) n h = c * additive p f n h := by
  induction n generalizing h with
  | zero => simp [additive]
  | succ n ih =>
      simp only [additive, ih, mul_add, Finset.mul_sum]
      congr 1
      apply Finset.sum_congr rfl
      intro a _
      ring

/-- The full Adaptive bound, derived from the exact tree recursion and all
three continuation-TV bounds. Local assumptions stop at the horizon N. -/
theorem surrogate_error_adaptive (p q : Policy A) (R : List A → ℝ)
    (B ε δ : ℝ) (hB : 0 ≤ B) (hR : ∀ y, |R y| ≤ B) (N : ℕ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (hTV : ∀ h, h.length < N → localTV p q h ≤ ε)
    (hKL : ∀ h, h.length < N → localReverseKL p q h ≤ δ)
    (n : ℕ) (h : List A) (hh : h.length + n ≤ N) :
    |value q R n h - value p R n h - surrogate p q R n h| ≤
      4 * B * additive p (fun k y => localTV p q y * adaptiveFutureCap ε δ k) n h := by
  rw [surrogate_error_future_identity]
  have hb := additive_abs_le_within p (futureInteraction p q R)
    (fun k y => 4 * B * (localTV p q y * adaptiveFutureCap ε δ k)) N
    (fun k y hy => futureInteraction_abs_le p q R B ε δ hB hR N hs hTV hKL k y hy)
    n h hh
  rwa [additive_const_mul] at hb

/-- Rollout-expected local TV at each position, weighted by the remaining
future horizon. Index k is paper position t-1. -/
noncomputable def adaptiveCost (p q : Policy A) (ε δ : ℝ) (N : ℕ) : ℝ :=
  ∑ k ∈ Finset.range N, value p (localTV p q) k [] * adaptiveFutureCap ε δ (N - 1 - k)

theorem adaptiveCost_eq_additive (p q : Policy A) (ε δ : ℝ) (N : ℕ) :
    adaptiveCost p q ε δ N =
      additive p (fun k y => localTV p q y * adaptiveFutureCap ε δ k) N [] := by
  rw [additive_eq_prefix_sum]
  unfold adaptiveCost
  apply Finset.sum_congr rfl
  intro k _
  simpa only [mul_comm] using
    (value_const_mul p (adaptiveFutureCap ε δ (N - 1 - k)) (localTV p q) k []).symm

theorem surrogate_error_adaptive_root (p q : Policy A) (R : List A → ℝ)
    (B ε δ : ℝ) (hB : 0 ≤ B) (hR : ∀ y, |R y| ≤ B) (N : ℕ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (hTV : ∀ h, h.length < N → localTV p q h ≤ ε)
    (hKL : ∀ h, h.length < N → localReverseKL p q h ≤ δ) :
    |value q R N [] - value p R N [] - surrogate p q R N []| ≤
      4 * B * adaptiveCost p q ε δ N := by
  rw [adaptiveCost_eq_additive]
  exact surrogate_error_adaptive p q R B ε δ hB hR N hs hTV hKL N [] (by simp)

/-- Bound the actual reward-weighted trajectory remainder, with the same
ratios as the exact change-of-measure identity in TreeSurrogate. -/
theorem trajectory_remainder_adaptive (p q : Policy A) (R : List A → ℝ)
    (B ε δ : ℝ) (hB : 0 ≤ B) (hR : ∀ y, |R y| ≤ B) (N : ℕ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (hTV : ∀ h, h.length < N → localTV p q h ≤ ε)
    (hKL : ∀ h, h.length < N → localReverseKL p q h ≤ δ) :
    |value p (fun y => R y * remainderRhs (ratioTrace p q y)) N []| ≤
      4 * B * adaptiveCost p q ε δ N := by
  have hid := trajectory_surrogate_remainder_identity p q R (fun h a => (hs h a).mp) N
  rw [trajectory_surrogate_eq p q R (fun h a => (hs h a).mp)] at hid
  have hb := surrogate_error_adaptive_root p q R B ε δ hB hR N hs hTV hKL
  have heq : value q R N [] - value p R N [] - surrogate p q R N [] =
      -value p (fun y => R y * remainderRhs (ratioTrace p q y)) N [] := by linarith
  rwa [heq, abs_neg] at hb

theorem performance_lower_bound_adaptive (p q : Policy A) (R : List A → ℝ)
    (B ε δ : ℝ) (hB : 0 ≤ B) (hR : ∀ y, |R y| ≤ B) (N : ℕ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (hTV : ∀ h, h.length < N → localTV p q h ≤ ε)
    (hKL : ∀ h, h.length < N → localReverseKL p q h ≤ δ) :
    surrogate p q R N [] - 4 * B * adaptiveCost p q ε δ N ≤
      value q R N [] - value p R N [] := by
  have hb := (abs_le.mp (surrogate_error_adaptive_root p q R B ε δ hB hR N hs hTV hKL)).1
  linarith

section PromptFamily
variable {X : Type*} [Fintype X]

/-- The prompt mixture has exactly the paper's expected per-position TV. -/
theorem family_adaptive_cost (μ : X → ℝ) (p q : X → Policy A) (ε δ : ℝ) (N : ℕ) :
    (∑ x, μ x * adaptiveCost (p x) (q x) ε δ N) =
      ∑ k ∈ Finset.range N,
        (∑ x, μ x * value (p x) (localTV (p x) (q x)) k []) *
          adaptiveFutureCap ε δ (N - 1 - k) := by
  simp only [adaptiveCost, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro x _
  ring

theorem family_remainder_adaptive
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hμ : ∀ x, 0 ≤ μ x) (B ε δ : ℝ) (hB : 0 ≤ B)
    (hR : ∀ x y, |R x y| ≤ B) (N : ℕ)
    (hs : ∀ x h a, (p x).prob h a = 0 ↔ (q x).prob h a = 0)
    (hTV : ∀ x h, h.length < N → localTV (p x) (q x) h ≤ ε)
    (hKL : ∀ x h, h.length < N → localReverseKL (p x) (q x) h ≤ δ) :
    |familyRemainder μ p q R N| ≤
      4 * B * ∑ k ∈ Finset.range N,
        (∑ x, μ x * value (p x) (localTV (p x) (q x)) k []) *
          adaptiveFutureCap ε δ (N - 1 - k) := by
  have hb := abs_weighted_sum_le_sum μ
    (fun x => value (p x) (fun y => R x y * remainderRhs (ratioTrace (p x) (q x) y)) N [])
    (fun x => 4 * B * adaptiveCost (p x) (q x) ε δ N) hμ
    (fun x => trajectory_remainder_adaptive (p x) (q x) (R x) B ε δ hB (hR x) N
      (hs x) (hTV x) (hKL x))
  change |familyRemainder μ p q R N| ≤ _ at hb
  have heq : (∑ x, μ x * (4 * B * adaptiveCost (p x) (q x) ε δ N)) =
      4 * B * ∑ x, μ x * adaptiveCost (p x) (q x) ε δ N := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro x _
    ring
  rwa [heq, family_adaptive_cost] at hb

theorem family_performance_lower_bound_adaptive
    (μ : X → ℝ) (p q : X → Policy A) (R : X → List A → ℝ)
    (hμ : ∀ x, 0 ≤ μ x) (B ε δ : ℝ) (hB : 0 ≤ B)
    (hR : ∀ x y, |R x y| ≤ B) (N : ℕ)
    (hs : ∀ x h a, (p x).prob h a = 0 ↔ (q x).prob h a = 0)
    (hTV : ∀ x h, h.length < N → localTV (p x) (q x) h ≤ ε)
    (hKL : ∀ x h, h.length < N → localReverseKL (p x) (q x) h ≤ δ) :
    familySurrogate μ p q R N -
      4 * B * ∑ k ∈ Finset.range N,
        (∑ x, μ x * value (p x) (localTV (p x) (q x)) k []) *
          adaptiveFutureCap ε δ (N - 1 - k) ≤
      familyValue μ q R N - familyValue μ p R N := by
  rw [family_surrogate_remainder_identity μ p q R (fun x h a => (hs x h a).mp)]
  have hb := (abs_le.mp (family_remainder_adaptive μ p q R hμ B ε δ hB hR N hs hTV hKL)).2
  linarith
end PromptFamily

end REINFORCEProMax.AutoregressiveTree
