import PrefixConcentration

/-!
Total variation on the actual finite autoregressive prefix distributions.
The upper bound uses rollout-weighted conditional TVs, not a maximum over
histories. The lower bound records that a complete prefix cannot forget an
earlier discrepancy. Neither statement requires likelihood ratios or support
assumptions; common absorbing EOS transitions are allowed.
-/
namespace REINFORCEProMax.AutoregressiveTree

open scoped BigOperators
variable {A : Type*} [Fintype A]

theorem pathSum_add (f g : List A → ℝ) (n : ℕ) (h : List A) :
    pathSum (fun y => f y + g y) n h = pathSum f n h + pathSum g n h := by
  induction n generalizing h with
  | zero => rfl
  | succ n ih => simp only [pathSum, ih, Finset.sum_add_distrib]

theorem pathSum_const_mul (c : ℝ) (f : List A → ℝ) (n : ℕ) (h : List A) :
    pathSum (fun y => c * f y) n h = c * pathSum f n h := by
  induction n generalizing h with
  | zero => rfl
  | succ n ih => simp only [pathSum, ih, Finset.mul_sum]

theorem pathSum_mono (f g : List A → ℝ) (hfg : ∀ y, f y ≤ g y)
    (n : ℕ) (h : List A) : pathSum f n h ≤ pathSum g n h := by
  induction n generalizing h with
  | zero => exact hfg h
  | succ n ih => exact Finset.sum_le_sum fun a _ => ih _

/-- Expand the last token, so that the outer sum is the actual prefix law. -/
theorem pathSum_succ_right (f : List A → ℝ) (n : ℕ) (h : List A) :
    pathSum f (n + 1) h = pathSum (fun y => ∑ a, f (y ++ [a])) n h := by
  induction n generalizing h with
  | zero => rfl
  | succ n ih =>
      change (∑ a, pathSum f (n + 1) (h ++ [a])) =
        ∑ a, pathSum (fun y => ∑ b, f (y ++ [b])) n (h ++ [a])
      simp only [ih]

/-- Both directions of the L1 estimate for one extension of a fixed prefix.
The upper bound is weighted by that prefix's rollout mass `u`. -/
theorem prefix_row_l1_bounds (p q : Policy A) (h : List A)
    (u v : ℝ) (hu : 0 ≤ u) :
    |v - u| ≤ (∑ a, |v * q.prob h a - u * p.prob h a|) ∧
    (∑ a, |v * q.prob h a - u * p.prob h a|) ≤
      |v - u| + u * (∑ a, |q.prob h a - p.prob h a|) := by
  constructor
  · have hb := Finset.abs_sum_le_sum_abs
      (fun a => v * q.prob h a - u * p.prob h a) Finset.univ
    simpa only [Finset.sum_sub_distrib, ← Finset.mul_sum, p.mass, q.mass, mul_one] using hb
  · calc
      _ ≤ ∑ a, (|v - u| * q.prob h a + u * |q.prob h a - p.prob h a|) := by
        apply Finset.sum_le_sum
        intro a _
        have hid : v * q.prob h a - u * p.prob h a =
            (v - u) * q.prob h a + u * (q.prob h a - p.prob h a) := by ring
        rw [hid]
        simpa only [abs_mul, abs_of_nonneg (q.nonneg h a), abs_of_nonneg hu] using
          abs_add ((v - u) * q.prob h a) (u * (q.prob h a - p.prob h a))
      _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, q.mass, mul_one]

/-- One-step drift of the generated distributions, with the expected local
TV under rollout occupancy. This retains the paper's sharper weighting. -/
theorem prefixTV_step_weighted (p q : Policy A) (n : ℕ) :
    acceptedPrefixTV p q (fun _ => 1) (n + 1) ≤
      acceptedPrefixTV p q (fun _ => 1) n + value p (localTV p q) n [] := by
  have hrow : ∀ h,
      (∑ a, |pathMass q [] (h ++ [a]) - pathMass p [] (h ++ [a])|) ≤
        |pathMass q [] h - pathMass p [] h| +
          pathMass p [] h * (∑ a, |q.prob h a - p.prob h a|) := by
    intro h
    simpa only [pathMass_snoc] using
      (prefix_row_l1_bounds p q h (pathMass p [] h) (pathMass q [] h)
        (pathMass_nonneg p [] h)).2
  unfold acceptedPrefixTV
  simp only [one_mul]
  rw [pathSum_succ_right]
  calc
    _ ≤ (1 / 2 : ℝ) * pathSum (fun h => |pathMass q [] h - pathMass p [] h| +
        pathMass p [] h * (∑ a, |q.prob h a - p.prob h a|)) n [] :=
      mul_le_mul_of_nonneg_left (pathSum_mono _ _ hrow n []) (by norm_num)
    _ = _ := by
      rw [pathSum_add, mul_add]
      congr 1
      have hid : (fun h => pathMass p [] h * (∑ a, |q.prob h a - p.prob h a|)) =
          (fun h => 2 * (pathMass p [] h * localTV p q h)) := by
        funext h; unfold localTV; ring
      rw [hid, pathSum_const_mul, pathSum_mass_value]
      simp [pathMass]

/-- Complete sequence-TV bound on a normalized history tree. The summand at
zero-based `k` is the paper's expected token TV at position `t = k + 1`. -/
theorem sequenceTV_le_expected_tokenTV (p q : Policy A) (n : ℕ) :
    acceptedPrefixTV p q (fun _ => 1) n ≤
      ∑ k ∈ Finset.range n, value p (localTV p q) k [] := by
  induction n with
  | zero => simp [acceptedPrefixTV, pathSum, pathMass]
  | succ n ih =>
      rw [Finset.sum_range_succ]
      exact (prefixTV_step_weighted p q n).trans (add_le_add_right ih _)

/-- Marginalizing the last token recovers the preceding full prefix. -/
theorem prefixTV_le_next (p q : Policy A) (n : ℕ) :
    acceptedPrefixTV p q (fun _ => 1) n ≤ acceptedPrefixTV p q (fun _ => 1) (n + 1) := by
  have hrow : ∀ h, |pathMass q [] h - pathMass p [] h| ≤
      ∑ a, |pathMass q [] (h ++ [a]) - pathMass p [] (h ++ [a])| := by
    intro h
    simpa only [pathMass_snoc] using
      (prefix_row_l1_bounds p q h (pathMass p [] h) (pathMass q [] h)
        (pathMass_nonneg p [] h)).1
  unfold acceptedPrefixTV
  simp only [one_mul]
  rw [pathSum_succ_right]
  exact mul_le_mul_of_nonneg_left (pathSum_mono _ _ hrow n []) (by norm_num)

theorem prefixTV_mono (p q : Policy A) :
    Monotone (acceptedPrefixTV p q (fun _ => 1)) :=
  monotone_nat_of_le_succ (prefixTV_le_next p q)

theorem prefixTV_positive_propagates (p q : Policy A) {s t : ℕ} (hst : s ≤ t)
    (hs : 0 < acceptedPrefixTV p q (fun _ => 1) s) :
    0 < acceptedPrefixTV p q (fun _ => 1) t :=
  hs.trans_le (prefixTV_mono p q hst)

theorem pathSum_congr_length (f g : List A → ℝ) (n : ℕ) (h : List A)
    (hfg : ∀ y, y.length = h.length + n → f y = g y) : pathSum f n h = pathSum g n h := by
  induction n generalizing h with
  | zero => exact hfg h (by omega)
  | succ n ih =>
      apply Finset.sum_congr rfl
      intro a _
      apply ih
      intro y hy
      apply hfg y
      simp only [List.length_append, List.length_singleton] at hy
      omega

theorem pathSum_nonneg (f : List A → ℝ) (hf : ∀ y, 0 ≤ f y) (n : ℕ) (h : List A) :
    0 ≤ pathSum f n h := by
  induction n generalizing h with
  | zero => exact hf h
  | succ n ih => exact Finset.sum_nonneg fun a _ => ih _

theorem pathSum_pos_of_path (f : List A → ℝ) (hf : ∀ y, 0 ≤ f y) (h ys : List A)
    (hys : 0 < f (h ++ ys)) : 0 < pathSum f ys.length h := by
  induction ys generalizing h with
  | nil => simpa [pathSum] using hys
  | cons a ys ih =>
      apply Finset.sum_pos' (fun b _ => pathSum_nonneg f hf ys.length (h ++ [b]))
      refine ⟨a, Finset.mem_univ a, ih (h ++ [a]) ?_⟩
      simpa only [List.append_assoc, List.singleton_append] using hys

/-- At the first disagreement, the preceding prefix masses are equal, and
the next prefix TV equals the rollout-weighted local TV exactly. -/
theorem prefixTV_next_of_equal_prefix_mass (p q : Policy A) (n : ℕ)
    (heq : ∀ h, h.length = n → pathMass p [] h = pathMass q [] h) :
    acceptedPrefixTV p q (fun _ => 1) (n + 1) = value p (localTV p q) n [] := by
  have hrow : ∀ h, h.length = n →
      (∑ a, |pathMass q [] (h ++ [a]) - pathMass p [] (h ++ [a])|) =
        2 * (pathMass p [] h * localTV p q h) := by
    intro h hh
    simp only [pathMass_snoc, ← heq h hh, ← mul_sub, abs_mul,
      abs_of_nonneg (pathMass_nonneg p [] h), ← Finset.mul_sum]
    unfold localTV
    ring
  unfold acceptedPrefixTV
  simp only [one_mul]
  rw [pathSum_succ_right,
    pathSum_congr_length _ (fun h => 2 * (pathMass p [] h * localTV p q h)) n []
      (by intro h hh; exact hrow h (by simpa using hh)),
    pathSum_const_mul, pathSum_mass_value]
  simp [pathMass]

theorem expected_localTV_pos_of_reachable (p q : Policy A) (h : List A)
    (hp : 0 < pathMass p [] h) (htv : 0 < localTV p q h) :
    0 < value p (localTV p q) h.length [] := by
  have hn : ∀ y, 0 ≤ pathMass p [] y * localTV p q y := by
    intro y
    exact mul_nonneg (pathMass_nonneg p [] y)
      (mul_nonneg (by norm_num) (Finset.sum_nonneg fun _ _ => abs_nonneg _))
  have hb := pathSum_pos_of_path (fun y => pathMass p [] y * localTV p q y) hn [] h
    (by simpa using mul_pos hp htv)
  simpa only [pathSum_mass_value, pathMass, one_mul] using hb

/-- Agreement is required only at rollout-reachable histories before depth
`n`; no assumption is made on later kernels or unreachable histories. -/
theorem pathMass_eq_of_agree_before (p q : Policy A) (n : ℕ)
    (hag : ∀ h, h.length < n → 0 < pathMass p [] h → ∀ a, p.prob h a = q.prob h a)
    (y : List A) (hy : y.length ≤ n) : pathMass p [] y = pathMass q [] y := by
  induction y using List.reverseRecOn with
  | nil => rfl
  | append_singleton y a ih =>
      have hlen : y.length < n := by simpa only [List.length_append, List.length_singleton] using hy
      have heq := ih (by omega)
      rw [pathMass_snoc, pathMass_snoc, ← heq]
      by_cases hz : pathMass p [] y = 0
      · simp [hz]
      · rw [hag y hlen (lt_of_le_of_ne (pathMass_nonneg p [] y) (Ne.symm hz)) a]

/-- The complete first-disagreement propagation claim of `rem:coupling`.
Here depth `n` is the context before the paper's token `s = n + 1`, and
every later prefix depth `t >= n + 1` retains a positive discrepancy. -/
theorem first_disagreement_propagates (p q : Policy A) (n : ℕ)
    (hag : ∀ h, h.length < n → 0 < pathMass p [] h → ∀ a, p.prob h a = q.prob h a)
    (h : List A) (hh : h.length = n) (hp : 0 < pathMass p [] h) (htv : 0 < localTV p q h)
    (t : ℕ) (ht : n + 1 ≤ t) : 0 < acceptedPrefixTV p q (fun _ => 1) t := by
  apply prefixTV_positive_propagates p q ht
  rw [prefixTV_next_of_equal_prefix_mass p q n
    (fun y hy => pathMass_eq_of_agree_before p q n hag y (le_of_eq hy))]
  simpa only [hh] using expected_localTV_pos_of_reachable p q h hp htv

end REINFORCEProMax.AutoregressiveTree
