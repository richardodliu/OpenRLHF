import AutoregressiveTree
import ProMax

/-!
Connect the finite history tree to the manuscript's trajectory formula.
The terminal history records the actual token ratios along that history.
Expectations are the normalized, recursively generated tree expectations,
not arbitrary weights supplied with a change-of-measure identity as a premise.
-/
namespace REINFORCEProMax.AutoregressiveTree

variable {A : Type*} [Fintype A]

noncomputable def tokenRatio (p q : Policy A) (h : List A) (a : A) : ℝ :=
  q.prob h a / p.prob h a

noncomputable def traceStep (p q : Policy A) (s : List A × List ℝ) (a : A) :
    List A × List ℝ := (s.1 ++ [a], s.2 ++ [tokenRatio p q s.1 a])

noncomputable def ratioTrace (p q : Policy A) (h : List A) : List ℝ :=
  (h.foldl (traceStep p q) ([], [])).2

theorem trace_history (p q : Policy A) (xs h : List A) (rs : List ℝ) :
    (xs.foldl (traceStep p q) (h, rs)).1 = h ++ xs := by
  induction xs generalizing h rs with
  | nil => simp
  | cons a xs ih =>
      simp only [List.foldl_cons, traceStep]
      rw [ih]
      simp [List.append_assoc]

theorem ratioTrace_snoc (p q : Policy A) (h : List A) (a : A) :
    ratioTrace p q (h ++ [a]) = ratioTrace p q h ++ [tokenRatio p q h a] := by
  simp only [ratioTrace, List.foldl_append, List.foldl_cons, List.foldl_nil, traceStep]
  rw [trace_history]
  simp

theorem surrogateRhs_snoc (xs : List ℝ) (r : ℝ) :
    surrogateRhs (xs ++ [r]) = surrogateRhs xs + (r - 1) := by
  induction xs with
  | nil => simp [surrogateRhs]
  | cons x xs ih => simp only [List.cons_append, surrogateRhs]; rw [ih]; ring

theorem value_add (p : Policy A) (R S : List A → ℝ) (n : ℕ) (h : List A) :
    value p (fun y => R y + S y) n h = value p R n h + value p S n h := by
  induction n generalizing h with
  | zero => rfl
  | succ n ih => simp only [value, ih, mul_add, Finset.sum_add_distrib]

theorem value_sub (p : Policy A) (R S : List A → ℝ) (n : ℕ) (h : List A) :
    value p (fun y => R y - S y) n h = value p R n h - value p S n h := by
  induction n generalizing h with
  | zero => rfl
  | succ n ih => simp only [value, ih, mul_sub, Finset.sum_sub_distrib]

theorem prob_mul_ratio (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (h : List A) (a : A) :
    p.prob h a * tokenRatio p q h a = q.prob h a := by
  by_cases hp : p.prob h a = 0
  · simp [hp, hs h a hp]
  · dsimp [tokenRatio]
    field_simp

/-- The terminal ratio-sum payoff is precisely the rollout-occupancy
surrogate plus the already accumulated prefix-signal offset. -/
theorem trajectory_surrogate_bridge (p q : Policy A) (R : List A → ℝ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) (h : List A) :
    value p (fun y => R y * surrogateRhs (ratioTrace p q y)) n h =
      value p R n h * surrogateRhs (ratioTrace p q h) + surrogate p q R n h := by
  induction n generalizing h with
  | zero => simp [value, surrogate, additive]
  | succ n ih =>
      simp only [value, ih, ratioTrace_snoc, surrogateRhs_snoc]
      change (∑ a, p.prob h a *
          (value p R n (h ++ [a]) * (surrogateRhs (ratioTrace p q h) +
            (tokenRatio p q h a - 1)) + surrogate p q R n (h ++ [a]))) =
        (∑ a, p.prob h a * value p R n (h ++ [a])) * surrogateRhs (ratioTrace p q h) +
          (gain p q R n h + ∑ a, p.prob h a * surrogate p q R n (h ++ [a]))
      rw [Finset.sum_mul]
      unfold gain
      rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro a _
      have hr := prob_mul_ratio p q hs h a
      have hres : p.prob h a * (tokenRatio p q h a - 1) = q.prob h a - p.prob h a := by
        rw [mul_sub, hr]
        ring
      calc
        p.prob h a *
            (value p R n (h ++ [a]) * (surrogateRhs (ratioTrace p q h) +
              (tokenRatio p q h a - 1)) + surrogate p q R n (h ++ [a])) =
            p.prob h a * value p R n (h ++ [a]) * surrogateRhs (ratioTrace p q h) +
              (p.prob h a * (tokenRatio p q h a - 1)) * value p R n (h ++ [a]) +
              p.prob h a * surrogate p q R n (h ++ [a]) := by ring
        _ = p.prob h a * value p R n (h ++ [a]) * surrogateRhs (ratioTrace p q h) +
              (q.prob h a - p.prob h a) * value p R n (h ++ [a]) +
              p.prob h a * surrogate p q R n (h ++ [a]) := by rw [hres]
        _ = p.prob h a * value p R n (h ++ [a]) * surrogateRhs (ratioTrace p q h) +
              ((q.prob h a - p.prob h a) * value p R n (h ++ [a]) +
                p.prob h a * surrogate p q R n (h ++ [a])) := by ring

/-- Full change of measure is derived by induction from per-context support
and normalized policy kernels. Off-policy current policies are allowed. -/
theorem trajectory_change_of_measure (p q : Policy A) (R : List A → ℝ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) (h : List A) :
    value p (fun y => R y * (ratioTrace p q y).prod) n h =
      value q R n h * (ratioTrace p q h).prod := by
  induction n generalizing h with
  | zero => rfl
  | succ n ih =>
      simp only [value, ih, ratioTrace_snoc, List.prod_append, List.prod_cons, List.prod_nil,
        mul_one]
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro a _
      have hr := prob_mul_ratio p q hs h a
      calc
        p.prob h a * (value q R n (h ++ [a]) *
            ((ratioTrace p q h).prod * tokenRatio p q h a)) =
          (p.prob h a * tokenRatio p q h a) * value q R n (h ++ [a]) *
            (ratioTrace p q h).prod := by ring
        _ = _ := by rw [hr]

/-- The exact formula from the paper, on a constructed autoregressive model:
expected reward change equals expected reward-weighted surrogate minus
expected reward-weighted remainder, both sampled under the rollout policy. -/
theorem trajectory_surrogate_remainder_identity (p q : Policy A) (R : List A → ℝ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) :
    value q R n [] - value p R n [] =
      value p (fun y => R y * surrogateRhs (ratioTrace p q y)) n [] -
      value p (fun y => R y * remainderRhs (ratioTrace p q y)) n [] := by
  have hpoint : (fun y => R y * surrogateRhs (ratioTrace p q y) -
      R y * remainderRhs (ratioTrace p q y)) =
      (fun y => R y * (ratioTrace p q y).prod - R y) := by
    funext y
    rw [← mul_sub, surrogate_minus_remainder, ← finite_horizon_product_telescoping]
    ring
  rw [← value_sub, hpoint, value_sub, trajectory_change_of_measure p q R hs]
  simp [ratioTrace]

/-- A terminal sequence-reward ratio sum equals the standard surrogate at
the root, so the tree performance bounds refer to the same objective. -/
theorem trajectory_surrogate_eq (p q : Policy A) (R : List A → ℝ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) :
    value p (fun y => R y * surrogateRhs (ratioTrace p q y)) n [] =
      surrogate p q R n [] := by
  simpa [ratioTrace, surrogateRhs] using trajectory_surrogate_bridge p q R hs n []

theorem trajectory_masked_surrogate_eq (p q : Policy A) (R : List A → ℝ)
    (m : List A → ℝ)
    (hm : ∀ h, m h = 0 ∨ m h = 1)
    (hinactive : ∀ h, m h = 0 → ∀ a, q.prob h a = p.prob h a)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) :
    maskedSurrogate p q R m n [] =
      value p (fun y => R y * surrogateRhs (ratioTrace p q y)) n [] := by
  rw [masked_surrogate_eq_surrogate p q R m hm hinactive]
  exact (trajectory_surrogate_eq p q R hs n).symm

theorem trajectory_remainder_bound (p q : Policy A) (R : List A → ℝ)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0)
    (B ε : ℝ) (hB : 0 ≤ B) (hε : 0 ≤ ε)
    (hR : ∀ h, |R h| ≤ B) (hTV : ∀ h, localTV p q h ≤ ε) (n : ℕ) :
    |value p (fun y => R y * remainderRhs (ratioTrace p q y)) n []| ≤
      min (2 * B * (n : ℝ) * ((n : ℝ) - 1) * ε ^ 2) (4 * B * driftCost p q n []) := by
  have hid := trajectory_surrogate_remainder_identity p q R hs n
  rw [trajectory_surrogate_eq p q R hs] at hid
  have hq := surrogate_error_quadratic p q R B ε hB hε hR hTV n []
  have hl := surrogate_error_linear p q R B hR n []
  have heq : value q R n [] - value p R n [] - surrogate p q R n [] =
      -value p (fun y => R y * remainderRhs (ratioTrace p q y)) n [] := by linarith
  rw [heq, abs_neg] at hq hl
  exact le_min hq hl

end REINFORCEProMax.AutoregressiveTree
