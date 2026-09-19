import TreeSurrogate

/-!
Finite-history concentration for the actual autoregressive prefix law.
All token kernels may depend on the full history. Mutual support allows zero
probabilities (including common absorbing EOS transitions); no token or history
independence is assumed. The centered MGF is evaluated only at lambda = 2,
which follows from the manuscript's stronger all-lambda assumption.
-/
namespace REINFORCEProMax.AutoregressiveTree

open scoped BigOperators
variable {A : Type*} [Fintype A]

noncomputable def logRatioMean (p q : Policy A) (h : List A) : ℝ :=
  ∑ a, p.prob h a * Real.log (tokenRatio p q h a)

/-- Finite conditional KL sign, with common zero-probability actions allowed. -/
theorem logRatioMean_nonpos (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0) (h : List A) :
    logRatioMean p q h ≤ 0 := by
  have hterm : ∀ a, p.prob h a * Real.log (tokenRatio p q h a) ≤
      q.prob h a - p.prob h a := by
    intro a
    by_cases hz : p.prob h a = 0
    · simp [hz, (hs h a).mp hz]
    · have hp := lt_of_le_of_ne (p.nonneg h a) (Ne.symm hz)
      have hq := lt_of_le_of_ne (q.nonneg h a) (Ne.symm (mt (hs h a).mpr hz))
      have hr : 0 < tokenRatio p q h a := div_pos hq hp
      have hb := mul_le_mul_of_nonneg_left (Real.log_le_sub_one_of_pos hr) (p.nonneg h a)
      rw [mul_sub, prob_mul_ratio p q (fun h a => (hs h a).mp), mul_one] at hb
      exact hb
  calc
    logRatioMean p q h ≤ ∑ a, (q.prob h a - p.prob h a) := Finset.sum_le_sum fun a _ => hterm a
    _ = 0 := by rw [Finset.sum_sub_distrib, p.mass, q.mass]; ring

theorem local_ratio_second_moment_of_mgf (p q : Policy A) (σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (h : List A)
    (hmgf : ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2)) :
    ∑ a, p.prob h a * (tokenRatio p q h a) ^ 2 ≤ Real.exp (2 * σ ^ 2) := by
  let ph : Policy A := ⟨fun _ => p.prob h, fun _ => p.nonneg h, fun _ => p.mass h⟩
  have hb := local_second_moment_of_centered_log_mgf ph
    (fun _ a => Real.log (tokenRatio p q h a)) (fun _ => logRatioMean p q h) (fun _ => σ)
    (fun _ => logRatioMean_nonpos p q hs h) (fun _ => hmgf) []
  change (∑ a, p.prob h a * Real.exp (Real.log (tokenRatio p q h a)) ^ 2) ≤ _ at hb
  convert hb using 1
  apply Finset.sum_congr rfl
  intro a _
  by_cases hz : p.prob h a = 0
  · simp [hz]
  · have hp := lt_of_le_of_ne (p.nonneg h a) (Ne.symm hz)
    have hq := lt_of_le_of_ne (q.nonneg h a) (Ne.symm (mt (hs h a).mpr hz))
    have hr : 0 < tokenRatio p q h a := div_pos hq hp
    rw [Real.exp_log hr]

theorem value_const (p : Policy A) (c : ℝ) (n : ℕ) (h : List A) :
    value p (fun _ => c) n h = c := by
  induction n generalizing h with
  | zero => rfl
  | succ n ih => simp only [value, ih, ← Finset.sum_mul, p.mass, one_mul]

theorem value_nonneg (p : Policy A) (R : List A → ℝ)
    (hR : ∀ h, 0 ≤ R h) (n : ℕ) (h : List A) : 0 ≤ value p R n h := by
  induction n generalizing h with
  | zero => exact hR h
  | succ n ih => exact Finset.sum_nonneg fun a _ => mul_nonneg (p.nonneg h a) (ih _)

theorem value_mono (p : Policy A) (R S : List A → ℝ)
    (hRS : ∀ h, R h ≤ S h) (n : ℕ) (h : List A) : value p R n h ≤ value p S n h := by
  induction n generalizing h with
  | zero => exact hRS h
  | succ n ih => exact Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (ih _) (p.nonneg h a)

theorem value_const_mul (p : Policy A) (c : ℝ) (R : List A → ℝ) (n : ℕ) (h : List A) :
    value p (fun y => c * R y) n h = c * value p R n h := by
  induction n generalizing h with
  | zero => rfl
  | succ n ih => simp only [value, ih, Finset.mul_sum]; apply Finset.sum_congr rfl; intros; ring

theorem trajectory_second_moment (p q : Policy A) (n : ℕ) (h : List A) :
    value p (fun y => (ratioTrace p q y).prod ^ 2) n h =
      (ratioTrace p q h).prod ^ 2 * prefixSecondMoment p (tokenRatio p q) n h := by
  induction n generalizing h with
  | zero => simp [value, prefixSecondMoment]
  | succ n ih =>
      simp only [value, ih, ratioTrace_snoc, List.prod_append, List.prod_cons, List.prod_nil,
        mul_one, prefixSecondMoment, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intros
      ring

/-- Probability of a specified continuation, obtained by multiplying the
history-dependent kernels, rather than postulating a trajectory distribution. -/
noncomputable def pathMass (p : Policy A) : List A → List A → ℝ
  | _, [] => 1
  | h, a :: ys => p.prob h a * pathMass p (h ++ [a]) ys

theorem pathMass_nonneg (p : Policy A) (h ys : List A) : 0 ≤ pathMass p h ys := by
  induction ys generalizing h with
  | nil => exact zero_le_one
  | cons a ys ih => exact mul_nonneg (p.nonneg h a) (ih _)

theorem pathMass_append (p : Policy A) (h xs ys : List A) :
    pathMass p h (xs ++ ys) = pathMass p h xs * pathMass p (h ++ xs) ys := by
  induction xs generalizing h with
  | nil => simp [pathMass]
  | cons a xs ih => simp [pathMass, ih, List.append_assoc, mul_assoc]

theorem pathMass_snoc (p : Policy A) (h : List A) (a : A) :
    pathMass p [] (h ++ [a]) = pathMass p [] h * p.prob h a := by
  rw [pathMass_append]; simp [pathMass]

theorem pathMass_ratio (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (y : List A) :
    pathMass q [] y = pathMass p [] y * (ratioTrace p q y).prod := by
  induction y using List.reverseRecOn with
  | nil => simp [pathMass, ratioTrace]
  | append_singleton y a ih =>
      rw [pathMass_snoc, pathMass_snoc, ratioTrace_snoc, List.prod_append]
      simp only [List.prod_cons, List.prod_nil, mul_one]
      rw [ih]
      have hr := prob_mul_ratio p q hs y a
      calc
        pathMass p [] y * (ratioTrace p q y).prod * q.prob y a =
            pathMass p [] y * (ratioTrace p q y).prod * (p.prob y a * tokenRatio p q y a) := by rw [hr]
        _ = _ := by ring

/-- Only the histories within the specified finite horizon need a local bound. -/
theorem prefixSecondMoment_bound_within (p : Policy A) (rho : List A → A → ℝ)
    (C : ℝ) (hC : 0 ≤ C) (N : ℕ)
    (hlocal : ∀ h, h.length < N → ∑ a, p.prob h a * (rho h a) ^ 2 ≤ C)
    (n : ℕ) (h : List A) (hh : h.length + n ≤ N) :
    prefixSecondMoment p rho n h ≤ C ^ n := by
  induction n generalizing h with
  | zero => simp [prefixSecondMoment]
  | succ n ih =>
      calc
        prefixSecondMoment p rho (n + 1) h ≤ ∑ a, p.prob h a * (rho h a) ^ 2 * C ^ n := by
          apply Finset.sum_le_sum
          intro a _
          exact mul_le_mul_of_nonneg_left (ih (h ++ [a]) (by simp only [List.length_append, List.length_singleton]; omega))
            (mul_nonneg (p.nonneg h a) (sq_nonneg _))
        _ = (∑ a, p.prob h a * (rho h a) ^ 2) * C ^ n := (Finset.sum_mul _ _ _).symm
        _ ≤ C * C ^ n := mul_le_mul_of_nonneg_right (hlocal h (by omega)) (pow_nonneg hC _)
        _ = C ^ (n + 1) := by ring

/-- Unreachable histories need no local moment bound: their incoming factor
is exactly zero. This matches the paper's rollout-reachable MGF assumption. -/
theorem prefixSecondMoment_bound_reachable (p : Policy A) (rho : List A → A → ℝ)
    (C : ℝ) (hC : 0 ≤ C) (N : ℕ)
    (hlocal : ∀ h, h.length < N → 0 < pathMass p [] h →
      ∑ a, p.prob h a * (rho h a) ^ 2 ≤ C)
    (n : ℕ) (h : List A) (hh : h.length + n ≤ N) (hr : 0 < pathMass p [] h) :
    prefixSecondMoment p rho n h ≤ C ^ n := by
  induction n generalizing h with
  | zero => simp [prefixSecondMoment]
  | succ n ih =>
      calc
        prefixSecondMoment p rho (n + 1) h ≤ ∑ a, p.prob h a * (rho h a) ^ 2 * C ^ n := by
          apply Finset.sum_le_sum
          intro a _
          by_cases hz : p.prob h a = 0
          · simp [hz]
          have hp : 0 < p.prob h a := lt_of_le_of_ne (p.nonneg h a) (Ne.symm hz)
          have hreach : 0 < pathMass p [] (h ++ [a]) := by
            rw [pathMass_snoc]
            exact mul_pos hr hp
          exact mul_le_mul_of_nonneg_left
            (ih (h ++ [a]) (by simp only [List.length_append, List.length_singleton]; omega) hreach)
            (mul_nonneg (p.nonneg h a) (sq_nonneg _))
        _ = (∑ a, p.prob h a * (rho h a) ^ 2) * C ^ n := (Finset.sum_mul _ _ _).symm
        _ ≤ C * C ^ n := mul_le_mul_of_nonneg_right (hlocal h (by omega) hr) (pow_nonneg hC _)
        _ = C ^ (n + 1) := by ring

/-- The paper's E_p[W_t^2] bound, from normalized mutually supported policies
and their finite conditional MGF; the expectation is built from those policies. -/
theorem trajectory_second_moment_of_mgf (p q : Policy A) (σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (n : ℕ)
    (hmgf : ∀ h, h.length < n → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2)) :
    value p (fun y => (ratioTrace p q y).prod ^ 2) n [] ≤
      Real.exp (2 * σ ^ 2 * (n : ℝ)) := by
  rw [trajectory_second_moment]
  simp only [ratioTrace, List.foldl_nil, List.prod_nil, one_pow, one_mul]
  calc
    prefixSecondMoment p (tokenRatio p q) n [] ≤ (Real.exp (2 * σ ^ 2)) ^ n :=
      prefixSecondMoment_bound_reachable p (tokenRatio p q) _ (Real.exp_nonneg _) n
        (fun h hh hr => local_ratio_second_moment_of_mgf p q σ hs h (hmgf h hh hr))
        n [] (by simp) (by simp [pathMass])
    _ = Real.exp (2 * σ ^ 2 * (n : ℝ)) := by rw [← Real.exp_nat_mul]; congr 1; ring

theorem trajectory_ratio_mass (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) :
    value p (fun y => (ratioTrace p q y).prod) n [] = 1 := by
  have hb := trajectory_change_of_measure p q (fun _ => 1) hs n []
  simpa [value_const, ratioTrace] using hb

theorem trajectory_ratio_variance (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0) (n : ℕ) :
    value p (fun y => ((ratioTrace p q y).prod - 1) ^ 2) n [] =
      value p (fun y => (ratioTrace p q y).prod ^ 2) n [] - 1 := by
  have hid : (fun y => ((ratioTrace p q y).prod - 1) ^ 2) =
      (fun y => (ratioTrace p q y).prod ^ 2 - 2 * (ratioTrace p q y).prod + 1) := by
    funext y; ring
  rw [hid, value_add, value_sub, value_const_mul, trajectory_ratio_mass p q hs, value_const]
  ring

/-- Cauchy--Schwarz for the normalized recursive tree expectation. -/
theorem value_cauchy_sq (p : Policy A) (f g : List A → ℝ) (n : ℕ) (h : List A) :
    value p (fun y => f y * g y) n h ^ 2 ≤
      value p (fun y => f y ^ 2) n h * value p (fun y => g y ^ 2) n h := by
  have hquad : ∀ x : ℝ, 0 ≤ value p (fun y => f y ^ 2) n h * (x * x) +
      (2 * value p (fun y => f y * g y) n h) * x + value p (fun y => g y ^ 2) n h := by
    intro x
    have hb := value_nonneg p (fun y => (x * f y + g y) ^ 2) (fun _ => sq_nonneg _) n h
    have hid : (fun y => (x * f y + g y) ^ 2) =
        (fun y => x ^ 2 * f y ^ 2 + (2 * x) * (f y * g y) + g y ^ 2) := by
      funext y; ring
    rw [hid, value_add, value_add, value_const_mul, value_const_mul] at hb
    nlinarith only [hb]
  have hd := discrim_le_zero hquad
  dsimp [discrim] at hd
  nlinarith only [hd]

/-- The acceptance-coupled bound holds for every prefix-measurable gate in
[0,1], hence in particular the paper's binary geometric-mean prefix gate.
No assumption of independence between the gate and the prefix weight is used. -/
theorem accepted_prefix_residual_of_mgf (p q : Policy A) (σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (n : ℕ)
    (hmgf : ∀ h, h.length < n → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2))
    (M : List A → ℝ) (hM0 : ∀ h, 0 ≤ M h) (hM1 : ∀ h, M h ≤ 1) :
    value p (fun y => M y * |(ratioTrace p q y).prod - 1|) n [] ≤
      Real.sqrt (value p M n []) * Real.sqrt (Real.exp (2 * σ ^ 2 * (n : ℝ)) - 1) := by
  have hcs := value_cauchy_sq p M (fun y => |(ratioTrace p q y).prod - 1|) n []
  simp only [sq_abs] at hcs
  have hm := value_mono p (fun y => M y ^ 2) M (fun y => by
    simpa only [pow_two] using mul_le_of_le_one_right (hM0 y) (hM1 y)) n []
  have hv := value_nonneg p (fun y => ((ratioTrace p q y).prod - 1) ^ 2)
    (fun _ => sq_nonneg _) n []
  have ha := value_nonneg p M hM0 n []
  have hvar : value p (fun y => ((ratioTrace p q y).prod - 1) ^ 2) n [] ≤
      Real.exp (2 * σ ^ 2 * (n : ℝ)) - 1 := by
    rw [trajectory_ratio_variance p q (fun h a => (hs h a).mp)]
    exact sub_le_sub_right (trajectory_second_moment_of_mgf p q σ hs n hmgf) 1
  have hsq := hcs.trans ((mul_le_mul_of_nonneg_right hm hv).trans
    (mul_le_mul_of_nonneg_left hvar ha))
  have he : 0 ≤ Real.exp (2 * σ ^ 2 * (n : ℝ)) - 1 := le_trans hv hvar
  have hroot : (Real.sqrt (value p M n []) *
      Real.sqrt (Real.exp (2 * σ ^ 2 * (n : ℝ)) - 1)) ^ 2 =
      value p M n [] * (Real.exp (2 * σ ^ 2 * (n : ℝ)) - 1) := by
    rw [mul_pow, Real.sq_sqrt ha, Real.sq_sqrt he]
  have hn := mul_nonneg (Real.sqrt_nonneg (value p M n []))
    (Real.sqrt_nonneg (Real.exp (2 * σ ^ 2 * (n : ℝ)) - 1))
  nlinarith only [hsq, hroot, hn]

/-- Unweighted sum over the finite continuation tree. -/
noncomputable def pathSum (f : List A → ℝ) : ℕ → List A → ℝ
  | 0, h => f h
  | n + 1, h => ∑ a, pathSum f n (h ++ [a])

theorem pathSum_mass_value (p : Policy A) (R : List A → ℝ) (n : ℕ) (h : List A) :
    pathSum (fun y => pathMass p [] y * R y) n h = pathMass p [] h * value p R n h := by
  induction n generalizing h with
  | zero => rfl
  | succ n ih =>
      simp only [pathSum, ih, pathMass_snoc, value, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intros; ring

theorem pathMass_normalized (p : Policy A) (n : ℕ) :
    pathSum (pathMass p []) n [] = 1 := by
  have hb := pathSum_mass_value p (fun _ => 1) n []
  simpa [value_const, pathMass] using hb

/-- Half L1 distance of the two sub-probability masses produced by the same
prefix gate. For unequal masses this specifies the TV convention explicitly. -/
noncomputable def acceptedPrefixTV (p q : Policy A) (M : List A → ℝ) (n : ℕ) : ℝ :=
  (1 / 2 : ℝ) * pathSum (fun y => |M y * pathMass q [] y - M y * pathMass p [] y|) n []

theorem acceptedPrefixTV_identity (p q : Policy A)
    (hs : ∀ h a, p.prob h a = 0 → q.prob h a = 0)
    (M : List A → ℝ) (hM : ∀ h, 0 ≤ M h) (n : ℕ) :
    acceptedPrefixTV p q M n =
      (1 / 2 : ℝ) * value p (fun y => M y * |(ratioTrace p q y).prod - 1|) n [] := by
  have hpoint : (fun y => |M y * pathMass q [] y - M y * pathMass p [] y|) =
      (fun y => pathMass p [] y * (M y * |(ratioTrace p q y).prod - 1|)) := by
    funext y
    rw [pathMass_ratio p q hs]
    have hid : M y * (pathMass p [] y * (ratioTrace p q y).prod) - M y * pathMass p [] y =
        pathMass p [] y * (M y * ((ratioTrace p q y).prod - 1)) := by ring
    rw [hid, abs_mul, abs_mul, abs_of_nonneg (pathMass_nonneg p [] y), abs_of_nonneg (hM y)]
  unfold acceptedPrefixTV
  rw [hpoint, pathSum_mass_value]
  simp [pathMass]

/-- Complete finite-history version of the paper's filtered TV corollary. -/
theorem acceptedPrefixTV_of_mgf (p q : Policy A) (σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (n : ℕ)
    (hmgf : ∀ h, h.length < n → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2))
    (M : List A → ℝ) (hM0 : ∀ h, 0 ≤ M h) (hM1 : ∀ h, M h ≤ 1) :
    acceptedPrefixTV p q M n ≤ (1 / 2 : ℝ) *
      Real.sqrt (value p M n []) * Real.sqrt (Real.exp (2 * σ ^ 2 * (n : ℝ)) - 1) := by
  rw [acceptedPrefixTV_identity p q (fun h a => (hs h a).mp) M hM0]
  simpa only [mul_assoc] using mul_le_mul_of_nonneg_left
    (accepted_prefix_residual_of_mgf p q σ hs n hmgf M hM0 hM1) (by norm_num : (0 : ℝ) ≤ 1 / 2)

theorem prefixTV_of_mgf (p q : Policy A) (σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (n : ℕ)
    (hmgf : ∀ h, h.length < n → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2)) :
    acceptedPrefixTV p q (fun _ => 1) n ≤
      (1 / 2 : ℝ) * Real.sqrt (Real.exp (2 * σ ^ 2 * (n : ℝ)) - 1) := by
  simpa [value_const] using acceptedPrefixTV_of_mgf p q σ hs n hmgf
    (fun _ => 1) (fun _ => zero_le_one) (fun _ => le_rfl)

/-- Sum of the accepted-prefix drift bounds, with the exact effective horizon
sum_t E_p[M_t]. Gates may vary by position and depend on the current action. -/
theorem effective_horizon_of_mgf (p q : Policy A) (σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (N : ℕ)
    (hmgf : ∀ h, h.length < N → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2))
    (M : Fin N → List A → ℝ)
    (hM0 : ∀ t h, 0 ≤ M t h) (hM1 : ∀ t h, M t h ≤ 1) :
    (∑ t : Fin N, acceptedPrefixTV p q (M t) (t.val + 1)) ≤
      (1 / 2 : ℝ) * Real.sqrt (∑ t : Fin N, value p (M t) (t.val + 1) []) *
        Real.sqrt (∑ t : Fin N, (Real.exp (2 * σ ^ 2 * ((t.val + 1 : ℕ) : ℝ)) - 1)) := by
  have he : ∀ t : Fin N, 0 ≤ Real.exp (2 * σ ^ 2 * ((t.val + 1 : ℕ) : ℝ)) - 1 := by
    intro t
    have hx : 0 ≤ 2 * σ ^ 2 * ((t.val + 1 : ℕ) : ℝ) := by positivity
    have hb := Real.exp_le_exp.mpr hx
    simpa using sub_nonneg.mpr hb
  have hcs := Real.sum_sqrt_mul_sqrt_le (Finset.univ : Finset (Fin N))
    (fun t => value_nonneg p (M t) (hM0 t) (t.val + 1) []) he
  calc
    (∑ t : Fin N, acceptedPrefixTV p q (M t) (t.val + 1)) ≤
        ∑ t : Fin N, (1 / 2 : ℝ) * Real.sqrt (value p (M t) (t.val + 1) []) *
          Real.sqrt (Real.exp (2 * σ ^ 2 * ((t.val + 1 : ℕ) : ℝ)) - 1) := by
      apply Finset.sum_le_sum
      intro t _
      exact acceptedPrefixTV_of_mgf p q σ hs (t.val + 1)
        (fun h hh => hmgf h (by omega)) (M t) (hM0 t) (hM1 t)
    _ = (1 / 2 : ℝ) * (∑ t : Fin N, Real.sqrt (value p (M t) (t.val + 1) []) *
          Real.sqrt (Real.exp (2 * σ ^ 2 * ((t.val + 1 : ℕ) : ℝ)) - 1)) := by
      simp only [Finset.mul_sum, mul_assoc]
    _ ≤ _ := by
      simpa only [mul_assoc] using mul_le_mul_of_nonneg_left hcs (by norm_num : (0 : ℝ) ≤ 1 / 2)

theorem abs_value_le_value_abs (p : Policy A) (R : List A → ℝ) (n : ℕ) (h : List A) :
    |value p R n h| ≤ value p (fun y => |R y|) n h := by
  induction n generalizing h with
  | zero => exact le_rfl
  | succ n ih =>
      calc
        |value p R (n + 1) h| ≤ ∑ a, |p.prob h a * value p R n (h ++ [a])| :=
          Finset.abs_sum_le_sum_abs _ _
        _ = ∑ a, p.prob h a * |value p R n (h ++ [a])| := by
          simp only [abs_mul, abs_of_nonneg (p.nonneg _ _)]
        _ ≤ value p (fun y => |R y|) (n + 1) h :=
          Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (ih _) (p.nonneg h a)

/-- A bounded test function on prefix contexts, under the two actual laws. -/
theorem value_drift_of_mgf (p q : Policy A) (σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (n : ℕ)
    (hmgf : ∀ h, h.length < n → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2))
    (f : List A → ℝ) (B : ℝ) (hB : 0 ≤ B) (hf : ∀ y, |f y| ≤ B) :
    |value q f n [] - value p f n []| ≤ B * Real.sqrt (Real.exp (2 * σ ^ 2 * (n : ℝ)) - 1) := by
  have hchange := trajectory_change_of_measure p q f (fun h a => (hs h a).mp) n []
  rw [show (ratioTrace p q []).prod = 1 from rfl, mul_one] at hchange
  have hid : (fun y => f y * (ratioTrace p q y).prod - f y) =
      (fun y => f y * ((ratioTrace p q y).prod - 1)) := by funext y; ring
  have hres := accepted_prefix_residual_of_mgf p q σ hs n hmgf
    (fun _ => 1) (fun _ => zero_le_one) (fun _ => le_rfl)
  simp only [one_mul, value_const, Real.sqrt_one] at hres
  calc
    |value q f n [] - value p f n []| =
        |value p (fun y => f y * ((ratioTrace p q y).prod - 1)) n []| := by
      rw [← hchange, ← value_sub, hid]
    _ ≤ value p (fun y => |f y * ((ratioTrace p q y).prod - 1)|) n [] := abs_value_le_value_abs _ _ _ _
    _ ≤ value p (fun y => B * |(ratioTrace p q y).prod - 1|) n [] := by
      apply value_mono
      intro y
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_right (hf y) (abs_nonneg _)
    _ = B * value p (fun y => |(ratioTrace p q y).prod - 1|) n [] := value_const_mul _ _ _ _ _
    _ ≤ _ := mul_le_mul_of_nonneg_left hres hB

/-- The recursive cumulative signal is the sum of its prefix-law expectations. -/
theorem additive_eq_prefix_sum (p : Policy A) (f : ℕ → List A → ℝ) (n : ℕ) (h : List A) :
    additive p f n h = ∑ k ∈ Finset.range n, value p (f (n - 1 - k)) k h := by
  induction n generalizing h with
  | zero => simp [additive]
  | succ n ih =>
      rw [additive, Finset.sum_range_succ']
      simp only [Nat.add_sub_cancel, Nat.sub_zero, value]
      simp_rw [ih, Finset.mul_sum]
      have hindex : ∀ k : ℕ, n - (k + 1) = n - 1 - k := by intro k; omega
      rw [Finset.sum_comm, add_comm]
      congr 1
      apply Finset.sum_congr rfl
      intro k _
      rw [hindex k]

theorem localTV_le_one (p q : Policy A) (h : List A) : localTV p q h ≤ 1 := by
  have hb : (∑ a, |q.prob h a - p.prob h a|) ≤ 2 := by
    calc
      _ ≤ ∑ a, (q.prob h a + p.prob h a) := Finset.sum_le_sum fun a _ => by
        rw [abs_sub_le_iff]
        constructor <;> linarith [p.nonneg h a, q.nonneg h a]
      _ = 2 := by rw [Finset.sum_add_distrib, p.mass, q.mass]; ring
  unfold localTV
  linarith

/-- The paper's per-position surrogate-error bound, now derived on the same
finite autoregressive model as the concentration results and actual return. -/
theorem surrogate_error_of_mgf (p q : Policy A) (σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (N : ℕ)
    (hmgf : ∀ h, h.length < N → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2))
    (R : List A → ℝ) (B : ℝ) (hB : 0 ≤ B) (hR : ∀ y, |R y| ≤ B) :
    |value q R N [] - value p R N [] - surrogate p q R N []| ≤
      2 * B * ∑ k ∈ Finset.range N, Real.sqrt (Real.exp (2 * σ ^ 2 * (k : ℝ)) - 1) := by
  have hf : ∀ k h, |gain p q R k h| ≤ 2 * B := by
    intro k h
    exact (gain_abs_le p q R B hR k h).trans
      (by simpa using mul_le_mul_of_nonneg_left (localTV_le_one p q h) (by positivity : 0 ≤ 2 * B))
  rw [performance_difference, surrogate, additive_eq_prefix_sum, additive_eq_prefix_sum,
    ← Finset.sum_sub_distrib]
  calc
    _ ≤ ∑ k ∈ Finset.range N,
        |value q (gain p q R (N - 1 - k)) k [] - value p (gain p q R (N - 1 - k)) k []| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k ∈ Finset.range N, 2 * B * Real.sqrt (Real.exp (2 * σ ^ 2 * (k : ℝ)) - 1) := by
      apply Finset.sum_le_sum
      intro k hk
      have hkN := Finset.mem_range.mp hk
      exact value_drift_of_mgf p q σ hs k (fun h hh => hmgf h (by omega))
        (gain p q R (N - 1 - k)) (2 * B) (by positivity) (hf _)
    _ = _ := (Finset.mul_sum _ _ _).symm

theorem performance_lower_bound_of_mgf (p q : Policy A) (σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (N : ℕ)
    (hmgf : ∀ h, h.length < N → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2))
    (R : List A → ℝ) (B : ℝ) (hB : 0 ≤ B) (hR : ∀ y, |R y| ≤ B) :
    surrogate p q R N [] -
      2 * B * ∑ k ∈ Finset.range N, Real.sqrt (Real.exp (2 * σ ^ 2 * (k : ℝ)) - 1) ≤
        value q R N [] - value p R N [] := by
  have hb := (abs_le.mp (surrogate_error_of_mgf p q σ hs N hmgf R B hB hR)).1
  linarith

/-- A sufficient condition for strict true-reward improvement, expressed in
the raw population surrogate rather than the altered finite-batch PPO loss. -/
theorem performance_improves_of_mgf (p q : Policy A) (σ : ℝ)
    (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (N : ℕ)
    (hmgf : ∀ h, h.length < N → 0 < pathMass p [] h → ∑ a, p.prob h a *
      Real.exp (2 * (Real.log (tokenRatio p q h a) - logRatioMean p q h)) ≤
        Real.exp (2 * σ ^ 2))
    (R : List A → ℝ) (B : ℝ) (hB : 0 ≤ B) (hR : ∀ y, |R y| ≤ B)
    (hgain : 2 * B * ∑ k ∈ Finset.range N,
      Real.sqrt (Real.exp (2 * σ ^ 2 * (k : ℝ)) - 1) < surrogate p q R N []) :
    value p R N [] < value q R N [] := by
  have hb := performance_lower_bound_of_mgf p q σ hs N hmgf R B hB hR
  linarith

end REINFORCEProMax.AutoregressiveTree
