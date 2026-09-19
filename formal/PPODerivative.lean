import Mathlib.Tactic

/-!
The exact derivative of standard PPO with detached reduction weights,
prefix gates, snapshot/rollout ratios, and advantages. Only the genuine
sign-dependent clipping kinks are excluded; clipping and filtering remain
in the objective. These are real-arithmetic statements, not a specification
of an autodiff convention at a nondifferentiable boundary.
-/
namespace REINFORCEProMax.PPODerivative

open scoped BigOperators Topology

noncomputable def ppoTerm (lo hi A r : ℝ) : ℝ :=
  min (r * A) (max lo (min hi r) * A)

theorem ppoTerm_of_nonneg (lo hi A r : ℝ) (hband : lo ≤ hi) (hA : 0 ≤ A) :
    ppoTerm lo hi A r = min r hi * A := by
  unfold ppoTerm
  rcases le_total r lo with hr | hr
  · rw [min_eq_right (hr.trans hband), max_eq_left hr,
      min_eq_left (mul_le_mul_of_nonneg_right hr hA), min_eq_left (hr.trans hband)]
  · rcases le_total r hi with hr' | hr'
    · rw [min_eq_right hr', max_eq_right hr, min_self, min_eq_left hr']
    · rw [min_eq_left hr', max_eq_right hband,
        min_eq_right (mul_le_mul_of_nonneg_right hr' hA), min_eq_right hr']

theorem ppoTerm_of_nonpos (lo hi A r : ℝ) (hband : lo ≤ hi) (hA : A ≤ 0) :
    ppoTerm lo hi A r = max r lo * A := by
  unfold ppoTerm
  rcases le_total r lo with hr | hr
  · rw [min_eq_right (hr.trans hband), max_eq_left hr,
      min_eq_right (mul_le_mul_of_nonpos_right hr hA), max_eq_right hr]
  · rcases le_total r hi with hr' | hr'
    · rw [min_eq_right hr', max_eq_right hr, min_self, max_eq_left hr]
    · rw [min_eq_left hr', max_eq_right hband,
        min_eq_left (mul_le_mul_of_nonpos_right hr' hA), max_eq_left hr]

/-- One exactly on a branch that differentiates through the current ratio. -/
noncomputable def clipActive (lo hi A r : ℝ) : ℝ :=
  if 0 < A then (if r < hi then 1 else 0)
  else if A < 0 then (if lo < r then 1 else 0) else 0

theorem clipActive_binary (lo hi A r : ℝ) :
    clipActive lo hi A r = 0 ∨ clipActive lo hi A r = 1 := by
  unfold clipActive
  split_ifs <;> simp

theorem ppoTerm_hasDerivAt (lo hi A r : ℝ) (hband : lo ≤ hi)
    (hpos : 0 < A → r ≠ hi) (hneg : A < 0 → r ≠ lo) :
    HasDerivAt (ppoTerm lo hi A) (A * clipActive lo hi A r) r := by
  rcases lt_trichotomy A 0 with hA | hA | hA
  · simp only [clipActive, not_lt.mpr hA.le, hA, ↓reduceIte]
    by_cases hr : lo < r
    · simp only [hr, ↓reduceIte, mul_one]
      have hid : HasDerivAt (fun s : ℝ => s * A) A r := by
        simpa using (hasDerivAt_id r).mul_const A
      apply hid.congr_of_eventuallyEq
      filter_upwards [eventually_gt_nhds hr] with s hs
      rw [ppoTerm_of_nonpos lo hi A s hband hA.le, max_eq_left hs.le]
    · simp only [hr, ↓reduceIte, mul_zero]
      have hrl : r < lo := lt_of_le_of_ne (le_of_not_gt hr) (hneg hA)
      apply (hasDerivAt_const r (lo * A)).congr_of_eventuallyEq
      filter_upwards [eventually_lt_nhds hrl] with s hs
      rw [ppoTerm_of_nonpos lo hi A s hband hA.le, max_eq_right hs.le]
  · subst A
    have hz : ppoTerm lo hi 0 = (fun _ : ℝ => 0) := by
      funext s
      simp [ppoTerm]
    rw [hz]
    convert hasDerivAt_const r (0 : ℝ) using 1 <;> ring
  · simp only [clipActive, hA, ↓reduceIte]
    by_cases hr : r < hi
    · simp only [hr, ↓reduceIte, mul_one]
      have hid : HasDerivAt (fun s : ℝ => s * A) A r := by
        simpa using (hasDerivAt_id r).mul_const A
      apply hid.congr_of_eventuallyEq
      filter_upwards [eventually_lt_nhds hr] with s hs
      rw [ppoTerm_of_nonneg lo hi A s hband hA.le, min_eq_left hs.le]
    · simp only [hr, ↓reduceIte, mul_zero]
      have hrl : hi < r := lt_of_le_of_ne (le_of_not_gt hr) (Ne.symm (hpos hA))
      apply (hasDerivAt_const r (hi * A)).congr_of_eventuallyEq
      filter_upwards [eventually_gt_nhds hrl] with s hs
      rw [ppoTerm_of_nonneg lo hi A s hband hA.le, min_eq_right hs.le]

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The current log-probability may have an arbitrary real normed parameter
space. All other arguments are fixed during this derivative. -/
theorem gated_ppo_hasFDerivAt (ell : E → ℝ) (dell : E →L[ℝ] ℝ) (θ : E)
    (hd : HasFDerivAt ell dell θ) (a M w old lo hi A : ℝ) (hband : lo ≤ hi)
    (hpos : 0 < A → Real.exp (ell θ - old) ≠ hi)
    (hneg : A < 0 → Real.exp (ell θ - old) ≠ lo) :
    HasFDerivAt (fun u => a * M * w * ppoTerm lo hi A (Real.exp (ell u - old)))
      ((a * M * w * Real.exp (ell θ - old) * A *
        clipActive lo hi A (Real.exp (ell θ - old))) • dell) θ := by
  have he := (hd.sub_const old).exp
  have hp := (ppoTerm_hasDerivAt lo hi A (Real.exp (ell θ - old)) hband hpos hneg).comp_hasFDerivAt θ he
  simpa [Function.comp_def, smul_smul, mul_assoc, mul_comm, mul_left_comm] using
    hp.const_mul (a * M * w)

/-- The finite objective keeps all sample/action positions, and therefore
also permits both action-token and response-mean reductions. -/
theorem finite_gated_ppo_hasFDerivAt {I : Type*} [Fintype I]
    (ell : I → E → ℝ) (dell : I → E →L[ℝ] ℝ) (θ : E)
    (hd : ∀ i, HasFDerivAt (ell i) (dell i) θ)
    (a M w old A : I → ℝ) (lo hi : ℝ) (hband : lo ≤ hi)
    (hpos : ∀ i, 0 < A i → Real.exp (ell i θ - old i) ≠ hi)
    (hneg : ∀ i, A i < 0 → Real.exp (ell i θ - old i) ≠ lo) :
    HasFDerivAt (fun u => ∑ i, a i * M i * w i *
      ppoTerm lo hi (A i) (Real.exp (ell i u - old i)))
      (∑ i, (a i * M i * w i * Real.exp (ell i θ - old i) * A i *
        clipActive lo hi (A i) (Real.exp (ell i θ - old i))) • dell i) θ := by
  exact HasFDerivAt.sum (fun i _ => gated_ppo_hasFDerivAt (ell i) (dell i) θ
    (hd i) (a i) (M i) (w i) (old i) lo hi (A i) hband (hpos i) (hneg i))

/-- In uniform-scale mode, a retained, differentiating token has exactly the
reward's sign; a rejected/clipped/zero-reward token has zero coefficient. -/
theorem uniform_scale_token_coefficient_sign (a w r reward n : ℝ)
    (ha : 0 < a) (hw : 0 < w) (hr : 0 < r) (hn : 0 < n) :
    (0 < a * w * r * (reward / n) ↔ 0 < reward) ∧
    (a * w * r * (reward / n) < 0 ↔ reward < 0) ∧
    (a * w * r * (reward / n) = 0 ↔ reward = 0) := by
  have h : 0 < a * w * r := mul_pos (mul_pos ha hw) hr
  have hinv : 0 < (1 / n : ℝ) := one_div_pos.mpr hn
  have hcoef : 0 < a * w * r * (1 / n : ℝ) := mul_pos h hinv
  have hrew : a * w * r * (reward / n) =
      (a * w * r * (1 / n : ℝ)) * reward := by
    field_simp [ne_of_gt hn]
  constructor
  · rw [hrew, mul_pos_iff_of_pos_left hcoef]
  constructor
  · rw [hrew]
    constructor
    · intro hx
      rcases (mul_neg_iff.mp hx) with hcase | hcase
      · exact hcase.2
      · linarith [hcase.1, hcoef]
    · intro hx
      exact mul_neg_of_pos_of_neg hcoef hx
  · rw [hrew]
    constructor
    · intro hz
      rcases mul_eq_zero.mp hz with hz | hz
      · linarith [hcoef]
      · exact hz
    · intro hz
      simp [hz]

theorem zero_reward_ppo_term (lo hi r : ℝ) : ppoTerm lo hi 0 r = 0 := by
  simp [ppoTerm]

theorem zero_gate_ppo_term (lo hi A r a w : ℝ) :
    a * 0 * w * ppoTerm lo hi A r = 0 := by ring

end REINFORCEProMax.PPODerivative
