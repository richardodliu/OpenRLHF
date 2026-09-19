import PolicyCover

/-!
Distribution-free bounds for a normalized positive tilt of a finite policy.
The normalization is retained explicitly. In a softmax policy the weights
are exponentiated logit changes, so only their range matters; no minimum
token probability or independent-token assumption is needed.
-/
namespace REINFORCEProMax.LogitBounds

open scoped BigOperators
noncomputable section
variable {A : Type*} [Fintype A]

def tiltMean (p w : A → ℝ) : ℝ := ∑ a, p a * w a
def tiltRow (p w : A → ℝ) (a : A) : ℝ := p a * w a / tiltMean p w
def tiltEnergy (p w : A → ℝ) : ℝ := ∑ a, p a * (w a / tiltMean p w - 1) ^ 2
def tiltKL (p w : A → ℝ) : ℝ := ∑ a, p a * Real.log (tiltMean p w / w a)
def tiltCap (lo hi : ℝ) : ℝ := (hi - lo) ^ 2 / (4 * lo * hi)

theorem tilt_mean_bounds (p w : A → ℝ) (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (lo hi : ℝ) (hwlo : ∀ a, lo ≤ w a) (hwhi : ∀ a, w a ≤ hi) :
    lo ≤ tiltMean p w ∧ tiltMean p w ≤ hi := by
  constructor
  · calc
      lo = ∑ a, p a * lo := by rw [← Finset.sum_mul, hp, one_mul]
      _ ≤ _ := Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (hwlo a) (hp0 a)
  · calc
      _ ≤ ∑ a, p a * hi := Finset.sum_le_sum fun a _ =>
        mul_le_mul_of_nonneg_left (hwhi a) (hp0 a)
      _ = hi := by rw [← Finset.sum_mul, hp, one_mul]

theorem tilt_row_mass (p w : A → ℝ) (hZ : tiltMean p w ≠ 0) :
    ∑ a, tiltRow p w a = 1 := by
  simp only [tiltRow, ← Finset.sum_div]
  exact div_self hZ

theorem tilt_second_moment (p w : A → ℝ) (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (lo hi : ℝ) (hwlo : ∀ a, lo ≤ w a) (hwhi : ∀ a, w a ≤ hi) :
    (∑ a, p a * (w a) ^ 2) ≤ (lo + hi) * tiltMean p w - lo * hi := by
  have hh : ∀ a, (w a) ^ 2 ≤ (lo + hi) * w a - lo * hi := by
    intro a
    nlinarith [mul_nonneg (sub_nonneg.mpr (hwlo a)) (sub_nonneg.mpr (hwhi a))]
  calc
    _ ≤ ∑ a, p a * ((lo + hi) * w a - lo * hi) := Finset.sum_le_sum fun a _ =>
      mul_le_mul_of_nonneg_left (hh a) (hp0 a)
    _ = _ := by
      simp only [mul_sub, Finset.sum_sub_distrib, tiltMean]
      simp_rw [← mul_assoc, mul_comm (p _) (lo + hi), mul_assoc]
      rw [← Finset.mul_sum, ← Finset.sum_mul, hp]
      ring

theorem tilt_energy_identity (p w : A → ℝ) (hp : ∑ a, p a = 1)
    (hZ : tiltMean p w ≠ 0) :
    tiltEnergy p w = ((∑ a, p a * (w a) ^ 2) - (tiltMean p w) ^ 2) /
      (tiltMean p w) ^ 2 := by
  have hterm : ∀ a, p a * (w a / tiltMean p w - 1) ^ 2 =
      (p a * (w a) ^ 2 - 2 * tiltMean p w * (p a * w a) + (tiltMean p w) ^ 2 * p a) /
        (tiltMean p w) ^ 2 := by
    intro a
    field_simp
    ring
  simp only [tiltEnergy, hterm, ← Finset.sum_div, Finset.sum_add_distrib,
    Finset.sum_sub_distrib, ← Finset.mul_sum, hp]
  change _ = _
  unfold tiltMean
  ring

/-- Kantorovich bound for the actual normalized tilt, including its normalizer. -/
theorem tilt_energy_bound (p w : A → ℝ) (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (lo hi : ℝ) (hlo : 0 < lo) (hwlo : ∀ a, lo ≤ w a) (hwhi : ∀ a, w a ≤ hi) :
    tiltEnergy p w ≤ tiltCap lo hi := by
  obtain ⟨hlZ, hZh⟩ := tilt_mean_bounds p w hp0 hp lo hi hwlo hwhi
  have hZ : 0 < tiltMean p w := lt_of_lt_of_le hlo hlZ
  have hhi : 0 < hi := lt_of_lt_of_le hZ hZh
  have hmoment := tilt_second_moment p w hp0 hp lo hi hwlo hwhi
  have hprod := mul_nonneg (show 0 ≤ 4 * lo * hi by positivity) (sub_nonneg.mpr hmoment)
  have hsq := sq_nonneg ((lo + hi) * tiltMean p w - 2 * lo * hi)
  rw [tilt_energy_identity p w hp (ne_of_gt hZ)]
  unfold tiltCap
  apply (div_le_div_iff₀ (sq_pos_of_pos hZ) (by positivity : 0 < 4 * lo * hi)).mpr
  nlinarith only [hprod, hsq]

theorem tilt_reciprocal_moment (p w : A → ℝ) (hp0 : ∀ a, 0 ≤ p a)
    (hp : ∑ a, p a = 1) (lo hi : ℝ) (hlo : 0 < lo)
    (hwlo : ∀ a, lo ≤ w a) (hwhi : ∀ a, w a ≤ hi) :
    tiltMean p w + lo * hi * (∑ a, p a / w a) ≤ lo + hi := by
  have hh : ∀ a, w a + lo * hi / w a ≤ lo + hi := by
    intro a
    have hw := lt_of_lt_of_le hlo (hwlo a)
    have hb : lo * hi / w a ≤ lo + hi - w a := (div_le_iff₀ hw).mpr (by
      nlinarith [mul_nonneg (sub_nonneg.mpr (hwlo a)) (sub_nonneg.mpr (hwhi a))])
    linarith
  have hsum := Finset.sum_le_sum (s := Finset.univ) (fun a _ =>
    mul_le_mul_of_nonneg_left (hh a) (hp0 a))
  have hid : (∑ a, p a * (w a + lo * hi / w a)) =
      tiltMean p w + lo * hi * (∑ a, p a / w a) := by
    simp only [tiltMean, mul_add, Finset.sum_add_distrib, Finset.mul_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro a _
    ring
  rwa [hid, ← Finset.sum_mul, hp, one_mul] at hsum

theorem tilt_kl_bound (p w : A → ℝ) (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (lo hi : ℝ) (hlo : 0 < lo) (hwlo : ∀ a, lo ≤ w a) (hwhi : ∀ a, w a ≤ hi) :
    tiltKL p w ≤ tiltCap lo hi := by
  obtain ⟨hlZ, hZh⟩ := tilt_mean_bounds p w hp0 hp lo hi hwlo hwhi
  have hZ : 0 < tiltMean p w := lt_of_lt_of_le hlo hlZ
  have hhi : 0 < hi := lt_of_lt_of_le hZ hZh
  have hU : 0 ≤ ∑ a, p a / w a := Finset.sum_nonneg fun a _ =>
    div_nonneg (hp0 a) (le_trans hlo.le (hwlo a))
  have hrec := tilt_reciprocal_moment p w hp0 hp lo hi hlo hwlo hwhi
  have hrec_sq := (sq_le_sq₀ (by positivity : 0 ≤ tiltMean p w + lo * hi * (∑ a, p a / w a))
    (by positivity : 0 ≤ lo + hi)).mpr hrec
  have hdiff := sq_nonneg (tiltMean p w - lo * hi * (∑ a, p a / w a))
  have hlog : tiltKL p w ≤ tiltMean p w * (∑ a, p a / w a) - 1 := by
    calc
      _ ≤ ∑ a, p a * (tiltMean p w / w a - 1) := Finset.sum_le_sum fun a _ =>
        mul_le_mul_of_nonneg_left (Real.log_le_sub_one_of_pos
          (div_pos hZ (lt_of_lt_of_le hlo (hwlo a)))) (hp0 a)
      _ = _ := by
        simp only [mul_sub, mul_one, Finset.sum_sub_distrib, hp, Finset.mul_sum]
        congr 1
        apply Finset.sum_congr rfl
        intro a _
        ring
  apply hlog.trans
  unfold tiltCap
  apply (le_div_iff₀ (by positivity : 0 < 4 * lo * hi)).mpr
  nlinarith only [hrec_sq, hdiff]

theorem weighted_mean_square_le (p f : A → ℝ) (hp0 : ∀ a, 0 ≤ p a)
    (hp : ∑ a, p a = 1) : (∑ a, p a * f a) ^ 2 ≤ ∑ a, p a * (f a) ^ 2 := by
  let c := ∑ a, p a * f a
  have hn : 0 ≤ ∑ a, p a * (f a - c) ^ 2 :=
    Finset.sum_nonneg fun a _ => mul_nonneg (hp0 a) (sq_nonneg _)
  have hterm : ∀ a, p a * (f a - c) ^ 2 =
      p a * (f a) ^ 2 - 2 * c * (p a * f a) + p a * c ^ 2 := by intro a; ring
  simp_rw [hterm, Finset.sum_add_distrib, Finset.sum_sub_distrib] at hn
  rw [← Finset.mul_sum, ← Finset.sum_mul, hp] at hn
  change 0 ≤ (∑ a, p a * (f a) ^ 2) - 2 * c * c + 1 * c ^ 2 at hn
  change c ^ 2 ≤ _
  nlinarith only [hn]

theorem tilt_tv_bound (p w : A → ℝ) (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (lo hi : ℝ) (hlo : 0 < lo) (hwlo : ∀ a, lo ≤ w a) (hwhi : ∀ a, w a ≤ hi) :
    (1 / 2 : ℝ) * (∑ a, |tiltRow p w a - p a|) ≤ Real.sqrt (tiltCap lo hi) / 2 := by
  have he := tilt_energy_bound p w hp0 hp lo hi hlo hwlo hwhi
  have hsq := weighted_mean_square_le p (fun a => |w a / tiltMean p w - 1|) hp0 hp
  simp only [sq_abs] at hsq
  have hsum : (∑ a, |tiltRow p w a - p a|) =
      ∑ a, p a * |w a / tiltMean p w - 1| := by
    apply Finset.sum_congr rfl
    intro a _
    have hid : tiltRow p w a - p a = p a * (w a / tiltMean p w - 1) := by
      unfold tiltRow
      ring
    rw [hid, abs_mul, abs_of_nonneg (hp0 a)]
  rw [hsum]
  have hb : (∑ a, p a * |w a / tiltMean p w - 1|) ≤ Real.sqrt (tiltCap lo hi) :=
    (Real.le_sqrt (by exact Finset.sum_nonneg fun a _ => mul_nonneg (hp0 a) (abs_nonneg _))
      (le_trans (Finset.sum_nonneg fun a _ => mul_nonneg (hp0 a) (sq_nonneg _)) he)).mpr
      (hsq.trans he)
  linarith

theorem tilt_relative_deviation (p w : A → ℝ) (hp0 : ∀ a, 0 ≤ p a)
    (hp : ∑ a, p a = 1) (lo hi : ℝ) (hlo : 0 < lo)
    (hwlo : ∀ a, lo ≤ w a) (hwhi : ∀ a, w a ≤ hi) (a : A) :
    |w a / tiltMean p w - 1| ≤ (hi - lo) / lo := by
  obtain ⟨hlZ, hZh⟩ := tilt_mean_bounds p w hp0 hp lo hi hwlo hwhi
  have hZ := lt_of_lt_of_le hlo hlZ
  have hdiff : |w a - tiltMean p w| ≤ hi - lo := abs_le.mpr ⟨by linarith [hwlo a], by linarith [hwhi a]⟩
  calc
    _ = |w a - tiltMean p w| / tiltMean p w := by
      rw [div_sub_one (ne_of_gt hZ), abs_div, abs_of_pos hZ]
    _ ≤ (hi - lo) / tiltMean p w := div_le_div_of_nonneg_right hdiff hZ.le
    _ ≤ _ := div_le_div_of_nonneg_left (by linarith) hlo hlZ

def softmaxRow (z : A → ℝ) (a : A) : ℝ := Real.exp (z a) / ∑ b, Real.exp (z b)
def logitCap (s : ℝ) : ℝ := (Real.exp s - 1) ^ 2 / (4 * Real.exp s)

theorem softmax_sum_pos [Nonempty A] (z : A → ℝ) : 0 < ∑ a, Real.exp (z a) :=
  Finset.sum_pos (fun a _ => Real.exp_pos _) Finset.univ_nonempty

theorem softmax_tilt [Nonempty A] (z z' : A → ℝ) (a : A) :
    softmaxRow z' a = tiltRow (softmaxRow z) (fun b => Real.exp (z' b - z b)) a := by
  have hbase := ne_of_gt (softmax_sum_pos z)
  have hnew := ne_of_gt (softmax_sum_pos z')
  have hterm : ∀ b, softmaxRow z b * Real.exp (z' b - z b) =
      Real.exp (z' b) / (∑ c, Real.exp (z c)) := by
    intro b
    unfold softmaxRow
    rw [Real.exp_sub]
    field_simp
    ring
  have hZ : tiltMean (softmaxRow z) (fun b => Real.exp (z' b - z b)) =
      (∑ b, Real.exp (z' b)) / (∑ b, Real.exp (z b)) := by
    simp only [tiltMean, hterm, Finset.sum_div]
  unfold tiltRow
  rw [hterm, hZ]
  unfold softmaxRow
  field_simp

theorem tilt_cap_exp (lo hi : ℝ) : tiltCap (Real.exp lo) (Real.exp hi) = logitCap (hi - lo) := by
  unfold tiltCap logitCap
  rw [Real.exp_sub]
  field_simp
  ring

theorem tilt_row_scale (p w : A → ℝ) (c : ℝ) (hc : c ≠ 0) (a : A) :
    tiltRow p (fun b => c * w b) a = tiltRow p w a := by
  have hmean : tiltMean p (fun b => c * w b) = c * tiltMean p w := by
    unfold tiltMean
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro b _
    ring
  unfold tiltRow
  rw [hmean, show p a * (c * w a) = c * (p a * w a) by ring]
  exact mul_div_mul_left _ _ hc

theorem softmax_centered_tilt [Nonempty A] (z z' : A → ℝ) (c : ℝ) (a : A) :
    softmaxRow z' a = tiltRow (softmaxRow z) (fun b => Real.exp (z' b - z b - c)) a := by
  have hw : (fun b => Real.exp (z' b - z b - c)) =
      (fun b => Real.exp (-c) * Real.exp (z' b - z b)) := by
    funext b
    rw [← Real.exp_add]
    congr 1
    ring
  rw [hw, tilt_row_scale _ _ _ (Real.exp_ne_zero _)]
  exact softmax_tilt z z' a

/-- The energy bound is attained by a two-token distribution. -/
theorem tilt_energy_bound_sharp (lo hi : ℝ) (hlo : 0 < lo) (hhi : 0 < hi) :
    tiltEnergy (fun b : Bool => if b then lo / (lo + hi) else hi / (lo + hi))
      (fun b : Bool => if b then hi else lo) = tiltCap lo hi := by
  have hsum : lo + hi ≠ 0 := ne_of_gt (by positivity : 0 < lo + hi)
  have hp : hi * lo + lo * hi ≠ 0 := ne_of_gt (by positivity : 0 < hi * lo + lo * hi)
  simp only [tiltEnergy, tiltMean, Fintype.sum_bool, Bool.false_eq_true, reduceIte, tiltCap]
  field_simp
  <;> ring

end
end REINFORCEProMax.LogitBounds

namespace REINFORCEProMax.AutoregressiveTree
open scoped BigOperators
open LogitBounds
noncomputable section
variable {A : Type*} [Fintype A] [DecidableEq A]

theorem policy_tilt_energy (p q : Policy A) (h : List A) (w : A → ℝ)
    (heq : ∀ a, q.prob h a = tiltRow (p.prob h) w a) :
    localRatioEnergy p q h = tiltEnergy (p.prob h) w := by
  apply Finset.sum_congr rfl
  intro a _
  by_cases hp : p.prob h a = 0
  · simp [hp]
  · have hr : tokenRatio p q h a = w a / tiltMean (p.prob h) w := by
      unfold tokenRatio
      rw [heq]
      unfold tiltRow
      rw [div_right_comm, mul_div_cancel_left₀ _ hp]
    rw [hr]

theorem policy_tilt_kl (p q : Policy A) (h : List A) (w : A → ℝ)
    (hZ : tiltMean (p.prob h) w ≠ 0) (hw : ∀ a, w a ≠ 0)
    (heq : ∀ a, q.prob h a = tiltRow (p.prob h) w a) :
    localReverseKL p q h = tiltKL (p.prob h) w := by
  apply Finset.sum_congr rfl
  intro a _
  by_cases hp : p.prob h a = 0
  · simp [hp]
  · have hr : tokenRatio q p h a = tiltMean (p.prob h) w / w a := by
      unfold tokenRatio
      rw [heq]
      unfold tiltRow
      field_simp [hp, hw a, hZ]
      ring
    rw [hr]

/-- All three certificate inputs, on the actual normalized policy pair. -/
theorem policy_tilt_local_bounds (p q : Policy A) (h : List A) (w : A → ℝ)
    (lo hi : ℝ) (hlo : 0 < lo) (hwlo : ∀ a, lo ≤ w a) (hwhi : ∀ a, w a ≤ hi)
    (heq : ∀ a, q.prob h a = tiltRow (p.prob h) w a) :
    localRatioEnergy p q h ≤ tiltCap lo hi ∧ localReverseKL p q h ≤ tiltCap lo hi ∧
      localTV p q h ≤ Real.sqrt (tiltCap lo hi) / 2 := by
  have hZ : 0 < tiltMean (p.prob h) w := lt_of_lt_of_le hlo
    (tilt_mean_bounds _ _ (p.nonneg h) (p.mass h) lo hi hwlo hwhi).1
  refine ⟨?_, ?_, ?_⟩
  · rw [policy_tilt_energy p q h w heq]
    exact tilt_energy_bound _ _ (p.nonneg h) (p.mass h) lo hi hlo hwlo hwhi
  · rw [policy_tilt_kl p q h w (ne_of_gt hZ) (fun a => ne_of_gt (lt_of_lt_of_le hlo (hwlo a))) heq]
    exact tilt_kl_bound _ _ (p.nonneg h) (p.mass h) lo hi hlo hwlo hwhi
  · simpa only [localTV, heq] using tilt_tv_bound _ _ (p.nonneg h) (p.mass h) lo hi hlo hwlo hwhi

theorem policy_softmax_local_bounds [Nonempty A] (p q : Policy A) (h : List A)
    (z z' : A → ℝ) (hp : ∀ a, p.prob h a = softmaxRow z a)
    (hq : ∀ a, q.prob h a = softmaxRow z' a)
    (lo hi : ℝ) (hl : ∀ a, lo ≤ z' a - z a) (hu : ∀ a, z' a - z a ≤ hi) :
    localRatioEnergy p q h ≤ logitCap (hi - lo) ∧ localReverseKL p q h ≤ logitCap (hi - lo) ∧
      localTV p q h ≤ Real.sqrt (logitCap (hi - lo)) / 2 := by
  have heq : ∀ a, q.prob h a = tiltRow (p.prob h) (fun b => Real.exp (z' b - z b)) a := by
    intro a
    simp only [hq, show p.prob h = softmaxRow z from funext hp]
    exact softmax_tilt z z' a
  simpa only [tilt_cap_exp] using policy_tilt_local_bounds p q h (fun b => Real.exp (z' b - z b))
    (Real.exp lo) (Real.exp hi) (Real.exp_pos lo)
    (fun a => Real.exp_le_exp.mpr (hl a)) (fun a => Real.exp_le_exp.mpr (hu a)) heq

/-- A context-dependent common shift turns logit oscillation into a weight
envelope; this does not change the normalized softmax policy. -/
theorem policy_softmax_envelope [Nonempty A] (p q : Policy A) (h : List A)
    (z z' : A → ℝ) (c ω : ℝ)
    (hp : ∀ a, p.prob h a = softmaxRow z a)
    (hq : ∀ a, q.prob h a = softmaxRow z' a)
    (hl : ∀ a, 0 ≤ z' a - z a - c) (hu : ∀ a, z' a - z a - c ≤ ω) :
    (∀ a, 1 ≤ Real.exp (z' a - z a - c)) ∧
    (∀ a, Real.exp (z' a - z a - c) ≤ Real.exp ω) ∧
    (∀ a, q.prob h a = tiltRow (p.prob h) (fun b => Real.exp (z' b - z b - c)) a) := by
  refine ⟨fun a => ?_, fun a => Real.exp_le_exp.mpr (hu a), fun a => ?_⟩
  · simpa only [Real.exp_zero] using Real.exp_le_exp.mpr (hl a)
  · simp only [hq, show p.prob h = softmaxRow z from funext hp]
    exact softmax_centered_tilt z z' c a

/-- A ratio cover can be transferred from a logit/tilt cover without a
lower bound on the rollout probability of a rare token. -/
theorem token_ratio_cover_of_tilt (p q q' : Policy A) (h : List A) (w : A → ℝ)
    (lo hi C : ℝ) (hlo : 0 < lo) (hC : 0 ≤ C)
    (hwlo : ∀ a, lo ≤ w a) (hwhi : ∀ a, w a ≤ hi)
    (heq : ∀ a, q.prob h a = tiltRow (q'.prob h) w a)
    (hanchor : ∀ a, tokenRatio p q' h a ≤ C) (a : A) :
    |tokenRatio p q h a - tokenRatio p q' h a| ≤ C * ((hi - lo) / lo) := by
  have hnonneg : 0 ≤ tokenRatio p q' h a := div_nonneg (q'.nonneg h a) (p.nonneg h a)
  have hid : tokenRatio p q h a - tokenRatio p q' h a =
      tokenRatio p q' h a * (w a / tiltMean (q'.prob h) w - 1) := by
    unfold tokenRatio
    rw [heq]
    unfold tiltRow
    ring
  rw [hid, abs_mul, abs_of_nonneg hnonneg]
  exact mul_le_mul (hanchor a)
    (tilt_relative_deviation _ _ (q'.nonneg h) (q'.mass h) lo hi hlo hwlo hwhi a)
    (abs_nonneg _) hC

theorem token_ratio_upper_of_tilt (p q : Policy A) (h : List A) (w : A → ℝ)
    (lo hi : ℝ) (hlo : 0 < lo) (hwlo : ∀ a, lo ≤ w a) (hwhi : ∀ a, w a ≤ hi)
    (heq : ∀ a, q.prob h a = tiltRow (p.prob h) w a) (a : A) :
    tokenRatio p q h a ≤ hi / lo := by
  obtain ⟨hlZ, hZh⟩ := tilt_mean_bounds _ _ (p.nonneg h) (p.mass h) lo hi hwlo hwhi
  have hZ := lt_of_lt_of_le hlo hlZ
  have hhi := lt_of_lt_of_le hZ hZh
  by_cases hp : p.prob h a = 0
  · simp only [tokenRatio, hp, div_zero]
    positivity
  · have hr : tokenRatio p q h a = w a / tiltMean (p.prob h) w := by
      unfold tokenRatio
      rw [heq]
      unfold tiltRow
      rw [div_right_comm, mul_div_cancel_left₀ _ hp]
    rw [hr]
    exact (div_le_div_of_nonneg_right (hwhi a) hZ.le).trans
      (div_le_div_of_nonneg_left hhi.le hlo hlZ)

/-- The complete token-cover to response-cover bridge for two consecutive
normalized tilts, useful for a fixed cover in logit coordinates. -/
theorem response_ratio_cover_of_tilts (p q q' : Policy A)
    (w0 w1 : List A → A → ℝ) (lo0 hi0 lo1 hi1 : ℝ)
    (hl0 : 0 < lo0) (hh0 : 0 ≤ hi0) (hl1 : 0 < lo1) (T : ℕ)
    (hw0lo : ∀ h a, h.length < T → lo0 ≤ w0 h a)
    (hw0hi : ∀ h a, h.length < T → w0 h a ≤ hi0)
    (hw1lo : ∀ h a, h.length < T → lo1 ≤ w1 h a)
    (hw1hi : ∀ h a, h.length < T → w1 h a ≤ hi1)
    (hcenter : ∀ h a, h.length < T → q'.prob h a = tiltRow (p.prob h) (w0 h) a)
    (hquery : ∀ h a, h.length < T → q.prob h a = tiltRow (q'.prob h) (w1 h) a)
    (y : Fin T → A) :
    |responseRatioResidual p q T y - responseRatioResidual p q' T y| ≤
      (T : ℝ) * ((hi0 / lo0) * ((hi1 - lo1) / lo1)) := by
  apply response_residual_difference_of_token_cover
  intro h a hh
  exact token_ratio_cover_of_tilt p q q' h (w1 h) lo1 hi1 (hi0 / lo0) hl1
    (div_nonneg hh0 hl0.le) (fun a => hw1lo h a hh) (fun a => hw1hi h a hh)
    (fun a => hquery h a hh)
    (fun a => token_ratio_upper_of_tilt p q' h (w0 h) lo0 hi0 hl0
      (fun a => hw0lo h a hh) (fun a => hw0hi h a hh) (fun a => hcenter h a hh) a) a

/-- Explicit logit-cover radius for the additive response residual, on
every represented history, without a minimum rollout-token probability. -/
theorem response_ratio_cover_of_softmax [Nonempty A] (p q q' : Policy A)
    (z z' zq : List A → A → ℝ) (c0 c1 : List A → ℝ) (ω r : ℝ) (T : ℕ)
    (hp : ∀ h a, h.length < T → p.prob h a = softmaxRow (z h) a)
    (hc : ∀ h a, h.length < T → q'.prob h a = softmaxRow (z' h) a)
    (hq : ∀ h a, h.length < T → q.prob h a = softmaxRow (zq h) a)
    (h0lo : ∀ h a, h.length < T → 0 ≤ z' h a - z h a - c0 h)
    (h0hi : ∀ h a, h.length < T → z' h a - z h a - c0 h ≤ ω)
    (h1lo : ∀ h a, h.length < T → 0 ≤ zq h a - z' h a - c1 h)
    (h1hi : ∀ h a, h.length < T → zq h a - z' h a - c1 h ≤ r)
    (y : Fin T → A) :
    |responseRatioResidual p q T y - responseRatioResidual p q' T y| ≤
      (T : ℝ) * (Real.exp ω * (Real.exp r - 1)) := by
  have h0 := fun h hh => policy_softmax_envelope p q' h (z h) (z' h) (c0 h) ω
    (fun a => hp h a hh) (fun a => hc h a hh) (fun a => h0lo h a hh) (fun a => h0hi h a hh)
  have h1 := fun h hh => policy_softmax_envelope q' q h (z' h) (zq h) (c1 h) r
    (fun a => hc h a hh) (fun a => hq h a hh) (fun a => h1lo h a hh) (fun a => h1hi h a hh)
  simpa only [div_one] using response_ratio_cover_of_tilts p q q'
    (fun h a => Real.exp (z' h a - z h a - c0 h))
    (fun h a => Real.exp (zq h a - z' h a - c1 h))
    1 (Real.exp ω) 1 (Real.exp r) (by norm_num) (Real.exp_pos ω).le (by norm_num) T
    (fun h a hh => (h0 h hh).1 a) (fun h a hh => (h0 h hh).2.1 a)
    (fun h a hh => (h1 h hh).1 a) (fun h a hh => (h1 h hh).2.1 a)
    (fun h a hh => (h0 h hh).2.2 a) (fun h a hh => (h1 h hh).2.2 a) y

theorem policy_tilt_energy_horizon (p q : Policy A) (w : List A → A → ℝ)
    (lo hi : ℝ) (hlo : 0 < lo) (T : ℕ)
    (hwlo : ∀ h a, h.length < T → lo ≤ w h a)
    (hwhi : ∀ h a, h.length < T → w h a ≤ hi)
    (heq : ∀ h a, h.length < T → q.prob h a = tiltRow (p.prob h) (w h) a) :
    ratioEnergy p q T ≤ (T : ℝ) * tiltCap lo hi := by
  apply ratio_energy_horizon_bound
  intro h hh
  exact (policy_tilt_local_bounds p q h (w h) lo hi hlo
    (fun a => hwlo h a hh) (fun a => hwhi h a hh) (fun a => heq h a hh)).1

theorem adaptive_cost_of_local_tv (p q : Policy A) (ε δ : ℝ) (hε : 0 ≤ ε)
    (T : ℕ) (hTV : ∀ h, h.length < T → localTV p q h ≤ ε) :
    adaptiveCost p q ε δ T ≤ ε * ∑ k ∈ Finset.range T, adaptiveFutureCap ε δ (T - 1 - k) := by
  unfold adaptiveCost
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro k hk
  have hv := value_le_of_terminal_length p (localTV p q) ε k []
    (fun h hh => hTV h (by have := Finset.mem_range.mp hk; simp only [List.length_nil, zero_add] at hh; omega))
  apply mul_le_mul_of_nonneg_right hv
  unfold adaptiveFutureCap
  positivity

/-- Explicit reward remainder bound from a normalized-weight envelope.
The sharper occupancy-weighted Adaptive cost remains available. -/
theorem policy_tilt_performance_bound (p q : Policy A) (w : List A → A → ℝ)
    (R : List A → ℝ) (B lo hi : ℝ) (hB : 0 ≤ B) (hR : ∀ y, |R y| ≤ B)
    (hlo : 0 < lo) (T : ℕ) (hs : ∀ h a, p.prob h a = 0 ↔ q.prob h a = 0)
    (hwlo : ∀ h a, h.length < T → lo ≤ w h a)
    (hwhi : ∀ h a, h.length < T → w h a ≤ hi)
    (heq : ∀ h a, h.length < T → q.prob h a = tiltRow (p.prob h) (w h) a) :
    let C := tiltCap lo hi
    let ε := Real.sqrt C / 2
    surrogate p q R T [] - 4 * B * ε *
      (∑ k ∈ Finset.range T, adaptiveFutureCap ε C (T - 1 - k)) ≤
      value q R T [] - value p R T [] := by
  dsimp only
  have hlocal := fun h hh => policy_tilt_local_bounds p q h (w h) lo hi hlo
    (fun a => hwlo h a hh) (fun a => hwhi h a hh) (fun a => heq h a hh)
  have hb := performance_lower_bound_adaptive p q R B (Real.sqrt (tiltCap lo hi) / 2)
    (tiltCap lo hi) hB hR T hs (fun h hh => (hlocal h hh).2.2) (fun h hh => (hlocal h hh).2.1)
  have hc := adaptive_cost_of_local_tv p q (Real.sqrt (tiltCap lo hi) / 2) (tiltCap lo hi)
    (by positivity) T (fun h hh => (hlocal h hh).2.2)
  nlinarith [mul_le_mul_of_nonneg_left hc (show 0 ≤ 4 * B by positivity)]

section PromptFamily
variable {X : Type*} [Fintype X] [DecidableEq X]

def envelopeCost (B C : ℝ) (T : ℕ) : ℝ :=
  4 * B * (Real.sqrt C / 2) *
    ∑ k ∈ Finset.range T, adaptiveFutureCap (Real.sqrt C / 2) C (T - 1 - k)

/-- Concrete upper bounds for both unknown drift inputs to the existing
finite-sample certificate, including the finite prompt mixture. -/
theorem family_tilt_certificate_inputs (μ : X → ℝ) (p q : X → Policy A)
    (w : X → List A → A → ℝ) (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B lo hi : ℝ) (hB : 0 ≤ B) (hlo : 0 < lo) (T : ℕ)
    (hwlo : ∀ x h a, h.length < T → lo ≤ w x h a)
    (hwhi : ∀ x h a, h.length < T → w x h a ≤ hi)
    (heq : ∀ x h a, h.length < T → (q x).prob h a = tiltRow ((p x).prob h) (w x h) a) :
    familyRatioEnergy μ p q T ≤ (T : ℝ) * tiltCap lo hi ∧
      familyAdaptiveCost μ p q B (Real.sqrt (tiltCap lo hi) / 2) (tiltCap lo hi) T ≤
        envelopeCost B (tiltCap lo hi) T := by
  have hlocal := fun x h hh => policy_tilt_local_bounds (p x) (q x) h (w x h) lo hi hlo
    (fun a => hwlo x h a hh) (fun a => hwhi x h a hh) (fun a => heq x h a hh)
  constructor
  · exact family_ratio_energy_horizon_bound μ p q hμ0 hμ (tiltCap lo hi) T
      (fun x h hh => (hlocal x h hh).1)
  · have hc : ∀ x, adaptiveCost (p x) (q x) (Real.sqrt (tiltCap lo hi) / 2) (tiltCap lo hi) T ≤
        (Real.sqrt (tiltCap lo hi) / 2) *
          ∑ k ∈ Finset.range T, adaptiveFutureCap (Real.sqrt (tiltCap lo hi) / 2) (tiltCap lo hi) (T - 1 - k) :=
      fun x => adaptive_cost_of_local_tv (p x) (q x) _ _ (by positivity) T (fun h hh => (hlocal x h hh).2.2)
    have hs := Finset.sum_le_sum (s := Finset.univ) (fun x _ => mul_le_mul_of_nonneg_left (hc x) (hμ0 x))
    rw [← Finset.sum_mul, hμ, one_mul] at hs
    unfold familyAdaptiveCost envelopeCost
    nlinarith [mul_le_mul_of_nonneg_left hs (show 0 ≤ 4 * B by positivity)]

/-- The finite-prompt certificate inputs from actual softmax logits with
a uniform bound on their centered increments. -/
theorem family_softmax_certificate_inputs [Nonempty A]
    (μ : X → ℝ) (p q : X → Policy A)
    (z z' : X → List A → A → ℝ) (c : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B ω : ℝ) (hB : 0 ≤ B) (T : ℕ)
    (hp : ∀ x h a, h.length < T → (p x).prob h a = softmaxRow (z x h) a)
    (hq : ∀ x h a, h.length < T → (q x).prob h a = softmaxRow (z' x h) a)
    (hl : ∀ x h a, h.length < T → 0 ≤ z' x h a - z x h a - c x h)
    (hu : ∀ x h a, h.length < T → z' x h a - z x h a - c x h ≤ ω) :
    familyRatioEnergy μ p q T ≤ (T : ℝ) * logitCap ω ∧
      familyAdaptiveCost μ p q B (Real.sqrt (logitCap ω) / 2) (logitCap ω) T ≤
        envelopeCost B (logitCap ω) T := by
  have hb := fun x h hh => policy_softmax_envelope (p x) (q x) h (z x h) (z' x h) (c x h) ω
    (fun a => hp x h a hh) (fun a => hq x h a hh) (fun a => hl x h a hh) (fun a => hu x h a hh)
  simpa only [tiltCap, logitCap, mul_one] using family_tilt_certificate_inputs μ p q
    (fun x h a => Real.exp (z' x h a - z x h a - c x h)) hμ0 hμ B 1 (Real.exp ω)
    hB (by norm_num) T (fun x h a hh => (hb x h hh).1 a)
    (fun x h a hh => (hb x h hh).2.1 a) (fun x h a hh => (hb x h hh).2.2 a)

open BatchReduction

/-- Substitute the explicit energy envelope into the actual RLOO-group
uniform confidence bound. Candidate policies stay fixed before evaluation. -/
theorem tilt_surrogate_uniform_tail {n m T : ℕ} (hn : n ≠ 0) (hm : m ≠ 0)
    {J : Type*} [Fintype J]
    (μ : X → ℝ) (p : X → Policy A) (q : J → X → Policy A)
    (w : J → X → List A → A → ℝ) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B : ℝ) (lo hi η : J → ℝ) (hB : 0 ≤ B) (hlo : ∀ j, 0 < lo j) (hη : ∀ j, 0 < η j)
    (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ j x h a, (p x).prob h a = 0 → (q j x).prob h a = 0)
    (hwlo : ∀ j x h a, h.length < T → lo j ≤ w j x h a)
    (hwhi : ∀ j x h a, h.length < T → w j x h a ≤ hi j)
    (heq : ∀ j x h a, h.length < T → (q j x).prob h a = tiltRow ((p x).prob h) (w j x h) a) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → RolloutBlock X A m T => ∃ j,
      η j ≤ |sampleMean (blockSurrogate p (q j) R) s - familySurrogate μ p (q j) R T|),
        iidMass (blockMass μ p m T) s) ≤
      ∑ j, (4 * B ^ 2 * ((T : ℝ) * tiltCap (lo j) (hi j))) / ((n : ℝ) * (η j) ^ 2) := by
  classical
  apply (rescaled_surrogate_uniform_tail hn hm μ p q R hμ0 hμ B η hB hη hR hs).trans
  apply Finset.sum_le_sum
  intro j _
  apply div_le_div_of_nonneg_right _ (by positivity)
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  exact (family_tilt_certificate_inputs μ p (q j) (w j) hμ0 hμ B (lo j) (hi j)
    hB (hlo j) T (hwlo j) (hwhi j) (heq j)).1

/-- The same deterministic envelope gives the reward bound from the
observed, exactly corrected ProMax statistic. -/
theorem tilt_performance_from_estimate (μ : X → ℝ) (p q : X → Policy A)
    (w : X → List A → A → ℝ) (R : X → List A → ℝ)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (B lo hi η estimate : ℝ) (hB : 0 ≤ B) (hlo : 0 < lo) (T : ℕ)
    (hR : ∀ x y, |R x y| ≤ B)
    (hs : ∀ x h a, (p x).prob h a = 0 ↔ (q x).prob h a = 0)
    (hwlo : ∀ x h a, h.length < T → lo ≤ w x h a)
    (hwhi : ∀ x h a, h.length < T → w x h a ≤ hi)
    (heq : ∀ x h a, h.length < T → (q x).prob h a = tiltRow ((p x).prob h) (w x h) a)
    (hest : |estimate - familySurrogate μ p q R T| ≤ η) :
    estimate - η - envelopeCost B (tiltCap lo hi) T ≤
      familyValue μ q R T - familyValue μ p R T := by
  have hlocal := fun x h hh => policy_tilt_local_bounds (p x) (q x) h (w x h) lo hi hlo
    (fun a => hwlo x h a hh) (fun a => hwhi x h a hh) (fun a => heq x h a hh)
  have hb := rescaled_performance_from_estimate μ p q R T hμ0 B (Real.sqrt (tiltCap lo hi) / 2)
    (tiltCap lo hi) η estimate hB hR hs (fun x h hh => (hlocal x h hh).2.2)
    (fun x h hh => (hlocal x h hh).2.1) hest
  have hc := (family_tilt_certificate_inputs μ p q w hμ0 hμ B lo hi hB hlo T hwlo hwhi heq).2
  linarith

end PromptFamily

end
end REINFORCEProMax.AutoregressiveTree
