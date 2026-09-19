import CoarseGrain
import KLEquality

/-! KL contraction and its full equality condition for finite partitions.
The first measure may vanish inside a positive-mass cell. Only p << q is
required; zero-p cells impose no condition on the second measure. -/
namespace REINFORCEProMax

noncomputable def gibbsGap (p q : ℝ) : ℝ := p * Real.log (p / q) - p + q

theorem gibbsGap_nonneg {p q : ℝ} (hp : 0 ≤ p) (hq : 0 ≤ q)
    (hs : 0 < p → 0 < q) : 0 ≤ gibbsGap p q := by
  by_cases hz : p = 0
  · simpa [gibbsGap, hz] using hq
  have hp' : 0 < p := lt_of_le_of_ne hp (Ne.symm hz)
  have hq' := hs hp'
  have hlog := Real.log_le_sub_one_of_pos (div_pos hq' hp')
  have hmul := mul_le_mul_of_nonneg_left hlog hp
  have hinv : Real.log (p / q) = -Real.log (q / p) := by
    rw [Real.log_div hz (ne_of_gt hq'), Real.log_div (ne_of_gt hq') hz]
    ring
  have hcancel : p * (q / p - 1) = q - p := by field_simp
  rw [hcancel] at hmul
  dsimp [gibbsGap]
  rw [hinv]
  linarith

theorem gibbsGap_eq_zero_iff {p q : ℝ} (hp : 0 ≤ p) (_hq : 0 ≤ q)
    (hs : 0 < p → 0 < q) : gibbsGap p q = 0 ↔ p = q := by
  by_cases hz : p = 0
  · simp [gibbsGap, hz, eq_comm]
  have hp' : 0 < p := lt_of_le_of_ne hp (Ne.symm hz)
  have hq' := hs hp'
  constructor
  · intro hg
    by_contra hne
    have hratio : q / p ≠ 1 := by
      intro hr
      have := (div_eq_one_iff_eq hz).mp hr
      exact hne this.symm
    have hlog := Real.log_lt_sub_one_of_pos (div_pos hq' hp') hratio
    have hmul := mul_lt_mul_of_pos_left hlog hp'
    have hinv : Real.log (p / q) = -Real.log (q / p) := by
      rw [Real.log_div hz (ne_of_gt hq'), Real.log_div (ne_of_gt hq') hz]
      ring
    have hcancel : p * (q / p - 1) = q - p := by field_simp
    rw [hcancel] at hmul
    dsimp [gibbsGap] at hg
    rw [hinv] at hg
    linarith
  · intro heq
    simp [gibbsGap, heq, ne_of_gt hq']

theorem cell_mass_support {I : Type*} (S : Finset I) (p q : I → ℝ)
    (hp : ∀ i ∈ S, 0 ≤ p i) (hq : ∀ i ∈ S, 0 ≤ q i)
    (hs : ∀ i ∈ S, 0 < p i → 0 < q i)
    (hP : 0 < ∑ i ∈ S, p i) : 0 < ∑ i ∈ S, q i := by
  by_contra hn
  have hzero := (Finset.sum_eq_zero_iff_of_nonneg hq).mp (le_antisymm (le_of_not_gt hn) (Finset.sum_nonneg hq))
  have hpzero : ∀ i ∈ S, p i = 0 := by
    intro i hi
    by_contra hpi
    have := hs i hi (lt_of_le_of_ne (hp i hi) (Ne.symm hpi))
    rw [hzero i hi] at this
    exact lt_irrefl _ this
  have := Finset.sum_eq_zero hpzero
  linarith

theorem cell_gibbsGap_identity {I : Type*} (S : Finset I) (p q : I → ℝ)
    (hp : ∀ i ∈ S, 0 ≤ p i) (hq : ∀ i ∈ S, 0 ≤ q i)
    (hs : ∀ i ∈ S, 0 < p i → 0 < q i)
    (hP : 0 < ∑ i ∈ S, p i) :
    (∑ i ∈ S, gibbsGap (p i) (((∑ i ∈ S, p i) / (∑ i ∈ S, q i)) * q i)) =
      (∑ i ∈ S, p i * Real.log (p i / q i)) -
        (∑ i ∈ S, p i) * Real.log ((∑ i ∈ S, p i) / (∑ i ∈ S, q i)) := by
  let P := ∑ i ∈ S, p i
  let Q := ∑ i ∈ S, q i
  have hQ : 0 < Q := cell_mass_support S p q hp hq hs hP
  have hscale : 0 < P / Q := div_pos hP hQ
  have hpoint : ∀ i ∈ S, gibbsGap (p i) ((P / Q) * q i) =
      p i * Real.log (p i / q i) - p i * Real.log (P / Q) - p i + (P / Q) * q i := by
    intro i hi
    by_cases hz : p i = 0
    · simp [gibbsGap, hz]
    have hpi : 0 < p i := lt_of_le_of_ne (hp i hi) (Ne.symm hz)
    have hqi := hs i hi hpi
    dsimp [gibbsGap]
    rw [Real.log_div hz (ne_of_gt (mul_pos hscale hqi)),
      Real.log_mul (ne_of_gt hscale) (ne_of_gt hqi), Real.log_div hz (ne_of_gt hqi)]
    ring
  change (∑ i ∈ S, gibbsGap (p i) ((P / Q) * q i)) = _ - P * Real.log (P / Q)
  rw [Finset.sum_congr rfl hpoint]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
    ← Finset.sum_mul, ← Finset.mul_sum]
  change _ - P * Real.log (P / Q) - P + (P / Q) * Q = _
  rw [div_mul_cancel₀ _ (ne_of_gt hQ)]
  ring

/-- The exact log-sum gap also covers cells with zero first-measure mass.
It quantifies the deviation from proportionality inside every cell. -/
theorem finite_cell_kl_gap_identity {I : Type*} (S : Finset I) (p q : I → ℝ)
    (hp : ∀ i ∈ S, 0 ≤ p i) (hq : ∀ i ∈ S, 0 ≤ q i)
    (hs : ∀ i ∈ S, 0 < p i → 0 < q i) :
    (∑ i ∈ S, p i * Real.log (p i / q i)) -
        (∑ i ∈ S, p i) * Real.log ((∑ i ∈ S, p i) / (∑ i ∈ S, q i)) =
      ∑ i ∈ S, gibbsGap (p i) (((∑ i ∈ S, p i) / (∑ i ∈ S, q i)) * q i) := by
  by_cases hz : (∑ i ∈ S, p i) = 0
  · have hzero := (Finset.sum_eq_zero_iff_of_nonneg hp).mp hz
    simp only [hz, zero_div, zero_mul, sub_zero]
    apply Finset.sum_congr rfl
    intro i hi
    simp [hzero i hi, gibbsGap]
  exact (cell_gibbsGap_identity S p q hp hq hs
    (lt_of_le_of_ne (Finset.sum_nonneg hp) (Ne.symm hz))).symm

/-- Full KL minus coarse KL is exactly the sum of nonnegative within-cell
Gibbs gaps, rather than merely an unspecified approximation error. -/
theorem finite_coarse_kl_gap_identity {n m : ℕ} (p q : Fin n → ℝ)
    (cell : Fin n → Fin m) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hs : ∀ i, 0 < p i → 0 < q i) :
    (∑ i : Fin n, p i * Real.log (p i / q i)) -
        (∑ j : Fin m, fiberSum cell p j * Real.log (fiberSum cell p j / fiberSum cell q j)) =
      ∑ j : Fin m, fiberSum cell
        (fun i => gibbsGap (p i) ((fiberSum cell p j / fiberSum cell q j) * q i)) j := by
  have hsum : (∑ j : Fin m, fiberSum cell (fun i => p i * Real.log (p i / q i)) j) =
      ∑ i : Fin n, p i * Real.log (p i / q i) := by
    simpa [fiberSum] using (Finset.sum_fiberwise (s := (Finset.univ : Finset (Fin n)))
      cell (fun i => p i * Real.log (p i / q i)))
  rw [← hsum, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  exact finite_cell_kl_gap_identity _ p q
    (fun i _ => hp i) (fun i _ => hq i) (fun i _ => hs i)

theorem finite_cell_kl_contraction {I : Type*} (S : Finset I) (p q : I → ℝ)
    (hp : ∀ i ∈ S, 0 ≤ p i) (hq : ∀ i ∈ S, 0 ≤ q i)
    (hs : ∀ i ∈ S, 0 < p i → 0 < q i) :
    (∑ i ∈ S, p i) * Real.log ((∑ i ∈ S, p i) / (∑ i ∈ S, q i)) ≤
      ∑ i ∈ S, p i * Real.log (p i / q i) := by
  have hP := Finset.sum_nonneg hp
  by_cases hz : (∑ i ∈ S, p i) = 0
  · have hzero := (Finset.sum_eq_zero_iff_of_nonneg hp).mp hz
    rw [hz, zero_mul]
    apply Finset.sum_nonneg
    intro i hi
    simp [hzero i hi]
  have hP' : 0 < ∑ i ∈ S, p i := lt_of_le_of_ne hP (Ne.symm hz)
  have hQ := cell_mass_support S p q hp hq hs hP'
  have hscale := div_pos hP' hQ
  have hgap : 0 ≤ ∑ i ∈ S,
      gibbsGap (p i) (((∑ i ∈ S, p i) / (∑ i ∈ S, q i)) * q i) := by
    apply Finset.sum_nonneg
    intro i hi
    exact gibbsGap_nonneg (hp i hi) (mul_nonneg (le_of_lt hscale) (hq i hi))
      (fun hpi => mul_pos hscale (hs i hi hpi))
  rw [cell_gibbsGap_identity S p q hp hq hs hP'] at hgap
  linarith

theorem finite_cell_kl_equality_iff {I : Type*} (S : Finset I) (p q : I → ℝ)
    (hp : ∀ i ∈ S, 0 ≤ p i) (hq : ∀ i ∈ S, 0 ≤ q i)
    (hs : ∀ i ∈ S, 0 < p i → 0 < q i) :
    (∑ i ∈ S, p i) * Real.log ((∑ i ∈ S, p i) / (∑ i ∈ S, q i)) =
      (∑ i ∈ S, p i * Real.log (p i / q i)) ↔
    (0 < ∑ i ∈ S, p i → ∀ i ∈ S,
      p i = ((∑ i ∈ S, p i) / (∑ i ∈ S, q i)) * q i) := by
  have hP := Finset.sum_nonneg hp
  by_cases hz : (∑ i ∈ S, p i) = 0
  · have hzero := (Finset.sum_eq_zero_iff_of_nonneg hp).mp hz
    have hkl : (∑ i ∈ S, p i * Real.log (p i / q i)) = 0 := by
      apply Finset.sum_eq_zero
      intro i hi
      simp [hzero i hi]
    simp [hz, hkl]
  have hP' : 0 < ∑ i ∈ S, p i := lt_of_le_of_ne hP (Ne.symm hz)
  have hQ := cell_mass_support S p q hp hq hs hP'
  have hscale := div_pos hP' hQ
  have hg : ∀ i ∈ S, 0 ≤ gibbsGap (p i)
      (((∑ i ∈ S, p i) / (∑ i ∈ S, q i)) * q i) := by
    intro i hi
    exact gibbsGap_nonneg (hp i hi) (mul_nonneg (le_of_lt hscale) (hq i hi))
      (fun hpi => mul_pos hscale (hs i hi hpi))
  rw [show (_ = _) ↔ (∑ i ∈ S,
      gibbsGap (p i) (((∑ i ∈ S, p i) / (∑ i ∈ S, q i)) * q i)) = 0 by
    rw [cell_gibbsGap_identity S p q hp hq hs hP']; constructor <;> intro h <;> linarith]
  rw [Finset.sum_eq_zero_iff_of_nonneg hg]
  simp only [hP', true_implies]
  apply forall₂_congr
  intro i hi
  exact gibbsGap_eq_zero_iff (hp i hi) (mul_nonneg (le_of_lt hscale) (hq i hi))
    (fun hpi => mul_pos hscale (hs i hi hpi))

theorem finite_coarse_kl_contraction {n m : ℕ} (p q : Fin n → ℝ)
    (cell : Fin n → Fin m) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hs : ∀ i, 0 < p i → 0 < q i) :
    (∑ j : Fin m, fiberSum cell p j * Real.log (fiberSum cell p j / fiberSum cell q j)) ≤
      ∑ i : Fin n, p i * Real.log (p i / q i) := by
  calc
    _ ≤ ∑ j : Fin m, fiberSum cell (fun i => p i * Real.log (p i / q i)) j := by
      apply Finset.sum_le_sum
      intro j _
      exact finite_cell_kl_contraction _ p q (fun i _ => hp i)
        (fun i _ => hq i) (fun i _ => hs i)
    _ = _ := by
      simpa [fiberSum] using (Finset.sum_fiberwise (s := (Finset.univ : Finset (Fin n)))
        cell (fun i => p i * Real.log (p i / q i)))

theorem finite_coarse_kl_equality_iff {n m : ℕ} (p q : Fin n → ℝ)
    (cell : Fin n → Fin m) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hs : ∀ i, 0 < p i → 0 < q i) :
    (∑ j : Fin m, fiberSum cell p j * Real.log (fiberSum cell p j / fiberSum cell q j)) =
      (∑ i : Fin n, p i * Real.log (p i / q i)) ↔
    (∀ j, 0 < fiberSum cell p j → ∀ i, cell i = j →
      p i = (fiberSum cell p j / fiberSum cell q j) * q i) := by
  let f := fun j => fiberSum cell (fun i => p i * Real.log (p i / q i)) j
  let g := fun j => fiberSum cell p j * Real.log (fiberSum cell p j / fiberSum cell q j)
  have hsum : (∑ j : Fin m, f j) = ∑ i : Fin n, p i * Real.log (p i / q i) := by
    simpa [f, fiberSum] using (Finset.sum_fiberwise (s := (Finset.univ : Finset (Fin n)))
      cell (fun i => p i * Real.log (p i / q i)))
  have hgap : ∀ j ∈ (Finset.univ : Finset (Fin m)), 0 ≤ f j - g j := by
    intro j _
    exact sub_nonneg.mpr (finite_cell_kl_contraction _ p q
      (fun i _ => hp i) (fun i _ => hq i) (fun i _ => hs i))
  have heq : (∑ j : Fin m, g j) = (∑ j : Fin m, f j) ↔ ∀ j, g j = f j := by
    rw [← sub_eq_zero, ← Finset.sum_sub_distrib]
    constructor
    · intro h
      have hzero : (∑ j : Fin m, (f j - g j)) = 0 := by
        rw [Finset.sum_sub_distrib] at h ⊢
        linarith
      have hz := (Finset.sum_eq_zero_iff_of_nonneg hgap).mp hzero
      intro j
      have := hz j (Finset.mem_univ _)
      linarith
    · intro h
      apply Finset.sum_eq_zero
      intro j _
      rw [h j, sub_self]
  rw [← hsum]
  change (∑ j : Fin m, g j) = (∑ j : Fin m, f j) ↔ _
  rw [heq]
  apply forall_congr'
  intro j
  have hc := finite_cell_kl_equality_iff
    (Finset.univ.filter (fun i : Fin n => cell i = j)) p q
    (fun i _ => hp i) (fun i _ => hq i) (fun i _ => hs i)
  simpa [f, g, fiberSum] using hc

theorem finite_coarse_kl_equality_iff_proportional {n m : ℕ} (p q : Fin n → ℝ)
    (cell : Fin n → Fin m) (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hs : ∀ i, 0 < p i → 0 < q i) :
    (∑ j : Fin m, fiberSum cell p j * Real.log (fiberSum cell p j / fiberSum cell q j)) =
      (∑ i : Fin n, p i * Real.log (p i / q i)) ↔
    (∀ j, 0 < fiberSum cell p j → ∃ c : ℝ, 0 < c ∧
      ∀ i, cell i = j → p i = c * q i) := by
  rw [finite_coarse_kl_equality_iff p q cell hp hq hs]
  apply forall_congr'
  intro j
  apply forall_congr'
  intro hP
  have hQ : 0 < fiberSum cell q j := cell_mass_support _ p q
    (fun i _ => hp i) (fun i _ => hq i) (fun i _ => hs i) hP
  constructor
  · intro h
    exact ⟨_, div_pos hP hQ, h⟩
  · rintro ⟨c, _, hc⟩
    have hsum : fiberSum cell p j = c * fiberSum cell q j := by
      dsimp [fiberSum]
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i hi
      exact hc i (Finset.mem_filter.mp hi).2
    have hratio : fiberSum cell p j / fiberSum cell q j = c := by
      rw [hsum, mul_div_cancel_right₀ _ (ne_of_gt hQ)]
    simpa [hratio] using hc

end REINFORCEProMax
