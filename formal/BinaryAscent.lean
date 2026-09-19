import BinaryLength
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Calculus.Deriv.Slope

/-!
Local true-reward improvement for the population binary ProMax update.
The response coefficient includes arbitrary content-dependent lengths,
implemented normalization safeguards and optional uniform scale. The on-policy,
accepted-gate, inactive-clipping regime is inherited from BinaryLength.
The covariance margin is derived on the same finite differentiable policy;
neither the gradient identity nor reward improvement is an extra premise.
-/
namespace REINFORCEProMax.BinaryAscent
open scoped BigOperators Topology
open BatchReduction UniformScale BinaryUpdate BinaryLength InnerProductSpace
noncomputable section
variable {A E : Type*} [Fintype A] [DecidableEq A]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

def successDifferential (p : E → A → ℝ) (b : A → Bool) (θ : E)
    (D : A → E →L[ℝ] ℝ) : E →L[ℝ] ℝ :=
  ∑ a, restrictedMass (p θ) b a • policyScore p θ D a

def successGradient (p : E → A → ℝ) (b : A → Bool) (θ : E)
    (D : A → E →L[ℝ] ℝ) : E :=
  (toDual ℝ E).symm (successDifferential p b θ D)

def updateDifferential {m : ℕ} (p : A → ℝ) (S : A → E →L[ℝ] ℝ)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) : E →L[ℝ] ℝ :=
  ∑ y : Fin (m + 1) → A, iidMass p y • ∑ i, C y i • S (y i)

def populationUpdate {m : ℕ} (p : A → ℝ) (S : A → E →L[ℝ] ℝ)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) : E :=
  (toDual ℝ E).symm (updateDifferential p S C)

omit [DecidableEq A] [CompleteSpace E] in
theorem update_differential_apply {m : ℕ} (p : A → ℝ) (S : A → E →L[ℝ] ℝ)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) (v : E) :
    updateDifferential p S C v =
      ∑ y : Fin (m + 1) → A, iidMass p y * ∑ i, C y i * S (y i) v := by
  simp [updateDifferential, ContinuousLinearMap.sum_apply]

omit [DecidableEq A] in
/-- The Riesz representation is the expected sum of response score vectors. -/
theorem population_update_eq_expected_score {m : ℕ} (p : A → ℝ)
    (S : A → E →L[ℝ] ℝ)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) :
    populationUpdate p S C =
      ∑ y : Fin (m + 1) → A, iidMass p y •
        ∑ i, C y i • (toDual ℝ E).symm (S (y i)) := by
  apply ext_inner_right ℝ
  intro v
  simp [populationUpdate, toDual_symm_apply, sum_inner,
    real_inner_smul_left, update_differential_apply]

theorem success_gradient_correct (p : E → A → ℝ) (b : A → Bool) (θ : E)
    (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a)
    (hnorm : ∀ᶠ η in 𝓝 θ, ∑ a, p η a = 1) :
    HasGradientAt (fun η => successMass (p η) b) (successGradient p b θ D) θ :=
  (finite_policy_uniform_correction p b θ D 0 1 0 hd hpos hnorm).1.hasGradientAt

omit [DecidableEq A] in
/-- Symmetry of the real inner product connects the score projection along
the true gradient to the derivative along the actual expected update. -/
theorem ascent_pairing {m : ℕ} (p : E → A → ℝ) (b : A → Bool) (θ : E)
    (D : A → E →L[ℝ] ℝ)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) :
    successDifferential p b θ D (populationUpdate (p θ) (policyScore p θ D) C) =
      updateDifferential (p θ) (policyScore p θ D) C (successGradient p b θ D) := by
  unfold populationUpdate successGradient
  rw [← toDual_symm_apply, real_inner_comm, toDual_symm_apply]

omit [DecidableEq A] in
theorem success_score_gradient_norm (p : E → A → ℝ) (b : A → Bool) (θ : E)
    (D : A → E →L[ℝ] ℝ) :
    successScore (p θ) (fun a => policyScore p θ D a (successGradient p b θ D)) b =
      ‖successGradient p b θ D‖ ^ 2 := by
  have h := toDual_symm_apply (𝕜 := ℝ) (E := E)
    (x := successGradient p b θ D) (y := successDifferential p b θ D)
  change ⟪successGradient p b θ D, successGradient p b θ D⟫_ℝ = _ at h
  rw [real_inner_self_eq_norm_sq] at h
  simpa [successScore, successDifferential, ContinuousLinearMap.sum_apply] using h.symm

theorem population_ascent_bound {m : ℕ} (p : E → A → ℝ) (b : A → Bool) (θ : E)
    (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a)
    (hnorm : ∀ᶠ η in 𝓝 θ, ∑ a, p η a = 1)
    (hmass : ∀ q, classMass (p θ) b q ≠ 0)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ) (d : ℝ)
    (hlower : d ≤ classMultiplier (p θ) b C) :
    d * ‖successGradient p b θ D‖ ^ 2 -
      Real.sqrt (coefficientVariance (p θ) b C * ((m + 1 : ℝ) *
        scoreVariance (p θ) (fun a => policyScore p θ D a (successGradient p b θ D)) b)) ≤
      successDifferential p b θ D (populationUpdate (p θ) (policyScore p θ D) C) := by
  let f := fun a => policyScore p θ D a (successGradient p b θ D)
  have hp0 : ∀ a, 0 ≤ p θ a := hpos.self_of_nhds
  have hp : ∑ a, p θ a = 1 := hnorm.self_of_nhds
  have hf : ∑ a, p θ a * f a = 0 := by
    simpa [f, ContinuousLinearMap.sum_apply] using
      congrArg (fun L : E →L[ℝ] ℝ => L (successGradient p b θ D))
        (finite_policy_score_mean_zero p θ D hd hpos hnorm)
  rw [ascent_pairing, update_differential_apply,
    general_update_decomposition (p θ) f b hmass hf C,
    success_score_gradient_norm]
  have hb := (abs_le.mp (covariance_error_abs_bound (p θ) f b hp0 hp C)).1
  have hm := mul_le_mul_of_nonneg_right hlower (sq_nonneg ‖successGradient p b θ D‖)
  linarith

omit [CompleteSpace E] in
/-- Differentiability alone turns a strictly positive directional derivative
into strict improvement for every sufficiently small positive step. -/
theorem local_improvement_of_positive_derivative (J : E → ℝ) (θ v : E)
    (G : E →L[ℝ] ℝ) (hJ : HasFDerivAt J G θ) (hv : 0 < G v) :
    ∃ radius : ℝ, 0 < radius ∧ ∀ t : ℝ, 0 < t → t < radius → J θ < J (θ + t • v) := by
  have hline : HasDerivAt (fun t : ℝ => θ + t • v) v 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add θ
  have hcomp : HasDerivAt (fun t : ℝ => J (θ + t • v)) (G v) 0 := by
    have hJ' : HasFDerivAt J G (θ + (0 : ℝ) • v) := by simpa using hJ
    exact hJ'.comp_hasDerivAt 0 hline
  have hevent := hcomp.tendsto_slope_zero_right.eventually (Ioi_mem_nhds hv)
  obtain ⟨radius, hr, hsub⟩ := mem_nhdsGT_iff_exists_Ioo_subset.mp hevent
  refine ⟨radius, hr, ?_⟩
  intro t ht htr
  have h := hsub ⟨ht, htr⟩
  simp only [zero_add, zero_smul, add_zero, smul_eq_mul] at h
  have hinv : 0 < t⁻¹ := inv_pos.mpr ht
  exact sub_pos.mp ((mul_pos_iff_of_pos_left hinv).mp h)

/-- Use the actual class multiplier, without replacing it by a potentially
small clamp-based lower bound. This applies to the complete response coefficient. -/
theorem population_local_improvement {m : ℕ}
    (p : E → A → ℝ) (b : A → Bool) (θ : E) (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a)
    (hnorm : ∀ᶠ η in 𝓝 θ, ∑ a, p η a = 1)
    (hmass : ∀ q, classMass (p θ) b q ≠ 0)
    (C : (Fin (m + 1) → A) → Fin (m + 1) → ℝ)
    (hmargin : Real.sqrt (coefficientVariance (p θ) b C * ((m + 1 : ℝ) *
        scoreVariance (p θ) (fun a => policyScore p θ D a (successGradient p b θ D)) b)) <
      classMultiplier (p θ) b C * ‖successGradient p b θ D‖ ^ 2) :
    let U := populationUpdate (p θ) (policyScore p θ D) C
    HasGradientAt (fun η => successMass (p η) b) (successGradient p b θ D) θ ∧
      0 < successDifferential p b θ D U ∧
      ∃ radius : ℝ, 0 < radius ∧ ∀ t : ℝ, 0 < t → t < radius →
        successMass (p θ) b < successMass (p (θ + t • U)) b := by
  dsimp only
  have hb := population_ascent_bound p b θ D hd hpos hnorm hmass C
    (classMultiplier (p θ) b C) le_rfl
  have ha := lt_of_lt_of_le (sub_pos.mpr hmargin) hb
  have hderiv := (finite_policy_uniform_correction p b θ D 0 1 0 hd hpos hnorm).1
  exact ⟨success_gradient_correct p b θ D hd hpos hnorm, ha,
    local_improvement_of_positive_derivative _ θ _ _ hderiv ha⟩

/-- A covariance margin gives actual local improvement for arbitrary
content-dependent response lengths, with all real-arithmetic safeguards. -/
theorem response_local_improvement {m : ℕ} (hm : m ≠ 0)
    (p : E → A → ℝ) (b : A → Bool) (θ : E) (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a)
    (hnorm : ∀ᶠ η in 𝓝 θ, ∑ a, p η a = 1)
    (hmass : ∀ q, 0 < classMass (p θ) b q)
    (uniform : Bool) (length : A → ℝ) (ε cap maxScale Lmax : ℝ)
    (hε : 0 < ε) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale)
    (hposL : ∀ a, 0 < length a) (hmax : ∀ a, length a ≤ Lmax) (hL : 0 < Lmax)
    (hmargin :
      Real.sqrt (coefficientVariance (p θ) b
          (responseCoefficient (m := m) uniform b length ε cap maxScale) *
        ((m + 1 : ℝ) * scoreVariance (p θ)
          (fun a => policyScore p θ D a (successGradient p b θ D)) b)) <
        (ε / Lmax) * ‖successGradient p b θ D‖ ^ 2) :
    let U := populationUpdate (p θ) (policyScore p θ D)
      (responseCoefficient (m := m) uniform b length ε cap maxScale)
    HasGradientAt (fun η => successMass (p η) b) (successGradient p b θ D) θ ∧
      0 < successDifferential p b θ D U ∧
      ∃ radius : ℝ, 0 < radius ∧ ∀ t : ℝ, 0 < t → t < radius →
        successMass (p θ) b < successMass (p (θ + t • U)) b := by
  dsimp only
  have hp0 := hpos.self_of_nhds
  have hp := hnorm.self_of_nhds
  have hlower := response_class_multiplier_lower_bound hm (p θ) b uniform length
    ε cap maxScale Lmax hp0 hp hε.le hε1 hεK hposL hmax hmass hL
  have hb := population_ascent_bound p b θ D hd hpos hnorm
    (fun q => ne_of_gt (hmass q))
    (responseCoefficient (m := m) uniform b length ε cap maxScale)
    (ε / Lmax) hlower
  have ha := lt_of_lt_of_le (sub_pos.mpr hmargin) hb
  have hderiv := (finite_policy_uniform_correction p b θ D 0 1 0 hd hpos hnorm).1
  exact ⟨success_gradient_correct p b θ D hd hpos hnorm, ha,
    local_improvement_of_positive_derivative _ θ _ _ hderiv ha⟩

/-- With class-dependent lengths, nonstationarity alone is enough: the
complete expected update has a strictly positive common multiplier. -/
theorem class_length_local_improvement {m : ℕ} (hm : m ≠ 0)
    (p : E → A → ℝ) (b : A → Bool) (θ : E) (D : A → E →L[ℝ] ℝ)
    (hd : ∀ a, HasFDerivAt (fun η => p η a) (D a) θ)
    (hpos : ∀ᶠ η in 𝓝 θ, ∀ a, 0 ≤ p η a)
    (hnorm : ∀ᶠ η in 𝓝 θ, ∑ a, p η a = 1)
    (uniform : Bool) (length : Bool → ℝ) (ε cap maxScale Lmax : ℝ)
    (hε : 0 < ε) (hε1 : ε ≤ 1) (hεK : ε ≤ maxScale)
    (hposL : ∀ q, 0 < length q) (hmax : ∀ q, length q ≤ Lmax)
    (hg : successGradient p b θ D ≠ 0) :
    let C := fun y : Fin (m + 1) → A =>
      implementationCoefficient uniform length ε cap maxScale (fun j => b (y j))
    let U := populationUpdate (p θ) (policyScore p θ D) C
    HasGradientAt (fun η => successMass (p η) b) (successGradient p b θ D) θ ∧
      0 < successDifferential p b θ D U ∧
      ∃ radius : ℝ, 0 < radius ∧ ∀ t : ℝ, 0 < t → t < radius →
        successMass (p θ) b < successMass (p (θ + t • U)) b := by
  dsimp only
  have hdir := implementation_policy_direction hm p b θ D hd hpos hnorm
    uniform length ε cap maxScale Lmax hε hε1 hεK hposL hmax
  have hnormsq : 0 < ‖successGradient p b θ D‖ ^ 2 := sq_pos_of_pos (norm_pos_iff.mpr hg)
  have hself : successDifferential p b θ D (successGradient p b θ D) =
      ‖successGradient p b θ D‖ ^ 2 := by
    simpa [successDifferential, successScore, ContinuousLinearMap.sum_apply] using
      success_score_gradient_norm p b θ D
  have ha : 0 < successDifferential p b θ D
      (populationUpdate (p θ) (policyScore p θ D)
        (fun y : Fin (m + 1) → A =>
          implementationCoefficient uniform length ε cap maxScale (fun j => b (y j)))) := by
    rw [ascent_pairing, update_differential_apply, hdir.2.2.2]
    change 0 < _ * successDifferential p b θ D (successGradient p b θ D)
    rw [hself]
    exact mul_pos hdir.2.1 hnormsq
  exact ⟨success_gradient_correct p b θ D hd hpos hnorm, ha,
    local_improvement_of_positive_derivative _ θ _ _ hdir.1 ha⟩

end
end REINFORCEProMax.BinaryAscent
