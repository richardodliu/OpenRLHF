import BinaryAscent

/-! Finite prompt aggregation of normalized population updates. Prompt-specific
positive scaling alone does not imply ascent of the common reward objective. -/
namespace REINFORCEProMax.PromptAscent
open scoped BigOperators
open InnerProductSpace
noncomputable section
variable {X E : Type*} [Fintype X]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

def mean (μ f : X → ℝ) : ℝ := ∑ x, μ x * f x
def variance (μ f : X → ℝ) : ℝ := ∑ x, μ x * (f x - mean μ f)^2
def covariance (μ f h : X → ℝ) : ℝ :=
  ∑ x, μ x * (f x - mean μ f) * (h x - mean μ h)

theorem covariance_identity (μ f h : X → ℝ) (hμ : ∑ x, μ x = 1) :
    mean μ (fun x => f x*h x) = mean μ f * mean μ h + covariance μ f h := by
  have heq : ∀ x, μ x * (f x-mean μ f)*(h x-mean μ h) =
      μ x*(f x*h x) - (μ x*f x)*mean μ h - (μ x*h x)*mean μ f +
      μ x*(mean μ f*mean μ h) := by intro x; ring
  simp only [covariance, heq, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.sum_mul, hμ, one_mul]
  unfold mean
  ring

theorem covariance_bound (μ f h : X → ℝ) (hμ : ∀ x, 0 ≤ μ x) :
    |covariance μ f h| ≤ Real.sqrt (variance μ f * variance μ h) := by
  have hcs := Finset.sum_sq_le_sum_mul_sum_of_sq_eq_mul
    (s := Finset.univ) (r := fun x => μ x*(f x-mean μ f)*(h x-mean μ h))
    (f := fun x => μ x*(f x-mean μ f)^2)
    (g := fun x => μ x*(h x-mean μ h)^2)
    (fun x _ => mul_nonneg (hμ x) (sq_nonneg _))
    (fun x _ => mul_nonneg (hμ x) (sq_nonneg _)) (fun x _ => by ring)
  change (covariance μ f h)^2 ≤ variance μ f*variance μ h at hcs
  have hs := Real.sq_sqrt ((sq_nonneg _).trans hcs)
  have hn := Real.sqrt_nonneg (variance μ f*variance μ h)
  rw [abs_le]
  constructor <;> nlinarith

def averageGradient (μ : X → ℝ) (g : X → E) : E := ∑ x, μ x • g x

def averageUpdate (μ k : X → ℝ) (g e : X → E) : E :=
  ∑ x, μ x • (k x • g x + e x)

omit [CompleteSpace E] in
theorem projection_mean (μ : X → ℝ) (g : X → E) :
    mean μ (fun x => ⟪averageGradient μ g, g x⟫_ℝ) = ‖averageGradient μ g‖^2 := by
  have h : mean μ (fun x => ⟪averageGradient μ g, g x⟫_ℝ) =
      ⟪averageGradient μ g, averageGradient μ g⟫_ℝ := by
    simp [mean, averageGradient, inner_sum, real_inner_smul_right]
  rw [h, real_inner_self_eq_norm_sq]

theorem average_ascent_bound (μ k : X → ℝ) (g e : X → E)
    (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1) :
    mean μ k * ‖averageGradient μ g‖^2 -
      Real.sqrt (variance μ k * variance μ (fun x => ⟪averageGradient μ g, g x⟫_ℝ)) -
      mean μ (fun x => |⟪averageGradient μ g, e x⟫_ℝ|) ≤
        ⟪averageGradient μ g, averageUpdate μ k g e⟫_ℝ := by
  have hid : ⟪averageGradient μ g, averageUpdate μ k g e⟫_ℝ =
      mean μ (fun x => k x*⟪averageGradient μ g,g x⟫_ℝ) +
      mean μ (fun x => ⟪averageGradient μ g,e x⟫_ℝ) := by
    simp [averageUpdate, mean, inner_sum, inner_add_right, real_inner_smul_right,
      mul_add, Finset.sum_add_distrib]
  rw [hid, covariance_identity μ k _ hμ, projection_mean]
  have hc := (abs_le.mp (covariance_bound μ k
    (fun x => ⟪averageGradient μ g,g x⟫_ℝ) hμ0)).1
  have he : -mean μ (fun x => |⟪averageGradient μ g,e x⟫_ℝ|) ≤
      mean μ (fun x => ⟪averageGradient μ g,e x⟫_ℝ) := by
    have hh := Finset.sum_le_sum (s := Finset.univ) (fun x _ =>
      mul_le_mul_of_nonneg_left (neg_abs_le ⟪averageGradient μ g,e x⟫_ℝ) (hμ0 x))
    simpa [mean] using hh
  linarith

theorem finite_prompt_gradient (μ : X → ℝ) (J : X → E → ℝ) (g : X → E) (θ : E)
    (hJ : ∀ x, HasGradientAt (J x) (g x) θ) :
    HasGradientAt (fun η => ∑ x, μ x*J x η) (averageGradient μ g) θ := by
  have hh := HasFDerivAt.sum (fun x (_ : x ∈ (Finset.univ : Finset X)) =>
    ((hJ x).hasFDerivAt.const_mul (μ x)))
  rw [hasGradientAt_iff_hasFDerivAt]
  simpa [averageGradient] using hh

theorem finite_prompt_local_improvement (μ k : X → ℝ) (J : X → E → ℝ)
    (g e : X → E) (θ : E) (hμ0 : ∀ x, 0 ≤ μ x) (hμ : ∑ x, μ x = 1)
    (hJ : ∀ x, HasGradientAt (J x) (g x) θ)
    (hmargin : Real.sqrt (variance μ k * variance μ (fun x => ⟪averageGradient μ g,g x⟫_ℝ)) +
      mean μ (fun x => |⟪averageGradient μ g,e x⟫_ℝ|) < mean μ k * ‖averageGradient μ g‖^2) :
    ∃ r : ℝ, 0 < r ∧ ∀ t : ℝ, 0 < t → t < r →
      (∑ x, μ x*J x θ) < ∑ x, μ x*J x (θ+t • averageUpdate μ k g e) := by
  have hb := average_ascent_bound μ k g e hμ0 hμ
  have hd := (finite_prompt_gradient μ J g θ hJ).hasFDerivAt
  apply BinaryAscent.local_improvement_of_positive_derivative _ θ _ _ hd
  change 0 < ⟪averageGradient μ g, averageUpdate μ k g e⟫_ℝ
  linarith

/-- Positive individual multipliers need not preserve the direction of the
mean gradient. This algebraic example is not a ProMax execution trace. -/
theorem positive_multipliers_can_reverse :
    let μ : Bool → ℝ := fun _ => 1/2
    let g : Bool → ℝ := fun x => if x then 1 else -2
    let k : Bool → ℝ := fun x => if x then 5 else 1
    (∀ x, 0 < k x) ∧
      averageGradient μ g * averageUpdate μ k g (fun _ => 0) = -3/4 := by
  dsimp only
  constructor
  · intro x; cases x <;> norm_num
  · norm_num [averageGradient, averageUpdate, Fintype.sum_bool]

end
end REINFORCEProMax.PromptAscent
