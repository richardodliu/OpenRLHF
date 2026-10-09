import BatchPerformance

/-!
Simultaneous certificates for a finite, fixed family evaluated on the same
rollout data. The selector may use all evaluation data; the candidate family
and its probability laws are fixed before that data is drawn. No independence
among candidates' estimates is required.
-/
namespace REINFORCEProMax.BatchReduction

open scoped BigOperators
noncomputable section
set_option maxHeartbeats 400000
variable {A J : Type*} [Fintype A] [DecidableEq A] [Fintype J]

theorem finite_event_union_bound (w : A → ℝ) (hw : ∀ a, 0 ≤ w a)
    (E : J → A → Prop) [∀ j, DecidablePred (E j)]
    [DecidablePred (fun a => ∃ j, E j a)] :
    (∑ a ∈ Finset.univ.filter (fun a => ∃ j, E j a), w a) ≤
      ∑ j, ∑ a ∈ Finset.univ.filter (E j), w a := by
  classical
  simp only [Finset.sum_filter]
  rw [Finset.sum_comm]
  apply Finset.sum_le_sum
  intro a _
  by_cases he : ∃ j, E j a
  · rw [if_pos he]
    obtain ⟨j, hj⟩ := he
    calc
      w a = (if E j a then w a else 0) := (if_pos hj).symm
      _ ≤ ∑ k, if E k a then w a else 0 :=
        Finset.single_le_sum (f := fun k => if E k a then w a else 0)
          (fun k _ => by dsimp; split_ifs; exact hw a; exact le_rfl) (Finset.mem_univ j)
  · rw [if_neg he]
    exact Finset.sum_nonneg fun j _ => by split_ifs; exact hw a; exact le_rfl

theorem iid_token_mean_uniform_tail {n : ℕ} (hn : n ≠ 0)
    (p L : A → ℝ) (Z : J → A → ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (l : ℝ) (κ ε : J → ℝ) (hl : 0 < l) (hε : ∀ j, 0 < ε j)
    (hL : ∀ a, l ≤ L a)
    (hcenter : ∀ j, expectation p (fun a => Z j a - κ j * L a) = 0) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → A => ∃ j,
      ε j ≤ |sampleMean (Z j) s / sampleMean L s - κ j|), iidMass p s) ≤
      ∑ j, expectation p (fun a => (Z j a - κ j * L a) ^ 2) /
        ((n : ℝ) * l ^ 2 * (ε j) ^ 2) := by
  classical
  exact (finite_event_union_bound (iidMass p) (iid_mass_nonneg p hp0) _).trans
    (Finset.sum_le_sum fun j _ =>
      iid_token_mean_tail hn p (Z j) L hp0 hp l (κ j) (ε j) hl (hε j) hL (hcenter j))

/-- Data-dependent selection is safe against false improvement certificates
within the fixed candidate family, at the sum of the individual error budgets. -/
theorem selected_false_improvement_bound {n : ℕ} (hn : n ≠ 0)
    (p L : A → ℝ) (Z : J → A → ℝ)
    (hp0 : ∀ a, 0 ≤ p a) (hp : ∑ a, p a = 1)
    (l μ : ℝ) (κ cost ε gain : J → ℝ)
    (hl : 0 < l) (hμ : 0 < μ) (hε : ∀ j, 0 < ε j)
    (hL : ∀ a, l ≤ L a)
    (hcenter : ∀ j, expectation p (fun a => Z j a - κ j * L a) = 0)
    (hperf : ∀ j, μ * κ j - cost j ≤ gain j)
    (select : (Fin n → A) → J) :
    (∑ s ∈ Finset.univ.filter (fun s : Fin n → A =>
      gain (select s) ≤ 0 ∧ cost (select s) / μ + ε (select s) <
        sampleMean (Z (select s)) s / sampleMean L s), iidMass p s) ≤
      ∑ j, expectation p (fun a => (Z j a - κ j * L a) ^ 2) /
        ((n : ℝ) * l ^ 2 * (ε j) ^ 2) := by
  classical
  let E : J → (Fin n → A) → Prop := fun j s =>
    gain j ≤ 0 ∧ cost j / μ + ε j < sampleMean (Z j) s / sampleMean L s
  have hsub : Finset.univ.filter (fun s : Fin n → A => E (select s) s) ⊆
      Finset.univ.filter (fun s => ∃ j, E j s) := by
    intro s hs
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, select s, (Finset.mem_filter.mp hs).2⟩
  exact ((Finset.sum_le_sum_of_subset_of_nonneg hsub
    (fun s _ _ => iid_mass_nonneg p hp0 s)).trans
      (finite_event_union_bound (iidMass p) (iid_mass_nonneg p hp0) E)).trans
    (Finset.sum_le_sum fun j _ => false_improvement_certificate_bound hn p (Z j) L hp0 hp
      l (κ j) μ (cost j) (ε j) (gain j) hl hμ (hε j) hL (hcenter j) (hperf j))


/-- Average conditional failure bounds over a finite training-data law.
The evaluation law may depend on training; no unconditional independence
between the trained policy and its evaluation trajectories is assumed. -/
theorem conditional_failure_average {D : Type*} [Fintype D]
    (train : D → ℝ) (eval : D → A → ℝ) (budget : D → ℝ)
    (event : D → A → Prop) [∀ d, DecidablePred (event d)]
    (htrain : ∀ d, 0 ≤ train d)
    (hconditional : ∀ d, (∑ a ∈ Finset.univ.filter (event d), eval d a) ≤ budget d) :
    (∑ d, ∑ a ∈ Finset.univ.filter (event d), train d * eval d a) ≤
      ∑ d, train d * budget d := by
  apply Finset.sum_le_sum
  intro d _
  rw [← Finset.mul_sum]
  exact mul_le_mul_of_nonneg_left (hconditional d) (htrain d)

theorem conditional_failure_uniform_budget {D : Type*} [Fintype D]
    (train : D → ℝ) (eval : D → A → ℝ) (α : ℝ)
    (event : D → A → Prop) [∀ d, DecidablePred (event d)]
    (htrain : ∀ d, 0 ≤ train d) (ht : ∑ d, train d = 1)
    (hconditional : ∀ d, (∑ a ∈ Finset.univ.filter (event d), eval d a) ≤ α) :
    (∑ d, ∑ a ∈ Finset.univ.filter (event d), train d * eval d a) ≤ α := by
  have hb := conditional_failure_average train eval (fun _ => α) event htrain hconditional
  simpa [← Finset.sum_mul, ht] using hb

/-- The conditional iid evaluation construction is a normalized joint law. -/
theorem training_evaluation_mass_normalized {D : Type*} [Fintype D]
    (train : D → ℝ) (p : D → A → ℝ)
    (ht : ∑ d, train d = 1) (hp : ∀ d, ∑ a, p d a = 1) (n : ℕ) :
    (∑ d, ∑ s : Fin n → A, train d * iidMass (p d) s) = 1 := by
  simp_rw [← Finset.mul_sum, iid_mass_normalized _ (hp _) n, mul_one]
  exact ht

/-- Policies, candidates, thresholds and the selector may depend on finite
training data. Given that data, evaluation blocks obey the stated iid law.
The unconditional false-certificate probability is at most the average
conditional budget, including when the selector sees all evaluation blocks. -/
theorem trained_selected_false_improvement_bound {D : Type*} [Fintype D]
    {n : ℕ} (hn : n ≠ 0)
    (train : D → ℝ) (p L : D → A → ℝ) (Z : D → J → A → ℝ)
    (ht0 : ∀ d, 0 ≤ train d)
    (hp0 : ∀ d a, 0 ≤ p d a) (hp : ∀ d, ∑ a, p d a = 1)
    (l μ : D → ℝ) (κ cost ε gain : D → J → ℝ)
    (hl : ∀ d, 0 < l d) (hμ : ∀ d, 0 < μ d)
    (hε : ∀ d j, 0 < ε d j) (hL : ∀ d a, l d ≤ L d a)
    (hcenter : ∀ d j, expectation (p d) (fun a => Z d j a - κ d j * L d a) = 0)
    (hperf : ∀ d j, μ d * κ d j - cost d j ≤ gain d j)
    (select : D → (Fin n → A) → J) :
    (∑ d, ∑ s ∈ Finset.univ.filter (fun s : Fin n → A =>
      gain d (select d s) ≤ 0 ∧ cost d (select d s) / μ d + ε d (select d s) <
        sampleMean (Z d (select d s)) s / sampleMean (L d) s),
        train d * iidMass (p d) s) ≤
      ∑ d, train d * ∑ j,
        expectation (p d) (fun a => (Z d j a - κ d j * L d a)^2) /
          ((n : ℝ) * (l d)^2 * (ε d j)^2) := by
  classical
  apply conditional_failure_average
  · exact ht0
  · intro d
    exact selected_false_improvement_bound hn (p d) (L d) (Z d)
      (hp0 d) (hp d) (l d) (μ d) (κ d) (cost d) (ε d) (gain d)
      (hl d) (hμ d) (hε d) (hL d) (hcenter d) (hperf d) (select d)

end
end REINFORCEProMax.BatchReduction
