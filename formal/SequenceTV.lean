import Mathlib.Tactic

/-!
  Finite-state causal (Markov-kernel) checks for the sequence-TV accumulation
  statement in `tex/main/5-theory.tex`.  A state represents the complete prefix
  context at a given time; kernels may therefore depend on that context.
-/
namespace REINFORCEProMax

noncomputable def tv {n : ℕ} (p q : Fin n → ℝ) : ℝ :=
  (1 / 2 : ℝ) * ∑ i : Fin n, |p i - q i|

def push {n m : ℕ} (d : Fin n → ℝ)
    (K : Fin n → Fin m → ℝ) (j : Fin m) : ℝ :=
  ∑ i : Fin n, d i * K i j

/-- One causal transition changes TV by at most the largest row-wise kernel TV.
The hypotheses state that `p,q` and every row of `K,L` are probability vectors.
The proof uses the asymmetric decomposition `(p-q)L + p(K-L)`, so the
rollout-side assumptions are the ones needed by the contraction estimate;
the symmetric assumptions are retained to keep the statement distributional. -/
theorem finite_kernel_tv_step
    {n m : ℕ} (p q : Fin n → ℝ)
    (K L : Fin n → Fin m → ℝ) (ε : ℝ)
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hps : ∑ i : Fin n, p i = 1) (hqs : ∑ i : Fin n, q i = 1)
    (hK : ∀ i j, 0 ≤ K i j) (hL : ∀ i j, 0 ≤ L i j)
    (hKs : ∀ i, ∑ j : Fin m, K i j = 1)
    (hLs : ∀ i, ∑ j : Fin m, L i j = 1)
    (hrow : ∀ i, tv (K i) (L i) ≤ ε) :
    tv (push p K) (push q L) ≤ tv p q + ε := by
  have hrow_l1 : ∀ i : Fin n, (∑ j : Fin m, |K i j - L i j|) ≤ 2 * ε := by
    intro i
    have hi := hrow i
    dsimp [tv] at hi
    linarith
  have hdecomp : ∀ j : Fin m,
      push p K j - push q L j =
        (∑ i : Fin n, (p i - q i) * L i j) +
          ∑ i : Fin n, p i * (K i j - L i j) := by
    intro j
    dsimp [push]
    calc
      (∑ i : Fin n, p i * K i j) - ∑ i : Fin n, q i * L i j =
          ∑ i : Fin n, (p i * K i j - q i * L i j) := by
            rw [Finset.sum_sub_distrib]
      _ = ∑ i : Fin n, ((p i - q i) * L i j + p i * (K i j - L i j)) := by
            apply Finset.sum_congr rfl
            intro i hi
            ring
      _ = (∑ i : Fin n, (p i - q i) * L i j) +
          ∑ i : Fin n, p i * (K i j - L i j) := by
            rw [Finset.sum_add_distrib]
  have hA : ∑ j : Fin m, |∑ i : Fin n, (p i - q i) * L i j| ≤
      ∑ i : Fin n, |p i - q i| := by
    calc
      (∑ j : Fin m, |∑ i : Fin n, (p i - q i) * L i j|) ≤
          ∑ j : Fin m, ∑ i : Fin n, |(p i - q i) * L i j| := by
        apply Finset.sum_le_sum
        intro j hj
        exact Finset.abs_sum_le_sum_abs
          (fun i : Fin n => (p i - q i) * L i j) Finset.univ
      _ = ∑ j : Fin m, ∑ i : Fin n, |p i - q i| * L i j := by
        apply Finset.sum_congr rfl
        intro j hj
        apply Finset.sum_congr rfl
        intro i hi
        rw [abs_mul, abs_of_nonneg (hL i j)]
      _ = ∑ i : Fin n, |p i - q i| * (∑ j : Fin m, L i j) := by
        rw [Finset.sum_comm]
        apply Finset.sum_congr rfl
        intro i hi
        rw [Finset.mul_sum]
      _ = ∑ i : Fin n, |p i - q i| := by
        apply Finset.sum_congr rfl
        intro i hi
        rw [hLs i, mul_one]
  have hB : ∑ j : Fin m, |∑ i : Fin n, p i * (K i j - L i j)| ≤ 2 * ε := by
    calc
      (∑ j : Fin m, |∑ i : Fin n, p i * (K i j - L i j)|) ≤
          ∑ j : Fin m, ∑ i : Fin n, |p i * (K i j - L i j)| := by
        apply Finset.sum_le_sum
        intro j hj
        exact Finset.abs_sum_le_sum_abs
          (fun i : Fin n => p i * (K i j - L i j)) Finset.univ
      _ = ∑ j : Fin m, ∑ i : Fin n, p i * |K i j - L i j| := by
        apply Finset.sum_congr rfl
        intro j hj
        apply Finset.sum_congr rfl
        intro i hi
        rw [abs_mul, abs_of_nonneg (hp i)]
      _ = ∑ i : Fin n, p i * (∑ j : Fin m, |K i j - L i j|) := by
        rw [Finset.sum_comm]
        apply Finset.sum_congr rfl
        intro i hi
        rw [Finset.mul_sum]
      _ ≤ ∑ i : Fin n, p i * (2 * ε) := by
        apply Finset.sum_le_sum
        intro i hi
        exact mul_le_mul_of_nonneg_left (hrow_l1 i) (hp i)
      _ = 2 * ε := by
        rw [← Finset.sum_mul, hps]
        ring
  dsimp [tv]
  rw [show (∑ j : Fin m, |push p K j - push q L j|) =
      ∑ j : Fin m, |(∑ i : Fin n, (p i - q i) * L i j) +
          ∑ i : Fin n, p i * (K i j - L i j)| by
        apply Finset.sum_congr rfl
        intro j hj
        rw [hdecomp j]]
  have hsplit : ∀ j : Fin m,
      |(∑ i : Fin n, (p i - q i) * L i j) +
          ∑ i : Fin n, p i * (K i j - L i j)| ≤
        |∑ i : Fin n, (p i - q i) * L i j| +
          |∑ i : Fin n, p i * (K i j - L i j)| := by
    intro j
    exact abs_add _ _
  have htotal := Finset.sum_le_sum (s := (Finset.univ : Finset (Fin m))) (fun j hj => hsplit j)
  have htotal' :
      (∑ j : Fin m, |(∑ i : Fin n, (p i - q i) * L i j) +
          ∑ i : Fin n, p i * (K i j - L i j)|) ≤
        (∑ i : Fin n, |p i - q i|) + 2 * ε := by
    calc
      _ ≤ ∑ j : Fin m, (|∑ i : Fin n, (p i - q i) * L i j| +
            |∑ i : Fin n, p i * (K i j - L i j)|) := htotal
      _ = (∑ j : Fin m, |∑ i : Fin n, (p i - q i) * L i j|) +
            ∑ j : Fin m, |∑ i : Fin n, p i * (K i j - L i j)| := by
              rw [Finset.sum_add_distrib]
      _ ≤ (∑ i : Fin n, |p i - q i|) + 2 * ε := by linarith
  nlinarith [htotal']

/-- If each causal transition obeys a one-step recurrence, its TV drift
accumulates additively over a finite horizon (the recurrence may already
encode an embedding of time-varying prefix spaces into a common finite state
space). -/
theorem finite_tv_accumulation
    {T n : ℕ} (d e : ℕ → Fin n → ℝ)
    (ε : ℕ → ℝ)
    (hrec : ∀ t < T,
      tv (d (t + 1)) (e (t + 1)) ≤
        tv (d t) (e t) + ε t) :
    tv (d T) (e T) ≤
      tv (d 0) (e 0) + ∑ t ∈ Finset.range T, ε t := by
  induction T with
  | zero => simp
  | succ T ih =>
      have hstep := hrec T (Nat.lt_succ_self T)
      have hprev : ∀ t < T,
          tv (d (t + 1)) (e (t + 1)) ≤ tv (d t) (e t) + ε t := by
        intro t ht
        exact hrec t (lt_trans ht (Nat.lt_succ_self T))
      have hi := ih hprev
      rw [Finset.sum_range_succ]
      linarith

end REINFORCEProMax
