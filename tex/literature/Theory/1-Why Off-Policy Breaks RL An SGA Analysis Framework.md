---
title: "Part 1: Why Off-Policy Breaks RL — An SGA Analysis Framework"
date: 2025-10-30
math: true
authors:
  - admin
  - jiacai-liu
tags:
  - Reinforcement Learning
  - Off-Policy
  - PPO
  - TRPO
categories:
  - Research
series:
  - RL Collapse
series_order: 1
---

**Authors:** [Yingru Li](http://richardli.xyz), [Jiacai Liu](https://www.jiacailiu.cn/)

{{% callout note %}}
**Original Blog:** [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://richardli.xyz/rl-collapse)
{{% /callout %}}

---

## The Problem

In reinforcement learning, we often cannot sample directly from the policy $\pi_\theta$ we are optimizing. Instead, we sample from a different **behavior policy** $\mu$. This **off-policy** setting ($\mu \neq \pi$) arises from multiple sources:

1. **Standard off-policy RL**: Using a replay buffer or behavior policy for sample efficiency
2. **PPO's inherent off-policiness**: Reusing rollout samples across multiple gradient updates (the behavior policy $\mu$ is the policy at rollout time, while $\pi$ evolves during updates)
3. **Distributed LLM-RL systems**: Inference engine discrepancies (vLLM vs FSDP), quantization differences, or hardware-specific kernels

This off-policy mismatch is not a minor technicality—it's a fundamental mathematical problem that creates two distinct failure modes.

{{% callout note %}}
### TL;DR: Two Failure Modes

| Failure Mode | What Happens | Measured By | Consequence |
|--------------|--------------|-------------|-------------|
| **Bias (Wrong Direction)** | Optimizer pushed toward wrong solution | $D_{TV}$ (Total Variation) | Convergence to suboptimal policy |
| **Variance (Stalled Progress)** | Gradient noise forces tiny learning rate | $\chi^2$-divergence | Training flatlines |

**Key Insight:** These two metrics are not interchangeable. A small TV distance can hide a massive $\chi^2$-divergence. Confusing them leads to suboptimal solutions.

**When is bias tolerable?** When off-policiness is solely from policy parameter updates and controlled by algorithms (e.g., PPO's clipping).

**When does bias become catastrophic?** When off-policiness has diverse, uncontrolled sources—such as distributed system discrepancies, MoE expert routing shifts, or large policy changes.
{{% /callout %}}

---

## Citation

```bibtex
@online{liu-li-2025-rl-collapse,
  title = {When Speed Kills Stability: Demystifying {RL} Collapse from the Training-Inference Mismatch},
  author = {Liu, Jiacai and Li, Yingru and Fu, Yuqian and Wang, Jiawei and Liu, Qian and Shen, Yu},
  year = {2025},
  month = sep,
  url = {https://richardli.xyz/rl-collapse}
}
```

---

## 1. Setup: Policy Optimization as Stochastic Gradient Ascent

### 1.1 The Optimization Goal

In policy-based RL, we optimize a policy $\pi_\theta$ to maximize expected reward:

{{< math >}}
$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}[R(y|x)]
$$
{{< /math >}}

We do this via stochastic gradient ascent:

{{< math >}}
$$
\theta_{k+1} = \theta_k + \eta \hat{g}_k
$$
{{< /math >}}

where $\hat{g}_k$ is our gradient estimator.

### 1.2 The Off-Policy Problem

**Ideally:** We sample $y \sim \pi_\theta$ and compute an unbiased gradient estimator $\hat{g}$.

**Reality:** We sample $y \sim \mu$ (a behavior policy) and must use importance sampling to correct for the off-policy mismatch.

{{% callout note %}}
Because we sample from $\mu \neq \pi$, our gradient estimator $\hat{g}$ may become biased (if we don't fully correct) or have high variance (if we do correct via importance sampling). How significant is this effect? We need a formal tool to quantify it.
{{% /callout %}}

### 1.3 The MDP Formulation

To analyze this rigorously, we model the problem as a Markov Decision Process. For **autoregressive LLM generation**, this becomes:

| RL Concept | LLM Interpretation |
|------------|-------------------|
| **State** $s_t$ | The prefix $(x, y_{\lt t})$: prompt + previously generated tokens |
| **Action** $a_t$ | The next token $y_t$ |
| **Policy** $\pi(a_t \| s_t)$ | Token distribution $\pi_\theta(y_t \| x, y_{\lt t})$ |
| **Transition** $P(s_{t+1} \| s_t, a_t)$ | **Deterministic**: appending $y_t$ to $(x, y_{\lt t})$ gives $s_{t+1} = (x, y_{\lt t+1})$ |
| **Horizon** $T$ | Sequence length |

This deterministic transition structure is crucial: once you choose an action, the next state is fully determined. This applies to LLM generation and many other sequential decision problems.

---

## 2. The SGA Lemma: Quantifying Off-Policy Effects

The **Stochastic Gradient Ascent (SGA) Lemma** gives us a precise formula for optimization progress. This is our primary analytical tool—it applies to any gradient-based policy optimization, including PPO, GRPO, REINFORCE, and their variants.

For an $L$-smooth objective, the expected progress per step is:

{{< math >}}
$$
\begin{aligned}
\mathbb{E}[J(\theta_{k+1})] - J(\theta_k) \geq\; & \underbrace{\eta \left(1 - \frac{L\eta}{2}\right)\|\nabla J\|^2}_{\text{Term A: True Progress}} \\
& + \underbrace{\eta(1 - L\eta)\langle \nabla J, \mathbf{Bias}(\hat{g}) \rangle}_{\text{Term B: Bias Error}} \\
& - \underbrace{\frac{L\eta^2}{2}\left[\mathbf{Var}(\hat{g}) + \|\mathbf{Bias}(\hat{g})\|^2\right]}_{\text{Term C: Noise Penalty}}
\end{aligned}
$$
{{< /math >}}

where:
- **Term A:** Ideal progress with the true gradient
- **Term B:** Effect of systematic error (can be negative!)
- **Term C:** Penalty from noise (always negative)

This decomposition reveals exactly how mismatch affects optimization progress.

{{< spoiler text="Derivation of the SGA Lemma" >}}

**1. Start with the $L$-smoothness assumption:**
An objective $J$ is $L$-smooth if its gradient is $L$-Lipschitz. The descent lemma states:

$$
J(\theta_{k+1}) \geq J(\theta_k) + \langle \nabla J(\theta_k), \eta \hat{g}_k \rangle - \frac{L\eta^2}{2}\|\hat{g}_k\|^2
$$

**2. Take the expectation:**

$$
\mathbb{E}[J(\theta_{k+1})] - J(\theta_k) \geq \eta\langle \nabla J, \mathbb{E}[\hat{g}_k] \rangle - \frac{L\eta^2}{2}\mathbb{E}[\|\hat{g}_k\|^2]
$$

**3. Decompose using Bias and Variance:**

- $\mathbf{Bias}(\hat{g}) = \mathbb{E}[\hat{g}] - \nabla J$
- $\mathbf{Var}(\hat{g}) = \mathbb{E}[\|\hat{g}\|^2] - \|\mathbb{E}[\hat{g}]\|^2$

**4. Substitute and expand:**

Using $\mathbb{E}[\hat{g}] = \nabla J + \mathbf{Bias}(\hat{g})$:

$$
\mathbb{E}[\|\hat{g}\|^2] = \mathbf{Var}(\hat{g}) + \|\nabla J + \mathbf{Bias}(\hat{g})\|^2
$$

Expanding the squared term:

$$
= \mathbf{Var}(\hat{g}) + \|\nabla J\|^2 + 2\langle \nabla J, \mathbf{Bias}(\hat{g}) \rangle + \|\mathbf{Bias}(\hat{g})\|^2
$$

**5. Collect terms:**

- $\|\nabla J\|^2$ terms: $\eta (1 - \frac{L\eta}{2})\|\nabla J\|^2$ (Term A)
- $\langle \nabla J, \mathbf{Bias} \rangle$ terms: $\eta(1 - L\eta)\langle \nabla J, \mathbf{Bias} \rangle$ (Term B)
- $\mathbf{Var}$ and $\|\mathbf{Bias}\|^2$ terms: $- \frac{L\eta^2}{2}[\mathbf{Var} + \|\mathbf{Bias}\|^2]$ (Term C)

{{< /spoiler >}}

---

## 3. The Two Failure Modes

### 3.1 Failure Mode 1: Bias (Converging to the Wrong Solution)

The bias is the systematic error:

{{< math >}}
$$
\mathbf{Bias}(\hat{g}) = \mathbb{E}[\hat{g}] - \nabla J
$$
{{< /math >}}

**Term B** measures alignment between true gradient and this error: $\langle \nabla J, \mathbf{Bias} \rangle$

- If bias is small or random → Term B ≈ 0 → OK
- If bias is systematic and opposes $\nabla J$ → **Term B becomes highly negative**

**Consequence:** A negative Term B means the optimization direction opposes the true objective direction. This not only slows convergence but leads to convergence to the **wrong solution**.

### 3.2 Failure Mode 2: Variance (Stalled Progress)

The variance is the noise:

{{< math >}}
$$
\mathbf{Var}(\hat{g}) = \mathbb{E}[\|\hat{g}\|^2] - \|\mathbb{E}[\hat{g}]\|^2
$$
{{< /math >}}

**Term C** is always negative and scales with $\eta^2$.

- High variance → huge negative Term C
- To ensure net positive progress → must use tiny $\eta$

**Consequence:** High variance forces $\eta = O(1/\mathbf{Var})$. Training stalls—not because the optimum has been reached, but because the learning rate is too small for effective updates.

---

## 4. The Right Tools: TV Distance vs. $\chi^2$-Divergence

We've identified two failure modes. We need the right mathematical tools to measure each.

### 4.1 Total Variation (TV) Distance → Measures Bias

{{< math >}}
$$
D_{TV}(\pi \| \mu) = \frac{1}{2} \sum_y |\pi(y|x) - \mu(y|x)|
$$
{{< /math >}}

**Why TV for bias?** Bias is a difference of expectations. The key identity is:

{{< math >}}
$$
|\mathbb{E}_\pi[f] - \mathbb{E}_\mu[f]| \leq 2 \|f\|_\infty \cdot D_{TV}(\pi \| \mu)
$$
{{< /math >}}

TV distance directly bounds how much expectations can differ between two distributions.

**Our metric:** $\Delta_{\max}$ = maximum per-token TV distance

### 4.2 Chi-Square ($\chi^2$) Divergence → Measures Variance

{{< math >}}
$$
\chi^2(\pi\|\mu) = \mathbb{E}_\mu\left[\left(\frac{\pi(y)}{\mu(y)}\right)^2\right] - 1 = \mathbb{E}_\mu[\rho^2] - 1
$$
{{< /math >}}

**Why $\chi^2$ for variance?** The variance of any importance-sampled estimator depends on $\mathbb{E}_\mu[\rho^2]$. If this is large or infinite, variance explodes.

**Our metric:** $\chi^2$ at sequence level

### 4.3 The Critical Insight: These Metrics Are Not Interchangeable

The Pinsker-type inequality $D_{TV}(\pi\|\mu) \leq \sqrt{\frac{1}{2}\chi^2(\pi\|\mu)}$ reveals:

> **A tiny TV distance can hide a massive $\chi^2$-divergence.**

The converse does not hold: bounding TV distance does NOT bound $\chi^2$-divergence. You cannot use TV distance to analyze variance. This is why the TRPO/PPO framework has a blind spot—it uses $D_{TV}$ or $D_{KL}$, which cannot detect variance explosions.

---

## 5. Connection to Trust Region Methods (PPO/TRPO)

A natural question: "Don't PPO and TRPO already solve this?"

The answer reveals a critical gap between theory and practice.

### 5.1 The Surrogate Objective

TRPO optimizes a **surrogate objective** instead of $J(\pi)$ directly:

{{< math >}}
$$
L_{\mu}(\pi) = J(\mu) + \mathbb{E}_{s \sim d_\mu} \mathbb{E}_{a \sim \pi(\cdot|s)} [A_\mu(s, a)]
$$
{{< /math >}}

where $d_\mu$ is the state visitation distribution under $\mu$, and $A_\mu$ is the advantage function.

**Why use a surrogate?** It satisfies two key properties at $\pi = \mu$:

| Property | At $\pi = \mu$ | Away from $\pi = \mu$ |
|----------|----------------|----------------------|
| Value | $L_\mu(\mu) = J(\mu)$ ✓ | $L_\mu(\pi) \neq J(\pi)$ |
| Gradient | $\nabla L_\mu = \nabla J$ ✓ | $\nabla L_\mu \neq \nabla J$ |

The surrogate is a **first-order Taylor approximation** of $J(\pi)$ around $\pi = \mu$.

### 5.2 The Key Equations: Token-Level IS = Surrogate Gradient

The gradient PPO/GRPO actually computes is:

{{< math >}}
$$
\begin{aligned}
&\sum_{t=0}^{T-1} \mathbb{E}_{s_t \sim d_{\mu,t}} \mathbb{E}_{y_t \sim \mu(\cdot|s_t)} \left[ \frac{\pi_\theta(y_t|s_t)}{\mu(y_t|s_t)} A_\mu(s_t, y_t) \nabla_\theta \log \pi_\theta(y_t|s_t) \right] \\
&= \nabla_\theta L_\mu \quad \text{(Token-level IS gradient, what PPO computes)}
\end{aligned}
$$
{{< /math >}}

The **true** policy gradient is:

{{< math >}}
$$
\begin{aligned}
\nabla_\theta J = \sum_{t=0}^{T-1} \mathbb{E}_{s_t \sim d_{\pi,t}} \mathbb{E}_{y_t \sim \pi(\cdot|s_t)} \left[ A_\pi(s_t, y_t) \nabla_\theta \log \pi_\theta(y_t|s_t) \right]
\end{aligned}
$$
{{< /math >}}

**Critical observation:** Token-level IS corrects for the **token distribution** mismatch (via the $\pi/\mu$ ratio), but the expectation over states is still under $d_{\mu,t}$, not $d_{\pi,t}$. The **prefix distribution mismatch** is NOT corrected. This uncorrected state mismatch is the source of the $O(T^2 \Delta_{\max})$ bias in token-level methods.

### 5.3 The TRPO Lower Bound

The **Performance Difference Lemma** connects surrogate and true objectives:

{{< math >}}
$$
J(\pi) - J(\mu) = \mathbb{E}_{s \sim d_{\pi}} \mathbb{E}_{a \sim \pi(\cdot|s)} [A_\mu(s, a)]
$$
{{< /math >}}

Notice: true improvement uses $d_\pi$, surrogate uses $d_\mu$. TRPO bounds this gap:

{{< math >}}
$$
\boxed{J(\pi) \geq L_\mu(\pi) - C \cdot T^2 \cdot D_{TV}^{\max}(\pi, \mu)}
$$
{{< /math >}}

where:
- $C$ = constant depending on max advantage $\max_{s,a}|A_\mu(s,a)|$
- $T$ = horizon (sequence length)
- $D_{TV}^{\max} = \max_s D_{TV}(\pi(\cdot|s) \| \mu(\cdot|s))$

**The penalty scales quadratically with horizon $T^2$** because state distribution errors accumulate linearly with $t$, and summing over all timesteps gives $O(T^2)$.

{{< spoiler text="Proof Sketch: The TRPO Lower Bound" >}}

**1. The Core Problem: Changing State Distributions**

The error between surrogate and true objective is:

$$
|J(\pi) - L_\mu(\pi)| = \left| \sum_s (d_\pi(s) - d_\mu(s)) \cdot \mathbb{E}_{a \sim \pi} [A_\mu(s,a)] \right|
$$

This simplifies to $O(\|d_\pi - d_\mu\|_1)$.

**2. The Simulation Lemma**

State distribution divergence accumulates linearly with time:

$$
D_{TV}(d_{\pi,t} \| d_{\mu,t}) \leq t \cdot D_{TV}^{\max}(\pi, \mu)
$$

**Simulation Lemma Proof:**

**Lemma:** $D_{TV}(d_{\pi,t} \| d_{\mu,t}) \leq t \cdot D_{TV}^{\max}(\pi, \mu)$

**Proof by Induction:**

Let $\delta_t = \|d_{\pi,t} - d_{\mu,t}\|_1 = 2 \cdot D_{TV}(d_{\pi,t} \| d_{\mu,t})$.

**Base case:** $\delta_0 = 0$ (same initial distribution).

**Inductive step:** Using the recursive state distribution:

$$
\delta_t \leq \delta_{t-1} + \epsilon_{\max}
$$

where $\delta_{t-1}$ is the propagated divergence, $\epsilon_{\max} = 2 \cdot D_{TV}^{\max}$ is the new divergence added at step $t$.

Unrolling: $\delta_t \leq t \cdot \epsilon_{\max}$.

**3. Total Error**

Summing over all timesteps:

$$
\text{Total Error} = O\left(\sum_{t=0}^{T-1} t \cdot D_{TV}^{\max}\right) = O(T^2 \cdot D_{TV}^{\max})
$$

**4. Connection to Discounted Setting**

The original TRPO paper used $\gamma$-discounting with a tighter bound:

$$
J(\pi) \geq L_\mu(\pi) - \frac{4\gamma \epsilon}{(1-\gamma)^2} \cdot (D_{TV}^{\max})^2
$$

where $\epsilon = \max_{s,a} |A_\mu(s,a)|$. Note this bound is quadratic in $D_{TV}^{\max}$, derived via a different technique (using KL divergence and Pinsker's inequality). The key insight for both bounds: **the penalty scales quadratically with effective horizon** ($T^2$ or $1/(1-\gamma)^2$).

{{< /spoiler >}}

### 5.4 Why TRPO Theory Does Not Solve This Problem: Three Reasons

**Reason 1: Theory-Practice Gap**

TRPO theory requires the trust region to **shrink with horizon**: $\delta \propto 1/T^2$.

PPO uses a **constant** clipping factor ($\epsilon = 0.2$) regardless of sequence length. This violates the theoretical requirement for long sequences.

**Reason 2: PPO is Really SGA**

PPO doesn't optimize the TRPO lower bound—it computes a clipped gradient estimator and feeds it to Adam. It's a clever **SGA method**, and our SGA Lemma is the right framework to analyze it.

**Reason 3: Blind Spot for Variance**

The TRPO framework uses $D_{TV}$ (or $D_{KL}$), which cannot measure variance. It provides no constraint on $\chi^2$-divergence, which causes variance explosion. This can be misleading: token-level IS keeps TV small (low bias in controlled settings), but there is no theoretical framework to compare its variance against alternatives like sequence-level IS.

---

## 6. When Bias is Tolerable vs. Catastrophic

### 6.1 Tolerable Bias: Controlled Off-Policy Setting

In standard PPO/GRPO with controlled off-policiness, the mismatch comes solely from **policy parameter updates**, which are actively controlled:

- PPO's clipping keeps $\pi$ close to $\mu$
- $\Delta_{\max}$ is small **by design**
- The $O(T^2 \Delta_{\max})$ bias is tolerable

**Focus:** Variance is the main concern → token-level IS is a reasonable solution.

### 6.2 Catastrophic Bias: Uncontrolled Off-Policy Setting

In settings with **uncontrolled off-policiness**, the mismatch has diverse sources:

- **Large policy changes**: Aggressive updates that move $\pi$ far from $\mu$
- **Stale samples**: Using old rollouts after many gradient updates
- **Distributed systems**: Inference engine discrepancies (vLLM vs FSDP kernels), quantization differences
- **MoE routing variations**: Expert selection changes between inference and training

These mismatches are **persistent and large**—not controlled by the algorithm.

**Result:** $\Delta_{\max}$ is no longer small. The $O(T^2 \Delta_{\max})$ bias becomes a significant error causing **optimization instability and collapse**.

**Focus:** Bias is now the primary concern.

---

## 7. Summary and Preview of Part 2

### Summary Table

| Aspect | Controlled Off-Policy | Uncontrolled Off-Policy |
|--------|----------------------|-------------------------|
| **Source** | Policy parameter updates | Large policy changes, stale samples, system discrepancies |
| **$\Delta_{\max}$** | Small (by design) | Large (persistent) |
| **Bias** | Tolerable | **Catastrophic** |
| **Variance** | Main concern | Secondary concern |
| **Token-level IS** | Reasonable | Creates intolerable bias |

### The Core Issue

The inequality $D_{TV} \leq \sqrt{\frac{1}{2}\chi^2}$ reveals a key asymmetry: small $\chi^2$ implies small TV, but **small TV does NOT imply small $\chi^2$**. You can have small bias but large variance.

- **Token-level IS (PPO/GRPO):** Low variance, but $O(T^2 \Delta_{\max})$ bias
- **Sequence-level IS:** Zero bias, but $O((1 + \bar{\chi}^2_{\max})^T)$ variance

We need estimators that balance this trade-off.

### Preview: Part 2

In [**Part 2**](https://richardli.xyz/rl-collapse-2), we will:

1. Prove that **Token-Level IS (PPO/GRPO)** has $O(T^2 \Delta_{\max})$ bias
2. Prove that **Sequence-Level IS** has $O((1 + \bar{\chi}^2_{\max})^T)$ variance
3. Derive **Sequence-Level Truncated IS (Seq-TIS)** that achieves a controllable bias-variance trade-off
