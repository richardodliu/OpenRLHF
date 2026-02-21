---
title: "Part 2: Applying the SGA Framework — Token v.s. Sequence-level Correction"
date: 2025-10-31
math: true
authors:
  - admin
  - jiacai-liu
tags:
  - Reinforcement Learning
  - Off-Policy
  - PPO
  - Importance Sampling
categories:
  - Research
series:
  - RL Collapse
series_order: 2
---

**Authors:** [Yingru Li](http://richardli.xyz), [Jiacai Liu](https://www.jiacailiu.cn/)

{{% callout note %}}
**Original Blog:** [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://richardli.xyz/rl-collapse)
{{% /callout %}}

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

## Recap: Analytical Framework

**In [Part 1](https://richardli.xyz/rl-collapse-1), we established our analytical framework. We learned that the Stochastic Gradient Ascent (SGA) Lemma is the core tool for analyzing any gradient estimator, $\hat{g}$, used in an off-policy system where we sample from $\mu$ but optimize $\pi$.**

The SGA Lemma reveals two distinct failure modes:

- **Bias (Term B):** The estimator points in the wrong direction, pushing the optimizer toward an incorrect solution. We measure this with $D_{TV}$.
- **Variance (Term C):** The estimator is so noisy it forces our learning rate to zero, stalling all progress. We measure this with $\chi^2$-divergence.

Our goal is simple: find an estimator $\hat{g}$ that simultaneously controls both bias and variance.

We will now analyze the most common estimators. For this analysis, we'll define

{{< math >}}
$$
f(y) := \nabla_\theta \log \pi(y|x) \cdot R(y|x)
$$
{{< /math >}}

as our target function, where the true gradient is $g = \mathbb{E}_\pi[f(y)]$.

---

## Analysis 1: Sequence-Level Importance Sampling (Seq-IS)

This is the most direct estimator in theory. It corrects for the mismatch by re-weighting every sample by the full sequence-level ratio, $\rho(y) = \pi(y) / \mu(y)$.

- **The Estimator:** {{< math >}}$\hat{g}_{\text{seq}} = \rho(y) \cdot f(y)${{< /math >}}

### The Analysis

**Term B (Bias):**
This estimator is perfectly unbiased. By the definition of importance sampling:

{{< math >}}
$$
\mathbb{E}_\mu[\hat{g}_{\text{seq}}] = \mathbb{E}_\mu\left[ \frac{\pi(y)}{\mu(y)} f(y) \right] = \sum_y \mu(y) \frac{\pi(y)}{\mu(y)} f(y) = \sum_y \pi(y) f(y) = \mathbb{E}_\pi[f(y)] = g
$$
{{< /math >}}

It will always converge to the correct solution... if it ever converges.

**Term C (Variance):**
The variance of this estimator scales with the second moment of the IS ratio,

{{< math >}}
$$
\mathbb{E}_\mu[\rho(y)^2].
$$
{{< /math >}}

The problem is that for autoregressive sequences, this second moment explodes exponentially with sequence length $T$. Even a tiny, 1% per-token mismatch ($\bar{\chi}^2_{\max} = 0.01$) over a 200-token sequence ($T=200$) leads to a variance multiplier of $(1.01)^{200} \approx 7.3$. A 5% mismatch gives $(1.05)^{200} \approx 17,292$. This is impractical for real applications.

{{< spoiler text="Derivation: The Exponential Variance of Seq-IS O((1 + χ²_max)^T)" >}}

A direct computation yields that

$$
\mathbf{Var}(\hat{g}_{\mathrm{seq}}) = \mathbb{E}\_{\mu}\left[ \left\| \hat{g}\_{\mathrm{seq}} \right\|^2 \right] - \left\| \mathbb{E}\_{\mu}\left[ \hat{g}\_{\mathrm{seq}} \right] \right\|^2
$$

$$
= \mathbb{E}\_{\mu}\left[ \rho^2(y) R^2(y) \left\| \nabla\_{\theta}\log \pi(y) \right\|^2 \right] - \left\| g \right\|^2
$$

Assuming that $\sup\_y \left\| \nabla\_{\theta}\log \pi(y) \right\|^2 \le M$ and $\forall y: |R(y)| \le R\_{\text{max}}$, then the variance is upper-bounded by the second moment:

$$
\mathbf{Var}(\hat{g}\_{\mathrm{seq}}) \le M \cdot R\_{\max} \cdot \mathbb{E}\_{\mu}\left[ \rho^2(y) \right] - \left\| g \right\|^2
$$

Let's analyze the second moment.

The ratio $\rho(y)$ is a product of per-token ratios: $\rho(y) = \prod\_{t=0}^{T-1} \rho\_t = \prod\_{t=0}^{T-1} \frac{\pi(y\_t|x,y\_{\lt t})}{\mu(y\_t|x,y\_{\lt t})}$

Using the tower property of expectation and conditional independence:

$$
\mathbb{E}\_\mu[\rho(y)^2] = \mathbb{E}\_\mu\left[ \left(\prod_t \rho_t\right)^2 \right] = \mathbb{E}\_\mu\left[ \rho_{0:T-2}^2 \cdot \mathbb{E}\_\mu[\rho_{T-1}^2 | s_{T-1}] \right]
$$

Each per-token expectation is, by definition, $1$ plus its $\chi^2$-divergence:

$$
\mathbb{E}\_\mu[\rho_t^2 | s_t] = 1 + \chi^2(\pi(\cdot|s_t) \| \mu(\cdot|s_t)) \le 1 + \bar{\chi}^2\_{\max}
$$

Let $\bar{\chi}^2\_{\max} = \max\_{t, s\_t} \chi^2(\pi(\cdot|s\_t) \| \mu(\cdot|s\_t))$. If any mismatch exists, $\bar{\chi}^2\_{\max} \gt 0$.
This gives the exponential upper bound:

$$
\mathbb{E}\_\mu[\rho(y)^2] \le \prod_{t=0}^{T-1} (1 + \bar{\chi}^2\_{\max}) = (1 + \bar{\chi}^2\_{\max})^T
$$

This proves the variance is $\mathbf{Var}(\hat{g}\_{\text{seq}}) = O((1 + \bar{\chi}^2\_{\max})^T)$, which grows too fast for long sequences.

{{< /spoiler >}}

### Self-Normalized Sequence-Level IS (SNIS)

Before we give up on Sequence-Level IS and resort to Clipping or Token-level methods, is there a statistically principled way to fix the variance? This leads to Self-Normalized Importance Sampling (SNIS).

Standard IS estimates the expectation as an average:

{{< math >}}
$$
\hat{g}_{\text{is}} = \frac{1}{N} \sum_{i=1}^N \rho(y_i) f(y_i).
$$
{{< /math >}}

SNIS replaces the fixed denominator $N$ with the sum of the weights:

{{< math >}}
$$
\hat{g}_{\text{snis}} = \frac{\sum_{i=1}^N \rho(y_i) f(y_i)}{\sum_{i=1}^N \rho(y_i)}
$$
{{< /math >}}

**The Logic:**

This acts as a stabilizer. If a specific sample trajectory $y_k$ has a massive weight $\rho_k \to \infty$, it blows up the numerator, but it *also* blows up the denominator. The ratio stays bounded (approaching $f(y_k)$).

**The Analysis:**
**Bias:** Unlike standard IS, SNIS is **biased** for finite $N$ (bias is $O(1/N)$). However, it is consistent (bias vanishes as $N \to \infty$).
**Variance:** It successfully caps the variance of the *gradient norm* You won't get a gradient of $10^{6}$.

**Conclusion: The "Effective Sample Size" Problem**
While SNIS prevents numerical explosion, it fails in a different way: **Sample Collapse**.
In high-dimensional sequence generation, the variance of the weights $\rho$ is so high that often a single "lucky" sample dominates the entire sum.

{{< math >}}
$$
\hat{g}_{\text{snis}} \approx \frac{\rho(y_{\text{lucky}}) f(y_{\text{lucky}})}{\rho(y_{\text{lucky}})} = f(y_{\text{lucky}})
$$
{{< /math >}}

When this happens, the estimator effectively ignores all your data except for that one sample. Your "Effective Sample Size" (ESS) drops to 1.

{{< math >}}
$$
\text{ESS} = \frac{(\sum_i \rho_i)^2}{\sum_i \rho_i^2} \approx 1
$$
{{< /math >}}

So while SNIS is "stable" (it doesn't explode), it becomes incredibly inefficient. We are discarding $99\%$ of our batch data. We need a way to keep more data while controlling the outliers. We need Truncation.

---

## Analysis 2: Naive & Token-Level IS

This family of estimators "solves" the exponential variance problem by avoiding the full sequence-level product $\rho = \prod_t \rho_t$. The Naive estimator ignores it entirely, and Token-Level IS (Token-IS) handles each token's ratio $\rho_t$ independently.

- **The Estimator (Naive):** {{< math >}}$\hat{g}_{\text{naive}}(y) = R(y|x) \cdot \nabla_\theta \log \pi(y|x)${{< /math >}}
- **The Estimator (Token-IS / PPO-like):**

{{< math >}}
$$
\hat{g}_{\text{tok}}(y) = \sum_{t=0}^{T-1} \frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})} A(s_t, y_t) \nabla_\theta \log \pi(y_t|x, y_{\lt t})
$$
{{< /math >}}

### The Analysis

By breaking the exponential product, both estimators have a variance that is polynomial in $T$. This is because the score function $f(y)$ itself has a variance that scales polynomially, $O(T^2)$.

- **Naive:** The variance is bounded by a mismatch-independent constant, $\mathbf{Var}(\hat{g}_{\text{naive}}) \le O(T^2)$.
- **Token-IS:** The variance is also polynomial, scaling as $O(T^2(1+\bar{\chi}^2))$.

{{< spoiler text="Derivation: The Polynomial Variance" >}}

The variance is always bounded by the second moment: $\mathbf{Var}(\hat{g}) \le \mathbb{E}[\|\hat{g}\|^2]$. We just need to show this second moment is $O(\text{poly}(T))$.

**1. Naive Estimator:**

Let $f(y) = \hat{g}\_{\text{naive}}(y) = R(y|x) \cdot \nabla\_\theta \log \pi(y|x)$.

As we show in the "Factor 1" part of the bias proof below, the score function $\nabla\_\theta \log \pi(y|x) = \sum\_t \nabla\_\theta \log \pi\_t$ has a magnitude that scales as $O(T)$.

$$
\|f(y)\| = |R(y)| \cdot \|\nabla\_\theta \log \pi(y|x)\| \le R\_{\max} \cdot O(T)
$$

The second moment is therefore bounded by the square of this:

$$
\mathbb{E}[\|\hat{g}\_{\text{naive}}\|^2] \le \mathbb{E}[(\sup_y \|f(y)\|)^2] = O(T^2)
$$

The variance is definitively polynomial.

**2. Token-Level IS (Token-IS) Estimator:**
This estimator is a sum of $T$ terms: $\hat{g}\_{\text{tok}} = \sum\_{t=0}^{T-1} X\_t$.
The variance of a sum is $\mathbf{Var}(\sum\_t X\_t) = \sum\_t \mathbf{Var}(X\_t) + \sum\_{t \neq t'} \mathbf{Cov}(X\_t, X\_{t'})$.
This has $T$ diagonal (variance) terms and $T(T-1) \approx O(T^2)$ off-diagonal (covariance) terms.

**Diagonal Terms ($T$ of them):** We bound $\mathbb{E}[\|X\_t\|^2]$.

$$
\mathbb{E}[|X_t|^2] = \mathbb{E}\left[\left\| \frac{\pi_t}{\mu_t} A_t \nabla\_\theta \log \pi_t \right\|^2\right] \le (\sup \|A_t \nabla\_\theta \log \pi_t\|)^2 \cdot \mathbb{E}\left[\left(\frac{\pi_t}{\mu_t}\right)^2\right]
$$

The $A\_t$ and $\nabla\_\theta \log \pi\_t$ terms are bounded by constants. The expectation is $\mathbb{E}[\rho\_t^2] \le (1+\bar{\chi}^2)$.
So, each diagonal term is $O(1+\bar{\chi}^2)$. The sum of all $T$ diagonal terms is $O(T(1+\bar{\chi}^2))$.

**Cross Terms ($O(T^2)$ of them):** We bound $\mathbf{Cov}(X\_t, X\_{t'})$.

By the Cauchy-Schwarz inequality, $|\mathbb{E}[X\_t^\top X\_{t'}]| \le \sqrt{\mathbb{E}[\|X\_t\|^2] \mathbb{E}[\|X\_{t'}\|^2]}$.

Since we just showed $\mathbb{E}[\|X\_t\|^2]$ is $O(1+\bar{\chi}^2)$, the covariance/cross-term is also $O(1+\bar{\chi}^2)$.

The sum of all $O(T^2)$ cross terms is $O(T^2(1+\bar{\chi}^2))$.

**Total Variance:**

$$
\mathbf{Var}(\hat{g}\_{\text{tok}}) = O(T(1+\bar{\chi}^2)) + O(T^2(1+\bar{\chi}^2)) = O(T^2(1+\bar{\chi}^2))
$$

Both estimators have variance that is polynomial in $T$, not exponential.

{{< /spoiler >}}

---

Both estimators fail for the same reason: by breaking the sequence-level importance sampling, they introduce a systematic bias. The core issue is that this bias scales quadratically with sequence length $T$.

This happens because these estimators are optimizing a **biased surrogate objective** $L_\mu(\pi)$ (based on the "old" state distribution $d_\mu$) instead of the **true objective** $J(\pi)$ (based on the "new" state distribution $d_\pi$). They *both* inherit the $O(T^2)$ bias of this incorrect objective. The derivation below proves this bias for the Naive estimator, and the conclusion explains why it applies to Token-IS as well.

{{< spoiler text="Derivation: The O(T² Δ_max) Bias" >}}

Let's analyze the bias of the **Naive estimator** directly. The bias is the difference between what we *compute* (expectation under $\mu$) and what we *want* (expectation under $\pi$):

$$
\text{Bias}(\hat{g}\_{\text{naive}}) = \mathbb{E}\_\mu[f(y)] - \mathbb{E}\_\pi[f(y)]
$$

where $f(y) = \hat{g}\_{\text{naive}}(y) = R(y|x) \cdot \nabla\_\theta \log \pi(y|x)$.

This is the difference in the expectation of the same function $f(y)$ under two different distributions, $\mu$ and $\pi$. We can bound this difference:

$$
\|\text{Bias}(\hat{g}\_{\text{naive}})\|_2 = \left\| \sum_y (\mu(y) - \pi(y)) f(y) \right\|_2 \le \sup_y \|f(y)\| \cdot \sum_y |\mu(y) - \pi(y)|
$$

This is bounded by the "worst-case" value of $f(y)$ multiplied by the total difference in probabilities. The total probability difference is exactly twice the Total Variation (TV) distance:

$$
\sum_y |\mu(y) - \pi(y)| = 2 D_{TV}(\mu \| \pi)
$$

$$
\|\text{Bias}(\hat{g}\_{\text{naive}})\|_2 \le 2 \sup_y \|f(y)\| \cdot D_{TV}(\mu \| \pi)
$$

The $O(T^2)$ bias comes from the fact that both of these terms scale linearly with $T$.

---

**Factor 1: Bounding the Function,** $\sup\_y \|f(y)\| = O(T)$

Our function $f(y)$ is the product of the reward (bounded by $R\_{\max}$) and the full-sequence score function. The score function is a sum of $T$ per-token score functions:

$$
\nabla\_\theta \log \pi(y|x) = \sum_{t=0}^{T-1} \nabla\_\theta \log \pi(y_t|x, y_{\lt t})
$$

Since each per-token score is bounded (let's say by a constant $K\_t$) for softmax parameterization, the full sequence score is bounded by $O(T \cdot K\_t)$. Therefore:

$$
\sup_y \|f(y)\| \le R\_{\max} \cdot O(T) = O(T)
$$

---

**Factor 2: Bounding the Divergence,** $D\_{TV}(\mu \| \pi) = O(T \Delta\_{\max})$

This is the **Simulation Lemma** (or hybrid argument). It proves that the total sequence-level TV distance is bounded by the *sum* of the per-token TV distances.

a. Define Hybrid Distributions:

We create a "bridge" of distributions from $\pi$ to $\mu$, swapping one token at a time.

- $H\_0(y) = \pi(y)$
- $H\_t(y) = \left( \prod\_{i=0}^{t-1} \mu(y\_i|s\_i) \right) \cdot \left( \prod\_{j=t}^{T-1} \pi(y\_j|s\_j) \right)$ (First $t$ tokens from $\mu$, the rest from $\pi$)
- $H\_T(y) = \mu(y)$

b. Use the Triangle Inequality (Telescoping Sum):

The total distance is bounded by the sum of the distances of each "step" in the bridge.

$$
D_{TV}(\pi \| \mu) = D_{TV}(H_0 \| H_T) \le \sum_{t=0}^{T-1} D_{TV}(H_t \| H_{t+1})
$$

**c. Analyze a Single Step** $D\_{TV}(H\_t \| H\_{t+1})$:

$H\_t$ and $H\_{t+1}$ are identical *except* at token $t$, where one uses $\pi(y\_t|s\_t)$ and the other uses $\mu(y\_t|s\_t)$.

$$
D_{TV}(H_t \| H_{t+1}) = \frac{1}{2} \sum_y |H_t(y) - H_{t+1}(y)|
$$

Let's call the common prefix $q(s\_t) = \prod\_{i\lt t} \mu(y\_i|s\_i)$ and the common suffix $p(y\_{\gt t}) = \prod\_{j\gt t} \pi(y\_j|s\_j)$.

$$
D_{TV}(H_t \| H_{t+1}) = \frac{1}{2} \sum_{s_t} q(s_t) \sum_{y_t} |\pi(y_t|s_t) - \mu(y_t|s_t)| \sum_{y_{\gt t}} p(y_{\gt t})
$$

Factor out the common terms. The innermost sum $\sum\_{y\_{\gt t}} p(y\_{\gt t})$ is 1 (it's the probability of all possible futures):

$$
D_{TV}(H_t \| H_{t+1}) = \sum_{s_t} q(s_t) \cdot D_{TV}(\pi(\cdot|s_t) \| \mu(\cdot|s_t))
$$

We recognize: $q(s\_t)$ is $d\_t^\mu(s\_t)$, the probability of being in state $s\_t$ under $\mu$. This gives:

$$
D_{TV}(H_t \| H_{t+1}) = \mathbb{E}_{s_t \sim d_t^\mu} [D_{TV}(\pi(\cdot|s_t) \| \mu(\cdot|s_t))]
$$

**d. Combine and Conclude:**

Substitute this back into the sum from (b):

$$
D_{TV}(\pi \| \mu) \le \sum_{t=0}^{T-1} \mathbb{E}_{s_t \sim d_t^\mu} [D_{TV}(\pi(\cdot|s_t) \| \mu(\cdot|s_t))]
$$

Let $\Delta\_{\max} = \max\_{t,s\_t} D\_{TV}(\pi(\cdot|s\_t) \| \mu(\cdot|s\_t))$. The bound becomes:

$$
D_{TV}(\pi \| \mu) \le T \cdot \Delta\_{\max}
$$

---

**Final Conclusion: The $O(T^2)$ Bias Applies to Both**

We have just proved that the **Naive estimator** has a bias of $O(T^2 \Delta\_{\max})$:

$$
\|\text{Bias}(\hat{g}\_{\text{naive}})\|_2 \le (2 \cdot O(T)) \cdot (O(T \Delta\_{\max})) = O(T^2 \Delta\_{\max})
$$

**Why does this also apply to the Token-Level IS (PPO-like) estimator?**

Because the $O(T^2 \Delta\_{\max})$ bias doesn't come from the specific *form* of the estimator (like using $R(y)$ vs $A(s\_t, y\_t)$). It comes from the **sampling distribution** itself.

1. The **true gradient** $g = \nabla J(\pi)$ is an expectation over the *new* policy's state distribution, $d\_\pi$.
2. *Both* $\hat{g}\_{\text{naive}}$ and $\hat{g}\_{\text{tok}}$ are computed on samples from $\mu$, so their expectations are taken over the *old* policy's state distribution, $d\_\mu$.

The $O(T \Delta\_{\max})$ error from the Simulation Lemma is the *fundamental difference* between the true state distribution $d\_\pi$ and the sampling distribution $d\_\mu$. Any estimator that computes its expectation under $d\_\mu$ (like Naive and Token-IS) is optimizing a surrogate objective $L\_\mu(\pi)$, not the true objective $J(\pi)$.

Therefore, both estimators are biased relative to the true gradient $g$, and this bias is $O(T^2 \Delta\_{\max})$.

{{< /spoiler >}}

---

## Analysis 3: Token-Level Truncated IS (Token-TIS and The PPO Paradigm)

Standard RL practice (specifically **Proximal Policy Optimization**, or PPO) addresses the variance problem effectively. Can we apply the same clipping technique here?
This brings us to **Token-Level Truncated IS (Token-TIS)**. This is the theoretical backbone of PPO. The idea is straightforward: if the importance ratio $\rho_t$ gets too large, we clip it to prevent the gradient from exploding.

- **The Estimator:**

{{< math >}}
$$
\hat{g}_{\text{tl-tis}}(y) = \sum_{t=0}^{T-1} \min\left(\frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})}, C\right) A(s_t, y_t) \nabla_\theta \log \pi(y_t|x, y_{\lt t})
$$
{{< /math >}}

(Note: While PPO uses a specific "pessimistic lower bound" objective, its core mechanism is identical to this estimator: it limits the magnitude of the update based on the per-token ratio deviation.)

### The Analysis

This estimator is perfectly stable. By clipping each per-token ratio $\rho_t \le C$, the variance of each term in the sum is controlled.

**The total variance remains polynomial, scaling as $O(T^2 C^2)$.**

{{< spoiler text="Derivation: The Polynomial Variance" >}}

This proof is nearly identical to the variance proof for Token-IS in Analysis 2.

The estimator is a sum of $T$ terms: $\hat{g}\_{\text{tl-tis}} = \sum\_{t=0}^{T-1} X\_t'$, where $X\_t' = \min(\rho\_t, C) A\_t \nabla\_\theta \log \pi\_t$.

The variance is $\mathbf{Var}(\sum\_t X\_t') = \sum\_t \mathbf{Var}(X\_t') + \sum\_{t \neq t'} \mathbf{Cov}(X\_t', X\_{t'})$. This is again $O(T)$ diagonal terms and $O(T^2)$ cross terms.

**Diagonal Terms ($T$ of them):** We bound $\mathbb{E}[\|X\_t'\|^2]$.

$$
\mathbb{E}[|X_t'|^2] = \mathbb{E}\left[\left| \min(\rho_t, C) A_t \nabla\_\theta \log \pi_t \right|^2\right] \le (\sup |A_t \nabla\_\theta \log \pi_t|)^2 \cdot \mathbb{E}\left[\min(\rho_t, C)^2\right]
$$

Since $\min(\rho\_t, C)^2 \le C^2$ by definition, the expectation is bounded by $C^2$. This is even *better* than the $O(1+\bar{\chi}^2)$ bound for the unclipped Token-IS. Each diagonal term is $O(C^2)$. The sum of $T$ terms is $O(T C^2)$.

**Cross Terms ($O(T^2)$ of them):**

By Cauchy-Schwarz, $|\mathbb{E}[X\_t'^\top X\_{t'}']| \le \sqrt{\mathbb{E}[\|X\_t'\|^2] \mathbb{E}[\|X\_{t'}'\|^2]}$. Since the second moments are $O(C^2)$, each cross-term is also $O(C^2)$. The sum of all $O(T^2)$ cross terms is $O(T^2 C^2)$.

**Total Variance:**

$$
\mathbf{Var}(\hat{g}\_{\text{tl-tis}}) = O(TC^2) + O(T^2 C^2) = O(T^2 C^2)
$$

The variance is successfully controlled. This is why PPO is widely used—it rarely exhibits numerical instability.

{{< /spoiler >}}

---

**Term B (Bias):**

This is the critical insight. This estimator does not fix the $O(T^2 \Delta\_{\max})$ bias we proved in Analysis 2.

Why? Because that bias comes from the **state distribution mismatch** ($d\_\mu \neq d\_\pi$), not from the variance of the individual importance weights $\rho\_t$.

Clipping $\rho\_t$ is a *variance reduction* technique, but it doesn't fix the underlying *objective*. This estimator just introduces a new, *secondary* truncation bias on top of the *existing,* $O(T^2 \Delta\_{\max})$ bias.

{{< spoiler text="Derivation: The 'Bias-on-Bias' Problem" >}}

Let's be precise about the two sources of bias.

**1. The Bias (from Analysis 2):**

This is the bias we already have, before we even add clipping. It's the difference between the true gradient $g$ and the expectation of the unclipped token-level estimator $\hat{g}\_{\text{tok}}$. This bias is large because it comes from optimizing on the wrong state distribution $d\_\mu$.

$$
B\_{\text{fatal}} = \mathbb{E}[\hat{g}\_{\text{tok}}] - g = O(T^2 \Delta\_{\max})
$$

**2. The "New" Truncation Bias:**

Our new estimator, $\hat{g}\_{\text{tl-tis}}$, isn't an unbiased estimator for $g$. It's not even an unbiased estimator for $\mathbb{E}[\hat{g}\_{\text{tok}}]$. By clipping $\rho\_t \to \min(\rho\_t, C)$, we have introduced a *new* bias, which is the difference between the expectation of the clipped estimator and the unclipped one:

$$
B\_{\text{trunc}} = \mathbb{E}[\hat{g}\_{\text{tl-tis}}] - \mathbb{E}[\hat{g}\_{\text{tok}}]
$$

**3. The Total Bias:**

The total bias of our new estimator, relative to the *true* gradient $g$, is the sum of both:

$$
\text{Total Bias} = \mathbb{E}[\hat{g}\_{\text{tl-tis}}] - g
$$

We can add and subtract $\mathbb{E}[\hat{g}\_{\text{tok}}]$ (the unclipped estimator's expectation) to see both parts:

$$
\text{Total Bias} = \underbrace{(\mathbb{E}[\hat{g}\_{\text{tl-tis}}] - \mathbb{E}[\hat{g}\_{\text{tok}}])}\_{B\_{\text{trunc}}} + \underbrace{(\mathbb{E}[\hat{g}\_{\text{tok}}] - g)}\_{B\_{\text{fatal}}}
$$

$$
\text{Total Bias} = B\_{\text{trunc}} + O(T^2 \Delta\_{\max})
$$

**Conclusion:**

The total bias is still dominated by the $O(T^2 \Delta\_{\max})$ term. All we've done is add more bias ($B\_{\text{trunc}}$) in an attempt to solve a variance problem, without ever addressing the fundamental $O(T^2)$ bias.

This proves that **the problem cannot be solved at the token level if either the sequence length or off-policiness is large.** PPO is a variance control method that prevents explosion but cannot address the true sequence-level objective optimization problem. The bias is a sequence-level problem, and it requires a sequence-level solution.

{{< /spoiler >}}

---

## Trade-offs: Sequence-Level Truncated IS (Seq-TIS)

We are trapped.

- **Seq-IS:** 0 Bias (Good!), $O((1 + \bar{\chi}^2_{\max})^T)$ Variance
- **Naive/Token-IS/Token-TIS:** $O(T^2)$ Variance (Good!), $O(T^2 \Delta_{\max})$ Bias

Neither is usable. This brings us to the **Sequence-Level Truncated IS (Seq-TIS)** estimator.

The key insight is this:

1. Let's go back to **Seq-IS**. Its *only* problem is exponential variance.
2. Why? Because *rare* samples can have an *enormous* sequence ratio $\rho \to \infty$.
3. What's the simplest fix? **Just clip the full sequence ratio** $\rho$**.**

- **The Estimator:** {{< math >}}$\hat{g}_{\text{sl-tis}}(y) = \min\left(\prod_{t=0}^{T-1} \frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})}, C\right) \cdot f(y)${{< /math >}}

### The Analysis

This estimator introduces a new "knob" $C$. Let's see how $C$ affects our two failure modes.

- **Term C (Variance):**
We fixed it! By definition, the ratio $\min(\rho, C)$ can never be larger than $C$.

{{< math >}}
$$
\mathbf{Var}(\hat{g}_{\text{sl-tis}}) \le \mathbb{E}_\mu[\|\min(\rho, C) f(y)\|^2] \le \mathbb{E}_\mu[C^2 \|f(y)\|^2]
$$
{{< /math >}}

- The variance is now bounded by $\mathbf{Var} \le K^2 C^2 = O(T^2 C^2)$. It is **finite and controllable.**

- **Term B (Bias): ...A NEW TRADE-OFF!**
We are no longer unbiased. By clipping, we've introduced a new bias (Term B).

{{< math >}}
$$
\mathbf{Bias}(\hat{g}_{\text{sl-tis}}) = \mathbb{E}[\hat{g}_{\text{sl-tis}}] - g = \mathbb{E}_\mu[\min(\rho, C) f(y)] - \mathbb{E}_\mu[\rho f(y)]
$$
{{< /math >}}

- This bias is $\mathbf{Bias} = \mathbb{E}_\mu[(\min(\rho, C) - \rho) f(y)]$*. This term is only non-zero for the tail (*$\rho \gt C$*), and **it can be shown that** this bias is inversely proportional to* $C$*:*

{{< math >}}
$$
\|\mathbf{Bias}(\hat{g}_{\text{sl-tis}})\|_2 \le \frac{2 K (1 + \chi^2)}{C}
$$
{{< /math >}}

- *Critically, this bias is not the* $O(T^2 \Delta_{\max})$ state-mismatch bias. This is a **controllable truncation bias.**

### The Optimal Balance

We have just derived the Trade-off in its final, explicit form.

- To lower **Bias**, we must **increase** $C$.
- To lower **Variance**, we must **decrease** $C$.

We are no longer stuck with (0, $\infty$) or ($\infty$, 0). We can now choose a $C$ that minimizes the total error: $\text{MSE}(C) \approx \text{Bias}^2 + \text{Var}$.

{{< math >}}
$$
\text{MSE}(C) \le \frac{4K^2(1+\chi^2)^2}{C^2} + K^2 C^2
$$
{{< /math >}}

This is a stable, U-shaped function. We can use basic calculus to find the $C$ that minimizes this MSE bound. The provably optimal solution is:

{{< math >}}
$$
C^* = \sqrt{2(1+\chi^2)}
$$
{{< /math >}}

The optimal clipping threshold $C^*$ is a direct function of the $\chi^2$-divergence (our variance metric). It *uses* the mismatch to set the bias-variance trade-off correctly.

Plugging $C^*$ *back in gives the optimal MSE:*

{{< math >}}
$$
\text{MSE} \le \frac{4K^2(1+\chi^2)^2}{2(1+\chi^2)} + K^2(2(1+\chi^2)) = 4 K^2 (1+\chi^2)
$$
{{< /math >}}

This is the fundamental limit. We have found an estimator that achieves this limit.

---

## Conclusion

The analysis is complete.

- **Seq-IS** is theoretically unbiased but impractical due to exponential variance.
- **Naive/Token-IS/Token-TIS** have polynomial variance but bias that grows quadratically with sequence length.
- **Seq-TIS** correctly addresses the sequence-level trade-off and provides a tunable parameter ($C$) to balance bias and variance.

| **Estimator** | **Bias (Term B)** | **Variance (Term C)** |
| --- | --- | --- |
| Naive/Token-TIS/PPO | $O(T^2 \Delta_{\max})$ | $O(T^2(1+\bar{\chi}^2))$ |
| Seq-IS | $0$ | $O((1 + \bar{\chi}^2_{\max})^T)$ |
| Seq-TIS | $O(T(1+\chi^2)/C)$  | $O(T^2 C^2)$  |

---

## Preview: Part 3 — Trust Region Optimization via Sequence Masking

We have mathematically "solved" the problem. We found **Seq-TIS**, the estimator that achieves a controllable bias-variance trade-off in a standard statistical setting.

But as we deploy RL to **Reasoning Models (CoT)** and **Long-Horizon Agents**, we encounter two phenomena that break our clean theoretical assumptions:

1. **Out-of-Distribution (OOD) High-Weight Samples:** In theory, a sample with a massive weight ($\rho = 10,000$) is a "highly informative" signal. In reality, such extreme ratios typically arise when $\mu(y)$ is near the numerical precision floor—these are OOD samples caused by numerical precision artifacts, reward exploitation, or distribution shift beyond the valid IS regime. **Clipping** only limits the weight while the problematic sample still participates in gradient updates. We need to **Reject** it.
2. **Length-Dependent Rejection Bias:** Standard importance sampling systematically penalizes long sequences. If we use Seq-TIS on a Chain-of-Thought model, it will suppress long reasoning chains and bias the model toward short, shallow answers.

In [**Part 3**](https://richardli.xyz/rl-collapse-3), we move from theoretical analysis to practical application. We will introduce **Sequence-Level Masked IS (Seq-MIS)** and **Geometric Rejection Sampling (Geo-RS)**—the technical improvements needed to stabilize Agentic RL.

**[[Read Part 3: Trust Region Optimization via Sequence Masking](https://richardli.xyz/rl-collapse-3)]**
