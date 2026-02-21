---
title: "Part 3: Trust Region Optimization via Sequence Masking"
date: 2025-11-04
math: true
authors:
  - admin
  - jiacai-liu
tags:
  - Reinforcement Learning
  - Off-Policy
  - Trust Region
  - Chain-of-Thought
categories:
  - Research
series:
  - RL Collapse
series_order: 3
---

**Authors:** [Yingru Li](http://richardli.xyz), [Jiacai Liu](https://www.jiacailiu.cn/)

{{% callout note %}}
**Original Blog:** [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://richardli.xyz/rl-collapse)
{{% /callout %}}

{{% callout note %}}
**Series Context**

- [**Part 1**](https://richardli.xyz/rl-collapse-1): We established the SGA (Stochastic Gradient Ascent) framework and identified two failure modes of off-policy mismatch: Bias (measured by $D_{TV}$) and Variance (measured by $\chi^2$-divergence).
- [**Part 2**](https://richardli.xyz/rl-collapse-2): We analyzed gradient estimators and showed that Token-level IS (PPO/GRPO) has $O(T^2 \Delta_{\max})$ bias, while Sequence-Level Truncated IS (Seq-TIS) achieves a controllable bias-variance trade-off via clipping: $\rho(y) \to \min(\rho(y), C)$.
{{% /callout %}}

{{% callout note %}}
**TL;DR**

In a standard statistical setting, Part 2 solved the problem with Seq-TIS.

However, when training Agents or Reasoning Models (Chain-of-Thought), two practical phenomena violate the assumptions underlying Seq-TIS:

1. **Out-of-Distribution (OOD) High-Weight Samples:** Extremely high importance weights ($\rho \gg C$) often correspond to samples outside the behavior policy's support—numerical errors or distribution shift artifacts. Clipping these samples still includes them in the gradient update. **Solution: Enforce a Hard Trust Region via Rejection/Masking (Seq-MIS).**
2. **Length-Dependent Rejection Bias:** The importance ratio $\rho(y) = \prod_t \rho_t$ grows exponentially with sequence length $T$, causing systematic rejection of long sequences regardless of per-step quality. **Solution: Geometric Sequence Masking (Geo-Mask), which enforces a Per-Token Trust Region using a length-normalized KL divergence criterion.**

| Pathology | What Breaks | Why Standard Methods Fail |
|-----------|-------------|---------------------------|
| OOD High-Weight Samples | Samples with $\rho \gg C$ lie outside trust region | Clipping retains them with weight $C$—still corrupts gradients |
| Length-Dependent Bias | $\rho(y) = \prod_t \rho_t$ grows exponentially with $T$ | Fixed threshold $C$ systematically rejects long sequences |
| Theory-Practice Gap | TRPO requires trust region $\propto 1/T^2$ | PPO/GRPO use fixed clipping $\epsilon$ regardless of $T$ |
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

## Recap: The Trust Region Framework

In Part 1, we established the theoretical foundation for trust region optimization. Here we recap the key results that motivate the methods in this part.

### The Surrogate Objective and Its Limitations

Consider an autoregressive language model generating a sequence $y = (y_0, y_1, \ldots, y_{T-1})$ given prompt $x$. When optimizing a policy $\pi$ using samples from a behavior policy $\mu$, we cannot directly optimize the true objective $J(\pi)$. Instead, we optimize a **surrogate objective**:

{{< math >}}
$$
L_\mu(\pi) = J(\mu) + \sum_{t=0}^{T-1} \mathbb{E}_{(x, y_{\lt t}) \sim d_{\mu,t}} \mathbb{E}_{y_t \sim \pi(\cdot|x, y_{\lt t})} [A_\mu(x, y_{\le t})]
$$
{{< /math >}}

where $d_{\mu,t}$ is the context distribution at step $t$ under $\mu$, and $A_\mu(x, y_{\le t})$ is the advantage of generating token $y_t$ in context $(x, y_{\lt t})$.

The surrogate is a first-order approximation ([Kakade & Langford, 2002](https://dl.acm.org/doi/10.5555/645531.656005); [Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) that satisfies:
- $L_\mu(\mu) = J(\mu)$ (equal values at $\pi = \mu$)
- {{< math >}}$\nabla L_\mu(\pi)\big|_{\pi=\mu} = \nabla J(\pi)\big|_{\pi=\mu}${{< /math >}} (equal gradients at $\pi = \mu$)

However, the approximation degrades as $\pi$ moves away from $\mu$.

**The Surrogate Gradient (Token-IS):** Taking the gradient of $L_\mu(\pi)$:

{{< math >}}
$$
\nabla L_\mu(\pi) = \sum_{t=0}^{T-1} \mathbb{E}_{(x, y_{\lt t}) \sim d_{\mu,t}} \mathbb{E}_{y_t \sim \pi(\cdot|x, y_{\lt t})} [A_\mu(x, y_{\le t}) \nabla \log \pi(y_t|x, y_{\lt t})]
$$
{{< /math >}}

To estimate this from samples $y \sim \mu$, we apply importance sampling at each token:

{{< math >}}
$$
\nabla L_\mu(\pi) = \mathbb{E}_{y \sim \mu} \left[ \sum_{t=0}^{T-1} \rho_t \cdot A_t \nabla \log \pi(y_t|x, y_{\lt t}) \right]
$$
{{< /math >}}

where $\rho_t = \frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})}$ and $A_t = A_\mu(x, y_{\le t})$. This is the **Token-IS gradient**—the foundation for PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) and GRPO ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)).

### The TRPO Lower Bound

The **Performance Difference Lemma** ([Kakade & Langford, 2002](https://dl.acm.org/doi/10.5555/645531.656005)) quantifies the gap between the surrogate and true objectives:

{{< math >}}
$$
J(\pi) - J(\mu) = \sum_{t=0}^{T-1} \mathbb{E}_{(x, y_{\lt t}) \sim d_{\pi,t}} \mathbb{E}_{y_t \sim \pi(\cdot|x, y_{\lt t})} [A_\mu(x, y_{\le t})]
$$
{{< /math >}}

The key difference from the surrogate is that the true improvement uses the context distribution $d_{\pi,t}$ (under the new policy), while the surrogate uses $d_{\mu,t}$ (under the old policy).

Using the **Simulation Lemma** ([Kearns & Singh, 2002](https://dl.acm.org/doi/10.5555/645531.757765)), which bounds how context distributions diverge over the sequence:

{{< math >}}
$$
D_{TV}(d_{\pi,t} \| d_{\mu,t}) \leq t \cdot D_{TV}^{\max}(\pi, \mu)
$$
{{< /math >}}

we derive the **TRPO lower bound** ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)):

{{< math >}}
$$
\boxed{J(\pi) \geq L_\mu(\pi) - C \cdot T^2 \cdot D_{TV}^{\max}(\pi, \mu)}
$$
{{< /math >}}

where:
- $C$ is a constant depending on the maximum advantage $\max_{x, y_{\le t}}|A_\mu(x, y_{\le t})|$
- $T$ is the sequence length
- $D_{TV}^{\max} = \max_{x, y_{\lt t}} D_{TV}(\pi(\cdot|x, y_{\lt t}) \| \mu(\cdot|x, y_{\lt t}))$ is the maximum per-token TV distance

### The Trust Region Requirement

For surrogate optimization to guarantee improvement in the true objective, the policy must stay within a **trust region**:

{{< math >}}
$$
D_{TV}^{\max}(\pi, \mu) \leq \delta
$$
{{< /math >}}

The critical insight is that **the valid trust region size must shrink with sequence length**:

{{< math >}}
$$
\delta \propto \frac{1}{T^2}
$$
{{< /math >}}

This $T^2$ dependence arises because context distribution errors accumulate linearly over $T$ tokens, and the total error (summed over all tokens) scales quadratically.

### Soft vs. Hard Trust Regions

The TRPO framework suggests two approaches to enforce trust regions:

| Type | Mechanism | Implementation |
|------|-----------|----------------|
| **Soft (Clipping)** | Down-weight samples outside the region | $\min(\rho, C)$ — sample included with bounded weight |
| **Hard (Rejection)** | Exclude samples outside the region | $\mathbb{I}(\rho \leq C)$ — sample excluded entirely |

**Soft trust regions** use clipped importance sampling: $\min(\rho_t, 1+\epsilon)$. This is computationally efficient but retains potentially problematic samples.

**This part develops hard trust region methods**—Seq-MIS and Geo-Mask—that completely exclude samples outside the trusted region. We show when and why hard rejection outperforms soft clipping.

---

## 1. OOD High-Weight Samples: Why Rejection Outperforms Clipping

### 1.1 The Problem: Clipping Retains Problematic Samples

In Part 2, we derived the Seq-TIS estimator:

{{< math >}}
$$
\hat{g}_{\text{seq-tis}}(y) = \min(\rho(y), C) \cdot f(y)
$$
{{< /math >}}

where $f(y) = R(y) \cdot \nabla \log \pi(y)$ is the score function (reward-weighted gradient), and $\rho(y) = \pi(y)/\mu(y)$ is the sequence-level importance ratio.

The implicit assumption was that all samples $y \sim \mu$ are valid learning signals—samples with high weights $\rho(y)$ simply require variance control via clipping.

However, this assumption fails in practice. Consider a sample with $\rho(y) = 10,000$. This means:

{{< math >}}
$$
\frac{\pi(y)}{\mu(y)} = 10,000 \implies \mu(y) = \frac{\pi(y)}{10,000}
$$
{{< /math >}}

Such extreme ratios typically arise when $\mu(y)$ is near the numerical precision floor (e.g., $\mu(y) \approx 10^{-9}$). These are **Out-of-Distribution (OOD)** samples—samples that lie outside the trust region where importance sampling is valid. They occur due to:

1. **Numerical precision artifacts** in probability computation
2. **Distribution shift** between $\pi$ and $\mu$ beyond the valid IS regime

**The Clipping Problem:** When we apply Seq-TIS, we compute $\min(10000, C) \cdot f(y) = C \cdot f(y)$. The sample is still included in the gradient update with weight $C$. If $f(y)$ is malformed (due to the OOD nature of $y$), we introduce a systematic error into every gradient step.

**Why This Violates the Trust Region:** Recall from the TRPO lower bound that the surrogate objective $L_\mu(\pi)$ is only valid when $D_{TV}^{\max}(\pi, \mu) \leq \delta$. Samples with extreme importance ratios $\rho(y) \gg C$ indicate that $\pi$ and $\mu$ have diverged far beyond this trust region for that particular sample. Clipping the weight does not fix the underlying problem—the sample itself lies outside the region where the first-order approximation holds.

### 1.2 The Solution: Hard Trust Region via Rejection (Seq-MIS)

Instead of soft clipping, we enforce a **Hard Trust Region**: samples outside the trusted region are rejected entirely.

{{% callout note %}}
**Definition: Sequence-Level Masked IS (Seq-MIS)**

$$
\hat{g}\_{\text{seq-mis}}(y) = \mathbb{I}(\rho(y) \le C) \cdot \rho(y) \cdot f(y)
$$

where $\mathbb{I}(\cdot)$ is the indicator function.
{{% /callout %}}

**Mathematical Interpretation:** The trust region is defined as:

{{< math >}}
$$
\mathcal{T}_C = \{y : \rho(y) \le C\} = \left\{y : \frac{\pi(y)}{\mu(y)} \le C\right\}
$$
{{< /math >}}

Only samples within this region contribute to the gradient. Samples with $\rho(y) \gt C$ are treated as unreliable and excluded.

**Connection to TRPO:** This implements the trust region concept from TRPO theory ([Part 1](https://richardli.xyz/rl-collapse-1)), where the trust region constraint $D_{TV}(\pi \| \mu) \le \delta$ must be enforced to guarantee improvement. Seq-MIS enforces this constraint via **rejection** (hard trust region) rather than **penalization** (soft trust region), ensuring that gradient updates only use samples from the valid trust region.

### 1.3 Bias-Variance Analysis of Seq-MIS

**Bias:** The Seq-MIS estimator is biased. By importance sampling identity:

{{< math >}}
$$
\mathbb{E}_\mu[\hat{g}_{\text{seq-mis}}] = \mathbb{E}_\mu[\mathbb{I}(\rho \le C) \cdot \rho \cdot f] = \sum_y \mu(y) \cdot \mathbb{I}(\rho(y) \le C) \cdot \frac{\pi(y)}{\mu(y)} \cdot f(y)
$$
{{< /math >}}

{{< math >}}
$$
= \sum_y \pi(y) \cdot \mathbb{I}(\rho(y) \le C) \cdot f(y) = \mathbb{E}_\pi[f \cdot \mathbb{I}(\rho \le C)]
$$
{{< /math >}}

The true gradient is $g = \mathbb{E}_\pi[f]$. Therefore:

{{< math >}}
$$
\text{Bias} = \mathbb{E}_\mu[\hat{g}_{\text{seq-mis}}] - g = \mathbb{E}_\pi[f \cdot \mathbb{I}(\rho \le C)] - \mathbb{E}_\pi[f] = -\mathbb{E}_\pi[f \cdot \mathbb{I}(\rho > C)]
$$
{{< /math >}}

The bias is the negative of the contribution from rejected samples (weighted by $\pi$).

**Variance:** Since $\mathbb{I}(\rho \le C) \cdot \rho \le C$ when the indicator is 1, and 0 otherwise:

{{< math >}}
$$
\mathbf{Var}(\hat{g}_{\text{seq-mis}}) \le \mathbb{E}_\mu[\|\hat{g}_{\text{seq-mis}}\|^2] \le C^2 \cdot \mathbb{E}_\mu[\|f\|^2] = O(T^2 C^2)
$$
{{< /math >}}

### 1.4 When to Use Each Estimator

| **Estimator** | **Mechanism** | **Use Case** |
| --- | --- | --- |
| **Seq-TIS** | $\min(\rho, C) \cdot f$ | Moderate mismatch; maximize sample efficiency |
| **Seq-MIS** | $\mathbb{I}(\rho \le C) \cdot \rho \cdot f$ | Large mismatch; OOD samples likely; prioritize robustness |

The choice depends on the reliability of high-weight samples. When the behavior policy $\mu$ is well-calibrated and mismatch is controlled, Seq-TIS extracts more information. When OOD samples are prevalent, Seq-MIS provides a Hard Trust Region that prevents gradient corruption.

### 1.5 Practical Consequences for LLM-RL Training

The distinction between soft and hard trust regions has concrete effects on training stability:

**When using Seq-TIS (soft clipping):**

1. **Sample efficiency is higher** — all samples contribute to gradient updates
2. **Risk of gradient corruption** — OOD samples with $\rho \gg C$ still influence updates with weight $C$
3. **Systematic bias accumulation** — if many samples are OOD, clipping introduces consistent directional bias

**When using Seq-MIS (hard rejection):**

1. **Training is more robust** — OOD samples are completely excluded
2. **Lower sample efficiency** — rejected samples provide no learning signal
3. **Cleaner gradient estimates** — only samples within the trust region contribute

In practice, monitoring the **rejection rate** (fraction of samples with $\rho > C$) provides insight into policy-behavior divergence and can guide the choice between estimators.

---

## 2. Length-Dependent Rejection Bias: The Failure of Sequence-Level IS for Long Horizons

### 2.1 The Problem: Exponential Growth of the Importance Ratio

For autoregressive generation, the sequence-level importance ratio is a product of per-token ratios:

{{< math >}}
$$
\rho(y) = \prod_{t=0}^{T-1} \rho_t, \quad \text{where} \quad \rho_t = \frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})}
$$
{{< /math >}}

Even when the per-token mismatch is small, this product grows (or shrinks) exponentially with sequence length $T$ **almost everywhere** along the trajectory.

**Formal Analysis:** Let $\bar{\rho} = \mathbb{E}[\rho_t]$ denote the expected per-token ratio. If we assume (for simplicity) that the $\rho_t$ are independent with $\bar{\rho} \gt 1$, then:

{{< math >}}
$$
\mathbb{E}[\rho(y)] = \prod_{t=0}^{T-1} \mathbb{E}[\rho_t] = \bar{\rho}^T
$$
{{< /math >}}

Similarly, for the log-ratio:

{{< math >}}
$$
\log \rho(y) = \sum_{t=0}^{T-1} \log \rho_t
$$
{{< /math >}}

If each $\log \rho_t$ has mean $\delta \gt 0$ (i.e., $\pi$ assigns slightly higher probability than $\mu$ on average), then:

{{< math >}}
$$
\mathbb{E}[\log \rho(y)] = T \cdot \delta \implies \rho(y) \approx e^{T\delta}
$$
{{< /math >}}

**Numerical Example:** Consider $\bar{\rho} = 1.001$ (0.1% per-token drift):

| Sequence Length $T$ | $\rho(y) \approx 1.001^T$ | Within Trust Region ($C=5$)? |
| --- | --- | --- |
| 10 | 1.01 | Accepted |
| 1,000 | 2.72 | Accepted |
| 2,000 | 7.39 | Rejected |
| 5,000 | 148.4 | Rejected |

### 2.2 The Consequence: Systematic Length Bias

This exponential scaling creates a **length-dependent acceptance probability**. For any fixed threshold $C$, there exists a critical length $T^*$ beyond which almost all samples are rejected:

{{< math >}}
$$
T^* = \frac{\log C}{\log \bar{\rho}}
$$
{{< /math >}}

For $C = 5$ and $\bar{\rho} = 1.001$: $T^* = \frac{\log 5}{\log 1.001} \approx 1609$ tokens.

**Challenge 1: Structural Length Bias.** Chain-of-Thought (CoT) models and agents often generate sequences of 2,000-10,000+ tokens. With standard Seq-TIS or Seq-MIS:

1. Short responses (< 1000 tokens) are almost always accepted
2. Long reasoning chains (> 2000 tokens) are almost always rejected or heavily clipped
3. The model receives **systematically biased feedback** favoring short outputs

This is not a variance problem—it is a **structural bias** against long-horizon reasoning, independent of the quality of individual reasoning steps.

**Challenge 2: The Sequence-Level IS Dilemma.** We face a fundamental trade-off:

- **Use a large threshold** ($C \gg 1$): Accepts long sequences, but introduces high variance and allows OOD samples
- **Use a small threshold** ($C \approx 1$): Controls variance and rejects OOD samples, but systematically biases against long sequences

There is no value of $C$ that simultaneously:

1. Accepts high-quality long reasoning chains (requires $C$ large enough for $\rho(y) \le C$ when $T$ is large)
2. Rejects low-quality or OOD samples (requires $C$ small enough to filter outliers)
3. Provides a length-invariant acceptance criterion (requires $C$ to adapt with $T$, contradicting it being a fixed constant)

### 2.3 The Gap Between Theory and Practice

Here's a critical insight: **Standard PPO/GRPO implementations do not properly account for the horizon dependence of the trust region**.

The TRPO lower bound shows that for the surrogate objective to remain valid, the trust region must satisfy:

{{< math >}}
$$
D_{TV}^{\max}(\pi, \mu) \leq \delta, \quad \text{where} \quad \delta \propto \frac{1}{T^2}
$$
{{< /math >}}

However, PPO and GRPO use a **constant clipping factor** (e.g., $\epsilon = 0.2$ for token-level clipping, or $C = 5$ for sequence-level clipping) regardless of sequence length. This creates a fundamental mismatch:

| Aspect | TRPO Theory | PPO/GRPO Practice |
|--------|-------------|-------------------|
| Trust region size | Shrinks as $O(1/T^2)$ | Fixed constant $\epsilon$ or $C$ |
| Sequence-level threshold | Should be $C \approx O(1/T^2)$ | Typically $C \in [2, 10]$ regardless of $T$ |
| Long sequences ($T \gg 100$) | Requires very tight trust region | Uses same threshold as short sequences |

**Why This Matters for Long-Horizon Tasks:** When $T = 2000$, the theory suggests we should use $C \approx O(10^{-7})$, but in practice we use $C = 5$. This means:

1. **For short sequences:** The fixed threshold is too conservative (rejects valid samples)
2. **For long sequences:** The fixed threshold is too permissive (accepts out-of-trust-region samples)

The sequence-level estimators (Seq-TIS, Seq-MIS) suffer from this mismatch because $\rho(y) = \prod_t \rho_t$ is an **extensive quantity** that naturally grows with $T$, while the trust region constraint requires an **intensive** (length-normalized) measure.

This motivates the need for a length-invariant trust region mechanism—which is exactly what Geometric Sequence Masking provides.

---

## 3. Geometric Sequence Masking: A Per-Token Trust Region

### 3.1 From Extensive to Intensive Metrics

The fundamental problem with sequence-level IS is that $\rho(y) = \prod_t \rho_t$ is an **extensive quantity**—it scales with sequence length. We need an **intensive quantity** that measures the *average* per-token divergence, independent of length.

{{% callout note %}}
**Definition: Geometric Mean of the Importance Ratio**

$$
\rho\_{\text{geo}}(y) = \left( \prod\_{t=0}^{T-1} \rho\_t \right)^{1/T} = \rho(y)^{1/T}
$$
{{% /callout %}}

This is the geometric mean of the per-token ratios. It is length-invariant: if every $\rho_t = r$, then $\rho_{\text{geo}} = r$ regardless of $T$.

### 3.2 Mathematical Foundation: Connection to Per-Token KL Divergence

The geometric mean has a natural interpretation in terms of KL divergence. Taking the logarithm:

{{< math >}}
$$
\log \rho_{\text{geo}}(y) = \frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})}
$$
{{< /math >}}

This is the **sample average of the per-token log-ratios** along trajectory $y$.

**Connection to KL Divergence:** Recall that at each step $t$, given context $(x, y_{\lt t})$:

- **Forward KL:** {{< math >}}$D_{KL}(\pi \| \mu) = \mathbb{E}_{y_t \sim \pi}\left[\log \frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})}\right]${{< /math >}}
- **Reverse KL:** {{< math >}}$D_{KL}(\mu \| \pi) = \mathbb{E}_{y_t \sim \mu}\left[\log \frac{\mu(y_t|x, y_{\lt t})}{\pi(y_t|x, y_{\lt t})}\right] = -\mathbb{E}_{y_t \sim \mu}\left[\log \frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})}\right]${{< /math >}}

Since samples are drawn from $\mu$ (the behavior policy), each term $\log \rho_t = \log \frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})}$ is a single-sample estimate of the **negative reverse KL**:

{{< math >}}
$$
\mathbb{E}_{y_t \sim \mu}\left[\log \frac{\pi(y_t|x, y_{\lt t})}{\mu(y_t|x, y_{\lt t})}\right] = -D_{KL}(\mu(\cdot|x, y_{\lt t}) \| \pi(\cdot|x, y_{\lt t}))
$$
{{< /math >}}

Therefore, $\log \rho_{\text{geo}}(y)$ can be interpreted as a trajectory-averaged sample of the negative reverse KL:

{{< math >}}
$$
\mathbb{E}_{y \sim \mu}\left[\log \rho_{\text{geo}}(y)\right] = -\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}_{(x, y_{\lt t}) \sim d_\mu}\left[D_{KL}(\mu(\cdot|x, y_{\lt t}) \| \pi(\cdot|x, y_{\lt t}))\right]
$$
{{< /math >}}

**Connection to k1 Estimation of KL Divergence:** Schulman's framework ([Schulman, 2016](http://joschu.net/blog/kl-approx.html)) defines three estimators for $D_{KL}(\mu \| \pi)$ based on ratio $r = \mu/\pi$:

- **k1:** $-\log r = \log(\pi/\mu) = \log \rho$ (unbiased, high variance)
- **k2:** $(\log r)^2 / 2 = (\log \rho)^2 / 2$ (low variance, biased)
- **k3:** $(r - 1) - \log r = (1/\rho - 1) + \log \rho$ (unbiased, low variance)

In our notation with $\rho = \pi/\mu$, we have $r = 1/\rho$, so the k1 estimator equals $\log \rho$.

Given a single trajectory sample $y \sim \mu$:

{{< math >}}
$$
\log \rho_{\text{geo}}(y) = \frac{1}{T} \sum_{t=0}^{T-1} \log \rho_t = \frac{1}{T} \sum_{t=0}^{T-1} \text{(k1 estimator at step } t\text{)}
$$
{{< /math >}}

Therefore, **$\log \rho_{\text{geo}}$ is the average k1 estimate** along the trajectory. Since $\mathbb{E}_{\mu}[\log \rho] = -D_{KL}(\mu \| \pi)$, we have:

{{< math >}}
$$
\mathbb{E}_{y \sim \mu}\left[\log \rho_{\text{geo}}(y)\right] = -\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}_{(x, y_{\lt t}) \sim d_\mu}\left[D_{KL}(\mu(\cdot|x, y_{\lt t}) \| \pi(\cdot|x, y_{\lt t}))\right] = -\bar{D}_{KL}
$$
{{< /math >}}

where $\bar{D}_{KL}$ is the average per-token reverse KL along the trajectory distribution.

**Why k1-based filtering is natural:** Filtering by $\log \rho_{\text{geo}} > 0$ means $\pi$ assigns higher probability than $\mu$ on average, corresponding to negative reverse KL (or positive forward KL). Alternative trust region metrics could use k2 or k3, but k1-based filtering via $\log \rho_{\text{geo}}$ directly corresponds to the log of the geometric mean and provides an unbiased estimator of the KL divergence.

This connects our geometric ratio directly to the trust region constraint: filtering by $|\log \rho_{\text{geo}}| \leq \epsilon$ enforces that the **average per-token KL** (estimated via k1) remains bounded.

**Interpretation:** $\log \rho_{\text{geo}}(y)$ measures the average per-step log-likelihood ratio along the specific trajectory $y$. Unlike the sequence-level ratio, this quantity:

1. **Does not grow with $T$** (it's an average, not a sum)
2. **Can be positive or negative** ($\log \rho_{\text{geo}} \gt 0$ when $\pi$ assigns higher probability, $\lt 0$ when $\mu$ assigns higher probability)
3. **Detects both directions of drift** ($\rho_{\text{geo}} \ll 1$: policy forgetting; $\rho_{\text{geo}} \gg 1$: policy collapse)
4. **Is the average k1 estimate** (where k1 = $\log \rho$ in our notation with $\rho = \pi/\mu$)

{{% callout note %}}
**Connection to TRPO Theory (Part 1):** In [Part 1](https://richardli.xyz/rl-collapse-1), we showed that TRPO requires the trust region size to shrink with horizon: $\delta \propto 1/T^2$. This ensures the surrogate objective remains a valid approximation.

However, enforcing a sequence-level constraint that shrinks as $O(1/T^2)$ is impractical—it would require different thresholds for every sequence length and make long sequences nearly impossible to accept.

**Geo-Mask achieves length-invariance via per-token KL control.** By constraining $|\log \rho\_{\text{geo}}| \le \epsilon$, we enforce:

$$
\left| \frac{1}{T} \sum\_{t=0}^{T-1} \log \frac{\pi(y\_t|x, y\_{\lt t})}{\mu(y\_t|x, y\_{\lt t})} \right| \le \epsilon
$$

This bounds the **average per-token KL divergence** (via the k1 estimator) along the trajectory. The key insight is that this constraint is **independent of sequence length $T$**:

- **Sequence-level trust region:** Requires $\rho(y) = \prod\_t \rho\_t \le C$, where $C$ must shrink as $O(1/T^2)$ to satisfy TRPO
- **Per-token trust region (Geo-Mask):** Requires $\rho\_{\text{geo}}(y) = (\prod\_t \rho\_t)^{1/T} \le C$, where $C$ can be fixed (e.g., $C=2$) for all $T$

Since $\log \rho\_{\text{geo}}(y) = \frac{1}{T}\sum\_t \log \rho\_t$ is the average k1 estimate (where k1 = $\log \rho$), the constraint $|\log \rho\_{\text{geo}}| \le \epsilon$ approximately enforces that the average per-token KL divergence remains bounded. This is precisely the intensive (per-token) version of the trust region constraint, automatically adapting to sequence length by measuring the *average* rather than the *total* divergence.

Thus, Geo-Mask is a **practical implementation of the TRPO hard trust region** in the LLM context, with the crucial property that the acceptance criterion is **length-invariant**.
{{% /callout %}}

### 3.3 The Two-Sided Hard Trust Region (Geo-Mask)

With the geometric ratio, we can define a **Per-Token Trust Region** that is independent of sequence length:

{{% callout note %}}
**Definition: Geometric Sequence Masking (Geo-Mask)**

$$
\hat{g}\_{\text{geo-mask}}(y) = \mathbb{I}\left( C\_{\text{low}} \le \rho\_{\text{geo}}(y) \le C\_{\text{high}} \right) \cdot f(y)
$$

Equivalently, in log-space:

$$
\hat{g}\_{\text{geo-mask}}(y) = \mathbb{I}\left( \log C\_{\text{low}} \le \frac{1}{T}\sum\_{t=0}^{T-1} \log \rho\_t \le \log C\_{\text{high}} \right) \cdot f(y)
$$
{{% /callout %}}

**Note on IS Weighting:** Unlike Seq-MIS which uses $\rho \cdot f$, Geo-Mask uses just $f(y)$ without importance weighting. This is because Geo-Mask is designed as a **pure filtering operation**—it determines which samples are within the trust region. Within the trust region where $\rho_{\text{geo}} \approx 1$, we have $\pi \approx \mu$ on average, so IS correction is less critical. For full IS correction, combine with Token-TIS (see Section 3.5).

**Why Two-Sided?** The trust region enforces constraints in both directions:

| Condition | Meaning | Detection |
| --- | --- | --- |
| $\rho_{\text{geo}} \lt C_{\text{low}}$ | $\pi$ assigns much lower probability than $\mu$ on average | Policy has drifted away from high-likelihood regions of $\mu$ |
| $\rho_{\text{geo}} \gt C_{\text{high}}$ | $\pi$ assigns much higher probability than $\mu$ on average | Policy may be collapsing/overfitting to specific patterns |

**Typical Values:** $C_{\text{low}} = 0.5$, $C_{\text{high}} = 2.0$ (or equivalently, $|\log \rho_{\text{geo}}| \le \log 2 \approx 0.69$).

### 3.4 Trust Region Interpretation of Geo-Mask

**Connection to TRPO:** The masking operation in Geo-Mask directly enforces the trust region constraint from TRPO theory. Recall from the TRPO lower bound:

{{< math >}}
$$
J(\pi) \geq L_\mu(\pi) - C \cdot T^2 \cdot D_{TV}^{\max}(\pi, \mu)
$$
{{< /math >}}

The surrogate objective $L_\mu(\pi)$ is only a valid approximation to $J(\pi)$ within the trust region. Samples outside this region violate the approximation assumptions—gradient updates from such samples do not guarantee improvement in the true objective. Geo-Mask excludes these samples entirely, ensuring that **all gradient contributions come from the valid trust region**.

**Variance Control:** Since there is no importance weight multiplication, the variance is naturally bounded:

{{< math >}}
$$
\mathbf{Var}(\hat{g}_{\text{geo-mask}}) \le \mathbb{E}_\mu[\|f\|^2] = O(T^2)
$$
{{< /math >}}

**Length Invariance:** The acceptance criterion $C_{\text{low}} \le \rho_{\text{geo}} \le C_{\text{high}}$ is length-independent. A 100-token sequence and a 10,000-token sequence are judged by the same per-token divergence threshold.

### 3.5 Combining Geo-Mask with Token-TIS

The TRPO surrogate objective is naturally a sum over tokens, so its gradient is essentially a **Token-level IS** gradient. This motivates combining Geo-Mask with Token-TIS:

1. **Geo-Mask filter:** Mask samples with extreme per-token divergence (length-invariant safety)
2. **Token-TIS weighting:** Apply per-token clipped importance sampling for accepted samples

{{% callout note %}}
**Definition: Geo-Mask-Token-TIS**

$$
\hat{g}\_{\text{geo-mask-token-tis}}(y) = \mathbb{I}\left( C\_{\text{low}} \le \rho\_{\text{geo}}(y) \le C\_{\text{high}} \right) \cdot \sum\_{t=0}^{T-1} \min(\rho\_t, C) \cdot A\_t \nabla \log \pi(y\_t|x, y\_{\lt t})
$$
{{% /callout %}}

This estimator:
1. First checks if the sample is within the per-token trust region (Geo-Mask)
2. Then applies per-token clipped importance weighting for accepted samples (Token-TIS)

**When to use which component:**

| Component | Purpose |
| --- | --- |
| Geo-Mask ($\rho_{\text{geo}}$ filter) | Ensures length-invariant acceptance; detects per-token drift |
| Token-TIS ($\min(\rho_t, C)$ weight) | Aligns with TRPO surrogate gradient structure |

---

## 4. Summary: Hierarchy of Estimators and Selection Guidelines

We have developed a hierarchy of estimators, each addressing specific failure modes:

| **Estimator** | **Formula** | **Trust Region Type** | **Primary Use Case** |
| --- | --- | --- | --- |
| **Token-IS (PPO)** | $\sum_t \min(\rho_t, C) A_t \nabla \log \pi_t$ | Per-token, soft | Stable but has $O(T^2\Delta_{\max})$ bias |
| **Seq-TIS** | $\min(\rho, C) \cdot f$ | Sequence-level, soft | Optimal bias-variance when all samples are valid |
| **Seq-MIS** | $\mathbb{I}(\rho \le C) \cdot \rho \cdot f$ | Sequence-level, hard | OOD sample filtering; large mismatch scenarios |
| **Geo-Mask** | $\mathbb{I}(C_{\text{low}} \le \rho_{\text{geo}} \le C_{\text{high}}) \cdot f$ | Per-token, hard | Long-horizon tasks; length-invariant masking |
| **Geo-Mask-Token-TIS** | Geo-Mask filter × Token-TIS weight | Hybrid | Long-horizon + TRPO-aligned gradient |

For long-horizon reasoning tasks, **Geometric Sequence Masking (Geo-Mask)** provides a principled, length-invariant Hard Trust Region that prevents the systematic length bias inherent in standard importance sampling estimators.

---

## References

**Trust Region Methods:**

- Kakade, S. & Langford, J. (2002). "Approximately Optimal Approximate Reinforcement Learning." *ICML*. https://dl.acm.org/doi/10.5555/645531.656005
- Kearns, M. & Singh, S. (2002). "Near-Optimal Reinforcement Learning in Polynomial Time." *Machine Learning*.
- Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). "Trust Region Policy Optimization." *ICML*. https://arxiv.org/abs/1502.05477
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347. https://arxiv.org/abs/1707.06347

**KL Divergence Estimation:**

- Schulman, J. (2016). "Approximating KL Divergence." Blog post. http://joschu.net/blog/kl-approx.html

**LLM Reinforcement Learning:**

- Shao, Z., et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300. https://arxiv.org/abs/2402.03300
