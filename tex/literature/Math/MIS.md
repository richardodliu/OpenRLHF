# When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch

Authors:    [Jiacai Liu](https://www.jiacailiu.cn/)    [Yingru Li](http://richardli.xyz)    Yuqian Fu    Jiawei Wang    Qian Liu    Yu Shen

## **TL;DR:**

The relentless push for faster inference has created a dangerous "training-inference mismatch" that can silently kill reinforcement learning with LLMs. Our investigation reveals a vicious cycle that is particularly acute in modern reasoning and agentic RL:

- **OOD Contexts Drive Low-Probability Sampling:** Agentic workflows expose models to external inputs and dynamic environments, forcing frequent generation of **low-probability tokens** that are essential for novel reasoning, tool calls, and adaptive responses. [3.4 OOD Tool Responses Amplifies the Mismatch](https://www.notion.so/3-4-OOD-Tool-Responses-Amplifies-the-Mismatch-271211a558b78050b3b1dbbad2a1eb40?pvs=21)
- **Low-Probability Tokens Amplify Training Collapse:** These tokens become the weakest link—the training-inference mismatch is most severe for them, causing catastrophically large gradients that lead to silent degradation and sudden training failure. [**3.3 The Smoking Gun: The Low-Probability Token Pitfall**](https://www.notion.so/3-3-The-Smoking-Gun-The-Low-Probability-Token-Pitfall-271211a558b780439a6bf4222291f060?pvs=21)
- **Hardware Variability Complicates the Problem:** Different GPU architectures exacerbate the mismatch unpredictably, meaning the same agentic training setup can succeed on one machine and catastrophically fail on another. [**3.5 The Environmental Factor: The Critical Role of Hardware**](https://www.notion.so/3-5-The-Environmental-Factor-The-Critical-Role-of-Hardware-271211a558b780c9ad90f8aa1353bfe1?pvs=21)
- **Sequence-Level Correction is the Principled Solution:** **Sequence-level Correction** emerges as the theoretically grounded fix. It corrects the biased gradients by accounting for the full state trajectory, restoring training stability **across different hardware and complex tasks.** [4.2.1 A Principled Solution: Distribution Correction](https://www.notion.so/4-2-1-A-Principled-Solution-Distribution-Correction-27b211a558b78099ba48fa8849ab54c8?pvs=21) 

## 📖**Deeper Analysis:**

For a rigorous theoretical breakdown of this problem, we've published a 3-part blog series to give more insights:

- [**[Part 1: Why Off-Policy Breaks RL — An SGA Analysis Framework\]**](https://richardli.xyz/rl-collapse-1) **Known (TRPO theory):** The surrogate objective $L_\mu(\pi)$ is a first-order Taylor approximation of RL objective $J(\pi)$; the TRPO lower bound $J(\pi) \ge L_\mu(\pi) - C \cdot T^2 \cdot D_{TV}^{\max}$ shows the approximation error grows with $T^2$, requiring the trust region to shrink as $\delta \propto 1/T^2$. **Our insight:** Token-level IS (PPO/GRPO) computes $\nabla L_\mu$, not $\nabla J$—it corrects the token distribution but not the state distribution mismatch ($d_\mu \ne d_\pi$), resulting in $O(T^2 D_{TV}^{\max})$ bias. **We formalize two failure modes via the SGA Lemma**: **Bias** (measured by $D_{TV}$) and **Variance** (measured by $\chi^2$-divergence)—these metrics are not interchangeable.
- [**[Part 2: Applying the SGA Framework — Token v.s. Sequence-level Correction\]**](https://richardli.xyz/rl-collapse-2) **Our analysis:** Token-level IS (PPO/GRPO) has $O(T^2 D_{TV}^{\max})$ bias from the surrogate's first-order approximation error; Sequence-level IS is unbiased but has exponential variance $O((1+\bar{\chi}^2_{\max})^T)$. **Our solution:** Seq-TIS achieves controllable bias-variance trade-off via clipping $\rho(y) \to \min(\rho(y), C)$. **Key conclusion:** This bias is a sequence-level problem requiring sequence-level solutions **when non-negligible**.
- [**[Part 3: Trust Region Optimization via Sequence Masking\]**](https://richardli.xyz/rl-collapse-3) **Known (TRPO theory):** Trust region constraints ensure the surrogate remains a valid approximation. **Our solutions:** (1) **Seq-MIS** enforces a Hard Trust Region via rejection $\mathbb{I}(\rho \le C) \cdot \rho \cdot f$, excluding OOD samples entirely. (2) **Geo-RS** uses the geometric mean $\rho_{\text{geo}}=\rho^{1/T}$ to achieve a **length-invariant Per-Token Trust Region**—a practical implementation of TRPO's hard trust region for LLMs that avoids systematic rejection of long sequences. 

# Citation

```jsx
@online{liu-li-2025-rl-collapse,
  title = {When Speed Kills Stability: Demystifying {RL} Collapse from the Training-Inference Mismatch},
  author = {Liu, Jiacai and Li, Yingru and Fu, Yuqian and Wang, Jiawei and Liu, Qian and Jiang, Zhuo},
  year = {2025},
  month = sep,
  url = {<https://richardli.xyz/rl-collapse>}
}
```

# **1. The Mystery of the Sudden Collapse**

In the rapidly advancing field of reinforcement learning for large language models (LLM-RL), a frustrating pattern of sudden training collapse is emerging. Whether in complex **reasoning RL** or multi-turn **agentic RL**, many have observed training runs that, after a period of stable learning, catastrophically fail.

We recently encountered this firsthand while conducting agentic RL experiments for multi-turn tool-integrated-reasoning (TIR) on Qwen3 models. This occurred across both on-policy and off-policy variants of the GRPO algorithm on our L20 GPU cluster. Figure 1 shows the reward and gradient norm dynamics of our four crashed experiments on Qwen3-14B-Base. As training progresses, the gradient norms suddenly explode, leading to model collapse. Our initial investigation focused on common culprits:

- We examined the code and confirmed that our agent loop follows a token-in-token-out process.

- We tuned the hyperparameters `beta1` and `beta2` in the Adam optimizer.

- We also applied batch normalization to the advantages to balance the updates.

  ...

However, none of these standard fixes worked. Since even the simpler on-policy experiments failed, we suspected the issue was not with the RL algorithm but with a more fundamental part of the training stack. This led us to investigate a critical and increasingly prevalent challenge in modern LLM-RL: the unavoidable gap between highly-optimized inference engines and faithful training frameworks.

------

# **2. A Fundamental Conflict: The Growing Gap Between Inference and Training**

Rollout speed is a core bottleneck in LLM-RL. To achieve the massive throughput required, modern inference engines (e.g., vLLM, SGLang, TensorRT-LLM) employ aggressive optimization strategies like speculative decoding, low-precision computation (INT8/FP8), and specialized, batch-variant CUDA kernels. While maintaining sampling fidelity, **the primary objective of modern inference engines is to maximize throughput**, ****often measured in tokens per second. **Conversely, training frameworks (e.g., FSDP, DeepSpeed, Megatron-LM)** **must strike a different balance, prioritizing numerical stability and precision for gradient computation,** often using higher-precision formats like FP32 for master weights and optimizer states. **This divergence in optimization priorities and constraints creates an inevitable training-inference mismatch.** The relentless push for faster rollouts is making this gap wider, not smaller. While one might propose enforcing identical calculations (e.g., using ["batch invariant kernels"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)), these solutions come with a severe performance penalty, defeating the purpose of using a high-speed inference engine in the first place. **This speed-vs-consistency trade-off is at the heart of the problem, making it a persistent challenge rather than a simple engineering fix.**

In our stack, this mismatch manifested between our vLLM inference sampler and our FSDP trainer. The actual parameter update was:

$$ \mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \textcolor{red}{\pi _{\theta}^{\mathrm{vllm}}}\left( \cdot |x \right)}\left[ R\left( x,y \right) \nabla _{\theta}\log \textcolor{blue}{\pi _{\theta}^{\mathrm{fsdp}}}\left( y|x \right) \right], $$

whereas the theoretical parameter update should be

$$ \mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \textcolor{blue}{\pi _{\theta}^{\mathrm{fsdp}}}\left( \cdot |x \right)}\left[ R\left( x,y \right) \nabla _{\theta}\log \textcolor{blue}{\pi _{\theta}^{\mathrm{fsdp}}}\left( y|x \right) \right]. $$

Here $x$ is the prompt sampled from the distribution $\mathcal{D}$, $y$ is the response, $R$ is the reward function, $\theta$ is the parameter of LLM,  $\textcolor{red}{\pi^\text{vllm}*\theta}$ and $\textcolor{blue}{\pi^\text{fsdp}*\theta}$ are the policy implemented in vLLM engine and FSDP engine respectively. To investigate this, we first needed a way to measure it.

------

# **3. Anatomy of the Training Collapse**

## **3.0** Experiments Setup

Unless otherwise specified, the experiments presented in **Section 3** and **Section 4** are conducted on the [**VeRL](https://github.com/search?q=verl&type=repositories)** framework under the TIR setting, using the vLLM v1 sampler (AsyncvLLMServer), the Qwen3-14B-Base model, and the GRPO algorithm, all running on an L20 GPU cluster.

------

## **3.1 Measuring the Mismatch: The `vllm-kl` Metric**

A very straightforward metric for measuring the training-inference mismatch is **vllm-kl**:

$$ \small{\mathbb{E}*{s\sim d*{\textcolor{red}{\pi^\text{vllm}*\theta}}}\left[\text{KL}\left(\textcolor{red}{\pi^\text{vllm}*\theta}\left(\cdot|s\right),\textcolor{blue}{\pi^\text{fsdp}*\theta}\left(\cdot|s\right)\right)\right] = \mathbb{E}*{s\sim d_{\textcolor{red}{\pi^\text{vllm}*\theta}},a\sim {\textcolor{red}{\pi^\text{vllm}*\theta}\left(\cdot|s\right)}} \left[\log\left(\frac{\textcolor{red}{\pi^\text{vllm}*\theta}(a|s)}{\textcolor{blue}{\pi^\text{fsdp}*\theta}(a|s)}\right)\right],} $$

where $d_\pi$ is the state-occupancy of a policy $\pi$, $s$ is the context prefix (state) and $a$ is the token (action).  Note that our experiments involve tool calls, meaning the response  $y$ may include tool responses. Therefore, our  `vllm-kl` metric only considers tokens generated by the model itself. The following code provides an implementation in [**VeRL**](https://github.com/search?q=verl&type=repositories) for calculating the vllm-kl metric using the [K3 estimator](http://joschu.net/blog/kl-approx.html), assuming that the token probabilities of inference engine are already accessible:

- codeblock of K3 estimator for `vllm-kl`

  ```python
  rollout_log_probs = batch.batch["rollout_log_probs"] # pi_vllm
  actor_old_log_probs = batch.batch["old_log_probs"] # pi_fsdp
  response_mask = batch.batch["response_mask"]
  log_ratio = actor_old_log_probs - rollout_log_probs 
  vllm_k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
  vllm_k3_kl = masked_mean(vllm_k3_kl_matrix,response_mask)
  ```

------

## 3.2 **The Warning Signs: Correlated Instability**

Our first clue was that high `vllm-kl` values were not isolated events. They correlated strongly with other signs of instability.

### 3.2.1 Fluctuations **in FSDP Entropy and Rewards**

In many of our experiments, we have observed that abnormal spikes in `vllm-kl` typically trigger simultaneous anomalous fluctuations in the entropy of FSDP policy $\textcolor{blue}{\pi^\text{fsdp}*\theta}$ and rewards. The experimental results presented in Figure 2 serve as an intuitive example.  As it can be seen in the figure, **the locations where entropy spikes occur almost perfectly correspond with the locations of `vllm-kl` spikes**. While no equally pronounced correlation is observed in the rewards, a massive `vllm-kl` spike around step 250 can be seen, which triggered the generation of a low-quality batch and resulted in a noticeable dip there as well.  **This means that when the mismatch is large, both the vLLM policy** $\textcolor{red}{\pi^\text{vllm}*\theta}$ **and the FSDP policy** $\textcolor{blue}{\pi^\text{fsdp}_\theta}$  **enter an unstable region.**

------

### **3.2.2 Rising FSDP PPL and Gradient Norm Leading to Policy Collapse**

More critically, we observed a spike in `vllm-kl` simultaneously triggering an explosion in both the fsdp-[ppl](https://www.wikiwand.com/en/articles/Perplexity) metric and the gradient norm. In our experiments, the fsdp-ppl metric of a response $y$ is calculated as follows:

$$ \text{fsdp-ppl}:  \exp\left(\frac{-1}{\left| \mathcal{T} *{\mathcal{M}}\left( y \right) \right|}\sum*{t\in \mathcal{T} _{\mathcal{M}}\left( y \right)}{\log \textcolor{blue}{\pi *{\theta}^{\text{fsdp}}}\left( y_t|y*{<t} \right)}\right) $$

where $\mathcal{T} _{\mathcal{M}}\left( y \right)$ is the index set of the tokens generated by the model itself in the response $y$.  The final fsdp-ppl metric is the average of the fsdp-ppl metrics for all responses in the batch. Figure 3  present the experimental results for an on-policy version and an off-policy version of GRPO. In both experiments, the spikes in `vllm-kl` almost precisely triggered corresponding explosions in the fsdp-ppl and gradient norm. Moreover, it can be observed that before the training reward collapses, there is a significant  rising in the `vllm-kl` metric.

In our experiments, the sequences generated by the model itself consist of at least several hundred tokens. Therefore, in the later stages of training, it is more reasonable for the ppl metric to remain around 1. However, in batches with significant training-inference mismatch—that is, where `vllm-kl` is significant high—an explosion in the fsdp-ppl metric was observed**. This indicates that the FSDP engine assigning catastrophically low probabilities to tokens that the inference policy sampled, leading to gradient explosion.** This observation helped us further pinpoint where mismatches are more likely to occur. In fact, as we will see later, when these extremely low-fsdp-probability tokens were sampled, their probabilities in vLLM engine were not nearly as low.

------

## **3.3 The Smoking Gun: The Low-Probability Token Pitfall**

**The mismatch was not uniform.**

By analyzing batches with varying levels of `vllm-kl`, we found a stark pattern: **the divergence is most severe for tokens that have a low probability according to the vLLM inference engine.** As a token's inference probability approaches zero, the training probability can become orders of magnitude smaller, leading to an infinite PPL and gradient. To ensure the conclusions are as generalizable as possible, we selected batches sampled from various experiments prior to training collapse, taken at different training steps. All of these batches exhibit relatively high `vllm-kl` values, enabling us to study mismatch patterns under pronounced conditions. Rollout batches were collected within the following three `vllm-kl` ranges, with five batches (around 5M tokens) in each group:

- **Group 1 (low):**  Each rollout batch has a `vllm-kl` no greater than 1e-3, and the batches are sampled using H20 GPUs.
- **Group 2 (medium):**  Each rollout  batch has a `vllm-kl` belongs to [1e-3, 2e-2], and the batches are sampled using L20 GPUs.
- **Group 3 (high):**  Each rollout  batch has a `vllm-kl` belongs to [2e-2, 1e-1], and the batches are sampled using L20 GPUs.

Figure 4 (a)(b)(c) below shows the relationship between the output probabilities of the vLLM engine, i.e. $\textcolor{red}{\pi^\text{vllm}*\theta}(a|s)$ and the mismatch—measured by $\log\left(\textcolor{blue}{\pi^\text{fsdp}*\theta}(a|s)\right)-\log\left(\textcolor{red}{\pi^\text{vllm}_\theta}(a|s)\right)$ —under different magnitudes of `vllm-kl`.

From the three figures above, we can clearly observe that:

1. The degree of mismatch tends to be more pronounced when the vLLM probability $\textcolor{red}{\pi^\text{vllm}*\theta}$ approaches zero, and the extreme values of $\log\left(\textcolor{blue}{\pi^\text{fsdp}*\theta}\right)-\log\left(\textcolor{red}{\pi^\text{vllm}_\theta}\right)$ are more likely to occur under these conditions.
2. The batches collected on L20 GPUs, namely those in groups 2 and 3, exhibit a training-inference mismatch that is primarily manifested as the FSDP probability $\textcolor{blue}{\pi^\text{fsdp}*\theta}$ being significantly smaller than the vLLM probability $\textcolor{red}{\pi^\text{vllm}*\theta}$.

------

## 3.4 OOD Tool Responses Amplifies the Mismatch

### 3.4.1 The Mismatch Is More Severe in Non-First-Round Outputs

The finding in Section 3 explained why the problem was so acute in our multi-turn TIR experiments, **particularly in the non-first-round model outputs**. The process works as follows:

1. The **agent** receives a tool response, which is often structured text (e.g., context enveloped by `<python_output>` and `</python_output>` tags) that is OOD compared to its pre-training and SFT data.
2. Faced with this unfamiliar OOD context, the **agent's policy** becomes more uncertain, making it more likely to sample low-probability tokens in its subsequent turns (which is also observed in [**SimpleTIR**](https://github.com/ltzheng/SimpleTIR)).
3. As we just established, these low-probability tokens are the primary site where severe mismatch occurs, creating the conditions for a `fsdp-ppl` and gradient explosion.

Next, we plot the mismatch in the batches of the three groups (around 5k trajectories per group), highlighting the differences between the first-round model outputs and non-first-round model outputs. We consider the following two methods to visualize the mismatch:

1. **Log-ppl scatter plot**: The x-axis represents the logarithm of the ppl metric calculated by the vLLM policy $\textcolor{red}{\pi^\text{vllm}*\theta}$, denoted as **`vllm-log-ppl`**, and the y-axis represents the logarithm of the ppl  metric calculated by FSDP policy  $\textcolor{blue}{\pi^\text{fsdp}*\theta}$, denoted as **`fsdp-log-ppl`**.
2. **Probability scatter plot**: The x-axis represents the token probability of vLLM policy $\textcolor{red}{\pi^\text{vllm}*\theta}$ and the y-axis represents the token probability of FSDP policy  $\textcolor{blue}{\pi^\text{fsdp}*\theta}$.

We present the visualization results of the mismatch for three groups below.

From the visualization results, we can observe that:

1. The `vllm-log-ppl` of non-first-round outputs is generally larger compared to the first-round outputs, which means more low-probability tokens are sampled when faced with unfamiliar OOD context.
2. The mismatch primarily occurs in non-first-round model outputs, manifested by a larger mean absolute difference in both log-ppl and token probability between FSDP policy and vLLM policy compared to the first-round output, along with a lower Pearson correlation coefficient.
3. As the `vllm-kl` value increases, the training-inference mismatch primarily worsens in non-first-turn outputs.
4. The mismatch consistently show that the `fsdp-log-ppl` is greater than the `vllm-log-ppl` indicating that the FSDP engine produces more extreme low-probability tokens.

------

### **3.4.2 More Tool Calls, More Training Instability**

The following experiments also demonstrate that the OOD tool responses exacerbates the training-inference mismatch and training instability. We conducted off-policy GRPO experiments on H20 GPUs using Qwen3-14B-Base as the base model with clip higher=0.28 and 4 mini-batches. We set different maximum number of tool calls in a single trajectory (the hyperparameter `max_tool_turn`) at 20 and 100. The experimental results are shown in Figure 5 and 6. It can be observed that as the number of tool calls increases, the time to training collapse occurs earlier. Upon collapse, gradient explosion and an explosion in `vllm-kl` are observed in all cases.

------

## **3.5 The Environmental Factor: The Critical Role of Hardware**

Finally, we discovered that the physical hardware is a critical variable. **The exact same code and model produced drastically different levels of mismatch across different GPU hardware**. To evaluate the extent of mismatch across different hardware, we ran on-policy algorithms with identical code environments and hyperparameters, only switching between different GPUs for inference and training. Figure 7 illustrates the training dynamics on the L20, H20, and A100, respectively.

From the figure, it can be observed that in our experiments, the magnitude of the `vllm-kl` essentially follows: **H20 < L20 < A100**. Specifically, the `vllm-kl` for the H20 is generally on the order of 5e-4 to 1e-3, for the L20 it is around 1e-3 to 1e-2, and for the A100, it is predominantly between 1e-2 and 1. Due to the severe training-inference mismatch on the A100, normal training becomes unfeasible, resulting in highly unstable rewards curves.

We found that disabling cascade attention in the vLLM engine was particularly helpful in reducing the mismatch in our experiments **run on** A100 GPUs. **We present these results** in Section [**4.2.4 Diable Cascade Attention in vLLM**](https://www.notion.so/4-2-4-Diable-Cascade-Attention-in-vLLM-27a211a558b780fab2d9d85708b21084?pvs=21).

The most compelling proof came when we took a failed L20 experiment and resumed it from a checkpoint on H20 GPUs (see Figure 8). The training immediately stabilized and recovered, proving the hardware's first-order impact on the problem.

------

## **3.6 The Mismatch is Not Static: A Vicious Cycle Driven by Optimization**

One might assume the training-inference mismatch is a static property of the hardware and software stack. However, Our following "batch-filter" experiment proved that **the mismatch is coupled with the training dynamics and model’s state**.

We set up the following policy update strategy in our “**batch-filter”** experiment : for each training step, if the collected batch yields a `vllm-kl` metric greater than the threshold, we skip updating the model parameters on this batch, as such updates are prone to cause training collapse. Instead, we proceed directly to the next step, continuing data collection until a batch with a `vllm-kl` value below the threshold is obtained, at which point the model is updated.

The logic behind this experiment is that if the degree of mismatch were entirely independent of the model's output distribution and the training dynamics, the magnitude of `vllm-kl` should exhibit the same distributions across different training steps. However, the experimental results presented in Figure 9 show that **once the model entered a certain state, it began consistently generating high-mismatch batches**, effectively halting training. This, along with the spiraling `vllm-kl` and `fsdp-ppl` seen in other runs (Figure 10), points to a dangerous feedback loop.

**We hypothesize this is due to following two-stage failure cascade:**

1. **Stage 1: Increased Numerical Sensitivity.** The RL optimizer pushes the model's weights into numerical ranges where the `bfloat16` data type has lower relative precision (e.g., very small or very large values).
2. **Stage 2: Kernel-Driven Error Amplification.** These initial, tiny `bfloat16` quantization errors are then fed into the different kernel implementations of vLLM and FSDP. The differing calculation orders act as non-linear amplifiers, causing the small initial deviations to snowball into large differences in the final logits.

This creates a **vicious feedback loop**: mismatch leads to biased and noisy gradients, which might push the parameters further into a numerically sensitive region, which in turn worsens the mismatch for the next iteration, until the system collapses.

------

# **4. Attempts to Alleviate Training-Inference Mismatch**

Next, we will list the methods we have attempted to mitigate the training-inference mismatch. Some of these methods were helpful, while others were not.

## **4.1 Ineffective Attempts**

### 4.1.1 Use FP32 LM Head

Inspired by the [**Minimax-M1 technical report**](https://arxiv.org/abs/2506.13585) and the blog post《[**Your Efficient RL Framework Secretly Brings You Off-Policy RL Training**](https://fengyao.notion.site/off-policy-rl)》, we patched vLLM to casting lm_head to fp32 precision. However, in our experiments, the mismatch problem still persisted after the patch and the model collapse was unavoidable. Figure 11 shows a failed on-policy experiment on the L20 GPU using a bf16 lm_head in the vLLM engine, and an experiment that started using an fp32 lm head on the vLLM engine after the 200th training step from the collapsed experiment. It can be observed that both experiments ultimately collapsed, and the experiment with the fp32 lm_head still exhibited a `vllm-kl` explosion.

------

### **4.1.2 Disable Chunked Prefill**

We also attempted to resume the RL training from the 200th training step of the failed  experiment with bf16 lm_head  in Section 4.1.1 , disabling [chunked prefill](https://docs.vllm.ai/en/v0.4.2/models/performance.html) to see if it would resolve the crash. However, our experimental results (as shown in Figure 12) indicate that this approach did not resolve the crash issue.

### 4.1.3 Enable `enforce_eager`  and `free_cache_engine`

The [VeRL's official recipe of DAPO](https://verl.readthedocs.io/en/latest/algo/dapo.html) mentions that enabling CUDA graphs ( `enforce_eage=False`) may cause model performance degradation.  To investigate whether it affects the training-inference mismatch, we conducted an ablation study to examine the impact of the vLLM engine hyperparameter `enforce_eager`  with the consideration of another hyperparameter `free_cache_engine`.  We conducted experiments on reasoning RL. We performed on-policy GRPO experiments on H100 GPUs using Qwen3-4B-Base as the base model, running a total of four experimental settings: an exhaustive combination of the  hyperparameters `enforce_eager` and `free_cache_engine`, each set to either `True` or `False`. The performance was evaluated on the AIME24 benchmark. The experimental results are presented in Figure 13. As can be seen from the figure, adjusting the values of `enforce_eager` and `free_cache_engine` has no significant impact on the training-inference mismatch and test performance.

------

## **4.2 Effective Attempts**

### 4.2.1 A Principled Solution: Distribution Correction

The training-inference mismatch turns our otherwise on-policy RL problem into an off-policy one, where the policy used for generating rollouts (the *behavior policy*, $\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}$) differs from the policy being trained (the *target policy*, $\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}$). One theoretically sound way to correct for this distributional shift is **Importance Sampling (IS)**. However, the specific formulation of IS is critical for maintaining an unbiased gradient and achieving stable training.

Inspired by the findings in [(Yao et al., 2025)](https://fengyao.notion.site/off-policy-rl), which first highlighted this implicit off-policy issue due to training-inference mismatch, we analyze two primary forms of IS: the theoretically sound **sequence-level IS** and the common but biased **token-level IS** approximation.

------

### The Principled Estimator: Sequence-Level IS

The correct, unbiased policy gradient estimator applies a single importance ratio over the entire generated sequence (trajectory) $y$. This correctly re-weights the expectation from the behavior policy to the target policy, yielding the true gradient of the objective function $J(\theta)$.

Let's derive the Sequence-Level IS estimator $g_{\mathrm{seq}}(\theta)$ step-by-step.

1. The objective is to maximize the expected reward under the target **FSDP policy**:

   $$ J(\theta) = \mathbb{E}*{x \sim \mathcal{D}, y \sim \textcolor{blue}{\pi*{\theta}^{\mathrm{fsdp}}}(\cdot|x)}[R(x,y)] $$

2. The true policy gradient is therefore:

   $$ g(\theta) = \nabla_{\theta} J(\theta) = \mathbb{E}*{x \sim \mathcal{D}, y \sim \textcolor{blue}{\pi*{\theta}^{\mathrm{fsdp}}}(\cdot|x)}\left[R(x,y) \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}(y|x)\right] $$

3. Since we can only sample from the **vLLM policy**, we use importance sampling to change the distribution of the expectation:

   $$ g_{\mathrm{seq}}(\theta) = \mathbb{E}*{x \sim \mathcal{D}, y \sim \textcolor{red}{\pi*{\theta}^{\mathrm{vllm}}}(\cdot|x)}\left[ \frac{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}(y|x)}{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}(y|x)} \cdot R(x,y) \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}(y|x) \right] $$

This is essentially off-policy **REINFORCE algorithm ([Williams, 1992](https://doi.org/10.1007/BF00992696))** . This estimator is mathematically equivalent to the standard advantage form of the policy gradient. The key is to show that the importance sampling ratio precisely corrects the expectation, revealing the true on-policy gradient underneath, which can then be refined.

- Click to see the detailed derivation

  ### Step 1: Convert the Expectation to its On-Policy Form

  The IS estimator takes its expectation with respect to the behavior policy $\textcolor{red}{\pi^{\mathrm{vllm}}}$. By writing out the definition of expectation as an integral, the behavior policy density $\textcolor{red}{p_{\theta}^{\mathrm{vllm}}}(y|x)$ cancels out:

  $$ g_{\mathrm{seq}}(\theta) = \int \left( \frac{\pi_{\textcolor{blue}{\theta}}^{\mathrm{fsdp}}(y|x)}{\pi_{\textcolor{red}{\theta}}^{\mathrm{vllm}}(y|x)} \cdot R(x,y) \cdot \nabla_{\theta} \log \pi_{\textcolor{blue}{\theta}}^{\mathrm{fsdp}}(y|x) \right) \pi_{\textcolor{red}{\theta}}^{\mathrm{vllm}}(y|x) \,dy = \int \left( R(x,y) \cdot \nabla_{\theta} \log \pi_{\textcolor{blue}{\theta}}^{\mathrm{fsdp}}(y|x) \right) \pi_{\textcolor{blue}{\theta}}^{\mathrm{fsdp}}(y|x) \,dy $$

  ------

  This leaves us with the true on-policy gradient, where the expectation is now over the target policy $\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}$. This proves that $g_{\mathrm{seq}}(\theta)$ is an unbiased estimator of the true policy gradient.

  $$ g(\theta) = \mathbb{E}_{y \sim \textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}} \left[ R(x,y) \cdot \nabla_{\theta} \log \pi_{\textcolor{blue}{\theta}}^{\mathrm{fsdp}}(y|x) \right] $$

  ### Step 2: Decompose into Timesteps and Apply Causality

  Now working with the simpler on-policy expression, we expand the trajectory-level terms and apply the principle of **causality** (an action at step $t$ only affects future rewards). This allows us to replace the total reward $R(x,y)$ with the **return-to-go** $G_t = \sum_{k=t}^{|y|-1} r(s_k, a_k)$.

  $$ g(\theta) = \mathbb{E}*{y \sim \textcolor{blue}{\pi*{\theta}}} \left[ \sum_{t=0}^{|y|-1} G_t \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}}(a_t|s_t) \right] $$

  ------

  ### Step 3: Introduce the Advantage Function

  For variance reduction, we subtract a state-dependent **baseline,** the value function $V^{\textcolor{blue}{\pi_{\theta}}}(s_t)$. This converts the return-to-go into the **Advantage Function**, $A^{\textcolor{blue}{\pi_{\theta}}}(s_t, a_t) = G_t - V^{\textcolor{blue}{\pi_{\theta}}}(s_t)$.

  $$ g(\theta) = \mathbb{E}*{y \sim \textcolor{blue}{\pi*{\theta}}} \left[ \sum_{t=0}^{|y|-1} A^{\textcolor{blue}{\pi_{\theta}}}(s_t, a_t) \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}}(a_t|s_t) \right] $$

  ------

  ### Step 4: Convert to State-Level Expectation

  The final step rewrites the expectation over trajectories into an equivalent expectation over the state-visitation distribution $d_{\textcolor{blue}{\pi_{\theta}}}$ induced by the target policy.

  - The detailed math from Step 3 to Step 4

    ### **Step A: Start with the Trajectory-Level Expectation**

    $$ g(\theta) = \mathbb{E}*{y \sim \textcolor{blue}{\pi*{\theta}}} \left[ \sum_{t=0}^{|y|-1} A^{\textcolor{blue}{\pi_{\theta}}}(s_t, a_t) \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}}(a_t|s_t) \right] $$

    ### **Step B: Apply Linearity of Expectation**

    Swap the expectation and summation operators. We can extend the sum to infinity because any terms for timesteps $t$ beyond a trajectory's finite length $|y|$ are zero, so we are just adding zeros.

    $$g(\theta) = \sum_{t=0}^{\infty} \mathbb{E}*{y \sim \textcolor{blue}{\pi*{\theta}}} \left[ A^{\textcolor{blue}{\pi_{\theta}}}(s_t, a_t) \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}}(a_t|s_t) \right] $$

    ### **Step C: Expand the Expectation into an Explicit Sum**

    Rewrite the inner expectation by summing over every possible state $s$ and action $a$, weighted by their joint probability $P(s_t=s, a_t=a) = P(s_t=s) \cdot \textcolor{blue}{\pi_{\theta}}(a|s)$.

    $$ g(\theta) = \sum_{t=0}^{\infty} \sum_{s \in \mathcal{S}} P(s_t=s | \textcolor{blue}{\pi_{\theta}}) \sum_{a \in \mathcal{A}} \textcolor{blue}{\pi_{\theta}}(a|s) \cdot \left[ A^{\textcolor{blue}{\pi_{\theta}}}(s, a) \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}}(a|s) \right] $$

    ### **Step D: Introduce the State Occupancy Measure**

    Rearrange the summations to group all terms related to state $s$.

    $$ g(\theta) = \sum_{s \in \mathcal{S}} \left( \sum_{t=0}^{\infty} P(s_t=s | \textcolor{blue}{\pi_{\theta}}) \right) \left( \sum_{a \in \mathcal{A}} \textcolor{blue}{\pi_{\theta}}(a|s) \cdot \left[ \dots \right] \right) $$

    The first term, $\sum_{t=0}^{\infty} P(s_t=s)$, is the definition of the **state occupancy measure**, $d_{\textcolor{blue}{\pi_{\theta}}}(s)$. The second term is the definition of an expectation over actions, $\mathbb{E}*{a \sim \textcolor{blue}{\pi*{\theta}}(\cdot|s)}[\dots]$.

    $$ g(\theta) = \sum_{s \in \mathcal{S}} d_{\textcolor{blue}{\pi_{\theta}}}(s) \cdot \mathbb{E}*{a \sim \textcolor{blue}{\pi*{\theta}}(\cdot|s)} \left[ A^{\textcolor{blue}{\pi_{\theta}}}(s, a) \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}}(a|s) \right] $$

    ### **Step E: Arrive at the Final Form**

    The sum over all states weighted by the state occupancy measure is, by definition, an expectation over the state distribution $d_{\textcolor{blue}{\pi_{\theta}}}$.

    $$g(\theta) = \mathbb{E}*{s \sim d*{\textcolor{blue}{\pi_{\theta}}}} \left[ \mathbb{E}*{a \sim \textcolor{blue}{\pi*{\theta}}(\cdot|s)} \left[ A^{\textcolor{blue}{\pi_{\theta}}}(s,a) \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}}(a|s) \right] \right] $$

This derivation leads us to the final advantage form of the policy gradient:

$$g_{\mathrm{seq}}(\theta) = \mathbb{E}*{s \sim d*{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}}} \mathbb{E}*{a \sim \textcolor{blue}{\pi*{\theta}^{\mathrm{fsdp}}}(\cdot|s)} \left[ A^{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}}(s,a) \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}(a|s) \right] $$

Here, $s=(x,y_{<t})$ **is the state (prefix), $a=y_t$ is the action (token). The term** $d_{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}}$ **is the state occupancy measure** under the target FSDP policy. It is formally defined as the expected number of times a state $s$ is visited when following policy $\pi$:

$$ \textcolor{blue}{d_{\pi}(s) := \mathbb{E}*{x' \sim \mathcal{D}, y' \sim \pi(\cdot|x')} \left[ \sum*{t'=0}^{|y'|-1} \mathbb{I}\{ (x', y'*{<t'}) = s \} \right] = P(x) \cdot \prod*{k=0}^{t-1} \pi(y_k|x,y_{<k})} $$

This estimator is **unbiased**, meaning $g_{\mathrm{seq}}(\theta) = g(\theta)$. For numerical stability, **Truncated Importance Sampling (TIS)** is used, which clips the sequence-level ratio $\rho(y|x)$ at a constant $C$.

------

### A Common Biased Estimator: Token-Level IS

A common heuristic, often inspired by algorithms like PPO and used in [(Yao et al., 2025)](https://fengyao.notion.site/off-policy-rl), applies a per-token importance ratio. While this typically has lower variance than the sequence-level ratio, it is a **biased estimator** that is theoretically unsound for autoregressive models.

Let's derive the Token-Level IS gradient estimator, $g_{\mathrm{tok}}(\theta)$.

1. The formulation begins by incorrectly applying the importance sampling ratio inside the sum over timesteps: i.e., $g_{\mathrm{tok}}(\theta)$ is defined as

   $$ \mathbb{E}*{x \sim \mathcal{D}, y \sim \textcolor{red}{\pi*{\theta}^{\mathrm{vllm}}}(\cdot|x)}\left[ R(x,y) \cdot \sum_{t=0}^{|y|-1} \frac{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}(y_t|x,y_{<t})}{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}(y_t|x,y_{<t})} \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}(y_t|x,y_{<t}) \right] $$

2. We can rewrite this expectation over trajectories as an expectation over the states visited under the **vLLM policy**:

   $$ g_{\mathrm{tok}}(\theta) = \mathbb{E}*{s \sim d*{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}}} \mathbb{E}*{a \sim \textcolor{red}{\pi*{\theta}^{\mathrm{vllm}}}(\cdot|s)} \left[ \frac{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}(a|s)}{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}(a|s)} \cdot A^{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}}(s,a) \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}(a|s) \right] $$

3. *Note: Here,* $R(x,y)$ *is the empirical return from the full trajectory sampled by* $\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}$*, which serves as a Monte Carlo estimate for the state-action value* $Q^{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}}(s,a)$*. I*ntroducing a baseline and changing the expectation over actions gives the final form:

   $$ g_{\mathrm{tok}}(\theta) = \mathbb{E}*{s \sim d*{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}}} \mathbb{E}*{a \sim \textcolor{blue}{\pi*{\theta}^{\mathrm{fsdp}}}(\cdot|s)} \left[ A^{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}}(s,a) \cdot \nabla_{\theta} \log \textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}(a|s) \right] $$

This final expression clearly reveals the gradient bias of token-level IS.  It can be observed clearly that $J(\theta)$ can be optimized by $g_\text{tok}(\theta)$ **only when**  $\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}$ stays in the trust region of $\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}$, where $d_{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}}\approx d_{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}}$ and $A^{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}}\approx A^{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}}$.

------

### The Source of Bias in Token-Level IS

Comparing $g_{\mathrm{tok}}(\theta)$ with the true gradient $g_{\mathrm{seq}}(\theta)$ reveals two distinct and significant errors that render the token-level estimator biased.

### **Source 1: State Occupancy Mismatch 🗺️**

A sound off-policy correction must account for two distributional shifts: the action probabilities and the state visitation probabilities. The token-level method only corrects for the first.

- **True Gradient (**$g_{\mathrm{seq}}$**):** The expectation is over states visited under **the correct target fsdp distribution**, $\mathbb{E}*{s \sim d*{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}}}$.
- **Flawed Gradient (**$g_{\mathrm{tok}}$**):** The expectation is over states visited under the **incorrect behavior vLLM distribution**, $\mathbb{E}*{s \sim d*{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}}}$.

This implicitly assumes the state occupancy ratio is 1, i.e. ${d_{\textcolor{blue}{\pi^{\mathrm{fsdp}}}}(s)} /{d_{\textcolor{red}{\pi^{\mathrm{vllm}}}}(s)} = 1$. This assumption is catastrophically violated in autoregressive models; due to deterministic transitions, a single different token choice guarantees that state trajectories diverge completely. By ignoring this, $g_{\mathrm{tok}}(\theta)$ introduces a large, uncontrolled bias.

### **Source 2: Mismatched Reward Signal 🎯**

The second critical error is that the token-level gradient weights the update with a reward signal from the **wrong policy**.

- **True Gradient (**$g_{\mathrm{seq}}$**):** The update is scaled by the advantage function of the **target fsdp policy**, $A^{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}}$, representing the expected future reward under that policy.
- **Flawed Gradient (**$g_{\mathrm{tok}}$**):** The update is scaled by the advantage function of the **behavior vLLM policy**, $A^{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}}$.

The gradient for the target policy is being scaled by a reward signal that belongs to the behavior policy. Because both the state distribution and the reward signal are fundamentally mismatched, the token-level gradient is a biased and theoretically unsound estimator.

 📌 These theories suggest that despite the potential for lower variance in token-level methods, gradient bias nonetheless persists, potentially leading to training instability— **a prediction our experiments go on to confirm.**  We also present the detailed bias and variance analysis ([**part 1**](https://richardli.xyz/rl-collapse-1) & [**part2**](https://richardli.xyz/rl-collapse-2)) on both token-level and sequence-level methods.

------

### Experimental Validation

Our theoretical analysis predicts that the biased token-level IS will be unstable and ultimately fail, while the unbiased sequence-level IS will be robust. Our experiments confirm this.

### **IS Prevents Gradient Explosion, but Token-Level Still Fails**

We resumed a crashed experiment on an L20 GPU with both token-level TIS and sequence-level TIS ($C=2$), starting from its 200th training step. As shown in Figure 14, while both initially prevent the gradient explosion seen in the vanilla experiment, the run with **token-level TIS still collapses later**. The run with sequence-level TIS remains stable, validating our theory that the biased gradients from the token-level method eventually lead to failure.

### **Token-Level TIS in Reasoning RL**

Although token-level TIS fails in our complex TIR experiments, it can help prevent collapse in simpler reasoning RL, where the mismatch is smaller. In on-policy GRPO and RLOO experiments (Figure 15), token-level TIS prevented gradient explosion, but training remained unstable and did not achieve better final performance, likely due to the underlying gradient bias.

### **TIS Prolongs Training But Can Suffer from Instability**

We conducted an on-policy TIR experiment from scratch on L20 GPUs with sequence-level TIS ($C=2$). As shown in Figure 16, while the method prevented a complete collapse, the reward curve exhibited continuous fluctuations after reaching a plateau, and its test performance did not exceed the peak achieved by the vanilla experiment before it crashed.

### **Masked Importance Sampling (MIS)**

To improve upon TIS, we propose **Masked Importance Sampling (MIS)**, which masks the policy loss for sequences where the IS ratio exceeds the threshold $C$ (i.e., $\rho(y|x) \gets \rho(y|x) \mathbb{I}\{\rho(y|x) \le C\}$). As shown in Figure 17, MIS not only stabilized training but also surpassed the peak training rewards and test accuracy of both the vanilla and TIS experiments.

### **Token-Level MIS vs. Sequence-Level MIS**

Finally, we compared token-level MIS against sequence-level MIS. As expected, Figure 18 shows that while both prevent the initial gradient explosion, the **token-level MIS experiment still collapses**. This reinforces our conclusion that for complex, long-horizon autoregressive tasks, only a theoretically sound, sequence-level correction is reliable.

### 4.2.2 Top-p Sampling

As we discussed above, we observe that low probability tokens of **vLLM** policy  are more prone to severe training-inference mismatch issues, leading to extremely low probabilities of **FSDP** policy. To further corroborate this, we conducted the following top-p ablation study. We ran on-policy GRPO experiments on L20 GPUs and set the `top-p` hyperparameters for vLLM sampling policy to 0.98, 0.99, and 0.999 respectively (note that we did not apply importance sampling for gradient correction). By setting a smaller `top-p` value, we aim to decrease the frequency of extremely low-vllm-probability tokens during the inference stage, thereby mitigating the training-inference mismatch. Our ablation results are shown in Figure 19. As expected, the `vllm-kl` metric indicates that a smaller `top-p` reduces the occurrence of `vllm-kl` spikes. However, it is important to note that a smaller `top-p` also increases the distribution divergence between the **vLLM** policy  $\textcolor{red}{\pi^{\text{vllm}}*\theta}$ and the **FSDP** policy   $\textcolor{blue}{\pi^{\text{fsdp}}*\theta}$. Consequently, without applying TIS, the gradient bias becomes larger, leading to slower reward improvement as `top-p` decreases.

### **4.2.3 Use other GPU series**

After discovering that the training–inference mismatch in experiments run on the H20 GPUs was significantly smaller than that on the L20 GPUs, we switched all our TIR experiments to the H20 GPUs, which greatly reduced the occurrence of training crashes. Figure 20 shows the results of the two on-policy GRPO experiments trained from scratch under identical configurations on the L20 and H20 GPUs, respectively. It can be observed that the on-policy experiment running on the H20 GPUs significantly extended the stable training duration and achieved better performance.

### **4.2.4 Diable Cascade Attention in vLLM**

According to this [GitHub issue](https://github.com/vllm-project/flash-attention/pull/87), we set `disable_cascade_attn=True` when initializing the vLLM engine and found that it significantly helps reduce the training-inference mismatch in experiments conducted on A100 GPUs. We performed two on-policy GRPO experiments using Qwen3-14B-Base as the base model on A100 GPUs, with `disable_cascade_attn` set to `True` and `False`, respectively. The results are shown in Figure 21. It can be observed that after disabling cascade attention, the `vllm-kl` metric decreased from the range of 5e-2 to 1e-1 to around 1e-3, indicating a substantial reduction in training-inference mismatch. In addition, the rewards on the training set also increased appropriately.

------

# **5. Conclusion & Key Takeaways for Practitioners**

The training-inference mismatch is not a niche bug but a fundamental, growing challenge across modern **reasoning and agentic RL**, driven by the necessary pursuit of performance. Our investigation provides a clear roadmap for diagnosing and mitigating it. 📝

- **Mismatch is an Inevitable Trade-Off:** Accept that high-speed inference will always diverge from training calculations. This is a core trade-off, not a temporary flaw.
- **Monitor Your Health:** The `vllm-kl` metric is a vital early warning system. Track it alongside Perplexity (PPL) and gradient norms to predict and diagnose instability before it leads to collapse.
- **Identify the Real Culprit:** The problem is not random. It is systematically triggered by **low-probability tokens**, which are generated more frequently when the model processes out-of-distribution (OOD) inputs—a common scenario in tool-use and multi-turn applications.
- **Hardware is a First-Order Variable:** The same code can collapse on one GPU architecture and train perfectly on another. Always validate your setup on your target hardware, as results may not be exactly portable.
- **Use Theoretically-Sound Corrections:** While changing hardware or tuning samplers can help, the most robust and principled solution is algorithmic. Our work demonstrates that theoretically biased, **token-level corrections are insufficient and can still fail in our experiments**. In contrast, **sequence-level methods like Truncated (Seq-TIS) and Masked (Seq-MIS) Importance Sampling** directly address the gradient bias by correcting for the full state trajectory. These methods are essential for maintaining stability and should be considered a default for any serious LLM-RL training stack.

We hypothesize that this mismatch between inference engines (e.g., vLLM) and training frameworks (e.g., Megatron-LM) will also be a significant issue for **Mixture-of-Experts (MoE) RL**, which represents an interesting and critical direction for future investigation.