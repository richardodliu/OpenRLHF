# Your Efficient RL Framework Secretly Brings You Off-Policy RL Training



**TL;DR**

In modern RL training frameworks (e.g., VeRL), different implementations are used for  rollout generation (e.g., vLLM) and model training (e.g., FSDP). Here, we show the **implementation gap** implicitly turns the on-policy RL to be **off-policy**, and discuss a simple yet effective importance sampling technique for handling such discrepancy. 



# The Mismatch Problem

For simplicity, we use the REINFORCE algorithm as an example, which supposedly updates the policy $\pi$ — an LLM parameterized by $\theta$ — via:

$$
\theta \gets \theta + \mu \cdot  \mathbb{E}_{\underbrace{a \sim{\pi}(\theta)}_{rollout}} [R(a)\cdot \underbrace{\nabla_\theta \log {\pi}(a, \theta)}_{\tiny{training}}].
$$

In practice, rollout generation is expensive and modern RL frameworks (e.g., [VeRL](https://github.com/volcengine/verl)) typically employ highly optimized inference engines (e.g., **vLLM, SGLang**) to boost throughput, while using a separate backend (e.g., **FSDP, Megatron**) for model training. Such hybrid design makes the updating:

$$
\theta \gets \theta + \mu \cdot  \mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} [R(a)\cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)].
$$

*Here, we use $\pi_{\rm sampler}$ to represent the model loaded with the inference engine (e.g., vLLM, SGLang) and $\pi_{\rm learner}$ to denote the same model instantiated with the training backend (e.g., FSDP, Megatron). Unless unspecified, our experiments use vLLM and FSDP as sampler and learner backends.* 

There is unexpected **rollout-training mismatch** observed. ****As shown in [Figure 1](https://www.notion.so/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56?pvs=21), despite  $\textcolor{blue}{\pi_{\text{fsdp}}}$ and $\textcolor{red}{\pi_{\text{vllm}}}$ sharing the same model parameters $\theta$, they can produce **significantly different token probabilities**. For certain tokens $a$, they even yield contradictory predictions, i.e., $\textcolor{red}{\pi_{\text{vllm}}}(a, \theta)\!=\!1$ and $\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta)\!=\!0$. This unexpected behavior implicitly breaks the **on-policy** assumption, secretly making the RL training become **off-policy**.

# How to Fix It?

## Mitigate the system-level mismatch

Does higher-precision vLLM help? We first hypothesized that vLLM is the root cause, and thus we patched vLLM to address two commonly suspected contributors to the mismatch problem. 

- **Inaccessible true sampling probabilities:**  vLLM v1 engine [does not support](https://docs.vllm.ai/en/v0.10.0/usage/v1_guide.html?h=immediately#semantic-changes-to-logprobs) directly returning the adjusted probabilities used for sampling, introducing an additional gap.
  
    → Our patch forces vLLM to return the actual probabilities used for sampling [[upstreamed]](https://github.com/vllm-project/vllm/pull/22387). 
    
- **Backend numerical differences:** vLLM lm_head’s precision [does not match](https://discuss.vllm.ai/t/numerical-difference-between-vllm-logprobs-and-huggingface-logprobs/151) that of HuggingFace transformers, which is also denoted in the MiniMax-M1 [technical report](https://arxiv.org/pdf/2506.13585#page=8).
  
    → Our patch provides the option to force vLLM casting lm_head to fp32.
    

However, as shown in [Figure 1](https://www.notion.so/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56?pvs=21), the mismatch problem **still exists** after applying both patches. 

## Embrace the mismatch — Apply algorithm-level fix

Instead of mitigating the distribution mismatch at the system level, we propose to adapt the model update such that it’s aware of this mismatch. A simple way is via importance-sampling correction. Specifically, we handle the mismatch between $\textcolor{blue}{\pi_{\text{learner}}}$ and $\textcolor{red}{\pi_{\text{sampler}}}$ by adding the importance ratio to the model update, i.e., changing the current gradient computation from

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} [R(a)\cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)],
$$

to 

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \Bigl[\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)} \cdot R(a)\cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)\Bigr].
$$

While there has been extensive study on how to design a stable and effective importance sampling, in practice we find it usually sufficient to use a classical technique, [truncated importance sampling](https://ionides.github.io/pubs/ionides08-jcgs.pdf): 

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \Bigl[\underbrace{\min\Bigl(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)}, C\Bigr)}_{\text{truncated importance ratio}} \cdot R(a) \cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)\Bigr],
$$

where C is a hyper parameter. 

### Extension to Other Algorithms

It is straightforward to extend the above analyses to other algorithms, as one can switch the exact form for gradient computation from REINFORCE $R(a) \cdot\nabla \log {\pi}(a, \theta)$ to any form. Here, we provide similar analyses to the commonly used PPO algorithm as an additional example. 

The policy gradient of PPO  $\nabla_\theta L^{\mathrm{CLIP}}(\theta)$ is defined as:

$$
\small{ \mathbb{E}_{a\sim\pi_{\theta_{\mathrm{old}}}}
\Bigl[
\nabla_\theta \min\Bigl(
\frac{\pi_\theta(a)}{\pi_{\theta_{\mathrm{old}}}(a)}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\pi_\theta(a)}{\pi_{\theta_{\mathrm{old}}}(a)},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]}.
$$

**To improve throughput**, hybrid RL systems adopt vLLM engine for rollout generation —sampling tokens $a$ from $\pi_{\theta_{\text{old}}}$, while using FSDP backend both to sample from $\pi_\theta$ and to [recompute](https://github.com/volcengine/verl/blob/3e2bceb1afcaa77ebc40106a64f7b440509b67e1/verl/workers/fsdp_workers.py#L782) the token probabilities for $\pi_{\theta_{\mathrm{old}}}$ for gradient computation: 

$$
\small{
\mathbb{E}_{a\sim\textcolor{red}{\pi_{\text{sampler}}}(\theta_{\mathrm{old}})}
\Bigl[
\nabla_\theta \min\Bigl(
\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\mathrm{old}})}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\mathrm{old}})},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]
}.
$$

Similar to the analysis above, the gap between $\textcolor{blue}{\pi_{\text{learner}}}$ and $\textcolor{red}{\pi_{\text{sampler}}}$ shows up again, and we fix it with truncated importance sampling:

$\small{\mathbb{E}_{a\sim\textcolor{red}{\pi_{\mathrm{sampler}}}(\theta_{\mathrm{old}})}\Bigl[\underbrace{\min\Bigl(  \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})},  C\Bigr)}_{\text{truncated importance ratio}}\cdot\nabla_{\theta}\,\min\Bigl(  \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}\,\hat{A},  \mathrm{clip}\Bigl(    \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})},    1-\epsilon,\;1+\epsilon  \Bigr)\,\hat{A}\Bigr)\Bigr]}$where C is a hyper-parameter.

**Additional Discussion on PG, Sequence, and Token**

Our discussion above does not touch the specific formulation of the state and the action. We discuss policy gradient at both token and sequence levels, how they connect to each other, and the impact of learner-sampler mismatch:

[Policy Gradient, Sequence, and Token— Part I: Basic Concepts](https://www.notion.so/Policy-Gradient-Sequence-and-Token-Part-I-Basic-Concepts-28b721e3f6c480b88b5be1d89512ac3a?pvs=21)

[Policy Gradient, Sequence, and Token— Part II: Learner-Sampler Mismatch](https://www.notion.so/Policy-Gradient-Sequence-and-Token-Part-II-Learner-Sampler-Mismatch-28b721e3f6c480f8a4b0e1f8301d90ac?pvs=21)

### **Connection to Classical Wisdom**

**Importance Sampling** 

When direct Monte Carlo estimation of the expected value under a target distribution is difficult, importance sampling allows us to sample from an alternative distribution instead. In our case, the target distribution is $\textcolor{blue}{\pi_{\text{learner}}}$, but it is extremely slow to sample from. Using a separate backend (e.g., vLLM) for rollout generation means that we are sampling from $\textcolor{red}{\pi_{\text{sampler}}}$ instead.The discrepancy is then corrected by weighting each sample with an importance ratio:

$$
\mathbb{E}_{a \sim \textcolor{blue}{\pi_{\text{learner}}}(\theta)} [R(a)] 
= \mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \left[ 
\underbrace{\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)}}_{\tiny\text{importance ratio}} \cdot R(a) 
\right].
$$

**Decoupled PPO** 

[Decoupled PPO](https://arxiv.org/pdf/2110.00641) is a special case of using importance sampling to bridge the gap between the rollout generation and gradient computation, which has been adopted in async-RL frameworks like [AReaL](https://arxiv.org/pdf/2505.24298#page=6). It is worth mentioning that AReaL didn’t implement the truncated importance ratio as we discussed here. Instead, AReaL [will drop the training sample entirely](https://github.com/inclusionAI/AReaL/blob/main/realhf/impl/model/utils/ppo_functional.py#L127), if the importance ratio exceeds a pre-defined threshold. 

# Experiments

We further conducted empirical analyses to elaborate on the impact of the distribution gap and the effectiveness of the proposed Truncated Importance Sampling (TIS) fix. 

## Does the gap matter a lot?

We conduct experiments using Qwen2.5-32B dense model with the popular [DAPO](https://github.com/volcengine/verl/tree/main/recipe/dapo) recipe. The data is processed following the [community guidance](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/discussions/3). The result is visualized in [Figure 1](https://www.notion.so/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56?pvs=21). 

Due to the resource constraints, we only finished the first 250 steps of the training, yet the gap-aware fix TIS has already helped boost the performance significantly. As the only difference between these two runs is the introduced term, i.e., $\min(\frac{\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta)}{\textcolor{red}{\pi_{\text{vllm}}}(a, \theta)}, C)$, the improvement showcased the potential impact of the distribution gap. 

## How well can TIS fix it?

![gsm8k_int8.png](attachment:766b9627-d7c4-4f0d-ba10-6eda045390a1:gsm8k_int8.png)

***Figure 2.** **Left**: Token-level probability differences. **Right**: Performance comparison for normal RL training on GSM8K and RL training with INT8 quantized rollouts. Experiments are conducted on Qwen2.5-0.5B dense model using one node of 4 A6000 GPU.*

We designed a controlled experiment to measure how well TIS fixes the issue. We conduct RL training following [GSM8K example in the verl tutorial](https://verl.readthedocs.io/en/latest/start/quickstart.html) and use two different settings:

1. **Normal RL training**: the maximum token probability difference is considerably smaller (~0.4) than the previous setting (1.0 on DAPO on Qwen-2.5-32B dense), 
2. [**RL training with INT8 quantized rollouts instead of bf16 rollouts**](https://fengyao.notion.site/flash-rl): the maximum token probability difference is considerably larger (1.0) than normal RL training. 

We conduct regular PPO training in setting 1, which is “almost” on-policy, and both regular PPO training and PPO training with truncated importance sampling in setting 2, whose generation rollout and gradient computation have a larger gap. 

As visualized in [Figure 2](https://www.notion.so/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56?pvs=21), performing PPO in setting 2 leads to significant performance degradation, comparing to the PPO in setting 1. At the same time, applying truncated importance sampling manages to mitigate the gap greatly, effectively allowing the setting 2 run achieves a similar performance with setting 1. 

More analysis are provided in [Section Analysis](https://www.notion.so/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56?pvs=21) below.

## Does TIS always help?

![dapo_1.5B.png](attachment:40ff1a9f-0eda-4aab-924e-fc99bae3d95c:dapo_1.5B.png)

***Figure 3.** **Left**: Token probability differences brought by the mismatch problem. **Right**: Performance comparison between normal RL training and after fixing the mismatch problem. The experiment is conducted on the DeepSeek-R1-Distill-Qwen-1.5B model using 4 nodes of 8 H100 GPUs.* *In this case, the mismatch is not huge because we use a standard bfloat16 rollout in both runs and the model is relatively small. [[wandb log](https://wandb.ai/llychinalz/Flash-DAPO/?nw=xv3gvup0lw8)]*

We also observed that, in cases where the probability difference is relatively small, introducing the additional Truncated Importance Sampling term cannot bring performance gain. Meanwhile, it is worth mentioning that, the importance sampling ratio term would have a value of 1.0, in the strict on-policy RL setting. 

# TIS Analysis

## Analysis about different TIS-Variants

We also summarize two alternative solutions for mitigating the distribution gap. 

- **PPO Importance Sampling (PPO-IS)**
  
    $$
    \small{ \mathbb{E}_{a\sim\textcolor{red}{\pi_{\mathrm{sampler}}}(\theta_{\mathrm{old}})}\Bigl[\nabla_{\theta}\,\min\Bigl(  \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\mathrm{old}})}\,\hat{A},  \mathrm{clip}\Bigl(    \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\mathrm{old}})},    1-\epsilon,\;1+\epsilon  \Bigr)\,\hat{A}\Bigr)\Bigr]}
    $$
    
    *Note: Colossal framework uses this implementation.
    
- **Vanilla Importance Sampling (vanilla-IS)**
  
    $\small{\mathbb{E}_{\textcolor{red}{\pi_{\mathrm{sampler}}}(\theta_{\mathrm{old}})}\Bigl[\underbrace{\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})} }_{\text{importance ratio}}\cdot\nabla_{\theta}\,\min\Bigl(  \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}\,\hat{A},  \mathrm{clip}\Bigl(    \frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})},    1-\epsilon,\;1+\epsilon  \Bigr)\,\hat{A}\Bigr)\Bigr]}$*Note: Nemo-RL uses this implementation.
    

To assess the effectiveness of TIS and understand the impact of its design choices, we conducted experiments comparing TIS with the two variants above. TIS outperforms both variants consistently, especially in cases where the gap is large (e.g., FP8/INT8). 

**Figure 4.** We ablate different rollout-training mismatch mitigation strategies on Qwen2.5-0.5B with GSM8k. Note PPO-IS and Vanilla-IS achieves near 0 accuracy for INT8 rollouts thus being highly overlapped. We also plot the KL divergence between vLLM sampled distribution and the FSDP distribution on the right. 



**Vanilla-IS v.s. TIS**

Regarding vanilla-IS, the instability is mainly from the cases when the rollout $a\sim\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})$ is sampled with low probability and thus the importance ratio is large, amplifying gradient variance by $(\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})})^2$. Therefore, we use a clamp operation in our truncated importance sampling (TIS) for training stabilization. For example, when the ratio $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})}$ reaches 16 for one token, the gradient noise for that token would be amplified by 256 times via **Vanilla-IS**, 4 times via **TIS-2**, or 64 times via **TIS-8**.

**PPO-IS v.s. TIS**

Since the release of our blog, a lot of people have asked why we do not directly incorporate the importance sampling into PPO (i.e., the PPO-IS variant above). Frankly speaking, we start by changing the ppo clip directly as in PPO-IS, but it doesn't perform well in our experiment setting.

As to the underlying rationale, by doing the PPO-IS, the gradient is actually still biased from the on-policy version of PPO. In other words, although it may still optimize towards the unbiased objective, it may be less effective compared to PPO.

Additionally, we remark that the PPO trust region technique is proposed to constrain the probability ratio between rollout $\theta_{\rm old}$ and current model $\theta$ to be close to 1 to approximate on-policy REINFORCE gradient. However in PPO-IS, even when $\theta=\theta_{\rm old}$, the probability ratio $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\rm old})}$ is already not equal to 1 due to the mismatch — this makes the clipping happen with high possibility and the training much less informative. Furthermore, in our TIS method, we separately clip $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\theta_{\mathrm{old}})}$ and $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}$ and thus much more mild; notice $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta)}{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\mathrm{old}})}$ equals to 1 when $\theta=\theta_{\rm old}$ which is suitable for the trust region constraint.



## From Ill-conditioned to Benign

Beyond rollout acceleration, rollout quantization serves as an effective testbed for examining the impact of distribution gaps between rollout generation and gradient computation. We demonstrate that RL training with quantized rollouts exhibits characteristic instabilities commonly observed in other scenarios when this gap is not addressed. Additionally, introducing the TIS term making RL training stable and benign.

### Entropy Collapse and Abnormal Response Length

A number of prior works have shown that RL training in LLM will lead to entropy collapse — the categorical distribution on the token level becomes close to one-hot distribution, restricting RL training from exploration effectively. 

Our INT8 rollout experiments revealed severe entropy collapse. [Figure 5](https://www.notion.so/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56?pvs=21) shows entropy dropping below 0.2 and continuing to decrease throughout training. We also observed abnormally long response generation—another failure mode in RL training. Incorporating the TIS term reverses this trend, enabling the model to be trained in a stable and benign manner. 

In contrast, BF16 rollout experiments showed no severe entropy collapse. Nevertheless, the TIS term still increased entropy values. With a smaller distribution gap compared to INT8 rollouts, response lengths remained within reasonable bounds.



### On the Impact of Distribution Gap: A Case Study on KL Estimation

One unbiased kl estimator for $\rm{KL}(\textcolor{blue}{\pi_{\rm old}^{\rm fsdp}} \Vert \textcolor{blue}{\pi^{\rm fsdp}})$ is the $k_1$ [estimator](http://joschu.net/blog/kl-approx.html):  $\log \textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}(a) - \log \textcolor{blue}{\pi^{\rm fsdp}} (a)$ where $a\sim \textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}(a)$. However,  modern RL training framework generate rollouts from $\textcolor{red}{\pi_{\rm old}^{\rm vllm}}$ instead of $\textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}$, introducing bias to the kl estimation similar to the gradient estimation bias as discussed earlier.

Accordingly, we can use KL estimation as a case study to explore the impact of the mismatch between $\textcolor{red}{\pi_{\rm old}^{\rm vllm}}$ and $\textcolor{blue}{\pi_{\rm old}^{\rm fsdp}}$. Without any bias, the KL divergence is a non-negative by definition. However, the substantial distribution mismatch in INT8 rollouts causes the biased $k_1$ estimator to frequently yield negative values, as shown in [Figure 5](https://www.notion.so/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56?pvs=21). These negative KL estimates signal ill-conditioned training dynamics.

Meanwhile, when TIS is incorporated into RL training, the same $k_1$ estimator—while still affected by the underlying distribution mismatch—maintains positive values throughout most of the training process. This preservation of the expected sign indicates that TIS successfully restores well-conditioned training behavior.

### Biased Reward in Training Log

An interesting phenomenon of integrating TIS is that it may lead to worse reward logging while bringing better downstream performance. This is because the gap between $\textcolor{red}{\pi_{\text{sampler}}}$ and $\textcolor{blue}{\pi_{\text{learner}}}$ introduce biases in not only gradient estimation but also the reward estimation in logging. Particularly, the logged rewards is from the rollout policy, i.e.,  $E_{\textcolor{red}{\pi_{\text{sampler}}}}[{\rm R}]$ instead of $E_{\textcolor{blue}{\pi_{\text{learner}}}}[{\rm R}]$. Specifically, as shown in [Figure 6](https://www.notion.so/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56?pvs=21)(right two subplots), the logged reward metric shows that BF16-Rollout is better than BF16-Rollout w. TIS. However, if we look at the downstream performance of AIME accuracy, BF16-Rollout w. TIS significantly outperforms the vanilla BF16-Rollout.

### Intuitions of TIS’s Working Mechanism

While the exact mechanism of TIS remains an open question, we provide high-level intuitions on how TIS mitigating the distribution gap. 

Particularly, neglecting the bias on rollouts having  $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\rm old})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\rm old})} < 1$ could lead to entropy collapse through the following mechanism: For rollouts with negative advantages, policy gradients tend to reduce $\textcolor{blue}{\pi_{\mathrm{learner}}}$. When a large distribution gap exists after parameter updates, the reduction in $\textcolor{blue}{\pi_{\mathrm{learner}}}$ may not be reflected in $\textcolor{red}{\pi_{\mathrm{sampler}}}$. Consequently, policy gradient continues pointing towards further reductions in $\textcolor{blue}{\pi_{\mathrm{learner}}}$. Intuitively, such penalization may force the model to overly commit to a output distribution with a small entropy.

At the same time, TIS sticks to the un-truncated importance ratio for $\frac{\textcolor{blue}{\pi_{\mathrm{learner}}}(a,\;\theta_{\rm old})}{\textcolor{red}{\pi_{\mathrm{sampler}}}(a,\;\theta_{\rm old})} < 1$,  thereby eliminating bias for this subset of rollouts and breaks this mechanism. 

# Rollout-Training Mismatch Analysis

We conduct a series of controlled experiments to identify the factors that introduce or amplify the discrepancy between rollout generation and gradient computation. Specifically, we find that **differences** in **parallelism strategies** and **long response length** contribute to the mismatch, and the choice of the **sampler backends alone** have only marginal impacts. 

## Analysis Setup

**Model & Data.**  We experiment with two representative models — [DAPO-32B](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B) and [Polaris-7B](https://huggingface.co/POLARIS-Project/Polaris-7B-Preview) trained with the [DAPO](https://arxiv.org/pdf/2503.14476) and [POLARIS](https://www.notion.so/1dfa954ff7c38094923ec7772bf447a1?pvs=21) RL recipes. For evaluation, we use the first 512 prompts from [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset to evaluate the discrepancy between sampler and learner outputs.

**Metric.**  We measure response-level mismatches using two metrics:

- **Max Mismatch per response:**                           $\max_{a\,\in\, \text{response}} |p_{\tiny\text{sampler}}(a) - p_{\tiny\text{learner}}(a)|$
- **Mean Mismatch per response:**      $\frac{1}{|\text{response}|}\sum _{a\,\in\, \text{response}} |p_{\tiny\text{sampler}}(a) - p_{\tiny\text{learner}}(a)|$

These metrics allow us to capture both the worst-case token divergence and the average level of discrepancy within a response. We compute them for responses to the same prompts under different settings to isolate the impact of specific factors.

**Visualization.**  We present both metrics using the visualization format shown on the right. This is an illustrative example for interpreting the figures.



## Larger Parallelism Difference, Larger Max Gap

We observe that parallelism discrepancies between the sampler and learner contribute nontrivially to the Max Mismatch metric.

**Simplest Setting**

Using the DAPO-32B model, we begin with the simplest configuration: the sampler runs on vLLM with TP1 and the learner uses FSDP with SP1. Since the sampler and learner have the same parallelism setting, we refer to this as ***Same Parallelism*** and its distribution gap attributes to factors other than parallelism difference***.***

**Adding Tensor Parallelism**

To study the impact of TP difference, we change the sampler from TP1 to TP2 while keeping the learner at *SP1* (***Different TP***). As shown in Figure 7 Left, the number of responses with a high maximum mismatch (> 0.5) increases as parallelism differences grow. The ***Same Parallelism*** case yields only one such response, and ***Different TP*** increases this to two.

**Adding Sequence Parallelism**

To study the impact of [Ulysses Sequence Parallel](https://arxiv.org/abs/2309.14509) difference, we then change the learner from SP1 to SP8 (***Different TP* & *SP***). As shown in Figure 7 Middle, additional SP difference increases the number of high maximum mismatch from two to double digits. 

**Disentangling Parallelism and Sharding**

As shown in Figure 8 Left below, for a similar distributed world size (e.g., 8 devices), using Tensor Parallelism (TP8) in the learner leads to a smaller mismatch with the TP2 sampler than using Sequence Parallelism (SP8). We hypothesize this is because the implementation differences between a TP8 learner and a TP2 sampler are less pronounced than those between an SP8 learner and a TP2 sampler. This reinforces the finding that minimizing parallelism differences between the sampler and learner consistently reduces the gap.

We then measured the maximum mismatch when using the same tensor parallelism in learner & sampler, denoted as ***Same Parallelism (TP2)*** and ***Same Parallelism (TP4).*** Different from the simplest setting, these two configurations shard the model computation across multiple devices and thus are more scalable. As shown in Figure 8 Middle and Right, ***Same Parallelism (TP2)*** and ***Same Parallelism (TP4)*** have a small number responses with a high maximum mismatch (> 0.5)  , suggesting that sharding the model in the same way for both sampler and learner helps reduce the mismatch and should be preferable.

**Mean Mismatch and KL**

Desipte we observe consistent patterns for Max Mismatch, it is worth mentioning that we didn’t observe any significant difference on the Mean Mismatch/KL divergence of those configurations.

## Longer Response, Larger Max Gap

Our experiments consistently show that longer generation sequences lead to a larger **maximum** **mismatch**, while the **mean mismatch** is much less affected. We ablate the effect of sequence length using both  DAPO-32B and Polaris-7B models.

As shown in Figure 9, responses capped at 20K tokens exhibit a higher maximum mismatch than those capped at 4K. In contrast, the mean mismatch remains similar across both settings. This suggests that longer sequences provide more opportunities for a single, large probability divergence, even when the average per-token difference remains stable.

To verify that this effect is driven by sequence length rather than the total number of tokens generated, we conduct a control experiment comparing batches of single 20K-token responses with batches of five separate 4K-token responses to the same set of prompts.

As shown in Figure 10 Left, generating multiple shorter responses (5×4K) yields only a modest increase in maximum mismatch compared to a single 4K-token response. However, a continuous 20K-token response produces a **much larger mismatch** than either. This confirms that the discrepancy is exacerbated by the uninterrupted length of the sequence.

Interestingly, we observe the mismatch accumulates as generation progresses: the maximum mismatch within just the *first 4K tokens* of a 20K-token response often exceeds the maximum mismatch of an independent 4K-token response. This indicates that the internal states of sampler and learner diverge increasingly over long generation contexts.

## Altering Sampler Alone, Gap Still There

Finally, we investigate whether the choice of the sampler backend itself is a major contributor to the mismatch. We compared three configurations for the sampler: 1) vLLM; 2) SGLang; and 3) [SGLang](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) with [deterministic kernel](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) enabled.

The results show that the sampler backend alone does not have a decisive impact. With the DAPO-32B model, SGLang yields a smaller **mean mismatch**, whereas with the Polaris-7B model, vLLM performs better (i.e., vLLM has a smaller mean mismatch). Thus, no single sampler backend consistently dominates across different settings.

Notably, enabling deterministic sampling in SGLang, without aligning the training configuration, does not noticeably reduce the gap. This suggests the mismatch originates primarily from deeper implementation differences (e.g., parallelism or numerical precision), rather than from stochastic sampling alone.

## What’s More

There are other dimensions may influence the rollout–training mismatch, including **model type** (e.g., dense vs. MoE, base vs. post-trained), prompt **data characteristics** (e.g., difficulty, domain), **GPU hardware**, and the choice of training **backend**. For example, we find it relatively consistent that dense and MoE models of comparable scales (32B and 30B) exhibit different levels of mismatch, and base models have smaller rollout–training mismatch than their post-trained counterparts. We are continuously working towards a deeper understanding and better utilization of rollout-training mismatch for practical LLM post-training. Stay tuned!

# Discussion

We discuss the potential impact of our fix, Truncated Importance Sampling (TIS), on the RL of MoE architecture specifically. We also highlight TIS’s connection to recent works (e.g., [GSPO](https://arxiv.org/pdf/2507.18071), [GMPO](https://arxiv.org/pdf/2507.20673)) that aim to improve the importance-sampling ratio in policy update.

### The gap can be amplified in MoE RL

While our current experiments and analyses focus on dense models, we believe this distribution gap also exists in MoE RL and can be even more severe. There are two main reasons:

- **Dynamic Routing:** Different from dense models, MoE utilize a router to dynamically activate specific experts for This routing mechanism is inherently precision-sensitive; even slight numerical discrepancies can result in substantially different expert activations.
- **Specially Optimized Kernels:** MoE models are usually large-scale and modern inference engines (e.g., vLLM) have unique optimization for MoE models compared to dense models, which makes the backend numerical inconsistencies even larger.

Together, these characteristics can significantly amplify the distribution mismatch, making solutions like TIS particularly valuable in MoE RL.

### TIS is orthogonal and compatible with existing GxPOs

Recent works improve the stability of policy update by renovating the calculation of the importance-sampling ratio. For example, [GSPO](https://arxiv.org/pdf/2507.18071) calculates the ratio at the sequence level instead of the token level, while [GMPO](https://arxiv.org/pdf/2507.20673) computes the geometric mean instead of the arithmetic mean.

Orthogonal to these works, our TIS fix addresses the distribution mismatch problem rooted in the system level, which is brought by the different compute kernels used in rollout generation and model training. Such a problem widely exists in RL training frameworks that adopt a hybrid computation design. Thus, our fix can be applied irrespective of the specific RL algorithms used.

# Citation

```jsx
@misc{yao2025offpolicy,
  title = {Your Efficient RL Framework Secretly Brings You Off-Policy RL Training},
  url = {https://fengyao.notion.site/off-policy-rl},
  author = {Yao, Feng and Liu, Liyuan and Zhang, Dinghuai and Dong, Chengyu and Shang, Jingbo and Gao, Jianfeng},
  journal = {Feng Yao's Notion},
  year = {2025},
  month = aug,
}
```

We're excited to share our early results and welcome feedback from the community as we continue to refine and expand Flash-RL’s capabilities. If you have any questions or feedback, please feel free to contact us at [`fengyao@ucsd.edu`](mailto:fengyao@ucsd.edu) and [`llychinalz@gmail.com`](mailto:llychinalz@gmail.com)