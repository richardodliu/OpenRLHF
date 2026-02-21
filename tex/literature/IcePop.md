### Small Leak Can Sink a Great Ship—Boost RL Training on MoE with 𝑰𝒄𝒆𝑷𝒐𝒑!

**TL;DR**

Recent work[[1\]](https://www.notion.so/Small-Leak-Can-Sink-a-Great-Ship-Boost-RL-Training-on-MoE-with-271c8705a03280378b98d7f8da794ed0?pvs=21) has highlighted a mismatch issue between the model training and rollout generation in the current reinforcement learning (RL) training framework. We observed that this problem can be **exacerbated** ☹️ in the **Mixture-of-Experts (MoE)**  architectures. Especially when the model tends to generate long responses, the discrepancy becomes **further amplified** 😱. Although prior work[[1\]](https://www.notion.so/Small-Leak-Can-Sink-a-Great-Ship-Boost-RL-Training-on-MoE-with-271c8705a03280378b98d7f8da794ed0?pvs=21) proposed to mitigate it by introducing importance-sampling correction, we argue that such a technique may hit a performance plateau on the benchmark as training proceeds.  Instead, we propose a simple yet effective method— :icepop:𝑰𝒄𝒆𝑷𝒐𝒑 🤩 that delivers both stable training and superior downstream performance with RL even on a strong SFT-ed MoE model.

# The Mismatch Issue on MoE

The mismatch issue refers to the probability discrepancy between the training backend and inference engine in the current RL training framework, which inevitably turns on-policy training to off-policy[[1\]](https://www.notion.so/Small-Leak-Can-Sink-a-Great-Ship-Boost-RL-Training-on-MoE-with-271c8705a03280378b98d7f8da794ed0?pvs=21). We observed that such an implementation gap becomes more significant during RL training on MoE models. Unlike dense models, to achieve higher parameter efficiency, MoE architectures adopt a routing mechanism, which selects only a handful of top-ranked “experts” for each token during training and inference. However, we found that such structural design exacerbates the mismatch issue during on-policy RL training, preventing the MoE models from fully unlocking the potential of reinforcement learning.

According to the policy gradient equation below, we can see that another mismatch problem arises, i.e., $\textcolor{red}{\theta_{\rm infer}}$ and $\textcolor{blue}{\theta_{\rm train}}$. In MoE models, the router function $\texttt{TopK}(\cdot)$ dynamically activates a subset of  “experts” (i.e., networks) for each input token. Ideally, for a fixed input, the output of $\texttt{TopK}(\cdot)$ is expected to be identical regardless of the engine on which the policy model is deployed. Nevertheless, when there is a substantial gap between $\textcolor{red}{\theta_{\rm infer}}$ and $\textcolor{blue}{\theta_{\rm train}}$, it will inevitably lead to a greater divergence between  $\textcolor{red}{\pi_{\rm infer}}$ and $\textcolor{blue}{\pi_{\rm train}}$.

$$ \small{\begin{equation}\theta \leftarrow \theta + \mu \cdot \mathbb{E}*{a\sim \textcolor{red}{\pi*{{\rm{infer}}}}(\textcolor{red}{\theta_{\rm infer}} ), \ \textcolor{red}{\theta_{\rm infer}} \sim \mathtt{TopK}*{\rm infer}(a)}\left[ R(a) \cdot \nabla*{\theta}\log \textcolor{blue}{\pi_{\rm{train}}}(a;\textcolor{blue}{\theta_{\rm train}});\textcolor{blue}{\theta_{\rm train}} \!\sim \!\texttt{TopK}_{\rm train}(a) \right]\end{equation}} $$

For MoE models, we identified two major causes of the training-inference discrepancy problem:

- **The selected experts may vary between training and inference stages.**

  > Our prior analysis indicates that, even in the first MoE layer, a tiny fraction of tokens already activate different experts in the training backend and inference engine.

  For example, when the probabilities of selecting the Top-k and Top-(k+1) experts are very close, even a slight precision discrepancy can cause different experts to be chosen during training and during inference, resulting in a large discrepancy in the computed probabilities.

- **The mismatch becomes pronounced as more routing networks are stacked.**

  > We further noticed that, as the MoE layers deepen, the proportion of tokens that invoke the same experts in both training backend and inference engine rapidly drop by around 10%.

  At each layer, the routing network determines which experts to activate. In a deep MoE model, it selects multiple experts per layer, thus even small discrepancies in each call to $\texttt{TopK}(\cdot)$ can accumulate and become increasingly amplified as the depth grows.

## What Effects Will It Bring to MoE RL?

 🧵 The probability discrepancy becomes magnified, especially for long sequences.

At the very beginning of training, we find that the noticeable probability discrepancies are already evident at certain token positions. Due to the autoregressive nature of prediction, tokens that occur in later positions are more susceptible to the accumulation of discrepancies, resulting in a wider spread of variation.

As training progresses, the problem intensifies: the probability gap for the same token between training and inference engines continues to increase across token positions, even affecting those preceding tokens in long sequences and destabilizing the optimization process.

 📉 The mismatch issue quickly causes crashes during on-policy MoE RL training.

In on-policy RL training, compared to dense models, we observed that MoE models tasked with generating long sequences are more vulnerable to such mismatch problem, often leading training to crashes.

For instance, the figures above show that the discrepancy gradually increases after step 150, and the training essentially fails once the discrepancy exceeds 0.05.  As the implementations differ, probability discrepancies could become larger with the compounding effect.

- **Lemma** (Compounding Probability Discrepancy)

  Let $\pi_{\text{infer}}(\cdot;\theta)$ and $\pi_{\text{train}}(\cdot;\theta)$ be the inference and training policies. Define the probability discrepancy at step $t$ as

  $$ \delta_t \;=\; D_{\mathrm{KL}}\!\big(\pi_{\text{infer}}(\cdot;\theta_t)\,\|\,\pi_{\text{train}}(\cdot;\theta_t)\big). $$

  which measures how much the inference engine’s distribution deviates from the training engine’s distribution.

  During RL training with mismatched engines, the parameter update is

  $$ \theta_{t+1} \;=\; \theta_t \;+\; \mu\,g_t, \qquad g_t \;=\; \mathbb{E}*{a\sim \pi*{\text{infer}}(\theta_t)}\!\big[R(a)\,\nabla_\theta \log \pi_{\text{train}}(a;\theta_t)\big]. $$

  The on-policy update is $g_t^\star = \mathbb{E}*{a\sim \pi{*\text{train}}(\theta_t)}[R(a)\,\nabla_\theta \log \pi_{\text{train}}(a;\theta_t)]$  and the bias is $b_t = g_t - g_t^\star$. Assume the following local conditions hold for $\theta$.

  1. **Smooth discrepancy.**

     $\delta(\theta)$ is $L$-smooth with $\big|\,\delta(\theta+\Delta)-\delta(\theta)\,\big| \;\le\; L\,\|\Delta\| \;+\; c_0 \|\Delta\|^2$ where $c_0$ is curvature constant.

     This means that small parameter updates cause only small changes in discrepancy.

  2. **Bias scales with discrepancy.**

     $\|b_t\| \;\ge\; c_1\,\delta_t, ~~\big\langle \nabla_\theta \delta(\theta_t),\, b_t \big\rangle \;\ge\; c_2\,\delta_t$, where $c_1$ is bias magnitude coefficient and $c_2$ bias alignment coefficient.

     The larger the mismatch, the more the bias pushes in the direction that worsens it.

  3. **Bounded on-policy drift.**

     There exists $M\ge 0$ such that $\big|\langle \nabla_\theta \delta(\theta_t),\, g_t^\star \rangle\big| \le M$

     On-policy training alone does not cause runaway divergence; instability arises mainly from the bias.

# Unleash MoE RL with *IcePop*: Discard All Noisy Gradient Updates!

To address the aforementioned issues, we propose a simple yet effective technique,  𝑰𝒄𝒆𝑷𝒐𝒑 🤩. We apply double-sided masking to mitigate the harmful compounding effects of probability discrepancies, preserving only healthy gradient updates.

- **Double-sided clipping**: not only clipping tokens where *training probability ≫ inference probability*, but also when *training probability ≪ inference probability*.
- **Masking**: tokens *with excessive discrepancy* are removed from gradient updating.

$$ \small{\begin{align*}\mathcal{J}*{{\text{IcePop}}}(\theta) &= \mathbb{E}*{x \sim \mathcal{D}, \{y_i\}*{i=1}^G \sim \pi*{\textcolor{red}{\text{infer}}}(\cdot \mid x; \theta_{\rm old})} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \Big[\mathcal{M}\Bigl(\frac{\pi_{\textcolor{blue}{\text{train}}}(y_{i,t} \mid x, y_{i,<t};\theta_{\text{old}})}{\pi_{\textcolor{red}{\text{infer}}}(y_{i,t} \mid x, y_{i,<t}; \theta_{\mathrm{old}})}; \alpha, \beta\Bigr) \right. \\ &\left. \qquad \qquad \qquad \qquad \quad \qquad \cdot \min \left( r_{i,t}\widehat{A}*{i,t}, \text{clip} \left( r*{i,t}, 1 - \varepsilon, 1 + \varepsilon \right) \widehat{A}_{i,t} \right) \right]\Bigg] &\end{align*}} $$

where $r_{i,t} = \frac{\pi_{\textcolor{blue}{\text{train}}}(y_{i,t} \mid x, y_{i,<t}; \ \theta)}{\pi_{\textcolor{blue}{\text{train}}}(y_{i,t} \mid x, y_{i,<t}; \ \theta_{\text{old}})}$, with the masking function below,

$$ \begin{equation} \mathcal{M}(k) =\begin{cases} k & \text{if \ } k \in [\alpha, \beta] \\ 0 & \text{otherwise}\end{cases} \end{equation} $$

and two hyperparameters $\alpha$,  $\beta$ to control the lower and upper limits.

The gradient of IcePop:

$$ \small{\nabla_\theta \mathcal{J}*{\text{IcePop}}(\theta) \sim \small{\begin{equation}\mathbb{E}*{a \sim \textcolor{red}{\pi_{\text{infer}}}(\theta_{\text{old}})} \Bigg[\mathcal{M}\Bigg(\frac{\textcolor{blue}{\pi_{\text{train}}}(a;\theta_{\text{old}})}{\textcolor{red}{\pi_{\text{infer}}}(a;\theta_{\text{old}})}\Bigg ) \cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{train}}}(a;\theta) \cdot \hat{A} \cdot r(a)\Bigg)\Bigg].\end{equation}}} $$

The difference between our work and TIS[[1\]](https://www.notion.so/Small-Leak-Can-Sink-a-Great-Ship-Boost-RL-Training-on-MoE-with-271c8705a03280378b98d7f8da794ed0?pvs=21):

- When $\dfrac{\textcolor{blue}{\pi_{\text{train}}}(a;\theta_{\text{old}})}{\textcolor{red}{\pi_{\text{infer}}}(a;\theta_{\text{old}})} < \alpha$, the $\textcolor{blue}{\pi_{\text{train}}}$ tends to assign a smaller value to the action, inversely, $\textcolor{red}{\pi_{\text{infer}}}$ outputs a higher probability, when the ratio is smaller enough, indicating that there is huge divergence between training and inference engines. TIS multiplies a small coefficient to mitigate the noisy gradient updates, however, as training proceeds, we found the such small disturbance can be gradually amplified, and eventually causes plateau in benchmark performance.

- When $\dfrac{\textcolor{blue}{\pi_{\text{train}}}(a;\theta_{\text{old}})}{\textcolor{red}{\pi_{\text{infer}}}(a;\theta_{\text{old}})} > \beta$,  TIS continues to update the policy gradient by multiplying a modest coefficient. For IcePop, the gradient is zero, which means that we forgo all noisy updates and only keep those healthy policy gradients.

- 👂🏼 Wanna hear the story behind the name of **IcePop**?

  😄 We came up with the name while enjoying ice pops!

  - Much like an ice pop cools down overheating, the algorithm “**cools**” unstable on-policy training by clipping extreme probability ratios on both sides and masking tokens with excessive discrepancy.
  - This selective correction **pops out** unstable contributions while preserving efficient updates, thereby stabilizing training without slowing inference.

# Experiments

In this article, we compare three settings on the [Ring-mini-2.0](https://huggingface.co/inclusionAI/Ring-mini-2.0), which is a MoE model developed by [InclusionAI](https://inclusionai.github.io/).  It owns 16.8B total parameters and 0.75B activated parameters:  **(1) IcePop**, **(2) [TIS](https://fengyao.notion.site/off-policy-rl),** and **(3) Baseline**—Vanilla GRPO without the KL-term, which consistently failed and collapsed across repeated runs. We collected challenging reasoning problems as the training dataset. With IcePop, we found that the instability of on-policy training can be effectively resolved, and even achieve better performance than TIS.

 🔥

On the downstream task, **IcePop** outperforms both **TIS** and the **baseline**.

- **Model Evaluation**: On the challenging benchmark AIME25, **IcePop** consistently outperforms **TIS** with a large gain along the training process, and finally improves the base score (63%) by over **14%**, and expands the performance gap with **TIS** by relative **6%**.

# More Analysis

- **Probability Discrepancy**

Without addressing the mismatch issues, the probability difference grows rapidly, as shown in the baseline setting. In contrast, both TIS and IcePop keep the KL divergence of training-inference probability within a reasonable range. Although the maximum probability difference rises for all three methods as training proceeds, the discrepancy of IcePop remains relatively low and even decreases within 400 steps. We also notice that TIS consistently shows larger extreme discrepancies and faster growth than ours, probably due to including the noisy policy updates during training.

- **Training Stability**

We believe that a stable training process serves as a solid foundation and sufficient space for showcasing the power of reinforcement learning. It is worth noting that both IcePop and TIS mitigate the instability of RL training within 600 gradient steps, avoiding rapid training crashes occurring in the baseline setting.

- **Exploration Space**

We observed that the log probabilities of IcePop consistently maintain relatively lower values than those of TIS, which implicitly indicates that our method avoids overconfident predictions, thus ensuring a larger scope for exploring space, where low-probability tokens are more likely to be chosen, eventually increasing the diversity of responses.

- **Ill-conditioned Tokens**

In our experiments, we found that the clipping ratio from our masking mechanism stays around 1–2‰ of training tokens. As training progresses, the clipping ratio rises sharply, suggesting that increasingly subtle but harmful gradient updates occur and necessitate a higher clipping ratio. We also conducted a detailed analysis of the clipped tokens. The following right figure shows that, compared to all tokens, clipped tokens exhibit higher entropy, indicating that the clipped tokens play an important role in training.

# Conclusion and Future Work

- We provide preliminary analysis on the training-inference probability mismatch issues on MoE. The instability of on-policy RL training could arise from a growing probability discrepancy between training and inference engines. **IcePop** addresses this problem by correcting the mismatch at the loss level.
- An important direction is to formally characterize the *collapse boundary*, defined as the critical probability discrepancy beyond which on-policy training becomes unstable, and to study how this boundary scales with model size, batch configuration, and engine mismatch. </aside>

🫡 *Thank you for reading. We are continuing to refine this work and welcome your feedback.*

# Citation

```json
@misc{IcePop2025,
  title = {Small Leak Can Sink a Great Ship--Boost RL Training on MoE with IcePop!},
  url = {<https://ringtech.notion.site/icepop>},
  author = {Xin Zhao and Yongkang Liu and Kuan Xu and Jia Guo and Zihao Wang and Yan Sun and Xinyu Kong and Qianggang Cao and Liang Jiang and Zujie Wen and Zhiqiang Zhang and Jun Zhou},
  year = {2025},
  month = {Sep},
}
```

## Reference

[1] Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang and Jianfeng Gao. [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl). *Notion Blog*. 2025.