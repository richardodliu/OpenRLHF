# Theory of On-Policy Distillation

> **Authors:   [Jiacai Liu](https://www.jiacailiu.cn/)*  Zhuo Jiang*  [Yuqian Fu](https://scholar.google.com/citations?user=oRcXbE0AAAAJ&hl=en)**
> 

***Co-First Authors. **Work done at ByteDance Seed. First published at February 15, 2026. 

# 1. On-Policy Distillation

## 1.1 Objective

As large language models continue to expand in size and computational requirements, the need for efficient training methods that produce capable smaller models or merge multiple expert models into one has become increasingly critical. On-policy distillation (OPD) represents a powerful approach to post-training that combines the advantages of on-policy training with dense reward signals, addressing fundamental limitations in both traditional knowledge distillation and reinforcement learning methods for training language models. We begin with the optimization objective of OPD. Suppose we have training prompts that are sampled from a certain distribution $\mathcal{D}$, and for each prompt $x$ we can access to a corresponding **teacher model** $\pi_{T(x)}$. Then OPD minimize the reverse KL divergence between current parameterized model $\pi_\theta$ (also known as **student model**) and the teach model, i.e.,

$$
\begin{align}\text{OPD\,:}\quad \underset{\theta}{\min}\,\,\mathcal{L} \left(\theta\right) =\mathbb{E} _{x\sim \mathcal{D}}\left[ \mathrm{KL}\left( \pi _{\theta}\left( \cdot |x \right) ,\pi _{T\left( x \right)}\left( \cdot |x \right) \right) \right]. \end{align}
$$

By the definition of KL divergence, one can show that 

$$
\begin{align} \mathrm{KL}\left( \pi _{\theta}\left( \cdot |x \right) ,\pi _{T\left( x \right)}\left( \cdot |x \right) \right) =-\mathbb{E}_{y\sim\pi_{\theta}(\cdot|x)}\left[\sum_{t=0}^{\left|y\right|-1}\log\frac{\pi_{T(x)}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}  \right],\end{align}
$$

where $s_t:=(x,y_0,...,y_{t-1}),a_t:=y_t$. Plugging it into (1) yields that 

$$
\small{\begin{align}\text{OPD\,:}\quad \underset{\theta}{\max}\,\,\mathcal{V} \left(\theta \right) &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E}_{y\sim\pi_{\theta}(\cdot|x)}\left[\sum_{t=0}^{\left|y\right|-1}\log\frac{\pi_{T(x)}\left(a_t|s_t\right)}{\pi_{\theta}\left(a_t|s_t\right)}  \right]. \\ &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E}_{y\sim\pi_{\theta}(\cdot|x)}\left[\sum_{t=0}^{\left|y\right|-1}\log \pi_{T(x)}\left(a_t|s_t\right)+\mathcal{H}\left(\pi_{\theta}\left(\cdot|s_t\right)\right)  \right],\end{align}}
$$

which implies that OPD is a special entropy-regularized finite horizon RL problem with the immediate reward $r(s_t,a_t)$  given by the teacher’s log probability $\log \pi_{T(x)}(a_t|s_t)$. 

## 1.2 Policy Gradient of OPD

Since OPD is a special RL problem, we can use policy gradient methods to update the student model. We directly present the policy gradient of OPD in the following along with the proof.

---

**Theorem 1 (Policy Gradient of OPD).** *For arbitrary parameter $\theta$, the policy gradient of OPD is*

$$
\begin{align}\nabla _{\theta}\mathcal{V} \left( \theta \right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\hat{A}_{\theta}\left( s_t,a_t \right)  \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right],\end{align}
$$

*where $s_t:=(x,y_{<t}),a_t:=y_t,$ and the advantage $\hat{A}_{\theta}\left( s_t,a_t \right)$ can be one of the following:*

*(i) $\sum_{t^{\prime}=0}^{\left| y \right|-1}{\log \frac{\pi _{T\left( x \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}}=\frac{\log \pi _{T\left( x \right)}\left( y|x \right)}{\log \pi _{\theta}\left( y|x \right)}$ : the log ratio of full trajectory.* 

*(ii) $\sum_{t^{\prime}=\textcolor{red}{t}}^{\left| y \right|-1}{\log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}}$:  log-ratio-to-go.*

*(iii) $\sum_{t^{\prime}=\textcolor{red}{t}}^{\left| y \right|-1}{\log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} -b(s_{\textcolor{red}{t}})$: log-ratio-to-go with baseline.*

*(iv) $\log \frac{\pi _{T(x)}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}{\pi _{\theta}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}-\sum_{t^{\prime}=\textcolor{red}{t}+1}^{\left| y \right|-1}{\mathrm{KL}\left( \pi _{\theta}\left( \cdot |s_{t^{\prime}} \right) ,\pi _{T\left( x \right)}\left( \cdot |s_{t^{\prime}} \right) \right)}$.*

*(v) $\log \frac{\pi _{T(x)}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}{\pi _{\theta}\left( a_{\textcolor{red}{t}}|s_{\textcolor{red}{t}} \right)}-\sum_{t^{\prime}=\textcolor{red}{t}+1}^{\left| y \right|-1}{\mathrm{KL}\left( \pi _{\theta}\left( \cdot |s_{t^{\prime}} \right) ,\pi _{T\left( x \right)}\left( \cdot |s_{t^{\prime}} \right) \right)} -b(s_{\textcolor{red}{t}})$: baselined version of (iv).*

- **Proof for (i)**
    
    *A direct computation yields that*
    
    $$
    \small{\begin{align*}\nabla _{\theta}\mathcal{V} \left( \theta \right) &=\nabla _{\theta}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \log \frac{\pi _{T\left( x \right)}\left( y|x \right)}{\pi _{\theta}\left( y|x \right)} \right] \\\,\,         &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \log \frac{\pi _{T\left( x \right)}\left( y|x \right)}{\pi _{\theta}\left( y|x \right)}\nabla _{\theta}\log \pi _{\theta}\left( y|x \right) +\nabla _{\theta}\left( \log \frac{\pi _{T\left( x \right)}\left( y|x \right)}{\pi _{\theta}\left( y|x \right)} \right) \right] \\\,\,         &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \log \frac{\pi _{T\left( x \right)}\left( y|x \right)}{\pi _{\theta}\left( y|x \right)}\nabla _{\theta}\log \pi _{\theta}\left( y|x \right) \right] \\\,\,         &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \left( \sum_{t^{\prime}=0}^{\left| y \right|-1}{\log \frac{\pi _{T(x)}\left( y_{t^{\prime}}|x,y_{<t^{\prime}} \right)}{\pi _{\theta}\left( y_{t^{\prime}}|x,y_{<t^{\prime}} \right)}} \right) \sum_{t=0}^{\left| y \right|-1}{\nabla _{\theta}\log \pi _{\theta}\left( y_t|x,y_{<t} \right)} \right] \\\,\,         &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=0}^{\left| y \right|-1}{\log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right].\end{align*}}
    $$
    
- **Proof for (ii)**
    
    *Consider two fixed time indexes  $0\le t^{\prime}<t\le \left| y \right|-1$, one can show that*
    
    $$
    \small{\begin{align*}\mathbb{E} \left[ \log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] &=\mathbb{E} \left[ \mathbb{E} \left[ \log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |\left( s_{t^{\prime}},a_{t^{\prime}} \right) \right] \right] \\\,\,                                               &=\mathbb{E} \left[ \log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \mathbb{E} \left[ \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |\left( s_{t^{\prime}},a_{t^{\prime}} \right) \right] \right] \\\,\,                                               &=\mathbb{E} \left[ \log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \mathbb{E} \left[ \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] \right] \\\,\,                                               &=\mathbb{E} \left[ \log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot 0 \right] \\\,\,                                               &=0,\end{align*}}
    $$
    
    *where the expectation $\mathbb{E}$ is taken over $x\sim \mathcal{D},y\sim \pi_{\theta}.$ This causality means that the policy gradient of OPD at each visited token $a_t$, i.e. , $\nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)$ , does **not** need to consider the impact on the KL divergence of **previously** encountered tokens—it only needs to consider the impact on the KL divergence of tokens that may be encountered **subsequently**. Thus, combining it with the proof of (i), one has*
    
    $$
    \small{\begin{align*}\nabla _{\theta}\mathcal{V} \left( \theta \right)   &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=0}^{\left| y \right|-1}{\log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right]\\\ &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=\textcolor{red}{t}}^{\left| y \right|-1}{\log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right].   \end{align*}}
    $$
    
- **Proof for (iv)**
    
     *For arbitrary two fixed time indexes  $0\le t < t^{\prime}\le \left| y \right|-1$, a direct computation yields that*
    
    $$
    \small{\begin{align*}\mathbb{E} \left[ \log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] &=\mathbb{E} \left[ \mathbb{E} \left[ \log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}\cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |\left( s_t,a_t \right) \right] \right] \\\,\,                                               &=\mathbb{E} \left[ \mathbb{E} \left[ \log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}|\left( s_t,a_t \right) \right] \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] \\\,\,                                               &=\mathbb{E} \left[ \mathbb{E} \left[ \log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}|s_{t^{\prime}} \right] \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] \\\,\,                                               &=\mathbb{E} \left[-\mathrm{KL}\left( \pi _{\theta}\left( \cdot |s_{t^{\prime}} \right) ,\pi _{T\left( x \right)}\left( \cdot |s_{t^{\prime}} \right) \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right],\end{align*}}
    $$
    
    *where the expectation $\mathbb{E}$ is taken over $x\sim \mathcal{D},y\sim \pi_{\theta}.$ Combining the last line with the proof for (ii), we have* 
    
    $$
    \small{\begin{align*}\nabla _{\theta}\mathcal{V} \left( \theta \right)   &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=\textcolor{red}{t}}^{\left| y \right|-1}{\log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right]\\\ &=
    \mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \log \frac{\pi _{T(x)}\left( a_t|s_t \right)}{\pi _{\theta}\left( a_t|s_t \right)}-\sum_{t^{\prime}=t+1}^{\left| y \right|-1}{\mathrm{KL}\left( \pi _{\theta}\left( \cdot |s_{t^{\prime}} \right) ,\pi _{T\left( x \right)}\left( \cdot |s_{t^{\prime}} \right) \right)} \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right] 
    .   \end{align*}}
    $$
    
- **Proof for (iii) & (iv)**
    
    *For any function that only depends on current state $s_t$, i.e. , $b(s_t)$, it’s easy to show that*
    
    $$
    \begin{align*}\mathbb{E} \left[ b\left( s_t \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] &=\mathbb{E} \left[ \mathbb{E} \left[ b\left( s_t \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |s_t \right] \right] \\\,\,                               &=\mathbb{E} \left[ b\left( s_t \right) \cdot \mathbb{E} \left[ \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |s_t \right] \right] \\\,\,                               &=\mathbb{E} \left[ b\left( s_t \right) \cdot \mathbb{E} \left[ \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] \right] \\\,\,                               &=\mathbb{E} \left[ b\left( s_t \right) \cdot 0 \right] \\\,\,                               &=0,\end{align*}
    $$
    
    *where the expectation $\mathbb{E}$ is taken over $x\sim \mathcal{D},y\sim \pi_{\theta}.$ This means that adding or subtracting a baseline term $b(s_t)$ does not change the policy gradient of OPD at each state-action pair $(s_t,a_t)$  in expectation, which justifies using (iii) and (iv) as $\hat{A}_{\theta}\left( s_t,a_t \right)$.* 
    

---

At each training step, one can sample multiple trajectories $\mathcal{T}=\{x_i,y_i\}$ from $\mathcal{D} \times \mathcal{\pi_\theta}$ and then compute the per-token advantage $\hat{A}_\theta\left(s_t,a_t\right)$ using any of the type (i) through (v) for each trajectory $(x,y)\in\mathcal{T}$. The student model can then be updated by any policy gradient method, such as PPO or GRPO,  using the calculated advantages. For example, one can use REINFORCE [REF] algorithm, whose policy loss is 

$$
\begin{align} \eta(\theta)=-\frac{1}{\left|\mathcal{T}\right|}\sum_{\left(x,y\right)\in\mathcal{T}}\sum_{t=0}^{\left|y\right|-1}{\hat{A}_{\text{sg}\left(\theta\right)}\left(s_t,a_t\right)\cdot \log\pi_\theta\left(a_t|s_t\right)}.\end{align}
$$

Here $\text{sg}(\cdot)$ is the stop gradient operator. The parameter $\theta$ can be updated by doing gradient descent w.r.t $\eta(\theta)$ which provides unbiased gradient since $\mathbb{E}_{\mathcal{T}}\left[\nabla_\theta \eta(\theta)\right] = - \nabla_\theta\mathcal{V}(\theta)$.

## 1.3 A Simple, Practical, Yet Biased, Method : Immediate Log Ratio as Per-Token Advantage

Theorem 1 indicates that, to produce an unbiased gradient, the per-token advantage should consider the future log ratios or KL divergence. However, A popular surrogate loss that has emerged in some studies [REF] involves using only the immediate log ratio as the per-token advantage, specifically $\hat{A}_\theta(s_t,a_t)=\log\frac{\pi_{T(x)}(a_t|s_t)}{\pi_\theta(a_t|s_t)}$, within the policy loss. To illustrate, consider again the REINFORCE algorithm, the policy loss then becomes

$$
\begin{align} \hat{\eta}(\theta)=-\frac{1}{\left|\mathcal{T}\right|}\sum_{\left(x,y\right)\in\mathcal{T}}\sum_{t=0}^{\left|y\right|-1}{\log\frac{\pi_{T(x)}(a_t|s_t)}{\pi_{\text{sg}\left(\theta\right)}(a_t|s_t)}\cdot \log\pi_\theta\left(a_t|s_t\right)}.\end{align}
$$

It’s easy to see that, by Theorem 1,

$$
\begin{align*}\mathbb{E}_{\mathcal{T}}\left[\nabla_\theta \hat{\eta}(\theta)\right]&=-\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\log \frac{\pi_{T(x)}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right]  \\ &\ne - \nabla_\theta\mathcal{V}(\theta).\end{align*}
$$

To take a deeper analysis into this bias, we introduce the concept of **state occupancy** which is also commonly used in RL literatures [REF]. For arbitrary policy $\pi,$ we define the state occupancy of $\pi$ as 

$$
\begin{align}d_{\pi}\left( s \right) :=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}} \right].\end{align}
$$

We also define $S$ as the **state space** which is the set of all possible values of  $(x,y_{<t})$.  Following lemma shows that we can exchange between $\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y-1 \right|}{\cdot} \right]$  and $\mathbb{E} _{s\sim d_{\pi}}\left[ \cdot \right]$.

---

**Lemma 1 (Expectation Exchange Under State Occupancy).** *For arbitrary policy $\pi$ and real valued function $f:\mathcal{S}\rightarrow \mathbb{R}$, one has*

$$
\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{f\left(x,y_{<t} \right)} \right] =\mathbb{E} _{s\sim d_{\pi}}\left[ f\left( s \right) \right]
$$

- **Proof**
    
    *Note that the indicator function satisfies*
    
    $$
    \sum_{s\in \mathcal{S}}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}}=1
    $$
    
    *Then a direct computation yields that*
    
    $$
    \begin{align*}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{f\left( x,y_{<t} \right)} \right] &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\underset{=1}{\underbrace{\left( \sum_{s\in \mathcal{S}}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}} \right) }}f\left( x,y_{<t} \right)} \right] \\\,\,                                     &=\sum_{s\in \mathcal{S}}{\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\} f\left( x,y_{<t} \right)} \right]}\\\,\,                                     &=\sum_{s\in \mathcal{S}}{\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\} f\left( s \right)} \right]}\\\,\,                                     &=\sum_{s\in \mathcal{S}}{\underset{=d_{\pi}\left( s \right)}{\underbrace{\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}} \right] }}\cdot f\left( s \right)}\\\,\,                                     &=\mathbb{E} _{s\sim d_{\pi}}\left[ f\left( s \right) \right] \end{align*}
    $$
    

---

By combining Lemma 1 and (3), one can rewrite the objective function of OPD as

$$
\begin{align}
\mathcal{V} \left( \theta \right) =\mathbb{E} _{s\sim d_{\theta}}\underset{=-\mathrm{KL}\left( \pi _{\theta}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right)}{\underbrace{\left[ \mathbb{E} _{a\sim \pi _{\theta}\left( \cdot |s \right)}\left[ \log \frac{\pi _{T(s)}\left( a|s \right)}{\pi _{\theta}\left( a|s \right)} \right] \right] }}
.\end{align}
$$

where $d_\theta$ is short for $d_{\pi_\theta}$ and $T(s)$ is the corresponding teacher at state $s.$ According to the chain rule,  the gradient of $\mathcal{V}(\theta)$  satisfies

$$
\begin{align*}
	\nabla _{\theta}\mathcal{V} \left( \theta \right) &=\underset{:=g_1(\theta)}{\underbrace{\sum_{s\in \mathcal{S}}{d_{\theta}\left( s \right) \cdot \nabla _{\theta}\mathbb{E} _{a\sim \pi _{\theta}\left( \cdot |s \right)}\left[ \log \frac{\pi _{T\left( s \right)}\left( a|s \right)}{\pi _{\theta}\left( a|s \right)} \right]}}}\\
	&+\underset{:=g_2(\theta)}{\underbrace{\sum_{s\in \mathcal{S}}{\nabla _{\theta}d_{\theta}\left( s \right) \cdot \mathbb{E} _{a\sim \pi _{\theta}\left( \cdot |s \right)}\left[ \log \frac{\pi _{T\left( s \right)}\left( a|s \right)}{\pi _{\theta}\left( a|s \right)} \right]}}}.\\

\end{align*}
$$

By a direct computation, we can show that,  the first partial derivative ****of $\nabla_\theta\mathcal{V}(\theta),$ i.e., $g_1(\theta),$  satisfies

$$
\begin{align*}g_1\left(\theta\right)&=\sum_{s\in \mathcal{S}}{d_{\theta}\left( s \right) \cdot \nabla _{\theta}\mathbb{E} _{a\sim \pi _{\theta}\left( \cdot |s \right)}\left[ \log \frac{\pi _{T\left( s \right)}\left( a|s \right)}{\pi _{\theta}\left( a|s \right)} \right]}\\&=\mathbb{E}_{s\sim d_\theta}\mathbb{E}_{a\sim\pi_\theta(\cdot|s)}\left[\log\frac{\pi_{T(s)}(a|s)}{\pi_\theta(a|s)}\nabla_\theta \log \pi_\theta (a|s)\right]\\&\overset{(a)}{=}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\log \frac{\pi_{T(x)}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right] \\ & \overset{(b)}{=} - \mathbb{E}_{\mathcal{T}}\left[\nabla_\theta \hat{\eta}(\theta)\right],\end{align*}
$$

where (a) is due to Lemma 1 and (b) is due to (7).  **Intuitively,  performing gradient ascent with $g_1(\theta)$  minimizes the reverse KL divergence, i.e., $\mathrm{KL}\left( \pi _{\theta}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right),$  for those states $s$ that the student finds itself. This update effectively uses the immediate log ratio as the per-token advantage in the policy gradient, i.e., $\hat{A}_\theta(s_t,a_t)=\log \frac{\pi_{T(x)}(a_t|s_t)}{\pi_\theta(a_t|s_t)},$ introduced in (5). However, such minimization does not address how states are generated — which is the source of bias.**  The optimization of state occupancy is instead handled by the second partial derivative, denoted as $g_2(\theta).$  The analytic form of $\sum_{s} f(s)\nabla_\theta d_\theta(s)$ for arbitrary real valued function $f$ is provided in Theorem 2.

---

**Theorem 2 (Gradient of Parameterized State Occupancy).** *For arbitrary parameter $\theta$ and real valued function $f:\mathcal{S}\rightarrow \mathbb{R}$:* 

$$
\begin{align*}
\nabla _{\theta}\mathbb{E} _{s\sim d_{\theta}}\left[ f\left( s \right) \right] =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |s \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=t+1}^{\left| y \right|-1}{f\left( s_{t^{\prime}} \right)} \right) \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right].
\end{align*}
$$

*where $s_t:=(x,y_{<t}),a_t:=y_t.$* 

- **Proof**
    
    *Recall that* 
    
    $$
    \begin{align*}d_{\theta}\left( s \right) :=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}} \right].\end{align*}
    $$
    
    *Thus a direct computation yields that*
    
    $$
    \begin{align*}\nabla _{\theta}\mathbb{E} _{s\sim d_{\theta}}\left[ f\left( s \right) \right] &=\sum_{s\in \mathcal{S}}{\nabla _{\theta}d_{\theta}\left( s \right) f\left( s \right)}\\\,\,                  &=\sum_{s\in \mathcal{S}}{\nabla _{\theta}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\}} \right] \cdot f\left( s \right)}\\\,\,                  &=\nabla _{\theta}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{f\left( s_t \right)} \right] \\\,\,                  &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \left( \sum_{t=0}^{\left| y \right|-1}{f\left( s_t \right)} \right) \nabla _{\theta}\log \pi _{\theta}\left( y|x \right) \right] \\\,\,                  &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=0}^{\left| y \right|-1}{f\left( s_{t^{\prime}} \right)} \right) \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right] \\\,\,                  &\overset{\left( a \right)}{=}\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=t+1}^{\left| y \right|-1}{f\left( s_{t^{\prime}} \right)} \right) \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right] \end{align*}
    $$
    
    *where (a) is due to that, for any two fixed time indexes  $0\le t^{\prime}\le t\le \left| y \right|-1,$*
    
    $$
    \begin{aligned}
    	\mathbb{E} \left[ f\left( s_{t^{\prime}} \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) \right] &=\mathbb{E} \left[ \mathbb{E} \left[ f\left( s_{t^{\prime}} \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |s_t \right] \right]\\
    	\,\,&=\mathbb{E} \left[ f\left( s_{t^{\prime}} \right) \cdot \mathbb{E} \left[ \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right) |s_t \right] \right]\\
    	\,\,&=\mathbb{E} \left[ f\left( s_{t^{\prime}} \right) \cdot 0 \right]\\
    	\,\,&=0.\\
    \end{aligned}
    
    $$
    

---

Applying Theorem 2 directly yields that

$$
\begin{align*}g_2(\theta)&=\sum_{s\in \mathcal{S}}{\nabla _{\theta}d_{\theta}\left( s \right) \cdot \mathbb{E} _{a\sim \pi _{\theta}\left( \cdot |s \right)}\left[ \log \frac{\pi _{T\left( s \right)}\left( a|s \right)}{\pi _{\theta}\left( a|s \right)} \right]} \\\,\,                                              &=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\left( \sum_{t^{\prime}=t+1}^{\left| y \right|-1}{\log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right) \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right],\end{align*}
$$

which essentially uses the future log ratio, i.e., $\sum_{t^{\prime}=t+1}^{\left| y \right|-1}{\log \frac{\pi _{T(x)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}}$  , as the per token advantage in (5). By combining the analytical forms of $g_1(\theta)$  and $g_2(\theta),$ where the former uses the immediate log ratio and the latter uses the future log ratio as the per-token advantage, we recover the result of Theorem 1: the full OPD gradient effectively uses the log-ratio-to-go or any other unbiased version of it as the advantages in the policy gradient.

## 2.4 Do We Really Need Log-Ratio-To-Go to Produce Unbiased Policy Gradient ?

Consider the policy loss and the configurations presented in Section 2.1.3. In this section, we compare the performance of following two types of per-token OPD advantages: 

1. immediate log ratio:  $\hat{A}_{\theta}^{\mathrm{OPD}}\left( s_t,a_t \right) =\log \frac{\pi _{T\left( s \right)}\left( a_t|s_t \right)}{\pi _{\theta}\left( a_t|s_t \right)}.$
2. log-ratio-to-go:  $\hat{A}_{\theta}^{\mathrm{OPD}}\left( s_t,a_t \right) =\sum_{t^{\prime}=t}^{\left| y \right|-1}{\log \frac{\pi _{T\left( s \right)}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}}.$

We summarize the key properties of both advantages in the table below.

| Type of OPD Advantage | immediate log ratio | log-ratio-to-go |
| --- | --- | --- |
| Bias | biased  | unbiased |
| Variance | low | high |
| Optimality in the Stationary Point | Yes | Yes |

The bias analysis for both types of OPD advantages is presented in Sections 1.2 and 1.3, respectively. Regarding the variance, it is evident that the immediate log-ratio as the per-token advantage involves fewer random variables compared to the log-ratio-to-go used in stochastic policy gradients, thus suffering from the lower variance. As for optimality at the stationary point, we now present the theoretical results below.

---

**Theorem 3 (Optimality At The Stationary Point).**  *Consider the tabular softmax policy $\pi_\theta$ with parameter $\theta$ and the gradient function of following form:*

$$
g\left( \theta \right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{\left| y \right|-1}{\hat{A}_{\theta}^{\mathrm{OPD}}\left( s_t,a_t \right) \cdot \nabla _{\theta}\log \pi _{\theta}\left( a_t|s_t \right)} \right],
$$

*where the per-token OPD advantage $\hat{A}_{\theta}^{\mathrm{OPD}}\left( s_t,a_t \right)$ can either be the immediate log ratio or log-ratio-to-go. If the gradient norm at $\theta =\hat{\theta}$ equals to zero, i.e., $\left\| g\left( \hat{\theta} \right) \right\| _2=0,$ then $\pi _{\hat{\theta}}$ is the optimal policy for OPD objective $\mathcal{V}(\theta)$ in (3), i.e.,* 

$$
\mathbb{E} _{x\sim \mathcal{D}}\left[ \mathrm{KL}\left( \pi _{\hat{\theta}}\left( \cdot |x \right) ,\pi _{T\left( x \right)}\left( \cdot |x \right) \right) \right] = 0,
$$

*which is equivalent to $\forall x\in \mathrm{support}\left( \mathcal{D} \right) :  \pi _{\hat{\theta}}\left( \cdot |x \right) =\pi _{T\left( x \right)}\left( \cdot |x \right).$*

- **Proof for immediate log ratio**
    
    *Based on the bias analysis in Section 1.3, when the per-token OPD advantage is immediate log ratio, the gradient $g(\theta)$ can be rewritten as* 
    
    $$
    g\left( \theta \right) =-\sum_s{d_{\theta}\left( s \right) \cdot \nabla _{\theta}\mathrm{KL}\left( \pi _{\theta}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right)},
    $$
    
     *which is actually a partial derivative of the OPD objective $\mathcal{V}(\theta),$ lacking the gradient on the state occupancy $d_\theta$ (see (8) for the definition). In the tabular setting, different states share different parameters. Thus for arbitrary state $s$ that is supported on $d_{\hat{\theta}},$ one has*
    
    $$
    \begin{align}\left\| g\left( \hat{\theta} \right) \right\| _2=0 \Rightarrow \,\,\nabla _{\theta}\mathrm{KL}\left( \pi _{\hat{\theta}}\left( \cdot |s \right) ,\pi _{T\left( s \right)}\left( \cdot |s \right) \right) =0,\end{align}
    $$
    
    *which implies that*
    
    $$
    \begin{align}\forall s\in \mathrm{support}\left( d_{\hat{\theta}} \right) :   \pi _{\hat{\theta}}\left( \cdot |s \right) =\pi _{T\left( s \right)}\left( \cdot |s \right)\end{align}
    $$
    
    *Without loss of generality, we assume the response $y$ is no longer than a positive integer $H.$ If we define the state space at time $t$ as $\mathcal{S}_t,$ the state occupancy at time $t$ as* 
    
    $$
    d_{\theta}^{t}\left( s \right) :=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi \left( \cdot |x \right)}\left[ \mathbb{I} \left\{ \left( x,y_{<t} \right) =s \right\} \right],
    $$
    
    *then it’s easy to see that the state occupancy satisfies $d_{\theta}\left( s \right)  =\sum_{t=0}^{H-1}{d_{\theta}^{t}\left( s \right)}.$ We now prove by induction that, for $\forall\ 0\le t \le H-1,$*
    
    $$
    \begin{align} \forall s_t\in \mathcal{S} _t, d_{\hat{\theta}}^{t}\left( s_t \right) =d_{\pi _T}^{t}\left( s_t \right).\end{align}
    $$
    
    *We first assume the condition (13) is satisfied by $0\le t\le m.$  Note that for any state $s_{m+1}=(x,y_{< m+1}) \in \mathcal{S}_{m+1},$  a direct computation yields that*
    
    $$
    \begin{align*}d_{\hat{\theta}}^{m+1}\left( s_{m+1} \right) &=d_{\hat{\theta}}^{m}\left( x,y_{<m} \right) \cdot \pi _{\hat{\theta}}\left( y_m|x,y_{<m} \right) 
    \\
    \,\,              &\overset{\left( a \right)}{=}d_{\pi _T}^{m}\left( x,y_{<m} \right) \cdot \pi _{\hat{\theta}}\left( y_m|x,y_{<m} \right) 
    \\
    \,\,              &\overset{\left( b \right)}{=}d_{\pi _T}^{m}\left( x,y_{<m} \right) \cdot \pi _{T\left( x \right)}\left( y_m|x,y_{<m} \right) \\ &=d^{m+1}_{\pi_T}(s_{m+1}) \end{align*}
    $$
    
    *where (a) is due to the assumption, (b) is due to (12).  This implies that the condition (13) can be satisfied for t=m+1 under the assumption. Note that for t=0, $S_0$ is simply the set of prompts and the state occupancy for any policy is $\mathcal{D},$ thus $d_{\hat{\theta}}^{0}\left( \cdot \right) =\mathcal{D} =d_{\pi _T}^{0}\left( \cdot \right),$ and the condition (13) is satisfied for m=0.  By induction,  condition (13) therefore holds for all $\forall\ 0\le t \le H-1,$  which directly completes the proof  together with (12).*
    
- **Proof for log-ratio-to-go**
    
    *By Theorem 1, when the per-token OPD advantage is log-ratio-to-go, the gradient $g(\theta)$ is unbiased w.r.t to the objective $\mathcal{V}(\theta),$ i.e.,*  
    
    $$
    g\left( \theta \right) =\nabla _{\theta}\mathcal{V} \left( \theta \right) =-\mathbb{E} _{x\sim \mathcal{D}}\left[ \nabla _{\theta}\mathrm{KL}\left( \pi _{\theta}\left( \cdot |x \right) ,\pi _{T\left( x \right)}\left( \cdot |x \right) \right) \right].
    $$
    
    *In the tabular setting, different prompt $x$ share different parameters, thus*
    
    $$
    \begin{align*}\left\| g\left( \hat{\theta} \right) \right\| _2=0&\Leftrightarrow \forall x\in \mathrm{Support}\left( \mathcal{D} \right) , \nabla _{\theta}\mathrm{KL}\left( \pi _{\theta}\left( \cdot |x \right) ,\pi _{T\left( x \right)}\left( \cdot |x \right) \right) =0\\\,\,               &\Leftrightarrow \forall x\in \mathrm{Support}\left( \mathcal{D} \right) , \pi _{\theta}\left( \cdot |x \right) =\pi _{T\left( x \right)}\left( \cdot |x \right) .\end{align*}
    $$
    

---