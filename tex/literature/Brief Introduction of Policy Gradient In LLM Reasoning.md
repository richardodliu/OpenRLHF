# Brief Introduction of Policy Gradient In LLM Reasoning

Author :  Jiacai Liu 

Email    :  23110980012@m.fudan.edu.cn

url : [https://www.notion.so/Brief-Introduction-of-Policy-Gradient-In-LLM-Reasoning-1c04795a3e8b805abbd6ccc9f1a34ac0](Brief%20Introduction%20of%20Policy%20Gradient%20In%20LLM%20Reaso%201c04795a3e8b805abbd6ccc9f1a34ac0.md)

# Basic Ideas

### 1. Notation & Objective

Recently,  reinforcement learning (RL) methods, especially the policy gradient methods,  have achieved great success in enhancing the reasoning abilities of LLM by optimizing its token generation policy. The token generation policy of one LLM can be represented as a policy $\pi$.  Given the text prefix $s$,  the next token $a$ is sampled from the distribution  $\pi(\cdot|s) \in \Delta(\mathcal{V})$. Here we use $\Delta(\mathcal{V})$ as the probability simplex on vocabulary $\mathcal{V}$.  With a slightly abuse of this notation, for any sequence of tokens $\hat{y}$  with length $\hat{T}$ and input prefix $x$, we also use $\pi(\hat{y}|\hat{x})$ as the sequence generation policy and

$$
\begin{align}\pi \left( \hat{y}|\hat{x} \right) =\prod_{t=0}^{\hat{T}-1}{\pi \left( \hat{y}_t|\left( \hat{x},\hat{y}_{<t} \right) \right) .}\end{align}
$$

Here $\hat{y}_{t}$ is the token in the sequence at index $t$ and $\hat{y}_{<t}:=(\hat{y}_0,...,\hat{y}_{t-1})$.   In order to enhance the reasoning abilities of LLM,  we  often want to find a new policy that maximizes the reward function $\mathcal{r}$ in the trust region of reference policy $\pi_\text{ref}$, i.e. 

$$
\begin{align}
\underset{\pi}{\max}\,\,\left\{ \mathcal{J} \left( \textcolor{red}{\pi} \right) :=\mathbb{E} _{x\sim \mathcal{D}}\left[ \mathbb{E} _{y\sim \textcolor{red}{\pi} \left( \cdot |x \right)}\left[ r\left( x,y \right) \right] -\beta \,\mathrm{KL}\left( \textcolor{red}{\pi_x},\pi _{x}^{\mathrm{ref}} \right) \right] \right\}，
\end{align}
$$

where $y=(a_0,...,a_{T-1})$   is the response of prompt $x$, $a_t$ is the token in the response at index $t$,  $\mathcal{D}$ is the sampling distribution of $x$, $\beta$ is the coefficient of  KL regularization,  $\pi_x,\pi^{\text{ref}}_x$ is short for $\pi(\cdot|x)$ and $\pi_{\text{ref}}(\cdot|x)$  respectively.  For parameterized policy $\pi_\theta$,  we  use  $\mathcal{J}(\theta):=\mathcal{J}(\pi_\theta)$ for ease of notation.

### 2. Policy Gradient

Policy gradient methods solve the optimal policy by gradient ascent directly, i.e. 

$$
\theta ^{k+1}=\theta ^k+ \eta_k\nabla _{\theta}\mathcal{J} \left( \theta ^k \right).
$$

where $\eta_k$ is the step size. Following theorem gives an expression of  $\nabla_\theta\mathcal{J}(\theta)$.

**Theorem 1 (Policy Gradient Theorem)** *For parameterized policy $\pi_\theta$,* 

$$
\begin{align}
\nabla _{\theta}\mathcal{J} \left( \textcolor{red}{\theta} \right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \left( r\left( x,y \right) -\beta\log \frac{\pi _{\textcolor{red}{\theta}}\left( y|x \right)}{\pi _{\mathrm{ref}}\left( y|x \right)} \right) \nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( y|x \right) \right] 
\end{align}
$$

Note that  the response  $y$ is a sequence of tokens, thus by (1), we know that 

$$
\begin{align*}\log \pi _{\theta}\left( y|x \right) &=\log \prod_{t=0}^{T-1}{\pi _{\theta}\left( a_t|\left( x,a_0,...,a_{t-1} \right) \right) .}
\\
\,\,             &=\log \prod_{t=0}^{T-1}{\pi _{\theta}\left( a_t|s_t \right)}
\\
\,\,             &=\sum_{t=0}^{T-1}{\log \pi _{\theta}\left( a_t|s_t \right)},\end{align*}
$$

where $s_t:=\left( x,a_0,...,a_{t-1} \right)$. Plugging it into (3) we obtain following token-level policy gradient theorem directly.

**Theorem 2 (Token-level Policy Gradient Theorem 1)** *For parameterized policy $\pi_\theta$,* 

$$
\begin{align}\nabla _{\theta}\mathcal{J} \left( \textcolor{red}{\theta}\right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right) \left( r\left( x,y \right) -\beta\sum_{t^\prime=0}^{T-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( a_{t^\prime}|s_{t^\prime} \right)}{\pi _{\text{ref}}\left( a_{t^\prime}|s_{t^\prime} \right)}} \right)} \right],\end{align}
$$

*where $y=\left(a_0,....,a_{T-1} \right)$, $s_t=\left( x,a_0,...,a_{t-1} \right)$,* $a_t\sim \pi_\theta(\cdot|s_t)$.

### 3. Implementation  in A**utomatic Differentiation Frameworks**

Here are instructions on how to implement the policy gradient method in modern automatic differentiation frameworks like PyTorch. These frameworks support updating parameters when a scalar loss involving those parameters is computed on a data batch.  This implies that the frameworks can only optimize parameters that explicitly appear in the loss expression, while parameters involved in the distribution of the input data fed to the loss function will not be optimized.  Thus for those objective functions with parameter appears in the expectation, we need to construct the surrogate objective function. Suppose current parameter value is $\theta_k$, according to(4), we can construct 

$$
\begin{align}\mathcal{L}_k\left( \textcolor{red}{\theta} \right) =-\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\frac{\pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right)}{\pi _{\theta _k}\left( a_t|s_t \right)}\cdot \left( r\left( x,y \right) -\beta \sum_{t^{\prime}=0}^{T-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right)} \right] \end{align}
$$

as the policy loss function.  It can be shown that performing one step **exact** gradient descent w.r.t $\mathcal{L_k}(\theta)$ is actually doing one step policy gradient ascent. This can be verified as 

$$
\frac{\nabla _{\theta}\pi _{\theta _k}\left( a_t|s_t \right)}{\pi _{\theta _k}\left( a_t|s_t \right)}=\left. \left( \frac{\pi _{\textcolor{red}{\theta} }\left( a_t|s_t \right)}{\pi _{\theta _k}\left( a_t|s_t \right)}\nabla _{\theta}\log \pi _{\textcolor{red}{\theta} }\left( a_t|s_t \right) \right) \right|_{\textcolor{red}{\theta} =\theta _k}=\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right),
$$

And

$$
\begin{align*}
\nabla _{\theta}\mathcal{L} _k\left( \theta _k \right) &=-\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\frac{\nabla _{\theta}\pi _{\theta _k}\left( a_t|s_t \right)}{\pi _{\theta _k}\left( a_t|s_t \right)}\left( r\left( x,y \right) -\beta \sum_{t^{\prime}=0}^{T-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right)} \right] 
\\
\,\,           &=-\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right) \left( r\left( x,y \right) -\beta \sum_{t^{\prime}=0}^{T-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}} \right)} \right] 
\\
\,\,          &=-\nabla _{\theta}\mathcal{J} \left( \theta _k \right) .
\end{align*}
$$

Thus, at each step $k \in \mathbb{N}$,  we can sample a batch of N prompts $x_1,…,x_N \sim \mathcal{D}$ and M iid responses $y_{i1},y_{i2},…y_{iM} \sim \pi_{\theta_k}(\cdot|x_i)$ for each prompt $x_i$.  We denote that 

$y_{ij}=(a^{(ij)}_0,...,a^{(ij)}_{T_{ij}-1})$ and  $s^{(ij)}_t=(x,a^{(ij)}_0,...,a^{(ij)}_{t-1})$.  Suppose the policy loss for each sample pair $(x_i,y_{ij})$ is  $\hat{\mathcal{L}}_k\left( \theta|x_i,y_{ij} \right)$. Then we can compute the total policy loss as 

$$
\begin{align}\hat{\mathcal{L}}_k\left( \theta \right) =\frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\hat{\mathcal{L}}_k\left( \theta |x_i,y_{ij} \right)}}.\end{align}
$$

If the each policy loss is unbiased, i.e.  $\mathbb{E} [\nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta_k|x_i,y_{ij} \right)]=-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$,  then we can see that the total policy loss is also unbiased, i.e.  $\mathbb{E} [\nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta_k \right)]=-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$.

### 4. REINFORCE

**REINFORCE** updates the parameter by performing the stochastic gradient descent w.r.t $\mathcal{L}_k(\theta)$ using pure **Monte-Carlo (MC) sampling**.  According to (5),  ****the policy loss for each sample pair $(x_i,y_{ij})$  in **REINFORCE** is computed as 

$$
\begin{align}
\hat{\mathcal{L}}_k\left( \textcolor{red}{\theta} |x_i,y_{ij} \right) =-\sum_{t=0}^{T_{ij}-1}\left[{\frac{\pi _{\textcolor{red}{\theta}}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}{\pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}\cdot \left( r\left( x_i,y_{ij} \right) -\beta \sum_{t^{\prime}=0}^{T_{ij}-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}} \right)}\right],
\end{align}
$$

It’s easy to see that the each sample policy loss is unbiased, i.e.  $\mathbb{E} [\nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta_k|x_i,y_{ij} \right)]=-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$.  However, the pure MC sampling applied by **REINFORCE**  often suffers from high variance especially in the KL terms.  More advanced tricks are introduced in order to reduce the variance.  

### 5. REINFORCE with KL Trick

The first trick we show is that we do not need to compute the KL penalty of the full trajectory, i.e. $\sum_{t^{\prime}=0}^{T-1}{\log \frac{\pi_{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi_{\mathrm{ref}}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}}$, as the weight of $\nabla _{\theta}\log \pi _{\theta }\left( a_t|s_t \right)$ in the gradient, we can only use the KL penalty starts from each time step $t$,  i.e. $\sum_{t^{\prime}=\textcolor{red}{t}}^{T-1}{\log \frac{\pi_{\theta}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}{\pi_{\mathrm{ref}}\left( a_{t^{\prime}}|s_{t^{\prime}} \right)}}$ as the weight of $\nabla _{\theta}\log \pi _{\theta }\left( a_t|s_t \right)$.  

**Theorem 3 (Token-level Policy Gradient Theorem 2)** *For parameterized policy $\pi_\theta$,* 

$$
\begin{align}\nabla _{\theta}\mathcal{J} \left( \textcolor{red}{\theta}\right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right) \left( r\left( x,y \right) -\beta\sum_{t^\prime=\textcolor{red}{t}}^{T-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( a_{t^\prime}|s_{t^\prime} \right)}{\pi _{\text{ref}}\left( a_{t^\prime}|s_{t^\prime} \right)}} \right)} \right]\end{align}
$$

*where $y=\left(a_0,....,a_{T-1} \right)$, $s_t=\left( x,a_0,...,a_{t-1} \right)$,* $a_t\sim \pi_\theta(\cdot|s_t)$.

Based on Theorem 3,  A more refined **REINFORCE** policy loss for each sample pair $(x_i,y_{ij})$   is

$$
\begin{align}
\hat{\mathcal{L}}_k\left( \textcolor{red}{\theta} |x_i,y_{ij} \right) =-\sum_{t=0}^{T_{ij}-1}\left[{\frac{\pi _{\textcolor{red}{\theta}}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}{\pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}\cdot \left( r\left( x_i,y_{ij} \right) -\beta \sum_{t^{\prime}=\textcolor{red}{t}}^{T_{ij}-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}} \right)}\right],
\end{align}
$$

Compared with (7),  (9) reduces the variance due to less random variables are involved.  However, the term $r(x_i,y_{ij})$  also introduces some variance.  

### 6. Policy Gradient with Q Values

A popular class of methods called Actor-Critic (AC)  is further designed to reduce the variance of the reward term, i.e. $r(x_i,y_{ij})$  in **REINFORCE**  by leveraging the value functions.  we first define following **value functions.**  For any text prefix $s$, we define the state value function $V^{\pi _{\theta}}\left( s \right)$ as the expected reward of the completion $y$ that is sampled from $\pi_\theta(\cdot|s)$, i.e. 

$$
\begin{align*}V^{\pi _{\theta}}\left( s \right) =\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |s \right)}\left[ r\left( s,y \right) \right].\end{align*}
$$

For any text prefix $s$ and token $a$, we define the state-action value function $Q^{\pi _{\theta}}\left( s,a\right)$  as the the expected reward of the completion $y$ that is sampled from $\pi_\theta(\cdot|(s,a))$, i.e.

$$
\begin{align*}Q^{\pi _{\theta}}\left( s,a \right) =\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |s^\prime\right)}\left[ r\left( s^\prime ,y \right) \right],\end{align*}
$$

where $s^\prime=\left( s,a \right)$.  It’s easy to see that $Q^{\pi _{\theta}}\left( s,a \right) =V^{\pi _{\theta}}\left( s^{\prime} \right)$.  Following theorems states that we can use the Q function of the current policy as the weight of  $\nabla _{\theta}\log \pi _{\theta }\left( a_t|s_t \right)$.

**Theorem 4 (Token-level Policy Gradient Theorem 3)** *For parameterized policy $\pi_\theta$,* 

$$
\begin{align}\nabla _{\theta}\mathcal{J} \left( \textcolor{red}{\theta}\right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right) \left( Q^{\pi _{\textcolor{red}{\theta}}}\left( s_t,a_t \right) -\beta\sum_{t^\prime=\textcolor{red}{t}}^{T-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( a_{t^\prime}|s_{t^\prime} \right)}{\pi _{\text{ref}}\left( a_{t^\prime}|s_{t^\prime} \right)}} \right)} \right]\end{align}
$$

*where $y=\left(a_0,....,a_{T-1} \right)$, $s_t=\left( x,a_0,...,a_{t-1} \right)$,* $a_t\sim \pi_\theta(\cdot|s_t)$.

The proof of this theorem is based on following key observation

$$
\begin{align*}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ r\left( x,y \right) |s_t,a_t \right] =Q^{\pi _{\theta}}\left( s_t,a_t \right) \,\,\forall x.
\end{align*}
$$

**AC methods train an independent critic function to predict the true value function $Q^{\pi_{\theta_k}}$ or $V^{\pi_{\theta_k}}$.** More specifically, at each step $k\in\mathbb{N}$,  for each sample pair $(x_i,y_{ij})$ defined in section 3,  the estimated Q values  $\hat{Q}(s^{(ij)}_t,a^{(ij)}_t) \approx Q^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)$ are constructed for each state-action pair $(s^{(ij)}_t,a^{(ij)}_t)$.  After that,  we can use 

$$
\begin{align}
\hat{\mathcal{L}}_k\left( \textcolor{red}{\theta}|x_i,y_{ij} \right) =-\sum_{t=0}^{T_{ij}-1}\left[{\frac{\pi _{\textcolor{red}{\theta}}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}{\pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}\cdot \left( \hat{Q}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\beta \sum_{t^{\prime}=\textcolor{red}{t}}^{T_{ij}-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}} \right)}\right],\end{align}
$$

as the policy loss for sample pair $(x_i,y_{ij})$.  Compared with policy loss of **REINFORCE** in (7) (9),  (11) further reduces the variance by replacing the term $r(x_i,y_{ij})$ with $\hat{Q}(s^{(ij)}_t,a^{(ij)}_t)$ since $r(x_i,y_{ij})$ is still uncertain given with $(s^{(ij)}_t,a^{(ij)}_t)$  whereas $\hat{Q}(s^{(ij)}_t,a^{(ij)}_t)$ is a constant given with $(s^{(ij)}_t,a^{(ij)}_t)$. However,  $\hat{Q}(s^{(ij)}_t,a^{(ij)}_t)$ may not equal to $Q^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)$ exactly, the bias is introduced inevitable. That is the policy loss may **not** satisfies  $\mathbb{E} [\nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta_k|x_i,y_{ij} \right)]\ne-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$. 

### 6. Policy Gradient with Advantages

However, even we have access to the exact Q value in the policy loss (11), i.e.  $\hat{Q}(s^{(ij)}_t,a^{(ij)}_t) =Q^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)$, there stills remains room to reduce the variance further  in finite samples scenarios.  Suppose $\beta=0$, performing one step gradient descent on $\hat{\mathcal{L}}_k\left( \theta|x_i,y_{ij} \right)$ in (11) directly will generally increase the probability for all tokens in $y_{ij}$ since the Q values  $Q^{\pi _{\theta _k}}(s^{(ij)}_t,a^{(ij)}_t)$  are all positive.  However for those tokens with lower Q values, the token probabilities should be decreased.  Such training variance will be only  reduced when the batch size $N$ and number of responses $M$ are large enough, i.e. $N,M \rightarrow \infty$

.  **So in finite samples scenarios,  we need a baseline term** $b(s^{(ij)}_t)$  **to determine the direction (increase or decrease) and magnitude of probability adjustments for each state-action pair** $(s^{(ij)}_t,a^{(ij)}_t)$.  Following theorem shows that adding a baseline term will not influence the updating direction of the gradient. 

**Theorem 5 (Token-level Policy Gradient Theorem 4)** *For parameterized policy $\pi_\theta$ and function* $b$, 

$$
\begin{align}\nabla _{\theta}\mathcal{J} \left( \textcolor{red}{\theta}\right) =\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\textcolor{red}{\theta}}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\nabla _{\theta}\log \pi _{\textcolor{red}{\theta}}\left( a_t|s_t \right) \left( Q^{\pi _{\textcolor{red}{\theta}}}\left( s_t,a_t \right)-\textcolor{red}{b(s_t)} -\beta\sum_{t^\prime=\textcolor{red}{t}}^{T-1}{\log \frac{\pi _{\textcolor{red}{\theta}}\left( a_{t^\prime}|s_{t^\prime} \right)}{\pi _{\text{ref}}\left( a_{t^\prime}|s_{t^\prime} \right)}} \right)} \right]\end{align}
$$

*where $y=\left(a_0,....,a_{T-1} \right)$, $s_t=\left( x,a_0,...,a_{t-1} \right)$,* $a_t\sim \pi_\theta(\cdot|s_t)$.

According to Theorem 5,  we can use 

$$
\begin{align}
\hat{\mathcal{L}}_k\left( \textcolor{red}{\theta}|x_i,y_{ij} \right) =-\sum_{t=0}^{T_{ij}-1}\left[{\frac{\pi _{\textcolor{red}{\theta}}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}{\pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}\cdot \left( \hat{Q}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)-b(s^{(ij)}_t) -\beta \sum_{t^{\prime}=\textcolor{red}{t}}^{T_{ij}-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}} \right)}\right],\end{align}
$$

as the policy loss for sample pair $(x_i,y_{ij})$ where  $\hat{Q}(s^{(ij)}_t,a^{(ij)}_t) \approx Q^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)$.  **A natural question arises : which baseline term should I choose ?**   In practice ****choosing $b\left( s_t \right) =V^{\pi _{\theta}}\left( s_t \right)$ generally yields minimum variance in finite sample scenarios. We define the advantage function as $A^{\pi _{\theta}}\left( s,a \right) =Q^{\pi _{\theta}}\left( s,a \right) -V^{\pi _{\theta}}\left( s \right)$.  Directly analysing the variance matrix of (13) is difficult and beyond the scope of this blog.  Note that 

$$
V^{\pi _{\theta}}\left( s \right) =\underset{c\in \mathbb{R}}{\mathrm{arg}\min}\,\,\mathbb{E} _{a\sim \pi _{\theta}\left( \cdot |s \right)}\left( Q^{\pi _{\theta _k}}\left( s,a \right) -c \right) ^2.
$$

This provide an  intuition on why $b\left( s_t \right) =V^{\pi _{\theta}}\left( s_t \right)$ may be a good choice as the baseline term.

# Analysis of Popular Policy Gradient Methods in LLM Reasoning

At each step $k\in \mathbb{N}$,  suppose  a batch of N prompts $x_1,…,x_N \sim \mathcal{D}$ and M iid responses $y_{i1},y_{i2},…y_{iM} \sim \pi_{\theta_k}(\cdot|x_i)$ for each prompt $x_i$ are sampled.  By policy gradient theorems  (Theorem 1-5) and the ideas in section 3,   the policy loss of policy gradient based methods applied in deep-learning frameworks  generally take the form of 

$$
\begin{align}
\hat{\mathcal{L}}_k\left( \textcolor{red}{\theta} \right) =-\frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \underset{:=\hat{\mathcal{L}}_k\left( \textcolor{red}{\theta} |x_i,y_{ij} \right)}{\underbrace{\sum_{t=0}^{T_{ij}-1}{\frac{\pi _{\textcolor{red}{\theta}}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}{\pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}\cdot \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)}}} \right]}}
\end{align}
$$

And the gradient at current parameter value $\theta=\theta_k$ satisfies 

$$
\begin{align}\nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) =-\frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \underset{:=\nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k|x_i,y_{ij} \right)}{\underbrace{\sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)}}} \right]}}\end{align}
$$

where $y_{ij}=(a^{(ij)}_0,...,a^{(ij)}_{T_{ij}-1})$  , $s^{(ij)}_t=(x,a^{(ij)}_0,...,a^{(ij)}_{t-1})$ and $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)$ is the estimated advantage.  **From (14) we know that we should sum the policy loss of the tokens within each sample pair $(x_i,y_{ij})$ then take average on the total sample pairs.**  Note that for arbitrary index $(i,j,t$), 

$$
\begin{align*}
\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] =\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] \right] 
\\
\,\,                                                  =\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \textcolor{red}{\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] }\right],
\end{align*}
$$

Thus the expectation of the gradient (15) satisfies

$$
\begin{align}\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) \right]&=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)} \right]}} \right] \\\,\,                    &=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \textcolor{red}{\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]}} \right]}} \right] \end{align}
$$

Thus in the following analysis, we mainly focus on :  1) The design  of the token advantage  $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)$    2)  The expectation and variance of the token advantage. For ease of notations in the following analysis, we define  $\mathbb{X} :=\left\{ x_1,..,x_n \right\}$ ,$\mathbb{Y} :=\left\{ y_{11},..,y_{NM} \right\}$, $\mathbb{R}_i:=\{r(x_i,y_{i1}),...,(x_i,y_{iM})\}$ and $\mathbb{R}:=\{r(x_1,y_{11}),...,(x_{N},y_{NM})\}$.

## 1. Advantage Estimations of Popular Policy Gradient Methods

Current popular policy gradient methods mainly focus on the design of $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)$.  The choices of  $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)$ of different algorithms are summarized in following tables:

| Algorithm | Token Advantage $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)$ |
| --- | --- |
| **REINFORCE** | $r\left( x_i,y_{ij} \right) -\beta \sum_{t^{\prime}=0}^{T_{ij}-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}}$  |
| **REINFORCE w KL trick** | $r\left( x_i,y_{ij} \right) -\beta \sum_{t^{\prime}=t}^{T_{ij}-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}}$  |
| **REINFORCE ++** | $r\left( x_i,y_{ij} \right) -\beta \sum_{t^{\prime}=t}^{T_{ij}-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}}$  with global batch normalization |
| **RLOO** | $R_{ij}-\frac{1}{M-1}\sum_{l\ne j}{R_{ij}}, \text{  where } R_{ij}:=r\left( x_i,y_{ij} \right) -\beta \sum_{t^{\prime}=0}^{T_{ij}-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}}$ |
| **REMAX** | $r\left( x_i,y_{ij} \right) -r\left( x_i,\hat{y}_i \right) -\beta \sum_{t^{\prime}=t}^{T_{ij}-1}{\log \frac{\pi _{\theta _k}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}{\pi _{\mathrm{ref}}\left( a_{t^{\prime}}^{\left( ij \right)}|s_{t^{\prime}}^{\left( ij \right)} \right)}}$
, where $\hat{y}_i$ is sampled from greedy policy of $\theta_k$ |
| **PPO** | $\sum_{t^{\prime}=t}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}\delta _{t^{\prime}}^{\left( ij \right)}}$  where $\delta _{t}^{\left( ij \right)}=\mathbb{I} \left\{ t=T_{ij}-1 \right\} \cdot r\left( x_i,y_{ij} \right) -\beta \log \frac{\pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}{\pi _{\mathrm{ref}}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}+\gamma V_{\phi}\left( s_{t+1}^{\left( ij \right)} \right) -V_{\phi}\left( s_{t}^{\left( ij \right)} \right)$  |
| **ORZ**  | $r\left( x_i,y_{ij} \right) -V_{\phi}\left( s_{t}^{\left( ij \right)} \right)$   **(PPO with $\lambda=\gamma=1$, $\beta=0$)** |
| **GRPO (origin verson)** | $\frac{1}{T_{ij}}\left[ \frac{r\left( x_i,y_{ij} \right) -\text{mean}(\mathbb{R}_i)}{\text{std}(\mathbb{R}_i)}+\beta \left( \frac{\pi _{\mathrm{ref}}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}{\pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}-1 \right) \right]$ |
| **GRPO (R1 verson)** |  $\frac{r\left( x_i,y_{ij} \right) -\text{mean}(\mathbb{R}_i)}{\text{std}(\mathbb{R}_i)}+\beta \left( \frac{\pi _{\mathrm{ref}}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}{\pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right)}-1 \right)$ |
| **DAPO** | $\frac{1}{\hat{T}_i}\left[ \frac{r\left( x_i,y_{ij} \right) -\text{mean}(\mathbb{R}_i)}{\text{std}(\mathbb{R}_i)} \right]$ , where  $\hat{T}_i=\frac{1}{M}\sum_{j=1}^M{T_{ij}}$. |
| **DR.GRPO** | $r\left( x_i,y_{ij} \right) -\text{mean}(\mathbb{R}_i)$ |

## 2. Analysis in Binary Reward Scenarios

we further analyze these algorithms in binary reward scenarios, i.e.  $r(x,y)\in \{0,1\}$. For simplicity, we assume $\beta=0$. 

### Bias Analysis

We first study the expectation of the gradient of the policy loss.  **Please refer the subsections of each algorithm for details** (e.g. the notations). 

| Algorithm | Token Advantage $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)$ | Expectation of Token Advantage $\mathbb{E}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]$ 
 | Expectation of  the Gradient of Policy Loss $\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}\left( \theta _k \right) \right]$   |
| --- | --- | --- | --- |
| **REINFORCE** | $r\left( x_i,y_{ij} \right)$ | $Q^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)$ | $-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$  |
| **REINFORCE ++** | $\frac{1}{\text{std}(\mathbb{R})}\left[ r\left( x_i,y_{ij} \right) -\text{mean}(\mathbb{R})\right]$  | $\textcolor{red}{\frac{1}{\sigma _k}}\left( Q^{\pi _{\theta _k}}(s_{t}^{(ij)},a_{t}^{(ij)})-\textcolor{red}{\mu _k}\right)$  | $-\textcolor{red}{\frac{1}{\sigma _k}}\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$   asymptotically as NM is large enough |
| **RLOO** | $r\left( x_i,y_{ij} \right) -\frac{1}{M-1}\sum_{l\ne j}{r\left( x_i,y_{il} \right)}$ | $Q^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)-\textcolor{red}{V^{\pi_{\theta_k}}(x_i)}$ | $-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$  |
| **REMAX** | $r\left( x_i,y_{ij} \right) -r\left( x_i,\hat{y}_i \right)$, where $\hat{y}_i$ is sampled from the greedy policy of $\hat{\pi}_{\theta_k}$ | $Q^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)-\textcolor{red}{V^{\hat{\pi}_{\theta_k}}(x_i)}$ | $-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$ |
| **PPO** | $\left( \lambda \gamma \right) ^{T_{ij}-1-t}r\left( x_i,y_{ij} \right) +\sum_{t^{\prime}=t+1}^{T_{ij}-1}{\left( \frac{1}{\lambda}-1 \right) \left( \lambda \gamma \right) ^{t^{\prime}-t}V_{\phi}\left( s_{t^{\prime}}^{\left( ij \right)} \right)} -V_{\phi}\left( s_{t}^{\left( ij \right)} \right)$   | $A_{\lambda,\gamma}^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)$.  If $V_\phi=V^{\pi_{\theta_k}}$ and $\gamma=1$, $\textcolor{red}{A_{\lambda,\gamma}^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)=A^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)}$ | $-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$ If $V_\phi=V^{\pi_{\theta_k}}$ and $\gamma=1$.   |
| **ORZ** | $r\left( x_i,y_{ij} \right) -V_{\phi}\left( s_{t}^{\left( ij \right)} \right)$(PPO with $\lambda=\gamma=1$) | $Q^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)-\textcolor{red}{V_{\phi}\left( s_{t}^{\left( ij \right)} \right)}$  | $-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$ |
| **GRPO (origin version)** | $\frac{1}{T_{ij}}\left[ \frac{r\left( x_i,y_{ij} \right) -\hat{p}_i}{\sqrt{\hat{p}_i(1-\hat{p}_i)}} \right]$ | Hard to analyze due to the length normalization term $\textcolor{red}{\frac{1}{T_{ij}}}$ | $-\mathbb{E}_{x\sim \mathcal{D},y\sim\pi_{\theta_k}(\cdot|x)} \left[ \frac{r\left( x,y \right) \nabla _{\theta}\log \pi _{\theta _k}\left( x,y \right)}{\textcolor{red}{T\sqrt{p_{\theta_k}\left( x \right) \left( 1-p_{\theta_k}\left( x \right) \right)}}} \right]\ne -\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$  asymptotically as M is large enough |
| **GRPO (R1 version)** |  $\frac{r\left( x_i,y_{ij} \right) -\hat{p}_i}{\sqrt{\hat{p}_i(1-\hat{p}_i)}}$  | $\frac{Q^{\pi _{\theta _k}}(s_{t}^{(ij)},a_{t}^{(ij)})-\textcolor{red}{p_{\theta _k}\left( x_i \right)}}{\textcolor{red}{p_{\theta _k}\left( x_i \right) \sqrt{1-p_{\theta _k}\left( x_i \right)}}}$ asymptotically as M is large enough | $~~-\mathbb{E}_{x\sim \mathcal{D},y\sim\pi_{\theta_k}(\cdot|x)} \left[ \frac{r\left( x,y \right) \nabla _{\theta}\log \pi _{\theta _k}\left( x,y \right)}{\textcolor{red}{\sqrt{p_{\theta_k}\left( x \right) \left( 1-p_{\theta_k}\left( x \right) \right)}}} \right] \ne -\nabla _{\theta}\mathcal{J} \left( \theta _k \right)~~$  asymptotically as M is large enough |
| **DAPO** | $\frac{1}{\hat{T}_i}\left[ \frac{r\left( x_i,y_{ij} \right) -\hat{p}_i}{\sqrt{\hat{p}_i(1-\hat{p}_i)}}   \right]$ , where  $\hat{T}_i=\frac{1}{M}\sum_{j=1}^M{T_{ij}}$ | $\frac{Q^{\pi _{\theta _k}}(s_{t}^{(ij)},a_{t}^{(ij)})-\textcolor{red}{p_{\theta _k}\left( x_i \right)}}{\textcolor{red}{T_{\theta_k}(x_i)p_{\theta _k}\left( x_i \right) \sqrt{1-p_{\theta _k}\left( x_i \right)}}}$ asymptotically as M is large enough | $-\mathbb{E}_{x\sim \mathcal{D},y\sim\pi_{\theta_k}(\cdot|x)} \left[ \frac{r\left( x,y \right) \nabla _{\theta}\log \pi _{\theta _k}\left( x,y \right)}{\textcolor{red}{T_{\theta_k}(x)\cdot\sqrt{p_{\theta_k}\left( x \right) \left( 1-p_{\theta_k}\left( x \right) \right)}}} \right] \ne -\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$ asymptotically as M is large enough |
| **DR.GRPO** | $r\left( x_i,y_{ij} \right) -\hat{p}_i$ |  $\textcolor{red}{\frac{M-1}{M}}\left(Q^{\pi_{\theta_k}}(s^{(ij)}_t,a^{(ij)}_t)-\textcolor{red}{V^{\pi_{\theta_k}}\left(x_i \right)}\right)$ | $-\textcolor{red}{\frac{M-1}{M}}\nabla _{\theta}\mathcal{J} \left( \theta _k \right)$ |

Conclusions :

(1) All algorithms are doing policy gradient except **GRPO (original/R1 version) and DAPO** due to the prompt-based term $p_{\theta_k(x)}, T_{\theta_k(x)}$ and the length term $T$.

(2) The  adaptive learning rate $1/\sigma_k$ and $1-1/M$ are implicitly  applied in  **REINFORCE ++** and **DR.GRPO** respectively**.**

(3) Due to the prompt-based term $1/\sqrt{p_{\theta_k}\left( x \right) \left( 1-p_{\theta}\left( x \right) \right)}$,   **GRPO (original/R1 version)**, and **DAPO** are prefer to learn easy $(p_{\theta_{k}}(x)\rightarrow1)$ or hard problems $(p_{\theta_{k}}(x)\rightarrow 0)$.

(4) **GRPO (original version)** prefer short but correct answer due to length normalization term $1/T$.

(5) **DAPO** prefer to learn those questions that generally have short answers  due to the prompt-based  length normalization term $1/T_{\theta_k}(x)$.

(6)  Suppose $\lambda \gamma <1$.  As  $t \rightarrow0$,  $\left( \lambda \gamma \right) ^{T_{ij}-1-t} \rightarrow 0$. The Q estimation $\hat{Q}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) :=\left( \lambda \gamma \right) ^{T_{ij}-1-t}r\left( x_i,y_{ij} \right) +\sum\nolimits_{t^{\prime}=t+1}^{T_{ij}-1}{\left( \frac{1}{\lambda}-1 \right) \left( \lambda \gamma \right) ^{t^{\prime}-t}V_{\phi}\left( s_{t^{\prime}}^{\left( ij \right)} \right)}$ in **PPO** is dominated by prediction of the critic  $V_{\phi}\left( s_{t^{\prime}}^{\left( ij \right)} \right)$ and may introduce a large  if $V_\phi$ is a bad estimation of $V^{\pi_{\theta_k}}$. **ORZ** use $\lambda=\gamma=1$ and the Q estimation is not influenced by the critic.

### Variance Analysis

We now study the variances of these algorithms. Note that computing the variance of these algorithm explicitly is often hard. Thus we focus on the sign of  token advantage  $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)$  which is the weight of the gradient vector  $\nabla _{\theta}\log \pi _{\theta _k}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)$  and indicates the  the probability updating direction of this token. Generally speaking, if $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)>0$, the algorithm try to increase the probability of this token and vice-visa. 

| Algorithm | Token Advantage $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)$ | sign of $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)$  | Advantage vary with token index $t$ |
| --- | --- | --- | --- |
| **REINFORCE** | $r\left( x_i,y_{ij} \right)$ | positive  for $r(x_i,y_{ij})=0$ | No |
| **REINFORCE ++** | $\frac{1}{\text{std}(\mathbb{R})}\left[ r\left( x_i,y_{ij} \right) -\text{mean}(\mathbb{R})\right]$  | non-negative when $r(x_i,y_{ij})=1$ and  non-positive when $r(x_i,y_{ij})=0$ | No |
| **RLOO** | $r\left( x_i,y_{ij} \right) -\frac{1}{M-1}\sum_{l\ne j}{r\left( x_i,y_{il} \right)}$ | non-negative when $r(x_i,y_{ij})=1$ and  non-positive when $r(x_i,y_{ij})=0$ | No |
| **REMAX** | $r\left( x_i,y_{ij} \right) -r\left( x_i,\hat{y}_i \right)$, where $\hat{y}_i$ is sampled from the greedy policy of $\hat{\pi}_{\theta_k}$ | non-negative when $r(x_i,y_{ij})=1$ and  non-positive when $r(x_i,y_{ij})=0$ | No |
| **PPO** | $\left( \lambda \gamma \right) ^{T_{ij}-1-t}r\left( x_i,y_{ij} \right) +\sum_{t^{\prime}=t+1}^{T_{ij}-1}{\left( \frac{1}{\lambda}-1 \right) \left( \lambda \gamma \right) ^{t^{\prime}-t}V_{\phi}\left( s_{t^{\prime}}^{\left( ij \right)} \right)} -V_{\phi}\left( s_{t}^{\left( ij \right)} \right)$   | Depends on $V_\phi$.  If $V_\phi=V^{\pi_{\theta_k}}$,  $\mathrm{sign}\left( \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right) =\mathrm{sign}\left( A^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right)$  | Yes |
| **ORZ** | $r\left( x_i,y_{ij} \right) -V_{\phi}\left( s_{t}^{\left( ij \right)} \right)$(PPO with $\lambda=\gamma=1$) | non-negative when $r(x_i,y_{ij})=1$ and  non-positive when $r(x_i,y_{ij})=0$ | Yes |
| **GRPO (origin version)** | $\frac{1}{T_{ij}}\left[ \frac{r\left( x_i,y_{ij} \right) -\hat{p}_i}{\sqrt{\hat{p}_i(1-\hat{p}_i)}} \right]$ | non-negative when $r(x_i,y_{ij})=1$ and  non-positive when $r(x_i,y_{ij})=0$ | No |
| **GRPO (R1 version)** |  $\frac{r\left( x_i,y_{ij} \right) -\hat{p}_i}{\sqrt{\hat{p}_i(1-\hat{p}_i)}}$  | non-negative when $r(x_i,y_{ij})=1$ and  non-positive when $r(x_i,y_{ij})=0$ | No |
| **DAPO** | $\frac{1}{\hat{T}_i}\left[ \frac{r\left( x_i,y_{ij} \right) -\hat{p}_i}{\sqrt{\hat{p}_i(1-\hat{p}_i)}}   \right]$ , where  $\hat{T}_i=\frac{1}{M}\sum_{j=1}^M{T_{ij}}$ | non-negative when $r(x_i,y_{ij})=1$ and  non-positive when $r(x_i,y_{ij})=0$ | No |
| **DR.GRPO** | $r\left( x_i,y_{ij} \right) -\hat{p}_i$ |  non-negative when $r(x_i,y_{ij})=1$ and  non-positive when $r(x_i,y_{ij})=0$ | No |

Conclusions :

(1)  Since no baseline term is used in **REINFORCE**,  it try to increase the probabilities of all sampled tokens no matter the answer is correct or not. Thus **REINFORCE** suffers from large variance.

(2)  **All algorithms use $r(x_i,y_{ij})$  as the MC Q estimation except PPO**.  **Thus these algorithms generally try to increase the token probability  if the answer is correct  and decrease the token probability if the answer is incorrect no matter what baseline term is used.  So there is no fundamental difference in the variance reduction among those algorithms no matter which baseline term is used**. ****However, this update pattern indeed introduces training variance since there the sign of  the token’s true advantage value  may mismatch the final reward of the trajectory.  Following figure shows  a concrete example in reasoning step level.

![case_study_1_00.png](Brief%20Introduction%20of%20Policy%20Gradient%20In%20LLM%20Reaso/case_study_1_00.png)

The trajectory shown in this figure contains 14 reasoning steps and is a correct trajectory sampled from Llama-3.1-8B-instruct. The values and advantages of each reasoning step are also reported on the right-hand side of the figure. There is a noticeable drop in value (accuracy) in step 11, decreasing from 0.94 to 0.41, resulting in an advantage of -0.53. This means the model is more likely to make a mistake starting from step 11. A well-directed policy update may reduce the likelihood of step 11, rather than increasing it, even if the final answer is correct. **Thus, there exists trajectories that yield correct answers while containing reasoning steps (tokens) from which the model is likely to make mistakes, and vice versa.**  

(3) **The baseline term and the advantage values are all  prompt-level except in PPO and ORZ .**   However**,** ORZ  only adjusts the scale of advantages among different tokens while the sign of advantages keep the same. Thus only PPO can indeed achieve  fine-grained policy optimization.  

(4) **PPO  with a well trained critic in PPO  should reduce the variance of Q estimation and yields better training performance.**  That being said, the best challenge in the practice of PPO is how to train a well critic in outcome reward setting ? 

### **① REINFORCE**

The **token advantage** is $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) =r\left( x_i,y_{ij} \right)$ .  

The **expectation of token advantage** is 

$$
\begin{align*}\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] &=\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ r\left( x_i,y_{ij} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] \\\,\,                                      &=\mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right)}\left[ r\left( x_i,y \right) \right] \\\,\,                                      &=Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right), \end{align*}
$$

where the last line leverage the definition of value function. 

The **expectation  of policy loss gradient** is  

$$
\begin{align*}\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) \right] &=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]} \right]}} \right] \\\,\,                    &=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)} \right]}} \right] \\\,\,                    &=-\mathbb{E} _{x\sim \mathcal{D} ,y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right) \cdot Q^{\pi _{\theta _k}}\left( s_t,a_t \right)} \right] \\\,\,                    &=-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)\end{align*}
$$

where the last line is due to **Theorem 5.**

### **② REINFORCE ++**

The **token advantage** is  $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) =\frac{1}{\text{std}(\mathbb{R})}\left[ r\left( x_i,y_{ij} \right) -\text{mean}(\mathbb{R})\right].$

For the the **expectation of token advantage,  n**ote that as $NM\rightarrow \infty$, 

$$
\mathrm{mean}\left( \mathbb{R} \right) =\frac{1}{\sum_{i,j}{T_{ij}}}\sum_{i=1}^N{\sum_{j=1}^M{\sum_{t=0}^{T_{ij}}{r\left( x_i,y_{ij} \right)}}}\rightarrow \mu _k:=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ r\left( x,y \right) \frac{T}{T_k} \right]  \,\, \text{a.s}
$$

and 

$$
\begin{align*}\mathrm{std}\left( \mathbb{R} \right) &=\sqrt{\frac{1}{\sum_{i,j}{T_{ij}}}\sum_{i=1}^N{\sum_{j=1}^M{\sum_{t=0}^{T_{ij}}{\left( r\left( x_i,y_{ij} \right) -\mathrm{mean}\left( \mathbb{R} \right) \right)^2}}}}\\&\rightarrow \sigma _k:=\sqrt{\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \left( r\left( x,y \right) -\mu _k \right) ^2\frac{T}{T_k} \right]}\,\,\text{a.s},\end{align*}
$$

where $T_k$ is the expected average token length of current policy $\pi_{\theta_k}$, i.e.

$$
T_k:=\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ T \right].
$$

Thus, 

$$
\begin{aligned}	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] &=\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{\mathrm{std(}\mathbb{R} )}\left[ r\left( x_i,y_{ij} \right) -\mathrm{mean(}\mathbb{R} ) \right] |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]\\	&\rightarrow \textcolor{red}{\frac{1}{\sigma _k}}\left[ \mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ r\left( x_i,y_{ij} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] -\textcolor{red}{\mu _k} \right]\\	\,\,&=\textcolor{red}{\frac{1}{\sigma _k}}\left( Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\textcolor{red}{\mu _k} \right) .\\\end{aligned}
$$

The **expectation  of policy loss gradient** is  

$$
\begin{aligned}
	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) \right] &=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]} \right]}} \right]\\
	\,\,&=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot -\textcolor{red}{\frac{1}{\sigma _k}}\left( Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\textcolor{red}{\mu _k} \right)} \right]}} \right]\\
	\,\,&=-\textcolor{red}{\frac{1}{\sigma _k}}\mathbb{E} _{x\sim \mathcal{D} ,y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right) \cdot Q^{\pi _{\theta _k}}\left( s_t,a_t \right)} \right]\\
	\,\,&=-\textcolor{red}{\frac{1}{\sigma _k}}\nabla _{\theta}\mathcal{J} \left( \theta _k \right)\\
\end{aligned}
$$

### **③ RLOO**

The **token advantage** is

$$
\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) =r\left( x_i,y_{ij} \right) -\frac{1}{M-1}\sum_{l\ne j}{r\left( x_i,y_{il} \right)}.
$$

The **expectation of token advantage** is 

$$
\begin{aligned}	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] &=\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ r\left( x_i,y_{ij} \right) -\frac{1}{M-1}\sum_{l\ne j}{r\left( x_i,y_{il} \right)}|\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]\\	\,\,&=\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ r\left( x_i,y_{ij} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] -\frac{1}{M-1}\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \sum_{l\ne j}{r\left( x_i,y_{il} \right)}|x_i \right]\\	\,\,&=Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\textcolor{red}{V^{\pi _{\theta}}\left( x_i \right)}\\\end{aligned}
$$

The **expectation  of policy loss gradient** is  

$$
\begin{aligned}
	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) \right] &=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]} \right]}} \right]\\
	\,\,&=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \left( Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\textcolor{red}{V^{\pi _{\theta}}\left( x_i \right) }\right)} \right]}} \right]\\
	\,\,&=-\mathbb{E} _{x\sim \mathcal{D} ,y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right) \cdot Q^{\pi _{\theta _k}}\left( s_t,a_t \right)} \right]\\
	\,\,&=-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)\\
\end{aligned}
$$

### **④ REMAX**

The **token advantage** is $\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) =r\left( x_i,y_{ij} \right) -r\left( x_i,\hat{y}_i \right)$, where $\hat{y}_i$ is sampled from greedy policy $\hat{\pi}_{\theta_k}$. 

The **expectation of token advantage** is 

$$
\begin{aligned}
	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] &=\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ r\left( x_i,y_{ij} \right) -r\left( x_i,\hat{y}_i \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]\\
	\,\,&=\mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right)}\left[ r\left( x_i,y \right) \right] -r\left( x_i,\hat{y}_i \right)\\
	\,&=Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\textcolor{red}{V^{\hat{\pi}_{\theta _k}}\left( x_i \right)}\\
\end{aligned}
$$

The **expectation  of policy loss gradient** is  

$$
\begin{aligned}	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) \right] &=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]} \right]}} \right]\\	\,\,&=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \left( Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\textcolor{red}{V^{\hat{\pi}_{\theta _k}}\left( x_i \right)} \right)} \right]}} \right]\\	\,\,&=-\mathbb{E} _{x\sim \mathcal{D} ,y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right) \cdot Q^{\pi _{\theta _k}}\left( s_t,a_t \right)} \right]\\	\,\,&=-\nabla _{\theta}\mathcal{J} \left( \theta _k \right)\\\end{aligned}
$$

### **⑤ PPO / ORZ**

The **token advantage** is

$$
\begin{align*}\hat{A}_k\left( s_t^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) &=\sum_{t^{\prime}=t}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}\delta _{t^{\prime}}^{\left( ij \right)}}\\\,\,                &=\sum_{t^{\prime}=t}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}\left( \mathbb{I} \left\{ t^{\prime}=T_{ij}-1 \right\} r\left( x_i,y_{ij} \right) +\gamma V_{\phi}\left( s_{t^{\prime}+1}^{\left( ij \right)} \right) -V_{\phi}\left( s_{t^{\prime}}^{\left( ij \right)} \right) \right)}\\\,\,                &=\sum_{t^{\prime}=t}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}\mathbb{I} \left\{ t=T_{ij}-1 \right\} r\left( x_i,y_{ij} \right)}+\sum_{t^{\prime}=t}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}\gamma V_{\phi}\left( s_{t^{\prime}+1}^{\left( ij \right)} \right)}-\sum_{t^{\prime}=t}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}V_{\phi}\left( s_{t^{\prime}}^{\left( ij \right)} \right)}\\\,\,                &=\left( \lambda \gamma \right) ^{T_{ij}-t-1}r\left( x_i,y_{ij} \right) +\frac{1}{\lambda}\sum_{t^{\prime}=t}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}+1-t}V_{\phi}\left( s_{t^{\prime}+1}^{\left( ij \right)} \right)}-\sum_{t^{\prime}=t}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}V_{\phi}\left( s_{t^{\prime}}^{\left( ij \right)} \right)}\\\,\,                &=\left( \lambda \gamma \right) ^{T_{ij}-t-1}r\left( x_i,y_{ij} \right) +\frac{1}{\lambda}\sum_{t^{\prime}=t+1}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}V_{\phi}\left( s_{t^{\prime}}^{\left( ij \right)} \right)}-\sum_{t^{\prime}=t}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}V_{\phi}\left( s_{t^{\prime}}^{\left( ij \right)} \right)}\\\,\,                &=\left( \left( \lambda \gamma \right) ^{T_{ij}-t-1}r\left( x_i,y_{ij} \right) +\left( \frac{1}{\lambda}-1 \right) \sum_{t^{\prime}=t+1}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}V_{\phi}\left( s_{t^{\prime}}^{\left( ij \right)} \right)} \right) -V_{\phi}\left( s_{t}^{\left( ij \right)} \right).\end{align*}
$$

Suppose $V_\phi=V^{\pi_{\theta_k}}$. Note that, for any $i\in[N],j\in[M]\text{ and } t<t^\prime < T_{ij}$,  

$$
\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ r\left( x_i,y_{ij} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] =Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) 
$$

and 

$$
\mathbb{E} \left[ V_{\phi}\left( s_{t^{\prime}}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] =\mathbb{E} \left[ V^{\pi _{\theta _k}}\left( s_{t^{\prime}}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] =Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right).
$$

Hence the **expectation of token advantage** is 

$$
\begin{align*}\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] &=\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \left( \left( \lambda \gamma \right) ^{T_{ij}-t-1}r\left( x_i,y_{ij} \right) +\left( \frac{1}{\lambda}-1 \right) \sum_{t^{\prime}=t+1}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}V_{\phi}\left( s_{t}^{\left( ij \right)} \right)} \right) -V_{\phi}\left( s_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] \\\,\,                                      &=\left( \left( \lambda \gamma \right) ^{T_{ij}-t-1}+\left( \frac{1}{\lambda}-1 \right) \sum_{t^{\prime}=t+1}^{T_{ij}-1}{\left( \lambda \gamma \right) ^{t^{\prime}-t}} \right) Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -V^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)} \right) \\\,\,                                      &=\left( \left( \lambda \gamma \right) ^{T_{ij}-t-1}+\left( \frac{1}{\lambda}-1 \right) \frac{\lambda \gamma -\left( \lambda \gamma \right) ^{T_{ij}-t}}{1-\lambda \gamma} \right) Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -V^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)} \right) \\\,\,                                      &=\left( 1-\frac{1-\gamma}{1-\lambda \gamma}\left( 1-\left( \lambda \gamma \right) ^{T_{ij}-1-t} \right) \right) Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -V^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)} \right) \\\,\,                                      &:=\textcolor{red}{A_{\lambda ,\gamma}^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)}, \end{align*}
$$

The **expectation  of policy loss gradient** is  

$$
\begin{aligned}
	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) \right] &=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)} \right]}} \right]\\
	\,\,&=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \textcolor{red}{A_{\lambda ,\gamma}^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right)}} \right]}} \right]\\
	\,\,&=-\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right) \cdot \textcolor{red}{A_{\lambda ,\gamma}^{\pi _{\theta _k}}\left(s_t,a_t \right)}} \right] .\\
\end{aligned}
$$

When $\gamma=1$, one has 

$$
A_{\lambda ,\gamma}^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) =Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -V^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)} \right) =A^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) 
$$

and

$$
\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) \right] =-\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right) \cdot A^{\pi _{\theta _k}}\left( s_t,a_t \right)} \right] =-\nabla _{\theta}\mathcal{J} \left( \theta _k \right).
$$

### **⑤ GRPO  ([original version](https://arxiv.org/pdf/2402.03300))**

The **token advantage** is

$$
\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) =\frac{1}{T_{ij}}\left[ \frac{r\left( x_i,y_{ij} \right) -\text{mean}(\mathbb{R}_i)}{\text{std}(\mathbb{R}_i)} \right].
$$

we first denote 

$$
\left( 1 \right) p_{\theta _k}\left( x \right) :=\mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ r\left( x,y \right) \right] \qquad \left( 2 \right) \hat{p}_i:=\frac{1}{M}\sum_{j=1}^M{r\left( x_i,y_{ij} \right)}=\mathrm{mean}\left( \mathbb{R} _i \right) \qquad \left( 3 \right) \hat{p}_{i,-j}:=\frac{1}{M-1}\sum_{l=1,l\ne j}^M{r\left( x_i,y_{il} \right)}.
$$

It’s easy to show that 

$$
\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) =\frac{1}{T_{ij}}\left[ \frac{r\left( x_i,y_{ij} \right) -\hat{p}_i}{\sqrt{\hat{p}_i\left( 1-\hat{p}_i \right)}} \right].
$$

For the **expectation of token advantage,**  notice that for arbitrary  $i\in[N],j\in [M]$, 

$$
\hat{p}_i=\frac{1}{M}r\left( x_i,y_{ij} \right) +\frac{M-1}{M}\hat{p}_{i,-j}.
$$

Thus 

$$
\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) =\frac{1}{T_{ij}}\left[ \frac{r\left( x_i,y_{ij} \right) -\hat{p}_i}{\sqrt{\hat{p}_i\left( 1-\hat{p}_i \right)}} \right] =\frac{1}{T_{ij}}\left[ \frac{\left( 1-\frac{1}{M} \right) \left( r\left( x_i,y_{ij} \right) -\hat{p}_{i,-j} \right)}{\sqrt{\left( \frac{1}{M}r\left( x_i,y_{ij} \right) +\frac{M-1}{M}\hat{p}_{i,-j} \right) \left( \frac{1}{M}\left( 1-r\left( x_i,y_{ij} \right) \right) +\frac{M-1}{M}\left( 1-\hat{p}_{i,-j} \right) \right)}} \right].
$$

Note that $\hat{p}_{i,-j}$   converges to  $p_{\theta_k}(x_i)$ almost surely and $r$ is a bound variable. Thus 

$$
\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] \rightarrow \frac{1}{\sqrt{p_{\theta _k}\left( x_i \right) \left( 1-p_{\theta _k}\left( x_i \right) \right)}}\textcolor{red}{\mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x_i \right)}\left[ \frac{1}{T}\left( r\left( x_i,y \right) -p_{\theta _k}\left( x \right) \right) \right] }\,\,\text{a.s},
$$

as $M$ is large enough.  However, due to the length normalization term $\frac{1}{T}$,  the red term is hard to analyze. 

The **expectation  of policy loss gradient** is  

$$
\begin{aligned}
	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) \right] &=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \frac{1}{T_{ij}}\left[ \frac{r\left( x_i,y_{ij} \right) -\hat{p}_i}{\sqrt{\hat{p}_i\left( 1-\hat{p}_i \right)}} \right]} \right]}} \right]\\
	&=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \nabla _{\theta}\log \pi _{\theta _k}\left( y_{ij}|x_i \right) \cdot \frac{1}{T_{ij}}\frac{r\left( x_i,y_{ij} \right) -\textcolor{red}{p_{\theta _k}\left( x_i \right)}}{\textcolor{red}{\sqrt{p_{\theta _k}\left( x_i \right) \left( 1-p_{\theta _k}\left( x_i \right) \right)}}} \right]}} \right]\\
	&\rightarrow -\mathbb{E} _{x\sim \mathcal{D}}\left[ \mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \frac{\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right) \cdot \left( r\left( x,y \right) -\textcolor{red}{p_{\theta _k}\left( x \right)} \right)}{\textcolor{red}{T\sqrt{p_{\theta _k}\left( x_i \right) \left( 1-p_{\theta _k}\left( x_i \right) \right)}}} \right] \right]\\
	&=-\mathbb{E} _{x\sim \mathcal{D}}\left[ \mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \frac{\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right) \cdot r\left( x,y \right)}{\textcolor{red}{T\sqrt{p_{\theta _k}\left( x_i \right) \left( 1-p_{\theta _k}\left( x_i \right) \right)}}} \right] \right]\\
\end{aligned}
$$

### **⑥ GRPO  ([R1 version](https://arxiv.org/abs/2501.12948))**

The **token advantage** is

$$
\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) = \frac{r\left( x_i,y_{ij} \right) -\text{mean}(\mathbb{R}_i)}{\text{std}(\mathbb{R}_i)} 
$$

The analysis is similar to the analysis in *⑤ **GRPO  (original version).*  one can easily show that**

$$
\begin{align*}\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] &\rightarrow \frac{1}{\sqrt{p_{\theta _k}\left( x_i \right) \left( 1-p_{\theta _k}\left( x_i \right) \right)}}\mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x_i \right)}\left[ r\left( x_i,y \right) -p_{\theta _k}\left( x_i \right) \right] \,
\\
\,\,                                      &=\frac{Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\textcolor{red}{p_{\theta _k}\left( x_i \right)}}{\textcolor{red}{\sqrt{p_{\theta _k}\left( x_i \right) \left( 1-p_{\theta _k}\left( x_i \right) \right)}}}\end{align*}
$$

as $M$ is large enough.   And the **expectation  of policy loss gradient** is  

$$
\begin{aligned}
	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) \right] &=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]} \right]}} \right]\\
	\,\,&\rightarrow -\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \frac{Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\textcolor{red}{p_{\theta _k}\left( x_i \right)}}{\textcolor{red}{\sqrt{p_{\theta _k}\left( x_i \right) \left( 1-p_{\theta _k}\left( x_i \right) \right)}}}} \right]}} \right]\\
	\,\,&=-\mathbb{E} _{x\sim \mathcal{D} ,y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \frac{\nabla _{\theta}\log \pi _{\theta _k}\left( y|x \right) r\left( x,y \right)}{\textcolor{red}{\sqrt{p_{\theta _k}\left( x \right) \left( 1-p_{\theta _k}\left( x \right) \right)}}} \right]\\
\end{aligned}
$$

### **⑦ DAPO**

The **token advantage** is

$$
\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) =\frac{1}{\hat{T}_i}\left[ \frac{r\left( x_i,y_{ij} \right) -\hat{p}_i}{\sqrt{\hat{p}_i(1-\hat{p}_i)}}   \right],  \text{where } \hat{T}_i=\frac{1}{M}\sum_{j=1}^M{T_{ij}}.
$$

We define the expected average response length of prompt $x$ by $\pi_\theta$ is  $T_{\theta}\left( x \right) :=\mathbb{E} _{y\sim \pi _{\theta}\left( \cdot |x \right)}\left[ T \right]$. Thus when M is large enough,  one has $\hat{T}_i\rightarrow T_{\theta_k}\left(x_i \right), \text{a.s}.$
  Use a similar analysis argument in GRPO,  one can shows that 

$$
\begin{align*}\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]                                    \rightarrow \frac{Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\textcolor{red}{p_{\theta _k}\left( x_i \right)}}{\textcolor{red}{T_{\theta_k}(x_i)\sqrt{p_{\theta _k}\left( x_i \right) \left( 1-p_{\theta _k}\left( x_i \right) \right)}}}\end{align*},
$$

and 

$$
\begin{aligned}
	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) \right] \rightarrow -\mathbb{E} _{x\sim \mathcal{D}}\left[ \mathbb{E} _{y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \frac{\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right) \cdot r\left( x,y \right)}{\textcolor{red}{T_{\theta_k}(x)\sqrt{p_{\theta _k}\left( x \right) \left( 1-p_{\theta _k}\left( x \right) \right)}}} \right] \right].\\
\end{aligned}
$$

### **⑧ DR.GRPO**

The **token advantage** is

$$
\hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) =r\left( x_i,y_{ij} \right) -\frac{1}{M}\sum_{j=1}^{M}{r\left( x_i,y_{ij} \right)}.
$$

The **expectation  of policy loss gradient** is  

$$
\begin{aligned}
	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] &=\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ r\left( x_i,y_{ij} \right) -\frac{1}{M}\sum_{j=1}^M{r\left( x_i,y_{ij} \right)}|\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]\\
	\,\,&=\left( 1-\frac{1}{M} \right) \mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ r\left( x_i,y_{ij} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right] -\frac{1}{M}\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \sum_{l\ne j}{r\left( x_i,y_{il} \right)}|x_i \right]\\
	\,\,&=\textcolor{red}{\left( 1-\frac{1}{M} \right)} \left( Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\textcolor{red}{V^{\pi _{\theta _k}}\left( x_i \right)}\right)\\
\end{aligned}
$$

The **expectation  of policy loss gradient** is  

$$
\begin{aligned}
	\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \nabla _{\theta}\hat{\mathcal{L}}_k\left( \theta _k \right) \right] &=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \hat{A}_k\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) |\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) \right]} \right]}} \right]\\
	\,\,&=-\mathbb{E} _{\mathbb{X} ,\mathbb{Y}}\left[ \frac{1}{NM}\sum_{i=1}^N{\sum_{j=1}^M{\left[ \sum_{t=0}^{T_{ij}-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_{t}^{\left( ij \right)}|s_{t}^{\left( ij \right)} \right) \cdot \textcolor{red}{(1-\frac{1}{M})} \cdot\left( Q^{\pi _{\theta _k}}\left( s_{t}^{\left( ij \right)},a_{t}^{\left( ij \right)} \right) -\textcolor{red}{V^{\pi _{\theta}}\left( x_i \right) }\right)} \right]}} \right]\\
	\,\,&=- \textcolor{red}{(1-\frac{1}{M})} \cdot\mathbb{E} _{x\sim \mathcal{D} ,y\sim \pi _{\theta _k}\left( \cdot |x \right)}\left[ \sum_{t=0}^{T-1}{\nabla _{\theta}\log \pi _{\theta _k}\left( a_t|s_t \right) \cdot Q^{\pi _{\theta _k}}\left( s_t,a_t \right)} \right]\\
	\,\,&=-\textcolor{red}{(1-\frac{1}{M})} \cdot\nabla _{\theta}\mathcal{J} \left( \theta _k \right)\\
\end{aligned}
$$