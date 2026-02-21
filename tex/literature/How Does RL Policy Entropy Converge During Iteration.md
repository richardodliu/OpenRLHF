# 1. MOTIVATION

在强化学习中，策略熵顾名思义，即策略的信息熵。我们考虑离散的动作空间 $\mathcal{A}$ , 策略 $\pi$ 在状态 $s$ 处的信息熵定义为 

$$
\mathcal{H} \left( \pi \left( \cdot |s \right) \right) :=\mathbb{E} _{a\sim \pi \left( \cdot |s \right)}\left[ -\log \pi \left( a|s \right) \right] \\
$$

策略信息熵衡量了策略在状态s处的混乱与有序情况。当策略熵为0时，代表策略完全收敛到了一个固定的动作 $a$ 上，如果策略熵很大，代表策略的动作选取比较随机，更可能探索到多样性的轨迹。因此最大熵RL通常找到一个策略 $\pi$ 使得最大化累积奖励的同时尽可能最大化策略的熵，即

$$
\underset{\pi}{\max}\,\,\left\{ V_{\tau}^{\pi}\left( \mu \right) :=\mathbb{E} _{\pi}\left[ \sum_{t=0}^{\infty}{\gamma ^t\left( r\left( s_t,a_t \right) +\tau \mathcal{H} \left( \pi \left( \cdot |s_t \right) \right) \right)}|s_0\sim \mu \right] \right\}   \\
$$

一般而言，在RL的迭代过程中，如果策略存在一个明确的优化方向，通常伴随着策略熵的衰减，即策略倾向于收敛到固定的动作空间上。因此我们可以用策略熵来判断模型的收敛情况，如果策略熵衰减到0附近，代表策略完全收敛。**那么自然引出一个问题是，什么时候策略的熵减少？什么时候策略熵会增加？**本文旨在尝试提供一些理论上的insights来解答这个问题。

# 2. SETTINGS

在下文中，我们考虑求解策略最大化折扣累计奖励，即

$\underset{\pi}{\max}\,\,\left\{ V^{\pi}\left( \mu \right) :=\mathbb{E} _{\pi}\left[ \sum_{t=0}^{\infty}{\gamma ^tr\left( s_t,a_t \right)}|s_0\sim \mu \right] \right\} \\$我们 $\mu$ 是初始状态 $s_0$的分布。我们也定义 $Q$ 函数为

$$
Q^{\pi}\left( s,a \right) :=\mathbb{E} _{\pi}\left[ \sum_{t=0}^{\infty}{\gamma ^tr\left( s_t,a_t \right)}|s_0=s,a_0=a \right]  \\
$$

我们定义Advantage 函数$A^\pi(s,a):=Q^\pi(s,a)-V^\pi(s)\\$

为了方便起见，我们考虑策略是简单的softmax策略，即$\pi _{\theta}\left( a|s \right) =\frac{\exp \left\{ \theta _{s,a} \right\}}{\sum_{a^{\prime}\in \mathcal{A}}{\exp \left\{ \theta _{s,a^{\prime}} \right\}}} \\$其中状态-动作对上有一个参数 $\theta_{s,a}$ 代表logitis。在深度强化学习中 (例如LLM的RLHF), 我们使用neural softmax policy, 即logits通过神经网络来学习, $\theta_{s,a}=f_\theta(s,a)$, 其中 $f_\theta$ 是参数为 $\theta$的神经网络。由于涉及到神经网络往往涉及到更复杂的推导，为了比较简单说明算法背后的intuition，我们在此不考虑neural softmax policy。

我们考虑使用Natural Policy Gradient (NPG,[1]) 算法来更新策略。对于iteration $k$ , 给定任意基策略 $\pi_k$, 我们的新策略 $\pi_{k+1}$ 通过求解下面问题所得 : 

$$
\forall s:   \pi_{k+1}\left( \cdot |s \right) \in \underset{p\in \Delta \left( \mathcal{A} \right)}{\mathrm{arg}\max}\,\,\mathbb{E} _{a\sim p}\left[ Q^{\pi_k}\left( s,a \right) \right] -\frac{1}{\eta}\mathrm{KL}\left( p,\pi_k \left( \cdot |s \right) \right)  \\
$$

其中 $\eta$是KL约束系数。根据 [1] 我们可以知道，NPG 等价于策略空间上执行以下迭代:

$$
\begin{align*} \forall s\in \mathcal{S} ,a\in \mathcal{A} :\qquad \pi _{k+1}\left( a|s \right) &\propto \pi _k\left( a|s \right) \exp \left\{ \eta A^{\pi}\left( s,a \right) \right\} =\frac{\pi _k\left( a|s \right) \exp \left\{ \eta A^{\pi}\left( s,a \right) \right\}}{\mathbb{E} _{a^{\prime}\sim \pi _k\left( \cdot |s \right)}\left[ \exp \left\{ \eta A^{\pi}\left( s,a^{\prime} \right) \right\} \right]} \end{align*} \\
$$

或者等价的，在策略空间上:

$$
\forall s\in \mathcal{S} ,a\in \mathcal{A} :  \theta _{s,a}^{k+1}=\theta _{s,a}^{k}+\eta A^{\pi}\left( s,a \right)  \\
$$

**实际上，RLHF目标函数的求解，可以看成bandit问题上，执行单步NPG更新。**

# 3. RESULTS

下面我们来具体研究策略熵是如何变动的。由于我们的策略是参数化策略 $\pi_\theta$，因此策略熵也可以看成一个关于参数 $\theta$ 的函数。我们记

$$
\mathcal{H} \left( \theta |s \right) :=\mathcal{H} \left( \pi _{\theta}\left( \cdot |s \right) \right) \\
$$

于是我们知道，当容易知道$\eta$比较小的时候, 根据泰勒展开$\mathcal{H} \left( \theta ^{k+1}|s \right) \approx \mathcal{H} \left( \theta ^k|s \right) +\left< \nabla _{\theta}\mathcal{H} \left( \theta ^k|s \right) ,\theta ^{k+1}-\theta ^k \right>  \\$即状态 $s$ 处的策略熵的增减，取决于 策略熵梯度于参数的内积，即 $\left< \nabla _{\theta}\mathcal{H} \left( \theta ^k|s \right) ,\theta ^{k+1}-\theta ^k \right>$ , 下面我们来看一项这一项具体是什么。我们首先来获取策略熵的梯度向量

$$
\begin{align*} \nabla _{\theta}\mathcal{H} \left( \theta |s \right) &=\nabla _{\theta}\mathcal{H} \left( \pi _{\theta}\left( \cdot |s \right) \right)  \\ \,\,            &=\nabla _{\theta}\left( -\mathbb{E} _{a\sim \pi _{\theta}\left( \cdot |s \right)}\left[ \log \pi _{\theta}\left( a|s \right) \right] \right)  \\ \,\,            &=-\mathbb{E} _{a\sim \pi _{\theta}\left( \cdot |s \right)}\left[ \nabla _{\theta}\log \pi _{\theta}\left( a|s \right) +\log \pi _{\theta}\left( a|s \right) \nabla _{\theta}\log \pi _{\theta}\left( a|s \right) \right]  \\ \,\,            &=-\mathbb{E} _{a\sim \pi _{\theta}\left( \cdot |s \right)}\left[ \log \pi _{\theta}\left( a|s \right) \nabla _{\theta}\log \pi _{\theta}\left( a|s \right) \right]  \end{align*}  \\
$$

于是我们有 $\begin{align} \left< \nabla _{\theta}\mathcal{H} \left( \theta ^k|s \right) ,\theta ^{k+1}-\theta ^k \right> &=-\left< \mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \nabla _{\theta}\log \pi _{k}\left( a|s \right) \right] ,\theta ^{k+1}-\theta ^k \right>  \\ \,\,                              &=-\mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \left< \nabla _{\theta}\log \pi _{k}\left( a|s \right) ,\theta ^{k+1}-\theta ^k \right> \right]  \\ \,\,                              &=-\mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \sum_{s^{\prime}\in \mathcal{S} ,a^{\prime}\in \mathcal{A}}{\frac{\partial \log \pi _{\theta}\left( a|s \right)}{\partial \theta _{s^{\prime},a^{\prime}}}\cdot \left( \theta _{s^\prime,a^\prime}^{k+1}-\theta _{s^\prime,a^\prime}^{k} \right)} \right]  \\                               &=-\sum_{s^{\prime}\in \mathcal{S} ,a^{\prime}\in \mathcal{A}}{\mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \cdot \frac{\partial \log \pi _{\theta}\left( a|s \right)}{\partial \theta _{s^{\prime},a^{\prime}}}\cdot \left( \theta _{s^{\prime},a^{\prime}}^{k+1}-\theta _{s^{\prime},a^{\prime}}^{k} \right) \right]} \\ \,\,                             &=-\sum_{s^{\prime}\in \mathcal{S} ,a^{\prime}\in \mathcal{A}}{\left( \theta _{s^{\prime},a^{\prime}}^{k+1}-\theta _{s^{\prime},a^{\prime}}^{k} \right) \cdot \underset{I\left( s^{\prime},a^{\prime} \right)}{\underbrace{\mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \cdot \frac{\partial \log \pi _{\theta}\left( a|s \right)}{\partial \theta _{s^{\prime},a^{\prime}}} \right] }}} \end{align} \\$注意到对于softmax 策略$\begin{align} \frac{\partial \log \pi _{\theta}\left( a|s \right)}{\partial \theta _{s^{\prime},a^{\prime}}}&=\frac{\partial}{\partial \theta _{s^{\prime},a^{\prime}}}\left( \theta _{s,a}-\log \left( \sum_{a\in \mathcal{A}}{\exp \left\{ \theta _{s,a} \right\}} \right) \right)  \\ \,\,                &=\mathbf{1}\left\{ s=s^{\prime} \right\} \left( \mathbf{1}\left\{ a=a^{\prime} \right\} -\frac{\exp \left\{ \theta _{s,a^{\prime}} \right\}}{\sum_{a\in \mathcal{A}}{\exp \left\{ \theta _{s,a} \right\}}} \right)  \\ \,\,                &=\mathbf{1}\left\{ s=s^{\prime} \right\} \left( \mathbf{1}\left\{ a=a^{\prime} \right\} -\pi _{\theta}\left( a^{\prime}|s \right) \right)  \end{align} \\$于是

$\begin{align*} I\left( s^{\prime},a^{\prime} \right) &=\mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \cdot \frac{\partial \log \pi _{\theta}\left( a|s \right)}{\partial \theta _{s^{\prime},a^{\prime}}} \right]  \\ \,\,         &=\mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \cdot \mathbf{1}\left\{ s=s^{\prime} \right\} \cdot \left( \mathbf{1}\left\{ a=a^{\prime} \right\} -\pi _{k}\left( a^{\prime}|s \right) \right) \right]  \\ \,\,         &=\mathbf{1}\left\{ s=s^{\prime} \right\} \cdot \mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \cdot \left( \mathbf{1}\left\{ a=a^{\prime} \right\} -\pi _{k}\left( a^{\prime}|s \right) \right) \right]  \\ \,\,         &=\mathbf{1}\left\{ s=s^{\prime} \right\} \cdot \pi _{k}\left( a^{\prime}|s \right) \left( \log \pi _{k}\left( a^{\prime}|s \right) -\mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \right] \right)  \end{align*} \\$插入 $d_{\mathcal{H}}(s):=\left< \nabla _{\theta}\mathcal{H} \left( \theta ^k|s \right) ,\theta ^{k+1}-\theta ^k \right>$ 中我们可以得到

$\begin{align*} d_{\mathcal{H}}(s)&=-\sum_{s^{\prime}\in \mathcal{S} ,a^{\prime}\in \mathcal{A}}{\left( \theta _{s,a}^{k+1}-\theta _{s,a}^{k} \right) \cdot I\left( s^{\prime},a^{\prime} \right)} \\ \,\,                              &=-\sum_{s^{\prime}\in \mathcal{S} ,a^{\prime}\in \mathcal{A}}{\left( \theta _{s^{\prime},a^{\prime}}^{k+1}-\theta _{s^{\prime},a^{\prime}}^{k} \right) \cdot \mathbf{1}\left\{ s=s^{\prime} \right\} \cdot \pi _{k}\left( a^{\prime}|s \right) \left( \log \pi _{k}\left( a^{\prime}|s \right) -\mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \right] \right)} \\ \,\,                              &=-\sum_{a^{\prime}\in \mathcal{A}}{\left( \theta _{s,a^{\prime}}^{k+1}-\theta _{s,a^{\prime}}^{k} \right) \cdot \pi _{k}\left( a^{\prime}|s \right) \left( \log \pi _{k}\left( a^{\prime}|s \right) -\mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \right] \right)} \\ \,\,                              &=-\mathbb{E} _{a^{\prime}\sim \pi _{k}\left( \cdot |s \right)}\left[ \left( \theta _{s,a^{\prime}}^{k+1}-\theta _{s,a^{\prime}}^{k} \right) \left( \log \pi _{k}\left( a^{\prime}|s \right) -\mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \right] \right) \right]  \\ \,\,                              &=-\mathbb{E} _{a^{\prime}\sim \pi _{k}\left( \cdot |s \right)}\left[ \left( \theta _{s,a^{\prime}}^{k+1}-\theta _{s,a^{\prime}}^{k} \right) \log \pi _{k}\left( a^{\prime}|s \right) \right]  \\ \,\,                              &+\mathbb{E} _{a^{\prime}\sim \pi _{k}\left( \cdot |s \right)}\left[ \left( \theta _{s,a^{\prime}}^{k+1}-\theta _{s,a^{\prime}}^{k} \right) \right] \mathbb{E} _{a\sim \pi _{k}\left( \cdot |s \right)}\left[ \log \pi _{k}\left( a|s \right) \right]  \\ \,\,                              &=-\mathrm{Cov}_{a^{\prime}\sim \pi _{k}\left( \cdot |s \right)}\left( \log \pi _{k}\left( a^{\prime}|s \right) ,\theta _{s,a^{\prime}}^{k+1}-\theta _{s,a^{\prime}}^{k} \right)  \end{align*} \\$ 于是再插入回泰勒展开中，我们可以得到:

$\begin{align} \mathcal{H} \left( \theta ^{k+1}|s \right) -\mathcal{H} \left( \theta ^k|s \right) &\approx \left< \nabla _{\theta}\mathcal{H} \left( \theta ^k|s \right) ,\theta ^{k+1}-\theta ^k \right>  \\ \,\,                           &=-\mathrm{Cov}_{a\sim \pi _k\left( \cdot |s \right)}\left( \log \pi _k\left( a|s \right) ,\theta _{s,a}^{k+1}-\theta _{s,a}^{k} \right)  \\ \,\,                           &=-\eta \cdot \mathrm{Cov}_{a\sim \pi _k\left( \cdot |s \right)}\left( \log \pi _k\left( a|s \right) ,A^{\pi _k}\left( s,a \right) \right)  \end{align} \\$ 也就是说，对于softmax 策略而言， 状态 $s$ 处的策略熵的变动实际上取决当前策略的对数概率 $\log\pi_k(a|s)$ 与 每个动作上的策略logits的变化量 $\theta^{k+1}_{s,a}-\theta^{k}_{s,a}$ 的协方差！对于NPG这样更具体的算法而言，等于对数概率 与 advantage 函数的协方差！我们考虑整个策略轨迹上的动作策略熵，我们定义

$$
\mathcal{H} \left( \pi \right) :=\mathbb{E} _{s\sim d_{\mu}^{\pi}}\left[ \mathcal{H} \left( \pi \left( \cdot |s \right) \right) \right]  \\
$$

于是

$$
\begin{align} \mathcal{H} \left( \pi _{k+1} \right) -\mathcal{H} \left( \pi _k \right) &=\mathbb{E} _{s\sim d_{\mu}^{\pi _{k+1}}}\left[ \mathcal{H} \left( \pi _{k+1}\left( \cdot |s \right) \right) \right] -\mathbb{E} _{s\sim d_{\mu}^{\pi _k}}\left[ \mathcal{H} \left( \pi _k\left( \cdot |s \right) \right) \right]  \\ \,\,                      &\approx \mathbb{E} _{s\sim d_{\mu}^{\pi _k}}\left[ \mathcal{H} \left( \pi _{k+1}\left( \cdot |s \right) \right) \right] -\mathbb{E} _{s\sim d_{\mu}^{\pi _k}}\left[ \mathcal{H} \left( \pi _k\left( \cdot |s \right) \right) \right]  \\ \,\,                      &=\mathbb{E} _{s\sim d_{\mu}^{\pi _k}}\left[ \mathcal{H} \left( \pi _{k+1}\left( \cdot |s \right) \right) -\mathcal{H} \left( \pi _k\left( \cdot |s \right) \right) \right]  \\ \,\,                      &\approx \mathbb{E} _{s\sim d_{\mu}^{\pi _k}}\left[ \mathrm{Cov}_{a\sim \pi _k\left( \cdot |s \right)}\left( \log \pi _k\left( a|s \right) ,\theta _{s,a}^{k+1}-\theta _{s,a}^{k} \right) \right]  \\ \,\,                      &=-\eta \cdot \mathbb{E} _{s\sim d_{\mu}^{\pi _k}}\left[ \mathrm{Cov}_{a\sim \pi _k\left( \cdot |s \right)}\left( \log \pi _k\left( a|s \right) ,A^{\pi _k}\left( s,a \right) \right) \right]  \end{align} \\
$$

其中从第一行到第二行，第一项的期望从 $d^{\pi_{k+1}}_\mu $ 换成了 $d^{\pi_k}_\mu $ , 这是因为访问概率误差 $d^{\pi_{k+1}}_\mu-d^{\pi_{k}}_\mu$ 是关于策略向量误差 $\pi_{k+1}-\pi_k$ 的二阶项[2], 因此考虑一阶近似下，不会带来很大误差。上述的公式这意味着： 

1. 如果当前策略轨迹上，概率比较大的动作上，advantage也大，那么策略熵会倾向于减少，从而进一步减少策略熵
2. 如果当前策略轨迹上，概率比较大的动作上，advantage比较小，advantage较大的动作，概率比较小，那么会抑制策略熵的衰减，增大策略熵。
3. 这意味着，对于同一个状态 $s$ , 如果某个动作频繁的出现较大的advantage，那么会导致策略熵的降低并收敛！

这非常符合我们对策略熵运行方向判断的直觉，只不过我们给出了一个比较美妙优雅的表达式！

[1] Agarwal, A., Kakade, S. M., Lee, J. D., and Mahajan, G. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research, 22(98):1–76, 2021

[2] Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. (2015). Trust region policy optimization.  International Conference on Machine Learning,International Conference on Machine Learning.