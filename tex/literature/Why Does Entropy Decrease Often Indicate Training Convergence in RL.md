# RL训练中为什么熵减往往意味着训练收敛?

**Author:** skydownacai

**Date:** 2025-09-30

**Link:** https://zhuanlan.zhihu.com/p/1950579532802270647



最近半年以来，有关于RL+Entropy的研究非常的多。对于离散的动作空间 $\mathcal{A}$ , 策略 $\pi$ 在状态 $s$ 处的entropy为 

$$
\mathcal{H} \left( \pi \left( \cdot |s \right) \right) :=\mathbb{E} _{a\sim \pi \left( \cdot |s \right)}\left[ -\log \pi \left( a|s \right) \right] \\
$$

直观上而言，entropy收敛到0，意味着策略极化到某一个确定性的解上，并且不容易跳出来，这也是“收敛”两字的蕴含之意。但一个问题是，到底背后发生了什么，导致了该现象发生？对于softmax policy，即$\pi _{z_\theta}\left( a|s \right) =\frac{\exp \left\{ z_{\theta}(s,a) \right\}}{\sum_{a^{\prime}\in \mathcal{A}}{\exp \left\{ z_{\theta}(s,a^\prime) \right\}}} \\$

其中 $z_\theta(s,a)$ 为(s,a)处的关于参数\theta的logits函数。实际上，我们有以下两个理论可以说明，为什么entropy 收敛到0，模型训练往往意味着收敛  

## 1. Entropy 衰减，策略梯度幅度衰减

首先第一个理论结果我们已经写在[EMPG论文](https://arxiv.org/pdf/2509.09265)中。

![](https://pica.zhimg.com/v2-dcbc0c07a4797ac8869c53bdc71e20a2_r.jpg)

这个理论结果说明了，对于softmax 策略，状态s处，期望下关于logits的策略梯度范数 直接等于1-exp{-H2} 。Renyi-2 entropy, 即H2, 越小，例如接近0， 那么 期望的策略梯度范数也接近0。注意到我们一般讨论的entropy为信息熵，实际上是Renyi-1 entropy。Renyi entropy在order 上存在单调性，即 Renyi-1 entropy 大于 Renyi-2 entropy。从而上述的理论结果可以推到下面的不等式：

$\begin{align} \mathbb{E} _{a\sim \pi _{\theta}\left( \cdot |s \right)}\left[ \left\| \nabla _{z_{\theta}\left( s \right)}\log \pi _{\theta}\left( a|s \right) \right\| _{2}^{2} \right] \le 1-\exp \left( -H\left( \pi_\theta \left( \cdot |s \right) \right) \right) \end{align} \\$**实际上，这个定理的主要推导并不复杂。其背后的原理是：高概率action 的策略梯度范数更小。而entropy越低，更容易产生高概率的action，因此导致期望的梯度范数衰减。**

## 2. Entropy 衰减，策略Reverse KL移动幅度上界衰减

实际上，我们有下面一个定理，可以进一步佐证这个事情。假设我们有一个基础策略 $\pi_\theta$, 经过某个算法更新后(例如PG)，得到新的策略 $\pi_{\theta^+}$ 。我们考虑状态s处的更新前后的logits向量的差，即

$\Delta_s:=z_{\theta^+}(s)-z_{\theta}(s) \\$那我们可以得到如下的不等式证明：

$$
\mathrm{KL}\left( \pi _{z_{\theta ^+}}\left( \cdot |s \right) ,\pi _{z_{\theta}}\left( \cdot |s \right) \right) \le \frac{\left| \mathcal{A} \right|}{2}\cdot \left\| \Delta _s \right\| _{\infty}^{2}\cdot \left( 1-\exp \left( -\mathcal{H} \left( \pi _{\theta}\left( \cdot |s \right) \right) \right) \right) +o\left( \left\| \Delta _s \right\| _{2}^{2} \right) \\
$$

其中 $o\left( \left\| \Delta _s \right\| _{2}^{2} \right) $ 是一个根据泰勒展开得到的，关于logits 移动幅度距离的高阶项。在logits移动距离幅度的不大的时候可以忽略。因此如果只关注RHS的第一项，我们可以看到： 

1. logits的差$\Delta_s$的无穷范数 （即action上最大的logits变化）越小，新旧策略整体的KL移动幅度越小。
2. 当entropy $\mathcal{H} \left( \pi _{\theta}\left( \cdot |s \right) \right) $ 越接近0， $1-\exp \left( -\mathcal{H} \left( \pi _{\theta}\left( \cdot |s \right) \right) \right) $ 也越接近0，从而新旧在状态s处的KL移动幅度也越接近0

当然，细心的朋友可能会觉得 前面的常数项 $\frac{\left| \mathcal{A} \right|}{2}$，对于LLM来说可能太大了。实际上我们可以把这一项改进成例如top-p的action space，从而不会出现爆炸。出于简便本文不写详细结果。下面我们给出上面不等式的证明。 

- **不等式证明：**

为方便符号简单，我们记策略向量 $\pi_s:=\pi_{z_\theta}(\cdot|s)$ ， $\pi^+_{s}:=\pi_{z_{\theta^+}}(\cdot|s)$ 。我们定义向量函数

$f\left( z \right) :\mathbb{R} ^{\left| \mathcal{A} \right|}\rightarrow \mathbb{R} =\mathrm{KL}\left( \text{softmax} \left( z \right) ,\pi _s \right) \\$ 为以向量 $z$ 作为logits 与 $\pi_s$ 的KL距离。于是根据泰勒展开，容易知道:

$\begin{align*} \mathrm{KL(}\pi _{s}^{+},\pi _s)&=f(z_{\theta ^+}\left( s \right) ) \\ \,\,             &=f(z_{\theta}\left( s \right) )+\nabla _{z}f(z_{\theta}\left( s \right) )^T\Delta _s+\frac{1}{2}\Delta _{s}^{T}\nabla _{z}^{2}f(z_{\theta}\left( s \right) )\Delta _s+o\left( \left\| \Delta _s \right\| _{2}^{2} \right)  \end{align*}$ 

根据推导，我们可以得到 (此处省略):

$(1)f(z_{\theta}\left( s \right) )=0\qquad (2)\,\nabla _zf(z_{\theta}\left( s \right) )=0\qquad (3)\,\nabla _{z}^{2}f(z_{\theta}\left( s \right) )=\mathcal{F} (z_{\theta}\left( s \right) ). \\$ 其中 $\mathcal{F} \left( z \right)  $ 是softmax 策略在logits 向量 $z$ 处的Fisher information matrix，即

$\mathcal{F} \left( z \right) :=\mathbb{E} _{a\sim p_z}\left[ \nabla _z\log p_z\left( a \right) \nabla _z\log p_z\left( a \right)^T \right] \\$ 其中 $p_z:=\text{softmax} \left( z \right) $ . 将上述推导带入到泰勒展开中可以得到:

$\mathrm{KL(}\pi _{s}^{+},\pi _s)=\frac{1}{2}\Delta _{s}^{T}\mathcal{F} (z_{\theta}\left( s \right) )\Delta _s+o(\left\| \Delta _s \right\| _{2}^{2}). \\$ 注意到对于两个action $a,b \in \mathcal{A}$ , 根据softmax 策略求导，容易知道

$\frac{\partial \log p_z\left( a \right)}{\partial z\left( b \right)}=\mathbb{I} \left\{ a=b \right\} -p_z\left( b \right) \\$ 从而直接推导我们可以得到

$\begin{align*} \Delta _{s}^{T}\mathcal{F} (z_{\theta}\left( s \right) )\Delta _s&=\Delta _{s}^{T}\,\mathbb{E} _{a\sim \pi _s}\left[ \nabla _{z_{\theta}\left( s \right)}\log \pi _{\theta}(a|s)\nabla _{z_{\theta}\left( s \right)}\log \pi _{\theta}(a|s)^T \right] \Delta _s \\ \,\,                   &=\mathbb{E} _{a\sim \pi _s}\left[ \Delta _{s}^{T}\nabla _{z_{\theta}\left( s \right)}\log \pi _{\theta}(a|s)\nabla _{z_{\theta}\left( s \right)}\log \pi _{\theta}(a|s)^T\Delta _s \right]  \\ \,\,                   &=\mathbb{E} _{a\sim \pi _s}\left[ \left( \nabla _{z_{\theta}\left( s \right)}\log \pi _{\theta}(a|s)^T\Delta _s \right) ^2 \right]  \\ \,\,                   &=\mathbb{E} _{a\sim \pi _s}\left[ \left( \nabla _{z_{\theta}\left( s \right)}\log \pi _{\theta}(a|s)^T\Delta _s \right) ^2 \right]  \\ \,\,                   &=\mathbb{E} _{a\sim \pi _s}\left[ \left( \sum_{b\in \mathcal{A}}{\frac{\partial \log \pi _{\theta}(a|s)}{z_{\theta}\left( s,b \right)}\Delta _s\left( b \right)} \right) ^2 \right]  \\ \,\,                   &\overset{\left( a \right)}{\le}\left| \mathcal{A} \right|\cdot \mathbb{E} _{a\sim \pi _s}\left[ \sum_{b\in \mathcal{A}}{\left( \frac{\partial \log \pi _{\theta}(a|s)}{z_{\theta}\left( s,b \right)}\Delta _s\left( b \right) \right) ^2} \right]  \\ \,\,                   &\le \left| \mathcal{A} \right|\cdot \left\| \Delta _s \right\| _{\infty}^{2}\cdot \mathbb{E} _{a\sim \pi _s}\left[ \left\| \nabla _{z_{\theta}\left( s \right)}\log \pi _{\theta}\left( a|s \right) \right\| _{2}^{2} \right]  \\ \,\,                  &\overset{\left( b \right)}{\le}\left| \mathcal{A} \right|\cdot \left\| \Delta _s \right\| _{\infty}^{2}\cdot \left( 1-\exp \left( -\mathcal{H} \left( \pi _{\theta}\left( \cdot |s \right) \right) \right) \right)  \end{align*} \\$其中 (a) 使用了幂等不等式， (b) 实际上利用了第一个理论结果中的不等式。 将这个二次项带入 KL的泰勒展开，我们即得到了结果。

## 3. 总结

从上述推导可以看到，这两个性质的出现，即entropy 收敛导致的学习衰退，完全是由于softmax 参数化的特殊的曲率导致的。在之前的RL研究中 使用牛顿法（如NPG）或者换其他参数化 （比如我自己文章的[Hadamard 参数下的PG](https://academic.oup.com/imaiai/article-pdf/doi/10.1093/imaiai/iaaf003/62206681/iaaf003.pdf)) y一定程度上都能够克服这样的事情，避免陷在局部最优上出不来。我们的LLM+RL的研究，仍然有机会去改进softmax参数化导致的特殊的学习dynamic。