# REINFORCE Pro：causal prefix mask

当前论文与实验使用 rollout/current 两策略目标。历史 PPO 分支保留在框架中，但不是下述方法。

## 两个策略、一个比率

rollout 策略 $p$ 生成回答；current 策略 $q$ 提供训练前向概率。

$$\rho_t=q(y_t\mid c_t)/p(y_t\mid c_t).$$

给定 active-token mask $m_t$，前缀统计量与筛选规则为

$$g_t=\exp\left(\frac{\sum_{s\le t}m_s\log\rho_s}{\max(1,\sum_{s\le t}m_s)}\right),
\qquad M_t=\mathbf1\{\ell_-\le g_t\le\ell_+\}.$$

当前实验采用 $[\ell_-,\ell_+]=[0.8,1.25]$。Pro 的几何均值只生成 mask；保留的 token 使用一次 $\rho_t$。

$$\mathcal L=-\frac{\sum_{i,t}m_{i,t}M_{i,t}\rho_{i,t}A_{i,t}}{\sum_{i,t}m_{i,t}}.$$

Pro Max 仅将 $A$ 换成 Max 的正负分支缩放信号 $H$。Baseline 去掉 $M$。分母保留全部 active tokens。

## 实现入口

```bash
--policy_loss_type token_is \
--enable_vllm_is_correction \
--vllm_is_correction_type reinforce_pro \
--vllm_is_truncated_threshold 0.8 1.25
```

在 `token_is` 分支，以上 correction 开关只启用 prefix mask，不添加第二个 IS 权重。省略它即得到无 mask 的 token-IS 目标。

`openrlhf/models/loss.py` 使用 current/rollout log probabilities：可微比率乘 advantage，float32 前缀统计量生成 detached mask。没有 PPO clip；old log probabilities 仅可供历史分支或日志使用，不参与当前目标和 mask。训练 CLI 会为 `token_is` 请求 rollout log probabilities。

反向传播不对 mask 求导；每次前向仍根据 current 策略重新计算 mask。前缀允许重新进入区间；后续 token 不影响此前的筛选结果。

## 论文与参考技术

方法部分沿用 DPPO/TRM 的 masked objective 与 masking criterion 写法。DPPO 以局部分布散度及 advantage 方向筛选；TRM 用整序列筛选；Pro 用前缀几何均值逐位置筛选。它们的核心比较均可写成 rollout/current 两策略。

证明保留 U-G+N 分解、保留权重的下侧重入矩界、DPPO/TRM 条件序列误差界。总体奖励比较使用 S(q)-S(p)，M(p)=1，D 同时计入缩放误差与当前 mask 的拒绝贡献。

形式化范围见 `formal/PAPER_COVERAGE.json`。微批次和跨 rank 归约需单独按论文附录核对；局部 token mean 不自动等于全批次 token mean。当前运行的冻结源与配置见 `experiments/dapo_small/study-full-plan.json`。
