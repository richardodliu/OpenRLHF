# 模型模块

路径: `openrlhf/models/`

## actor.py - Actor 模型

**关键类**:
- `Actor` - Actor 模型封装

**功能**:
- 封装 HuggingFace 模型
- 支持 LoRA/QLoRA
- 计算 action log probabilities

**关键方法**:
- `forward()` - 计算 log probs
- `generate()` - 生成文本
- `gradient_checkpointing_enable()` - 启用梯度检查点

---

## loss.py - 损失函数

### PolicyLoss - PPO 策略损失

```python
PolicyLoss(
    clip_eps_low=0.2,
    clip_eps_high=0.28,
    dual_clip=None,
    policy_loss_type="ppo",
    enable_vllm_is_correction=False,
    vllm_is_truncated_threshold=None,
    use_icepop=False,
)
```

**功能**: 计算 PPO clip loss

### ValueLoss - Critic 损失

```python
ValueLoss(clip_eps=0.2)
```

**功能**: 计算 value function loss

### DPOLoss - DPO 损失

```python
DPOLoss(beta=0.1, label_smoothing=0.0, loss_type="dpo")
```

**支持类型**: `dpo`, `ipo`, `cdpo`, `hinge`

### KTOLoss - KTO 损失

```python
KTOLoss(beta=0.1)
```

---

## utils.py - 工具函数

### compute_approx_kl()

计算近似 KL 散度。

```python
kl = compute_approx_kl(
    log_probs,          # 当前策略 log probs
    log_probs_base,     # 参考策略 log probs
    kl_estimator="k1"   # k1, k2, k3
)
```

**KL 估计器**:
- `k1`: `log_probs - log_probs_base`
- `k2`: `0.5 * (log_probs - log_probs_base)^2`
- `k3`: `exp(log_probs_base - log_probs) - 1 - (log_probs_base - log_probs)`

### compute_reward()

计算带 KL 惩罚的奖励。

```python
reward = compute_reward(
    r,                  # 原始奖励
    kl_coef,            # KL 系数
    kl,                 # KL 散度
    action_mask=None,
    reward_clip_range=None,
)
```

### masked_mean()

计算带掩码的均值。

```python
mean = masked_mean(tensor, mask, dim=-1)
```

---

## model_utils.py

**关键函数**:
- `get_llm_for_sequence_regression()` - 获取序列回归模型 (RM/Critic)
