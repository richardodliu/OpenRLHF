# RL 算法详解

## 算法总览

| 算法 | `--advantage_estimator` | Critic | 特点 | 推荐场景 |
|------|------------------------|--------|------|----------|
| PPO | `gae` (default) | 需要 | 完整 GAE，最稳定 | 通用 RLHF |
| REINFORCE++ | `reinforce` | 不需要 | PPO tricks，无 Critic | 节省显存 |
| REINFORCE++-baseline | `reinforce_baseline` | 不需要 | 均值 baseline | 推理任务 (RLVR)，**推荐** |
| GRPO | `group_norm` | 不需要 | 组归一化 | 批量训练 |
| Dr. GRPO | `dr_grpo` | 不需要 | 简化 GRPO | 移除 `/std` |
| RLOO | `rloo` | 不需要 | Leave-one-out baseline | 多样本训练 |

---

## PPO (Proximal Policy Optimization)

**参数**: `--advantage_estimator gae` (默认)

**原理**:
- 使用 Critic 网络估计 value function
- GAE (Generalized Advantage Estimation) 计算优势
- PPO clip 限制策略更新幅度

**优势计算**:
```
A_t = δ_t + γλδ_{t+1} + (γλ)²δ_{t+2} + ...
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**配置**:
```bash
--advantage_estimator gae \
--critic_num_gpus_per_node 8 \
--critic_learning_rate 9e-6 \
--gamma 0.99 \
--lambd 0.95
```

**优点**: 稳定，效果好
**缺点**: 需要额外的 Critic 网络，显存占用大

---

## REINFORCE++

**参数**: `--advantage_estimator reinforce`

**原理**:
- 移除 Critic 网络
- 使用累积回报作为优势
- 保留 PPO 的 clip 和 KL 惩罚

**优势计算**:
```
A_t = R_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
```

**配置**:
```bash
--advantage_estimator reinforce \
--gamma 1.0 \
--init_kl_coef 0.01
```

**优点**: 无需 Critic，节省显存
**缺点**: 方差较大

---

## REINFORCE++-baseline (推荐)

**参数**: `--advantage_estimator reinforce_baseline`

**原理**:
- 在 REINFORCE++ 基础上增加均值 baseline
- 减去同一 prompt 的平均奖励
- 移除 GRPO 中的 `/std` 归一化

**优势计算**:
```
A_i = R_i - mean(R_1, R_2, ..., R_n)
```

其中 `R_1...R_n` 是同一 prompt 的 n 个样本的回报。

**配置**:
```bash
--advantage_estimator reinforce_baseline \
--n_samples_per_prompt 8 \
--init_kl_coef 1e-5 \
--use_kl_loss \
--kl_estimator k2
```

**优点**:
- 对不同奖励尺度鲁棒
- 适合推理任务 (RLVR)
- 训练稳定

**推荐场景**: 数学推理、代码生成等需要验证的任务

**相关论文**: [REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models](https://arxiv.org/abs/2501.03262)

---

## GRPO (Group Relative Policy Optimization)

**参数**: `--advantage_estimator group_norm`

**原理**:
- 同一 prompt 的多个样本形成一组
- 组内归一化：减均值，除标准差

**优势计算**:
```
A_i = (R_i - mean(R)) / (std(R) + ε)
```

**配置**:
```bash
--advantage_estimator group_norm \
--n_samples_per_prompt 8
```

**优点**: 组内相对比较
**缺点**: `/std` 可能导致数值不稳定

---

## Dr. GRPO

**参数**: `--advantage_estimator dr_grpo`

**原理**:
- 简化版 GRPO
- 移除 `/std` 归一化
- 与 REINFORCE++-baseline 类似

**优势计算**:
```
A_i = R_i - mean(R)
```

**配置**:
```bash
--advantage_estimator dr_grpo \
--n_samples_per_prompt 8
```

---

## RLOO (REINFORCE Leave-One-Out)

**参数**: `--advantage_estimator rloo`

**原理**:
- Leave-one-out baseline
- 对于样本 i，baseline 是其他样本的平均

**优势计算**:
```
A_i = R_i - (sum(R) - R_i) / (n - 1)
```

**配置**:
```bash
--advantage_estimator rloo \
--n_samples_per_prompt 8
```

**优点**: 更精确的 baseline 估计
**要求**: `n_samples_per_prompt > 1`

---

## 算法选择指南

### 通用 RLHF (对话、指令跟随)

```bash
# PPO - 最稳定
--advantage_estimator gae \
--critic_num_gpus_per_node 8

# 或 REINFORCE++ - 节省显存
--advantage_estimator reinforce
```

### 推理任务 (数学、代码)

```bash
# REINFORCE++-baseline - 推荐
--advantage_estimator reinforce_baseline \
--n_samples_per_prompt 8 \
--init_kl_coef 1e-5 \
--use_kl_loss
```

### 显存受限

```bash
# 无 Critic 的算法
--advantage_estimator reinforce_baseline
# 或
--advantage_estimator reinforce
```

### 多样本训练

```bash
# RLOO 或 GRPO
--advantage_estimator rloo \
--n_samples_per_prompt 8
```

---

## 源码位置

优势计算逻辑在 `openrlhf/trainer/ppo_utils/experience_maker.py`:

```python
# RemoteExperienceMaker.compute_advantages_and_returns()

if args.advantage_estimator == "rloo":
    baseline = (rewards.sum(-1, keepdim=True) - rewards) / (n - 1)
    rewards = rewards - baseline
elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
    rewards = rewards - rewards.mean(-1, keepdim=True)
elif args.advantage_estimator == "group_norm":
    rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
```

GAE 计算在 `get_advantages_and_returns()`，累积回报在 `get_cumulative_returns()`。
