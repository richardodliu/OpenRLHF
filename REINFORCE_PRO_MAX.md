# REINFORCE Pro Max

REINFORCE Pro Max = **RLOO baseline** + **自适应 token-level 归一化**

## 版本说明

| 模式 | 命令行参数 | 全对/全错处理 |
|------|-----------|---------------|
| **默认** | `--advantage_estimator reinforce_pro_max` | 所有组都走 RLOO baseline 和自适应归一化 |
| **`--uniform_scale`** | `--advantage_estimator reinforce_pro_max --uniform_scale` | 全同 reward 组跳过 RLOO 和归一化，使用 $r_i/n$ |

**默认行为**：对所有 prompt 组统一处理，全对/全错组经过 RLOO 后 shaped reward 为 0。但 KL penalty 仍会产生 token-level advantage，因此仍可能有梯度（除非 `init_kl_coef=0`）。

**启用 `--uniform_scale`**：检测全同 reward 组（全对或全错），跳过 RLOO 和归一化，保留梯度信号。

---

## 算法

### 符号定义

- 给定一个 prompt $q$，采样 $n$ 个回答 $\{o_1, o_2, \ldots, o_n\}$
- 每个回答 $o_i$ 包含 $T_i$ 个 action token
- $r_i$ 为回答 $o_i$ 的 reward
- $A_{i,t}$ 为回答 $o_i$ 第 $t$ 个 token 的 advantage

### Step 1: RLOO Baseline

> **启用 `--uniform_scale` 时**：全同组有特殊处理

对每个回答 $o_i$，使用 leave-one-out baseline：

$$b_i = \frac{\sum_{j \neq i} r_j}{n - 1}$$

$$\tilde{r}_i = r_i - b_i$$

#### 全对/全错处理

> **仅在启用 `--uniform_scale` 时生效**

当一个 prompt 的所有样本 reward 相同（全对或全错）时，RLOO baseline 会导致所有 $\tilde{r}_i = 0$，丢失梯度信号。

**默认行为**：所有组统一使用 RLOO，全同组 shaped reward $\tilde{r}_i = 0$（但 KL penalty 仍会产生 token-level advantage）。

**启用 `--uniform_scale`**：对于全同 reward 的 prompt：
1. **跳过 RLOO baseline**，shaped reward 直接使用：$\tilde{r}_i = \frac{r_i}{n}$
2. **跳过 Step 2 的归一化**

注意：shaped reward $\tilde{r}_i$ 后续仍会叠加 KL penalty 并通过 cumulative return 展开到 token level，因此最终的 token-level advantage 不等于 $\frac{r_i}{n}$，而是包含了 KL 惩罚项的累积回报。

这样可以保留梯度信号：全对时鼓励所有回答，全错时惩罚所有回答。

#### Token-Level 展开

将 $\tilde{r}_i$ 通过 KL penalty 和 cumulative return 展开到 token level，得到初始 advantage $A_{i,t}$。

### Step 2: 自适应 Token-Level 归一化

> **启用 `--uniform_scale` 时**：全同组跳过此步骤

将同一 prompt 的所有 $n$ 个回答的所有 token advantage 拼接为一个向量，按正负分离并分别缩放。

**默认行为**：所有组执行归一化。

**启用 `--uniform_scale`**：仅对混合组（有正有负 reward）执行归一化，全同组跳过。

定义集合：

$$\mathcal{P} = \{(i,t) \mid A_{i,t} > 0\}, \quad \mathcal{N} = \{(i,t) \mid A_{i,t} < 0\}$$

注意：$A_{i,t} = 0$ 的 token 不参与统计，归一化后保持为 0。

统计量：

$$S^+ = \sum_{(i,t) \in \mathcal{P}} A_{i,t}, \quad S^- = \sum_{(i,t) \in \mathcal{N}} A_{i,t}$$

$$Q^+ = \sum_{(i,t) \in \mathcal{P}} A_{i,t}^2, \quad Q^- = \sum_{(i,t) \in \mathcal{N}} A_{i,t}^2$$

归一化后的 advantage：

$$\hat{A}_{i,t} = \begin{cases} \alpha \cdot A_{i,t} & \text{if } A_{i,t} > 0 \\ \beta \cdot A_{i,t} & \text{if } A_{i,t} < 0 \\ 0 & \text{if } A_{i,t} = 0 \end{cases}$$

其中 $\alpha, \beta$ 由以下两个约束确定：

**约束 1（均值为 0）**：

$$\alpha \cdot S^+ + \beta \cdot S^- = 0 \implies \beta = -\alpha \cdot \frac{S^+}{S^-}$$

**约束 2（方差为 1，仅在非零 token 上）**：

$$\alpha^2 \cdot Q^+ + \beta^2 \cdot Q^- = N$$

其中 $N = |\mathcal{P}| + |\mathcal{N}|$ 是**非零 token 的数量**（不包括 $A=0$ 的 token）。

这样设计的好处是：
- 方差约束仅在非零 token 上满足，与均值约束一致
- 产生较小的 $\alpha, \beta$，梯度更保守、训练更稳定
- $A=0$ 的 token 不参与任何统计，保持为 0

联立求解：

$$\alpha = \sqrt{\frac{N}{Q^+ + \left(\frac{S^+}{S^-}\right)^2 Q^-}}, \quad \beta = -\alpha \cdot \frac{S^+}{S^-}$$

### 验证（仅在非零 token 上）

**均值**：

$$\mathbb{E}[\hat{A}] = \frac{\alpha S^+ + \beta S^-}{N} = \frac{\alpha S^+ - \alpha \frac{S^+}{S^-} S^-}{N} = 0 \checkmark$$

**方差**：

$$\text{Var}[\hat{A}] = \frac{\alpha^2 Q^+ + \beta^2 Q^-}{N} = \frac{N}{N} = 1 \checkmark$$

其中 $N = |\mathcal{P}| + |\mathcal{N}|$（非零 token 数量）。

### Fallback（跳过归一化）

当无法正常计算 $\alpha, \beta$ 时，**跳过归一化，直接返回原始 advantage**。触发条件：

1. **全同号**：$\mathcal{P} = \emptyset$ 或 $\mathcal{N} = \emptyset$（所有 advantage 同为正或同为负）
2. **数值问题**：$|S^+| < \epsilon$ 或 $|S^-| < \epsilon$，或 $\alpha, \beta$ 计算结果非有限

**设计理由**：
- 全同号情况说明组内样本质量一致，RLOO baseline 后的 advantage 本身已有意义
- 跳过归一化保持原始梯度信号，避免引入额外偏差
- 简化代码逻辑，减少边界情况处理

### 数值稳定性

为保证数值稳定，实现中包含以下保护措施：

1. **Sum 阈值检测**：当 $|S^+|$ 或 $|S^-|$ 小于 $\epsilon$（默认 `1e-8`）时，跳过归一化

2. **Alpha/Beta 裁剪**：$\alpha, \beta$ 被限制在 $[\epsilon, \text{max\_scale}]$ 范围内（默认 `max_scale=10.0`），防止极端缩放导致梯度爆炸

3. **中间计算裁剪**：`ratio^2 * Q-` 被限制在 `1e8` 以内，防止溢出

4. **有限性检查**：若 $\alpha$ 或 $\beta$ 为 `NaN`/`Inf`，跳过归一化

这些裁剪可能导致归一化后的均值/方差略微偏离 0/1，但在绝大多数情况下偏差很小（通常 `|mean| < 0.001`, `|var - 1| < 0.01`）。

---

## 与其他算法对比

| | REINFORCE++ Baseline | RLOO | Pro Max | Pro Max + `--uniform_scale` |
|---|---|---|---|---|
| **Baseline** | $b = \frac{1}{n}\sum r_i$ | $b_i = \frac{\sum_{j \neq i} r_j}{n-1}$ | RLOO (所有组) | RLOO (混合组) / $\frac{r_i}{n}$ (全同组) |
| **归一化** | 全局 $\frac{A - \mu}{\sigma}$ | 无 | 自适应 $\alpha/\beta$ (所有组) | 自适应 $\alpha/\beta$ (混合组) / 无 (全同组) |
| **约束** | $\mathbb{E}[A]=0, \text{Std}[A]=1$ | 无 | $\mathbb{E}[\hat{A}]=0, \text{Var}[\hat{A}]=1$ | 同左 (仅混合组) |
| **全同组梯度** | 有 | 无 | 仅 KL | 有 |
| **Critic** | 不需要 | 不需要 | 不需要 | 不需要 |

---

## 实现

### 核心代码

`openrlhf/trainer/ppo_utils/experience_maker.py` 中的 `compute_advantages_and_returns` 方法：

**Step 1: RLOO Baseline**
- **默认**: 所有组使用 RLOO
- **`--uniform_scale`**: 混合组使用 RLOO，全同组使用 $r_i/n$

**Step 2: 自适应归一化**
- 通过 `experience.index` 恢复原始样本顺序（处理 `use_dynamic_batch` 打乱的情况）
- **默认**: 所有组调用 `_adaptive_token_normalization_single_group()`
- **`--uniform_scale`**: 混合组调用归一化函数，全同组跳过

### 使用方法

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# 默认模式（全同组仅 KL penalty，shaped reward=0）
ray job submit --address="http://127.0.0.1:8265" \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --advantage_estimator reinforce_pro_max \
  --n_samples_per_prompt 8 \
  --pretrain Qwen/Qwen3-4B \
  --prompt_data your_dataset \
  --actor_learning_rate 5e-7 \
  --init_kl_coef 1e-5

# 启用 --uniform_scale（保留全同组梯度）
ray job submit --address="http://127.0.0.1:8265" \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --advantage_estimator reinforce_pro_max \
  --uniform_scale \
  --n_samples_per_prompt 8 \
  --pretrain Qwen/Qwen3-4B \
  --prompt_data your_dataset \
  --actor_learning_rate 5e-7 \
  --init_kl_coef 1e-5
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--advantage_estimator` | `reinforce_pro_max` | 必需 |
| `--uniform_scale` | flag | 可选，启用后全同组跳过 RLOO 和归一化 |
| `--n_samples_per_prompt` | 8 | 必需，$n > 1$ |
| `--actor_learning_rate` | `5e-7` | 推荐 |
| `--init_kl_coef` | `1e-5` | 推荐 |

**注意**: `reinforce_pro_max` 会强制 `gamma=1.0`（无折扣累积回报），与 REINFORCE/RLOO 一致。用户设置的 `--gamma` 值将被忽略。

### 监控指标

- `advantage_mean`：应接近 0（启用 `--uniform_scale` 时全同组除外）
- `advantage_std`：应接近 1（启用 `--uniform_scale` 时全同组除外）
- `reward_mean`：RLOO baseline 后的奖励均值
- `kl`：KL 散度

---

## 测试

```bash
python test_reinforce_pro_max.py
```

验证内容：
1. 顺序排列时新旧实现结果一致
2. 动态批处理打乱顺序后，新实现仍能正确按 prompt 分组（旧实现会出错）
3. 归一化后每个 prompt 组在**非零 token** 上满足 $\mathbb{E}[\hat{A}]=0, \text{Var}[\hat{A}]=1$
4. indices 完整性检查（必须是 0..B-1 的排列）
5. prompt 分组一致性检查（每个 group 内的 prompt 必须相同）

---

## 参考文献

- [REINFORCE++: A Simple and Efficient Approach](https://arxiv.org/abs/2501.03262)
- [REINFORCE++-baseline is all you need in RLVR](https://medium.com/@janhu9527/reinforce-baseline-is-all-you-need-in-rlvr-f5406930aa85)
