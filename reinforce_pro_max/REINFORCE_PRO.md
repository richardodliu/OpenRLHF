# REINFORCE Pro (reinforce_pro)

## 1. 背景：现有方法的局限

vLLM 生成 (rollout) 和 Actor 训练之间存在策略漂移，导致 off-policy 问题。现有 IS 校正方法各有局限：

| 方法 | IS 粒度 | 问题 |
|------|--------|------|
| **tis / icepop** | 单 token | 忽略 token 间依赖，早期偏离不影响后续 token 的权重 |
| **seq-mask-tis** | 整序列 | 粒度太粗，序列内不同位置的偏离程度被平均掉 |
| **GSPO** | 整序列（ratio） | 同上，且作用于 ratio 而非 mask |

**核心问题**：语言模型生成是自回归的，位置 t 的 token 依赖于前面所有 token。如果前缀（位置 0~t-1）的策略已经偏离，位置 t 的 IS 权重应该反映这种累积偏离，而不是只看单个 token 的偏离。

---

## 2. 算法动机：为什么需要前缀累积 IS

**因果结构**：自回归生成中，位置 t 的条件概率依赖于前缀 `x_{0:t-1}`。如果前缀的策略偏离较大，后续 token 的分布也会受到影响。因此，位置 t 的 IS 权重应该是从第 1 个 token 到第 t 个 token 的**累积**偏离，而不是单点偏离。

**前缀累积几何均值的直觉**：

```
prefix_is[t] = exp( (log_ratio[0] + log_ratio[1] + ... + log_ratio[t]) / (t+1) )
             = (ratio[0] * ratio[1] * ... * ratio[t])^{1/(t+1)}
```

即前 t+1 个 token 的 IS 比值的几何平均。这个值越偏离 1，说明前缀的策略漂移越严重。

---

## 3. 算法

`reinforce_pro` 是 `--vllm_is_correction_type` 的一个选项，用于校正 vLLM rollout 与 Actor 前向 logprob 的不一致。

**与其他校正方式的对比**：

| 类型 | 过滤粒度 | 校正系数 |
|------|---------|---------|
| `tis` | 无过滤，token 级 clamp | token 级 `vllm_is` |
| `icepop` | token 级过滤（单点） | token 级 `vllm_is` |
| `seq-mask-tis` | 序列级过滤（整序列 pass/drop） | token 级 `vllm_is` |
| `reinforce_pro` | **token 级过滤（前缀累积均值）** | token 级 `vllm_is` |

`reinforce_pro` 用前缀累积几何均值决定每个 token 是否参与 loss，通过阈值过滤后再乘 token 级 `vllm_is` 校正系数：

```python
log_ratio = old_log_probs - rollout_log_probs

# 1. 前缀累积几何均值（用于过滤）
cumsum_log_ratio = cumsum(log_ratio * action_mask, dim=-1)
positions = cumsum(action_mask.float(), dim=-1).clamp(min=1)
prefix_is = exp(cumsum_log_ratio / positions).detach()

# 2. token 级过滤
token_mask = (prefix_is >= low) & (prefix_is <= high)

# 3. token 级 IS 校正
vllm_is = exp(log_ratio).detach()
loss = token_mask * vllm_is * loss
```

**与 `seq-mask-tis` 的区别**：

| | `seq-mask-tis` | `reinforce_pro` |
|--|---------------|----------------|
| IS 统计量 | 序列级几何均值（单值） | 前缀累积几何均值（每个位置一个值） |
| 过滤粒度 | 整序列 pass/drop | 每个 token 独立 pass/drop |
| 因果结构 | 无（整序列同一决策） | **有**（早期偏离影响后续 token） |
| 适用场景 | 序列整体偏离 | 序列内局部/渐进偏离 |

---

## 4. 数学推导

### 梯度流

`reinforce_pro` 只影响 mask，不影响 PPO ratio 的计算，梯度流为：

```
log_probs [有梯度] ────┐
                       ↓
# PPO ratio（用于 loss 计算）
log_ratio_ppo = log_probs - old_log_probs [有梯度]
                       ↓
ratio = exp(log_ratio_ppo) [有梯度，标准 PPO ratio]
                       ↓
loss = -min(ratio * A, clip(ratio) * A) [有梯度]
                       ↓
# IS correction（用于 mask 和权重，均 detach）
log_ratio_is = old_log_probs - rollout_log_probs [无梯度方向]
prefix_is = exp(cumsum(log_ratio_is * mask) / positions).detach()
token_mask = (prefix_is >= low) & (prefix_is <= high) [无梯度，bool]
vllm_is = exp(log_ratio_is).detach() [无梯度]
                       ↓
loss = token_mask * vllm_is * loss [loss 有梯度]
```

**关键点**：
- PPO ratio 用 `log_probs - old_log_probs`，梯度通过此路径传播
- IS correction 用 `old_log_probs - rollout_log_probs`，两者均 detach，不贡献梯度
- `token_mask = 0` 的位置 loss 为 0，该位置不贡献梯度（loss 级别屏蔽）

### 因果结构保留

位置 t 的 `prefix_is[t]` 包含了位置 0 到 t 的累积信息：

```
prefix_is[0] = exp(log_ratio[0] / 1)
prefix_is[1] = exp((log_ratio[0] + log_ratio[1]) / 2)
prefix_is[t] = exp(sum(log_ratio[0:t+1]) / (t+1))
```

因此，如果序列前半段策略偏离较大，后半段的 `prefix_is` 也会偏离阈值，从而被 mask 掉。这与 `icepop` 只看单点不同——`icepop` 中前半段偏离不影响后半段的 mask 决策。

---

## 5. 使用方式

```bash
--enable_vllm_is_correction \
--vllm_is_correction_type reinforce_pro \
--vllm_is_truncated_threshold 0.5 5.0
```

`policy_loss_type` 保持默认 `ppo`，`reinforce_pro` 只影响 loss 中的 mask 策略。

**阈值选择建议**：
- 默认：`[0.5, 5.0]` - 适合大多数场景
- 保守：`[0.8, 2.0]` - 更严格，丢弃更多偏离样本
- 宽松：`[0.2, 10.0]` - 允许更大的前缀累积偏离
