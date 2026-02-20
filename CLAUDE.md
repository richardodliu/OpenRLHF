# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在此代码库中工作的指南。
默认用中文回复用户的问题和解释解决的思路

## 仓库概述

OpenRLHF 是基于 Ray + vLLM 分布式架构构建的高性能 RLHF 框架。当前版本: 0.9.2

**REINFORCE++ Baseline**: 无需 Critic 网络的高效 RL 算法，特别适合推理任务 (数学、代码、逻辑推理)

## 快速开始

### 安装
```bash
pip install openrlhf[vllm]  # 推荐: 包含 vLLM 0.13.0
```

### 训练命令
```bash
# 启动 Ray 集群
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# 启动训练
ray job submit --address="http://127.0.0.1:8265" \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --advantage_estimator reinforce_baseline \
  --pretrain Qwen/Qwen3-4B \
  --prompt_data your_dataset \
  --n_samples_per_prompt 8 \
  --actor_learning_rate 5e-7 \
  --init_kl_coef 1e-5
```

## REINFORCE++ Baseline 参考实现

### 算法原理

**传统 PPO**:
```
优势函数 A = GAE(rewards, values)  # 需要 Critic 估计 values
```

**REINFORCE++ Baseline**:
```
优势函数 A = rewards - mean(rewards)  # 无需 Critic，使用均值 baseline
```

### 核心代码位置

```
openrlhf/trainer/ppo_utils/experience_maker.py  # 优势计算核心
    └── make_experience_batch()                  # 第 745-789 行
```

### 关键代码实现

#### 1. 优势函数计算 (experience_maker.py:745-789)

```python
# openrlhf/trainer/ppo_utils/experience_maker.py

# 步骤 1: Reward Shaping (第 745-750 行)
if args.advantage_estimator == "reinforce_baseline":
    # ✨ 核心: 使用 batch 均值作为 baseline
    rewards = rewards - rewards.mean(-1, keepdim=True)
    # 注意: 移除了 /std 归一化，这是与 GRPO 的关键区别

# 步骤 2: 计算 Returns (第 772-789 行)
elif args.advantage_estimator in ["reinforce_baseline", ...]:
    # REINFORCE++ 强制 gamma=1.0 (无折扣)
    if args.gamma != 1.0:
        logger.warning("gamma is set to 1.0 for reinforce_baseline")
        args.gamma = 1.0

    # 计算累积回报
    experience.returns = self.get_cumulative_returns(
        reward,              # reward = r - mean(r) + KL_penalty
        experience.action_mask,
        gamma=1.0,
    )
    # advantages = returns (REINFORCE++ 中两者相同)
    experience.advantages = deepcopy(experience.returns)

# 步骤 3: 跨 batch 归一化 (第 798-818 行)
if args.advantage_estimator == "reinforce_baseline":
    # 收集所有 experience 的 advantages
    all_advantages = torch.cat([exp.advantages.flatten() for exp in experiences])
    all_action_masks = torch.cat([exp.action_mask.flatten() for exp in experiences])

    # 计算均值和标准差
    mean = (all_advantages * all_action_masks).sum() / num_actions
    var = ((all_advantages - mean).pow(2) * all_action_masks).sum() / num_actions
    rstd = var.clamp(min=1e-8).rsqrt()  # 1 / std

    # 归一化每个 experience
    for exp in experiences:
        exp.advantages = (exp.advantages - mean) * rstd
```

#### 2. Experience 数据结构 (experience_maker.py:35-109)

```python
@dataclass
class Experience:
    # 序列数据
    sequences: torch.Tensor           # (B, S) - 完整 token 序列
    attention_mask: torch.LongTensor  # (B, S) - 注意力掩码
    action_mask: torch.BoolTensor     # (B, S) - 动作位置掩码

    # REINFORCE++ Baseline 核心数据
    action_log_probs: torch.Tensor      # (B, S) - Actor 的 log π(a|s)
    base_action_log_probs: torch.Tensor # (B, S) - Reference 的 log π_ref(a|s)
    rewards: torch.Tensor               # (B,) - Reward 分数

    # 计算结果
    advantages: torch.Tensor  # (B, S) - rewards - mean(rewards)，已归一化
    returns: torch.Tensor     # (B, S) - 与 advantages 相同
    kl: torch.Tensor          # (B, S) - KL(π || π_ref)

    # 注意: 不需要 values (PPO 才需要)
```

#### 3. Actor 更新 (ppo_trainer.py:144-160)

```python
def ppo_train(self, global_steps: int) -> Dict:
    # REINFORCE++ Baseline 不需要训练 Critic
    run_critic = self.critic_model_group is not None  # False for reinforce_baseline
    run_actor = True

    # 只更新 Actor
    if run_actor:
        actor_status = self.actor_model_group.run_method_batch(
            method_name="train_step",
            num_steps=self.args.num_train_steps_per_episode,
        )
        status.update(actor_status)

    return status
```

### 核心参数

**必需参数**:
```bash
--advantage_estimator reinforce_baseline  # 算法选择
```

**推荐参数**:
```bash
# Batch 配置 (关键)
--n_samples_per_prompt 8        # 多样本估计 baseline
--train_batch_size 1024         # 大 batch size (无 Critic 显存占用)
--rollout_batch_size 128

# 学习率和 KL
--actor_learning_rate 5e-7      # Actor 学习率
--init_kl_coef 1e-5             # KL 惩罚系数 (比 PPO 小)

# 优化
--packing_samples               # 样本打包
--use_dynamic_batch             # 动态 batch
--gradient_checkpointing        # 梯度检查点

# 推理任务特有
--prompt_max_len 10240          # 长 prompt
--generate_max_len 64000        # 长思维链
--ring_attn_size 2              # RingAttention (超长序列)
```

**不需要的参数** (相比 PPO):
```bash
# 以下参数 REINFORCE++ Baseline 不使用:
# --critic_pretrain <model>      # 无 Critic
# --critic_learning_rate 9e-6    # 无 Critic
# --vf_coef 0.1                  # 无 Value loss
# --cliprange_value 0.2          # 无 Value clip
# --gamma 1.0                    # 强制为 1.0
# --lambd 0.95                   # GAE 参数，不需要
```

## 自定义奖励函数

REINFORCE++ Baseline 特别适合自定义奖励函数:

```python
# reward_func.py
import torch

def reward_func(queries, prompts, labels):
    """
    Args:
        queries: List[str] - 完整文本 (prompt + response)
        prompts: List[str] - 原始 prompts
        labels: List[str] - 真值标签 (来自 --label_key)

    Returns:
        dict {
            "rewards": Tensor - 用于优势计算
            "scores": Tensor - 用于动态过滤 (0-1 范围)
            "extra_logs": Dict - wandb 日志
        }
    """
    batch_size = len(queries)
    rewards = torch.zeros(batch_size)

    # 示例: 数学答案验证
    for i in range(batch_size):
        response = queries[i].replace(prompts[i], "")
        if labels[i] and labels[i].lower() in response.lower():
            rewards[i] = 1.0  # 正确
        else:
            rewards[i] = 0.0  # 错误

    return {
        "rewards": rewards,
        "scores": rewards,
        "extra_logs": {"accuracy": rewards.mean().item()}
    }
```

使用方式:
```bash
--remote_rm_url /path/to/reward_func.py \
--label_key label
```

## 代码调试技巧

### 1. 查看优势计算
```python
# 在 experience_maker.py:750 后添加
print(f"Rewards mean: {rewards.mean().item():.4f}")
print(f"Rewards std: {rewards.std().item():.4f}")
```

### 2. 检查归一化
```python
# 在 experience_maker.py:818 后添加
print(f"Advantages mean: {mean.item():.4f}")
print(f"Advantages std: {(1/rstd).item():.4f}")
```

### 3. 监控训练指标
```bash
# 启用 TensorBoard
--use_tensorboard ./logs

# 关键指标:
# - reward_mean: 奖励均值
# - advantage_mean: 应该接近 0
# - advantage_std: 应该接近 1
# - kl: KL 散度
```

## GPU 资源分配

**REINFORCE++ Baseline 不需要 Critic**，资源分配更简单:

```bash
# 8 GPUs 示例 (混合引擎)
--ref_num_gpus_per_node 4 \
--actor_num_gpus_per_node 4 \
--vllm_num_engines 2 \
--vllm_tensor_parallel_size 2 \
--colocate_all_models \
--vllm_gpu_memory_utilization 0.7

# 不需要 --critic_num_gpus_per_node
```

## 常见问题

1. **显存不足**: 减小 `--n_samples_per_prompt` 和 `--micro_train_batch_size`
2. **训练不稳定**: 增大 `--train_batch_size` (更准确的 baseline 估计)
3. **KL 过大**: 增大 `--init_kl_coef` 或减小 `--actor_learning_rate`
4. **序列太长**: 启用 `--ring_attn_size 2` (RingAttention)

## 核心文件路径

```
openrlhf/
├── cli/train_ppo_ray.py                     # 训练入口
├── trainer/
│   ├── ppo_trainer.py                       # 训练循环 (第 144-160 行)
│   └── ppo_utils/
│       └── experience_maker.py              # 优势计算 (第 745-818 行) ⭐
├── models/
│   └── actor.py                             # Actor 模型
└── datasets/
    └── prompts_dataset.py                   # 数据集加载

examples/scripts/
└── train_reinforce_baseline_hybrid_engine.sh  # 完整训练脚本示例
```

## 算法对比

| 特性 | REINFORCE++ Baseline | PPO |
|------|---------------------|-----|
| **Critic 网络** | ❌ 不需要 | ✅ 需要 |
| **显存占用** | 更低 (~25% 节省) | 更高 |
| **优势函数** | `r - mean(r)` | `GAE(r, V)` |
| **适用场景** | 推理任务 (0/1 奖励) | 通用 RLHF |
| **Gamma** | 强制 1.0 | 可配置 |
| **关键代码** | experience_maker.py:745-789 | + ppo_critic.py |

## 参考文献

- [REINFORCE++: A Simple and Efficient Approach](https://arxiv.org/abs/2501.03262)
- [REINFORCE++-baseline is all you need in RLVR](https://medium.com/@janhu9527/reinforce-baseline-is-all-you-need-in-rlvr-f5406930aa85)

## 重要性采样校正 (Off-Policy RL)

### 背景

vLLM 生成 (rollout) 和 Actor 训练之间存在策略漂移，导致 off-policy 问题。重要性采样 (IS) 通过权重校正来弥补分布差异。

参考: [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl)


---

## 现有技术

### TIS (Truncated Importance Sampling)

TIS 是用于校正 vLLM rollout 和 Actor 训练之间策略漂移的技术。

**问题背景**:
- vLLM 生成样本时使用的策略是 `π_rollout`
- Actor 训练时的策略已经更新为 `π_old`
- 两者之间存在差异，导致 off-policy 问题

**解决方案**:
TIS 通过一个额外的重要性采样权重来校正这个差异：

```python
# vllm_is: 校正 rollout 和训练时策略的差异
vllm_is = exp(old_log_probs - rollout_log_probs)  # π_old / π_rollout

# 将校正权重乘到 loss 上
loss = vllm_is * loss
```

**注意**: TIS 的 `vllm_is` 与 PPO 的 `ratio` 是不同的：
- **PPO ratio**: `exp(log_probs - old_log_probs)` = `π_current / π_old` - 用于 PPO clipping
- **TIS vllm_is**: `exp(old_log_probs - rollout_log_probs)` = `π_old / π_rollout` - 用于 off-policy 校正


**三种校正模式** (`--vllm_is_correction_type`):

| 模式 | 实现 | 效果 | 适用场景 |
|------|------|------|---------|
| **tis** (默认) | `vllm_is.clamp(low, high)` | token 级别裁剪到边界 | 通用 |
| **icepop** | `vllm_is * mask` | token 级别范围外设为 0 | 高方差场景 |
| **seq-mask-tis** | 序列级几何均值过滤 + token 级系数 | 序列级过滤，保留 token 级校正 | 序列整体偏离 |

**代码实现** (openrlhf/models/loss.py:182-201):
```python
log_ratio = old_log_probs - rollout_log_probs
if self.vllm_is_correction_type == "icepop":
    # ICEPOP: 范围外的权重直接置零
    vllm_is = torch.exp(log_ratio).detach()
    mask = (vllm_is >= low) & (vllm_is <= high)
    vllm_is = vllm_is * mask
    loss = vllm_is * loss
elif self.vllm_is_correction_type == "seq-mask-tis":
    # MIS: 序列级几何均值过滤 + token 级校正
    seq_log_ratio = masked_mean(log_ratio, action_mask, dim=-1)
    seq_is = torch.exp(seq_log_ratio)
    seq_mask = (seq_is >= low) & (seq_is <= high)
    vllm_is = torch.exp(log_ratio).detach()
    loss = seq_mask.unsqueeze(-1) * vllm_is * loss
else:
    # TIS: token 级别裁剪到边界
    vllm_is = torch.exp(log_ratio).clamp(low, high).detach()
    loss = vllm_is * loss
```

**阈值选择建议**:
- 默认: `[0.5, 5.0]` - 适合大多数场景
- 保守: `[0.8, 2.0]` - 更严格，更稳定但丢弃更多样本
- 宽松: `[0.2, 10.0]` - 允许更大的策略偏离

参考: [ICEPOP](https://www.emergentmind.com/topics/icepop)

### GSPO (序列级 IS)

GSPO 在序列级别计算 IS 系数，整个序列共享同一个 ratio。

**数学公式**:
```python
log_ratio = log_probs - rollout_log_probs
ratio = exp(sum(log_ratio * mask) / sum(mask))  # 序列级平均
```

**特点**:
- 粒度粗，无法区分序列中不同位置的策略偏离程度
- 适合策略偏离整体较大的场景

参考: [GSPO Paper](https://arxiv.org/pdf/2507.18071)

### 现有方法的局限

| 方法 | IS 粒度 | 问题 |
|------|--------|------|
| **Token-level IS (PPO)** | 单 token | 忽略 token 间依赖，早期偏离不影响后续 |
| **GSPO (序列级)** | 整序列 | 粒度太粗，无法区分不同位置的偏离程度 |

---

## 我们的方案: 前缀累积 IS (Prefix Cumulative IS)

解决 vLLM 推理与 Actor 前向 logprob 不一致的问题。详细设计文档参见 `REINFORCE_PRO.md`。

**注意**: `--vllm_is_correction_type reinforce_pro` 需配合 `--enable_vllm_is_correction` 使用。

### 使用方式

```bash
--enable_vllm_is_correction \
--vllm_is_truncated_threshold 0.5 5.0 \
--vllm_is_correction_type reinforce_pro
```

### 方法对比

| 方法 | 过滤粒度 | 因果结构 |
|------|---------|---------|
| Token-level (tis/icepop) | 单 token | 无 |
| seq-mask-tis | 整序列 | 无 |
| **reinforce_pro** | 前缀累积（token 级） | **有 ✓** |

**与现有技术的关系**: 详见 `REINFORCE_PRO.md`
- `reinforce_pro` 是 `--vllm_is_correction_type` 的选项，`policy_loss_type` 保持 `ppo`
- 核心优势：用前缀累积几何均值做 token 级 mask，保留因果结构

---

### 核心参数汇总

| 参数 | 说明 |
|------|------|
| `--enable_vllm_is_correction` | 启用 IS 校正 |
| `--vllm_is_truncated_threshold 0.5 5.0` | IS 权重截断区间 [low, high] |
| `--vllm_is_correction_type tis\|icepop\|seq-mask-tis\|reinforce_pro` | IS 校正模式 |
| `--policy_loss_type ppo\|gspo` | 策略损失类型 |

### 实现细节 (openrlhf/models/loss.py)

**PPO 模式的 IS 权重** (L152-181):
```python
log_ratio = old_log_probs - rollout_log_probs  # π_old / π_rollout
# 按 vllm_is_correction_type 处理:
# tis: vllm_is.clamp(low, high)
# icepop: vllm_is * mask (范围外置 0)
# seq-mask-tis: 序列级几何均值过滤 + token 级 vllm_is
# reinforce_pro: 前缀累积几何均值过滤 + token 级 vllm_is
```

**reinforce_pro 实现** (L169-177):
```python
cumsum_log_ratio = cumsum(log_ratio * action_mask, dim=-1)
positions = cumsum(action_mask.float(), dim=-1).clamp(min=1)
prefix_is = exp(cumsum_log_ratio / positions).detach()   # 前缀几何均值，用于过滤
token_mask = (prefix_is >= low) & (prefix_is <= high)
vllm_is = exp(log_ratio).detach()                        # token 级校正系数
loss = token_mask * vllm_is * loss
```

**GSPO (序列级别)**:
```python
# 使用 rollout_log_probs 计算 ratio
log_ratio = log_probs - rollout_log_probs
ratio = exp(sum(log_ratio * mask) / sum(mask))  # 序列级别
```
参考: https://arxiv.org/pdf/2507.18071

### ProRL v2 推荐配置

```bash
--enable_vllm_is_correction \
--vllm_is_truncated_threshold 0.5 5.0 \
--vllm_is_correction_type icepop
```

### 算法对比

| 特性 | TIS (tis) | ICEPOP (icepop) | seq-mask-tis | reinforce_pro |
|------|-----------|-----------------|--------------|---------------|
| 范围外处理 | clamp 到边界 | 设为 0 | 序列级过滤 | token 级过滤 |
| 过滤粒度 | 无过滤 | 单 token | 整序列 | 前缀累积（token 级） |
| 因果结构 | 无 | 无 | 无 | **有 ✓** |
| 适用场景 | 通用 | 高方差场景 | 序列整体偏离 | 序列内渐进偏离 |

## 算法设计
关于 REINFORCE Pro Max 算法的详细实现指南，请参考 `REINFORCE_MAX.md`。

关键参数:
- `--advantage_estimator reinforce_max`: 启用 REINFORCE Max 算法
- `--uniform_scale`: 可选，启用后全同 reward 组跳过 RLOO baseline 和归一化，使用 r_i/n

## 论文: REINFORCE Pro Max

### 论文目录结构

```
tex/
├── main.tex                    # 主控文件（摘要、宏定义、章节引入）
├── main/
│   ├── 1-intro.tex             # 引言 + 贡献列表
│   ├── 2-related.tex           # 相关工作
│   ├── 3-preliminaries.tex     # 符号、假设、surrogate 目标、IS 方法综述
│   ├── 4-method.tex            # 核心方法（REINFORCE Max + REINFORCE Pro）
│   ├── 5-theory.tex            # 统一框架、算法伪代码、方法对比表
│   ├── 6-experiment.tex        # 实验（占位中）
│   ├── 7-conclusion.tex        # 结论
│   └── appendix.tex            # 附录（RLOO 证明、γ=1 证明、prefix IS 示例、数值稳定性、uniform scale）
├── main/*_review.md            # 各章节审稿意见
└── literature/                 # 参考论文（Trust Region Masking 等）
```

### 论文章节与核心定理

| 章节 | 文件 | 核心内容 |
|------|------|---------|
| §4 REINFORCE Max | `4-method.tex` | RLOO baseline (`def:rloo`)、token expansion (`eq:token-advantage`)、adaptive normalization (`prop:alpha-beta`, `prop:gradient-direction`) |
| §4 REINFORCE Pro | `4-method.tex` | Prefix IS (`def:prefix-is`)、masking theorem (`thm:prefix-tighter`)、Adaptive bound 连接 (`thm:prefix-adaptive`) |
| §5 Unified Framework | `5-theory.tex` | 算法伪代码 (`alg:promax`)、方法对比表 (`tab:comparison`) |
| Appendix | `appendix.tex` | RLOO 性质证明 (`app:rloo-proof`)、γ=1 证明 (`app:gamma-one`)、prefix IS 示例 (`app:prefix-proof`)、数值稳定性 (`app:numerical`)、uniform scale (`app:uniform-scale`) |

### 论文修改规范

- 修改 theorem/proposition/proof 时，检查所有 `\Cref` 引用是否一致
- 从正文移动内容到附录时：保留 `\label`，正文用 `\Cref{app:xxx}` 引用
- 修改公式后，用 grep 检查相关符号在全文中的残留
- 摘要 (`main.tex`) 和贡献列表 (`1-intro.tex`) 的措辞需与 theorem 条件范围严格一致
- 附录中的 illustrative examples 是具体数值示例，不构成一般性证明

### 当前论文状态

- 不使用 reference model / KL penalty（已从公式中移除 `kl_t`, `\piref`, `\lambda_{\mathrm{KL}}`）
- Token expansion 简化为 $A_{i,t} = \tilde{r}_i$（sparse reward, γ=1）
- RLOO 详细证明在附录，正文只保留定义和简要说明
- Uniform scale 详细内容在附录，正文只有简短 remark
- 实验章节为占位文本

## 代码规范

- **行长度**: 119 字符 (black, isort, ruff)
- **Python 版本**: >= 3.10
