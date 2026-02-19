# PPO 工具模块

路径: `openrlhf/trainer/ppo_utils/`

## experience_maker.py - 经验收集与处理

### Experience 数据类

```python
@dataclass
class Experience:
    index: list[int]                    # 样本索引
    sequences: torch.Tensor             # (B, S) token 序列
    attention_mask: torch.Tensor        # (B, S) 注意力掩码
    action_mask: torch.Tensor           # (B, A) 动作掩码
    action_log_probs: torch.Tensor      # Actor log probs
    base_action_log_probs: torch.Tensor # Reference log probs
    rollout_log_probs: torch.Tensor     # vLLM rollout log probs
    values: torch.Tensor                # Critic values
    returns: torch.Tensor               # 回报
    advantages: torch.Tensor            # 优势
    rewards: torch.Tensor               # 奖励 (用于 advantage)
    scores: torch.Tensor                # 0-1 分数 (用于动态过滤)
    prompts: list[str]                  # 原始 prompt
    labels: list[str]                   # 标签
    info: dict                          # 额外信息 (用于日志)
```

**Experience 方法**:
- `to_device()` - 移动到设备
- `pin_memory()` - 固定内存
- `concat_experiences()` - 拼接多个经验
- `select()` - 选择特定字段

---

### SamplesGenerator - 样本生成器

**功能**: 从 prompt dataloader 拉取数据，分发到 vLLM 引擎生成

**关键函数**:
- `generate_samples()` - 生成训练样本
- `generate_eval_samples()` - 生成评估样本
- `_generate_vllm()` - vLLM 生成主逻辑
- `_dispatch_prompts_to_vllm()` - 分发 prompt 到 vLLM
- `_process_response_into_experience()` - 处理响应为 Experience

---

### RemoteExperienceMaker - 远程经验生成

**功能**: 调用远程模型计算 logprobs, values, rewards, kl

**关键函数**:
- `make_experience_batch()` - 生成经验批次 (完整流程)
- `make_experience()` - 计算 logprobs, values, rewards, kl
- `compute_advantages_and_returns()` - 计算优势和回报
- `split_rollout_samples()` - 分割样本到各 Actor
- `get_advantages_and_returns()` - GAE 计算
- `get_cumulative_returns()` - REINFORCE 累积回报

**优势估计方法**:
- `gae` - PPO GAE
- `reinforce` - REINFORCE++
- `reinforce_baseline` - REINFORCE++-baseline (推荐)
- `group_norm` - GRPO
- `rloo` - RLOO
- `dr_grpo` - Dr. GRPO

---

## replay_buffer.py - 经验回放

### NaiveReplayBuffer

**功能**: 简单的经验回放缓冲

**关键函数**:
- `append()` - 添加经验
- `clear()` - 清空缓冲
- `collate_fn()` - 批次整理函数
- `setup_dynamic_batch()` - 设置动态批次

### balance_experiences()

**功能**: 平衡经验分布到各 DP rank

---

## kl_controller.py - KL 控制

### FixedKLController

固定 KL 系数，推荐使用。

```python
controller = FixedKLController(init_kl_coef=0.01)
```

### AdaptiveKLController

自适应 KL 系数，根据实际 KL 调整。

```python
controller = AdaptiveKLController(
    init_kl_coef=0.01,
    target=6.0,
    horizon=10000
)
```

**关键函数**:
- `update()` - 更新 KL 系数
