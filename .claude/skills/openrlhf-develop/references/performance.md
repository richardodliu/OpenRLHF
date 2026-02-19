# 性能调优

## GPU 资源分配

### 推荐比例

```
vLLM : Actor : Critic = 1 : 1 : 1
```

### 配置示例

#### 8x A100 (80GB) - Hybrid Engine

```bash
--actor_num_gpus_per_node 8 \
--critic_num_gpus_per_node 8 \
--ref_num_gpus_per_node 8 \
--reward_num_gpus_per_node 8 \
--vllm_num_engines 4 \
--vllm_tensor_parallel_size 2 \
--colocate_all_models \
--vllm_gpu_memory_utilization 0.5 \
--vllm_enable_sleep \
--deepspeed_enable_sleep
```

#### 48x A100 - 70B 模型

```bash
# 16 GPUs → vLLM (TP=4, 4 engines)
# 16 GPUs → Actor (ZeRO-3)
# 16 GPUs → Critic (ZeRO-3)

--actor_num_nodes 2 \
--actor_num_gpus_per_node 8 \
--critic_num_nodes 2 \
--critic_num_gpus_per_node 8 \
--vllm_num_engines 4 \
--vllm_tensor_parallel_size 4 \
--zero_stage 3
```

---

## 速度优化

| 优化项 | 参数 | 效果 | 适用场景 |
|--------|------|------|----------|
| **Hybrid Engine** | `--colocate_all_models` | 减少 GPU 数量 | GPU 显存充足 |
| **vLLM Sleep** | `--vllm_enable_sleep` | 释放显存给训练 | Hybrid 模式 |
| **DeepSpeed Sleep** | `--deepspeed_enable_sleep` | 释放显存给推理 | Hybrid 模式 |
| **样本打包** | `--packing_samples` | 提高利用率 | 总是开启 |
| **动态批次** | `--use_dynamic_batch` | 平衡序列长度 | 变长序列 |
| **异步训练** | `--async_train` | 重叠生成和训练 | 验证收敛后 |
| **DeepCompile** | `--deepcompile` | 加速训练 | PyTorch 2.0+ |
| **通信重叠** | `--overlap_comm` | 隐藏通信延迟 | 显存充足 |
| **前缀缓存** | `--enable_prefix_caching` | 复用 KV cache | n_samples > 1 |

### vLLM 调优

```bash
# 最大化生成吞吐
--micro_rollout_batch_size 16 \      # 尽可能大
--vllm_tensor_parallel_size 2 \       # 尽可能小
--vllm_sync_backend nccl \            # 必须使用 nccl
--enforce_eager                        # 如果显存紧张
```

### 训练调优

```bash
# 最大化训练吞吐
--micro_train_batch_size 4 \          # 尽可能大
--packing_samples \                    # 样本打包
--gradient_checkpointing \             # 如果 OOM
--use_dynamic_batch \                  # 动态批次
--train_max_tokens_per_gpu 16384       # 每 GPU token 上限
```

---

## 内存优化

### 显存充足时

```bash
# 禁用 CPU 卸载
# 不使用 --adam_offload

# 启用通信重叠
--overlap_comm

# 启用模型共置
--colocate_all_models
--colocate_actor_ref
--colocate_critic_reward
```

### 显存不足时

```bash
# 禁用共置
# 不使用 --colocate_* 选项

# 启用梯度检查点
--gradient_checkpointing

# 启用 Adam 卸载
--adam_offload

# 减小批次大小
--micro_train_batch_size 1
--micro_rollout_batch_size 4

# 使用 ZeRO-3
--zero_stage 3

# Reference/Reward 模型卸载
--ref_reward_offload
```

### 超长序列

```bash
# 启用 Ring Attention
--ring_attn_size 4 \
--ring_head_stride 2

# 或 DeepSpeed TP
--ds_tensor_parallel_size 2
```

---

## 稳定性优化

### KL 散度控制

```bash
# 固定 KL 系数 (推荐)
--init_kl_coef 0.01

# 自适应 KL (可选)
--init_kl_coef 0.01 \
--kl_target 6.0 \
--kl_horizon 10000
```

### PPO Clip

```bash
# 标准设置
--eps_clip 0.2

# 更保守
--eps_clip_low_high 0.2 0.28

# Dual clip (更稳定)
--dual_clip 4.0
```

### 奖励归一化

```bash
# 训练奖励模型时
--normalize_reward

# PPO 训练时
--reward_clip_range -10 10
```

---

## 常见问题

### OOM (显存不足)

1. 减小 `--micro_train_batch_size`
2. 减小 `--micro_rollout_batch_size`
3. 启用 `--gradient_checkpointing`
4. 使用 `--zero_stage 3`
5. 禁用 `--colocate_*` 选项

### 训练不收敛

1. 降低学习率: `--actor_learning_rate 1e-7`
2. 增大 KL 系数: `--init_kl_coef 0.1`
3. 使用更保守的 clip: `--eps_clip 0.1`
4. 检查奖励模型质量

### vLLM 同步失败

1. 设置 `export NCCL_CUMEM_ENABLE=0`
2. 设置 `export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`
3. 使用 `--vllm_sync_backend nccl`

### 生成质量差

1. 调整温度: `--temperature 0.7`
2. 使用 top-p: `--top_p 0.9`
3. 增加生成长度: `--generate_max_len 2048`
4. 检查 prompt 格式

---

## 监控指标

### 关键指标

| 指标 | 正常范围 | 说明 |
|------|----------|------|
| `policy_loss` | -0.5 ~ 0.5 | PPO 策略损失 |
| `value_loss` | 0 ~ 10 | Critic 损失 |
| `kl` | 0 ~ 0.5 | KL 散度 |
| `reward` | 取决于任务 | 平均奖励 |
| `clip_ratio` | 0.1 ~ 0.3 | PPO clip 比例 |
| `entropy` | > 0 | 策略熵 |

### Wandb 配置

```bash
--use_wandb YOUR_API_KEY
```

### TensorBoard 配置

```bash
--use_tensorboard ./runs
tensorboard --logdir ./runs
```
