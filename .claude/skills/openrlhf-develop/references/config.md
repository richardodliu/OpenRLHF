# 配置参数参考

## 模型与数据

```bash
--pretrain MODEL             # 预训练模型路径或 HuggingFace 名称
--reward_pretrain MODEL      # 奖励模型路径
--critic_pretrain MODEL      # Critic 模型路径 (默认同 pretrain)
--prompt_data DATASET        # Prompt 数据集
--input_key KEY              # 数据集输入键名
--label_key KEY              # 数据集标签键名 (用于 reward_func)
--output_key KEY             # 数据集输出键名 (SFT)
--apply_chat_template        # 使用 HF tokenizer chat template
--input_template TEMPLATE    # 自定义输入模板
--max_samples N              # 最大样本数
--prompt_split SPLIT         # 数据集划分 (train/test)
```

---

## 分布式配置

### 模型放置

```bash
--actor_num_nodes N          # Actor 节点数
--actor_num_gpus_per_node N  # Actor 每节点 GPU 数
--critic_num_nodes N         # Critic 节点数
--critic_num_gpus_per_node N # Critic 每节点 GPU 数
--ref_num_nodes N            # Reference 节点数
--ref_num_gpus_per_node N    # Reference 每节点 GPU 数
--reward_num_nodes N         # Reward 节点数
--reward_num_gpus_per_node N # Reward 每节点 GPU 数
```

### vLLM 配置

```bash
--vllm_num_engines N         # vLLM 引擎数量
--vllm_tensor_parallel_size N # vLLM 张量并行度
--vllm_gpu_memory_utilization F # GPU 显存利用率 (0-1)
--vllm_sync_backend BACKEND  # 同步后端 (nccl/gloo)
--enforce_eager              # 禁用 CUDA Graph
--enable_prefix_caching      # 启用前缀缓存
```

### Hybrid Engine (模型共置)

```bash
--colocate_all_models        # 所有模型共享 GPU
--colocate_actor_ref         # Actor 和 Reference 共享
--colocate_critic_reward     # Critic 和 Reward 共享
--vllm_enable_sleep          # vLLM 睡眠模式节省显存
--deepspeed_enable_sleep     # DeepSpeed 睡眠模式
```

### DeepSpeed 配置

```bash
--zero_stage {0,1,2,3}       # ZeRO 阶段
--gradient_checkpointing     # 梯度检查点
--adam_offload               # Adam 卸载到 CPU
--overlap_comm               # 通信重叠
--ds_tensor_parallel_size N  # DeepSpeed 张量并行
--ring_attn_size N           # Ring Attention 大小
--ring_head_stride N         # Ring Attention head 步长
```

---

## 训练参数

### 批次大小

```bash
--train_batch_size N         # 训练批次大小
--rollout_batch_size N       # Rollout 批次大小
--micro_train_batch_size N   # 微批次大小 (训练)
--micro_rollout_batch_size N # 微批次大小 (Rollout)
--n_samples_per_prompt N     # 每个 prompt 采样数
```

### 学习率

```bash
--actor_learning_rate LR     # Actor 学习率
--critic_learning_rate LR    # Critic 学习率
--lr_scheduler TYPE          # 调度器类型
--lr_warmup_ratio R          # Warmup 比例
```

### 训练轮次

```bash
--max_epochs N               # 最大 epoch 数
--num_episodes N             # Episode 数
--max_steps N                # 最大步数
```

### KL 控制

```bash
--init_kl_coef F             # 初始 KL 系数
--kl_target F                # KL 目标值 (自适应)
--kl_horizon N               # KL 自适应窗口
--use_kl_loss                # 使用 KL 损失
--kl_estimator {k1,k2,k3}    # KL 估计器
```

### PPO 参数

```bash
--eps_clip F                 # PPO clip 范围
--eps_clip_low_high L H      # PPO clip 低高范围
--dual_clip F                # Dual clip 阈值
--gamma F                    # 折扣因子
--lambd F                    # GAE lambda
```

---

## 算法选择

```bash
--advantage_estimator TYPE   # 优势估计器:
                             #   gae - PPO (default)
                             #   reinforce - REINFORCE++
                             #   reinforce_baseline - REINFORCE++-baseline
                             #   group_norm - GRPO
                             #   rloo - RLOO
                             #   dr_grpo - Dr. GRPO
```

---

## Agent 模式

```bash
--agent_func_path PATH       # 多轮 Agent 函数路径
--remote_rm_url PATH_OR_URL  # 自定义奖励函数路径或 HTTP URL
--async_train                # 异步训练
--async_queue_size N         # 异步队列大小
```

### 动态过滤 (DAPO)

```bash
--dynamic_filtering          # 启用动态过滤
--dynamic_filtering_reward_range L H  # 过滤阈值范围 [L, H]
```

---

## 高级优化

### 样本处理

```bash
--packing_samples            # 样本打包
--use_dynamic_batch          # 动态批次
--train_max_tokens_per_gpu N # 每 GPU 最大 token (训练)
--rollout_max_tokens_per_gpu N # 每 GPU 最大 token (Rollout)
```

### 生成参数

```bash
--prompt_max_len N           # Prompt 最大长度
--generate_max_len N         # 生成最大长度
--temperature F              # 采样温度
--top_p F                    # Top-p 采样
--top_k N                    # Top-k 采样
```

### 精度与内存

```bash
--param_dtype {bf16,fp16,fp32}  # 参数精度
--attn_implementation IMPL   # 注意力实现 (flash_attention_2)
--load_in_4bit               # 4bit 量化加载
```

### LoRA

```bash
--lora_rank N                # LoRA rank
--lora_alpha N               # LoRA alpha
--lora_dropout F             # LoRA dropout
--target_modules MODULES     # LoRA 目标模块
```

### vLLM IS 校正

```bash
--enable_vllm_is_correction  # 启用 IS 校正
--vllm_is_truncated_threshold L H  # IS 截断阈值
--use_icepop                 # 使用 ICEPOP
```

---

## 检查点与日志

```bash
--save_path PATH             # 模型保存路径
--ckpt_path PATH             # 检查点路径
--load_checkpoint            # 加载检查点
--save_steps N               # 保存间隔
--save_hf_ckpt               # 保存 HF 格式
--max_ckpt_num N             # 最大检查点数
--logging_steps N            # 日志间隔
--eval_steps N               # 评估间隔
--use_wandb TOKEN            # 使用 Wandb
--use_tensorboard PATH       # 使用 TensorBoard
```

---

## 典型配置组合

### 8x A100 Hybrid Engine

```bash
--actor_num_gpus_per_node 8 \
--critic_num_gpus_per_node 8 \
--ref_num_gpus_per_node 8 \
--reward_num_gpus_per_node 8 \
--vllm_num_engines 4 \
--vllm_tensor_parallel_size 2 \
--colocate_all_models \
--vllm_enable_sleep \
--deepspeed_enable_sleep \
--vllm_gpu_memory_utilization 0.5
```

### REINFORCE++-baseline (推理任务)

```bash
--advantage_estimator reinforce_baseline \
--n_samples_per_prompt 8 \
--init_kl_coef 1e-5 \
--use_kl_loss \
--kl_estimator k2
```

### 动态过滤

```bash
--dynamic_filtering \
--dynamic_filtering_reward_range 0.0 1.0 \
--n_samples_per_prompt 8
```
