# 典型工作流

## 完整 RLHF Pipeline

### Step 1: SFT 监督微调

```bash
deepspeed --module openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --input_template $'User: {}\nAssistant: ' \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path ./checkpoint/llama3-8b-sft \
   --zero_stage 2 \
   --max_epochs 1 \
   --packing_samples \
   --param_dtype bf16 \
   --learning_rate 5e-6 \
   --gradient_checkpointing
```

### Step 2: 奖励模型训练

```bash
deepspeed --module openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3-8b-rm \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain ./checkpoint/llama3-8b-sft \
   --param_dtype bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset OpenRLHF/preference_dataset_mixture2 \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --packing_samples \
   --gradient_checkpointing
```

### Step 3: PPO 训练

```bash
# 启动 Ray
ray start --head --num-gpus 8

# 提交训练任务
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.5 \
   --pretrain ./checkpoint/llama3-8b-sft \
   --reward_pretrain ./checkpoint/llama3-8b-rm \
   --save_path ./checkpoint/llama3-8b-rlhf \
   --train_batch_size 128 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --param_dtype bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --use_dynamic_batch
```

---

## 自定义奖励函数

### 创建奖励函数

```python
# reward_func.py
import torch

def reward_func(queries, prompts, labels, **kwargs):
    """
    计算自定义奖励。

    Args:
        queries: List[str] - 完整文本 (prompt + response)
        prompts: List[str] - 原始 prompt
        labels: List[str] - 标签 (来自 --label_key)

    Returns:
        dict:
            - rewards: 用于 advantage 计算
            - scores: 0-1 范围，用于动态过滤
            - extra_logs: 额外日志 (Wandb)
    """
    rewards = []
    for query, label in zip(queries, labels):
        # 实现你的奖励逻辑
        # 例如：代码执行、数学验证、格式检查
        reward = your_reward_logic(query, label)
        rewards.append(reward)

    rewards = torch.tensor(rewards).float()
    return {
        "rewards": rewards,
        "scores": rewards.clamp(0, 1),  # 归一化到 0-1
        "extra_logs": {"custom_metric": rewards.mean().item()},
    }
```

### 使用奖励函数训练

```bash
ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --remote_rm_url /path/to/reward_func.py \
   --label_key answer \
   --prompt_data your_dataset \
   --advantage_estimator reinforce_baseline \
   --n_samples_per_prompt 4 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep
```

---

## 多轮 Agent

### 创建 Agent

```python
# agent_func.py
import torch
from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

class AgentInstance(AgentInstanceBase):
    async def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.max_steps = 3
        self.history = []

    async def reset(self, states, **kwargs):
        """初始化环境，返回初始观测。"""
        self.step_idx = 0
        self.history = []
        return {"observation": states["observation"]}

    async def step(self, states, **kwargs):
        """执行一步交互。"""
        observation_text = states["observation_text"]
        action_text = states["action_text"]
        label = states["label"]

        self.history.append(action_text)
        self.step_idx += 1

        # 检查是否完成
        done = self.step_idx >= self.max_steps or self.check_success(action_text, label)

        # 计算奖励
        if done:
            reward = self.calculate_final_reward(self.history, label)
        else:
            reward = torch.tensor(0.0)

        # 生成环境反馈
        if done:
            feedback = "\n\nHuman: [DONE]\n</s>"
        else:
            feedback = f"\n\nHuman: Step {self.step_idx} received. Continue.\n</s>\n\nAssistant: "

        return {
            "rewards": reward,
            "scores": reward.clamp(0, 1),
            "environment_feedback": feedback,
            "done": done,
            "sampling_params": states.get("sampling_params"),
            "extra_logs": {"step": self.step_idx},
        }

    def check_success(self, action, label):
        # 实现成功检查逻辑
        return False

    def calculate_final_reward(self, history, label):
        # 实现最终奖励计算
        return torch.tensor(1.0)


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
```

### 使用 Agent 训练

```bash
ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --agent_func_path /path/to/agent_func.py \
   --prompt_data your_dataset \
   --input_key prompt \
   --label_key label \
   --advantage_estimator reinforce_baseline \
   --async_train \
   --n_samples_per_prompt 4 \
   --dynamic_filtering \
   --dynamic_filtering_reward_range 0.0 1.0
```

---

## DPO 训练

```bash
deepspeed --module openrlhf.cli.train_dpo \
   --save_path ./checkpoint/llama3-8b-dpo \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain ./checkpoint/llama3-8b-sft \
   --param_dtype bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset OpenRLHF/preference_dataset \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --packing_samples \
   --gradient_checkpointing
```

---

## 迭代 DPO

```bash
# 第一轮
bash examples/scripts/train_iterative_dpo.sh --iteration 1

# 第二轮 (使用上一轮模型)
bash examples/scripts/train_iterative_dpo.sh --iteration 2 \
   --pretrain ./checkpoint/llama3-8b-dpo-iter1
```

---

## LoRA 训练与合并

### LoRA SFT

```bash
deepspeed --module openrlhf.cli.train_sft \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --lora_rank 64 \
   --lora_alpha 128 \
   --target_modules q_proj,v_proj,k_proj,o_proj \
   --save_path ./checkpoint/llama3-8b-sft-lora
```

### 合并 LoRA 权重

```bash
python -m openrlhf.cli.lora_combiner \
   --model_path meta-llama/Meta-Llama-3-8B \
   --lora_path ./checkpoint/llama3-8b-sft-lora \
   --output_path ./checkpoint/llama3-8b-sft-merged \
   --param_dtype bf16
```

---

## 多节点训练 (Slurm)

```bash
#!/bin/bash
#SBATCH --job-name=openrlhf
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

# 启动 Ray 集群
srun --nodes=1 --ntasks=1 ray start --head --port=6379 &
sleep 10
srun --nodes=3 --ntasks=3 ray start --address=$HEAD_NODE:6379 &
sleep 10

# 提交训练任务
ray job submit --address="http://$HEAD_NODE:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --actor_num_nodes 4 \
   --actor_num_gpus_per_node 8 \
   ...
```
