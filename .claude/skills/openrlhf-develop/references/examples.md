# 示例脚本参考

## 训练脚本 (`examples/scripts/`)

### SFT 和预训练

| 脚本 | 说明 |
|------|------|
| `train_sft.sh` | 监督微调 (SFT) |
| `train_sft_mixtral_lora.sh` | Mixtral LoRA SFT |
| `train_conditional.sh` | 条件 SFT |
| `train_knowledge_distillation.sh` | 知识蒸馏 |

### 奖励模型

| 脚本 | 说明 |
|------|------|
| `train_rm.sh` | 奖励模型训练 |
| `train_prm_mistral.sh` | Process Reward Model |
| `serve_remote_rm.sh` | 远程奖励模型服务 |

### PPO / REINFORCE++ 训练

| 脚本 | 说明 |
|------|------|
| `train_ppo_ray_hybrid_engine.sh` | PPO + Hybrid Engine (推荐) |
| `train_ppo_ray_slurm.sh` | PPO + Slurm 集群 |
| `train_ppo_with_reward_fn.sh` | 自定义奖励函数 |
| `train_reinforce_baseline_hybrid_engine.sh` | REINFORCE++-baseline |
| `train_reinforce_baseline_ray_agent_async.sh` | 异步多轮 Agent |
| `train_dapo_ray_hybrid_engine.sh` | DAPO 动态过滤 |

### DPO / KTO 训练

| 脚本 | 说明 |
|------|------|
| `train_dpo_llama.sh` | DPO/IPO/cDPO |
| `train_kto_llama.sh` | KTO |
| `train_iterative_dpo.sh` | 迭代 DPO |

### 其他

| 脚本 | 说明 |
|------|------|
| `train_rejection_sampling_llama.sh` | 拒绝采样 |
| `train_nonrl_slurm.sh` | 非 RL Slurm 训练 |
| `ckpt_ds_zero_to_universal.sh` | 检查点转换 |

---

## Python 示例 (`examples/python/`)

### 自定义奖励函数

**文件**: `reward_func.py`

```python
import torch

def reward_func(queries, prompts, labels, **kwargs):
    """
    Args:
        queries: List[str] - 完整文本 (prompt + response)
        prompts: List[str] - 原始 prompt
        labels: List[str] - 标签 (来自 --label_key)
    Returns:
        dict: rewards, scores, extra_logs
    """
    reward = torch.randint(0, 2, (len(queries),)).float()
    return {
        "rewards": reward,           # 用于 advantage 计算
        "scores": reward,            # 0-1 用于动态过滤
        "extra_logs": {"metric": reward.mean()},
    }
```

**使用**:
```bash
--remote_rm_url /path/to/reward_func.py --label_key answer
```

### 多轮 Agent

**文件**: `agent_func.py`

```python
from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor

class AgentInstance(AgentInstanceBase):
    async def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.max_steps = 3

    async def reset(self, states, **kwargs):
        return {"observation": states["observation"]}

    async def step(self, states, **kwargs):
        done = self.step_idx >= self.max_steps
        reward = torch.randint(0, 2, (1,)).float() if done else torch.tensor(0)
        self.step_idx += 1
        return {
            "rewards": reward,
            "scores": reward,
            "environment_feedback": "feedback text",
            "done": done,
            "sampling_params": states.get("sampling_params"),
            "extra_logs": {},
        }

class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
```

**使用**:
```bash
--agent_func_path /path/to/agent_func.py --async_train
```

### NeMo Gym 集成

**文件**: `agent_func_nemogym_executor.py`

用于与 NVIDIA NeMo Gym 环境集成的 Agent 执行器示例。

### GEM 多轮对话

**文件**: `agent_func_gem_multiturn.py`

用于 GEM benchmark 多轮对话任务的 Agent 示例。
