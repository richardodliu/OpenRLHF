---
name: openrlhf-develop
description: OpenRLHF 开发参考技能。当用户在 OpenRLHF 项目中工作、需要了解 RLHF 训练架构、编写 PPO/REINFORCE++/GRPO 训练代码、或配置分布式训练时自动激活。
---

# OpenRLHF 开发技能

OpenRLHF 是用于 LLM 强化学习后训练的高性能框架，基于 **Ray + vLLM** 分布式架构。

GitHub: https://github.com/OpenRLHF/OpenRLHF

## 核心架构

- **Ray**: 分布式调度，分离 Actor/Critic/Reward/Reference 模型
- **vLLM**: 高性能推理，RLHF 80% 时间用于生成
- **DeepSpeed**: ZeRO-3 内存优化，支持 70B+ 模型
- **Hybrid Engine**: `--colocate_all_models` 共享 GPU

## RL 算法

| 算法 | `--advantage_estimator` | 说明 |
|------|------------------------|------|
| PPO | (default) | 完整 Critic |
| REINFORCE++ | `reinforce` | 无 Critic |
| REINFORCE++-baseline | `reinforce_baseline` | 推理任务推荐 |
| GRPO | `group_norm` | 组归一化 |
| RLOO | `rloo` | Per-token KL |

## 关键源码

- PPO 训练: `openrlhf/trainer/ppo_trainer.py` → `PPOTrainer`, `train_step()`
- Ray Actor: `openrlhf/trainer/ray/ppo_actor.py` → `PolicyModelActor`, `broadcast_to_vllm()`
- 经验收集: `openrlhf/trainer/ppo_utils/experience_maker.py` → `Experience`, `RemoteExperienceMaker`
- vLLM 引擎: `openrlhf/trainer/ray/vllm_engine.py` → `LLMRayActor`, `create_vllm_engines()`
- CLI 入口: `openrlhf/cli/train_ppo_ray.py`

## 示例脚本

- PPO Hybrid: `examples/scripts/train_ppo_ray_hybrid_engine.sh`
- REINFORCE++: `examples/scripts/train_reinforce_baseline_hybrid_engine.sh`
- 自定义奖励: `examples/scripts/train_ppo_with_reward_fn.sh`
- 多轮 Agent: `examples/scripts/train_reinforce_baseline_ray_agent_async.sh`

## 常用配置

```bash
# 分布式
--actor_num_gpus_per_node 8 --vllm_num_engines 4 --vllm_tensor_parallel_size 2

# Hybrid Engine
--colocate_all_models --vllm_enable_sleep --deepspeed_enable_sleep

# 训练
--train_batch_size 128 --rollout_batch_size 1024 --packing_samples --use_dynamic_batch
```

## 详细参考

**源码模块**:
- `references/ray.md` - Ray 分布式模块 (`openrlhf/trainer/ray/`)
- `references/trainer.md` - 训练器模块 (`openrlhf/trainer/`)
- `references/ppo_utils.md` - PPO 工具 (`openrlhf/trainer/ppo_utils/`)
- `references/cli.md` - CLI 入口 (`openrlhf/cli/`)
- `references/models.md` - 模型模块 (`openrlhf/models/`)
- `references/datasets.md` - 数据集 (`openrlhf/datasets/`)
- `references/utils.md` - 工具函数 (`openrlhf/utils/`)

**算法详解**:
- `references/algorithms.md` - RL 算法详解 (PPO/REINFORCE++/GRPO/RLOO)

**使用指南**:
- `references/docs.md` - 文档和资源链接
- `references/examples.md` - 示例脚本详解
- `references/config.md` - 配置参数大全
- `references/workflows.md` - 典型工作流
- `references/performance.md` - 性能调优
