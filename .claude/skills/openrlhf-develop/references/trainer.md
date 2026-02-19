# 训练器模块

路径: `openrlhf/trainer/`

## ppo_trainer.py - PPO 训练主逻辑

**关键类**:
- `PPOTrainer` - Ray remote PPO 训练器
- `BasePPOTrainer` - 训练基类

**关键函数**:
- `fit()` - 主训练循环
- `train_step()` - 单步训练 (生成经验 + PPO 更新)
- `ppo_train()` - PPO 优化步骤
- `broadcast_to_vllm()` - 同步权重到 vLLM
- `save_logs_and_checkpoints()` - 保存日志和检查点
- `init_checkpoint_states()` - 初始化检查点状态
- `evaluate()` - 评估

**辅助函数**:
- `prepare_datasets()` - 准备数据集

---

## ppo_trainer_async.py - 异步 PPO 训练

**关键类**:
- `AsyncPPOTrainer` - 异步训练器，重叠生成和训练

**特点**:
- 使用 `--async_train` 启用
- 通过队列实现生成和训练的流水线
- 适合多轮 Agent 场景

---

## dpo_trainer.py - DPO 训练

**关键类**:
- `DPOTrainer` - DPO/IPO/cDPO 训练器

**关键函数**:
- `fit()` - 训练循环
- `training_step()` - 计算 DPO loss
- `concatenated_forward()` - 拼接 chosen/rejected 前向

---

## sft_trainer.py - SFT 训练

**关键类**:
- `SFTTrainer` - 监督微调训练器

**关键函数**:
- `fit()` - 训练循环
- `training_step()` - 计算交叉熵 loss

---

## rm_trainer.py - 奖励模型训练

**关键类**:
- `RewardModelTrainer` - 奖励模型训练器

**关键函数**:
- `fit()` - 训练循环
- `training_step()` - 计算 ranking loss
- `concatenated_forward()` - 拼接 chosen/rejected

---

## kto_trainer.py - KTO 训练

**关键类**:
- `KTOTrainer` - Kahneman-Tversky 优化训练器

**关键函数**:
- `fit()` - 训练循环
- `training_step()` - 计算 KTO loss

---

## kd_trainer.py - 知识蒸馏

**关键类**:
- `KDTrainer` - 知识蒸馏训练器

---

## prm_trainer.py - Process Reward Model

**关键类**:
- `PRMTrainer` - 过程奖励模型训练器
