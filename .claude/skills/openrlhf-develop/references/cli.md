# CLI 入口模块

路径: `openrlhf/cli/`

## train_ppo_ray.py - PPO/REINFORCE++/GRPO/RLOO 训练

**入口**: `python -m openrlhf.cli.train_ppo_ray`

**功能**:
- 解析参数，创建 Ray Actor 组
- 初始化 vLLM 引擎
- 启动 PPOTrainer

**关键函数**:
- `main()` - 主入口
- `create_actor_groups()` - 创建模型 Actor 组
- `parse_args()` - 参数解析

**支持算法** (`--advantage_estimator`):
- `gae` - PPO (default)
- `reinforce` - REINFORCE++
- `reinforce_baseline` - REINFORCE++-baseline
- `group_norm` - GRPO
- `rloo` - RLOO
- `dr_grpo` - Dr. GRPO

---

## train_sft.py - SFT 训练

**入口**: `deepspeed --module openrlhf.cli.train_sft`

**功能**: 监督微调训练

**关键参数**:
- `--dataset` - 数据集
- `--input_key` / `--output_key` - 输入输出键
- `--packing_samples` - 样本打包
- `--multiturn` - 多轮损失

---

## train_rm.py - 奖励模型训练

**入口**: `deepspeed --module openrlhf.cli.train_rm`

**功能**: 训练奖励模型

**关键参数**:
- `--chosen_key` / `--rejected_key` - chosen/rejected 键
- `--normalize_reward` - 奖励归一化
- `--value_head_prefix` - value head 前缀

---

## train_dpo.py - DPO 训练

**入口**: `deepspeed --module openrlhf.cli.train_dpo`

**功能**: DPO/IPO/cDPO 训练

**关键参数**:
- `--beta` - DPO beta 参数
- `--loss_type` - 损失类型 (dpo/ipo/cdpo)
- `--label_smoothing` - 标签平滑

---

## train_kto.py - KTO 训练

**入口**: `deepspeed --module openrlhf.cli.train_kto`

**功能**: Kahneman-Tversky 优化训练

---

## train_kd.py - 知识蒸馏

**入口**: `deepspeed --module openrlhf.cli.train_kd`

**功能**: 知识蒸馏训练

---

## train_prm.py - Process Reward Model

**入口**: `deepspeed --module openrlhf.cli.train_prm`

**功能**: 过程奖励模型训练

---

## batch_inference.py - 批量推理

**入口**: `python -m openrlhf.cli.batch_inference`

**功能**: 批量生成推理

---

## lora_combiner.py - LoRA 权重合并

**入口**: `python -m openrlhf.cli.lora_combiner`

**功能**: 合并 LoRA 权重到基础模型

**关键参数**:
- `--model_path` - 基础模型路径
- `--lora_path` - LoRA 权重路径
- `--output_path` - 输出路径
- `--is_rm` - 是否为奖励模型

---

## serve_rm.py - 奖励模型服务

**入口**: `python -m openrlhf.cli.serve_rm`

**功能**: 启动 HTTP 奖励模型服务

---

## interactive_chat.py - 交互聊天

**入口**: `python -m openrlhf.cli.interactive_chat`

**功能**: 交互式对话测试
