# Ray 分布式模块

路径: `openrlhf/trainer/ray/`

## ppo_actor.py - Actor 模型

**关键类**:
- `PolicyModelActor` - Ray Actor 封装
- `ActorPPOTrainer` - Actor 训练逻辑

**关键函数**:
- `init_model_from_pretrained()` - 模型初始化
- `fit()` - 训练一轮
- `forward()` - 前向计算 log_probs
- `broadcast_to_vllm()` - 同步权重到 vLLM
- `training_step()` - 单步训练
- `save_checkpoint()` - 保存检查点
- `reload_states()` / `offload_states()` - 睡眠模式

---

## ppo_critic.py - Critic 模型

**关键类**:
- `CriticModelActor` - Ray Actor 封装
- `CriticPPOTrainer` - Critic 训练逻辑

**关键函数**:
- `forward()` - 计算 value
- `fit()` - 训练一轮

---

## launcher.py - Ray Actor 组管理

**关键类**:
- `RayActorGroup` - Actor 组管理，支持批量调用
- `ReferenceModelActor` - Reference 模型 Actor
- `RewardModelActor` - Reward 模型 Actor
- `BaseModelActor` - 模型 Actor 基类
- `BaseDistributedActor` - 分布式 Actor 基类

**关键函数**:
- `async_run_method()` - 异步调用方法
- `async_run_method_batch()` - 批量异步调用
- `async_init_model_from_pretrained()` - 异步初始化
- `execute_batch()` - 执行批量任务

---

## vllm_engine.py - vLLM 引擎

**关键类**:
- `LLMRayActor` - vLLM Ray Actor，异步生成

**关键函数**:
- `create_vllm_engines()` - 创建 vLLM 引擎
- `batch_vllm_engine_call()` - 批量调用引擎
- `generate()` - Token 级生成
- `generate_responses()` - 生成响应 (调用 executor)
- `update_weight()` - 更新权重 (NCCL broadcast)
- `update_weight_cuda_ipc()` - CUDA IPC 更新权重
- `sleep()` / `wake_up()` - 睡眠/唤醒模式
- `init_process_group()` - 初始化进程组

---

## vllm_worker_wrap.py - vLLM Worker 扩展

**关键类**:
- `WorkerWrap` - Worker 扩展，支持权重同步

**功能**:
- 扩展 vLLM worker 支持从 DeepSpeed 接收权重更新
- 实现 `init_process_group()` 和 `update_weight()` RPC

---

## utils.py - 工具函数

**关键函数**:
- `get_physical_gpu_id()` - 获取物理 GPU ID
- `get_bundle_indices()` - 获取 placement group bundle 索引
- `ray_noset_visible_devices()` - 检查 RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES
