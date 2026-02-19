# 工具模块

路径: `openrlhf/utils/`

## agent.py - Agent 工具

### AgentExecutorBase

Agent 执行器基类，定义统一接口。

```python
class AgentExecutorBase(ABC):
    @abstractmethod
    async def execute(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        llm_engine,
    ) -> dict:
        """执行 agent 交互，返回 Experience 字典"""
        pass
```

### SingleTurnAgentExecutor

单轮执行器，支持可选奖励函数。

```python
executor = SingleTurnAgentExecutor(remote_rm_url=None)
```

**功能**:
- 单次生成
- 可选调用 reward_func 计算奖励

### MultiTurnAgentExecutor

多轮执行器，管理 Agent 实例生命周期。

```python
executor = MultiTurnAgentExecutor(AgentInstance)
```

**功能**:
- 管理多轮交互
- 调用 `reset()` 和 `step()`

### AgentInstanceBase

Agent 实例基类，用户需要继承实现。

```python
class AgentInstanceBase(ABC):
    @abstractmethod
    async def reset(self, states: dict, **kwargs) -> dict:
        """初始化环境，返回初始观测"""
        pass

    @abstractmethod
    async def step(self, states: dict, **kwargs) -> dict:
        """执行一步，返回 rewards, scores, feedback, done"""
        pass
```

---

## utils.py - 通用工具

### get_tokenizer()

获取 tokenizer。

```python
tokenizer = get_tokenizer(
    pretrain,
    model,
    padding_side="left",
    strategy=strategy,
    use_fast=True,
)
```

### zero_pad_sequences()

零填充序列。

```python
padded = zero_pad_sequences(
    sequences,
    side="right",  # left/right
    value=0,
)
```

---

## logging_utils.py - 日志工具

### WandbLogger

Wandb 日志记录器。

```python
logger = WandbLogger(args)
logger.log_train(step, metrics)
logger.log_eval(step, metrics)
logger.close()
```

### TensorboardLogger

TensorBoard 日志记录器。

```python
logger = TensorboardLogger(args)
logger.log_train(step, metrics)
logger.close()
```

### init_logger()

初始化 Python logger。

```python
logger = init_logger(__name__)
```

---

## deepspeed/ - DeepSpeed 工具

路径: `openrlhf/utils/deepspeed/`

### DeepspeedStrategy

DeepSpeed 训练策略。

**关键方法**:
- `setup_distributed()` - 设置分布式环境
- `prepare()` - 准备模型和优化器
- `backward()` - 反向传播
- `optimizer_step()` - 优化器步进
- `save_model()` - 保存模型
- `save_ckpt()` / `load_ckpt()` - 检查点管理
- `get_ds_train_config()` - 获取训练配置
- `get_ds_eval_config()` - 获取评估配置

### deepspeed_utils.py

**关键函数**:
- `offload_deepspeed_states()` - 卸载状态到 CPU
- `reload_deepspeed_states()` - 重新加载状态

---

## distributed_util.py - 分布式工具

**关键函数**:
- `stateless_init_process_group()` - 无状态初始化进程组
- `torch_dist_barrier_and_cuda_sync()` - 同步屏障

---

## seqlen_balancing.py - 序列长度平衡

**关键函数**:
- `get_seqlen_balanced_partitions()` - 获取平衡分区
- `get_minimum_num_micro_batch_size()` - 计算最小微批次数
