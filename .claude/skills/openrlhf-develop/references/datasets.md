# 数据集模块

路径: `openrlhf/datasets/`

## prompts_dataset.py - Prompt 数据集

**关键类**:
- `PromptDataset` - PPO 训练用 prompt 数据集

**输入格式**:
```json
{"input_key": "prompt text", "label_key": "answer"}
```

或使用 chat template:
```json
{"input_key": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"}
]}
```

**关键参数**:
- `--input_key` - 输入键名
- `--label_key` - 标签键名 (用于 reward_func)
- `--apply_chat_template` - 使用 chat template
- `--input_template` - 自定义输入模板

---

## reward_dataset.py - 奖励数据集

**关键类**:
- `RewardDataset` - 奖励模型训练数据集

**输入格式**:
```json
{
    "chosen_key": "better response",
    "rejected_key": "worse response"
}
```

**关键参数**:
- `--chosen_key` - chosen 响应键
- `--rejected_key` - rejected 响应键

---

## sft_dataset.py - SFT 数据集

**关键类**:
- `SFTDataset` - 监督微调数据集

**输入格式**:
```json
{"input_key": "prompt", "output_key": "response"}
```

**关键参数**:
- `--input_key` - 输入键名
- `--output_key` - 输出键名
- `--multiturn` - 多轮对话损失

---

## utils.py - 数据工具

### blending_datasets()

混合多个数据集。

```python
dataset = blending_datasets(
    datasets="dataset1,dataset2",
    probs="0.6,0.4",      # 采样概率
    strategy=strategy,
    seed=42,
    max_count=100000,
)
```

**使用**:
```bash
--prompt_data "dataset1,dataset2" \
--prompt_data_probs "0.6,0.4"
```

---

## 数据集示例

### Prompt 数据集 (PPO)

```bash
--prompt_data OpenRLHF/prompt-collection-v0.1 \
--input_key context_messages \
--apply_chat_template
```

### 偏好数据集 (RM/DPO)

```bash
--dataset OpenRLHF/preference_dataset_mixture2 \
--chosen_key chosen \
--rejected_key rejected \
--apply_chat_template
```

### SFT 数据集

```bash
--dataset Open-Orca/OpenOrca \
--input_key question \
--output_key response \
--input_template $'User: {}\nAssistant: '
```
