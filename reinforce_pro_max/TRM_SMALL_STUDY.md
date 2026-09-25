# 固定四组实验

唯一维护的方案、数据、奖励函数和执行入口见 [experiments/dapo_small](experiments/dapo_small/README.md)。

Baseline：advantage × current/rollout ratio。Pro：causal prefix mask × advantage × 同一个 ratio。无 PPO clip，无额外 IS 乘法。Max 仅替换优势缩放。四组各 100 步，seed 42，使用相同 DAPO 训练子集和本地 AIME25 评测。
