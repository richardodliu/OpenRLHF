# 固定四组实验

唯一维护的方案、数据、奖励函数和执行入口见 [experiments/dapo_small](experiments/dapo_small/README.md)。

Baseline：advantage × current/rollout ratio。Pro：causal prefix mask × advantage × 同一个 ratio。无 PPO clip，无额外 IS 乘法。Max 仅替换优势缩放。当前使用 Qwen2.5-Math-1.5B，8 卡串行运行四组，开启 CUDA Graph。四组各 544 步，seed 42，使用全部 17,381 道可用 DAPO 题目，固定重复 27 道题以补齐 batch，共 17,408 条；本地 AIME25 评测和奖励函数不变。配置见 `experiments/dapo_small/study-full-plan.json`。

## 2026-09-25 后续评测取消

用户指定此后只记录训练集上的 avg@8。运行中 Max-only 不重启；其后 Pro-only、Pro Max 正常串行训练，各544步。独立评测和 initial 评测跳过，已完成 Baseline AIME25 结果保留。主结果为训练过程中当批 prompt 的 online avg@8（每题8次），不声称独立泛化性能。实现、兼容旧调度器的跳过标记含义及汇总命令见 `experiments/dapo_small/README.md` 最新记录。
