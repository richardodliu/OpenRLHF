# REINFORCE Pro Max 文档/实现一致性核对

本文对照 `reinforce_pro_max/REINFORCE_MAX.md` 与当前代码实现，只保留仍未修复的不一致项；已修复项不再保留。

## 1. 核心算法一致性

`REINFORCE_MAX.md` 描述的算法主干与当前实现一致：

- Step1：RLOO baseline（`b_i=(sum(r)-r_i)/(n-1)`）
- `--uniform_scale`：全同 reward 组使用 `r_i/n`，并在 Step2 跳过归一化
- Step2：按 prompt 分组做自适应 token-level 归一化（正负分离缩放）
- Fallback：全同号、`sum_pos/sum_neg` 过小、`alpha/beta` 非有限时跳过归一化
- 数值保护：`eps=1e-8`、`max_scale=10.0`、`ratio^2*Q-` 上限裁剪

对应实现位置：

- `openrlhf/trainer/ppo_utils/experience_maker.py:24`
- `openrlhf/trainer/ppo_utils/experience_maker.py:827`
- `openrlhf/trainer/ppo_utils/experience_maker.py:952`

## 2. 当前不一致（未修复）

无。
