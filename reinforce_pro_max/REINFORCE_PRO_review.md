# REINFORCE_PRO 文档/实现一致性核对

## 1. 核对范围

- 文档：`reinforce_pro_max/REINFORCE_PRO.md`
- 实现：
  - `openrlhf/models/loss.py`
  - `openrlhf/trainer/ppo_utils/experience_maker.py`
  - `openrlhf/models/actor.py`
  - `openrlhf/trainer/ray/ppo_actor.py`

## 2. 一致项（算法层）

以下描述与代码实现一致：

- `reinforce_pro` 归属 `--vllm_is_correction_type`，不是 `policy_loss_type`
- IS correction 仅在 `enable_vllm_is_correction && policy_loss_type == "ppo"` 下生效
- `log_ratio_is = old_log_probs - rollout_log_probs`
- 前缀累积几何均值过滤：
  - `prefix_is = exp(cumsum(log_ratio_is * action_mask) / positions).detach()`
  - `token_mask = (prefix_is >= low) & (prefix_is <= high)`
- token 级校正系数：
  - `vllm_is = exp(log_ratio_is).detach()`
  - `loss = token_mask * vllm_is * loss`
- PPO ratio 梯度路径与文档一致（`log_probs - old_log_probs`）

## 3. 当前不一致（未修复）

无。
