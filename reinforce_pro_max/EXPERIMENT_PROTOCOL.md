# REINFORCE Pro Max：真实模型实验协议（待运行）

本文件定义拟执行的实验，不包含真实模型结果。基础模型 revision、数据 revision、GPU 拓扑和可用预算尚未确定；不得将合成实验、CLI 导入或 Lean 构建填入结果表。

## 已检查的入口与配置差异

训练入口为 `openrlhf.cli.train_ppo_ray`，数学奖励入口可复用 `examples/python/math_reward_func.py`，但运行前必须验证数据的 prompt/label 字段和 boxed-answer 提取。内置评测 `--eval_dataset` 要求设置 `--remote_rm_url`；默认评测 split 是 `train`，必须显式选择实际保留集，不能依赖默认值。

`examples/scripts/train_reinforce_baseline_hybrid_engine.sh` 是演示配置：启用 `dynamic_filtering`、`use_dynamic_batch`、`use_kl_loss`，设置 `init_kl_coef=1e-5`、`eval_steps=-1`，并默认加载 checkpoint。这些配置不能自动被标注为论文 KL-free、未筛选 iid 评估或全局 token-mean 模型。

## 固定条件

- 固定模型/tokenizer revision、chat template、数据 revision、训练/验证/测试样本 ID 和去重规则；先检查训练集与评测题目的重叠。
- 主实验采用 0/1 数学奖励，固定答案提取与评分代码版本；记录无法解析答案、截断和空输出比例。长度惩罚单列消融。
- KL-free 主配置设 `--init_kl_coef 0`，不启用 `--use_kl_loss`；不启用 `--uniform_scale` 和整组动态筛选。0/1 奖励的 raw RLOO 在 [-1,1]，默认 [-10,10] clipping 对该信号不激活；另加 shaping 后必须重新核对。
- 固定 prompt/生成长度上限、每 prompt 响应数、优化器、学习率、PPO clipping 和训练预算。以总生成 token 为主要预算，并同时报告更新次数、prompt 数、墙钟时间和 GPU 小时。
- 主配置使用 `temperature=1`、`top_p=1`，仍需检查引擎内的其他 logits processor、采样限制和 logprob 定义；这些参数本身不证明 mutual support 或全历史 ratio 界。
- 固定实际 loss reduction。关闭动态 batch 不足以证明全局 token mean；需记录 microbatch/跨卡 reduction，先验证实际加权方式，再解释与定理的关系。
- 标准 PPO 主配置不传 `--dual_clip`；`--aux_loss_coef 0`，`--entropy_loss_coef` 保持未设置或设为 0。dual-clip 会改变负 advantage 分支，使论文中的 clipping deduction 非负性质不再普遍成立；熵或 MoE 辅助损失也会改变完整参数更新，若另作消融必须独立标注，不能直接套用主配置的方向结论。
- 用预先选定的至少三个训练 seed 报告均值和跨 seed 离散程度。验证集用于选超参数；独立测试集只用于最终评估。

## 方法与消融矩阵

以下各行共用相同 backbone、数据和预算。所有 PPO 行保持 `policy_loss_type=ppo`；IS 行显式启用 `enable_vllm_is_correction`。

| 目的 | advantage_estimator | IS correction | 说明 |
|---|---|---|---|
| raw RLOO 对照 | rloo | icepop | 保留原始 RLOO 与实现 reduction |
| REINFORCE++ Baseline 对照 | reinforce_baseline | icepop | 包含其实际跨 batch 归一化 |
| GRPO-style 对照 | group_norm | icepop | 用实现名称描述，不能无条件等同所有 GRPO 配方 |
| 仅 Max | reinforce_max | icepop | 相对 raw RLOO 检查 normalization 效应 |
| 仅 Pro | rloo | reinforce_pro | 相对 raw RLOO 检查 prefix gate 效应 |
| Pro Max | reinforce_max | reinforce_pro | 主方法 |
| gate 对照 | reinforce_max | tis / seq-mask-tis | 保持 advantage 和阈值一致 |
| uniform-scale 消融 | reinforce_max | reinforce_pro | 单独开启 uniform_scale，保留 0/1 零点 |

若加入 critic PPO，必须另列 critic 模型、学习率、value loss 及额外算力；不能仅把 advantage_estimator 切为 gae 就称为公平对照。二元奖励下 mean/RLOO 经相同理想归一化可等价，不把等价系数重复当作独立创新证据。

## 评测与证据

对数学任务明确报告题目数、每题采样数、温度、长度上限和评分器。平均单次正确率的分母为全部评测响应；pass@k 是每题至少一次成功的题目比例，必须与前者分开。按题目聚类估计评测不确定性，另行报告训练 seed 差异，避免把同题多个响应当作独立题目。

保存每次运行的源码 revision 与 dirty patch、完整解析参数、数据 ID 清单、环境包版本、checkpoint、逐题预测和原始奖励、训练曲线、失败原因及资源计量。同步记录各 gate 接受率、有效 token 数、截断率、advantage 统计量、IS ratio 分布、归一化 fallback/cap/clamp 激活率。

若报告有限样本性能证书，必须独立记录未筛选 evaluation groups、候选族或 cover 的固定方式及合法的总体 cost/energy 上界。样本最大 ratio、样本 KL 或观测接受率不能替代全历史假设。标准 benchmark 正确率提升也不等于已经验证理论证书。

## 开跑前仍需提供

可用 GPU 环境及访问方式；基础 checkpoint 和 tokenizer 的固定版本；训练/验证/测试数据位置与字段；允许的运行预算。具备这些信息后先完成小规模端到端训练与独立评测，再扩展上述矩阵。当前不预填任何准确率、加速比或论文接收结论。

## 当前工作区检查（2026-09-18）

本工作区的可核验检查结果如下：

- `nvidia-smi` 不存在；`torch.cuda.is_available()` 为 `False`，`torch.cuda.device_count()` 为 `0`。
- 未发现可用的 Slurm `srun`/`sinfo` 命令，也未发现正在运行的 OpenRLHF 训练进程或本项目的训练输出目录。
- 本地存在若干 Hugging Face checkpoint（包括 `Qwen2.5-0.5B`、`Qwen3-0.6B-Base`、`Qwen2.5-Math-1.5B` 等），但本协议所需的训练/验证/独立测试数据版本、字段和预算尚未固定。
- 因此当前目录不能声称已经完成真实 LLM 训练或 benchmark 评测；合成结构实验和 Lean 构建不能替代这些结果。
