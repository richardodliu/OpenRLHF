# REINFORCE Max：实现与数学范围

REINFORCE Max 是 --advantage_estimator reinforce_max，其核心是对正、负 advantage
分别使用正缩放系数；当前实现以组内 RLOO 构造输入，并在每个 prompt 内计算系数。与 [Pro](REINFORCE_PRO.md) 的前缀 gate 组合才是 Pro Max。
实现见 [experience_maker.py](../openrlhf/trainer/ppo_utils/experience_maker.py)。

## 信号构造

同一 prompt 的 $n\ge2$ 个响应使用

$$A_i=r_i-\frac{\sum_{j\ne i}r_j}{n-1}.$$

gamma 强制为 1。只有 KL-free、稀疏终端奖励且 reward clipping 不激活时，
每个有效 token 的初始信号才等于上述 $A_i$；加入 KL 或 shaping 后须使用实际累计回报。
代码通过 experience.index 恢复 prompt-major 顺序，验证索引排列和同组 prompt，
按 action mask 排除 padding，归一化后再恢复原顺序和长度。

## 理想非对称归一化

在同一 prompt 的有效 token 中，令 $P,N$ 为正负 advantage 集合：

$$S_+=\sum_P A>0,\quad S_-=-\sum_N A>0,\quad
Q_+=\sum_P A^2,\quad Q_-=\sum_N A^2,\quad m=|P|+|N|.$$

两类均非空时，理想因子为

$$\kappa=S_+/S_-,\qquad
\alpha=\sqrt{\frac{m}{Q_++\kappa^2Q_-}},\qquad\beta=\kappa\alpha.$$

正负 token 分别乘以 $\alpha,\beta$，原始零值保持为零。理想形式在非零有效 token 上
满足均值为零、总体二阶矩为一，并保持符号及同号组内顺序。
这不保证完整参数梯度方向不变，也不证明训练更稳定。

实际二元 RLOO 组的正负有效 token 总数为 $P_{\rm tok},N_{\rm tok}$ 时：

$$\widehat A_+=\sqrt{N_{\rm tok}/P_{\rm tok}},\qquad
\widehat A_-=-\sqrt{P_{\rm tok}/N_{\rm tok}},\qquad
\beta/\alpha=\bar T_+/\bar T_-.$$

幅度比与因子比不同。固定相同 token 权重时，mean baseline 与 RLOO 的公共正比例
被理想归一化消去；二元奖励下，同组 token-weighted 标准化也给出相同理想系数。
不能把这些等价操作当作不同方法的独立优势。

## 实际 safeguards 与 fallback

_adaptive_token_normalization_single_group 当前使用：

- eps=1e-8；无有效 token、缺少任一符号、正负和过小或因子非有限时返回原信号。
- 中间项 ratio**2 * sum_sq_neg 上限为 1e8。
- 平方根分母加入 eps，最终因子裁剪到 [eps, 10.0]。

实际输出不严格满足理想零均值/单位方差，偏差由输入及保护分支决定；
没有通用的“通常误差小于某数”保证。精确实数公式和残差范围见
[数学审计](../formal/PROOF_AUDIT.md) 的 NormalizationSafeguards 映射。
标量系数分析不推出浮点 kernel、优化器或完整训练保证。

诊断口径：当前 `normed_var` 日志调用 `normed_nonzero.var()`，其默认
`correction=1` 使用样本方差分母（非零 token 数减一）。论文的经验方差和
二阶矩采用非零 token 数作分母。不能把该日志值直接与论文的理想单位方差
或 safeguards 公式比较；本说明没有改变日志或训练计算。

“未触发 clamp”也不足以推出零 token 均值：关闭 uniform scale，取两条响应
reward 为 `(d, 0)`、长度为 `(1, 2)`、`0 < d < eps`，RLOO 为 `(d, -d)`。
正质量 `d < eps` 触发直接返回分支，输出 token 均值为 `-d/3`，第二矩为 `d^2`。
该精确实数反例已由 `ImplementationBridge.small_signal_fallback_moments` 验证；
它不否定另外列明条件的二元 0/1 奖励结论。

## loss 分母与 batch 聚合

归一化在 prompt 组内进行，loss 的 token mean 则针对每次调用传入的有效 token。
先将每组除以自身 token 数再平均，与所有组共用一个 token 分母不同。
对 K 个独立完整组，后者的条件系数是 psi_K(L)=K E[1/(L+S)]，
应乘在尚未除以本组长度的响应系数上；S 是其余组的总长度。
有限模型证明及组内协方差连接见 `formal/BatchDenominator.lean`。
对二元 0/1 奖励、两类概率均为正、保护因子下限满足
`0 < eta <= min(1, max_scale)` 的情形，若完整组 token 数至多为 `l_max`，
共享分母诱导的类间乘子满足 `Lambda_x^(K) >= n * eta / l_max > 0`，
对任意 `K >= 1` 成立；二元 homogeneous uniform-scale 分支保持该下界。
若每条响应长度至多为 `L_max`，可取 `l_max = n * L_max`。
长度依赖组内内容时仍保留系数与 score 的协方差残差，正乘子本身不保证奖励提升。

动态 microbatch 路径显式按响应数比例缩放 loss；它不自动实现按 token 数加权的
全局 token mean。论文的总体奖励条件将同组 token 分母乘回后再取期望，
不将该动态路径的累计 loss 直接当作原始 reward surrogate。跨 prompt 的正方向还需相应的协方差 margin。

## 可选 uniform scale

默认关闭。开启 --uniform_scale 后，以 group_std < 1e-8 判定近似同奖励组，
使用 $r_i/n$ 并跳过自适应归一化；其余组仍执行 RLOO。

在 0/1 奖励且 KL-free 时，全零组仍为零，全一组使用 $1/n$。
不能声称全错组必然受惩罚，或所有同奖励组都有非零期望梯度。
该分支依赖奖励零点，奖励平移可能改变甚至反转更新。
有限模型方向结论还要求固定 prompt、on-policy、gate 接受、clipping 不激活及相应长度条件；
理论中的严格同奖励事件与实现的容差判定应区分。

## 使用与验证

~~~bash
--advantage_estimator reinforce_max \
--n_samples_per_prompt 8
~~~

响应数必须大于 1，8 只是配置示例。与 Pro 组合另加其 IS 参数。
理论 core mode 需显式 --init_kl_coef 0 并关闭 KL loss 等额外目标。
当前 GPU 实验按 [固定四组方案](experiments/dapo_small/README.md) 执行，旧实验记录已清理。

从仓库根目录运行：

~~~bash
python -m pytest -q reinforce_pro_max/test_reinforce_max.py \
  reinforce_pro_max/test_normalization_safeguards.py
~~~

这些测试检查分组、边界分支和数值公式，不是训练稳定性或 benchmark 结果。
