# REINFORCE Pro Max Lean 数学审计

## 范围与判定规则

本审计检查论文中的形式化数学命题、假设和推导，以及数学目标与实现公式的对应关系；不把公式核对扩大为训练运行、浮点语义或实验性能验证。Lean 文件位于 `formal/`（包括 [`formal/ProMax.lean`](/volume/pt-train/users/rbliu/github/OpenRLHF/formal/ProMax.lean)、`ErrorDecomp.lean`、`CoarseGrain.lean`、`KLChain.lean`、`SequenceTV.lean`、`KLEquality.lean`、`RLOOProbability.lean`、`RLOOGeneral.lean`、`SurrogateBridge.lean`、`ProMaxObjective.lean`、`AutoregressiveTree.lean`、`TreeSurrogate.lean`、`PromptMixture.lean`、`PrefixConcentration.lean`、`PrefixMasking.lean`、`PaddedPrefix.lean`、`NormalizationSafeguards.lean`、`TreeTV.lean`、`TreeKL.lean`、`CoarseKL.lean`、`K3Inlier.lean`、`SparseReturn.lean`、`Pinsker.lean`、`TreePinsker.lean`、`RLOOMoments.lean`、`PolicyGradient.lean`、`PromptGradient.lean`、`PromptGradientMeasure.lean`、`TreePolicyGradient.lean`、`BatchReduction.lean`、`BatchPerformance.lean`、`RLOOSurrogate.lean`、`PPODerivative.lean`、`UniformScale.lean`、`BinaryUpdate.lean`、`BinaryLength.lean` 和 `BinaryAscent.lean`），使用 Lean 4.19.0 和 Mathlib
`v4.19.0`。

当前版本保留实际 ProMax 目标到有限样本奖励保证的完整证明。
`RLOOCovariance.lean` 和 `GroupCertificate.lean` 已从实际 iid response law 推出
共享 baseline 的 RLOO group-average 精确方差，区分同一 prompt 内的响应采样误差
与 prompt 间的 surrogate 差异，并将二者接入原有观测 token 重缩放、统一置信界和
真实奖励证书。新增 `PolicyCover.lean` 进一步用预先固定的有限 cover，
把保证扩展到在评估数据上拟合的策略；策略类可以有连续参数，拟合结果无需等于
某个 cover center，但必须满足全类覆盖条件并支付显式逼近/复杂度代价。
方法中的 gate、归一化、clipping 与随机长度全部保留；
当前 `formal/` 中的 Lean 模块（包括后续新增的结构例子和筛选反例）均纳入默认公理审计。`AdaptiveMechanism.lean` 已
从实际有限 RLOO 奖励组及长度权重推导二元奖励的精确归一化机制，证明
mean/RLOO baseline 经同种理想归一化后的等价性，以及二元奖励下与同组
token 标准化的等价性；同时给出多值 RLOO 奖励下中心化翻转符号的严格反例。
`LogitBounds.lean` 已从保留 normalizer 的正权重倾斜及实际 softmax
logit 变化推导 ratio energy、反向 KL、TV、Adaptive cost 和固定 cover 半径的显式界，
并接入实际 RLOO 组的统一置信界及观测目标奖励证书。此校准不要求最小正 token
概率；它是原一般定理的可选实例化条件，不改 ProMax 的 gate、归一化或 clipping。
本轮新增 `UniformScale.lean`，从真实有限 iid 分组事件证明可选 uniform-scale
分支的二元奖励期望增量，导出 0/1 奖励下与单 prompt 奖励梯度同向的新增项；
同时严格验证奖励平移导致更新反向的反例，以及全支持恒定奖励下的零期望与
方向方差。保留可选方法本身，修正了“全同奖励就恢复信号”的过宽解释。

`BinaryAscent.lean` 将完整二元 population update 接到真实成功概率的梯度与局部改进。
对于任意内容相关长度，使用实际 class multiplier `Λ_C` 给出
`⟨∇s,U⟩ ≥ Λ_C‖∇s‖² − sqrt(V_C*n*V_f)`，其中 `Λ_C ≥ η/L_max > 0`。
当实际正向项超过协方差界时，所有足够小的正步长沿期望更新 `U` 都严格提高成功率；
`η/L_max` 仅作为可选的保守充分下界，不代替实际 multiplier 而收紧主要改进条件。
类别内长度固定时只需 `∇s ≠ 0`。这些结果保留实数 safeguards、随机 total-token 分母和
optional uniform-scale，范围明确为固定 prompt、on-policy、gate 全接受、clipping 不激活。

- `PROVEN`：对应的数学命题已经在 Lean 中以明确假设编译通过。
- `FAILED`：Lean 中有可编译的反例，或论文推导使用了可编译反例所否定的等式/不等式。
- `PENDING`：尚未完成足够的概率空间、过滤、策略可微性或有限状态模型形式化，因此不能据此断言论文正确或错误。

`PENDING` 不表示命题错误；它表示当前审计尚未完成相应证明或模型接口。假设作为定理前提被明确编码；Lean 证明不等于验证实际训练分布满足该假设。下列历史错误与修订后结论分开记录。

实验章节中的 baseline 数字和大规模 benchmark 结果不作为 Lean 数学证明的一部分。当前版本已提供
`reinforce_pro_max/structural_experiment.py` 及其固定 seed 的 JSON 输出，覆盖有限 ratio 模式的
结构性 synthetic Monte Carlo 检查；这些结果不是 Lean 定理，也不是 AIME/MATH 或真实 RLVR
benchmark 的训练证据。有限 ratio 模式的确定性 masking 表仍由 Lean 直接验证。大规模 LLM
benchmark 和真实分布式训练结果仍待完成。

本报告验证数学命题的形式化条件，不构成 ICLR 接收保证；接收还取决于实验、清晰度、相关工作和审稿判断。论文当前实验章节仍包含受控模拟和待完成的大规模 LLM 基准实验。

## 理论 core mode 与 CLI 默认值

稀疏回报、`gamma=1` 以及 `A_{i,t}=\tilde r_i` 的定理只覆盖 KL-free 且 RLOO
之后不发生 reward clipping 的 core mode。当前 `train_ppo_ray.py` 的 CLI 默认值是
`--init_kl_coef 0.01` 和 `--reward_clip_range (-10, 10)`；前者会加入 reference-model
KL reward，后者可能改变 RLOO shaped scalar reward。因此，使用默认值不能自动声称满足
这些定理。论文已补充说明：要实例化 core mode，必须显式设 `init_kl_coef=0`、关闭
KL loss，并证明 reward clip 区间在给定 reward 支持上不激活；否则应把 KL/clipping
纳入新的 return 定义和证明。这个范围修订不削弱 REINFORCE Pro Max 的方法，只避免把
默认工程配置误标为已验证的数学模型。

## 实现与理论的 ratio 边界

论文和代码现在明确区分三种策略对象：`pi_roll` 生成数据，`pi_old` 是 PPO
快照，`pi_theta` 是当前训练策略。代码的 prefix gate 使用
`w = pi_old/pi_roll`，PPO 的可微 ratio 使用 `r = pi_theta/pi_old`，而理论的
behavior-to-current ratio 满足 `rho = r*w`。因此，Lean 中关于 prefix 的有限序列
定理验证的是任意选定 policy pair 的 ratio 代数；把缓存的 `(pi_old, pi_roll)`
gate 解释为当前 `(pi_theta, pi_roll)` 的 proxy，需要额外的 PPO 更新 coupling
条件。`prefix_average_additive_perturbation` 严格证明了一个充分条件：若每个
`log r` 的绝对值不超过 `eta`，则任意 prefix 平均的偏移不超过 `eta`。未满足该
条件时，审计不把 implementation gate 宣称为当前策略的无偏 trust-region 保证。

实现检查还确认：`seq-mask-tis` 和 `reinforce_pro` 的 gate 之外，代码使用的是
detached 的未裁剪 token 权重 `exp(old_log_probs - rollout_log_probs)`；只有
`tis` 分支执行 token-level clamp。论文现已按这一实现语义表述，不能把
`reinforce_pro` 的 accepted coefficient 误称为 TIS-clipped coefficient。

uniform-scale 分支也已按实现的触发条件校正措辞：代码使用
`group_std < 1e-8` 的容差测试，因此可能包含近似均匀而非严格相等的 reward
组。论文现在把 `r_i/n` 视为该分支的直接定义；Lean 的
`uniform_scale_token_coefficient_sign` 对分支中的任意有限 `r_i` 给出标量系数
符号结论，不再把近似均匀判定误写成“所有 reward 必须严格相等”。

本轮实现一致性修订还固定了 `openrlhf/models/loss.py` 中的诊断变量复用：vLLM 校正分支会临时使用 `old_log_probs-rollout_log_probs`，但返回的 `ppo_kl` 仍必须计算 `-(log_probs-old_log_probs)`。现已保存 PPO 的 current/old log-ratio 后再计算该指标，因此监控量与 PPO ratio 的数学定义一致；这不改变校正 loss 或论文定理。

## 已通过的数学命题（注明模型范围）

| 论文位置 | Lean 名称 | 状态 | 说明 |
|---|---|---|---|
| `tex/main/4-method.tex`, `eq:rloo-baseline` | `rloo_scaling` | `PROVEN` | 对 `s = Σ_{j ≠ i} r_j`，`r-(s+r)/n = ((n-1)/n)(r-s/(n-1))`；假设分母非零。这里只证明确定性缩放，不包含无偏性。 |
| `tex/main/4-method.tex`, RLOO fallback | `rloo_shaped_sum_zero`, `rloo_shaped_sum_zero_erase` | `PROVEN` | 对 `n≥2`，精确 RLOO shaped rewards 在 prompt group 内求和为零；两种定式（`Σr-r_i` 与 `erase i`）均已验证。 |
| `tex/main/6-experiment.tex`, baseline 对照表 | `mean_baseline_shaped_sum_zero` (`RLOOMoments.lean`) | `PROVEN` | 对非空组，普通 mean baseline 的 response-level shaped rewards 同样精确零和。该性质不是 RLOO 独有优势；不等 response 长度下，两者的 token-weighted mean 都可能非零。 |
| `tex/main/4-method.tex`, RLOO fallback | `rloo_zero_if_nonpositive`, `rloo_zero_if_nonnegative` | `PROVEN` | 由零和性质可知，若所有 shaped rewards 同号（全非正或全非负），则它们逐项为零；这刻画 uniform-reward 分支的退化。 |
| `tex/main/4-method.tex`, `eq:alpha-beta` | `adaptive_closed_form` | `PROVEN` | 在 `P,N,Q_+,Q_-,m > 0` 下，闭式 `α,β` 为正，满足零均值方程和单位二阶矩方程。 |
| `tex/main/appendix.tex`, `prop:uniform-coefficient-sign` | `uniform_scale_coefficient_sign`, `ppoTerm_hasDerivAt`, `gated_ppo_hasFDerivAt`, `finite_gated_ppo_hasFDerivAt` (`PPODerivative.lean`) | `PROVEN` | 对正 group size `n`，系数 `r/n` 与标量 reward `r` 同号，且 `r=0` 当且仅当系数为零；保留 action/reduction 权重、prefix gate、detached rollout-to-snapshot 权重和 PPO clipping，在 clipping kink 之外验证精确导数。这只保证 scalar coefficient 的符号，不推出完整梯度向量方向保持。 |
| `tex/main/appendix.tex`, uniform-scale branch and implemented PPO loss | `ppoTerm_hasDerivAt`, `gated_ppo_hasFDerivAt`, `finite_gated_ppo_hasFDerivAt`, `uniform_scale_token_coefficient_sign` (`PPODerivative.lean`) | `PROVEN` | 保留 action/reduction 权重、prefix gate、detached rollout-to-snapshot 权重和 PPO clipping；在 clipping kink 之外给出当前 log-probability 的精确导数。uniform-scale 的 `r/n` 符号结论只适用于保留下来的 differentiating branch，不推出整批梯度方向。 |
| `tex/main/appendix.tex`, `prop:normalization-safeguards` | `NormalizationSafeguards.stabilized_closed_form`, `stabilizer_second_moment_lt_one`, `stabilizer_deficit_bound`, `capped_stabilized_moment`, `finite_scaled_moments`, `factor_mean_residual_bound`, `factor_second_moment_residual_bound` | `PROVEN` | 严格验证实现中的 additive `eps`、中间 product cap 和最终 factor clamp 对均值与二阶矩的精确公式及残差界；因此实际分支不再被误称为严格单位方差。 |
| `tex/main/4-method.tex`, prefix implementation bridge | `PaddedPrefix.implementationAverage_reindex`, `implementationMask_reindex`, `implementation_rejection_count` | `PROVEN` | 对任意有限二值 action-mask 的 active positions，递增重编号后累计和、clamp 分母、prefix mask 和拒绝计数与无 padding 的有限 prefix 定义完全一致。 |
| `formal/ProMax.lean`, 保留的 unit-amplitude 特例 | `binary_reward_scaling_sq` | `PROVEN` | 在正负原始优势为 `±1` 的特殊条件下，得到 `α²=N/P`、`β²=P/N`。这一代数本身正确，但真实 binary RLOO 通常不满足等幅条件；正文已改用下列一般结果，不能据此推出失败多时 `β<1`。 |
| `tex/main/4-method.tex`, `prop:binary-rloo-normalization` | `AdaptiveMechanism.finite_binary_rloo_normalization`, `rloo_success_amplitude`, `rloo_failure_amplitude`, `two_level_weighted_statistics` | `PROVEN` | 从 `k` 个高奖励、`m` 个低奖励的真实 finite reward sum 和 RLOO 公式开始，按实际长度形成五种统计量，再组合得到理想 token 系数 `+√(N/P)` 与 `−√(P/N)`。两类均非空，奖励差和总 token 权重为正。 |
| `tex/main/4-method.tex`, `eq:binary-length-balance` | `AdaptiveMechanism.rloo_scale_ratio_mean_lengths`, `raw_mass_ratio_mean_lengths`, `normalized_mass_balance`, `paired_factor_ratio` | `PROVEN` | `β/α=κ=平均高奖励长度/平均低奖励长度`；归一化后正负 aggregate scalar coefficient mass 相等。频率不平衡本身不推出 `β<1`，也不推出完整梯度向量平衡。 |
| `tex/main/4-method.tex`, `prop:normalization-scale-invariance` | `AdaptiveMechanism.weighted_statistics_rescale`, `weighted_normalization_scale_invariant`, `mean_baseline_is_scaled_rloo`, `mean_and_rloo_normalization_equal` | `PROVEN` | 固定相同 token/长度权重，理想 asymmetric normalization 对公共正比例缩放不变；mean baseline 与 RLOO 的系数因而完全相同。这不否定 raw RLOO 的独立 baseline 分析，且不自动扩展到固定 epsilon、clamps 或 fallback。 |
| `tex/main/4-method.tex`, `cor:binary-token-standardization` | `AdaptiveMechanism.two_level_token_zscore` | `PROVEN` | 两级正负幅度下，按同一 prompt 的同一 token 权重计算 population mean/std，中心化标准化得到相同的两类系数。因此精确二元奖励下，不能将这两个理想操作当作不同 normalization baselines。 |
| `tex/main/appendix.tex`, `par:multivalued-sign-example` | `AdaptiveMechanism.multivalued_rloo_centering_counterexample`, `multivalued_rloo_zscore_negative`, `multivalued_rloo_adaptive_positive` | `PROVEN` | 奖励 `[0,2/3,1]`、长度 `[1,1,4]` 的实际 RLOO 为 `[−5/6,1/6,2/3]`；token mean 为 `1/3`、variance 为 `11/36`。中间响应中心化/标准化后为负，而理想 asymmetric coefficient 严格为正。 |
| `tex/main/appendix.tex`, `app:adaptive-mechanism` 的 stabilizer 连接 | `AdaptiveMechanism.stabilizer_common_shrinkage`, `paired_factor_ratio` | `PROVEN` | product/factor clamps 均不激活且 normalization 分支被采用时，两类理想输出共同乘 `√(D/(D+ε))`。比值及零均值保留，但固定 `ε` 不随奖励缩放，不能据此声称实际实现完全 scale invariant。 |
| `tex/main/4-method.tex`, `par:adv-sign-preservation` | `positive_scaling_preserves_sign`, `positive_scaling_preserves_order` | `PROVEN` | 正数缩放保持正负号和组内大小顺序。 |
| `tex/main/4-method.tex`, `lem:prefix-stability` | `prefix_log_average_stability`、`prefix_geomean_stability` | `PROVEN` | 对正区间，prefix log-average 是两个区间内 log 值的凸组合；指数映射后仍在原接受区间。 |
| `tex/main/4-method.tex`, `thm:pre-causal-mechanism` | `accepted_prefix_density_envelope`, `prefix_gate_residual_bound`, `prefix_gate_residual_le_envelope`, `prefix_gate_tv_bound` (`TreePrefixGate.lean`) | `PROVEN` | 在完整有限历史树上定义实际 log-average 二值 gate，由正 rollout 质量前缀的接受条件证明密度比区间、残差界、乘以接受概率的 rollout 期望界以及过滤测度的 half-L1 界。允许共同零概率 token 和 action-dependent gate，不要求 token 或 gate 独立。 |
| `thm:pre-causal-mechanism` 的辅助代数 | `accepted_prefix_product_deviation_bound`, `finite_masked_expectation_bound` | `PROVEN` | 分别保留指数区间的标量残差界和任意有限非负权重下的 masked expectation 界；完整 gate 到历史树概率的组合已由上一行完成。 |
| `tex/main/4-method.tex`, implementation bridge | `prefix_average_additive_perturbation`, `finset_average_additive_perturbation` | `PROVEN` | 对有限 prefix 或任意非空有限 active set，若当前/old 的每个 active log-ratio 扰动满足 `|u_i|≤η`，则加入该扰动后 arithmetic mean 的变化绝对值不超过 `η`；后者明确排除 EOS padding，用于把缓存的 `(pi_old, pi_roll)` gate 转移到当前策略时的显式 coupling 条件。 |
| `tex/main/3-preliminaries.tex`, three-policy ratio decomposition and `tex/main/4-method.tex`, log/prefix threshold convention | `log_ratio_composition`; `ImplementationBridge.current_rollout_ratio_factorization`; `ImplementationBridge.prefix_exp_gate_iff_log_gate`; `ImplementationBridge.prefix_gate_thresholds_match` | `PROVEN` | 对正 ratio，严格证明 `log rho = log r + log w` 以及实现中的 `exp(logCurrent-logRollout)=exp(logCurrent-logOld)exp(logOld-logRollout)`；在显式正阈值条件下，代码先 exponentiate prefix average 再比较区间与论文比较 log-threshold 完全等价。 |
| `tex/main/4-method.tex`, `thm:prefix-tighter`(b) 的 pure single-spike 不等式 | `single_spike_dilution`, `single_spike_prefix_formulas`, `single_spike_prefix_rejects_sequence_accepts`, `single_spike_prefix_geomean_separation`, `single_spike_token_interval_pattern`, `finite_single_spike_mask_separation`, `prefix_tighter_single_spike_closed_loop`, `exists_single_spike_prefix_sequence_separation`, `single_spike_late_prefix_zero/accepted`, `early_deviation_upper/lower_window`, `list_average_interval`, `monotone_violation_average_gt/lt` | `PROVEN` | 验证正向单尖峰的稀释不等式、prefix/sequence 精确公式、非空有限 prefix 的区间平均稳定性，以及在 `k+1<n` 时显式构造 prefix 拒绝而 sequence 接受的有限见证；`prefix_tighter_single_spike_closed_loop` 将 late single-spike 的 token/prefix/sequence 三个 mask 结论合并为一个定理。完整的三分情形轨迹级 theorem 已由 `PrefixMasking.lean` 的 `prefix_tighter_complete`、`early_deviation_complete`、`late_deviation_complete` 和 `sustained_violation_complete` 编码。 |
| 修正后的 TV 恒等式 | `tv_ratio_identity_of_reverse_support` | `PROVEN` | 在有限支持上补充 `q(y)>0 → p(y)>0` 后，`TV(p,q)=1/2 E_p[|q/p-1|]` 成立。 |
| 修正后的密度比归一化 | `ratio_mass_eq_one_of_reverse_support` | `PROVEN` | 在 `q<<p` 和 `Σq=1` 下，`Σ p(q/p)=1` 成立。 |
| 修正后的有限支持换测度 | `finite_change_of_measure` | `PROVEN` | 在有限支持、非负 `p,q` 和反向支持 `q>0→p>0` 下，`Σ pR(q/p)=Σ qR`；`p=0` 处比值按 0 定义。 |
| `tex/main/appendix.tex`, `rem:coarse-grain-lower`/`rem:binary-tv-lower` | `REINFORCEProMax.finite_coarse_tv_lower_bound`, `REINFORCEProMax.binary_tv_lower_bound` (`formal/CoarseGrain.lean`) | `PROVEN` | 对有限 token 支持和任意有限 cell map，严格证明 coarse-grained TV 不超过 full TV；二元事件特例另外显式验证。 |
| `rem:coarse-grain-lower` / `rem:kl-equality` | `finite_cell_kl_contraction`, `finite_cell_kl_equality_iff`, `finite_coarse_kl_contraction`, `finite_coarse_kl_equality_iff_proportional` (`CoarseKL.lean`) | `PROVEN` | 任意有限非负 cell 与完整 partition，仅要求 `p<<q`，不要求严格正概率。KL 聚合取等号 iff 每个正 p-mass cell 上整个 cell 内 `p=(P/Q)q`；包括 p 为零而 q 为正的位置。零 p-mass cell 不施加额外限制。 |
| `rem:coarse-grain-lower` / `rem:kl-equality` 的 gap 分解 | `finite_cell_kl_gap_identity`, `finite_coarse_kl_gap_identity` (`CoarseKL.lean`) | `PROVEN` | full-minus-coarse KL 精确等于各 cell 的非负 Gibbs/log-sum gap 之和；因此 near-equality 应以总 gap 小来陈述，除非另行定义 ratio 误差度量。 |
| `tex/main/appendix.tex`, `rem:tv-gap` | `REINFORCEProMax.finite_tv_gap_slack_decomposition`, `finite_tv_gap_slack_nonneg`, `finite_tv_gap_slack_mass_bound` (`formal/CoarseGrain.lean`) | `PROVEN` | 对有限 cell map 严格证明 TV gap 的 cell-wise slack 分解、非负性及由 cell 总质量给出的上界（假设两分布逐点非负）。 |
| `tex/main/appendix.tex`, `rem:rloo-variance` | `finite_independent_baseline_score` (`formal/RLOOProbability.lean`) | `PROVEN` | 在有限乘积空间中，若 baseline 样本与 trajectory 独立且 score 在 trajectory 分布下均值为零，则 baseline control-variate 项严格消失；该独立 baseline 引理不负责可微性接口；有限可微策略与实际 RLOO 的连接现已由 PolicyGradient/TreePolicyGradient 单独证明，一般概率空间扩展仍需条件。 |
| `tex/main/appendix.tex`, `rem:rloo-variance` | `finite_iid_rloo_unbiased_two`, `finite_rloo_independent_baselines`, `finite_iid_rloo_unbiased`, `finite_iid_rloo_unbiased_vector` (`formal/RLOOProbability.lean`, `formal/RLOOGeneral.lean`) | `PROVEN` | 前者严格验证两样本 iid leave-one-out 估计量；`finite_rloo_independent_baselines` 验证任意有限 `n` 的独立 baseline law；`finite_iid_rloo_unbiased` 及其有限维向量版直接实例化其余 `n-1` 个 reward 的均值 baseline，并证明平均 RLOO 估计量的每个 score 坐标等于目标 score expectation。有限可微策略的梯度连接见 PolicyGradient/TreePolicyGradient；一般概率空间与积分/微分交换仍需额外 regularity 条件。 |
| `tex/main/appendix.tex`, `rem:rloo-variance` 的二阶矩分解 | `finite_independent_baseline_second_moment`, `finite_independent_baseline_cross_moment` (`formal/RLOOProbability.lean`) | `PROVEN` | 在有限独立乘积空间、baseline 均值为 `μ` 的条件下，严格证明标量平方和任意两 score 坐标的交叉二阶矩分解；逐坐标组合即得到有限维 outer-product 版本。 |
| `tex/main/appendix.tex`, `rem:rloo-variance` 的策略梯度连接 | `policy_mass_derivative_zero`, `policyScore_is_log_derivative`, `finite_policy_score_mean_zero`, `finite_policy_gradient`, `finite_iid_rloo_policy_gradient`, `tree_iid_rloo_policy_gradient` (`PolicyGradient.lean`, `TreePolicyGradient.lean`) | `PROVEN` | 在任意实赋范参数空间、局部非负且归一化的有限可微策略上，从微分和有限求和规则推出 score 零均值及实际 iid RLOO 期望等于回报导数。零概率响应的概率导数由局部最小值定理证为零。历史树版本从可微条件 token 概率构造路径质量，并接入与 performance-difference 相同的递归 terminal reward 目标；不要求 token 独立。此处不把带 gate/归一化/clipping 的完整 ProMax 宣称为无偏梯度。 |
| `rem:rloo-variance`：条件 RLOO 梯度到总体 prompt 目标 | `joint_prompt_response_mass_normalized`, `joint_prompt_response_mass_nonneg`, `finite_prompt_rloo_policy_gradient` (`PromptGradient.lean`) | `PROVEN` | 固定 prompt 下的期望对应 `∇J_x`；对与参数无关的有限 prompt law 求和后，联合 prompt/response 期望才对应总体 `∇J`。证明从实际 iid RLOO differential 及逐 prompt 的可微概率质量开始，不假设各 prompt 梯度相同。联合采样质量的非负性和归一化单列验证。 |
| `app:rloo-proof`：一般 prompt measure 的外层微分 | `ae_prompt_rloo_policy_gradient`, `measure_prompt_rloo_policy_gradient` (`PromptGradientMeasure.lean`) | `PROVEN` | 对任意 prompt 测度，以 Bochner 积分表示总体目标；在条件策略正则性、可测性、可积性及 integrable Lipschitz domination 下，几乎处处的有限-response RLOO 导数可交换到外层积分。条件独立性仍只是在给定 prompt 内成立；支配条件没有被隐藏。 |
| `tex/main/appendix.tex`, `rem:rloo-variance` 的实际 iid 构造 | `finite_iid_baseline_mean`, `finite_loo_baseline_marginal`, `finite_loo_baseline_variance`, `finite_loo_score_cross_moment` (`RLOOMoments.lean`) | `PROVEN` | 直接从任意有限 iid group 推出留一法 baseline 的边缘分布、均值和 `Var(b_i)=sigma²/(n-1)`，再推出任意两个 score 坐标的交叉二阶矩公式。没有把 baseline 方差作为前提。结论是单样本项的二阶矩，不是忽略共享 baseline 后的 group-average 方差。 |
| `tex/main/4-method.tex`, `rem:prefix-proxy` | `bernoulli_pinsker`, `finite_pinsker_squared`, `finite_pinsker`, `prefixTV_le_sqrt_jointPrefixKL`, `prefixTV_le_sqrt_chainKL` (`Pinsker.lean`, `TreePinsker.lean`) | `PROVEN` | 从二元 KL 的导数符号和有限 KL 聚合界推出精确常数 `2 TV² ≤ KL(p||q)` 与 `TV ≤ sqrt(KL/2)`，并接入实际有限历史树的联合前缀概率及 KL chain rule。允许共同零概率；当前泛型 theorem 采用与论文一致的 mutual support。 |
| 有限支持 surrogate 差分恒等式 | `finite_surrogate_difference_identity` | `PROVEN` | 在有限支持和反向支持下，`Σ pR(ρ−1)=Σ qR−Σ pR`；对应单步 rollout surrogate。 |
| 有限支持 performance-difference 恒等式 | `finite_performance_difference` | `PROVEN` | 将单步期望回报差直接写成 rollout 分布下的 ratio-residual surrogate；为有限 horizon 拼接提供方法相关接口。 |
| 有限条件 KL 的 log-ratio 恒等式 | `finite_kl_log_ratio` | `PROVEN` | 在有限支持 mutual support 下，`Σ p log(q/p) = -Σ p log(p/q)`；这是 autoregressive KL chain rule 的逐条件代数核心。 |
| 有限条件 KL 链式加权恒等式 | `finite_conditional_kl_chain` (`KLChain.lean`) | `PROVEN` | 对有限时域、context 权重和 token 条件分布，严格证明加权累计 `log(q/p)` 等于逐 context 条件反向 KL 的负和；context visitation 与自回归一致性作为显式建模前提保留。 |
| `rem:cumsum-kl`：联合前缀 KL 链式分解 | `trajectory_log_ratio_expectation`, `jointPrefixKL_eq_chainKL`, `expected_log_ratio_eq_neg_jointPrefixKL` (`TreeKL.lean`) | `PROVEN` | 从归一化有限历史树独立定义 joint-prefix KL，严格证明累计 log-ratio 的 rollout 期望等于联合前缀反向 KL 和累计条件反向 KL 的负值；实际路径质量、visitation 权重和共同零概率路径均已接入。 |
| 修正后的有限支持 `k_3` 代理恒等式 | `finite_k3_proxy_eq_reverse_kl` | `PROVEN` | 在有限支持、归一化和 mutual support 下，`Σ p(q/p−1−log(q/p)) = −Σ p log(q/p)`；对应附录中 `E_p[k_3(ρ)] = KL(p\|q)` 的归一化代数。 |
| 有限状态 TV 误差步骤 | `ErrorDecomp.finite_tv_expectation_bound` | `PROVEN` | 在独立文件 `formal/ErrorDecomp.lean` 中，对满足 `|g_i|≤G` 的有限 context 函数严格证明 `|E_q[g]-E_p[g]| ≤ 2G·TV(p,q)`；这是 `eq:error-decomp` 每个位置的有限状态代数核心。 |
| 有限时域误差项求和 | `ErrorDecomp.finite_error_decomposition_bound` | `PROVEN` | 在显式给出有限时域性能差分表示 `E=Σ_t(E_{q_t}g_t-E_{p_t}g_t)` 后，严格证明总误差由逐位置 `2G_t TV_t` 之和控制；概率/自回归 identity 仍由前提承载。 |
| `rem:seq-tv-sum`：完整历史树的加权 TV 界 | `prefixTV_step_weighted`, `sequenceTV_le_expected_tokenTV` (`TreeTV.lean`) | `PROVEN` | 从归一化路径概率直接推出 `TV(d_{t+1}^p,d_{t+1}^q) ≤ TV(d_t^p,d_t^q)+E_{d_t^p}[TV(p(.|h),q(.|h))]` 及其加和；不需要 support 假设。旧 `SequenceTV.finite_kernel_tv_step` 仅是每行统一 epsilon 上界，保留为辅助结果，不再代替加权定理。 |
| 有限 horizon 乘积展开 | `finite_horizon_product_telescoping` | `PROVEN` | 对任意有限 ratio 列表，`Πρ−1=Σ_t(ρ_t−1)Π_{j>t}ρ_j`，为 surrogate 恒等式的代数核心。 |
| 有限轨迹 surrogate 恒等式 | `surrogate_minus_remainder`, `finite_horizon_surrogate_identity`, `AutoregressiveTree.trajectory_surrogate_remainder_identity` | `PROVEN` | 先以有限自回归策略树递归构造每条历史的 token ratio，再严格验证 `J(q)-J(p)=L_p-Δ`；同时保留任意有限 trajectory support 的纯代数版本。 |
| 自回归树与论文 trajectory surrogate 的连接 | `AutoregressiveTree.trajectory_surrogate_bridge`, `trajectory_change_of_measure`, `trajectory_surrogate_eq`, `trajectory_remainder_bound` | `PROVEN` | 在每个历史条件分布归一化且 `q<<p` 下，从树递归而非外加 change-of-measure 前提推出完整轨迹密度比、sequence-ratio surrogate 与 per-step rollout surrogate 的等价，以及二次/线性 remainder 界。 |
| Prompt 混合分布的外层求和 | `AutoregressiveTree.family_surrogate_remainder_identity`, `family_remainder_quadratic`, `family_remainder_linear` | `PROVEN` | 对有限 prompt 支持及归一化 prompt 权重，逐 prompt 的 trajectory identity 和两种 remainder 界可严格加权求和；Adaptive bound 的对应组合由 `AdaptiveBound.lean` 补全。随机长度 reduction 的目标与误差单列在下方。 |
| EOS/action-mask 的有限树接口 | `AutoregressiveTree.masked_surrogate_eq_surrogate`, `trajectory_masked_surrogate_eq` | `PROVEN` | 对二值 context-predictable mask，若 mask 为零的 post-EOS 历史上两策略共享同一条件分布，则被 mask 的 surrogate 与完整树 surrogate 相等，且仍等于论文的 trajectory ratio-sum；这明确了 padding 不产生额外 ratio 项。active-position 重编号及随机长度 reduction 由 `PaddedPrefix.lean`、`BatchReduction.lean` 单列验证。 |
| `prop:surrogate-equivalence` 的有限条件分布定式 | `SurrogateBridge.finite_surrogate_baseline_offset`, `finite_surrogate_equivalence`, `finite_reward_ratio_surrogate_equivalence`, `finite_context_surrogate_equivalence` | `PROVEN` | 显式编码 rollout action law、给定 action 后的 rollout continuation law 及其归一化。在 `q<<p` 下，原始 trajectory surrogate 减去其 rollout 值精确等于逐步 advantage surrogate，也精确等于 sequence-reward ratio-residual surrogate；允许 off-policy 的当前策略、零概率 action、固定 context 权重和 predictable action mask。完整有限历史树的性能界另由 `AutoregressiveTree.lean` 给出；有限 prompt 混合和 EOS 接口已在相应树模块连接；神经网络参数化仍需独立的可微性建模。 |
| 独立 baseline 的边缘化 | `SurrogateBridge.finite_independent_baseline_average` | `PROVEN` | 在显式的 trajectory/独立 baseline 乘积质量上积分，随机 baseline 精确替换为其均值；不是把独立性消去等式直接写成前提。 |
| `prop:surrogate-equivalence` 的梯度关系 | `SurrogateBridge.finite_surrogate_derivative_equivalence` | `PROVEN` | 对任意实赋范参数空间，rollout/continuation/baseline 固定，当前策略族归一化且 `q<<p`，从 advantage surrogate 的 Fréchet 导数推出 trajectory surrogate 的同一导数；不声称 off-policy 时该导数等于真实回报导数。 |
| action-dependent prefix gate 的精确 baseline 项 | `SurrogateBridge.finite_masked_surrogate_identity` | `PROVEN` | 严格得到 `E_p[(R-b)rho M]=sum q M(Q-V)+(V-b)sum q M`。即使 gate 已 detach，最后的当前策略接受质量仍可能随策略改变，不能沿用无 gate 的常数抵消。 |
| `prop:promax-objective-split` | `Objective.finite_promax_objective_decomposition`, `finite_clipping_penalty_nonnegative`, `finite_clipped_le_unclipped`, `finite_promax_objective_error_bound` | `PROVEN` | 对实际有限 batch 的任意非负 reduction 权重，并显式要求 represented support 上的两个 ratio 系数 `r_k,w_k≥0`，保留原始/归一化 advantage、prefix gate 和标准 PPO clipping，严格证明 `S_PM=U-G+N-C_clip`、`C_clip≥0` 及绝对差上界。适用 token mean 或 response mean；浮点误差、dual-clip 模式和随机长度归一化的总体目标含义不在该定理内。 |
| 随机长度 reduction 与 token/response 目标 | `ratio_expectation_gap`, `iid_token_mean_mse_of_ratio`, `iid_token_mean_tail`, `microbatch_token_gap`, `sample_weighted_response_means` (`BatchReduction.lean`) | `PROVEN` | 精确分解 `E[Z/L]-E[Z]/E[L]` 的长度协方差项；对下界长度的 iid blocks 证明 self-normalized token ratio 的有限样本 MSE 和 Chebyshev 尾界，并形式化动态 microbatch 的 response-count weighting 与全局 token mean 的差异。 |
| 实际自回归 RLOO population bridge | `response_ratio_residual_mean_zero`, `iid_rloo_surrogate_increment_expectation`, `group_surrogate_increment_expectation`, `group_active_length_expectation`, `group_surrogate_ratio`, `group_surrogate_ratio_of_positive_length` (`RLOOSurrogate.lean`) | `PROVEN` | 在有限历史树、任意历史依赖 token policy 和 iid rollout group 下，trajectory ratio-sum residual 的 rollout 均值为零；实际 leave-one-out estimator 的期望等于 raw population surrogate。长度归一化后的 group ratio 代数上给出 surrogate/expected-length；其作为 token-normalized target 的统计解释使用 `group_surrogate_ratio_of_positive_length` 的严格正 expected active length 条件。 |
| 固定候选的有限 performance certificate | `performance_lower_bound_from_estimate`, `performance_positive_from_estimate`, `false_improvement_certificate_bound` (`BatchPerformance.lean`) | `PROVEN` | 将已知 population performance lower bound 与 ratio estimator 的有限误差合成；false-improvement bound 只对独立评估块上的固定 candidate policy 有效，不是同一批数据选候选后的 uniform confidence bound。 |
| `thm:promax-certificate`：实际目标到统计量 | `response_residual_eq_token_sum`, `group_increment_eq_token_sum`, `corrected_batch_change_eq_rloo_ratio`, `corrected_block_objective_eq` (`ProMaxBatch.lean`, `ProMaxCertificate.lean`) | `PROVEN` | 从实际 token ratio 和 RLOO baseline 推出 `ΔS_PM+ΔG−ΔN+ΔC_clip = ΣZ/ΣL`。保留任意 batch-dependent gate、实际归一化/可选 fallback、snapshot IS 系数和 PPO clipping；显式要求三策略 ratio 分解、参考策略 `r₀w=1` 和 padding residual 为零。参考端是 rollout 策略 p，不默认等于 frozen old；clipping penalty 的差分不保证非负。 |
| `thm:promax-certificate`：分组概率与总体目标 | `block_mass_nonneg`, `block_mass_normalized`, `block_increment_expectation`, `block_length_expectation`, `block_ratio_target`, `family_performance_from_block_target` (`ProMaxCertificate.lean`) | `PROVEN` | 从有限 prompt law 和归一化历史树构造实际 RLOO block law，证明 `E Z=n S`、`E L=n μ_L` 及 token target `κ=S/μ_L`。同一有限策略模型直接给出 Adaptive performance 下界，不把总体均值或性能关系作为独立前提。独立单位是完整组，不把共享 baseline 的贡献误作独立样本。 |
| `thm:promax-certificate`：统一置信界与回报保证 | `finite_event_union_bound`, `iid_token_mean_uniform_tail`, `selected_false_improvement_bound`, `rloo_uniform_estimation_bound`, `rloo_performance_from_estimate`, `rloo_selected_false_improvement_bound` (`FiniteSelection.lean`, `ProMaxCertificate.lean`) | `PROVEN` | 对评估前固定的有限候选族，实际 iid RLOO blocks 上同时误差及所选策略错误改进证书的概率均不超过 `Σ_j v_j/[K(nℓ)²η_j²]`。候选估计之间无需独立，选择器可以使用全部评估数据。用同批评估数据新拟合策略不自动满足 fixed-family 条件；`v_j`、`μ_L`、Adaptive cost 是总体量，实际数值使用需合法值或保守界。 |
| `cor:promax-energy-certificate`：历史树加性 energy | `local_ratio_residual_mean_zero`, `residual_square_step`, `trajectory_residual_second_moment`, `response_residual_energy`, `ratio_energy_eq_prefix_sum` (`SurrogateEnergy.lean`) | `PROVEN` | 实际归一化历史树上 `E[(Σ_t(ρ_t−1))²]=Σ_t E[χ²(q(·|c_t)‖p(·|c_t))]`。交叉项由条件均值为零消失，不假设 token 独立，不使用乘积 ratio 的指数二阶矩界；支持包含共同零概率动作。 |
| `cor:promax-energy-certificate`：实际 RLOO group 矩界 | `loo_baseline_abs_bound`, `rloo_estimator_square_bound`, `rloo_estimator_second_moment_bound`, `block_surrogate_second_moment` (`EmpiricalCertificate.lean`) | `PROVEN` | 在 `abs(R)≤B` 下，从实际 leave-one-out 平均证明 `abs(A_i)≤2B`，以有限 Cauchy–Schwarz 和 iid group 边缘分布推出 `E[(Z/n)²]≤4B² Eenergy`。保留 sibling 的共享 baseline；此界不声称额外 `1/n` 方差收益。 |
| `cor:promax-energy-certificate`：观测长度重缩放 | `observed_token_rescaling`, `corrected_objective_rescaling`, `block_surrogate_expectation`, `rescaled_surrogate_unbiased` (`EmpiricalCertificate.lean`) | `PROVEN` | `D/(Kn) × (ΔS_PM+ΔG−ΔN+ΔC_clip)=K⁻¹Σ_b Z_b/n`，其期望精确等于 raw population surrogate。使用实际 token 总数 `D` 的代数消去，不把样本平均长度直接代入旧的总体长度公式。保留 batch-dependent gate、实际 advantage/fallback、三策略 ratio 和标准 PPO clipping。 |
| `cor:promax-energy-certificate`：统一尾界和性能保证 | `iid_sample_mean_tail_of_second_moment`, `rescaled_surrogate_uniform_tail`, `rescaled_performance_from_estimate`, `rescaled_selected_false_improvement_bound` (`EmpiricalCertificate.lean`) | `PROVEN` | 实际独立 RLOO blocks 上统一失败概率 `≤Σ_j 4B² Eenergy_j/(K ξ_j²)`，真回报提升下界为 `S_hat_j−ξ_j−AdaptiveCost_j`。控制固定候选族中任意评估数据依赖选择器的错误改进证书，已消去未知总体平均长度和旧 residual 二阶矩 `v_j`；数值使用仍需合法 energy/Adaptive cost 上界。 |
| energy 的 horizon 与 EOS 界 | `local_ratio_energy_of_bound`, `ratio_energy_horizon_bound`, `local_ratio_energy_zero_of_equal`, `family_ratio_energy_of_ratio_bound` (`SurrogateEnergy.lean`, `EmpiricalCertificate.lean`) | `PROVEN` | 若 horizon 内所有条件分布的正 rollout 概率 action 满足 `abs(ρ−1)≤d`，则 `Eenergy≤T d²`；共同 EOS kernel 的 energy 为零。由此采样裕量的 horizon 阶可为 `sqrt(T/K)`，不宣称整个 Adaptive cost 或实际优化收敛率具有同样阶。共享零概率动作不强迫 `d≥1`；仅样本中的最大 ratio 不能证明总体上界。 |
| `prop:rloo-group-variance`：实际 group 与 sample covariance | `rloo_eq_sample_covariance`, `rloo_reward_translation` (`RLOOCovariance.lean`) | `PROVEN` | 对每个实际组证明 `U=(nΣXY−ΣXΣY)/(n(n−1))`，共同 reward 平移不改变 RLOO。使用实际 leave-one-out 均值，未替换成独立 baseline 副本。 |
| `prop:rloo-group-variance`：精确共享 baseline 方差 | `iid_sample_sum_square`, `iid_sum_product_cross`, `iid_centered_sums_mixed_square`, `centered_rloo_variance`, `actual_rloo_variance` (`RLOOCovariance.lean`) | `PROVEN` | 从有限 iid law 按样本数递推二阶和四阶混合矩，严格得到 `Var(U)=(E[X²Y²]−θ²)/n+(E[X²]E[Y²]+θ²)/(n(n−1))`；这里 `X=R−ER`、`EY=0`、`θ=E[XY]`。重叠 sample indices 的贡献完整保留，协方差不是独立前提；适用于 ratio residual 或任意零均值 scalar score。 |
| `prop:rloo-group-variance`：响应数收益与 energy 界 | `covariance_variance_coefficients_nonneg`, `actual_rloo_variance_bound`, `rloo_variance_factor_rate`, `response_centered_covariance`, `response_rloo_variance_exact`, `response_rloo_variance_bound` (`RLOOCovariance.lean`, `GroupCertificate.lean`) | `PROVEN` | 两个方差系数非负；在 `abs(R)≤B` 下实际条件方差 `≤c_n B² Eenergy_x`，`c_n=4/n+1/(n(n−1))≤5/n`。树模型上 `θ_x` 精确等于 prompt surrogate，`E[Y²]` 精确等于前述 ratio energy，未把这些对应关系作为独立假设。 |
| `prop:rloo-group-variance`：prompt 方差与完整证书 | `block_surrogate_variance_exact`, `family_rloo_variance_bound`, `family_rloo_variance_combined_bound`, `group_surrogate_mse_exact`, `group_surrogate_uniform_tail`, `group_selected_false_improvement_bound` (`GroupCertificate.lean`) | `PROVEN` | 实际 prompt/group law 给出 `V(n)=Var_x(S_x)+E_x Var(U|x)`，经验 surrogate 的 MSE 精确为 `V(n)/K`；固定候选族统一失败概率与错误改进证书概率 `≤Σ_j V_j(n)/(K ξ_j²)`。进一步 `V(n)≤min(4B²Eenergy,Var_x(S_x)+c_nB²Eenergy)`。固定 prompt 时采样裕量可按 `sqrt(Eenergy/(Kn))` 控制；异质 prompt 的差异项不因仅增加 sibling count 而消失。沿用实际 corrected ProMax 目标等式和 Adaptive 下界，不改 gate/归一化/clipping。 |
| `cor:covered-policy-certificate`：实际 RLOO 的 cover 逼近误差 | `rloo_estimator_difference_bound`, `family_surrogate_difference_of_residual_cover`, `empirical_surrogate_difference_of_residual_cover`, `surrogate_deviation_transfer` (`PolicyCover.lean`) | `PROVEN` | 若所有 prompt/response 上的加性 ratio residual 与固定 center 相差不超过 `ζ`，则实际 RLOO 经验均值之差 `≤2Bζ`、总体 surrogate 之差 `≤Bζ`；得到完整偏差转移 `abs(S_hatθ−Sθ)≤abs(S_hatcenter−Scenter)+3Bζ`。共享 baseline 保留，未假设组内贡献独立。 |
| `cor:covered-policy-certificate`：连续类拟合与真实回报 | `covered_surrogate_uniform_tail`, `covered_performance_lower_bound`, `covered_selected_false_improvement_bound` (`PolicyCover.lean`) | `PROVEN` | 任意参数类型 `Θ`（不要求有限）的预设策略类，若具有固定有限 cover，则全类偏差超过 `ξ_{ν(θ)}+3Bζ` 的概率 `≤Σ_j V_j(n)/(Kξ_j²)`。在相应 Adaptive 条件下，真回报下界为 `S_hatθ−AdaptiveCostθ−ξ_{ν(θ)}−3Bζ`，任意使用全部评估数据的拟合器均满足同一错误改进证书界。cover 及 class 必须在评估前确定；不声称 unrestricted neural policy class 的 cover 足够小。 |
| `cor:covered-policy-certificate`：token ratio 到 cover 半径 | `response_residual_difference_of_token_cover` (`PolicyCover.lean`) | `PROVEN` | 全部长度小于 `T` 的有限历史上 token ratio 的一致逼近误差 `≤d`，推出 response residual 的一致误差 `≤Td`。共同零概率动作使用有限模型的总定义 ratio；这不是仅在已观察 batch 上检查 ratio 的经验条件。 |
| `cor:logit-calibration`：归一化权重的局部分布界 | `tilt_energy_identity`, `tilt_energy_bound`, `tilt_reciprocal_moment`, `tilt_kl_bound`, `tilt_tv_bound`, `policy_tilt_local_bounds` (`LogitBounds.lean`) | `PROVEN` | 对实际 `q_a=p_a w_a/E_p[w]`、`0<lo≤w_a≤hi`，保留 normalizer，推出 `χ²(q‖p), KL(p‖q)≤C=(hi−lo)²/(4lo hi)`、`TV≤√C/2`。有限策略桥包含共同零概率动作，不需要最小正 token 概率，也不假设 token 独立。 |
| `app:logit-calibration`：energy 常数的达到例 | `tilt_energy_bound_sharp` (`LogitBounds.lean`) | `PROVEN` | 两个动作权重 `lo,hi`，概率 `hi/(lo+hi),lo/(lo+hi)` 的 energy 精确等于 `C`。Lean 定理直接验证该等式；所列概率在正端点下的非负性及和为 1 可由代数核对。只声称 energy 界 sharp，不声称 KL/TV 同时达到各自上界。 |
| `cor:logit-calibration`：实际 softmax 与 shift 不变性 | `softmax_tilt`, `tilt_row_scale`, `softmax_centered_tilt`, `policy_softmax_envelope`, `policy_softmax_local_bounds` (`LogitBounds.lean`) | `PROVEN` | 从 `softmax(z')` 的真实分母推出指数权重倾斜；公共 logit shift 可以逐 context 选择。若增量经 shift 后处于 `[0,ω]`，则权重在 `[1,exp ω]`、`C(ω)=(exp ω−1)²/(4exp ω)`。完整 softmax 使用非空有限 action 类型；shared mask/EOS 由一般 positive-tilt 版本覆盖，不允许改变采样支持。 |
| `cor:logit-calibration`：history tree 与有限 prompt 的证书输入 | `policy_tilt_energy_horizon`, `adaptive_cost_of_local_tv`, `policy_tilt_performance_bound`, `family_tilt_certificate_inputs`, `family_softmax_certificate_inputs` (`LogitBounds.lean`) | `PROVEN` | 全 horizon 的 uniform envelope 推出 `Eenergy≤TC` 和 `B_adaptive≤4B(√C/2)Σ_t min(1,(T−t)√C/2,√((T−t)C/2))`。真实有限 prompt 混合、全历史依赖和 reward lower bound 均保留；更紧的 occupancy-weighted cost 仍可单独使用。 |
| `cor:logit-calibration`：RLOO 统一尾界与奖励保证 | `tilt_surrogate_uniform_tail`, `tilt_performance_from_estimate` (`LogitBounds.lean`) | `PROVEN` | 将显式 energy 界代入实际 iid RLOO blocks 的尾界，失败概率 `≤Σ_j 4B²TC_j/(Kξ_j²)`；观测 corrected ProMax statistic 在偏差事件外给出 `J(q_j)−J(p)≥S_hat_j−ξ_j−Bbar_j`。与已有固定候选选择事件和实际目标等式组合；未假设共享 baseline 的各项独立，也未删去方法变换项。 |
| `cor:covered-policy-certificate` 的 logit 半径校准 | `token_ratio_cover_of_tilt`, `token_ratio_upper_of_tilt`, `response_ratio_cover_of_tilts`, `response_ratio_cover_of_softmax` (`LogitBounds.lean`) | `PROVEN` | center 相对 rollout 的 logit 增量跨度 `≤ω`，member 相对 center 的跨度 `≤r`，全 history 上推出 response residual 半径 `ζ=T exp(ω)(exp(r)−1)`。一般权重版本为 `T(hi₀/lo₀)(hi₁−lo₁)/lo₁`。固定 class/cover 的规模与全状态 envelope 仍需另行建立，评估 batch 上的最大差不够。 |
| `prop:uniform-population`：真实 homogeneous event 与期望增量 | `restricted_iid_mass`, `homogeneous_expectation`, `binary_correction_expectation`, `uniform_patch_difference`, `uniform_patch_expectation`, `rloo_homogeneous_zero` (`UniformScale.lean`) | `PROVEN` | 实际 iid response product law 上，on/off 两分支在 mixed groups 保持任意相同系数，homogeneous groups 从零替换为 `r/n`；sequence-score group mean 的期望差为 `[r₊s^(n−1)−r₋(1−s)^(n−1)]/n` 乘 success-restricted score expectation。全组事件由所有 sibling 共同决定，未假设事件与 score 独立。 |
| `prop:uniform-population`：策略微分及正面机制 | `finite_policy_uniform_correction`, `correction_factor_nonnegative`, `correction_factor_nonnegative_of_signed_rewards`, `zero_one_correction_factor`, `zero_one_correction_positive` (`UniformScale.lean`) | `PROVEN` | 在实际有限可微归一化策略上推导 score 零均值及 success-rate Fréchet 导数，任意参数方向均由同一 multiplier 缩放。`r₋≤0≤r₊` 保证新增项与单 prompt reward gradient 同向；0/1 奖励的 multiplier 精确为 `s^(n−1)/n`。只分析 on-policy、accepted gate、inactive clipping 的 sequence-score statistic；random token denominator、跨 prompt 共享参数聚合与一般 off-policy 更新不自动继承这个同向结论。 |
| `eq:uniform-offset-counterexample`：奖励平移及完整反向更新 | `correction_factor_reward_shift`, `bernoulli_mass_normalized`, `bernoulli_mass_nonnegative`, `bernoulli_score_derivative`, `shifted_reward_derivative`, `two_response_uniform_expectation`, `shifted_uniform_reverses_update` (`UniformScale.lean`) | `PROVEN` | 两个 one-token iid responses、rewards `3,4`、高奖励概率 `1/10`，真实 reward 导数为 1。mixed-group 系数为 `±a` 时，uniform-scale on 的平均 ascent update 是 `a−23/20`；`a≤1` 时 `≤−3/20`。关闭该选项的期望为 `a>0`。此反例针对可选分支的 reward-origin 依赖，不否定一般 corrected objective certificate。 |
| uniform 反例的数值 safeguards/触发条件 | `pair_scale_bounds`, `mixed_pair_sample_variance`, `mixed_pair_above_uniform_tolerance`, `shifted_uniform_reverses_with_safeguards`, `zero_one_two_response_with_safeguards` (`UniformScale.lean`) | `PROVEN` | 混合组 raw RLOO `±1` 的实际实数 scale 为 `min(max(sqrt(2/(2+ε)),ε),10)`，`0<ε≤1` 下在 `(0,1]`；product cap 不激活，反向结论保留 stabilizer/factor clamps。mixed reward 的 sample variance 为 `1/2`、std 大于 `1e−8`，因此不误入 uniform 分支。同模型 0/1 奖励时更新为 `a+s/2>0`。这核对实数数学公式，不宣称已经形式化 PyTorch 浮点 kernel。 |
| `app:uniform-scale`：零奖励与恒定奖励噪声 | `homogeneous_zero_reward`, `constant_reward_mean_zero`, `constant_reward_variance` (`UniformScale.lean`) | `PROVEN` | 全零组仍然零更新；若整个 response support 的奖励恒为 `c`，uniform branch 的 score group mean 期望为零，每个零均值 scalar score 方向的方差为 `c²/n³ × E[score²]`，普通 RLOO 恒为零。非零系数不等于有效 population signal。 |
| `prop:binary-complete-update`：完整二元更新的 population bridge（含随机长度分母） | `coordinate_expectation`, `complete_update_expectation`, `implementation_policy_direction` (`BinaryUpdate.lean`) | `PROVEN` | 对任意按完整 sibling 标签决定的系数，先逐坐标积分再得到共同 multiplier；在 score 零均值、正长度且长度上界有限、正因子下界 `ε` 等明确条件下，实际 token 分母不会破坏同向性，并给出 multiplier `≥ε/L_max`。uniform 0/1 boost 的非负 contrast 保留。该结果仍是有限 on-policy 模型；代码浮点/非有限 fallback 需满足对应模型条件。 |
| `prop:binary-length-covariance`：类别内长度变化的精确余项 | `class_covariance_decomposition`, `general_update_decomposition`, `covariance_error_square_bound`, `covariance_error_abs_bound`, `response_update_decomposition`, `response_class_multiplier_lower_bound` (`BinaryLength.lean`) | `PROVEN` | 任意内容相关长度/归一化系数下，完整更新精确分为类别均值 multiplier 乘成功 score 与类别内 coefficient--score covariance；有限 Cauchy--Schwarz 界为 `sqrt(V_C*(m+1)*V_f)`。不假设系数与 score 独立；类别内固定长度时 covariance 为零并恢复严格同向结论。 |
| `cor:binary-local-improvement`：完整期望更新的真实奖励局部改进 | `population_update_eq_expected_score`, `success_gradient_correct`, `population_ascent_bound`, `population_local_improvement`, `response_local_improvement`, `class_length_local_improvement` (`BinaryAscent.lean`) | `PROVEN` | 有限可微策略下，从 score 和 Riesz 表示导出实际奖励梯度与期望更新，而非将其身份作为额外前提。实际 class multiplier 乘梯度模平方严格大于类别内 covariance 上界时，所有充分小的正步长严格提高真实成功率；保留任意内容相关长度。`η/L_max` 只给出另一个充分条件。类别内长度固定且梯度非零时，无需额外 covariance margin。固定 prompt、on-policy、全接受 gate、inactive clipping；不声称每次随机 Adam 更新单调改善。 |
| 随机 reduction 的单位澄清 | `group_ratio_ne_response_mean` (`BatchReduction.lean`) | `PROVEN` | contributions `(1,-1)`、lengths `(1,3)` 时，response mean 为 `1/3`，group token ratio 为 `0`。将 RLOO 组作为独立采样单位不能同时把 mean-of-response-means 改为 ratio-of-group-sums。 |
| `rem:fh-tr-bound` | `AutoregressiveTree.surrogate_error_quadratic`, `surrogate_error_linear`, `performance_lower_bound` | `PROVEN` | 在有限词表、完整历史依赖的归一化自回归策略和有界终端奖励下，严格证明 rollout-occupancy surrogate 的二次界 `2 R_max T(T-1) ε²`、expected-sum-TV 线性界 `4 R_max Σ_t D̄_t` 及取较小误差界后的真实性能下界。 |
| `thm:adaptive-bound`, `eq:adaptive-bound`, `app:adaptive-proof` | `surrogate_error_future_identity`, `futureTV_le_adaptive`, `surrogate_error_adaptive_root`, `trajectory_remainder_adaptive`, `family_remainder_adaptive`, `family_performance_lower_bound_adaptive` (`AdaptiveBound.lean`) | `PROVEN` | 从精确历史树误差递推推出完整 `4 R_max Σ_t D̄_t min(1,(T-t)ε,sqrt((T-t)δ/2))` 界，直接接入实际 reward-weighted trajectory remainder、有限 prompt law 和真实回报下界。局部 TV/KL 条件只需在生成 horizon 内成立，不假设 token 独立，不把误差分解或换测度等式作为额外前提。 |
| `lem:wt-second-moment` 的有限树递推接口 | `AutoregressiveTree.prefix_second_moment_bound`, `local_second_moment_of_centered_log_mgf`, `finite_log_ratio_mean_nonpos` | `PROVEN` | 前者证明若每个历史的局部平方权重和 `Σ_a p(a|h)ρ(h,a)^2` 不超过 `C≥0`，则深度 `t` 的 prefix second moment 不超过 `C^t`；`local_second_moment_of_centered_log_mgf` 从中心化 log-ratio 的 λ=2 MGF 上界和显式非正条件均值推出 `C=exp(2σ²)` 的局部条件；`finite_log_ratio_mean_nonpos` 在严格正的有限归一化策略上证明该均值条件来自 KL 的非负性。一般随机历史上的 filtration/tower 构造仍待形式化。 |
| `lem:wt-second-moment` 的有限自回归模型 | `logRatioMean_nonpos`, `local_ratio_second_moment_of_mgf`, `prefixSecondMoment_bound_reachable`, `trajectory_second_moment_of_mgf` | `PROVEN` | 在有限词表、历史依赖的归一化策略、mutual support 和 horizon 内 rollout 可达历史的 lambda=2 中心化 MGF 界下，直接证明 `E_p[W_t²]≤exp(2σ²t)`。不可达分支的 incoming probability 为零，无需局部 MGF 假设。包含共同零概率动作和 EOS 吸收转移；无需 token 独立性。 |
| `lem:prefix-density-ratio`, `lem:tv-prefix-ratio`, `lem:tv-accepted-prefix` 的有限历史分布 | `pathMass_ratio`, `pathMass_normalized`, `pathSum_mass_value`, `acceptedPrefixTV_identity`, `trajectory_ratio_mass`, `trajectory_ratio_variance` | `PROVEN` | 用归一化策略逐步相乘构造实际 prefix 概率，证明总质量为 1、换测度和半 L1 恒等式，未把路径分布一致性或均值为 1 作为额外前提。对于过滤后可能不同质量的测度，TV 明确采用半 L1 约定，不等同于一般的事件上确界。 |
| `cor:tv-prefix-subgaussian`, `cor:tv-accepted-subgaussian`, `cor:effective-horizon` 的有限历史模型 | `value_cauchy_sq`, `accepted_prefix_residual_of_mgf`, `prefixTV_of_mgf`, `acceptedPrefixTV_of_mgf`, `effective_horizon_of_mgf` | `PROVEN` | 从同一个有限策略模型及上述 MGF 条件推得未过滤 TV 界、`sqrt(alpha_t)` 接受率界和 `sqrt(sum alpha_t)` 汇总界。gate 可以依赖完整的 action-extended prefix，与 ratio 无需独立；二值 prefix gate 是 `0≤M≤1` 的特例。这里衡量未重新归一化的过滤质量，不宣称条件分布更接近或 masked gradient 无偏。 |
| `thm:per-position-error-subgaussian`, `cor:concentration-improvement` 的有限历史模型 | `additive_eq_prefix_sum`, `value_drift_of_mgf`, `surrogate_error_of_mgf`, `performance_lower_bound_of_mgf`, `performance_improves_of_mgf` | `PROVEN` | 将逐位置 surrogate 与同一策略树生成的 prefix law 精确对应；从 MGF 条件和有界终端奖励证明误差界 `2R_max Σ_{k=0}^{T-1} sqrt(exp(2σ²k)-1)`，进而给出真实性能下界及 raw population surrogate 超过误差界时的严格回报改进。未将有限 batch PPO objective 的上升等同于总体 surrogate 的上升；sampling/reduction 和方法变换项仍需单独处理。 |
| `lem:k3-inlier-certificate` / `lem:anom-rate-implies-pre` | `logK3_interval_certificate`, `k3_log_inlier_bound`, `k3_symmetric_threshold_positive`, `k3_abnormal_prefix_average_bound` (`K3Inlier.lean`) | `PROVEN` | 用 `F(z)=exp(z)-1-z` 两侧严格单调性证明正确逆区间。若 `delta≤min(F(-tau0),F(tau0))`，样本级 k3 inlier 推出 `abs(log rho)≤tau0`，再与有限异常比例界合成；`tau0>0` 时容许的阈值严格正。总体 conditional KL 阈值不能直接代替样本级阈值。 |
| `rem:coupling`：首次分歧后的 context shift | `prefixTV_mono`, `prefixTV_next_of_equal_prefix_mass`, `first_disagreement_propagates` (`TreeTV.lean`) | `PROVEN` | 若首次分歧前所有 rollout 可达 histories 的 conditional policies 相同，且某个正 rollout mass context 的 local TV 为正，则此后完整 prefix 分布的 TV 严格为正。完整历史 TV 随深度不减；不需要原文的后续 mutual-support 条件。 |
| `app:gamma-one` / token expansion | `sparse_terminal_return`, `sparse_terminal_return_gamma_one`, `gamma_one_iff_constant_sparse_return` (`SparseReturn.lean`) | `PROVEN` | 从 reward-to-go 递归证明 sparse terminal return 为 `gamma^n R`。`gamma=1` 保证每个 active token 的 return 都等于 shaped reward；非零 reward 下，要求每种 suffix length 都有此性质也必然推出 `gamma=1`。 |

## 已证伪的论文命题或推导

固定 prompt 和总体目标的最新修订见本节第 15 项；其他历史错误保留原记录。

### 1. 单向 support 假设不足（已修正）

原版 [`tex/main/3-preliminaries.tex`](/volume/pt-train/users/rbliu/github/OpenRLHF/tex/main/3-preliminaries.tex) 的
`asm:support` 曾仅要求

```text
π_roll(y|c) > 0  ->  π_theta(y|c) > 0
```

即 `p << q`（这里 `p=π_roll`, `q=π_theta`）。它没有排除 `q` 在 `p=0` 的位置有额外质量。令有限支持上

```text
p = (1, 0), q = (1/2, 1/2).
```

这满足论文的单向 support 条件，且两者都是概率分布。Lean 定理
`tv_ratio_counterexample_has_paper_support` 验证这些条件；
`tv_ratio_counterexample_explicit` 验证

```text
TV(p,q) = 1/2,
1/2 * E_p[|q/p - 1|] = 1/4.
```

该反例证明原单向假设下以下内容为 `FAILED`：

- `rem:tv-ratio-identity`（当前位于 `tex/main/appendix.tex`）；
- `tex/main/4-method.tex` 的 `lem:tv-prefix-ratio`；
- `tex/main/appendix.tex` 的 `lem:tv-accepted-prefix`；
- 依赖 `E_p[W_t]=1` 的 `cor:tv-prefix-subgaussian`、`cor:tv-accepted-subgaussian`、
  `cor:effective-horizon` 和 `thm:per-position-error-subgaussian`。

同一个反例还由 `counterexample_prefix_ratio_mass_is_not_one` 验证：

```text
E_p[q/p] = 1/2,
```

而不是论文推导所使用的 `1`。论文现已将 `asm:support` 改为 mutual support（双向蕴含），补充了反向绝对连续性
`q(y|c)>0 -> p(y|c)>0`。在该修正假设下，有限支持版本由 `tv_ratio_identity_of_reverse_support` 和
`ratio_mass_eq_one_of_reverse_support` 形式化通过。`PrefixConcentration.lean` 进一步从真实有限历史树的路径质量证明 prefix ratio 归一化、TV 恒等式、集中界和有效 horizon；一般测度论扩展仍另列 `PENDING`。

### 2. 有限 horizon surrogate 的精确恒等式（原假设下失败，修正后有限历史树已证）

原版 [`tex/main/5-theory.tex`](/volume/pt-train/users/rbliu/github/OpenRLHF/tex/main/5-theory.tex) 的
`def:llm-surrogate` 和 `rem:fh-tr-bound` 在单向 Assumption `asm:support` 下声称

```text
J(q)-J(p) = L_p^LLM(q) - Δ^LLM(p,q).
```

取单步 horizon、上述 `p,q` 和奖励 `R=(0,1)`。Lean 定理
`finite_horizon_surrogate_counterexample` 验证：`J(q)-J(p)=1/2`，而 rollout 期望中的 ratio surrogate 和 remainder
都为 `0`。原因是 rollout 期望无法恢复 q-only 轨迹。故该精确恒等式在原 support 假设下为 `FAILED`。论文现已改为 mutual support，并明确 change-of-measure 条件。
`finite_change_of_measure` 证明了有限支持的换测度步骤，`finite_horizon_product_telescoping` 证明了逐轨迹乘积恒等式；
`finite_surrogate_difference_identity` 进一步证明了单步 surrogate 差分。`TreeSurrogate.lean` 已完成完整有限历史树上的路径换测度、trajectory ratio-sum 与 per-step surrogate 的等价及 remainder identity；`PromptMixture.lean` 已接上有限 prompt 混合，predictable absorbing EOS 也已处理。`ProMaxCertificate.lean` 进一步完成随机长度下实际 RLOO 组的总体 token target 与有限候选回报证书；`PolicyCover.lean` 将其扩展到具有固定有限 cover 的同批拟合策略类。一般测度模型，以及实际神经策略类的有效 cover 构造/复杂度，仍有后续义务。

### 3. `k_3` 的 KL proxy 恒等式（原假设下失败，修正后有限支持已证）

原版 [`tex/main/appendix.tex`](/volume/pt-train/users/rbliu/github/OpenRLHF/tex/main/appendix.tex) 在单向 support 下写道

```text
k_3(ρ)=ρ-1-log ρ,
E_p[k_3(ρ)] = KL(p || q).
```

这个计算再次使用 `E_p[ρ]=1`。同一个两点反例下，Lean 定理 `k3_proxy_counterexample` 验证两边不相等：左边是
`log 2 - 1/2`，右边是 `log 2`。因此该等式在原单向 support 下为 `FAILED`。论文现已在附录明确要求 mutual support；`finite_k3_proxy_eq_reverse_kl` 已完成有限支持的完整恒等式。一般条件期望版本仍为 `PENDING`。

### 4. abnormal-token lemma 的指数不等式错误（已修正）

历史版本的 [`tex/main/appendix.tex`](/volume/pt-train/users/rbliu/github/OpenRLHF/tex/main/appendix.tex) 使用

```text
exp(u) ≥ 1 + u + u²/2   for all u ∈ ℝ.
```

取 `u=-2` 时右边等于 `1`，而 `exp(-2)<1`。Lean 定理
`exp_quadratic_claim_is_false` 编译通过并否定该全称命题。因此原版 `lem:anom-rate-implies-pre` 证明为 `FAILED`。论文现已删除错误指数不等式，改为显式假设
`B_s=0→|z_s|≤τ₀` 并增加 `Λ_±≥τ₀`。对应的有限序列命题 `abnormal_prefix_average_bound` 已在 Lean 中
`PROVEN`。`K3Inlier.lean` 现已补上正确的双侧逆区间证书和非空正阈值选择，并以 `k3_abnormal_prefix_average_bound` 完成样本 proxy 阈值到 prefix 平均约束的合成，不再将 inlier 接口标为待证明。

### 5. 符号保持不推出整体梯度方向保持（已修正）

摘要原写 `preserves gradient direction`，超过已证明的 advantage sign/order 结论。
两个有符号向量贡献 `(1,0)` 和 `(0,-1)` 的和是 `(1,-1)`；不等正系数缩放可改变方向。
`Objective.asymmetric_scaling_direction_counterexample` 验证一般反例；
`Objective.normalized_scaling_direction_counterexample` 进一步满足实际的归一化约束：一个 `+1` token 和四个 `-1` token，
`alpha=2, beta=1/2`，归一化均值为零且二阶矩为一，但总梯度可从 `(1,-1)` 变为 `(2,-1/2)`，二者不成比例。
因此一般的整体梯度方向保持命题为 `FAILED`，现已将摘要和正文统一为 advantage 符号及同号组内顺序保持。

### 6. 原始 surrogate 的理论联系被错误写弱（已修正）

原 `3-preliminaries.tex` 将独立 baseline 的未裁剪 trajectory surrogate 称为逐步 surrogate 的非等价近似，
`5-theory.tex` 亦重复该说法。有限条件分布上的严格结论实际上是
`tilde_L_p(q)-tilde_L_p(p)=L_p(q)=E_p[R sum(rho-1)]`，并不要求 `q=p`；
独立 baseline、predictable EOS mask、固定 rollout 分布与 baseline sampling law 是关键条件。
相关 theorem、上下文求和、独立 baseline 边缘化及 Fréchet 导数转移均已在 `SurrogateBridge.lean` 编译通过。
论文已恢复这一更强联系，并用 `ProMaxObjective.lean` 的精确分解说明真实方法的 gate、归一化和 clipping 所带来的改变。

同时修正了由已证公式直接排除的表述：共同正缩放本身不会扭曲相对大小；所有 token log-ratio 都在区间内时，
其 prefix 平均不能因重复累积而越界。前者现在区分 centering 与 scaling，后者按已证 early/late/持续越界条件表述。
标题中的无条件 `Variance-Reduced` 被替换为已由矩约束定理支持的 `Adaptive Advantage Normalization`。

### 7. Coarse-grained KL 的旧等号条件不充分（已修正并完整证明）

只在 `p>0` 的 token 上检查 ratio 恒定，在单向绝对连续下不能推出聚合 KL 取等号。
`coarse_grain_kl_positive_support_condition_counterexample` 使用同一个两点分布反例严格否定旧说法。
修正后要求每个正 p-mass cell 上整个 cell 内的两分布成正比例。
`CoarseKL.lean` 已完成零质量 cell、p-only/q-only 支持边界和完整 partition assembly；在前提 `p<<q` 下排除 p-only 点，q-only 点则必须被等号条件检查。

### 8. Discounted return 的目标解释需要区分时间权重（措辞已修正）

原附录把 `gamma<1` 的逐 token reward-to-go 直接称为某个 discounted objective 的梯度。
实际公式是 `G_t=gamma^(T-t) R`；标准 discounted-objective policy gradient 还取决于外层时间权重约定。
现改为严格证明 `gamma=1` 与无折扣 terminal reward 的 token expansion 一致，并将小于 1 的情形准确描述为位置相关加权。
这项修订依据展开式和梯度约定核对，不标作已有 Lean policy-gradient 反例。

### 9. 普通 mean baseline 的零和性质被错误否定（已修正）

实验章节的确定性对照表原将 mean baseline 的组内 shaped-reward sum 写成 `not constrained`。
实际上 `Σ_i(r_i-mean(r))=0` 对所有非空有限组成立，已由
`mean_baseline_shaped_sum_zero` 严格证明；表格现将两种 baseline 都标为零。
零和是 response-level 性质，token expansion 后若长度不等，则两者都不保证 token-weighted mean 为零。
这不改变 RLOO 的独立 baseline 性质，也不改变 asymmetric normalization 的 token-moment 作用。

### 10. 有限长度假设应明确统一上限（已澄清）

每条响应各自有限并不自动给出统一的确定性 horizon。
论文现显式规定固定 generation cap `T`，并用共享吸收 EOS 补齐到该上限；这与模型的有限长度生成配置一致。
Adaptive bound 已保留一般奖励尺度的必要因子 `R_max`，完整组合现为 `PROVEN`。

### 11. 独立分组不能改变 reduction 的定义（已修正）

此前正文/附录把 `Z,L` 称为 block 的总贡献和总长度，却把 `E[Z/L]` 同时称为
response-level averaging 的目标。若 block 是完整 RLOO 组，实际 response mean 是
`(1/n)Σ_i Z_i/L_i`，并非 `Σ_i Z_i/Σ_i L_i`。Lean 定理
`group_ratio_ne_response_mean` 用正长度 `(1,3)` 和贡献 `(1,-1)` 验证两者为
`1/3` 与 `0`。现已明确 response 单位、独立 group 单位及各自的 reduction；新性能
证书使用完整 iid RLOO 组和全局 active-token ratio，不声称 response-count 加权的
动态 microbatch 自动重建该 token ratio。

### 12. 将等幅优势的解释外推到真实 binary RLOO（已证伪并修正）

旧机制解释将理想 `±1` 优势下的 `β²=P/N` 外推到二元奖励的实际 RLOO，
据此认为失败 token 较多就会衰减负优势。这里 **`FAILED` 的是外推和普遍的
`β<1` 结论**，不是 `binary_reward_scaling_sq` 的等幅代数。真实 RLOO 的正负
幅度为 `mΔ/(n−1)` 与 `kΔ/(n−1)`，通常不同。

一个高奖励 `1`、九个低奖励 `0`，每条回答长度相同，给出原始优势 `1` 与
`−1/9`。Lean 的 `rare_success_does_not_imply_negative_attenuation` 精确验证
`α=β=3`，所以负幅度从 `1/9` 增到 `1/3`。正文改为由实际 reward group 推出的
一般定理：输出幅度由正负 token 总量决定，系数比由两类平均回答长度决定。
该改动保留原方法，纠正其机制解释；平衡对象是 importance weighting 和
prefix filtering 之前的标量系数质量。

### 13. 归一化消去 baseline 尺度，以及二元奖励的等价操作（已澄清）

RLOO 的独立 baseline 性质适用于 raw estimator；不能把它的公共尺度差异
直接当成理想归一化后仍存在的机制差异。`mean_and_rloo_normalization_equal`
证明，使用同种理想 asymmetric normalization 和相同 active-token 权重后，
mean baseline 与 RLOO 的输出完全相同。

`two_level_token_zscore` 进一步证明，在精确二元奖励下，同一 prompt 内按
相同 token 权重进行普通中心化和 population-standard-deviation 标准化，也与
理想 asymmetric normalization 相同。正文、baseline 对照与实验方案已写明
这些等价范围，避免把等价操作当成两种机制。

多值奖励下保留符号仍有独立意义：`par:multivalued-sign-example` 中的实际
RLOO 反例已同时形式化原始优势、weighted moments 和两种最终符号。
所有 epsilon、clamps 和 fallback 行为保留；新的等价性声明限定理想分支，
实际数值分支继续使用既有精确公式与矩残差界。

### 14. Uniform scale 的零奖励例外和奖励原点依赖（已修正并补强）

原正文称全同奖励时可恢复原本消失的 signal，未排除所有 reward 为零的情形。
`homogeneous_zero_reward` 直接否定“所有全同奖励组都恢复非零信号”的全称解释。
当前文字明确只有非零 reward 会产生非零系数；即便如此，也不自动代表有效梯度。
原有单 token coefficient sign 命题仍正确，本轮没有将其误标为错误。

新增的正面机制结论在固定 prompt、on-policy sequence-score 统计量上证明：
0/1 奖励下，可选分支增加 `s^(n−1)/n × ∇J`，其中 `s` 是成功概率。更一般的
二元奖励 multiplier 为 `[r₊s^(n−1)−r₋(1−s)^(n−1)]/n` 乘 `∇s`，因此
`r₋≤0≤r₊` 是充分的同向条件。混合组仍使用原 normalization/safeguards，
没有为了证明而更换实际方法。随机 token reduction、跨 prompt 共享参数更新等
更一般情形继续由原完整目标/性能证书处理，不将本机制条件强加给所有定理。

同时，Lean 证明了一个完整的 reward-offset 反例：`n=2`、每条回答一个 token、
奖励 `3,4`、高奖励概率 `0.1`，真实 reward 导数为 `1`；开启 uniform scale 后
期望 ascent update `≤−3/20`，关闭时为正。结论保留实际 real-arithmetic
stabilizer 和 factor clamps，并核对 `1e−8` 的 branch tolerance。
把 reward 减去 3 后，同一模型的 0/1 reward update 又为正。这说明 optional
branch 不具有 RLOO 的 reward-translation invariance，而不是 0/1 机制定理失败。

全支持恒定 reward 的期望更新为零，方差可为正；该结果进一步区分非零样本系数
与 population learning signal。正文、算法解释、附录、确定性实验说明、结论及
实现注释已同步；没有删除或修改 uniform-scale 算法计算。

### 15. 固定 prompt 的条件梯度不能直接等同总体梯度（已修正）

原 `rem:rloo-variance` 固定 prompt `x`，却将该条件采样下的 RLOO 期望写为
全文总体目标 `J` 的梯度；全文的 `J` 包含对 prompt law 的平均。这一不加区分的
等式为 `FAILED`：两个等概率 prompt 的 Bernoulli 成功率分别为 `θ` 与 `1−θ`，
则总体奖励恒为 `1/2`、梯度为零，第一个 prompt 的奖励梯度却为 `1`。
`PromptGradient.two_prompt_policy_valid` 验证两组响应分布在 `0<θ<1` 均为
正的归一化概率；`two_prompt_reward` 连接 0/1 奖励，
`fixed_prompt_gradient_ne_global` 严格验证导数不相同。

论文现明确 `E[ĝ|x]=∇J_x`，并单列 `E_x E[ĝ|x]=∇J` 所需的外层平均。
`finite_prompt_rloo_policy_gradient` 在任意固定有限 prompt 权重和实际 iid
RLOO 分组模型上证明总体奖励的 Fréchet 导数等于联合 prompt/response law 下的
估计量期望；联合质量的非负性及归一化也分别证明。没有假设不同 prompt 的
奖励梯度相等，也没有把 baseline 与 response 的条件独立误写成跨 prompt
无条件独立。一般 prompt measure 的积分/微分交换仍保留明确的常规条件。
同样修正 uniform-scale population 命题中的 `∇J` 为固定 prompt 的 `∇J_x`。
这修正目标和采样层级的对应关系，保留方法、固定 prompt 的正面机制结果及
原有总体奖励证书。

## 正文与完整证明的结构对应

此前的结构整理保持当时的方法、定理条件与公式不变；以下记录的是该历史阶段：

- 前缀密度比、TV 恒等式和 accepted-prefix mechanism theorem 现在位于 `4-method.tex` 的 `sec:prefix-mechanism`，紧接前缀 mask 比较；正文保留其适用策略对、过滤质量解释和 stale-snapshot coupling 条件。
- `5-theory.tex` 集中呈现实际目标分解、有限样本奖励证书、观测长度/energy 证书、共享 baseline 精确方差以及有限 cover 扩展。详细方法比较移至附录，以免遮蔽这些主结果。
- `app:subgaussian-perpos` 集中保存可选的条件 MGF 假设、集中性推论及完整证明。这一额外假设没有加入主奖励证书的前提。
- 新增 `app:is-foundations` 保存标准 IS/TV/KL 结论、surrogate–remainder 恒等式及解释、有限 horizon 辅助界和详细比较；新增 `app:surrogate-equivalence` 保存总体 surrogate 等价性证明。原有 label 全部保留。
- 删除了“随机长度 reduction 接口仍未完成”的过时说明，改为引用已经完成的目标分解与奖励证书；一般测度接口仍明确列为未完成。
- **已纠正文案错误**：IcePop 的阈值 mask 原被描述为“消除 extreme-weight bias”。实际能保证的是移除越界样本系数，筛选本身可能引入偏差；现已按这一含义表述。此处是对既有 gate 解释的修正，没有新增或削弱数学定理。

该历史阶段整理前后的自动比对显示：当时全部 **247 个独立展示公式**按忽略空白后的文本逐项保留；**36 个 theorem/proposition/corollary/lemma/assumption/definition 环境**内容完全一致。原有 **112 个 label** 无丢失，新增三个位置标签，共 115 个。当时 42 个 Lean 源文件与结构整理前 SHA256 一致，因而该阶段没有重复运行未改动的 Lean 全量构建。

此前在此基础上新增实际 RLOO normalization 机制、scale invariance、binary token-standardization equivalence 及多值反例，证明集中在 `app:adaptive-mechanism`。摘要、贡献、相关工作、对照方案和结论同步修正；`experience_maker.py` 只修改 docstring/comment，标明理想矩约束与实际数值 safeguards 的区别，没有修改计算行为。

此前新增 `cor:logit-calibration` 与 `app:logit-calibration`，给出局部 energy/KL/TV、完整奖励证书输入及 fixed logit cover 的显式半径。摘要、第三条贡献和结论同步增加这些有条件的校准结果。Kantorovich 型有限加权不等式用于校准既有证书，未包装成新的普适算法优越性；原 energy、占用加权 Adaptive cost 和共享 baseline 精确方差均保留。

本轮在既有 `app:uniform-scale` 内新增 population mechanism、reward-offset 反例和 constant-reward 方差，正文只保留该选项的用途及精确结论引用。实现文件仅更新两行解释注释，计算不变。最新 Lean 与论文验证证据见下文。

本轮继续将 uniform-scale 机制接到完整 token-reduced 二元更新：新增 `BinaryUpdate.lean`，在成功/失败类别决定 active length 的明确充分条件下，形式化实际 additive stabilizer、intermediate cap、positive factor clamps、identity fallback、总 token 分母及 optional uniform boost，证明期望更新是成功概率梯度的共同严格正倍数。对于同一奖励类别内仍有内容相关长度变化的真实任务，论文保留额外 covariance/error 项的必要性，不把该充分条件误写为一般 LLM 训练保证。 本轮又新增 `BinaryLength.lean`，把同一 reward class 内依赖 response content 的长度情形写成精确的 class-mean multiplier 加 covariance 分解，并给出有限 Cauchy–Schwarz 误差界；因此正文只在 covariance 消失或被显式 margin 控制时宣称同向。

## 尚未完成的形式化与实现接口

以下是尚未覆盖的结果或接口，不是对上述已证有限结果的否定。

- **RLOO 一般概率空间扩展**：任意有限 iid group 的 baseline 方差、交叉二阶矩、score 零均值、有限可微策略的回报导数及历史树上的实际 RLOO 无偏性均已证明。此前已完成保留共享 baseline 的 scalar group-average 精确方差及 `1/n` 条件方差界，并实例化有限 prompt 混合和完整采样/回报证书。本轮 `PromptGradientMeasure.lean` 进一步在任意 prompt 测度上，以 Bochner 积分和显式 dominated differentiation 条件证明外层梯度交换；无限/连续 response space 的 RLOO 采样、可积性和条件期望接口仍未形式化。向量 group covariance 的完整矩阵形式未单列形式化；已有每个 scalar signal 的精确方差和单样本 outer-product 矩结果，二者不混称。
- **随机 reduction 与候选选择的剩余边界**：随机长度下 response mean 与 token ratio 的精确 bias decomposition、实际 iid RLOO groups 的总体 target、有限候选统一置信界和 Adaptive performance certificate 已完成；观测长度重缩放证书进一步消去未知总体平均长度。固定有限候选保证现已扩展到具有预设有限 cover 的策略类，允许同批评估数据上的连续参数拟合，额外代价为 `3Bζ` 和有限 center 失败概率之和。本轮已经从 uniform logit envelope 推导 energy、Adaptive cost 和 cover 半径的明确校准公式，但仍未证明一般神经策略类具有规模足够小的 cover，或实际训练满足全历史 envelope。评估 batch 上拟合/检查 cover、观察最大 logit 差，或只启用 prefix gate/PPO clipping，都不能替代这些一致条件。若 class/cover 来自独立训练数据，可以固定训练结果后应用。动态 microbatch 的 response-count weighting 仍不能自动视为全局 token mean；实际浮点训练过程也不在本精确实数证书内。
- **一般概率空间**：有限词表、有限 horizon、任意历史依赖、有限 prompt 混合和共同 absorbing EOS 的换测度、performance identity、KL 链式分解、加权 TV、集中界及改进充分条件均已证明。任意 prompt 测度下的外层 Bochner 积分/微分交换已在 `PromptGradientMeasure.lean` 以支配 Lipschitz 条件证明；可数/连续 response space、RLOO 条件期望以及一般响应空间的采样测度接口仍为 `PENDING`。当前集中界已经只要求 rollout 可达历史的 MGF，不再有“所有不可达历史也必须满足 MGF”的额外条件。
- **数学式到实际浮点训练**：理想归一化、additive stabilizer、product cap 和 factor clamp 的精确实数公式及矩残差已经证明。PyTorch 浮点误差、溢出/NaN、真实 tensor kernel 和 dual-clip 模式未形式化；这不属于宣称的精确实数证书。
- **过滤后的目标解释**：accepted-prefix TV 是未重新归一化过滤测度的 half-L1；gate 包含当前动作。已证接受率和 effective-horizon 界不意味着归一化后条件分布更接近，也不意味着 masked gradient 无偏。`finite_masked_surrogate_identity` 已保留相应 baseline 修正项。

数学假设的适用性也必须保留：例如 mutual support 取决于实际采样支持，MGF 界取决于策略对，cached old/rollout gate 到 current/rollout proxy 的转移取决于显式 coupling 条件。Lean 验证条件成立时的结论，不替实际训练配置证明这些条件。

## 可重复验证

### 2026-09-18：历史复核（PromptGradientMeasure 阶段）

在当前工作树重新执行：

```bash
LEAN_JOBS=3 bash formal/leanw build
```

命令以退出码 0 完成，并输出 `Build completed successfully.`。本次为 Lake 增量构建；输出对已有模块及 `AxiomAudit` 标记为 `Replayed`，
因此属于缓存有效性复核，并非删除产物后的全新编译。重放的审计结果显示 878 个项目 theorem 声明通过公理
依赖检查，仅使用 `propext`、`Classical.choice` 和 `Quot.sound`。输出中存在既有
Lean linter 提示（未使用变量、可简化 tactic 等），但没有编译错误、`sorryAx`、
自定义公理或公理审计失败。该结果只复核当前 Lean 形式化工程，不扩展
`PROVEN` 的模型范围，也不证明论文中列出的 `PENDING` 接口或实际浮点训练语义。

随后使用指定 TeX Live 环境重新运行 `compile.sh`（`PUSH=0`），pdfLaTeX/BibTeX
四轮编译退出码为 0，生成 `tex/main.pdf`。本次论文改动只补充了实现中
`uniform_scale`、近零/非有限 fallback、product cap、additive stabilizer 和 factor
clamp 的伪代码分支，以及 RLOO 条件独立性的准确措辞；没有新增未经 Lean 映射的
数学定理。该编译结果验证排版和引用闭合，不等同于实验结果或 ICLR 接收保证。

在仓库根目录执行：

```bash
LEAN_JOBS=3 bash formal/leanw build
```

本次验证结果：`Build completed successfully.` Lean 工具链、Mathlib、Lake 缓存、临时文件和构建输出都在
`/volume/pt-train/users/rbliu/` 挂载卷；包装脚本读取 cgroup 的 CPU/内存额度，用 `LEAN_NUM_THREADS` 控制 Lean 线程，并给单个进程设置保守的虚拟内存上限。没有向 `/tmp` 或 `/root` 写入构建数据。

全量构建的默认目标现在包含 `AxiomAudit.lean`。它导入全部项目证明模块，逐一遍历
`REINFORCEProMax` 命名空间下的 public theorem，并检查证明及类型的传递公理依赖。
只允许 `propext`、`Classical.choice`、`Quot.sound`；任何 `sorryAx`、自定义公理或
额外计算信任公理都会使构建失败。该检查源文件保存在 `formal/` 中，可随仓库复现，
不再依赖忽略目录 `.lake/` 中的临时手写名单。

本轮最终全量构建通过；AxiomAudit 报告覆盖 **878** 个项目 theorem 声明，仅使用 `propext`、`Classical.choice`、`Quot.sound`。计数包括自动生成的辅助定理，不等于 878 个论文命题。当时的 50 个项目 Lean 模块均可从 AxiomAudit 的导入图到达，无遗漏模块；所有模块的编译产物均晚于相应源文件。新增 `PromptGradient.lean` 与 `PromptGradientMeasure.lean` 共含 **8 个显式 theorem 声明**，验证联合 prompt/response 采样质量的非负性和归一化、有限 prompt 混合下实际 RLOO 的总体奖励导数、任意 prompt 测度下带支配条件的外层梯度交换，以及固定 prompt 梯度不等于总体梯度的反例。此前 `BinaryAscent.lean` 的完整二元期望更新与严格局部奖励改进结果保留。

完整日志为 `formal/.lake/build_incremental_backup_20260918/full_check_prompt_measure.log`，新增模块的定向构建日志为 `formal/.lake/build_incremental_backup_20260918/prompt_measure_targeted.log`。既有模块仍有非阻断的 Lean 风格 linter 提示；新增 `PromptGradient.lean` 与 `PromptGradientMeasure.lean` 构建没有此类提示，没有未完成证明或公理审计失败。新增模块的去注释源码扫描未发现 `sorry`、`admit`、自定义 `axiom` 或 `native_decide`；传递证明依赖另由 AxiomAudit 完整检查。使用 3 个 Lean 线程，最终 Lean 源文件 SHA256 清单保存在 `formal/.lake/build/prompt_measure_lean_sources.sha256`，验证产物全部新于源码。

论文编译使用用户指定环境：

```bash
source /volume/pt-train/users/rbliu/latex/env.sh
TMPDIR=/volume/pt-train/users/rbliu/latex/tmp \
PUSH=0 \
BUILD_DIR=/volume/pt-train/users/rbliu/latex/openrlhf_build_prompt_measure_20260917 \
TEX_CACHE_ROOT=/volume/pt-train/users/rbliu/latex/tmp/openrlhf_tex_cache_prompt_measure_20260917 \
./compile.sh
```

PDF 与 LaTeX 构建日志位于 `/volume/pt-train/users/rbliu/latex/openrlhf_build_prompt_measure_20260917/`，编译入口日志为 `/volume/pt-train/users/rbliu/latex/openrlhf_prompt_measure_compile_20260917.log`。实际使用 TeX Live 2026 的 **pdfLaTeX + BibTeX**，该阶段 PDF 为 **75 页、814014 bytes**；SHA256 为 `a20a11b3b1bddd45bd1516456353428ae151d5f1e3b18cca01847d913a2d9efc`。所有编译缓存和临时文件均放在指定挂载卷，`PUSH=0` 禁用脚本默认的 `commit/push`。

本轮修正的条件 RLOO 无偏性位于 **Remark C.1（第 39 页）**，其后的文字明确有限 prompt 外层平均及反例。uniform-scale population 机制为 Proposition C.9（第 52 页），其中固定 prompt 的梯度记为 `∇J_x`。此前的完整二元更新为 Proposition C.10（第 53 页），类别内长度协方差分解为 Proposition C.11（第 54 页），严格局部真实奖励改进为 Corollary C.12（第 54 页），reward-offset 反例为 Equation 39（第 56 页）。logit 校准为 Corollary 5.8（第 26 页）；前缀机制结果为 Theorem 4.11（第 18 页）；精确组方差为 Proposition 5.6（第 25 页）；cover 扩展为 Corollary 5.7（第 26 页）。页码与编号按本次 `main.aux` 核对。原有实际目标分解、off-policy correction 与有限样本证书均保留。

编译脚本的严格警告检查通过，无未定义引用、未定义 citation、重复 label、overfull 或 underfull box 提示。从主文件递归解析的 **11 个活动 TeX 文件**排除注释后，有 **136 个 label、375 条引用命令（403 个引用目标）**，没有重复或缺失 label。全部 **31 个带标签的 theorem/proposition/corollary/lemma** 在本报告中有形式化映射；命题内部的 equation label 不被重复算作论文命题。历史 uniform-scale 修订前快照中的 129 个 label 全部保留。仓库 `tex/main.pdf` 与构建目录中的 PDF SHA256 一致；全部活动 TeX、引用的 BibTeX 文件及本地 style/bibliography style 输入均早于该 PDF。

`git diff --check` 通过。本轮没有修改训练代码；此前 `experience_maker.py` 的 uniform-scale 注释修订仍保留，去除 docstring 后的 AST 与 `HEAD` 一致，计算行为未变。`loss.py` 中此前修正 PPO KL 诊断量的修改保留。最新 Lean/PDF/引用及 AST 核对数据保存在忽略目录 `formal/.lake/build/prompt_measure_artifact_validation.json`，验证脚本为同目录的 `validate_prompt_measure_artifacts.py`；本文件仍是唯一维护的审计报告。未执行 commit/push。

此前结构整理前的卷上快照位于 `/volume/pt-train/users/rbliu/latex/openrlhf_structure_snapshot_20260915T194610Z/`，机制修订前的快照位于 `/volume/pt-train/users/rbliu/latex/openrlhf_adaptive_snapshot_20260915T200210Z/`，logit 校准前的快照位于 `/volume/pt-train/users/rbliu/latex/openrlhf_logit_snapshot_20260915T203400Z/`。此前 uniform-scale 修订前的快照位于 `/volume/pt-train/users/rbliu/latex/openrlhf_uniform_snapshot_20260915T205408Z/`，保留此前 70 页论文及审计源文件。快照目录名使用运行环境的 UTC 时钟。

工具链和下载缓存位于 `/volume/pt-train/users/rbliu/tools/lean/`；Mathlib/Lake packages 与本项目 Lean 产物位于仓库的 `formal/.lake/`。两处都是挂载卷，不把所有 Lake 产物误记到 tools 目录。

公理检查验证的是所编码命题的逻辑依赖；上述 `PENDING` 接口、假设是否对应实际
采样配置、以及 tensor reduction 的总体目标解释仍需独立核对。

历史记录中的 2026-09-18 冷启动复核（878 定理阶段，并非当前 909 定理版本）：将原有 `formal/.lake/build` 目录移出后，从空构建目录再次执行
`LEAN_JOBS=3 bash formal/leanw build`。冷启动全量构建退出码为 0，最后的
`AxiomAudit` 仍报告 878 个项目 theorem 声明通过，且只使用
`propext`、`Classical.choice` 和 `Quot.sound`。这补充了对增量构建结果的独立复核；
它仍只覆盖当前 Lean 文件中编码的有限模型，不扩展 `PENDING` 接口，也不验证实际
浮点训练或论文投稿结果。


### 2026-09-18：结构实验、精确 masking 数值证明与统计口径修正

- 新增 `StructuralExamples.lean` 的 9 个显式 theorem，使用现有 `PrefixMasking` 的真实 gate 定义，而非独立复制 mask。对任意 `lo ≤ 0 < hi`，证明 early/late spike 的拒绝位置及 token/sequence/prefix 计数分别为 `(1,0,7)`、`(1,0,8)`；持续 `2*hi` 偏移的计数为 `(32,32,32)`。因此涵盖论文 `lo=log(0.9), hi=log(1.1)` 的数学阈值。序列边界相等被接受；Lean 的零基位置与论文的一基位置明确区分。该文件已加入 Lake library 和 `AxiomAudit` 导入图。定向 Lean 检查退出码为 0。

- 修正 `tex/main/5-theory.tex` 算法说明中残留的无条件 baseline 独立性表述：独立性要求给定 prompt 后条件 iid 的响应采样。
- `structural_experiment.py` 原 sign-flip 实现只在分子排除零 raw advantage，分母仍为所有 token，违背论文口径。现以非零 raw active token 数为分母，并保存翻转数、有效 token 数；空分母返回 `None` 而非伪造 0。矩统计也按 raw 非零位置取样。新增四个回归测试覆盖分母、空分母、非零信号被置零及 masking 位置/闭区间边界，`PYTHONPATH=. pytest -q reinforce_pro_max/test_structural_experiment.py` 为 **4 passed**。
- 固定 seed=20260918、groups=1000、n=8，在 CPU PyTorch 2.4.0+cu121、float32、3 个 intra-op 线程重算并保存 `structural_experiment_20260918.json`。有效 token 为 3,785,796，adaptive 翻转 0 个，global 翻转 176,197 个，rate=0.046541599177557375。baseline Pearson 相关性分别为 0.35465508699417114 和 0.0023529427126049995；adaptive 非零 token mean=1.8177719329770525e-9，population variance=1.0；RLOO 组和最大绝对残差=6.258487701416016e-7。与早先线程设置未固定的快照相比，极小 mean 残差变化属于浮点 reduction；当前线程数与 runtime 写入 JSON。
- 修正实验章节末尾“没有 Monte Carlo 结果/可执行脚本”的过时声明，说明 group 内 normalization、跨 response 的 Pearson 相关性、token 加权分母和 population variance。模拟不包含实现 safeguards；删除了没有当前通过证据的“单测已验证 safeguards”表述。所有数值仅为 synthetic structural checks，不能作为 LLM/RLVR benchmark。
- 重新执行现有 `PYTHONPATH=. pytest -q reinforce_pro_max/test_reinforce_max.py`，退出码 **2**，收集阶段报 `ModuleNotFoundError: No module named 'pylatexenc'`。未到达训练函数断言，不报告通过，也不把新增独立脚本测试当作训练集成测试。日志：`/volume/pt-train/users/rbliu/latex/promax_unit_collection_20260918.log`。
- 最新论文采用指定 TeX Live 环境、`PUSH=0` 重新编译，退出码 **0**，严格 warning 检查通过，生成 **76 页 / 822203 bytes** PDF。仓库与构建目录 PDF SHA256 一致：`4eba81a937b130ee369f800437ad93fdb0312d1fa63039f584bec43599024670`。日志：`/volume/pt-train/users/rbliu/latex/compile_structural_20260918.log`；构建目录：`/volume/pt-train/users/rbliu/latex/openrlhf_build_structural_20260918/`。本轮未做 PDF 渲染目检（当前 Python 无 PDF 读取/渲染包），不能将编译成功表述为视觉审阅完成。
- Python 语法检查与 `git diff --check` 通过；未 commit/push，未改动训练计算逻辑。上述修改不扩展一般 response 概率空间或浮点训练的形式化边界。
- （历史记录）包含新增模块的 `LEAN_JOBS=3 bash formal/leanw build` 已结束，退出码 **0**，输出 `Build completed successfully.`；当时 `AxiomAudit` 实际报告 **892 个项目 theorem 声明**，允许依赖仍只有 `propext`、`Classical.choice`、`Quot.sound`。本次是增量构建，新模块和 AxiomAudit 重新编译；计数包括自动生成的辅助声明，不能把 892 当成论文定理数量。新增文件 9 个显式定理，整体审计数较此前 878 增加 14。既有 linter warnings 非阻断，未发现公理审计失败。完整日志：`/volume/pt-train/users/rbliu/latex/lean_structural_20260918.log`。该段计数是历史快照，当前最新计数见文末的实现桥接记录。

### 2026-09-18：实际 PolicyLoss 的 CPU 数值与梯度核验

新增 `reinforce_pro_max/test_policy_loss_contract.py`，通过私有包名加载仓库完整的
`openrlhf/models/utils.py` 和 `openrlhf/models/loss.py` 源文件，避免
`models/__init__.py` 的模型/加速器依赖。未复制、替换或 mock `PolicyLoss`、
`masked_mean` 的实现；这验证实际 loss 源码，**不验证正常包导入或 Ray/vLLM 集成**。

使用独立 Python 标量计算而非另写一套 Tensor loss，交叉核对四种 IS 模式
（TIS、IcePop、sequence mask、prefix mask）、token/response 两种 reduction、
CPU float32/float64 的 loss 值和 current logprob 导数。样本同时包含正负 advantage、
进入/未进入 clipping 的 PPO ratio、不同 response 长度、内部 action-mask 空洞，
且 current/old 与 old/rollout ratio 明确不同。另检查 PPO KL 和 vLLM KL 的来源，
以及 correction 不对 rollout logprob 反传。测试容差分别为 1e-6 和 1e-12；
不构成所有浮点输入、极值/NaN/Inf 或 GPU kernel 的形式化证明。

额外用实际 `PolicyLoss` 实例化 `BatchReduction.dynamic_token_weight_counterexample`：
两个 microbatch 的 token contribution 总和为 0、3，active token 数为 1、3，
response 数相同。response-count weighting 得到 0.5，全局 token mean 得到 0.75，
active-token weighting 恢复 0.75。已核对当前 `replay_buffer.py` 以 partition response
数量计算 `dynamic_loss_scale`，且 `ppo_actor.py` 在 backward 前应用该 scale。
因此论文 §5 的 microbatch 限制有实际 loss 层证据；未将现有动态训练目标宣称为
全局 token mean 的性能证书实例。此测试不覆盖 DeepSpeed 分布式梯度累积。

命令 `PYTHONPATH=. pytest -q reinforce_pro_max/test_policy_loss_contract.py` 退出码 0，
**17 passed**。`git diff --check` 通过。本轮仅新增测试和审计记录，未改论文/Lean
定理/训练逻辑，因此未重新运行未受影响的 LaTeX 和 Lean 构建。此前训练套件
缺少 `pylatexenc` 的收集失败仍未解决，不能以此次 loss 层测试替代它。

### 2026-09-18：解除归一化单测收集阻塞并核对 safeguards

此前 `pylatexenc` 缺失已通过独立审计依赖目录解决，未替换系统 Python 环境：
`/volume/pt-train/users/rbliu/tools/openrlhf_audit_pydeps` 内安装
`pylatexenc==2.10`。随后正常导入又发现系统 torchdata 不提供
`torchdata.stateful_dataloader`，在同一目录加入与本机 torch 2.4 对应的
`torchdata==0.8.0`（两次均使用 `pip --no-deps --target`，不升级 torch）。
以该目录置于 PYTHONPATH 后，原有套件 **19 passed**、退出码 0：

```bash
PYTHONPATH=/volume/pt-train/users/rbliu/tools/openrlhf_audit_pydeps:. \
  pytest -q reinforce_pro_max/test_reinforce_max.py
```

本结果取代此前“该环境中套件尚未成功收集”的当前状态，历史失败记录保留。
日志：`/volume/pt-train/users/rbliu/latex/promax_normal_import_tests_20260918.log`。
仍有 DeepSpeed CPU accelerator 的既有 DeprecationWarning；CPU 单测通过不代表
vLLM CUDA 扩展可用或分布式训练已通过。此前导入日志实际显示缺少 libcuda.so.1。

新增 `test_normalization_safeguards.py`，通过正常包导入调用实际
`_adaptive_token_normalization_single_group`，在 float32/float64 上共 **10 passed**：

- stabilizer 示例 `[1,-2]`、eps=0.1，非零 token population variance 为 `20/21`；
  带一个 active 零 token 时若错用全部 active token，variance 为 `40/63`。
- product cap 示例 `[20000,-20000]`，variance 为 `8e8/(5e8+1e-8)`，约 **1.6**；
  因此不能把 cap 解释成一定降低方差或保持单位方差。
- upper factor clamp 示例 `[1e-4,-1e-3]`，两因子最终都是 10，mean 为
  **-0.0045**，population variance 为 **0.00003025**，符号保留。
- lower factor clamp 示例 `[100,-200]`、eps=0.1，输出 `[10,-20]`，mean 为
  **-5**，population variance 为 **225**。
- 小 signed sum 的 fallback 在实际函数中保持输入不变。

这些是 `NormalizationSafeguards.lean` 已证精确实数公式的具体实现核验，数值结果
与论文 residual 边界一致；未把理想归一化的零均值/单位方差外推到 safeguards。
没有新增数学断言或改变训练代码，本轮不重复未受影响的 Lean/LaTeX 构建。

### 2026-09-18：动态整组筛选与 RLOO 采样假设

核对 `SamplesGenerator._generate_vllm`：启用 `dynamic_filtering` 且各 response
均有 scores 时，按整组平均 score 的严格开区间 `(min_r,max_r)` 接受；默认范围
为 `(0,1)`。被拒绝的组不进入 batch，并请求新 prompt 补充。该操作不同于
loss 内的 prefix gate；不能把 accepted group 的采样律直接当作原始条件 iid
乘积律。固定 prompt 下也可能发生组内依赖，并非只需重新定义 prompt 权重。

新增 `formal/GroupFiltering.lean`，以两条 Bernoulli(1/2) 响应、reward=score
给出 8 个显式 theorem：默认筛选恰好保留 mixed pairs；接受概率 1/2；保留质量
为非负、归一化、按接受概率重归一化的条件质量；保留组的两个 reward 边缘均值
均为 1/2、乘积期望为 0、协方差为 -1/4。以 Bernoulli 成功概率为参数，在 1/2
处的 score 为成功 2 / 失败 -2，raw group RLOO 的期望从筛选前 **1** 变为
筛选后 **2**，真实期望奖励的导数仍为 **1**。这是精确有限反例，不是对所有
动态筛选配置偏差方向或训练效果的断言。定向 Lean 检查通过；已接入 Lake 和
AxiomAudit 导入图。

论文 §5 的性能证书后加入明确的采样说明和上述反例。已有证书本身要求原始
独立块 law，没有推翻该定理；修正的是其对可选训练配置的解释。允许 policy
在筛选训练数据上训练，再在满足其余条件的独立、未做整组筛选的条件 iid
评估 batch 上使用证书；对筛选后训练组直接应用则需要新的采样律分析。

论文以指定 TeX Live 环境、`PUSH=0` 重编译，退出码 0，严格 warning 检查通过。
当前 PDF **76 页**；仓库和构建目录 SHA256 均为
`3be0d5aa98b7680ea44e956d0168eec11f79a0e7379ec777f1a360abb4f6809c`。
日志 `/volume/pt-train/users/rbliu/latex/compile_group_filtering_20260918.log`，
构建目录 `/volume/pt-train/users/rbliu/latex/openrlhf_build_group_filtering_20260918/`。
在独立审计依赖目录加入 `pypdfium2==4.30.0` 后，已渲染目检第 **24 页**新增
采样说明及第 **31 页**结构模拟表，正文/公式/表格无可见截断或重叠；这仅覆盖
两页，不宣称全稿视觉审阅完成。对应 PNG 在同一 latex 根目录下
`group_filtering_review_page_24.png`、`group_filtering_review_page_31.png`。
`git diff --check` 通过，未改训练逻辑，未 commit/push。
- 包含 `GroupFiltering.lean` 后的全工程 `LEAN_JOBS=3 bash formal/leanw build` 已结束，退出码 0，`AxiomAudit` 报告 **906 个项目 theorem 声明**，允许依赖仍只有 `propext`、`Classical.choice`、`Quot.sound`。新增模块及审计目标均成功构建；既有 Lean linter warnings 非阻断。日志：`/volume/pt-train/users/rbliu/latex/lean_group_filtering_20260918.log`。

### 2026-09-18：动态筛选限制同步至贡献摘要和结论

全文检索发现引言贡献列表和结论仍可能让读者把 finite-sample guarantee 理解为适用于训练时筛选后的组。现已在 `tex/main/1-intro.tex` 和 `tex/main/7-conclusion.tex` 明确写出：证书适用于未条件化的 iid evaluation groups；训练时整组 `dynamic_filtering` 会改变保留组的 sampling law，必须使用独立未筛选评估 batch，或另行证明 accepted-group law。

重新运行相关测试集合：
`PYTHONPATH=/volume/pt-train/users/rbliu/tools/openrlhf_audit_pydeps:. pytest -q reinforce_pro_max/test_structural_experiment.py reinforce_pro_max/test_policy_loss_contract.py reinforce_pro_max/test_normalization_safeguards.py reinforce_pro_max/test_reinforce_max.py`
结果 **50 passed, 1 warning**；warning 是 DeepSpeed CPU accelerator 的既有 DeprecationWarning。

第一次 LaTeX 编译因新增结论文字使用 Markdown 反引号而触发 `Missing $ inserted`，已改成 `\\texttt{dynamic\\_filtering}` 后重编译通过。这次错误已记录为修复过程，不是当前构建结果。最新编译退出码 0、严格 warning 检查通过；PDF 76 页，SHA256 为
`e7de9df646b38a3eee82b1877b174103e69a1de614f0786cde102a540c6f2dc1`，构建目录 PDF 与仓库 PDF hash 一致。渲染目检新引言第 3 页和结论第 36 页，新增句子无截断、溢出或重叠。日志：`/volume/pt-train/users/rbliu/latex/compile_filtering_scope2_20260918.log`。

### 2026-09-18：实现公式桥接

新增 `formal/ImplementationBridge.lean`，并接入 `lakefile.toml` 与 `AxiomAudit.lean`。该模块形式化验证三项实现对应关系：

- `exp(logCurrent-logRollout) = exp(logCurrent-logOld) * exp(logOld-logRollout)`，对应代码中 current/old PPO ratio 与 old/rollout vLLM correction ratio 的分解；
- 对正阈值，代码对 prefix average 先 `exp` 再比较区间，与论文对 prefix log-average 比较对数阈值完全等价；
- 将一般阈值写成论文使用的 `1-epsLow` 与 `1+epsHigh` 后，上述等价性仍在显式正性条件下成立。

定向构建 `LEAN_JOBS=3 bash formal/leanw build ImplementationBridge AxiomAudit` 退出码为 0；AxiomAudit 报告 **909** 个项目 theorem 声明通过，仅使用 `propext`、`Classical.choice` 和 `Quot.sound`。这些定理覆盖实数代数对应关系，不声称形式化 PyTorch detach、浮点舍入或 GPU kernel。

论文同步修改后，以 `PUSH=0` 重新执行 `compile.sh`，四轮 pdfLaTeX/BibTeX 编译成功。生成的
`tex/main.pdf` 与构建目录 PDF SHA256 均为
`16cb3c99057c153a481026dc769d054d828225411bcf95c96afc4320705aedf7`；最终第四轮日志没有未定义引用、重复 label、overfull 或 underfull 警告。`git diff --check` 通过，未执行 commit/push。
用 `pypdfium2` 对新增内容所在的第 **5** 页和第 **16** 页进行渲染检查，ratio 分解、阈值等价性和定理起始位置均无截断、重叠或溢出。

### 2026-09-18：正文与附录数学环境覆盖核对

对 `tex/main/` 中所有 theorem/proposition/corollary/lemma 环境进行脚本化扫描，共得到 **31 个带 label 的正文与附录数学环境**。逐个以 label 搜索本审计报告的映射表，31/31 均命中；没有发现正文与附录数学环境缺少形式化映射的项目。该检查只证明每个结果都有对应的审计条目，不把映射条目本身当作 Lean 编译证据；Lean 编译和传递公理依赖仍以 `AxiomAudit.lean` 的构建结果为准。`tex/literature/` 下的引用论文环境不计入本论文命题覆盖。

### 验证记录溯源修正

此前维护误把最新的 906 计数写入了 PromptGradientMeasure 阶段的历史段落，现已恢复为 878。保存的 `formal/.lake/build_incremental_backup_20260918/full_check_prompt_measure.log` 第 527–528 行明确记录 878 和构建成功；GroupFiltering 阶段的 906 则由 `/volume/pt-train/users/rbliu/latex/lean_group_filtering_20260918.log` 第 528–529 行支持。ImplementationBridge 阶段的 909 来自后续定向构建及公理审计，不作为旧冷启动构建的证据。历史冷启动段落保留为当时记录，本次未重新执行冷启动构建，也未找到该次冷启动的独立日志；不据此宣称当前 909 定理版本已经冷启动验证。标签文本命中仅是覆盖索引检查，不证明自然语言命题与 Lean 类型的语义完全一致。


### 2026-09-18：性能证书的逐项语义核对与直接组合

本节核对 `tex/main/5-theory.tex` 中 `thm:promax-certificate` 的实际假设和公式，
不是仅搜索标签。读取了 `ProMaxCertificate.lean`、`ProMaxBatch.lean`、
`RLOOSurrogate.lean`、`PromptMixture.lean` 及论文引用的 Adaptive bound 前提。
此核对只覆盖这一条证书链，不能外推为全部 31 个数学环境均已完成语义验收。

| 论文对象或条件 | Lean 中的实际对应 | 核对结论与边界 |
|---|---|---|
| 有限 prompt、有限词表、固定 horizon | `Fintype X`、`Fintype A`、`Fin T → A` | 相同有限模型；Lean 的树表示不内置 EOS，padding 通过下列 `hpad` 条件接入。 |
| 每组响应数 n ≥ 2、独立组数 K ≥ 1 | 响应数为 `m + 1`，`hm : m ≠ 0`；组数为 Lean 参数 `n`，`hn : n ≠ 0` | 注意论文 n 对应 Lean m+1，论文 K 对应 Lean n，不能直接按变量名对应。 |
| prompt 后条件 iid 的完整组 | `blockMass μ p m T = μ(x) * iidMass (rolloutMass (p x) T)`，组间为 `iidMass law` | 不要求同组 RLOO 项独立；概率界需要 `hμ0` 与 `hμ` 保证 prompt law 非负且归一化。 |
| 实际 RLOO 信号及 Z | `groupAdvantage`、`groupSurrogateIncrement`、`group_increment_eq_token_sum` | baseline 排除自身；组求和与论文 Z 相同，非额外除以响应数的组平均。 |
| μ_L、κ 及 E Z = n S | `familyValue μ p L T`、`block_ratio_target`、`block_increment_expectation` | `familyValue` 对 L 给出单响应平均长度；不是整组长度。 |
| 每条响应有效长度至少 ℓ > 0 | `hl`、`hL`，导出 `block_length_lower_bound`、`family_length_positive` | Lean 在全部有限响应上要求长度下界；论文有效响应可用共同 EOS padding 及未采样路径上的长度扩展表示。 |
| 有限候选族预先固定，η_j > 0 | `J` 有限，`q : J → X → Policy A` 在样本外给定，`hη` | 允许选择器依赖整批数据；不能把由本批任意拟合的新策略自动代入固定候选族。 |
| 双策略支持、全历史 TV/KL、奖励有界 | `hs`、`hTV`、`hKL`、`hB`、`hR` | 与论文引用的 Adaptive 前提一致；这些是假设，不由 prefix gate 或训练日志自动保证。 |
| 同一 snapshot、gate、归一化信号、全局 token mean | `correctedBatchChange` 的共享 M,w,H；权重 `mask / Σ blockLength` | 允许 batch-dependent gate/normalization；不替代 response-mean 或动态 microbatch reduction。 |
| 三策略 ratio 与共同 EOS | `hw : r*w = responseTokenRatio`、`hw₀ : r₀*w = 1`、`hpad : mask*(ρ−1)=ρ−1` | 正概率支持上 ratio 分解成立；active 位置 mask=1，post-EOS 位置两策略共同确定性 padding 给出 ρ=1。这些是实例化条件，不是浮点 kernel 证明。 |
| 同时置信界及选择后的错误改进事件 | `rloo_uniform_estimation_bound`、`rloo_selected_false_improvement_bound` | 分母确为 K(nℓ)²η_j²；事件使用 ≥η，补事件给出严格 <η，再弱化到性能引理所需的 ≤η。 |

新增 `rloo_performance_from_corrected_objective`，在 Lean 内直接将
`corrected_block_objective_eq` 代入 `rloo_performance_from_estimate`，得到论文
`eq:promax-performance-certificate` 的校正目标形式。该组合定理不新增数学假设：
它显式携带两段已有结果需要的 ratio、padding、长度、支持、TV/KL 和估计事件前提。
组合定理本身是估计事件上的确定性结论，事件的概率仍由上述独立组定理控制。
论文在这次核对中未发现需要修改的公式；也未将该充分条件改写为训练单调改进保证。


验证结果：`LEAN_JOBS=3 bash formal/leanw build ProMaxCertificate AxiomAudit`
退出码为 **0**，`Build completed successfully.`；公理审计通过 **910 个项目 theorem 声明**，
仅允许 `propext`、`Classical.choice`、`Quot.sound`。相对上一阶段 909 增加了上述一条组合定理；
该计数包含辅助声明，不是论文定理数。此次为增量构建，重新检查了修改模块、受影响依赖及公理审计，
不作为冷启动验证证据。日志：`/volume/pt-train/users/rbliu/latex/lean_corrected_certificate_20260918.log`。
已有非阻断 linter 提示仍在；本轮不涉及训练代码或 TeX 修改，没有重新运行 CPU 训练测试或编译 PDF。


### 2026-09-18：前缀过滤五个数学环境的语义核对

逐项阅读 `4-method.tex` 的下列五个环境及对应 Lean 定义、假设和结论；
此处记录实际对应关系，不以标签命中代替语义核对。

| 论文环境 | 实际对应与实例化 | 核对范围 |
|---|---|---|
| `lem:prefix-stability` | `prefix_geomean_stability` 取 lo=1−epslow>0、hi=1+epshigh、θ=t/(t+1)；相邻前缀的对数平均递推给出该凸组合 | 要求旧 prefix 及新增 token 同时在界内；不是“被拒后永远拒绝”。Lean 的 θ∈[0,1] 泛化覆盖 t≥1。 |
| `thm:prefix-tighter` | `prefix_tighter_complete` 取 lo=−τ₋、hi=τ₊，Lean 索引 j 对应论文 t=j+1 | 早期偏离保留非零背景噪声，保证首 token 被拒及严格窗口 t<(z₁+ε)/(τ₊+ε)；下侧同理。晚期拒绝数 n−k.val 对应 T−t*+1；不保证 spike 本身必被 prefix 拒绝。持续单侧越界为严格不等式。 |
| 同一 theorem 的 spike 构造与末位置边界 | `upper_midpoint_spike_separation`、`lower_midpoint_spike_separation`、`final_prefix_eq_sequence` | 显式 spike 仅在 t*<T 时给出 prefix 拒绝而 sequence 接受；t*=T 两统计量相同。闭区间端点属于接受事件；不声称无条件 rejection-count 排序。 |
| `lem:prefix-density-ratio` | `pathMass_ratio` 给出 q路径质量=p路径质量×ratio乘积；论文在 p路径质量>0 时相除 | Lean 单向支持条件已足够，论文 mutual support 更强；长度 t 的路径对应论文 d_{t+1}，不能误写成 d_t。 |
| `lem:tv-prefix-ratio` | `acceptedPrefixTV_identity` 取 M≡1，`pathMass_normalized` 给出两边总质量均为1 | 有限分布 TV=half-L1，期望在 rollout p 下。共同零概率 token 允许存在，不要求词表每项严格正概率。 |
| `thm:pre-causal-mechanism` | `accepted_prefix_density_envelope`、`prefix_gate_residual_bound`、`prefix_gate_residual_le_envelope` | 接受前缀、正路径质量、t>0、τ₊,τ₋≥0；得到逐路径指数 envelope 及乘以接受质量的期望界。零质量路径通过 `value_mono_on_support_length` 排除，对概率结论无贡献。 |

五个环境没有发现假设不足或结论超出对应 Lean 结果的问题。论文的“on the event”按
rollout 随机前缀解释；路径密度比不用于 rollout 零质量路径。证明使用同一策略对 p,q，
实现中缓存的 old/rollout gate 若要控制 current/rollout，仍需要论文另列的 coupling 条件。
这里没有将 accepted-prefix 未归一化质量界解释为条件分布 TV，也没有证明 masked gradient 无偏。
本次未修改数学公式或训练代码；不增加 theorem 数量。

当前 910 定理版本的项目冷启动构建已经启动：构建前无正在运行的 Lean/Lake 进程；
原项目 `.lake/build` 被保留为 `.lake/build_before_cold_910_20260918T085407Z`，
从不存在的项目 build 目录开始。Mathlib/toolchain 依赖缓存保留，所以这是项目冷构建，
不是从零重编 Mathlib。启动时的 53 个 Lean 源文件和配置哈希、备份位置记录于
`/volume/pt-train/users/rbliu/latex/openrlhf_cold_910_20260918T085407Z.json`。
构建日志为 `/volume/pt-train/users/rbliu/latex/lean_cold_910_20260918.log`。
此启动记录本身不证明构建通过，最终状态应以日志末尾和进程退出码为准。


上述项目冷构建已完成，进程退出码 **0**，日志结尾为 `Build completed successfully.`；
AxiomAudit 实际输出 **910 个项目 theorem 声明通过**，只允许 `propext`、`Classical.choice`、
`Quot.sound`。构建结束后逐一比较启动清单：53 个 Lean 源文件及 lake/toolchain 配置的
SHA256 均未变化，53 个项目 `.olean` 均存在。验证结果补入同一 JSON 记录。
这是当前 910 定理版本的独立项目冷构建证据；不覆盖 Mathlib 从源码重建、浮点训练、
实际分布满足数学前提或全部论文表述的语义验收。既有非阻断 linter 警告仍保留。


### 2026-09-18：REINFORCE Max 三个归一化环境的语义核对

实际读取 `4-method.tex` 的三个环境、`AdaptiveMechanism.lean` 中的有限组定义和证明，
并对照 `_adaptive_token_normalization_single_group` 的有效 action mask、统计量和 safeguards。
同时检查 `main.tex` 摘要、`1-intro.tex` 贡献与 `7-conclusion.tex` 对这些结果的表述。

| 论文环境 | Lean 实例化及具体前提 | 与方法和实现的边界 |
|---|---|---|
| `prop:binary-rloo-normalization` | `finite_binary_rloo_normalization` 用 `Fin k ⊕ Fin m` 表示两类响应，k,m>0，hi>lo，wp/wn 取实际响应长度；两类长度总和为 P,N>0。`rloo_success_amplitude`、`rloo_failure_amplitude` 从真实奖励和得到 mΔ/(n−1) 与 −kΔ/(n−1)。`rloo_scale_ratio_mean_lengths` 与 `normalized_mass_balance` 给出因子比和总系数平衡。 | 二元指恰好两个奖励值，可不局限于 0/1；该理想结论不适用于 uniform-scale 同奖励组。P,N 是 token 总数，不是响应个数；平衡的是 scalar coefficient，不是完整 score-vector 梯度。 |
| `prop:normalization-scale-invariance` | `weighted_statistics_rescale` 保留非零质量、正负和随 c 缩放、平方和随 c² 缩放。`weighted_normalization_scale_invariant` 要求 c>0，同一组和同一权重；`mean_and_rloo_normalization_equal` 在组大小>1 时用 (n−1)/n>0 实例化。 | 论文要求两符号非空，使理想正因子有通常意义。Lean 对退化实数除法的代数扩展不是代码 fallback 语义。固定 eps、product cap、factor clamps 与分支阈值不随 c 同步缩放，不继承完整不变性。 |
| `cor:binary-token-standardization` | `two_level_token_zscore` 取 a=mΔ/(n−1)>0、b=kΔ/(n−1)>0；μ=(Pa−Nb)/(P+N)，v=[P(a−μ)²+N(−b−μ)²]/(P+N)>0，得到与理想归一化相同的 ±平方根系数。 | 分母是 P+N 的 population variance，不是 P+N−1 的样本方差；二元混合组没有零 raw advantage。对 multivalued reward 不能套用二元等价，代码排除零 advantage 的统计口径也不能改成所有 active tokens。 |

实现核对：helper 先用 `group_mask` 选择 active tokens，再分别统计严格正/负优势，
零优势不计入 `token_nonzero`；理想 κ 为正，而代码 `ratio=sum_pos/sum_neg` 为负，
`beta=-alpha*ratio` 与论文 β=ακ 的符号一致。实际分母包含 product cap 与 eps，
随后对两因子独立 clamp，且同号、小总和、非有限因子时返回原优势。
因此论文将精确幅度、零均值/单位方差和缩放不变性限定为 ideal normalization 是必要且一致的。
摘要、贡献和结论未将这三项理想结果扩大为实际 safeguarded 分支的严格恒等式。

本次没有发现上述三个数学环境需修正的公式。前两次与本次新增的逐项核对记录合计涉及
9 个不同带标签数学环境（性能证书1个、前缀相关5个、归一化3个）；这不是全部31个环境的验收。
没有修改 Lean/TeX/训练源码；复核当前 Lean 和配置哈希与 910 定理冷构建清单一致，
因此沿用那次编译与公理证据，未重复运行不受影响的构建或 CPU 测试。

### 2026-09-18：观测长度、RLOO 方差、有限覆盖与 logit 校准的语义核对

本轮逐项读取 `5-theory.tex` 中 `cor:promax-energy-certificate`、
`prop:rloo-group-variance`、`cor:covered-policy-certificate` 和
`cor:logit-calibration`，并对照 `SurrogateEnergy.lean`、`EmpiricalCertificate.lean`、
`GroupCertificate.lean`、`PolicyCover.lean`、`LogitBounds.lean` 的定义和定理。

| 论文结果 | Lean 对应 | 核对结论及限制 |
|---|---|---|
| 观测 token 数重缩放 | `observed_token_rescaling`、`corrected_objective_rescaling`、`rescaled_surrogate_unbiased` | `D/(Kn)` 乘全局长度归一化统计量严格等于独立组的 `sampleMean(blockSurrogate)`；其期望为 `familySurrogate`。要求样本长度分母非零；论文由每条响应至少 `ell>0` 给出该条件。 |
| additive ratio energy | `localRatioEnergy`、`trajectory_residual_second_moment`、`response_residual_energy` | 历史树上的 residual 是 martingale-difference 和，二阶矩为逐历史 local chi-square energy 之和；不假设 token 独立。共同 absorbing EOS 的相同 kernel energy 为零。 |
| energy 尾界 | `rescaled_surrogate_uniform_tail`、`rescaled_selected_false_improvement_bound` | 分母为 `K eta_j^2`（Lean 中外层组数为 `n`，论文记为 `K`），系数 `4 B^2 E_j`；来自 `|R|≤B` 从而 `|R-b|≤2B`。候选族必须评估前固定，selector 可以看完整评估数据。 |
| 共享 baseline 的精确方差 | `response_rloo_variance_exact`、`centered_rloo_variance` | 令 Lean `m+1` 对应论文每组响应数 `n`；精确式含 `(M-theta^2)/n + (sigma_R^2 E+theta^2)/(n(n-1))`，保留 sibling shared-baseline 相关项。 |
| prompt 方差分解 | `block_surrogate_variance_exact`、`group_surrogate_mse_exact` | `familyPromptVariance +` 条件 RLOO 方差是全方差分解；独立评估组均值再除以组数 `K`。prompt 异质性项不会因增加每 prompt 响应数而消失。 |
| 方差上界 | `family_rloo_variance_combined_bound`、`rlooVarianceFactor` | 同时保留粗的 `4B²E` 上界和 `tau² + (4/n+1/[n(n−1)])B²E` 上界，并取 `min`；`4/n+1/[n(n−1)]≤5/n` 已在 Lean 证明。 |
| 有限 cover 的 `3B zeta` | `response_residual_difference_of_token_cover`、`surrogate_deviation_transfer`、`covered_*` | 经验统计量差为 `≤2B zeta`，总体 surrogate 差为 `≤B zeta`，三角不等式给 `3B zeta`；覆盖条件必须对所有历史/响应成立，不能只在观测样本上检查。 |
| logit/positive tilt 校准 | `tilt_energy_bound`、`tilt_kl_bound`、`tilt_tv_bound`、`policy_softmax_local_bounds`、`family_softmax_certificate_inputs` | 正权重区间 `[lo,hi]` 且 `lo>0` 时，energy 与 reverse KL 均 `≤(hi-lo)^2/(4lo hi)`，TV `≤sqrt(C)/2`；softmax 通过相对 logit 的 context-dependent common shift 构造该 tilt。 |
| horizon 与支持边界 | `family_ratio_energy_horizon_bound`、`policy_tilt_local_bounds` | local 界要求在全部 `h.length<T` 的历史；零 rollout 概率 action 不需人为设最小 token 概率。实用时仍须证明整个规定历史范围内的 envelope，而非从 batch 最大 ratio 反推。 |

此次核对未发现这四组论文公式与 Lean 结论不一致。特别确认了：观测长度重缩放
不是把样本平均长度代入总体证书；RLOO 方差没有把共享 baseline 当作独立；有限 cover 的
`3B zeta` 同时包含经验和总体两项误差；logit 校准中的 `lo>0` 是归一化和 KL 证明的必要条件。
这些定理仍是有限词表/有限 horizon 的实数模型结果，未形式化浮点、GPU、Adam 或实际训练日志。

### 2026-09-18：附录 concentration / accepted-prefix / k3 链的语义核对

本轮逐项读取 `appendix.tex` 中 `lem:wt-second-moment`、
`cor:tv-prefix-subgaussian`、`thm:per-position-error-subgaussian`、
`cor:concentration-improvement`、`lem:tv-accepted-prefix`、
`cor:tv-accepted-subgaussian`、`cor:effective-horizon`、
`lem:k3-inlier-certificate` 和 `lem:anom-rate-implies-pre`，并对照
`PrefixConcentration.lean`、`TreePrefixGate.lean`、`K3Inlier.lean`。

| 论文环境 | Lean 对应 | 核对结论及边界 |
|---|---|---|
| `lem:wt-second-moment` | `trajectory_second_moment_of_mgf` | 由每个 rollout-reachable history 的 centered MGF 在 lambda=2 的条件界递归得到 `E_p[W_t²]≤exp(2 sigma²t)`；论文的 all-lambda 假设确实蕴含此条件。有限历史树、mutual support；不声称一般滤过概率空间。 |
| `cor:tv-prefix-subgaussian` | `prefixTV_of_mgf` 与 `trajectory_ratio_variance` | 用 `E_p[W_t]=1` 和 Cauchy-Schwarz 得 half-L1 TV 界，指数项为 `exp(2 sigma²t)-1`；位置 `d_{t+1}` 对应长度 t prefix。 |
| `thm:per-position-error-subgaussian`、`cor:concentration-improvement` | `surrogate_error_of_mgf`、`performance_lower_bound_of_mgf`、`performance_improves_of_mgf` | 使用 bounded reward 的 `|g_t|≤2Rmax` 和前缀 TV；论文的求和从 t=1 起用 prefix 长度 t−1，因此指数项 `exp(2 sigma²(t−1))-1` 一致。严格改进仍要求 raw population surrogate 大于 `B_T`。 |
| `lem:tv-accepted-prefix` | `acceptedPrefixTV_identity` | accepted measure 允许总质量小于 1，TV 明确定义为 half-L1；对应 rollout 期望 `E_p[M_t|W_t−1|]/2`，不是重新归一化后的条件分布 TV。 |
| `cor:tv-accepted-subgaussian` | `acceptedPrefixTV_of_mgf` | 任意 prefix-measurable `0≤M≤1`，不要求 M 与 ratio 独立；Cauchy-Schwarz 给 `sqrt(alpha_t)*sqrt(exp(2 sigma²t)-1)/2`。 |
| `cor:effective-horizon` | `effective_horizon_of_mgf` | Lean 用 `Fin N` 的 `t.val+1` 精确对应论文 t=1,…,T；Cauchy-Schwarz 汇总得到 `sqrt(sum alpha_t)*sqrt(sum exponential terms)/2`。不把 H_eff 当作归一化后策略 horizon。 |
| `lem:k3-inlier-certificate` | `k3_log_inlier_bound`、`logK3_interval_certificate` | 对正 ratio、非负上下 log 阈值和 `delta≤min(F(-tau_-),F(tau_+))`，点态 k3 inlier 推出 log-ratio 区间；不是由 conditional mean KL 或 batch 平均 k3 自动推出每个 token inlier。 |
| `lem:anom-rate-implies-pre` | `k3_abnormal_prefix_average_bound` 及 `abnormal_prefix_average_bound` | 需要每个 log-ratio 的全局上下界、`B=0` 的显式 inlier 条件和异常率 `≤xi`；得到前缀 log-average 的线性上下界。论文没有把“异常率小”单独当作充分条件。 |

核对中没有发现公式或索引错误。特别确认了 accepted-prefix 结论只控制未归一化
接受质量；MGF 定理只在有限树和 rollout 可达历史上使用；k3 结果是样本点态逆证书，
不能替代总体条件期望接口。截至本轮之前，已完成深度语义核对的带标签环境增至 22/31；本节之后的新增核对见下一节。


### 2026-09-18：剩余九个带标签环境的逐项语义核对

本轮继续读取 `3-preliminaries.tex`、`5-theory.tex` 和 `appendix.tex` 中剩余的九个 theorem/proposition/corollary 环境，并对照 `ProMaxObjective.lean`、`SurrogateBridge.lean`、`AdaptiveBound.lean`、`NormalizationSafeguards.lean`、`UniformScale.lean`、`PPODerivative.lean`、`BinaryUpdate.lean`、`BinaryLength.lean` 和 `BinaryAscent.lean` 的实际定义与定理。核对结果如下：

| 论文环境 | Lean 对应 | 核对结论及限制 |
|---|---|---|
| `prop:promax-objective-split` | `finite_promax_objective_decomposition`、`finite_promax_objective_error_bound`、`corrected_objective_delta` | 在同一批次、同一 gate/normalization/权重下，ProMax 目标精确分解为 raw surrogate、gate correction、normalization correction 和 clipping penalty；clipping penalty 非负所需的是 `a,M,w≥0`。不把 signed gate/normalization 项误当作单调改进。 |
| `prop:surrogate-equivalence` | `finite_surrogate_equivalence`、`finite_reward_ratio_surrogate_equivalence`、`finite_context_surrogate_equivalence`、`finite_surrogate_derivative_equivalence` | 有限 continuation、固定 rollout law、current policy 归一化及 rollout 支持包含条件下，trajectory reward-ratio surrogate 减去 rollout 常数等于 advantage surrogate。可预测 EOS mask 由 context 权重桥接；随机 baseline 需独立 baseline law，不能直接把随机 baseline 当作 Lean 中的固定标量。 |
| `thm:adaptive-bound` | `futureTV_le_linear`、`futureTV_le_pinsker`、`futureTV_le_adaptive`、`surrogate_error_adaptive_root`、`family_performance_lower_bound_adaptive` | 常数 `4R_max`、剩余 horizon `T-t` 和 `min(1,(T-t)ε,sqrt((T-t)δ/2))` 与 Lean 的 continuation recursion 一致。局部 TV/reverse-KL 前提覆盖全部 `h.length<T` 的历史；该结论不由 prefix gate 或 batch ratio 自动保证。 |
| `prop:normalization-safeguards` | `stabilized_closed_form`、`capped_stabilized_moment`、`factor_mean_residual_bound`、`factor_second_moment_residual_bound`、`positiveClamp_*` | 实现中的 additive epsilon 和 product cap 确实会改变理想单位二阶矩；factor clamp 后的均值及二阶矩残差界与论文一致，正 clamp 保留符号和同号排序。fallback 分支必须单独解释，不能套用 ideal normalization 恒等式。 |
| `prop:uniform-coefficient-sign` | `uniform_scale_token_coefficient_sign`、`ppoTerm_hasDerivAt`、`gated_ppo_hasFDerivAt`、`finite_gated_ppo_hasFDerivAt` | 保留 token 的局部 log-probability 系数符号与 `-r_i` 一致，但只在非 kink、未被 gate/clipping 去导的分支成立；不推出完整参数梯度的坐标级方向。 |
| `prop:uniform-population` | `binary_correction_expectation`、`zero_one_correction_factor`、Bernoulli score/finite policy identities in `UniformScale.lean` | 齐次 high/low 组的额外期望项给出 `λ_n(s)∇s`；对 0/1 reward 为 `s^{n-1}∇J/n`。结论限定在 on-policy、固定 prompt、全 gate 接受、无 clipping，并假定 uniformity test 恰好只选齐次组；近似齐次组或多 prompt 汇总不能直接使用该方向结论。 |
| `prop:binary-complete-update` | `implementation_factors_lower_bound`、`implementation_contrast_lower_bound`、`implementation_multiplier_lower_bound`、`implementation_policy_direction` | 在二元 reward 且长度只依赖 reward class 时，实际 epsilon/cap/clamp/fallback 和 token denominator 仍给出正的共同 multiplier，且下界为 `ε/L_max`。Lean 明确要求 `0<ε≤1`、`ε≤maxScale`、正长度、长度上界及两类正质量；这些是论文命题的实例化条件。 |
| `prop:binary-length-covariance` | `general_update_decomposition`、`covariance_error_abs_bound`、`response_update_decomposition` | 内容相关长度时，完整更新等于 class multiplier 主项加 within-class covariance error，误差界为 `sqrt(V_C n V_f)`；没有把随机长度误写成仍然精确沿 reward gradient。类内确定长度时 covariance 项归零。 |
| `cor:binary-local-improvement` | `response_local_improvement`、`class_length_local_improvement`、`population_local_improvement` | 任意内容相关长度需要 covariance margin 小于主项才保证局部真实奖励改进；class-dependent length 下，非零 success gradient 即足够。Lean 的局部改进结论还显式使用可微 policy、邻域非负/归一化和正类质量。 |

本轮没有发现上述九个环境的公式与其 Lean 对应结果不一致，也没有发现需要修改论文数学表达的实质性错误。特别保留了三个边界：Adaptive bound 的全历史条件、uniform-scale 的齐次组选取条件，以及任意内容相关长度下的 covariance margin。至此，31/31 个带标签环境都已完成逐项语义核对；这仍不等于一般概率空间、浮点/GPU/Adam/分布式执行或真实 LLM 实验已经形式化或验证。
### 2026-09-18：dual-clip 与标准 PPO 证明边界

实现的 `PolicyLoss` 还提供可选的 `dual_clip` 分支，但论文的
`finite_clipping_penalty_nonnegative` 只适用于标准 PPO 的
`min(surr1,surr2)`。本轮在 `ProMaxObjective.lean` 中加入
`dual_clip_negative_penalty_counterexample`：当 advantage 为 `-1`、
current/old ratio 为 `4`、上限 clip ratio 为 `6/5`、dual-clip 系数为 `3`
时，标准 PPO 的最大化贡献为 `-4`，dual-clip 贡献为 `-3`，因此相对于
未裁剪贡献的 deduction 为 `-1`，不再非负。Python contract 测试也覆盖了
该实例。故论文主定理明确限定 `dual_clip=None`，并将 KL、entropy 与 MoE
辅助损失排除在被证明的完整更新之外；启用这些分支需要额外的目标分解和
更新方向分析。

### 2026-09-18：实际 reduction 接口核对

`ppo_actor.py` 构造标准 `PolicyLoss` 时没有暴露独立的
`token_level_loss` CLI 参数；PPO 路径固定使用 active-token mean，只有
`policy_loss_type=gspo` 会强制切换为 sequence-level reduction。论文 §5
此前把 `token_level_loss=False` 描述成可直接选择的 response-mean PPO
配置，已改为说明：response-mean 只是形式化 reduction 对照，动态 replay
buffer 的 response-count weighting 不会在随机长度下自动重建全局 token
mean。该限定与 `BatchReduction.lean` 的反例及实际 `PolicyLoss` 调用一致。

同一轮还核对了 `compute_reward`：当主配置 `init_kl_coef=0` 时，
`kl_ctl.value` 使 KL reward 项严格为零；对 0/1 terminal reward，默认
`reward_clip_range=(-10,10)` 也不改变 reward。因而论文的 KL-free、稀疏
terminal-reward 分析与该限定配置一致；开启正 KL 系数、overlong penalty、
stop-properly penalty 或其它 reward shaping 后，需把新增 reward 项纳入
形式化目标，不能直接使用当前 sparse-reward 证明。
