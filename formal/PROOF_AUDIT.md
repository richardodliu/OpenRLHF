# 两策略版本的当前证明范围

更新：2026-09-25。以下是 rollout/current 修订后的当前状态；后面的历史记录仅对应旧版本，其中的三策略、cached mask、clipping 和旧正例数值不适用于当前稿。

- 唯一目标：rho=q/p，prefix mask × rho × sign-scaled advantage；没有第三策略和 PPO clip。
- 正文方法参照 DPPO/TRM 的 masked objective → masking criterion → error decomposition 组织，统一使用 mask。
- 目标分解 U-G+N；奖励比较包含 mask 随 q 变化的拒绝误差。总体定理保持原有 RLOO 条件独立性、支持、散度和 token-count reduction 条件。
- `TwoPolicyObjective.lean` 已检查点态残差及其绝对值界、精确目标分解、raw 下界；仅依赖标准 Lean 公理。
- 原来的有限树误差界与保留矩界继续作为通用两策略结果；旧三策略 clipped/fixed-mask 的总体封装不标作当前总体定理的完整机器证明。
- 正例与反例已重写；完整符号推导在论文内，`check_two_policy_examples.py` 的有理数枚举通过。新正例通用下界为 19/600。
- 本轮对应清单区分完整有限代数证据、带形式化组件的解析证明和带精确检查的解析例子。清单新鲜度不等于语义证明。
- 本轮变更不增加论文中的定理数量；原来的投稿判断不能自动替代对新版本的独立审稿。

## 历史记录（修订前，非当前验收结论）

# 核心数学与 Lean 对应

更新：2026-09-25。以下当前索引对应活动 ICLR 稿；后面的历史变更记录描述各自当时的状态，不作为当前清单或证明路线。

## 当前证明索引

唯一投稿入口为 `tex/iclr2027_submission.tex`。当前 `PAPER_COVERAGE.json` 包含 30 项显式数学环境（29 项有限范围条目、1 项支持假设），对应 36 个证据模块。数量包含定义、说明和例子，不代表独立贡献数。当前正文 9 页、全文 24 页。

| 当前核心环节 | Lean 入口 | 与稿件的对应 |
| --- | --- | --- |
| Max 的两系数及条件更新方向 | `ProMax.lean`、`NormalizationSafeguards.lean`、`BinaryUpdate.lean`、`BinaryAscent.lean` | 区分理想矩约束与数值保护后的实际系数；类内长度固定与协方差条件分别处理 |
| Pro 保留系数矩 | `AutoregressiveTree.retained_moment_lower_reentry_mass_bound` | `E[(Mw)^2] <= C E[Mw] + R^-`；下侧重入单列，第一矩为重要性加权保留量 |
| 实际目标与 raw surrogate 的逐组比较 | `Objective.raw_increment_ge_scaled_objective_increment` | 对整个总体期望固定的正尺度；保持实际 H、gate 和 clipping；包含参考策略的 clipping penalty |
| RLOO 总体期望 | `AutoregressiveTree.group_surrogate_increment_expectation` | 同 prompt 的 iid sibling；先乘回 token 分母，再消去独立 baseline |
| 奖励参考误差 | `AutoregressiveTree.family_reward_error_bound` | 同一误差分别由 Adaptive 与 Mixed 控制，再取最小值；前者使用条件 continuation TV，后者使用边缘 prefix TV |
| 实际 EOS 目标到奖励下界 | `AutoregressiveTree.population_performance_from_scaled_eos_objective` | 从 rollout/snapshot/candidate 比率、支持、随机 L、EOS 掩码导出点态比较，再接总体奖励界 |
| 多组 token 归约 | `AutoregressiveTree.restored_batch_margin_expectation` | 任意逐组贡献 F 均可代入，包括除以固定参考尺度的 margin；专用 `eos_batch_margin_expectation` 是单位尺度实例 |

表中的命名空间均位于 `REINFORCEProMax` 下。当前数学链不使用独立评估采样、有限候选选择、置信事件或有限样本奖励证书。未将旧形式化依赖删除，并不表示旧结论仍在投稿稿件中。

2026-09-25 核心前提复核：正文第 2 节给出 rollout/candidate mutual support、snapshot 在 rollout support 上正概率、共同吸收 EOS、标准 PPO 和固定有限 prompt law。总体定理使用同组缓存的 H/M；lambda 在整个总体期望外固定。Lean 的 `m+1` 对应稿件组大小 n；`hm : m != 0` 对应 n>=2。Lean 只要求 gate 非负，因此覆盖实际二值 gate；不要求 H 与组样本独立。一般 prompt 权重定理只用非负性，稿件的归一化 prompt law 是其特例。未发现这些核心陈述与其编码之间新增的前提缺口。

最新完整公理审计为 `uniform_mismatch_integration/build.log`（1309 项项目定理），构建退出 0，仅允许 propext、Classical.choice、Quot.sound；此前记录为 `retained_mass_bound/audit2.log`。Pro 组织修改后的 PDF 记录见 `pro_bound_narrative/validation.json`。前一轮直接运行 Lean 检查 EOSCertificateBridge.lean 与 PrefixGateMoments.lean，均退出 0。源码检查记录位于 `/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/core_chain_review/`。清单检查只验证新鲜度；自然语言到形式定义的对应仍须人工审阅。机器证明不证明研究新颖性或 ICLR 接收。

## 现有正例的通用奖励下界（2026-09-25）

SelectiveRolloutBound.actual_population_margin 由实际 safeguards、prefixLogGate 和标准 PPO 计算 rollout 基准的 margin=3/200。support_and_kl 构造实际两状态策略族的 KL cap；positive_actual_reward_bound 调用 surrogate_error_adaptive_root，使用 root TV=1/200、local TV<=1/30 和 Adaptive cost<=1/1500，得到 43/3000 的正奖励下界。没有把奖励改善作为假设；一维总体梯度步仍由 SelectiveRewardExample 原结果支撑。

SelectiveRolloutBound 与 AxiomAudit 构建成功，仅标准公理；当前 30 个显式数学环境、36 个证据模块。仅 ex:selective-positive-margin 陈述 hash 变化，其余 29 项保持。正文/附录的三步证明标题为叙事调整，公式与原证明前提保持。日志与验证见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/rollout_bound_integration/validation.json。

## Max 核心实现复核与证明复用（2026-09-25）

核对 `_adaptive_token_normalization_single_group`：同号/小总量 fallback 对应系数 1；常规分支使用带 additive epsilon 的平方和、10^8 product cap、正 factor clamp。二元全同组的 raw RLOO 为零，optional uniform 分支只在全成功组添加 1/n 信号，全失败组仍为零；实际容差的适用条件由 `lem:uniform-eligibility` 给出。单组系数 H_i/L 与共享批次的 psi_K(L)H_i 分别对应 `BinaryUpdate`、`BinaryLength` 与 `BatchDenominator`，未将二者混同。

`BinaryAscent.response_local_improvement` 现直接由 `population_local_improvement` 实例化：先沿用实际 normalizer 的 class-multiplier 下界，再由范数平方非负把保守 clamp margin 推为实际 multiplier margin。删除该特例内重复的方向导数和局部可微性推导，9 个 theorem 的签名全部不变，论文结论没有增加或削弱。验证记录见 `/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/max_proof_reuse/validation.json`。

归约术语修复：当 Z 与 L 为组贡献和组 token 数时，E[Z/L] 是组内 token 均值再平均；此前附录的 response means 表述不准确，现已纠正。原等式直接对应 BatchReduction.ratio_expectation_gap，公式、前提、Lean 源码均未变化。该次技术来源核对与 PDF 记录见 bound_attribution/validation.json。

Pro 误差界叙事调整：目标分解移到 mask 定义之后，条件序列 TV 的既有推导移入正文主线；thm:prefix-tighter 的陈述与完整证明逐字移至附录 C.3。全部 30 项陈述哈希、78 个受检 Lean/入口/模板文件哈希保持，清单更新文件与行号。新增正文中的二元 mask 残差等式是附录 D 已有分情况说明的公式化，不新增定理或前提。记录见 pro_bound_narrative/validation.json。

## 条件序列 TV 行间公式核验（2026-09-25）

正文 5.1–5.2 的三处行间式已用既有 Lean 结论直接实例化并通过 kernel 检查：同一历史 h 下条件 continuation ratio 的绝对偏差期望等于两倍 futureTV；futureTV 不超过从 h 出发、rollout 加权的后续局部 TV 之和；固定正 scale 与二值 mask 的缩放/丢弃残差精确拆分。检查文件为仓库外 conditional_tv_lean_check/InlineChecks.lean，未新增论文或项目定理。前两式只需既有支持条件（TV 求和式本身无需比率支持条件），没有引入 token 独立性或用边缘 TV 替换条件 TV。

运行 `bash formal/leanw build AdaptiveBound EOSCertificateBridge PrefixGateMoments AxiomAudit` 成功；未变模块及 1269 项公理审计为缓存复用。随后直接编译 InlineChecks.lean 成功，并重新输出六个关键结果的公理依赖，仅含 propext、Classical.choice、Quot.sound。首轮检查文件因用 Lean 保留字符 lambda 作变量名而解析失败，改名 scale 后通过；这是检查脚本语法修正，论文数学及项目 Lean 源码未修改。现有显式环境清单保持 30 项；正文与 PDF 哈希保持，24 页、正文第 9 页底部。完整记录见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/conditional_tv_lean_check/validation.json。只有源式或依赖变化时才需要重跑这些固定检查。

## 当前 CPU 接口回归（2026-09-25）

伪代码与实际 PolicyLoss 对照：Max 每 prompt 组缩放；Pro 以 cached old/rollout 构造前缀 mask，保留 token 权重；拒绝 token 仍保留在全 active-token 分母。源码中一条把 correction 写成 rollout/old 的注释已纠正，Python AST 不变。复用固定测试，loss、梯度、detach、诊断量和两种 reduction 共 18 项通过；stabilizer、product cap、factor clamp 和 fallback 共 10 项通过。首次归一化测试因缺少 pylatexenc 在收集时中断，使用现有隔离审计依赖目录后通过；没有安装全局依赖。测试全部在 CPU 执行，没有新增用例或运行 GPU 实验。记录见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/method_contract/validation.json。这些固定数值检查不替代 Lean，也不验证完整 Ray/vLLM 训练。

## 历史变更记录

以下保留既往变更及验证证据。诸如“当前”“本轮”和环境数量均指记录产生时的版本，可能已被上面的当前索引替代。

## 当前入口与清单

唯一 ICLR 投稿入口使用 tex/main/paper-body.tex 和 tex/main/core-appendix.tex。PAPER_COVERAGE.json 共 34 个显式环境：33 个有限范围条目和 1 个支持假设。包括定义、remark 和 example；并非 34 个定理或独立贡献。清单新鲜度不等于自动验证自然语言语义。

## 本轮搬移的语义核验

- 从原 75 个环境选取支撑算法主线的核心环境。按文本哈希对应原记录，搬移后更新文件、行号、入口集合及证据索引；标签保留，闭合全部活动引用。
- thm:adaptive-bound 仅移除“同一界还控制 trajectory remainder”的额外一句，保留实际使用的 reward/surrogate 不等式、有限模型与所有前提。原 AdaptiveBound 证明覆盖保留结果。
- rem:rloo-variance 将已失效的“下文 dominated-density 构造”说明替换成当前有限模型的微分/归一化有限和条件；四条结论及正则前提保留。简洁证明分别使用独立 baseline、有限策略梯度、确定性 mean/RLOO 比例及独立 sibling 二阶矩展开，对应原有 PolicyGradient、PromptGradient、RLOOMoments 等模块。
- 从旧 uniform-population 命题继承的二元更新模型前提已显式放回核心更新段：有限可微策略、邻域内非负归一化、固定奖励、正概率类别、score 及零质量约定、detached 系数；不因移除可选分支分析而丢失上下文。
- prefix 条件早晚比较保留原完整证明；Max 的二元更新、协方差、跨 prompt 与随机分母从旧 Uniform Scale 大节移入 Max 主证明部分。
- 组合目标恒等式、完整组采样证明、正 margin 例和必要共享参数反例保留。删去非必要的 cover、logit、energy 与精确组方差扩展，不削弱保留命题的前提。
- 伪代码依据实际归一化函数调整保护顺序：先排除退化/小信号组，计算 capped/stabilized 因子，检查有限性，然后 clamp 和缩放；uniform-scale 必须启用。训练代码未改。

## 奖励证书的使用边界

正文已明确：置信水平使用总体残差二阶矩 v_j，奖励下界还使用总体平均长度与全上下文 Adaptive 界。经验替代值不自动继承定理保证；实际应用须取得这些量或有效保守界，若从数据估计则须另行控制误差。训练批次的目标上升本身不是该证书。此次仅澄清现有定理的使用条件，37 项数学环境的文本哈希和 Lean 源码均未改变。

## 实现对应复核

实际归一化函数先检查符号类别和小总量，再计算 capped/epsilon 因子，检查有限性，最后 clamp 与缩放；与正文伪代码一致。优势计算位于 no_grad 路径，Pro 系数显式 detach。标准 PPO 调用 masked_mean(..., dim=None)，分母为所有 active tokens，拒绝的 token 不从分母中删除。动态批次另乘 replay 权重，不能直接视为全局 token mean；正文已保留这一限制。

覆盖检查新增追踪 openrlhf/models/utils.py（实际 masked_mean 定义）和 tex/iclr2027 下的公式宏文件，避免分母或符号定义变化后清单仍显示通过。隔离临时目录中的变更检测验证通过；没有改动训练计算或论文公式。

## 核心更新方向与正文层次

复核 BinaryUpdate.lean 的 implementationFactors、implementationCoefficient、implementation_policy_direction：实际符号因子由 length-weighted RLOO 统计构造，经 safeguard 下界、coordinate_expectation 和有限策略导数恒等式推出共同正乘子，未把目标方向恒等式直接设为前提。该结论仍要求类别内固定长度；任意内容相关长度使用后续 covariance 条件。

正文重复的 Prefix stability 引理改为短说明，附录 lem:prefix-stability 及 Lean 证据完整保留。当前显式环境因此为 36 项；其余陈述文本哈希均不变。这是删除重复陈述，不是移除证明或增加定理。

## 前缀筛选证明压缩

thm:prefix-tighter 只保留正文使用的早期偏离拒绝窗口与晚期偏离后缀局部性。删去单峰 midpoint 分离构造、末位置无法严格分离的说明和持续单侧越界的比较，不再将这些额外性质列入活动证明范围。题名改为 Conditional prefix rejection patterns，并在陈述中定义 tau 阈值。核对 PrefixMasking.lean 的 early_deviation_complete、late_deviation_complete：保留双侧窗口、正分母、严格边界及后缀计数，cached old/rollout 比率不变。相邻稳定性引理只保留一行凸组合恒等式及区间论证，陈述未变。没有增加数学结论，Lean 源码未改，其余 33 项环境哈希不变。

## Max 定义与基础代数证明压缩

def:adaptive-norm 合并原先重复展开的符号定义、token moment 约束、等价系数方程与闭式解。非零 active-token 集合、正负符号缩放、两个经验约束和唯一正解完全保留，eq:alpha-beta 引用不变。核对 finite_scaled_moments、adaptive_closed_form 与 adaptive_closed_form_unique；没有把 ideal 解当成 safeguards 后的实际解。仅该定义块哈希改变，其余 33 项陈述与 Lean 源码未变。

将二元 RLOO 闭式系数、尺度不变性与 token 标准化等价性的证明缩写为核心代数步骤；保留 P/N、a+/a- 的正性、总量与平方和、基线缩放恒等式及二点分布方差。保留稀有成功并不意味着负因子小于一的具体计算、多值奖励符号差异和 epsilon 共同比例收缩，去掉重复的 Lean 模块介绍与陈述重述。实际 safeguards、条件更新方向及协方差证明未改。36 项显式数学陈述及 Lean 源码哈希不变。

## RLOO 的最小基础性质

rem:rloo-variance 保留原始 RLOO 的条件无偏性与 mean-baseline 的逐组缩放关系；删除未被当前方向结论或奖励证书使用的 baseline 方差和 score 外积二阶矩展开。独立 sibling baseline 的作用直接放入证明。保留实向量参数、固定有限正支持、邻域内可微且归一化概率、参数无关奖励和 iid 条件。PolicyGradient.lean 的 finite_iid_rloo_policy_gradient 证明有限分布的导数等式；RLOOScaling.lean 的 raw_mean_estimator_scaling 按坐标给出缩放恒等式。其余 33 项陈述与 Lean 源码未变。不能将原始 estimator 的无偏性转用于归一化、gate、clipping 或随机 token 分母。

## Pro 筛选比率记号统一

附录的 prefix 定义不再引用 current/rollout 的 eq:log-ratio，而直接使用 w=pi_old/pi_rollout；prefix 和 sequence proxy 的显示符号同步为 w_pre/w_seq。稳定性和条件筛选定理统一使用该 w，支持条件显式指定 rollout/snapshot。原定义依靠“selected policy pair”的文字替换说明，不够直观；此次消除该重载，不改变 gate、阈值、证明或训练代码。PrefixMasking 的通用正比率结果及 PaddedPrefix 的 active-index 对应仍直接适用。重新审阅三个文本变动的陈述和共享宏影响，Lean 文件未改。

## 算法阈值接口复核

PPO clip 使用 eps_clip_low_high，Pro gate 使用独立的 vllm_is_truncated_threshold，且后者要求 enable_vllm_is_correction。正文配置明确启用校正，伪代码分别列出两个区间，并定义 clip(r) 的端点。finite_promax_objective_decomposition、finite_clipping_penalty_nonnegative 和 reward_gain_ge_objective_margin 的 clipping 值 c 与 gate M 独立，现有 Lean 证明直接覆盖该说明，不需要新定理。34 项显式陈述哈希、Lean 文件及训练源码均未改变。

## 可选分支的最小前提

lem:uniform-eligibility 的三个使用点均为 0/1 奖励。仅保留 n>=2、正容差及 n*tolerance^2<=1 下的全同组等价条件和实施容差特例，删除未被使用的任意奖励间隔扩展与 sharpness 构造。证明保留样本方差恒等式、混合组下界及严格不等号的边界处理。对应现有 uniform_test_iff_homogeneous 在 c=0,d=1 的实例与 zero_one_implementation_tolerance；Lean 源码不变，其余 33 项陈述哈希不变。

## 奖励证书的重复推导合并

删除 app:rloo-reduction 前置说明中重复的 RLOO 期望推导、随机分母 MSE/tail 公式和未被引用的两响应算例。thm:promax-certificate 后的完整证明已明确给出 E[Z]=n*S、比率与 EOS 对应、独立组残差二阶矩、Markov 与有限 union bound，以及奖励下界连接，因此没有删除这些依赖。正文保留 response-mean 与 global-token 的差异公式，附录保留其短推导说明及 ratio_expectation_gap 对应。34 项显式陈述、完整证书证明和 Lean 源码均未改。

## 证明依赖与验证

独立成功执行的 Lean 构建与公理审计记录为 `/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260924/current_core_lean.log`、current_core_axioms.log 和 current_core_verification.json。审计的 1243 项项目 theorem 只依赖 propext、Classical.choice、Quot.sound；这是底层辅助声明数量，不是论文贡献数。

精简后重新核对：当前 34 项显式环境（33 项有限范围、1 项支持假设）引用 35 个证据模块，全部在 AxiomAudit 的 71 个项目模块导入闭包中。所有 71 个 Lean 源码指纹与独立成功审计时一致；当前 PDF 和 PAPER_COVERAGE.json 指纹也与最新编译记录一致。证据见同目录 minimal_core_integrity.json。复用未变源码的既有审计，不将旧论文指纹或日志重放称为本轮重新证明。

当前逐项证据以 PAPER_COVERAGE.json 为准。论文构建与页面记录见 submission_wording_validation.json、submission_wording.log。清单只检查显式环境与依赖新鲜度，自然语言到形式定义的对应需要语义审阅；本文不声称全篇 TeX 被自动翻译验证。浮点执行、任意优化器行为、实证优越性和会议接收不在这次形式证明的范围内。

## 九页正文的内容搬移

prop:normalization-safeguards、ex:selective-positive-margin、prop:coupled-gate-failure 移至正文，陈述文本哈希保持不变。奖励证书主证明和二元更新条件因子化解释移至正文，Lean 标识符对应与 EOS 细节留在附录；删去重复概述。新增评估协议段落仅解释原定理前提。34 项陈述与全部 Lean 源码不变；清单按标签/原陈述哈希重新定位，未依靠会随搬移变化的自动序号继承证据。

## 历史记录

本轮前的论文、PDF、清单和审计记录：`/volume/pt-train/users/rbliu/workspace_backups/OpenRLHF/20260924T133539Z-single-core-paper/before.tar.gz`。更早完整证明审计在上一轮归档中，见该归档中的旧 PROOF_AUDIT.md。移出论文的扩展不是被证伪；仅不再承担当前算法主张。

后续仅对核心结论的真实缺口补证。精确实数有限模型、条件有效性、浮点执行、优化器行为和实证性能仍须区分。

## 投稿措辞清理（2026-09-24）

34 项显式环境保留；5 项标题或解释文字修改已逐项审阅：def:max-div 限定于当前有限模型，def:prefix-is 删除内部实现/验证标识，def:prefix-trm-proxy 标题改为 Causal Prefix Mask，thm:promax-certificate 将 core mode 改为明确目标配置，ex:selective-positive-margin 用构造条件取代重复免责声明。公式和数学结论不变，Lean 源码不变。论文中的证明路径与机器定理名从投稿文字移除，对应证据仍保存在 PAPER_COVERAGE.json。摘要和引言明确正例使用精确奖励恒等式，未将其表述为通用 Adaptive 余项界的非空性证明。

2026-09-25 投稿措辞复核：prop:normalization-safeguards 明确矩的对象为缩放优势；prop:binary-length-covariance、prop:batch-denominator、thm:prefix-tighter 将否定式限定改为直接数学说明。四项公式与结论不变，Lean 源码哈希未变，清单重新核对。最新构建与页面证据位于 `/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/submission_wording_validation.json`。

## 方法一致性的记号修复（2026-09-25）

二元更新正文证明中的 z 改为 sibling responses，H_i(a,z) 明确定义为当前响应标签 a 下的组 token 归约系数；这与 BinaryUpdate.coordinate_expectation 的响应乘积质量及标签映射一致。共享分母段落明确 H_i(B) 为未归约系数，避免与前文已经除以组长度的 H_i 混用；正文与 prop:batch-denominator 定义 factor-clamp 区间 [eta,s_max]，消除上限符号未定义。未增加定理，算法与 Lean 源码未改。

本轮直接执行 `bash formal/leanw env lean BinaryUpdate.lean` 和 `bash formal/leanw env lean BatchDenominator.lean`，均以退出码 0 完成；存在原有 unused-variable/tactic 风格警告，无证明错误。日志为 `/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/core_BinaryUpdate.log` 和 core_BatchDenominator.log。最新 PDF 和边界记录为同目录 core_alignment_validation.json；正文在第 9 页底部结束，总计 23 页。

本轮公理审计 `bash formal/leanw env lean AxiomAudit.lean` 已以退出码 0 完成：1243 项项目 theorem 的公理依赖仅包含 propext、Classical.choice、Quot.sound。此命令重新执行审计并检查导入证明的依赖，不等同于从源码重编译全部 71 个模块；本轮重新 elaboration 的两个相关模块如上。

## 组合奖励保证的闭合核查（2026-09-25）

本轮直接审阅源代码核对，而非再次运行未变证明。参数对应：正文 K 个评估组对应 Lean n，每组 n 个响应对应 Lean m+1，正文 mu_L 对应 familyValue mu p L T；Lean 的 B 是奖励绝对值上限，正文 B_j 对应 familyAdaptiveCost。核查链为：

1. sampled_certificate_ratio_padding 从正质量完整组与 EOS 假设推出三策略比率/补齐恒等式；unsupported_sample_mass_zero 排除零质量字符串。
2. corrected_block_objective_eq 将同一数据、gate、优势与全局 token 分母的校正目标增量化为实际 RLOO 统计量。
3. block_increment_expectation 与 block_ratio_target 从独立 sibling baseline 推出总体均值与长度比例目标。
4. rloo_uniform_estimation_bound 对固定候选集合给出完整组 iid 下的同时误差事件。
5. family_performance_from_block_target 使用已证 Adaptive 奖励界，rloo_performance_from_corrected_objective 将目标恒等式代入。
6. raw_increment_ge_objective_increment 与 reward_gain_ge_objective_margin 在同一事件上扣除 D_j 和参考 clipping 成本；末步 hcert 是前述奖励界的接口，不是新增的奖励改善假设。
7. rloo_selected_false_improvement_bound 允许在固定集合内依赖评估数据选择候选；不要求候选估计之间独立。

代码路径再次核对：cached ratio 为 old/rollout，gate 为前缀几何均值，保留系数为 token ratio；标准 PPO 的 token-level loss 用所有 active tokens 归约。动态微批次仍须遵循正文声明的全局归约条件。这些条件已在现稿出现，本轮没有修改论文、算法或 Lean 源码，也没有新增定理。逐步源文件指纹见 `/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/core_reward_chain_review.json`。

2026-09-25 参考工具约定：Adaptive 证明开头明确引用 TRM 的 token-shift/continuation 分解，以及 TV 次可加、KL 链式法则、Pinsker。仅补来源说明，34 项陈述和 Lean 源码未改。总页数约束为用户指定的 30 页，构建入口检查总页数和正文第 9 页；真实 PDF 为 23 页，正文页底几何与内容边界通过复核。模拟边界检查确认 30 页通过、31 页拒绝、正文第 8 页结束拒绝。


## EOS 下 Pro 系数矩与实现的连接（2026-09-25）

正文和 thm:prefix-reentry-moment 证明明确：公式中的 M_t 和重入项取完整 padded tree 的 gate。EOS 之后该分析 gate 可能与实现按 active count 冻结的原始 gate 不同，但这些位置的实际损失系数为零。EOSGateDrift.eos_effective_gate_implementation 已证明 active-mask 乘积相同；二元 mask 及 value_mono 给出实际 active 系数无条件平方矩不超过完整树的平方矩。因此直接沿用既有 PrefixGateMoments.retained_moment_reentry_bound；不把完整树重入项当成仅 active 位置的统计量，也不把无条件界当成条件于存活的界。

匿名组合检查保存在 `/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/eos_moment_bridge/Check.lean`，通过 `bash formal/leanw env lean` 检查，退出 0。该检查复用已有 EOS 恒等式与期望单调性，未添加项目 Lean 声明；对应两条已有定理的公理依赖仅为 propext、Classical.choice、Quot.sound。论文 34 项显式陈述哈希不变，只修正正文/证明中的适用解释并更新清单备注。最新构建和 PDF 几何证据在同目录 build.log、validation.json；全文 23 页、正文第 9 页末尾结束。


## 算法伪代码的结构化表达（2026-09-25）

将 alg:promax 中叙述式步骤改为显式赋值和 if/else。对照 experience_maker.py：默认 H 为 masked raw RLOO；uniform-scale 优先覆盖并绕过归一化；正负 token 质量的最小值达到 epsilon 才计算 capped/stabilized 因子，有限性检查通过后才 clamp 并缩放，否则保持 H。空符号组的质量为零，因此同一条件覆盖退化组回退。对照 loss.py：old/rollout 的 detached token 权重、按 active count 的前缀几何均值 gate、current/old PPO ratio、标准 min surrogate 与包含所有 active tokens 的分母均在伪代码中显式写出。仅改变论文表达，训练代码、数学陈述和 Lean 源码未改；未重跑未变证明。


### 2026-09-25：参考技术归属与核心伪代码

- 用户界定：Max 核心是正负 advantage 的不同正缩放系数；Pro 核心是 causal prefix mask/filter。伪代码由 22 个实现步骤改为 8 个核心步骤；输入已有 responses/RLOO advantages/masks，MaxScales 使用原系数及保护公式，fallback=(1,1)。H、cached w、M detached；loss 仍用包含被拒 token 的 active-token 分母。optional uniform-scale 单独引用原分支说明。训练源码及 Lean 定义未改。
- Adaptive TeX 证明用 DPPO B.1 乘积恒等式，给定完整当前 prefix 后用 tower property，按 TRM D 的三种 continuation TV 界收束。无当前 action/continuation 独立性假设；padding ratio=1；RLOO baseline 通过已有 surrogate 等价性消除。与现有 AdaptiveBound 的 finite-tree 递归证明等价，经人工复核。
- Pro 将既有连续接受 envelope + conditional normalization 的短推导移到正文，附录保留 horizon/EOS 细节。引用 Seq-MIS 的 envelope 思路，而不错误归属本文的 w²≤Cw 条件归一化步骤；Seq-MIS 原文实际展示 C² 点态界。
- Max 的局部方向分析引用 SGA 一阶 progress 的解释，不将系数矩当作完整随机梯度方差，也不引入新的 SGA 收敛定理。Theory 的三篇文章按具体页面署名、日期与技术博客类型引用。
- 34 项 statement 哈希均未改变；位置和 dependency hash 已刷新。重新执行 AdaptiveBound.lean、PrefixGateMoments.lean 源文件检查均退出 0；其他未变证明复用已有审计，不声称本轮重编译全部模块。
- 证据：/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/reference_techniques/validation.json 与 build6.log。PDF 25 页，正文第 9 页底部结束，第 10 页 AI USE STATEMENT；未改模板、字号、间距或分页规则。


### 2026-09-25：移除独立评估支线，改为方法核心的总体奖励条件

用户再次界定 Max 为正负 advantage 两系数缩放、Pro 为 causal prefix mask/filter，并明确独立评估采样不重要。论文移除有限候选、置信事件、独立评估与对应统计证书；既有 Lean 依赖保留，不能将其数量当作当前论文成果。

- 当前 30 项显式环境，28 项原文不变。数值保护残差移附录；二元缩放、长度协方差和因果拒绝模式移至正文并删除原处重复。只替换一项总体奖励结论并简化一个正例；未增加方法组件。
- 新总体 margin 为 E[L(Delta S_PM-D-C(p))]/n。逐组恒等式与 raw_increment_ge_objective_increment 给 L(Delta S_PM-D-C(p))<=Z；具体 iid RLOO law 给 E Z=n*S_p(q)，再应用 Adaptive bound。随机分母 L 保留在期望内部，不等同于用 E L 代替 L，也不保证未加权的训练 loss 增量足够。
- ProMaxCertificate.population_performance_from_objective_margin 已从源文件编译通过。Lean m+1 对应论文 n；其中点态 F<=blockIncrement 由上述 objective 结果及支持/EOS 对应实例化。正质量组上使用实际 margin，零质量支持外组可用 blockIncrement 延拓 F，不改变期望。该编码组合关系已人工复核，不声称自动验证 TeX 自然语言。
- 正例保留 E Z=delta、实际逐组 objective/distortion 公式、mixed-scale 正界与 exact reward gain=delta/2；删除 finite-sample tail、K>=960 和置信水平。D 比较 snapshot，r0=1。额外匿名组合检查 core_focus/PopulationExample.lean 退出 0，验证 E(Delta S-D)=(2s-1)delta/4>0。
- bash formal/leanw build ProMaxCertificate AxiomAudit 成功；改动模块及受影响下游重新编译，未变模块缓存复用。项目公理审计覆盖 1244 项 theorem，仅允许 propext、Classical.choice、Quot.sound。该计数不是论文定理数或新贡献数。
- 30 项清单中 29 项 reviewed_finite_scope、1 项 reviewed_assumption_scope；证据来自 33 个 Lean 模块。28 项不变陈述按哈希继承审阅；新总体结论与修改正例单独更新 evidence/review_note。清单检查只验证新鲜度。
- 构建、引用、diff 检查通过。PDF 22 页，正文第 9 页 y=732.269 结束，第 10 页 AI USE STATEMENT；已查看核心伪代码与第 9 页渲染。模板、字号、间距和分页规则未改，训练代码未改。
- 当前完整证据与改写前备份：/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/core_focus/；最终 build4.log、lean_build.log、validation.json、statement_review.json。当前科学结论是条件有效性，未据此认定原创成果强度与 TRM/DPPO 相同或达到会议接收保证。


### 2026-09-25：总体奖励条件的实际目标连接

上一轮完成四目录参考范围与技术对应整理，属于文档状态推进；本轮核查发现总体 Lean 定理的抽象 F 点态前提仍由人工实例化，因而将这一连接纳入内核检查。

- EOSCertificateBridge.eosGroupObjectiveMargin 显式定义 L*(Delta objective-D-reference clip)，包含同组 RLOO、实际 active-token 分母、三策略比率、gate 和缩放。population_performance_from_eos_objective 的参数不再要求调用者假设 F<=blockIncrement；证明内部从实际 objective 不等式导出它。
- 正质量组上的每个 response/token 都由模型概率推出正支持；EOS padding 残差为零；L>0 由正 horizon 和首 token active 推出。有限和处理零质量字符串，证明延拓不改变期望。随后复用具体 RLOO 期望与 DPPO/TRM 对应的 Adaptive bound。
- 固定 p/q/old 对总体组采样取期望；M/H 可依赖整个响应组。正文相应写明 fixed candidate，未添加独立评估或候选选择步骤。标准 clipping 作为 c/c0 代入，证明对任意这些数值成立，故不额外假设 clipping 恒等式。
- 正例的总体期望和正性从此前外部匿名检查迁入 SelectiveRewardExample.population_objective_margin；仍复用原有 group_moments 和 mixed_scale_bounds。没有新增论文定理或方法组件。
- 当前 30 项显式陈述哈希全部保持不变。清单新增直接连接定理和正例的持久化证据；33 个证据模块，29 项有限范围结果/定义/说明和 1 项支持假设。清单检查不代替语义对应审阅。
- 本轮 EOSCertificateBridge 源文件检查退出 0；论文构建退出 0，22 页，正文第 9 页底部 y=732.269，第 10 页从 AI USE STATEMENT 开始。已查看第 9 页渲染。模板、间距、字号、分页规则和训练源码未改。
- 证据目录：/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/population_bridge/；direct.log、build.log、latex.log、page9.png 和 validation.json。增量构建与公理审计最终状态见 validation.json。

本轮增量构建退出 0，AxiomAudit 检查 1246 项项目 theorem 通过，仅允许 propext、Classical.choice、Quot.sound。未变模块复用缓存；不声称重编译全部源文件。


### 2026-09-25：将 Pro 的全重入项收紧为下侧重入项

上轮实际目标到总体奖励界的 Lean 连接已完成并验证，属于实质推进。本轮检查 Pro 原有二阶矩证明时发现，前一前缀不必同时通过上下界，只需不低于下界即可导出保留 token 的包络。因此替换既有定理，而非增加论文陈述。

- 用 R_t^- = E[1{log W_(t-1)<-(t-1)tau_-} M_t w_t²] 替换统一界中的全重入项。当前接受且前一 prefix 不低于下界时，log w_t<=t*tau_+ +(t-1)*tau_-，条件归一化给局部矩<=C_t。对其余 histories 保留精确 R_t^-。
- 上侧越界后返回时 w_t<=exp(tau_+)，无需额外矩项。lower_reentry_le_all_reentry 验证 R_t^-<=R_t，因而新的统一 C_t+R_t^- 不大于原统一 C_t+R_t。此比较不声称逐点支配原来保留接受概率因子的更细表达式。
- PrefixGateMoments.lower_safe_retained_moment_bound 与 retained_moment_lower_reentry_bound 证明新主界；原有 bound_of_reentry/horizon_bound 更新为下侧项。PrefixGateExample.selective_lower_reentry_zero 将原半保留例的零重入结果传递到新项。
- EOS 的完整 padded-tree 定义不变：active coefficient 与实现一致，平方矩被 full-tree 矩控制；不把这一矩解释成完整随机梯度方差或 conditional-on-survival 的矩。
- 参考 Theory Part 3 §1.3 的 retained-weight envelope 技巧；本轮重新核对其原文是 C² 点态估计。本文条件归一化得到 C 型界及下侧拆分，正文按适配关系引用，未将具体新推导归给来源。
- 摘要、正文、附录均同步。论文仍 30 项显式环境：只有原 Pro 矩定理和半保留示例记号发生陈述变更，其他 28 项哈希不变。未改变 Max/Pro 算法或训练代码。
- PDF 22 页，正文第 9 页末行 y=732.269；第 10 页 AI USE STATEMENT。已查看第 6 页新推导与第 9 页渲染。通过调整文字保持页面，未改模板、字体、间距、分页或插入空白。
- 证据：/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/lower_reentry/；最终编译为 latex2.log（/tmp/openrlhf_iclr.0VRdva），Lean 最终构建 build2.log；初次 elaboration/排版失败日志保留，最终状态见 validation.json。

本轮最终增量构建退出 0，AxiomAudit 检查 1250 项项目 theorem 通过，仅允许 propext、Classical.choice、Quot.sound；未变模块复用缓存。


### 2026-09-25：Max 信号/系数记号与核心方向结论

上轮收紧 Pro 下侧重入矩界已完成并验证，属于实质推进。本轮发现 H 在方法定义中表示归约前信号、在 Max 更新论证中却表示归约后系数，存在读者误将 token 分母再除一次的风险。因此统一 C_i=H_i/L；H_i 是实际归约前信号，也涵盖可选 homogeneous replacement。

- 正文直接写出 U=E sum_i C_i u_i=Lambda*g、Lambda>=eta/Lmax，并在邻近文字给出二元奖励、iid 组、类内固定长度、on-policy、accepted gate、inactive clip 和数值 clamp 条件。其依据仍是 BinaryUpdate.implementation_policy_direction：同一 multiplier 对每个方向成立，implementationCoefficient 已包含 tokenDenominator 和 uniformBoost。
- 既有 covariance proposition 的含糊“方向符号一致”表述改成 <g,U>>=Lambda_C*||g||²-sqrt(V_C*n*V_f,g)；正裕量通过 differentiability 给局部改善。BinaryAscent.population_ascent_bound 取 d=classMultiplier，population_local_improvement 已使用该实际 multiplier 的严格裕量；未用更强的 eta/Lmax 条件冒充它。
- 合并重复展示的协方差等式/界，缩短已在附录证明的 ideal-standardization 说明。保留核心条件和推导，未增加论文数学环境、算法步骤、采样或优化器模型。
- 只有 prop:binary-complete-update 的记号和 prop:binary-length-covariance 的显式推论表述改变；28 项其他陈述哈希不变。Lean 全部源文件与上轮已审计状态一致，复用证明，不重复构建。
- 证据目录：/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/max_direction_clarity/；最终构建 latex3.log、validation.json 和页面渲染。初版多占页的构建日志保留；最终篇幅和页底由实际 PDF 验证。模板、字号、间距、分页规则与训练源码未改。


### 2026-09-25：组内 Max 与批次 loss 的伪代码范围

上轮消除 Max 信号/系数歧义并明确正文方向结论已完成，属于表述与对应关系的实质推进。本轮阅读当前训练源码，确认 experience_maker 对每个 prompt 组计算 Max 因子，PolicyLoss 在单次调用的全部 active tokens 上取均值。原伪代码只写一个 prompt 组，容易被读成逐组执行优化器更新。

- Algorithm 1 输入加入 prompt-group index b(i)，第一步对每组 b 求 alpha_b/beta_b，第二步按所属组缩放；原来的 batch token 分母和 PPO 更新保持。仍为 8 个步骤，无新增算法组件。
- 总体奖励段落明确 S_PM 是单组目标；这些组目标按 active-token 数比例组合成所描述的 batch objective。随机分母在期望内的要求保持不变，未将平均组 loss 与全局 token mean 混同。
- 代码核对：experience_maker.py 的分组恢复、逐组 normalization 和 fallback；loss.py 的 old/rollout ratio、active-prefix log average、detached token coefficient 和 masked_mean(dim=None)；ppo_actor/replay_buffer 的 dynamic_loss_scale 仍按响应数比例，并不自动实现整个 optimizer step 的全局 token mean。该路径差异已在 REINFORCE_MAX.md 和相应数学条件中保留，本轮不改实现。
- 数学依据复用 ProMaxObjective 的任意非负 reduction 恒等式、ProMaxBatch.corrected_batch_change_eq_rloo_ratio、BatchDenominator.batch_expectation/shared_batch_decomposition。伪代码不是 distributed training loop 的完整复现。
- 30 项显式陈述哈希全部不变；Lean 源码与既有公理审计时的版本一致，未重复编译。PDF 22 页，正文第 9 页底部 y=732.269；已查看第 7 页伪代码与第 9 页渲染。模板、字号、间距、分页规则和训练源码未改。
- 证据目录：/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/batch_algorithm_alignment/，构建 latex1.log（/tmp/openrlhf_iclr.WtJb5H）、validation.json、page7.png、page9.png。


## 2026-09-25：既有正例连接到实际总体梯度步

此前批次伪代码与目标归约已对齐。本轮加强 ex:selective-positive-margin，将同一标量策略族的可行候选连接到实际总体 PPO 目标的梯度上升步；没有新增论文环境。

- SelectiveRewardExample.scalarBlockObjective 保留原 rollout、snapshot、prefix mask、token ratio、四 token 分母、Max 信号和 PPO min/clip；scalar_population_objective_eq 在可行区间证明总体增量 F(delta)=s_epsilon*delta/4。
- scalar_population_objective_derivative 在内部点 0<delta_0<1/15 证明实际目标的导数为 s_epsilon/4。scalar_gradient_step_reward_gain 用 deriv 定义新参数；eta>0 且步后参数不超过 1/15 时，精确真实奖励增量 eta*s_epsilon/8>0。该结果针对明确的一维参数化的总体梯度，不扩展为随机优化器、任意神经策略或非对称缩放独有优势。
- 只修改一项 example，其他 29 项陈述哈希不变；证据与依赖哈希已刷新，清单检查通过。Lean 增量构建退出 0，AxiomAudit 覆盖 1255 项项目 theorem，仅允许 propext、Classical.choice、Quot.sound；未变模块复用缓存。底层声明计数不代表论文定理数。
- PDF 22 页，正文第 9 页底部 y=732.269；第 10 页从 AI USE STATEMENT 开始。已查看第 8、9 页渲染。模板、字号、间距、分页规则与训练源码未改。
- 证据及修改前备份：/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/selective_gradient_step/。最终构建 latex3.log（/tmp/openrlhf_iclr.1hErIY）、build.log、validation.json 和页面渲染。初版 direct.log 的重写失败已修复并由最终成功构建验证。


## 2026-09-25：理想平衡与实际缩放的文字范围

核对 Max 方向与 Pro 下侧重入矩界的假设后，继续审阅不在显式环境中的数学断言。正文结论的 token-balanced 容易被解读为实际保护分支始终精确平衡，因此改为 sign-dependent RLOO scaling；摘要、引言和方法明确平衡及单位二阶矩属于理想系数。附录平均长度比解释补明二元奖励条件。

- 对应既有 NormalizationSafeguards.stabilized_closed_form、capped_stabilized_moment、factor_mean_residual、factor_second_moment_residual_bound：未夹紧的共同缩放保持零质量，稳定项和乘积 cap 改变二阶矩；最终两个 factor clamp 可改变第一矩，fallback 也不保证 token 平衡。不是修改算法或增加新结果。
- 行文中的多值奖励数值例已有 AdaptiveMechanism.multivalued_rloo_centering_counterexample、multivalued_rloo_zscore_negative、multivalued_rloo_adaptive_positive 对应；稀有成功例已有 rare_success_does_not_imply_negative_attenuation。未将显式环境清单误作所有行文断言的自动覆盖证明。
- 30 项显式陈述与全部 Lean 源码哈希不变，复用 selective_gradient_step/build.log 的成功构建和 1255 项公理依赖审计。仅 TeX 文字与清单位置/依赖指纹更新。最新 PDF、页面核对和备份见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/ideal_actual_wording/validation.json。


## 2026-09-25：总体奖励界与 K 组批次 token loss 的连接

此前文字仅说组目标按 token 数加权，本轮将其写成现有 margin 的明确等价表达；奖励定理及全部 30 项显式陈述不变。

- K>=1 个 iid 完整 prompt 组，各含相同 n 个响应；策略和组内 Max/Pro 构造固定，按原 rollout law 取期望。每个批次的 objective、distortion、clipping 项使用同一总 active-token 分母。
- 逐组定义 F_b=L_b*(Delta S_b-D_b-C_b(p))。实际批次括号项等于 sum_b F_b/L_sum，因为这三项对 token reduction 权重均为线性。既有有限目标定义与 token 分区恒等式支持该对应。
- restored_batch_margin_expectation 在正长度的有限 iid 组 law 下证明 E[L_sum*(sum F_b/L_sum)]/(K*n)=E[F_b]/n。eos_batch_margin_expectation 将 F 实例化为实际 eosGroupObjectiveMargin，从 EOS 长度下界和 block_mass_normalized 导出前提。没有使用长度/信号独立性，也没有用 E[L_sum] 替换随机分母。
- 该等价表达仍比较固定候选 q，不赋予数据自适应候选或单次随机优化器额外保证。正文仍使用现有 DPPO/TRM 奖励分解；这里只做有限和及 iid 期望代数，不新增参考归属或研究定理。
- 新连接已从源码编译；首次实例化因 apply 的隐式参数展开超过心跳上限，改为明确给出实际长度和组目标后通过。最终构建、公理审计及 PDF 状态见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/batch_population_identity/validation.json，日志 build2.log、latex1.log。正文第 9 页底部结束，全文 22 页；已查看第 8、9 页渲染。未改模板、间距或训练源码。

本轮最终增量构建退出 0，公理审计覆盖 1257 项项目 theorem，仅允许 propext、Classical.choice、Quot.sound；该计数不是论文成果数。


## 2026-09-25：合并奖励误差界，展开条件序列 TV 推导

用户明确优先学习 DPPO/TRM 各种 bound 的证明；本轮围绕现有算法的误差链实施所选路线，不增加论文数学环境。

- AdaptiveBound.surrogate_error_mixed_tv 从 performance_difference 和边缘 prefix 期望差出发，以 gain_abs_le 控制每一步信号，再以 TV 控制有界函数期望差，并用 prefixTV_mono 收缩到完整响应 TV，得到 4*Rmax*T*min(1,epsilon)*sequenceTV。private value_difference_tv_at_length 只要求相应 prefix 长度上的函数包络，不对无关 histories 添加条件。
- family_reward_error_mixed_tv 对同一非负有限 prompt law 平均绝对误差界。ProMaxCertificate.family_reward_error_bound 通过 le_min 合并 Adaptive 和 Mixed，直接控制 |J(q)-J(p)-S_p(q)|；没有把 lower bound 当成绝对误差界。
- population_performance_from_objective_margin 和 EOSCertificateBridge.population_performance_from_eos_objective 改用 familyRewardCost。原实际目标、随机 token 分母、RLOO、EOS、三比率及组内 gate/scaling 对应保持，仍比较固定候选策略。
- 附录展开 DPPO B.2 的条件序列 TV 证明：概率乘积望远镜 → 三角不等式 → 候选 continuation 归一化 → rollout 加权 local TV 求和。该结果与 chain-rule/Pinsker 及 TV<=1 共同支撑 Adaptive 的逐位置最小值；完整序列 TV 仅在 Mixed 边缘分布路线使用。
- 在 DPPO/TRM 对应推导处引用已有技术，区分借用的通用 bound 与本算法的实际目标适配。统一梯度表达复用 PPODerivative.finite_gated_ppo_hasFDerivAt，只解释 cached gate、两系数信号与 PPO clipping 分支，不作为新的奖励误差成果。
- 30 项显式环境中仅 thm:adaptive-bound 的陈述哈希改变，29 项未变。thm:promax-population 的 B 定义已加强，虽 hash 不变仍人工复核其语义并重新编译 EOS 连接。清单现含 35 个证据模块；清单新鲜度与语义证明分开报告。
- 改动依赖只有 3 个 Lean 文件、摘要、正文和核心附录。训练源码、模板、字号、页边距、间距、分页规则不变。正文通过合并重复说明保持 9 页；PDF 共 23 页，第 9 页末行 y=732.26904296875，第 10 页 AI USE STATEMENT。已查看第 2、5、9、13、14 页渲染。
- 核心构建 build2.log 退出 0。先前 audit.log 通过，但启动早于最终绝对误差重构，不将其当作最终验证；audit2.log 运行以 143 中断，随后重跑最终审计 audit3.log。最终退出状态、公理审计和 PDF 检查记录在 mixed_integration/validation.json。证据目录为 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/mixed_integration/；latex2.log 对应 /tmp/openrlhf_iclr.6ayVuR。

这些 bound 的复用加强了现有奖励结论；借用结果不计为新的通用理论贡献，机器检查也不代表会议接收。总目标保持未完成。

最终 audit3.log 构建退出 0；公理依赖审计覆盖 1261 项项目 theorem，仅允许 propext、Classical.choice、Quot.sound。该计数不是论文成果数。清单新鲜度、交叉引用、PDF 页数与 git diff --check 均通过。


## 2026-09-25：固定正参考尺度下的实际目标误差界

本轮不是修复原不等式的错误，而是移除共同缩放可能造成的过松比较。路线评估在 scale_aware_error/route_assessment.json：满足 ideal Max 核心规则的三动作/三响应单步有限模型中，原 margin<0，固定 sqrt(2) 参考尺度的 margin 等于真实正奖励增量。该检查不作为新增论文例子或实验优越性证据。

- ProMaxObjective.raw_increment_ge_scaled_objective_increment 将原逐组比较用于 scale*A，证明 raw surrogate 差的线性缩放，再用 scale>0 除回。保留实际 H、gate、clipping、cached ratios。
- EOSCertificateBridge 将原完整连接改为 population_performance_from_scaled_eos_objective 和 eosGroupScaledObjectiveMargin。原采样支持、三策略比例、随机 token 分母和 EOS 证明复用，点态比较换成 scaled 版本；原 population_performance_from_eos_objective 由 scale=1 得到。通用批次还原恒等式适用于新的逐组 F。
- 论文中的固定 scale 提到有限期望外、与 n 合并为 scale*n，只是有限和除法代数；不允许用随机 batch scale 直接套此外提形式。未声称 common scaling 不改变所有随机优化器或参数更新。
- 总体奖励定理仍以 familyRewardCost 为余项，因此 Adaptive/Mixed 及其概率条件不变。候选固定，lambda 是分析单位而非新训练步骤。共同尺度消除只在实际 H=lambda*A、M=1 的明示条件下成立。
- 30 项显式环境中仅 thm:promax-population 的 hash 改变。原正反例明确使用单位尺度，29 项其他陈述不变；新增两项 Lean 连接服务于这一个被改写的论文结论。
- 构建、最终公理审计、指纹和页面结果见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/scale_aware_error/validation.json。不改模板、训练代码或算法伪代码；未运行 GPU 实验。

参考尺度版本的最终 AxiomAudit 构建退出 0，检查 1265 项项目 theorem，仅允许 propext、Classical.choice、Quot.sound。清单、引用和 diff 检查通过；PDF 23 页，正文末行 y=732.26904296875，第 10 页 AI USE STATEMENT。该计数不是新增论文成果数。


## 2026-09-25：Pro 证明脉络与结构

组织参考 DPPO performance identity → policy bound 与 TRM building blocks → tighter bounds 的递进方式。正文 Pro 先定义 filter/Mw，再界定单 token 风险、推导下侧重入矩界，最后陈述早晚拒绝模式。矩界证明拆成 local envelope、conditional normalization、history decomposition 三步；附录同步。

thm:prefix-tighter 的紧凑改写直接定义 sequence mask，保留所有数学前提及结论。复核 PrefixMasking.early_deviation_complete、late_deviation_complete：epsilon 非负及上下阈值限制仍明确，early 窗口仍严格，late 计数仍为 suffix 长度，sequence mask 仍 all-or-none。30 项环境中该项 hash 改变，其余 29 项不变。Lean 源文件与 scale_aware_error/ 最终公理审计一致，未重复编译；清单更新只记录新的位置和语义审阅。

正文 Pro 的已有梯度、矩界和示例只是重排；算法和训练代码不变。主体结构调整限定于 Pro，组合部分只压缩两句过渡以保持阅读与页面。最终证据见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/proof_organization/validation.json。


## 2026-09-25：Pro 保留质量的矩界

当前目标是在不增加论文定理的前提下，让 bound 保留实际 filter 的信息。PrefixGateMoments 新定义 localRetainedMass=sum p*M*w，逐项证明其非负和 <=1（含 p=0 的情形）。local_retained_moment_of_envelope_mass 在接受 token 上用 w<=C 得 p*M*w^2<=C*p*M*w，再求和。lower_safe_retained_moment_bound 由此升级为 C*localRetainedMass。

retained_moment_lower_reentry_mass_bound 在 lower-safe histories 使用该界，其余历史保留完整二阶矩 R^-；非负性允许受控一阶矩的求和范围扩大到所有 histories。既有 retained_moment_lower_reentry_bound 由整体 first moment<=1 导出，因此较粗 C+R^- 及 horizon 推论继续成立。consecutive_acceptance_moment_bound 删除重复局部推导并调用共享 envelope lemma。

TeX 只加强既有 thm:prefix-reentry-moment 的不等式链，主体三步证明顺序不变。full padded-tree first moment 与 reentry 项用于该界；active-token 二阶矩继续由它控制，不将任意 mask 接受事件或权重矩当作全梯度方差/奖励改善证明。未变的 29 项显式环境与其他依赖继续核对。构建、最终审计、清单及版面记录见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/retained_mass_bound/validation.json。

保留质量版本最终 audit2.log 退出 0，公理依赖审计覆盖 1269 项项目 theorem，仅允许 propext、Classical.choice、Quot.sound。首轮 audit.log 以 143 终止，未报告证明错误，最终重跑成功。清单与 diff 检查通过；PDF 23 页，正文末行 y=732.26904296875，第 10 页 AI USE STATEMENT。该计数不是论文成果数。

## 同一参数下的 Pro 矩界与奖励界（2026-09-25）

现有正例的 B=101/99、delta=1/30 与 0<=tau<log(B)/2 同时满足：两个深度的保留 correction 系数二阶矩为 1/2，而未过滤二阶矩为 10001/9999 和 1；实际总体奖励下界为 43/3000>0。正文将这两条已证明结果连接在同一正例中，明确采用同一 rollout law、cached snapshot 和候选策略。此处比较 correction 系数矩，不等同于完整梯度方差或跨方法训练优势。

仓库外 JointCheck.lean 直接实例化 selective_retained_moment、selective_unfiltered_root_moment 和 positive_actual_reward_bound，并验证第二步未过滤矩为 1；Lean 编译退出 0。没有新增项目 Lean 定理、论文数学环境或算法组件。仅现有正例陈述 hash 改变，其余 29 项保持；清单仍为 30 项环境、36 个证据模块。附录删除一处将 safeguard 证明错误指向正文的过时措辞。

最新 PDF 共 24 页，正文在第 9 页末行 y=732.26904296875 结束；模板与伪代码保持。最新验证记录：/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/joint_pro_bounds/validation.json。完整项目公理审计继续复用源码未变的 rollout_bound_integration/build.log，本轮直接核验上述联合实例。ICLR 贡献判断与接收仍不能由这些检查代替，研究总目标保持 active。

## 一般 cached mismatch 的正例替换（2026-09-25）

当前 SelectiveRolloutBound 使用任意 K>1 的候选策略族。reference_clipping_zero 替换旧的全位置 reference_clip 恒等式；大 K 时后者不成立，但被拒绝贡献为零。actual_population_margin 仍代入实际 safeguards 与 prefixLogGate，positive_actual_reward_bound 通过内部构造的支持/KL cap 及现有 Adaptive 定理，证明与 K 无关的 43/3000 正下界。

objective_eq_scalar 对任意可行 delta 证明新 candidate-minus-rollout clipped objective 与既有 scalarBlockObjective 相等；objective_expectation 直接复用 scalar_population_objective_eq，删除重复的目标期望枚举。candidate_gradient_step_reward_gain 使用原 scalar_population_objective_derivative 与新 reward_gain，保留实际总体梯度步的精确奖励增量。新增形式化连接用于替换同一个论文正例，未增加论文环境或算法步骤。直接编译 direct1/direct2 均通过；完整构建与审计结果见 uniform_mismatch_integration/build.log 和 validation.json。此前固定 K 记录保留作历史证据。

本次 SelectiveRolloutBound 与 AxiomAudit 完整构建退出 0，审计 1309 项项目定理，仅标准公理。UnboundedCheck.lean 进一步直接实例化：固定 tau=log(2)/4 和实际 epsilon=10^-8，对任意 target 选择足够大的 K，使未过滤首 token 二阶矩超过 target，同时两个深度的保留矩仍为 1/2、实际奖励下界仍为 43/3000。该检查不新增项目或论文定理；页面和最终 hash 记录在 uniform_mismatch_integration/validation.json。

## 当前论文证明的可移植复现（2026-09-25）

新增 formal/build_submission_supplement.py，从已核对的逐项清单提取 36 个证据入口及其 54 个本地依赖模块，按原始字节打包。补充包提供 Lean 4.19.0、Mathlib 锁文件、逐项数学陈述/证明索引、PaperAudit 和五份实现参考文件。源码与数学陈述不变，原有完整仓库 AxiomAudit 保留。

在仓库外的空项目构建目录编译全部 54 个模块，通过 PaperAudit 对导入闭包内 1120 项项目定理的公理检查，仅允许 propext、Classical.choice、Quot.sound。这是当前稿件依赖闭包的定理数，不是论文贡献数，也不是此前全仓库 1309 项审计的替代统计。复用已核对 commit 且 tracked sources 干净的 Mathlib 等第三方缓存，不复用项目编译产物；未测试首次联网安装。相同审计逻辑的负对照拒绝 sorryAx 和额外自定义公理；完整性检查拒绝被修改的证明文件。

最终包位于 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/submission_package/proof-supplement.zip；验证结果见同目录 validation.json。论文仅更新第 10 页复现声明：全文仍 24 页，正文第 9 页末行 y=732.26904296875。未新增数学结论或改变方法，未运行训练、上传投稿、提交或推送。

## 归约约定与动态微批次的对应（2026-09-25）

重新检查当前实现时确认：PolicyLoss 的 token-level 分支在单次调用内除以全部 active tokens；replay_buffer.py 的 dynamic_loss_scale 则按每个 partition 的响应数占比加权，ppo_actor.py 在 backward 前应用。既有反例及研究文档已经区分两者，但现稿缺少明确对应说明。

正文算法前补充 token-count accumulation 条件，附录 D.2 的 Active-token reduction 段说明 token-count 与 response-count 权重对应的 omega，展示两者的精确差。一般有限批次分解支持任意非负 omega；总体奖励定理继续使用全文约定的全批次 token mean。未改变训练实现、方法、30 项显式数学陈述或 Lean 源码。

外部 ReductionCheck.lean 直接实例化 BatchReduction.microbatch_token_gap 和 sample_weighted_response_means，通过 Lean，并打印标准公理依赖。复用既有 dynamic_token_weight_counterexample；CPU 的同名归约检查通过，两个单响应微批次得到 response-weighted=1/2、global-token=3/4。正文第 9 页底部、全文 24 页。

最新补充包和验证记录位于 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/reduction_contract/。包内 Lean 源码、构建配置、公理审计和 lock 与上一轮从零编译的版本逐字节相同，复用 1120 项审计证据；PDF、说明及清单已更新。前一轮 submission_package/ 是其历史版本。
