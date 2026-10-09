## 最新状态：2026-09-26 恢复全理论稿

用户要求移除投稿稿的全部实验部分。已恢复实验加入前的理论正文/附录，摘要、结论、复现声明及补充包范围同步更新。正文 9 页、全文 24 页；30 项数学陈述内容不变。实验记录保留在研究目录，不再进入稿件或补充包。

本轮验证与范围见 [THEORY_RESTORE_20260926.md](audits/THEORY_RESTORE_20260926.md)。下文及先前 Claude 审计为各自历史快照记录；当前构建与索引检查不等同于新一轮科学审稿或录用判断。

> 最新状态：用户将最终验收明确为“仅比较理论研究程度是否与 DPPO/TRM 相当”。已完成逐项比较，判断总体相当；依据见 `THEORY_DEPTH_COMPARISON.md`。该结论不包含实验效果或预期录用概率，下面此前更强标准的历史判断保留。

# 两策略版本的当前投稿状态

本轮完整验收见 `CURRENT_THEORY_REVIEW.md`，最终版面与补充包指纹见 `current_theory_validation.json`。当前结论：可作为条件理论分析稿投稿，主要录用风险为原创增量与适用广度。补充包独立构建、公理审计、精确例子及最终版面已核对。

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

# 当前论文状态

声明更新：第 10 页顺序现为 AI USE STATEMENT / ETHICS STATEMENT / REPRODUCIBILITY STATEMENT；AI 声明以辅助写作、语言润色和改善清晰度为首要表述，并保留技术协助的准确披露。正文 9 页和全文 23 页保持。当前配套补充包与检查记录见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/statement_order/。

验收状态：**已完成**。用户明确以“达到可投稿标准并完成有依据的审稿式评估”为验收；当前稿件按此标准通过，建议作为方法相关条件理论稿投稿。主要审稿风险是原创增量和适用广度，不将实际录用作为完成条件。最终评估见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/completion_review/review.md。

最新措辞与版面检查：2026-09-25，全文 **23 页**，正文最后一行位于第 9 页文字区底部（y=732.269 pt）。清理防御性和叙述流程式措辞，压缩附录重复解释；移除参考文献与附录之间的手动强制换页，消除整块留白。模板、字号、行距及页边距保持。30 项显式数学环境不增不减，5 项仅调整标题或解释文字，Lean 源码逐字节不变。当前 PDF、同步补充包和逐页版面证据见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/submission_language/；后文旧页数和指纹均为当时记录。

更新：2026-09-25。按用户最新界定，数学围绕 Max 的两系数缩放和 Pro 的 causal prefix mask/filter 组织；独立评估采样不再作为论文成果。

## 当前产物

| 项目 | 本轮改写前 | 当前 |
| --- | --- | --- |
| 总页数 | 25 | 23 |
| 正文 | 9 页 | 9 页，页底结束 |
| 显式数学环境 | 34 | 30（包含假设、定义和说明） |
| 伪代码 | 22 个实现步骤 | 8 个核心步骤 |

唯一入口 tex/iclr2027_submission.tex。正文、附录及摘要已同步。训练源码未改，无 GPU 实验、提交或推送。

当前正例已推广到任意满足阈值条件的 cached mismatch 参数 K>1：未过滤 correction 矩可无界增长，保留矩为 1/2，且奖励下界保持 43/3000。正式 Lean 构建、公理审计及固定阈值下的联合实例检查通过；最新记录为 uniform_mismatch_integration/validation.json。

## 本轮取舍

- 移除独立评估、有限候选选择、置信事件和有限样本奖励证书及其专用推导；正例保留精确总体结果，删除 K、概率阈值和 960 组等评估要求。
- 总体奖励条件现为 E[L(Delta S_PM-D_lambda-C(p))]/(lambda*n) > B(p,q)，其中 lambda>0 为固定参考尺度，lambda=1 恢复原式；直接连接实际逐组 loss、Max/Pro distortion 与 Adaptive/Mixed 最小奖励误差界。
- L 保持随机且在期望内部；该结论不声称未加权的平均训练 loss 一旦增加就必定增加奖励。
- Max 的实现保护残差移回附录。正文保留两系数解、二元系数、方向条件和一个多值奖励符号对照。
- Pro 的条件拒绝模式和证明移到方法旁，重入项解释 gate 统计量为何不同于保留 token 系数。
- 搬移内容删除原处重复；没有引入新的算法组件，也没有恢复已停用的 continuation 噪声研究支线。

## 正确性与验证范围

当前已核对 prompt 分组、实际缩放、cached prefix 统计量、token 分母与三策略比率。总体奖励定理使用固定正参考尺度和 Adaptive/Mixed 最小误差界；Pro 矩界保留 E[M*w]，并按定义、矩界、拒绝性质组织。当前前提与 Lean 对应索引见 formal/PROOF_AUDIT.md 开头；下方按日期保留历史改动，不能以某次历史 hash 变化描述全部当前改动。

Lean 的 population_performance_from_scaled_eos_objective 直接接收实际组内目标，在证明中处理 token 分母、三策略比率、RLOO、EOS padding 和零质量字符串，再应用 population_performance_from_objective_margin 的总体奖励界。population_performance_from_eos_objective 是 lambda=1 的特例。点态目标比较已在 Lean 中推导，未作为调用者须另行假设的 reward certificate。

已验证的梯度步结果：既有 scalarPopulationObjective 定义总体函数，SelectiveRolloutBound.objective_eq_scalar 证明它对应当前正例的实际 clipped objective，并复用原证明得到在 0<delta_0<1/15 时导数为 s_epsilon/4。若 eta>0 且 delta_1=delta_0+eta*s_epsilon/4<=1/15，则该总体梯度上升步的真实奖励增量恰为 eta*s_epsilon/8>0。起点取内部点以保证双侧导数；沿用同一 rollout、snapshot、cached gate 与组内信号，未改变算法。该例的两缩放系数相同，结果不构成非对称缩放独有优势或任意神经策略/随机优化器的保证。

批次连接：取 K 个同分布独立 prompt 组，各含 n 个条件 iid 响应。以 L_sum 为总 token 数，批次目标、distortion 和 clipping 均按 L_b/L_sum 聚合；E[L_sum*(Delta S_batch-D_lambda_batch-C_batch(p))]/(lambda*K*n) 等于现有单组 margin。Lean 的 restored_batch_margin_expectation 证明先还原随机分母、再取期望的恒等式；eos_batch_margin_expectation 是单位参考尺度的实际 EOS 组目标实例；任意固定正尺度使用同一通用恒等式代入 scaled group margin。该实例导出长度正性和组概率归一化。不要求组长度与目标独立，不引入独立评估或候选选择。

最新核心增量构建 PrefixGateMoments 退出 0；最终 AxiomAudit 状态记于下方最新记录。论文仍为 30 项显式数学环境、36 个证据模块，正文 9 页、总计 24 页。训练源码、模板与排版参数未改。

最新 PDF 记录为 core_result_framing/validation.json；最近数学集成为 uniform_mismatch_integration/validation.json；此前联合矩界记录为 joint_pro_bounds/validation.json；此前正例合并为 rollout_bound_integration/validation.json；此前相关工作记录为 prefix_related_work/validation.json；Pro 组织记录为 pro_bound_narrative/validation.json；核心前提复核见 core_chain_review/。矩界、证明与当时版面记录：/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/retained_mass_bound/；最终状态以 validation.json 为准。Pro 结构记录保留在 proof_organization/。mixed_integration/、batch_population_identity/ 及更早目录保留此前记录。论文到 Lean 的对应包含人工语义审阅；机器验证不是自然语言等价或会议接收证明。总目标未标为完成。

Pro 的当前组织以误差界为主：4.1 masked objective 与精确分解，4.2 保留权重矩界及拒绝机制短段落；第 5 节按条件序列误差、缩放/筛选误差、总体奖励界展开。早晚拒绝定理及证明完整搬至 C.3，30 项陈述哈希全部不变。正文末行 y=732.26904296875，全文 24 页；模板、算法和 Lean 源码未变。记录见 pro_bound_narrative/validation.json。

## 研究贡献判断

| 核心 | 本文提供的认识 | 贡献边界 |
| --- | --- | --- |
| Max | 两个正系数可纠正 response-to-token 的标量质量不平衡；二元系数有闭式解释；更新方向取决于系数与 score 的关系。 | 二元理想规则与匹配 token 标准化等价；部分正方向性质适用于一般正缩放，不能宣称独有优势。 |
| Pro | causal prefix 筛选对早晚偏移有不同影响；前一前缀满足下界即可控制保留权重，额外矩项只需计入下侧重入。 | 这是实际 gate 的条件权重矩结论，不是完整梯度方差或 benchmark 优越性。PNPO 的前缀均值与 CPPO 的前缀预算已纳入比较，使用前缀本身不是创新依据。 |
| 组合 | 实际目标的变换代价与 DPPO/TRM population remainder 明确连接；正反例说明何时目标增长具有奖励意义。 | 通用 Adaptive/Mixed bounds 属于参考论文；补偿后的 raw surrogate 身份不证明某个 gate 更优。 |

当前证明结构已与指定参考材料的目标分解、条件界和机制解释相接近，但不能据此认定原创成果强度或 ICLR 接收水平已经相当。本轮的实质改进是让结论直接服务于方法，减少了与方法核心关系较弱的统计评估支线。

## 已合并：现有正例中的通用奖励正下界

同一三动作族取 B=101/99、delta=1/30 和 0<tau<log(B)/2，两层均保留一半 rollout 质量。固定 lambda=s_epsilon，实际 PPO 目标增量期望为 s_epsilon/120，scaled distortion 为 s_epsilon/1200，参考 clipping 为零；还原 L=4 并除以 lambda*n 后，总体 margin=3/200。root TV=1/200，local TV<=1/30，Adaptive 误差上界=1/1500，从而奖励下界为 43/3000>0。

正式模块 SelectiveRolloutBound 使用实际 Max 系数和 prefixLogGate，证明参考 clipping、支持与构造的 KL 上界，并调用既有 Adaptive 误差定理。原一维梯度结果仍使用 snapshot 基准；数值通用界使用 rollout 基准，正文已区分。只替换现有正例的部分计算，没有新增正文数学环境或算法。该例不构成非对称 Max 或 Pro 相对其他方法的优势证明。

SelectiveRolloutBound 与 AxiomAudit 构建退出 0，审计 1309 项项目定理、仅标准公理；论文仍含 30 项数学环境，36 个证据模块。当前验证与 PDF 记录见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/rollout_bound_integration/validation.json。此前仓库外可行性记录保留在 rollout_positive_bound/。

## 指定参考材料的证明技术与贡献对照（2026-09-25）

本轮按用户要求阅读 Theory 三篇、TRM 的活动文档、DPPO 的理论及附录，并核对 policy-gradient、OPD、entropy Markdown 的相关推导。Max 的方法核心固定为正负 advantage 的两系数缩放；不新增算法组件或定理。

| 来源及位置 | 可复用技术 | 当前论文的具体适配与归属 |
| --- | --- | --- |
| DPPO v1 §3、附录 B.1–B.2 | 轨迹概率乘积望远镜展开；当前 token 偏移与 continuation 偏移分离；序列 TV 的条件求和界 | Adaptive 证明采用该恒等式，逐步标明来源；本文额外处理 RLOO baseline 与 EOS。此通用恒等式不计作本文原创结果。 |
| TRM v4 Adaptive theorem、附录 D | 完整 prefix 上的 tower property；未来 TV 的 1、线性 TV、KL chain rule + Pinsker 三种界 | 直接采用 Adaptive remainder；保留原有 horizon 阶数，不声称得到更强通用界。 |
| Theory Part 1 §2 | SGA 的方向内积与二阶矩分开分析 | Max 条件方向对应一阶 progress 项；本文分析采样正负系数与 token 分母。未引入新的 smoothness 假设或随机优化器收敛主张。 |
| Theory Part 2 Analyses 1–2 | 条件比率归一化、IS 二阶矩分析 | 明确 Pro 只控制保留 token 系数，完整梯度还含 advantage、score 与跨 token 相关。 |
| Theory Part 3 §1.3 | discarded-contribution 偏差分解；retained-weight envelope 二阶矩界 | 组合目标按 gate 丢弃项拆分；Pro 按前一前缀是否低于下界拆分。原文使用点态 C² 包络，本文利用条件归一化得到 C 型界并隔离下侧重入项，不能把这一具体推导误归给原文。 |
| Brief Introduction of Policy Gradient 的 score/baseline 推导 | 有限求导、条件期望、score 均值为零 | 沿用标准工具，论文引用 Williams 与 RLOO 原始文献；不把讲义的渐近 normalization 简化当作有限组独立性。 |
| OPD、entropy Markdown | occupancy 与动作分布区分；局部 Taylor/Fisher、协方差展开 | 用于核对技术适用范围；其 reverse-KL 目标、softmax/NPG 或局部近似未成为 Max 的额外模型，不搬入无关结论。 |

核对版本：TRM 源文件含两个串接文档，只使用首个 end-document 前的活动文档，对应 arXiv:2512.23075v4；DPPO 使用 arXiv:2602.04879v1，保证附录编号可追溯。Theory 三篇分别按页面署名 Yingru Li、Jiacai Liu 及 2025-10-30、10-31、11-04 日期引用，类型标明 Technical blog post。来源元数据已核对作者页面与 arXiv。

阅读材料中的简化不能直接转成本文结论：二阶矩的指数上界不等于必然指数增长；一次样本 gate 接受不推出所有 context 的 TV/KL 界；OPD 的 log ratio 不能读成两个 log 的比值。论文只借用逐步可核对且条件适用的技术。

贡献判断：本文已具备与这些工作相近的“目标分解—条件界—作用条件”证明结构，但本轮没有产生新的通用性能界，不能据此判定数学贡献已与 TRM/DPPO 相当。现有方法相关增量是两系数缩放的条件方向分析、cached prefix gate 的 reentry 矩边界，以及实际组合目标的误差分解与 RLOO 总体奖励连接。Max 的正乘子性质部分适用于更一般的正缩放，组合证书完全校正后还原 raw surrogate；这些事实决定了不能宣称本轮证明了独有性能优势。贡献表达应围绕上述具体认识，而不是扩大定理数或改造 Max。


## 归档

本轮改写前文件在 core_focus/*.before；旧完整章节归档位于 /volume/pt-train/users/rbliu/workspace_backups/OpenRLHF/20260924T133539Z-single-core-paper/before.tar.gz。历史实验与 Lean 底层依赖保留，不作为当前贡献数量。


## 四个优先参考目录（2026-09-25）

用户明确 DPPO/、TRM/、Theory/、Math/ 中的结论与技巧均可学习和参考。已核对 Math/ 的 7 份文件，它们与原 literature 根目录同名文件逐字节一致，不是新增的一套论文。复核了 policy-gradient 讲义的二元分析、OPD 的 occupancy 拆分、entropy 的局部展开及 TIS/MIS/IcePop 的校正/过滤机制；复用方向已整理进 PAPER_DESIGN.md。

重点提炼与方法的连接：Max 使用 score/baseline 恒等式、条件期望、方向内积和必要协方差；Pro 使用 change of measure、条件矩、截断/过滤偏差拆分；组合分析使用 DPPO/TRM 的奖励分解。局部 Taylor/Fisher 技巧保留为可选择的证明工具，未据此新增 entropy 收敛主张或改变优化器。该次资料归类只更新维护文档，未修改论文或 Lean；后续数学修改及验证见上方最新记录。


## 2026-09-25：剩余参考路线的取舍判断（已实施部分见后续记录）

用户要求先判断剩余路线能否强化现有核心结论，再决定替换或合并。以下保留当时的路线评估；后续已实施所选两条路线，当前证明与编译状态见下方记录。

### 决定一：保留 Adaptive，合并一条 Context Shift / Mixed-TV 路线

来源为 TRM 活动文档的 Context Shift、Mixed Bounds 及附录 Mixed 证明（main_arxiv.tex:201、246、710）。令 d_t^p、d_t^q 为完整 prefix 的边缘分布，g_t(h)=E_{a~q}[A_t^p(h,a)]。性能差异恒等式给出

    J(q)-J(p)-S_p(q) = sum_t (E_{d_t^q}g_t-E_{d_t^p}g_t).

在现有 |R|<=Rmax、localTV<=epsilon 条件下，|g_t|<=2 Rmax min(1,epsilon)。由 TV 控制有界函数期望差和 prefix 边缘化的 TV 收缩，可得

    B_ctx = 4 Rmax min(1,epsilon) sum_t E_x TV(d_{x,t}^p,d_{x,t}^q),
    B_ctx <= B_mix = 4 Rmax T min(1,epsilon) E_x TV(P_x^p,P_x^q).

P_x 为完整响应分布。第一步的 prefix 分布相同，因此也可写 T-1，但不为这一常数优化新增论文陈述。最小合并方案为 B_new=min(B_Adap,B_mix)，沿用当前总体 margin 和奖励定理的结构。以一条 Context Shift 推导支撑 Mixed，而不把全部 TRM 界分别列成定理。

收益：某些策略对上可严格减小真实奖励误差项，不改变 Max、Pro、采样或 loss。代价：引入一个完整响应 TV 量；它可在现有有限模型定义，不额外假定 policy/length 独立。若要将它变成可计算的训练诊断则需要额外估计工作，本轮不引入。Pro 的 cached old/rollout prefix mask 不自动控制 p/q 的完整响应 TV；不得声称已有 gate 直接满足新散度界。

关键边界：不能把无条件 sequence TV 直接塞进现有 Adaptive 证明的条件 future-TV 的 min 中；稀有 prefix 的条件分布偏移可能远大于整体偏移。必须走完整 prefix 的边缘分布差异这条独立推导，再在最终有效界上取最小值。

非支配性检查使用完全归一化的三步二动作正概率策略（只在仓库外保存代数检查，不加入论文例子）：一组得到 B_Adap=0.0770022、B_mix=0.0594；仅最后一步改变的另一组得到 B_Adap=0、B_mix=0.12。前者说明可真收紧，后者说明不能替换 Adaptive。策略质量及 TV 用有理数计算，KL 用浮点对数；这属于路线筛选证据，不是 Mixed 定理的 Lean 证明。明细见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/remaining_route_assessment/comparison.json。

当时的 Lean 基础包括 AutoregressiveTree.performance_difference、gain_abs_le、TreeTV.prefixTV_mono；后续已完成 gain 的 prefix 分布期望差到 Mixed 上界、prompt mixture、总体奖励定理和实际 EOS 目标的连接。

### 决定二：将 DPPO 的统一梯度表达合并进现有方法解释

来源为 DPPO 附录 Algorithmic Details for Stability Analysis（paper/app.tex:261）。在现有 detached omega、M、w、H 下，避开有符号 PPO clipping 的不可微点，实际目标的梯度为

    grad S_PM = sum_k omega_k M_k w_k r_k H_k chi_k grad log q_k,
    chi_k = 1{H_k>0,r_k<c_plus} + 1{H_k<0,r_k>c_minus}.

H_k=0 时整项为零。原论文的 mask/ratio/advantage 表达需适配本算法三策略比率：w=old/rollout，r=current/old；不能把 Pro mask 与 PPO clip-active mask 混为一个算法定义。两系数缩放在核心分支保持信号符号，因此可以清楚区分缩放幅度、因果筛选与 clipping 是否传递梯度。

收益是直接展示实际更新机制、连接现有条件方向分析；它本身不降低奖励误差、不证明随机优化器收敛，也不产生方法优越性。现有 PPODerivative.finite_gated_ppo_hasFDerivAt 已覆盖有限和导数及正确 clipping 分支，宜以公式替换现有重复文字，复用证明，不增设论文定理或新的方法比较模型。

### 其余路线

- Pinsker-Marginal / Coupling：在同样的 epsilon、delta 前提下，现有 Adaptive 保留实际逐位置 TV 并逐位置取更小余项，已支配这些粗化路线；不单列新增界。
- Mixed-KL：保留为无 TV 信息时的备选。当前最小方案先用 Mixed-TV，不同步扩展两个版本及多个等价推论。
- DPPO Binary / Top-K 散度近似：本文不计算该类词表粗粒化散度；现有下界无法直接证明本文所需的上界控制，不恢复到投稿稿。
- TRM k2/k3、LN-TRM/SER：涉及不同筛选量、估计任务或方法规则，不作为当前 Pro 的额外组件。其长度偏置问题已有当前 Max 长度系数和 Pro prefix 分析对应；不将别的 gate 的结论直接移植。
- 不增加独立评估采样、随机优化器收敛或比较实验。两项入选路线分别服务于现有奖励误差和实际梯度解释，正式修改须继续满足正文 9 页、全文<=30页和来源引用要求。


## 2026-09-25：以误差界为主线合并 Adaptive 与 Mixed

用户明确最关注 DPPO/TRM 各种 bound 的推导与证明。当前保留一条服务于实际算法的链：Max 缩放和 Pro 筛选的目标误差 → RLOO raw surrogate → 奖励近似误差 → 真实奖励下界。附录显式展开条件序列 TV 的望远镜推导与归一化步骤；Adaptive 与 Mixed 使用不同有效分解，在最终误差界处取最小值，不以无条件 TV 替代条件未来 TV。

- AdaptiveBound.surrogate_error_mixed_tv、family_reward_error_mixed_tv 证明 Mixed 的绝对误差界；ProMaxCertificate.family_reward_error_bound 合并两条绝对误差界。population_performance_from_objective_margin 和 EOSCertificateBridge.population_performance_from_eos_objective 已改为实际目标减去最小误差界。
- DPPO 统一梯度公式仅用于解释同一实际目标，复用 PPODerivative.finite_gated_ppo_hasFDerivAt；没有新增算法步骤或独立论文定理。Max 仍为正负优势两系数缩放，Pro 仍为 cached causal prefix mask/filter。
- 只修改一个显式 theorem，未增加论文数学环境。来源已在具体推导处引用；通用 bounds 归属参考论文，本文的适配包括实际缩放/筛选 distortion、RLOO、随机 token 分母和 EOS。不能将借用的通用界算作新的原创贡献。
- 全文 23 页，正文第 9 页末行 y=732.269，第 10 页为 AI USE STATEMENT。已查看第 2、5、9、13、14 页；没有修改模板、字体、间距或分页规则。最终机器验证状态见 mixed_integration/validation.json。

最终 audit3.log 构建退出 0；公理依赖审计覆盖 1261 项项目 theorem，仅允许 propext、Classical.choice、Quot.sound。该计数不是论文成果数。清单新鲜度、交叉引用、PDF 页数与 git diff --check 均通过。


## 2026-09-25：误差界与实际算法的针对性复核

以 mixed_integration/validation.json 的最终证明为基准，重新读取实际目标分解、正文奖励定理、EOS 实例化及 Max 更新条件；TeX、Lean、关联训练源码和 PDF 指纹与该次验证一致，复用已完成的内核检查，不重复扩大审计。

| 接口 | 核对结果 |
| --- | --- |
| Max 信号 | 误差项使用实际 H，包括数值保护和配置的 fallback；没有将所有分支假定为理想 token 平衡。核心方向结论另行保留二元奖励、on-policy、accepted gate 和 inactive clipping 条件。 |
| Pro gate | 目标分解不要求 M 与当前 token 或 reward 独立。实际 causal prefix mask 满足所需非负性；cached old/rollout gate 不被当成 candidate/rollout 全局 TV 上界。 |
| 两段比率 | 在正 rollout support 上，r*w=q/p，r0*w=1；EOS 证明显式完成该消去。 |
| clipping | C(q)>=0 允许省去候选扣减，参考项 C(p) 仍需保留，因 rollout 不必等于 frozen snapshot。 |
| 随机长度 | 先以实际 L 还原 token 加和，再取整个 RLOO 组的期望；没有把 E[Z/L] 换成 E[Z]/E[L]。 |
| 奖励余项 | Adaptive 与 Mixed 对同一绝对误差给界，再取最小值；条件 continuation TV 与 marginal prefix TV 的证明路径分开。 |

本轮针对这些接口未发现新的数学缺口，未改论文数学或增加辅助定理。修正研究 README 中仍为 22 页的过时状态。当前边界依然是条件奖励改善：通用 bound 的复用不能单独证明 Max/Pro 独有优势或 ICLR 接收。理论贡献的评估应聚焦缩放、筛选误差与条件有效性带来的具体认识，而不是增加被引用界的数量。


## 2026-09-25：参考尺度消除共同缩放的冗余误差

上一轮接口复核确认原 bound 有效，本轮发现它对共同正缩放可能过松。有限模型的路线检查：三个等概率动作，奖励 (1,1,0)，三个 iid 单 token 响应，ideal Max 在所有非零信号上给出 H=sqrt(2)*A；零信号组保持零。取 q-p=(1/50,-9/500,-1/500)，所有比例位于 PPO 未裁剪区间，old=p 且 gate 全接受。真实奖励增量为 1/500；原单位尺度 margin=(7-6*sqrt(2))/500<0，而参考尺度 sqrt(2) 的 margin=1/500>0。T=1 的 Adaptive 余项为零。这是实际 ideal Max 核心规则下的精确代数检查，不是实验结果，也没有加入论文的新 example。

据此替换现有总体奖励定理：D_lambda=mean[m*w*|r-r0|*|lambda*A-M*H|]，margin_lambda=E[L*(Delta S-D_lambda-C(p))]/(lambda*n)。对任意固定 lambda>0 应用原分解至 raw signal lambda*A、保留实际 H/M/clipping，再除以 lambda，得到同一 raw surrogate；随后沿用 RLOO 和 Adaptive/Mixed。选 lambda=1 可保留原条件，但任意给定的其他 lambda 不保证更紧。

- ProMaxObjective.raw_increment_ge_scaled_objective_increment 证明实际逐组比较；EOSCertificateBridge.population_performance_from_scaled_eos_objective 处理实际 token 分母和 EOS。原 theorem 保留为 scale-one 特例，依赖方无需改算法。
- lambda 固定在所有 prompt/组之外；没有移出随机组缩放，没有引入训练超参数、独立采样或候选选择。共同缩放不会产生 distortion 的断言限定在 H=lambda*A 且 M=1。其他 safeguards/fallback 仍通过实际 H 进入 D_lambda。
- 只修改既有 thm:promax-population，论文仍 30 项显式环境；29 项陈述 hash 不变。既有正反例使用 D=D_1、mathfrakM=mathfrakM_1，未改。
- 核心证明与 PDF 已构建；最终审计和版面证据见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/scale_aware_error/validation.json。这项加强来自同一目标误差分解的正尺度换算，不将它误归为参考论文现成定理。DPPO/TRM 仍为奖励分解及余项 bound 的引用来源。

参考尺度版本的最终 AxiomAudit 构建退出 0，检查 1265 项项目 theorem，仅允许 propext、Classical.choice、Quot.sound。清单、引用和 diff 检查通过；PDF 23 页，正文末行 y=732.26904296875，第 10 页 AI USE STATEMENT。该计数不是新增论文成果数。


## 2026-09-25：按 DPPO/TRM 的证明顺序重组 Pro

根据用户进一步明确的 Pro 范围，保留其他部分的总体结构，将 Pro 改为：4.1 Prefix filtering and the retained coefficient；4.2 A bound on retained-weight moments；4.3 Consequences for causal rejection。参考 DPPO 的恒等式到 bound、TRM 的分项控制到组合结论，先告诉读者要控制 Mw，再说明 prefix 平均值不能直接控制单 token，接着给出下侧重入矩界，最后解释早晚拒绝行为。

矩界推导按三个步骤呈现：由 log W_t-log W_(t-1) 得局部包络；由条件 ratio 归一化得 C 型矩界；按前一 prefix 是否低于下界拆出 R^-。早晚拒绝定理压缩重复文字，保留共同 interval、epsilon 条件、严格拒绝窗口及计数结论。附录使用一致的分组和证明步骤。新增过渡明确 retained-weight moment 与被丢弃贡献的奖励误差分别在哪处理。

本轮无新数学结论、算法步骤或 Lean 源码变更；30 项环境中只有 thm:prefix-tighter 的等价改写改变 hash。已有 early_deviation_complete、late_deviation_complete 和 retained-moment 证明逐项对照，复用 scale_aware_error/ 的最终成功审计。最终正文 9 页、全文 23 页，正文末行 y=732.26904296875；总体奖励定理从第 8 页开头开始，避免其开头被拆在上一页末。构建、页面和覆盖清单见 proof_organization/validation.json。


## 2026-09-25：在 Pro 的同一矩界中保留筛选质量

之前最终一步把 E_p[M_t*w_t] 放宽为 1。本轮保留这一加权保留量，得到 E_p[(M_t*w_t)^2]<=C_t*E_p[M_t*w_t]+R_t^-<=C_t+R_t^-。方法、假设和下侧重入项不变；当该一阶矩小于 1 时，新的显示上界严格小于原来的统一上界。它不是 rollout 的未加权接受概率。

证明仍是 Pro 当前的三步：局部包络给 M*w^2<=C*M*w；按历史拆分后，受控部分的一阶矩可由全体历史上的 E[M*w] 上界；条件归一化再给 E[M*w]<=1。原较粗结论由此导出，没有平行重复一套证明。consecutive_acceptance_moment_bound 中的重复有限和推导也改为复用局部包络引理。

来源仍为 Theory Part 3 §1.3 的 retained-weight envelope，以及 Part 2 的条件比率归一化；原来源使用 C^2 点态估计，本文的 C 型因子、保留质量及下侧重入拆分明确作为该技术的适配，不宣称是来源中的原定理。正文与附录保留相同证明顺序。只替换 thm:prefix-reentry-moment，未增加论文数学环境或算法步骤。最终验证见 retained_mass_bound/validation.json。

保留质量版本最终 audit2.log 退出 0，公理依赖审计覆盖 1269 项项目 theorem，仅允许 propext、Classical.choice、Quot.sound。首轮 audit.log 以 143 终止，未报告证明错误，最终重跑成功。清单与 diff 检查通过；PDF 23 页，正文末行 y=732.26904296875，第 10 页 AI USE STATEMENT。该计数不是论文成果数。

## Pro 主文与附录的推导衔接（2026-09-25）

对照 DPPO/TRM 原文后，正文 5.2 和附录 A.3 统一标出精确奖励分解、条件 continuation TV 控制、误差汇总三个步骤；Mixed 路线控制同一误差的边缘 prefix 分布变化。4.1 明确区分保留权重矩和奖励 distortion，避免把辅助矩界当作奖励下界的必要前提。修正例子前仍称全部例子使用单位尺度的过时引导句。算法、模板与其余 29 项显式数学陈述保持。最终正文第 9 页底部、全文 24 页，详见 rollout_bound_integration/validation.json。

## 同一参数下的 Pro 矩界与奖励界（2026-09-25）

正文现有正例已明确：同一 B=101/99、delta=1/30 设置下，保留 correction 系数二阶矩为 1/2，未过滤矩为 10001/9999 和 1，同时总体奖励下界为 43/3000>0。联合实例复用已有 Lean 定理并直接编译通过，没有新增论文环境或项目定理。完整证明对应见 formal/PROOF_AUDIT.md 同名记录；最新 PDF 验证见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/joint_pro_bounds/validation.json。正文仍在第 9 页底部、总计 24 页。

## 已合并：同一正例去掉 cached mismatch 的特定参数限制（2026-09-25）

SelectiveRolloutBound 和同一个正文正例已改为任意 K>1、2*tau<log(K) 的构造。候选首 token 从被拒绝动作 2 向动作 1 转移 1/200 概率，保留历史的变化仍为 delta；全部目标增量统一以 rollout 为基准。delta=1/30 时实际目标期望、distortion、margin 和 Adaptive 上界仍分别为 s_epsilon/120、s_epsilon/1200、3/200 和 1/1500，奖励下界为 43/3000，且不依赖 K。未过滤矩 (K+1/K)/2 可无界增长，而保留矩始终为 1/2。算法未改。

objective_eq_scalar 证明新 clipped objective 与既有标量目标逐组相同，objective_expectation 因而复用其总体期望证明，candidate_gradient_step_reward_gain 接到原导数结果。大 K 时被拒绝位置的参考 ratio 可越过 PPO 区间；reference_clipping_zero 正确证明的是 masked clipping penalty 为零。严格支持和有限 KL cap 在奖励证明内部构造，没有新增调用者假设。只替换现有 example，论文仍有 30 项显式数学环境、36 个证据模块；其他 29 项陈述保持，正文第 9 页底部、全文 24 页。最新验证：/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/uniform_mismatch_integration/validation.json；四份先期可行性文件保留在 uniform_mismatch_bound/。

## 核心结果展示同步（2026-09-25）

摘要、引言和结论现明确说明同一个两步策略族中有界保留二阶矩、正奖励下界和任意大未过滤二阶矩的联合结果，代替此前泛称正反例区分目标/奖励上升的概述。数学陈述、方法与证明源码不变；没有宣称一般梯度方差降低或训练性能优越性。最新 PDF 验证记录：core_result_framing/validation.json；24 页、正文第 9 页底部。

## 最新组织调整：Pro 的奖励证明主线（2026-09-25）

正文现在依次说明原始奖励误差（5.1）、实际目标的 scaling/filtering 误差（5.2）、总体奖励下界（5.3）。5.1 同时展开 Adaptive 的条件 continuation TV 和 Mixed 的优势幅度 × 边缘 prefix TV，分别引用 DPPO B.1–B.2、TRM D/C.2。保留权重矩作为独立性质；早晚拒绝只在正文简述，完整证明仍保留。附录标题与衔接同步。

30 项显式陈述 hash 全部不变；Lean、算法、模板和训练实现未变。此次为组织及推导展示修改，复用已有证明审计。最终 PDF 24 页，正文第 9 页底部结束。验证记录：/volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/pro_reading_order/validation.json。

## 当前论文证明的可移植复现（2026-09-25）

新增 `formal/build_submission_supplement.py`，按当前清单生成可移植证明包。36 个证据入口及 54 个本地依赖从空项目构建目录编译通过，PaperAudit 检查 1120 项导入闭包中的项目定理，公理与完整性负对照通过。复用已核对的第三方缓存，不复用项目编译产物；未测试首次联网安装。数学与方法保持，仅更新第 10 页复现声明；全文 24 页，正文第 9 页底部。详细证据见 formal/PROOF_AUDIT.md 同名记录，最终包及验证位于 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/submission_package/。

## 当前归约约定（2026-09-25）

正文与附录 D.2 已显式区分微批次 response-count 加权和全批次 token mean，并给出精确差；有限批次目标分解覆盖二者，总体奖励下界采用后者。等式复用已有 Lean 定理并实例化检查通过，既有 CPU 归约测试通过。数学环境仍 30 项，全文 24 页，正文第 9 页底部。最新 PDF 配套补充包及验证见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/reduction_contract/；详细对应见 formal/PROOF_AUDIT.md。

## 完成条件与贡献归属复核（2026-09-25）

现稿与补充包逐字节一致；当前核心结论在声明条件内有对应形式化证据。本轮按 DPPO/TRM 原文和 ICLR reviewer guide 区分通用已有界、本文方法适配及构造结果，未发现必须新增定理的缺口。研究价值和接收竞争力不由 Lean 数量决定，第 3 项的原待澄清状态已由用户后续确认解除；当前完成结论见本文件开头。详细自审（非独立评审）及当前产物指纹见 /volume/pt-train/users/rbliu/latex/rloo_group_validation_20260925/completion_review/。论文、方法和 Lean 均未修改。
