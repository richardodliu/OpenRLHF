# 当前两策略稿件的审稿核查

本记录针对当前工作树。最新验收按用户更新后的口径：不考虑实验，仅判断理论技术深度与 DPPO/TRM 是否相当。逐项比较见 `THEORY_DEPTH_COMPARISON.md`，结论为总体相当，按此口径完成。下文关于原创增量、适用性和此前验收错误的记录保留；本次结论不将它们改写为录用保证。

## 已核查的核心证明路线

| 来源 | 当前稿件采用的技术 | 审阅判断 |
| --- | --- | --- |
| DPPO `paper/llm_bound.tex`、`paper/app.tex` B.1–B.2 | terminal-reward surrogate 的乘积展开、条件序列 TV | 保留。它们直接连接当前/rollout token ratio 与真实奖励；属于借鉴技术，不计作本文新界。 |
| TRM `main_arxiv.tex` 首个完整文档 | Adaptive continuation-TV 与 Mixed marginal-prefix-TV 两条误差路线 | 保留并取最小值；无需再并列增加较弱的同类界。全上下文散度条件与样本 mask 判定分别陈述。 |
| Theory 第三篇 §1.3 | 拒绝贡献分解与保留权重包络 | 当前引用方向准确。原文约束完整序列权重，本文分析 token ratio；下侧重入项是完成这种转换所必需，不能直接移植原文方差结论。 |
| Theory 第二篇 | token correction 未修正全部状态访问分布 | 用于解释奖励余项的必要性；不把 Pro 写成无偏完整前缀重要性采样。 |

当前奖励证明按点态残差、RLOO 组期望、序列误差三步组织。乘回随机 token 分母后，独立 sibling baseline 才被条件归一化消去。mask 随候选策略重算的变化已包含在拒绝项内。

## 新颖性审阅

PNPO 已使用前缀几何均值，因此“使用前缀”本身不足以构成贡献。当前稿件已明确其几何均值同时用作权重和接受统计量，而 Pro 用其决定 mask，保留单个 token ratio。参考原文：https://arxiv.org/html/2608.01418v1 。这一区别支持研究不同的保留系数矩，但尚不能单凭定义差异判定贡献达到会议录用水平。

本文应集中评价的增量是：样本相关正负缩放的方向条件、前缀 mask 的下侧重入矩项、带缩放/拒绝代价的实际目标—奖励连接。Adaptive/Mixed 通用余项仍归属于参考论文。

CPPO 的 Theorem 1 通过加权 token divergence 和每个中间前缀的累计散度预算收紧奖励余项，证明使用剩余长度系数与 Abel 求和。Pro 的带符号 log-ratio 前缀平均不蕴含这些非负散度预算，因此不能将 CPPO 的阈值直接代入本文奖励界。当前相关工作对其 position-weighted divergence / cumulative prefix budget 的描述与原文一致。参考：https://arxiv.org/html/2606.10968v2 。

Max 复核：两类 RLOO 幅度为 mΔ/(n−1)、kΔ/(n−1)，代入两约束可得 sqrt(N/P)、−sqrt(P/N)；两系数比等于两类平均长度之比。条件方向证明先固定 sibling draw，再用成功/失败系数差及零均值 score 恒等式。下界来自正因子下限和总长度上限；类内变化及提示间共享参数分别通过协方差项保留。当前陈述是有条件的局部总体方向结论，没有将其写成任意 Adam 步保证。

## 其余参考路线的取舍

| 路线 | 决定及原因 |
| --- | --- |
| Theory 第一篇的完整 SGA 二阶进展式 | 保留已有梯度对齐引用；不新增随机收敛定理。完整二阶式需要目标光滑性及整个梯度估计量的矩，当前单个系数矩不能替代。 |
| Math 的 TIS、IcePop | 用于识别旧训练/推理校正与 PPO 的区别；不移植额外 IS 因子，以保持当前单比率算法。 |
| Math 的 MIS、policy-gradient introduction | 支持区分 token 条件动作校正与完整序列换测度；现有 DPPO/TRM 证明已包含需要的技术，不重复写一组定理。 |
| Math 的 on-policy distillation | 教师 log-prob 与熵正则构成不同目标；KL 链式分解可借鉴，但当前已有直接证明，不引入教师或蒸馏假设。 |
| Math 的两篇熵分析 | softmax Fisher/Taylor 界与 NPG 熵演化均需要额外参数化或更新条件，不能由当前 mask 推出，故不并入核心主线。 |

这些取舍基于参考文本的目标和前提，不代表逐式认可参考文档的所有结论。例如 OPD 文档将 full-trajectory log-ratio 一处写作两个 log 的比值；本文不复制该表达式。

## 实现对应关系

源码路径核查：`PolicyLoss.forward` 用全 active-token mean；`NaiveReplayBuffer.setup_dynamic_batch` 令每个微批次权重为其 response 数占本地训练批次的比例；`ActorPPOTrainer.training_step` 在 backward 前乘该权重、最后一个微批次才 optimizer_step；`DeepspeedStrategy.get_ds_train_config` 的动态配置令 gradient_accumulation_steps=1。等 response 数的数据并行 rank 取平均后仍是 response-count 加权微批次 token mean。

正文总体奖励定理要求 token-count 加权批次目标。附录 D.2 已列出两种归约的精确差值 E_red；本轮补上 ΔS_token=ΔS_dynamic−ΔE_red 的代入说明，因此可以沿用总体定理解释动态归约，D_lambda 仍按 token 数聚合。该恒等式比较固定候选、固定样本划分下的目标，不声称 Adam 或梯度裁剪的实际一步一定改善奖励。未修改冻结实验或训练配置。

## 审稿式判断

- **问题与相关性**：response 级优势展开为 token 更新，以及推理/训练策略差异，均直接对应算法的两个设计点。问题陈述清楚。
- **技术正确性**：当前复核的二元系数、条件方向、目标残差、条件序列 TV、Adaptive/Mixed 合并和归约恒等式一致。总体解析证明与 Lean 局部证据的覆盖范围须在补充材料中分别保留。
- **原创增量**：下侧重入矩项及实际样本缩放/归约的奖励连接是主要增量。前缀几何均值、masked objective、Adaptive/Mixed 余项不是本文首创。Max 二元理想规则与匹配 token 标准化等价，不能以该特例宣称独有优势。
- **意义与局限**：给出了何时目标增量具有奖励意义的可解释条件，并展示非空正例和失败边界。尚未证明 Pro 在一般条件下优于 TRM/DPPO；固定宽阈值的包络随位置增长，重入项也需控制。现有贡献支持条件理论分析，尚不能仅凭定理数量判断与参考工作的影响力相当。
- **清晰度**：Pro 已按 masked objective、masking criterion、保留矩、reward decomposition、目标代价和总体界展开；只用 current/rollout 两策略，弃用 gate/filter 机制术语。
- **当前建议**：作为条件理论分析稿已具备可投稿的完整性。预期评审分歧集中在贡献意义：正面评价是实际算法变换的精确界与非空条件；负面评价是部分结论接近一般代数恒等式，统一奖励余项来自已有工作，实际长序列适用性仍依赖条件。我的建议为可投稿、竞争力处于边缘，不据此预测录用，也不宣称与 TRM/DPPO 的贡献强度已经相同。

官方投稿规则核对来源：[ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)：正文至多 9 页，参考文献和附录不计入正文；AI use statement 必需，ethics/reproducibility statements 不计入正文。30 页总上限是本任务要求。此处只核查稿件要求，不声称已完成 OpenReview 提交、作者注册或会议录用。

## 术语和版面

TRM 方法节使用 masked surrogate、masking criterion；其实现说明出现 max-filtering。DPPO 方法节使用 dynamic mask。为统一本文机制名称，当前标题、摘要、正文、附录均采用 mask/masking，不再混用 gate 或 filter/filtering。不可见交叉引用标签及历史 Lean 名称保留，避免无意义破坏依赖。

本轮仅替换术语，公式和假设不变，未增加定理。重新运行 `bash compile_iclr.sh` 成功；PDF 为 23 页，正文第 9 页末行底部 y=730.056 pt，与本轮改动前一致。渲染文本无 gate/gating/filter/filtering/filtered/unfiltered；第 9 页目视检查未见新增明显空白。编译脚本已清理本次中间产物。这些结果证明本轮术语与版面要求，不替代数学正确性或投稿价值判断。


## 上轮验证证据（不作为整体目标完成结论）

| 要求 | 本轮证据与结论 |
| --- | --- |
| 四目录参考、重点 TRM/DPPO | 上述逐路线取舍及正文对应位置；借鉴技术已引用，不把未采用的熵/蒸馏路线增加为成果。 |
| 核心奖励误差和条件序列 TV | 正文第 2、5 节及附录 A、D；同一完整 prefix 条件下的 continuation TV、KL chain/Pinsker 与 Mixed 边缘路线已复核。 |
| Max/Pro 方法保持 | 正负两系数；current/rollout 单比率；前缀几何均值仅作 mask；无第三策略或额外 IS。 |
| 最小自洽证明链 | 本轮无新增数学环境，仍为 30 项（包含假设/定义/例子）；动态归约以已有恒等式代入，未新增定理。 |
| 正确引用与贡献区分 | DPPO/TRM 来源、Seq-MIS 技术来源及 PNPO/CPPO 差异均核对；未宣称已有通用界属于本文。 |
| 术语和排版 | 最终 PDF 全文 23 页，正文末行第 9 页 y=730.056 pt；全文页览及页 9 检查；声明顺序正确；无 gate/filter 机制用词。 |
| 补充材料一致 | 修正 cached-mask 残留；statement map 保留 review_status/review_note；补入新版参数例子检查脚本。 |
| 可复现检查 | 独立解包目录验证 67 个 payload 文件；7 组精确有理数例子通过；PaperAudit 从打包源码构建成功，940 个项目定理公理检查通过。复用已安装的固定版本 Mathlib 缓存，不是空机器网络安装测试。 |
| 科学审稿评估 | 上述问题、正确性、原创性、意义和清晰度评估已完成；可投稿不等于预期录用或神经网络随机优化保证。 |

最终补充包位于 `/tmp/promax_reviewed_supplement.zip`；其中 PDF 与仓库 PDF 字节一致，所有 Lean 源码与独立构建版本字节一致。最终指纹与页数见 `current_theory_validation.json`。本轮未提交 OpenReview、未 git commit/push、未变更冻结训练配置。


## 本轮限定补充：对照 TRM/DPPO

用户将当前任务限定为补齐与这两篇参考论文相比的必要内容。已完成以下修改：

1. 附录 A 在现有证明后直接推导 DPPO 的二次 TV 界与线性期望 TV 界：使用 barD_t≤epsilon、b_(T−t)≤(T−t)epsilon 或 b_(T−t)≤1。
2. 用 Pinsker 和 sqrt(k) 的积分上界推导 TRM Pinsker–Marginal 的 (4/3)R_max delta T^(3/2) 形式；引用来源，不计作新贡献。
3. 解释小幅均匀变化、局部偏移不均匀、长 continuation、较小完整序列 TV 下各项的作用；明确比较的是扣除缩放与拒绝代价后的目标增量。
4. 附录 D.1 合并与正文重复的完整证明，仅保留对应关系与必要前提。30 项数学环境的内容哈希全部不变，未新增定理或算法分支。

条件序列 TV、Adaptive/Mixed 完整推导与总体目标连接原已存在，本轮没有为追求数量重复加入。正文第 9 页底部保持，全文仍为 23 页，修改页 14、23 已目视检查。新版补充包 `/tmp/promax_trm_dppo_supplement.zip` 与当前 PDF 一致；Lean 源码与此前成功构建的补充包字节一致，复用此前公理审计结果。

本轮限定补充已完成；不将它升级为此前更强的“全部研究缺口解决”结论。普遍优于 TRM/DPPO、mask 自动保证全局散度、通用随机优化收敛均不作为本轮待追加的证明任务。


## 完整序列 KL 与交叉条件补充

按用户明确要求，附录 A 增补 TRM 的 Mixed-KL 形式：先在每个 prompt 上使用 K_x=KL(P_x^p||P_x^q)，对 capped square-root 取 prompt 期望，再由凹性给出使用平均 KL 的较粗表达。该界与原 Adaptive、Mixed-TV 界约束同一误差，可取最小值代入总体奖励定理。它不是精确 TV 界的统一改进。证明沿用局部优势界、Pinsker 和边缘化；新增部分为解析推导，未宣称新增 Lean 覆盖。

同时明确 h=T−t、epsilon>0 时交叉位置为 delta/(2 epsilon^2)，保留单位 cap、h=0 和 epsilon=0 情形。原 DPPO 两式和 Pinsker–Marginal 的推导仍保留，压缩重复文字后全文保持 23 页，正文第 9 页末行位置不变。30 项显式数学环境哈希未改，算法及模板未改。当前补充包为 `/tmp/promax_mixed_kl_supplement.zip`，已验证 manifest、PDF 一致性及 Lean 源码与上轮构建字节一致。


## 前缀 KL 与长度保留量补充

按用户要求完成两项补充，位置为附录 C.1 与 C.3 末尾。固定非随机前缀深度下，以条件期望和 KL 链式法则证明负对数几何均值的期望等于前缀 KL 除以深度；明确 EOS padding、随机有效长度和条件接受不能直接沿用同一无条件恒等式。引用 TRM 的 Sample-Based Estimators 附录。

长度部分引用 TRM 的 Length-Neutral Trust Region Masking 附录，给出按响应长度条件化的平均保留率，并将已有早期/晚期拒绝窗口转换为保留量界。未引入独立前缀假设或声称长度中立。接着证明理想 Max 经 mask 后的净有符号质量等于 B(eta_plus-eta_minus)，给出丢弃质量上界与实际保护系数版本；明确它在重要性加权和 score 向量之前成立，真实目标拒绝代价仍由 D_lambda 中的 rho 加权项承担。

新增推导为解析证明，不宣称新增 Lean 覆盖。原 30 个显式数学环境内容哈希不变，清单行号与依赖哈希已同步。正文未改，第 9 页末行保持 730.056 pt；全文 24 页，不改模板或添加空白。第 9、20、23、24 页目视检查通过，第 10 页声明边界核对通过。补充包 /tmp/promax_prefix_analysis_supplement.zip 的 manifest、PDF 和既有 Lean 源码一致性验证通过；既有 Lean 构建结果复用。未改训练代码或配置。


## 2026-09-26：TRM 风格实验部分

正文新增 Experiments：对已完成 Qwen2.5-Math-1.5B 四组544步数据，按 TRM 的性能曲线与 Training–Rollout Log Abs PPL Gap 配对组织。性能轴明确为 online training avg@8；不将训练得分称为 AIME 或泛化成绩。每题8次、每步32题、同一个seed42。表格列全程平均、最后50步平均和同窗口PPL Gap；图仅做20步后向滑动平均，统计表使用原始数据。三组相对Baseline有正向训练差异，但无额外组合优势且PPL Gap未一致降低，正文如实说明。

为保持正文9页，将正例与共享参数反例完整迁移到附录E，未删改其陈述或证明；附录F补充数据固定版本、重复样本、奖励/正确率区分、完整训练配置与聚合方式。30个数学环境哈希保持，清单更新文件身份和行号；Lean字节不变。摘要和结论同步提及单种子训练结果。正在运行的7B未加入完成结果。

最终PDF26页，正文第9页末行726.629pt，第10页AI声明；未改模板、字体、行距或添加空白。页面8、9、24、25、26目视检查。数据与绘图代码位于experiments/dapo_small/results/math15b_544和plot_paper_training.py。补充包/tmp/promax_experiments_supplement.zip含图、逐步CSV、匿名设置和统计JSON，80个payload指纹与PDF验证通过。


## 2026-09-26：展开实验、后移次要推导

按用户要求减少正文压缩：实验分为设置与指标、结果与解释，完整说明四组件对比、两项指标和聚合窗口。Max 的二元系数代入计算、类内长度协方差命题及证明、prompt 聚合和多组 token 加权奖励扩展移至附录；正文保留对应核心结论与条件，并补充附录局部上下文。原30项显式数学陈述哈希保持，证明迁移不新增定理；清单仅更新位置和依赖。

最终正文9页、全文28页，第9页末行729.801pt，声明自第10页开始。第7至9页版面及迁移上下文核对，最终第9页目视检查；未改模板或增加空白。补充包及验证数据以 current_theory_validation.json 为准。既有训练数据、图表数值和冻结训练源码不变。


## 2026-09-26：TRM 图形格式重绘

实际查看 TRM v4 第9页原图，采用逐方法对照 Baseline 的三个双面板图，顺序为左侧 PPL Gap、右侧训练 avg@8。统一范围、细实线与下方图例，采用矢量PDF。未改变数据、20步平滑和统计窗口，10个CSV/统计文件与上轮包字节一致。原四方法叠加PDF已由三个图替换，补充包包含全部新图。

为保留清晰图幅和完整实验解释，将二元系数完整命题/证明和总体奖励证明放入对应附录；正文保留核心结论及证明概述，已有附录的二阶矩证明不再在正文重复展开。30项陈述哈希不变，Lean源码字节不变。最终正文9页、末行732.269pt，全文28页；第7至9页目视核对，引用与分页检查通过。最新补充包 /tmp/promax_trm_pairwise_supplement.zip，82个payload及PDF验证通过。
