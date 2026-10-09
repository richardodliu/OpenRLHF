# REINFORCE Pro Max 论文维护

更新：2026-09-25，两策略目标修订。

## 唯一投稿入口与编译

- 分支 `iclr`；唯一入口 `tex/iclr2027_submission.tex`。
- 科学内容位于 `tex/main/paper-abstract.tex`、`paper-body.tex`、`core-appendix.tex` 和 `reference.bib`。
- 运行 `bash compile_iclr.sh`；独立临时目录编译、检查引用和篇幅，成功后只导出根目录 `iclr2027_submission.pdf`。成功、失败、中断均清理本次中间产物。
- 正文在第 9 页底部结束，全文不超过 30 页。只能调整文字，不改模板、字体、页边距或通过空白控制篇幅。
- 第 10 页声明依次为 AI USE STATEMENT、ETHICS STATEMENT、REPRODUCIBILITY STATEMENT。

## 方法边界

- Max：给正负 advantage 分别分配正缩放系数。RLOO 为输入，数值保护为实现细节。
- Pro：causal prefix mask。只比较 rollout p 与 current q；rho=q/p。
- 前缀几何均值只用于决定 0/1 mask；实际目标为 active-token mean of M*rho*H。
- 不引入独立 old policy，不做 PPO clipping，不额外乘 old/rollout IS。
- 当前前向重算 mask，反向 detach mask；rho 保留梯度。detach 不等于两个候选策略间 mask 不变。
- 统一使用 mask/masking；不用 gate，也不以 filter/filtering 作为 Pro 的机制名称。TRM/DPPO 的组织顺序用于介绍 masked objective、masking criterion，再进行目标分解。

## 最小证明主线

1. Max：保留两系数、二元系数及已有条件方向结论，不扩充辅助性质。
2. Pro：U-G+N 精确目标分解；对 M*rho 用条件归一化与下侧重入分解证明保留二阶矩界。
3. 奖励：沿用 DPPO 乘积展开、TRM 条件 continuation TV 与 Mixed 边缘前缀 TV。
4. 目标连接：比较各自重新计算 mask 的 S(q)、S(p)。rho(p)=1，M(p)=1。
   点态残差为 (rho-1)*(lambda*A-H)+rho*(1-M)*H；D_lambda 为两项绝对值之和的 token 平均。
   因而 J(q)-J(p) >= E[L*(Delta S-D_lambda)]/(lambda*n)-B(p,q)。无 clipping penalty。
5. 正例与反例均用实际 current/rollout mask。正例 zeta=1/100、delta=1/30 给出 19/600 的正下界；反例在同一 current-policy 参数族内展示目标上升、奖励下降。

## 证明技术来源

- DPPO：`tex/literature/DPPO/paper/method.tex` 的 masked objective，以及 `llm_bound.tex` 的奖励差恒等式。
- TRM：`tex/literature/TRM/main_arxiv.tex` 的序列 mask、Adaptive/Mixed 误差界及条件序列 TV 技术。
- Seq-MIS：保留权重的局部包络技术；Pro 增加前缀下侧重入项。
- 论文保留规范引用；前缀几何均值不等于全词表散度约束。

## 形式化范围与实现

`formal/TwoPolicyObjective.lean` 检查新版目标的点态分解、候选 mask 残差、绝对值界和缩放后的 raw 下界。总体期望连接和新版参数例子的完整推导写在论文中；有限例子的补充精确检查为 `reinforce_pro_max/check_two_policy_examples.py`。这些检查不能代替参数化证明。

旧 `ProMaxObjective`、`ProMaxCertificate`、`EOSCertificateBridge` 和旧正例模块包含三策略 clipped/fixed-mask 结果；保留依赖，但不当作新版总体定理的完整机器证明。当前对应范围见 `formal/PAPER_COVERAGE.json`。

仓库 `PolicyLoss` 的 `token_is` 分支与本轮冻结训练源保持核心公式一致；PPO 历史分支不是本文方法。梯度与 causal-mask 回归使用 `experiments/dapo_small/check_prefix_is.py`。

理论总体结论要求文中 token-count reduction；实际微批次/跨 rank 聚合的差别须按附录中的归约恒等式判断，不能由局部 loss 公式推断整个分布式训练已满足全部前提。训练中的冻结源码不得因论文修改而变更。


## 2026-09-26 恢复全理论投稿稿

当前稿件不含训练实验、经验比较、实验图表或实验配置附录。摘要、结论和 Reproducibility statement 同步移除经验结果及实验附件说明。
恢复实验整合前的正文/附录组织：二元系数推导、长度协方差、总体奖励证明、正例与失败边界回到正文；无新增定理，30 项数学陈述与回退前逐字一致。

正文 9 页，第 9 页末行底部 730.056 pt；全文 24 页。模板、字号、页边距及空白控制未变。第 10 页声明顺序保持 AI / Ethics / Reproducibility。
实验数据、脚本、曲线和运行记录仍在研究目录保存；此前投稿图迁至 `reinforce_pro_max/experiments/dapo_small/figures/math15b_submission_archive/`，不进入活动稿或理论补充包。论文回退不改变 GPU 训练任务。
验证记录见 `reinforce_pro_max/audits/THEORY_RESTORE_20260926.md`，当前产物指纹见 `reinforce_pro_max/current_theory_validation.json`。实验整合前后的历史审计保留原始范围，不能视为当前稿件的新审稿结论。
