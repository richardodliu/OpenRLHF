# Overall verdict

**NO MATERIAL CRITICAL ISSUES FOUND**

本结论仅针对本轮限定的、能够在不新增实验、不改变已有结果和核心结论的前提下直接修复的重大错误，不是录取判断，也不是数学正确性的绝对保证。

## 审计记录与最终裁定

Claude Code 2.1.282 已完成一次独立只读审计，实际模型为 `claude-opus-5-5[1m]`，会话 `3b8c2b4e-42cf-4cdd-8412-d66f03409378`。其完整原始报告保留在 [CLAUDE_DEADLINE_AUDIT_20260926.md](CLAUDE_DEADLINE_AUDIT_20260926.md)，未改写其结论。输入哈希和调用信息分别见同目录 `DEADLINE_AUDIT_INPUTS_20260926.json`、`CLAUDE_DEADLINE_AUDIT_PROVENANCE_20260926.json`。

我完整阅读了 Claude 的报告，并返回当前稿件、冻结实验实现、结果文件和引用原文逐项核查。Claude 的原判定为 NON-BLOCKING CRITICAL ISSUES，共两项；按用户要求的高 precision 标准，最终接受为 Critical 的数量为 **0**。两项均降级为非阻塞的局部完善建议。没有应用 Claude 的 patch，没有修改论文、图、实验记录或运行中的训练。

| Claude 项目 | 最终裁定 | 理由 |
|---|---|---|
| C1：低拒绝率与组件收益措辞矛盾 | 不接受为 Critical；可选措辞澄清 | 低拒绝率不推出更新量几乎相同，更不构成与较高实测分数的数学矛盾。当前文字明确限定于本次单种子训练比较。 |
| C2：模型、数据集和 vLLM 缺正式引用 | 降为引用完善建议 | 模型和数据集身份已经明确，附录还给出发布者及固定 revision；未发现错误引用、虚构来源或错误归属。正式引用可以补，但当前遗漏尚不满足本轮 material Critical 门槛。 |

## C1 的逐项复核

**Location / original content**

- `tex/main/paper-body.tex:481`：`Sign-dependent scaling improves training avg@8 in this run`。
- `tex/main/paper-body.tex:531–537`：比较单独组件和组合，明确写有 `in this run` 和 `within-study training comparisons under a common budget`。
- `tex/main/paper-body.tex:543–547`：如实报告 Pro-only / Pro Max 的拒绝率为 0.00030% / 0.00056%，并称为 low-rejection regime。
- `tex/main/paper-body.tex:561–573`：说明一组一个 seed、在线训练指标，并明确不建立方法的一般排序。
- `tex/main/paper-abstract.tex:15–17`：`reports higher online training accuracy` 是对观测结果的描述，没有声称普遍有效或因果机制已被验证。

**核查结果**

1. 低拒绝率本身真实，分数差也真实；两者可以同时成立。拒绝 token 的比例不是被删除损失、梯度范数或参数更新差的上界。尤其 mask 涉及重要性权重，少数 token 可能有不同的贡献大小；之后训练轨迹和采样也会改变。
2. Baseline 的反事实拒绝率与 Pro-only 接近，只说明该统计量接近，不能说明两组实际参数更新接近。它们并非在每一步共享同一策略、同一批响应、同一组梯度的配对计算。
3. 第一步采样分数不同说明样本不完全相同；这提醒我们不能把单次结果当作稳定效应估计，但不推翻稿件明确限定于本次运行的数值比较。
4. Claude 原 patch 加入 `the Pro-only and Baseline updates differ only marginally`。**这一句没有相应更新量证据，不应加入论文。** 它反而会引入用户本轮要求排除的无证据断言。
5. 单种子不足以确认组件的稳定收益属于 General。当前稿件没有声称显著性、跨种子稳定性或普遍规律；不把这个 General 问题升级为明确的数据矛盾。

**最小建议（可选，不是必须修复）**

若希望进一步减少“组件作用已经被识别”的歧义，只需把下面两处改成观察性表述，无需新增免责声明，也不要改数据或结论：

```diff
--- a/tex/main/paper-body.tex
+++ b/tex/main/paper-body.tex
@@
- Sign-dependent scaling improves training avg@8 in this run; the final-window PPL Gap remains above Baseline.}
+ Max-only has higher mean training avg@8 in this run; the final-window PPL Gap remains above Baseline.}
@@
-The component comparison distinguishes individual gains from a gain due
-to combining the methods. Pro-only and Pro Max have nearly equal
+The comparison reports scores for each component and their combination.
+Pro-only and Pro Max have nearly equal
```

这些修改只是收窄可能的阅读歧义，不代表接受 Claude 所称的“内部数据矛盾”。本轮没有应用此 patch。

## C2 的逐项复核

**Location / original content**

- `tex/main/paper-body.tex:425–431`：准确列出 Qwen2.5-Math-1.5B、DAPO-Math-17k-dedup、vLLM、ZeRO-3。
- `tex/main/core-appendix.tex:1469–1470`：明确数据来自 `YouJiacheng/DAPO-Math-17k-dedup`，revision 为 `6e26a33abdabd3e6aaa1d742326b790758f7dbc5`。
- `tex/main/reference.bib` 已有 `yu2025dapo`、`kwon2023efficient`，但正文未调用它们。

**核查结果与最小建议**

Claude 将“未添加正式 bibliographic citation”表述成“没有注明来源”，忽略了附录已给出的具体发布者和版本。这两件事不同。不存在虚构模型、错误数据源或无法识别所用发布版本的问题，因此降级，不作为提交阻塞项。

正式引用仍然是值得做的局部完善：模型名称后引用 [Qwen2.5-Math 技术报告](https://arxiv.org/abs/2409.12122)，DAPO 来源说明处引用 DAPO，保留 [YouJiacheng 的去重发布版本](https://huggingface.co/datasets/YouJiacheng/DAPO-Math-17k-dedup)及 revision。原始 DAPO 论文不能替代这个具体数据版本标识。vLLM 引用同样可补。若执行，需在最后一次编译中重新检查正文 9 页以及引用排版；不能默认新增 citation 不占空间。本轮未增加条目或改变排版。

## 其他意见的筛选

- **归约方式**：Claude 在 General 中指出实际训练为 rank-local token mean 后平均。附录 `core-appendix.tex:1328–1352` 已写出 response-count 与 token-count 归约差，并在 1347–1348 行明确涵盖等响应数的 rank–microbatch pairs。当前不是完全漏掉该数学区别，也不能据此宣称证明失效。可以把实验实际采用的 rank-local reduction 再写明确一句，但不需要新增理论或实验。
- **图中文字**：一次 PyMuPDF 预览出现 Figure 3 坐标文字缺失；独立 PDFium 渲染及新进程渲染显示完整。属于预览异常，排除出稿件问题。
- **必要边界说明**：有限模型、支持条件、单种子与 online training 指标的范围说明不能仅因语气保守而删除。
- **共享参数反例的端点**：附录参数族包含零作为参考点，命题明确讨论正参数变化；严格正负号按正变化理解。可加 `for 0<\delta\le1/45` 提高局部明确性，但不将其列为实质数学错误。

## Final contradiction pass

我在阅读 Claude 报告后，再次对照当前源码及独立复核结果检查了以下事项：

1. **数字与指标**：17,381 + 27 = 17,408 = 544 × 32；每组 139,264 个响应；最后 50 步 12,800 个响应。三组相对 Baseline 的全程增量 0.56 / 1.13 / 1.10 pp、末 50 步增量 1.34 / 2.00 / 0.95 pp 均与 CSV 一致。表中百分比、PPL Gap 的 ×10⁻³ 单位及 token-weighted 拒绝率一致。在线 avg@8 未冒充 pass@8 或 held-out 评测。
2. **模型和配置**：稿件仅包含完成的 Qwen2.5-Math-1.5B 四组 544 步；没有混入运行中的 7B 结果。学习率、长度、batch、seed、Pro 区间等与冻结设置一致。
3. **方法及数学**：一次 current/rollout 比率、detached causal prefix mask、被拒 token 留在分母，与冻结 loss 路径一致。归约差已在附录建模。核对了 Max 因子、Pro 目标分解、Adaptive/Mixed 的 TV/KL 界、population reward margin 与构造例子，未确认 material statement/proof 矛盾。现有精确有理数例子检查通过；它只验证例子，不代替全部证明。形式化清单检查只说明覆盖记录新鲜度，不作为语义正确性证据。
4. **摘要、贡献、结论**：经验陈述限定于本次训练比较，理论结论保留其条件，没有发现把现有实验写成普遍证明的句子。
5. **图、caption、引用**：三图的两两比较、平滑窗口、最终窗口排序与表格及正文一致。PDF 没有未解析的 `??`。DPPO/TRM 的分解与误差界来源、博客的偏差/二阶矩技术及相关方法引用经过原文交叉核对，未确认虚假引用或明显误引。不能把“补充可用引用”混同为“纠正错误引用”。
6. **内部文字**：结合上下文筛查 parent、TODO、placeholder、debugging、无必要的自我免责声明等，没有确认应按 Critical 清除的渲染文本。数据集 revision 与 AI USE STATEMENT 中的人工作者复核说明是正常语境。
7. **版面和输入完整性**：当前 PDF 共 28 页，正文结束于第 9 页，声明从第 10 页开始。三图及正文末页已视觉复核；其余页主要做文本/结构检查，没有声称逐页视觉检查。审计前后 9 个输入文件的 SHA-256 全部一致。

No additional critical inconsistencies found in the final contradiction pass

## General（仅记录，不展开）

- 单种子与仅在线训练指标的证据范围。
- 模型、数据集、工具正式引用及个别观察性措辞可进一步完善。
- 实验实际归约方式可以更直接连接到附录已有公式。
- 符号复用和相关工作覆盖属于一般完善项，不在本轮扩写。

## 建议的截止前处理

没有经复核确认的必须修复项，不建议照单应用 Claude 的整组 patch。若采用可选修改，优先仅做上述两处观察性措辞和来源引用补全，然后重新编译验证正文仍恰为 9 页、全文少于 30 页。不要加入“更新几乎相同”这样的未验证判断，不增加实验，不扩大理论，不改动既有结果。
