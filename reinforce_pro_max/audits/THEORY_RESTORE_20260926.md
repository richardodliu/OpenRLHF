# 2026-09-26 全理论投稿稿恢复记录

## 范围与结果

按用户要求恢复实验加入前的全理论稿，活动稿件中不保留实验章节、实验附录、经验结果表述或实验图表。实验数据、脚本、日志及已生成曲线作为独立研究记录保留；没有停止、重启或更改 GPU 任务。

- 投稿入口：`tex/iclr2027_submission.tex`。
- 最终 PDF：仓库根目录 `iclr2027_submission.pdf`。
- 正文恰为 9 页，第 9 页末行底部为 730.056 pt；全文 24 页，低于 30 页上限。
- 第 10 页声明顺序：AI USE STATEMENT / ETHICS STATEMENT / REPRODUCIBILITY STATEMENT。
- 模板、字号、行距、页边距及空白控制没有修改；没有添加占位内容。

## 恢复方法与内容保全

恢复来源为 `/tmp/promax_prefix_analysis_supplement.zip` 中保存的理论正文和附录，回退前状态单独备份至 `/tmp/promax_before_theory_restore_20260926_193212.zip`。

在恢复前，对比了该存档与当前全部数学环境：30 项 statement 的完整文本哈希集合一致。变化主要是实验整合时的正文/附录搬移；此次将二元系数推导、长度协方差、总体奖励证明、正例及失败边界恢复到原来的正文位置，删除其搬移后的重复位置。没有通过新增定理填补篇幅。

摘要末尾的四组实验结果、正文 Experiments、Limitations/Conclusion 中的经验描述、实验配置与聚合附录，以及复现声明中的实验补充材料说明均已移除。活动 PDF 不含 avg@8、PPL Gap、Qwen、DAPO、H200 或实验章节。

原 `tex/figures/training_*_vs_baseline.pdf` 三个实验图迁至 `reinforce_pro_max/experiments/dapo_small/figures/math15b_submission_archive/`。它们不再属于投稿源码依赖，也不进入理论补充包。

## 验证

1. `bash compile_iclr.sh` 成功；无未定义引用、重复标签、未收敛引用或 Overfull 错误。本次独立临时编译目录已由构建脚本清理，只导出根目录 PDF。
2. 对新 PDF 与恢复来源中的理论 PDF 逐页进行同一渲染器像素比较，24 页全部一致。
3. 视觉检查正文及附录接触表，并单独检查第 9–10 页：正文页底结束，声明顺序正确，未发现新增明显空白或排版异常。
4. `python reinforce_pro_max/check_submission_readiness.py` 通过：6 个 TeX 输入、75 个标签、16 个引用 key。此检查仅验证依赖及引用闭合。
5. `formal/PAPER_COVERAGE.json` 按数学文本哈希迁移文件位置和行号，保留原来的 review scope 与 Lean evidence；`python formal/check_paper_coverage.py` 通过，仍为 30 项。索引检查不代表新的语义证明。
6. `python reinforce_pro_max/check_two_policy_examples.py` 的 7 项精确有理数检查通过，不替代一般参数证明。
7. 新理论补充包 `/tmp/promax_theory_only_20260926.zip` 包含 67 个 payload 文件及 manifest；逐文件哈希验证通过，不含实验目录或实验图。51 个 Lean 源文件与前一已审计补充包逐字节一致，因此本轮未重复 Lean 构建，也未宣称完成新一轮形式化审计。
8. 定向 `git diff --check` 通过。既有其他工作区修改保留，没有 commit 或 push。

当前 PDF、补充包 SHA-256 和精确页数见 `reinforce_pro_max/current_theory_validation.json`。本轮是用户授权的结构恢复与格式验证，不重新判断录用可能性；此前 Claude 审计仍保留其原始快照范围。
