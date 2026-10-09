# REINFORCE Pro Max 研究入口

当前投稿稿为全理论版本，不含实验章节或经验结果。实验仅作为独立研究记录保留：[实验目录](experiments/dapo_small/README.md)。


当前主线是用户设计的 REINFORCE Max 与 REINFORCE Pro 算法；数学围绕组件正确性与组合更新的条件有效性组织。
当前理论稿已按用户确认的“达到可投稿标准并完成有依据的审稿式评估”完成验收；结论、证据和审稿风险见 [投稿状态](SUBMISSION_AUDIT.md)。有限模型证明和 CPU 检查不构成训练性能或实际录用保证。

| 入口 | 内容 |
| --- | --- |
| [论文结构与构建](../tex/PAPER_DESIGN.md) | 唯一投稿稿及正文、附录导航 |
| [投稿状态](SUBMISSION_AUDIT.md) | 当前检查、已修正文句与待解决问题 |
| [数学审计](../formal/PROOF_AUDIT.md) | 定理映射、反例、尚未覆盖的接口 |
| [逐项覆盖清单](../formal/PAPER_COVERAGE.json) | 当前陈述与 Lean 的语义核对进度；未录入不等于未证明 |
| [Max](REINFORCE_MAX.md) | RLOO、归一化、safeguards、uniform scale |
| [Pro](REINFORCE_PRO.md) | causal prefix mask 与一次 current/rollout token 比率 |
| [实验记录](experiments/dapo_small/README.md) | 独立研究记录，不属于当前投稿稿 |

structural_experiment.py 是固定 seed
合成检查及原始结果。保留其复现依据，不把它当作训练收益证据。
四个 test_*.py 文件覆盖实际实现和结构检查的统计口径。

从仓库根目录检查活动源码依赖和引用：

~~~bash
python reinforce_pro_max/check_submission_readiness.py
~~~

CPU 回归检查（需项目依赖）：

~~~bash
python -m pytest -q reinforce_pro_max/test_reinforce_max.py \
  reinforce_pro_max/test_normalization_safeguards.py \
  reinforce_pro_max/test_policy_loss_contract.py \
  reinforce_pro_max/test_structural_experiment.py
~~~

形式化构建：

~~~bash
LEAN_JOBS=3 bash formal/leanw build
~~~

当前稿件的可移植证明补充包可用 `python formal/build_submission_supplement.py /path/to/new.zip` 生成（先重建论文并核对清单）。补充包内提供 `python verify_manifest.py` 与 `lake build PaperAudit` 复现入口；只包含当前证明依赖及实现参考文件。最近的从零项目编译与公理审计记录见 [投稿状态](SUBMISSION_AUDIT.md)。

保留形式化反例及失败旧命题的解释，避免再次引入错误结论。
formal/.lake/packages 是当前构建依赖缓存，不是过时研究产物。

## 清理记录（2026-09-23）

已移除以下 10 个被替代或会造成误导的文件：

| 文件 | 原因 |
| --- | --- |
| `tex/iclr_main_draft.tex`、`tex/iclr_main_draft.pdf` | 旧压缩稿；当前投稿入口为 `tex/iclr2027_submission.tex` |
| `tex/plan.md`、`tex/plan/1.md`、`tex/plan/2.md`、`tex/plan/3.md` | 过时计划；当前结构和待办集中在 PAPER_DESIGN 与 SUBMISSION_AUDIT |
| `tex/math_commands.tex` | 活动论文入口不再依赖的旧宏文件；保留当前样式目录中的宏 |
| `REINFORCE_MAX_review.md`、`REINFORCE_PRO_review.md` | 旧版本审稿笔记；当前数学问题与范围以 PROOF_AUDIT 为准 |
| `test_n_design.py` | 使用旧保护参数及 fallback，并据此输出缺乏支持的训练稳定性结论 |

清理前备份位于仓库外
`/volume/pt-train/users/rbliu/workspace_backups/OpenRLHF/20260923T074935Z/`。
其中 `before-cleanup.tar.gz` 的 285 个文件已按 `manifest.json` 逐一核验大小及 SHA-256；
同时保留当时的 HEAD、工作区状态和补丁，可追溯删除前内容。

当时两份论文的输入、标签和文献引用检查通过；现仅维护下述唯一投稿版。该检查不证明数学正确性。
本次收尾未运行实验，也未重新运行 Lean 或训练回归测试。

投稿状态文档已精简，重复的历史构建过程移出当前状态页；精简前原文的仓库外位置见 [投稿状态](SUBMISSION_AUDIT.md)。详细证明映射和反例仍在 [数学审计](../formal/PROOF_AUDIT.md)，未作为过时产物删除。

### 旧 Lean 构建备份归档

两份历史构建备份已从 `formal/.lake/` 移至仓库外：

`/volume/pt-train/users/rbliu/workspace_backups/OpenRLHF/20260923T133302Z-lean-cache-cleanup/`

归档目录保留原相对路径，并以 `manifest.json` 逐文件核对大小和 SHA-256 后删除仓库内副本。
数学审计文档中的历史日志路径已更新。当前 `.lake/build`、`.lake/packages`、
全部 Lean 源码、论文源码和 PDF 均保留；归档不改变已有证明结论。

## 算法核心精简（2026-09-24）

已移出未实现 gate、范围外测度扩展及额外浓度假设链；旧“全部 Lean 源码均保留”描述属于上次清理历史，不再代表当前状态。备份和当前依赖验证见 SUBMISSION_AUDIT.md。不再以辅助定理数量作为研究进度指标。

## 当前论文结构（2026-09-24）

仅维护匿名 ICLR 投稿版，当前共 24 页，正文在第 9 页底部结束；paper-body.tex 和 core-appendix.tex 承载正文与证明，算法伪代码位于正文。署名版本及其专用构建文件已删除，使用 bash compile_iclr.sh 构建。原完整章节副本与扩展证明已归档，当前以 PAPER_DESIGN.md、SUBMISSION_AUDIT.md 为准。旧正文实验章节不进入当前投稿入口，合成检查脚本保留，旧结果已清理。
