# Overall verdict

**NON-BLOCKING CRITICAL ISSUES**

（这个判断只针对截止前能否局部修复，不代表对是否录取的判断。）

---

## 审计对象与范围

- **读过的源文件（全文）**：`tex/iclr2027_submission.tex`、`tex/main/paper-abstract.tex`、`tex/main/paper-body.tex`（574 行）、`tex/main/core-appendix.tex`（1507 行）、`tex/main/reference.bib`、`tex/iclr2027/math_extra.tex`。
- **PDF**：用 pymupdf 读了根目录 `iclr2027_submission.pdf`。共 28 页，正文到第 9 页的 Conclusion 结束，没有 `??` 未解析引用。抽查的正文和表格文字与当前源码一致，元数据中没有作者信息。
- **三张图**：把 `tex/figures/training_{max_only,pro_only,pro_max}_vs_baseline.pdf` 渲染成 PNG（放在 /tmp）后逐张看过。其余 PDF 页只做了文本抽取，没有逐页做视觉检查。
- **实验权威数据**：`results/math15b_544/{paper_settings,paper_summary,training_avg8_summary,plan}.json`、`paper_curves.csv`、四个 arm 的 `curve.csv` 和 `training_avg8.csv`，以及 `plot_paper_training.py`。另外只读核对了冻结源码中的 `loss.py`、`study_diagnostics.py`、`experience_maker.py` 和 `reward_func.py`。
- **引用核对**：
  - 本地原文：TRM `main_arxiv.tex`、DPPO `paper/app.tex`、Li & Liu 博客 Part 1–3。
  - 在线核对（arXiv 或 HF 页面）：2512.23075v4、2602.04879、2605.07331、2601.22718、2608.01418、2606.10968、2601.23135、2607.23364、2409.12122、HF `YouJiacheng/DAPO-Math-17k-dedup`，以及博客 Part 3 页面。
- **没有做的事**：没有修改任何文件，没有编译，没有训练，也没有做 git 操作。

---

## Critical issues（按严重程度排列）

### C1. 把 Pro-only 相对 Baseline 的提升写成 masking 的组件收益，与论文自己报告的拒绝率矛盾

**Location**
- `tex/main/paper-body.tex`
  - §6.2 "Training accuracy and component effects"，L531–537
  - §6.2 "Mismatch and masking activity"，L543–548
  - Figure 1 caption L479–481，Figure 3 caption L497–499
  - Conclusion L570–573
- `tex/main/paper-abstract.tex` L15–17（最后一句）

**Original text**
- L531–532: "The component comparison distinguishes individual gains from a gain due to combining the methods."
- L543–545: "Across training, Pro-only and Pro Max reject only 0.00030\% and 0.00056\% of active tokens. At the chosen interval, the mask changes a small fraction of the sampled token contributions."
- Fig. 1: "Sign-dependent scaling improves training avg@8 in this run; …"
- Fig. 3: "The combined method improves avg@8 over Baseline; …"
- Conclusion L571–572: "In the single-seed training study, all variants improve avg@8 over Baseline; Pro-only has the highest final-window score."

**Problem**
1. 按论文报告的数字，Pro-only 的 mask 只拒绝了 0.00030% 的 active token。在冻结数据 `pro_only/curve.csv` 中，544 步一共只拒绝约 385 个 token，而总 token 约为 544×232k≈1.26×10⁸，平均每次 update 不到 1 个 token。
2. 同一诊断量在**没有施加 mask** 的 Baseline arm 中（`study_diagnostics.record_policy` 只记录反事实 mask，不参与 loss）为 3.15×10⁻⁶，约 391 个 token，与 Pro-only 的 3.04×10⁻⁶ 几乎相同。也就是说，Pro-only 与 Baseline 每步的 update 在数值上几乎一致。
3. 第 1 步时四个 arm 都从同一个初始模型采样，但 avg@8 已经不同：Baseline 6.64%、Max-only 7.03%、Pro-only 5.47%、Pro Max 5.47%。这说明 run 之间本身存在采样差异。
4. 在这种情况下，把 Pro-only 比 Baseline 高 1.13 / 2.00 pp 称为 masking 的 "individual gain"，并在结论中写成 "all variants improve"，没有证据支持，属于因果归因过强。
5. 论文自己也写了 "low-rejection regime"，但没有指出这一点直接削弱了对 Pro-only 收益的归因。认真的读者对照 0.00030% 这个数字就能发现这个矛盾。

**Why critical**
- 这是论文自己数据内部的矛盾，读者容易注意到。
- 它会误导读者认为 causal prefix masking 是 Pro-only 取得最高 final-window 分数的原因，从而损害实验部分的可信度。
- 只需改措辞，不改变任何数值或核心结论。

**Minimal fix**
- 把 "improve / individual gains" 改成描述性说法（"attains higher … than"）。
- 在 masking activity 段落加一句：由于拒绝率极低，并且每个 arm 只有一个 seed，不把 Pro-only 与 Baseline 的差异归因于 mask。

**Patch**
```diff
--- a/tex/main/paper-body.tex
+++ b/tex/main/paper-body.tex
@@ -479,5 +479,5 @@
  \caption{Max-only versus Baseline. Left: training--rollout PPL Gap.
  Right: online training avg@8. Both curves use trailing 20-step means.
- Sign-dependent scaling improves training avg@8 in this run; the final-window PPL Gap remains above Baseline.}
+ Max-only attains higher mean training avg@8 than Baseline in this run; the final-window PPL Gap remains above Baseline.}
  \label{fig:training-max-only}
@@ -497,3 +497,3 @@
  \caption{Pro Max versus Baseline. Panels and smoothing follow
- \Cref{fig:training-max-only}. The combined method improves avg@8 over
+ \Cref{fig:training-max-only}. The combined method attains higher avg@8 than
  Baseline; its final-window gain is smaller than Pro-only's.}
@@ -531,3 +531,3 @@
-The component comparison distinguishes individual gains from a gain due
-to combining the methods. Pro-only and Pro Max have nearly equal
+The component comparison contrasts each single change with their
+combination. Pro-only and Pro Max have nearly equal
 all-step means, 25.14\% and 25.11\%, while Pro-only has the higher
@@ -543,6 +543,9 @@
 measured gap. Across training, Pro-only and Pro Max reject only
 0.00030\% and 0.00056\% of active tokens. At the chosen interval, the
-mask changes a small fraction of the sampled token contributions.
+mask changes very few sampled token contributions, so the Pro-only and
+Baseline updates differ only marginally. With one seed per arm, we do not
+attribute the Pro-only--Baseline accuracy difference to masking.
 This distinguishes the observed low-rejection regime from the large
 mismatch constructions used to establish the theoretical bounds.
@@ -570,4 +573,4 @@
 reward-improvement margin and a failure boundary. In the single-seed
-training study, all variants improve avg@8 over Baseline; Pro-only has the
-highest final-window score. These observations complement the conditional
+training study, all variants attain higher avg@8 than Baseline; Pro-only has the
+highest final-window score despite rejecting very few tokens. These observations complement the conditional
 theory without establishing a general ordering of the methods. The comparison
```
可选的摘要改动（同样只削弱措辞）：
```diff
--- a/tex/main/paper-abstract.tex
+++ b/tex/main/paper-abstract.tex
@@ -15,3 +15,3 @@
-and reward decrease. A four-arm mathematical-reasoning study reports higher
+and reward decrease. A single-seed four-arm mathematical-reasoning study reports higher
 online training accuracy than the baseline, with the strongest final-window
 result from Pro alone.
```

**交叉验证证据**
- `paper_summary.json` 中的 `token_weighted_prefix_rejection`：
  - baseline 3.151e-6（反事实诊断，mask 未施加）
  - pro_only 3.044e-6
  - pro_max 5.582e-6
- 由 `curve.csv` 按 `(1−prefix_acceptance)×active_tokens` 重算得到的拒绝 token 数：baseline 391，pro_only 385，pro_max 675，与上面一致。
- 冻结代码 `study_diagnostics.py` 中，`applied_gate` 在 Baseline arm 下为 `"none"`，`prefix_rejected` 仍然会被计算，确认 Baseline 的数值是反事实诊断。
- 第 1 步的 avg@8 来自 `paper_curves.csv`。

---

### C2. 实验使用的基座模型和数据集没有引用来源（相关 bib 条目已存在但未被引用）

**Location**
- `tex/main/paper-body.tex` §6.1 "Training protocol"，L425–431
- `tex/main/core-appendix.tex` §F，L1469–1470
- `tex/main/reference.bib`：`yu2025dapo` 和 `kwon2023efficient` 已有条目，但全文没有 `\cite`

**Original text**
"We train Qwen2.5-Math-1.5B on a fixed selection from the public DAPO-Math-17k-dedup dataset. … We use eight H200 GPUs, vLLM rollout, DeepSpeed ZeRO-3 training, and CUDA Graph."

**Problem**
- 唯一的实验中，基座模型 Qwen2.5-Math 和训练数据 DAPO-Math-17k（dedup 版本来自 BytedTsinghua-SIA/DAPO-Math-17k，已在 HF 页面核实）都没有引用。
- `reference.bib` 已经有 `yu2025dapo`（DAPO）和 `kwon2023efficient`（vLLM），但因为全文没有引用，它们不会出现在参考文献里。

**Why critical**
- 不注明所用模型和数据集的来源，不适合作为最终投稿稿；审稿人读实验设置时很容易发现。
- 修复只需补引用，不改变任何内容。

**Minimal fix**
补 `\citep`，并新增 Qwen2.5-Math 技术报告条目（arXiv:2409.12122，已在线核实：An Yang et al., "Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement"）。

**Patch**
```diff
--- a/tex/main/paper-body.tex
+++ b/tex/main/paper-body.tex
@@ -425,3 +425,4 @@
-We train Qwen2.5-Math-1.5B on a fixed selection from the public
-DAPO-Math-17k-dedup dataset. The training file contains 17,381 eligible
+We train Qwen2.5-Math-1.5B \citep{yang2024qwen25math} on a fixed selection from the public
+DAPO-Math-17k-dedup dataset, a deduplicated release of DAPO-Math-17k
+\citep{yu2025dapo}. The training file contains 17,381 eligible
@@ -430,2 +431,2 @@
-model and use the same data file and seed 42. We use eight H200 GPUs,
-vLLM rollout, DeepSpeed ZeRO-3 training, and CUDA Graph. The learning rate
+model and use the same data file and seed 42. We use eight H200 GPUs,
+vLLM rollout \citep{kwon2023efficient}, DeepSpeed ZeRO-3 training, and CUDA Graph. The learning rate
--- a/tex/main/reference.bib
+++ b/tex/main/reference.bib
@@
+@article{yang2024qwen25math,
+  title={{Qwen2.5-Math} Technical Report: Toward Mathematical Expert Model via Self-Improvement},
+  author={Yang, An and Zhang, Beichen and Hui, Binyuan and Gao, Bofei and Yu, Bowen and Li, Chengpeng and Liu, Dayiheng and Tu, Jianhong and Zhou, Jingren and Lin, Junyang and others},
+  journal={arXiv preprint arXiv:2409.12122},
+  year={2024}
+}
```

---

## Final contradiction pass

逐项核对的证据如下（全部用只读计算完成）：

- **相同数字的一致性**
  - 17,381 + 27 = 17,408 = 544×32；544×256 = 139,264；50×256 = 12,800。
  - lr 3e-6、prompt/gen/context 长度 1024/3072/4096、train batch 256、micro batch 32、vLLM 8 个 engine、显存利用率 0.6、seed 42、区间 [0.8,1.25]、reward 取值 {1,−0.5,−1}，均与 `plan.json` 的 `original_argv` 和 `paper_settings.json` 一致。
  - 表格数字（24.01/27.73/0.719 等）和正文增量（0.56/1.13/1.10、1.34/2.00/0.95）由 `paper_curves.csv` 重算后一致。
  - 拒绝率 0.00030%/0.00056% 一致；PPL Gap 列等于 `ppl_gap_current_rollout`。
- **方法 vs 算法 vs 实现**
  - Algorithm 1 与冻结代码 `loss.py` 的 `token_is` 分支一致：detached 前缀 mask、同一个 current/rollout 比率、`max(1,·)` 计数、`masked_mean` 分母保留被拒 token。
  - Max 的 ε=1e-8、乘积上限 1e8、factor clamp [1e-8,10]、fallback 条件与 Prop. B.x 以及 `_adaptive_token_normalization_single_group` 一致。
  - uniform-scale 的判定使用无偏 std，与 Lemma 的 k(n−k)/(n(n−1)) 一致。
- **数学逐项复算**
  - 二值因子、token 标准化等价、多值反例（token mean 1/3、方差 11/36）、1-vs-9 例子（α=β=3）。
  - Prop. C 的无界权重例子（K³/(K+1)²）、半保留例子（(K+K⁻¹)/2）。
  - 正 margin 例子：E f_q=2a、Cov、E D=s·aζ/3、𝔐=aδ、B_Adap≤4D̄₁δ、19/600、二阶矩 a(1+6δ²)。
  - 反例：乘积落在 [14/15,16/15]、2(1−9δ)≥8/5、ΔS=sδ/4、ΔJ=−δ/2。
  - 其余：早期偏离阈值不等式、DPPO 的 2RT(T−1)ε² 与 TRM 的 (4/3)RδT^{3/2}、B_Mix^KL 的 2RT√(δK̄)、Thm 5.1 Step 1 的残差恒等式、Prop. 4.1 分解。
  - 以上均成立。
- **abstract / intro / conclusion 与结果的强度**：除 C1 已处理的措辞外，理论部分的主张与附录陈述一致。
- **caption 与图**：三张图的视觉检查结果与 caption 一致（final-window PPL Gap 高于 Baseline；Pro-only 后段 avg@8 较高）。
- **引用是否支持前句**
  - DPPO 的 App. B.1–B.3、TRM 的 App. C.2 / D / Sample-Based Estimators / Length-Neutral（v4 结构）、博客 Part 3 §1.3、Part 1 §2 均与引用处的论述对应。
  - CTPO / MinPRO / PNPO / CPPO / Ge / Ding 的描述与原文一致。
  - 所有被引文献均真实存在。
- **残留草稿或内部痕迹**：在 `tex/main/*.tex` 中检索 TODO/parent/placeholder/review/debug/user/Lean/draft 等词，没有命中（唯一命中的 "revision" 是数据集 revision hash）。

No additional critical inconsistencies found in the final contradiction pass

---

## General（只记录，不在本次处理）

- **符号重载**：P/N（集合、计数、signed sums、残差 N）、m（action mask 与计数）、U（期望 update 与 raw objective）、η（step size、下限 clamp、ε_num、保留比例 η±）、B（误差界、例子参数、balance mass）、g（前缀统计与梯度）、𝓛（loss 与 surrogate）。
- **相关工作缺少最接近的先例**：没有提到 Geo-Mask 和 Geo-Mask-Token-TIS（li2025sequencemasking §3、§3.5，论文已经引用了这篇的 §1.3），也没有提到 GSPO 的几何均值统计量。
- **归约方式与算法描述不同**：Algorithm 1 写的是全局 active-token mean，实验实际是 8 个 DP rank 各自取 token mean 后等权平均。附录 D 把这种情形称为 "dynamic batching"，但实验并没有开启 dynamic batching。
- **实验本身的局限**：单 seed、没有 CI、只有训练集在线指标、没有 held-out；Max 的二值理论与实验中的三值 reward 不完全对应。
- **引用版本**：TRM 引用固定为 v4，arXiv 上已有 v5（2026-06-26），节号以 v4 为准。
- **交叉引用格式**：`\Cref{sec:population-transfer}` 渲染为 "Section D"，而其他附录引用渲染为 "Appendix"，格式不统一。