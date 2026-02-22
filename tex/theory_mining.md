# TRM/DPPO Theorem Mining Log

本文件记录对 `tex/literature/` 中 TRM 与 DPPO 参考材料的理论内容梳理结果，用于：

- 学习其 **定理设计方式、证明组织结构、与排版美学**（TRM/DPPO 作为 gold standard）。
- 识别可用于本文 `Prefix Causal Trust Region` 叙事的 **building blocks** 与 **proof techniques**。
- 对每个候选结果做 U/O/C 分类门禁，避免重复、避免矛盾、避免符号替换式搬运（学术合规红线见 `tex/plan.md`）。

注意：

- 这里的条目用于“研究与改稿”，不是本文正文的一部分。
- 任何来自参考论文的结论若进入论文正文，必须遵循：
  - 纯引用：statement-only + citation（不复现 proof）。
  - 本文延伸：必须明确新增点/差异点，并给出本文原创 proof（通常放附录）。

---

## 0. 分类规则（U/O/C）

- **Class U (Unique and Useful)**：可作为灵感来源，最终在本文中形成 *与本文方法强绑定* 的新结论（本文原创证明）。不能直接搬运原结论。
- **Class O (Overlapping)**：与已有结论等价或属于通用 building block，只用于引用或改写本文 proof 的组织方式，不作为本文新增命题。
- **Class C (Conflicting or Risky)**：与本文设定冲突、容易误用、或会诱发“强断言但不可证”的结果；禁止引入为正文承重结论。

---

## 1. TRM（Trust Region Masking for Long-Horizon LLM-RL）

### TRM-1：Advantage Bound（building block）

- 参考来源与位置：`tex/literature/TRM/main_arxiv.tex`，Lemma “Advantage Bound”，约 L196+（正文）与 L593+（附录 proof）。
- 结果类型：Lemma（building block）。
- 原文功能：给出 $\|g_t\|_\infty$ 的上界，用于 error decomposition 的 advantage factor。
- 关键证明技巧标签：martingale / tower property / bounded reward。
- 与本文挂钩点：对齐本文 `Preliminaries` 中 error decomposition 的 advantage factor 解释口径。
- 归类：**O**。
- 本文落点：引用性说明即可（不新增 theorem）；若需要，放在 `tex/main/5-theory.tex` 的 cited supporting results（statement + citation）。
- 优势映射：用于解释“我们主要通过 advantage estimator 稳定 advantage factor”，但不构成本文创新点。

### TRM-2：Context Shift（building block）

- 参考来源与位置：`tex/literature/TRM/main_arxiv.tex`，Lemma “Context Shift”，约 L201+（正文）与 L618+（附录 proof）。
- 结果类型：Lemma（building block）。
- 原文功能：界定 $\|d_t^{\pi_\theta}-d_t^{\pi_{\mathrm{roll}}}\|_{\mathrm{TV}}$ 的多条上界路线（coupling / Pinsker / data processing）。
- 关键证明技巧标签：coupling / KL chain rule / Pinsker / data processing。
- 与本文挂钩点：对应本文 prefix-level filtering 的“因果传播”叙事，解释 token-level 与 seq-level gate 的结构性缺陷。
- 归类：**O**。
- 本文落点：作为风格参考 + 引用 building block；避免在本文复现证明。

### TRM-3：Adaptive bound proof structure（proof technique）

- 参考来源与位置：`tex/literature/TRM/main_arxiv.tex`，Adaptive bound 推导说明约 L783+，并显式以四步组织。
- 结果类型：Proof technique（组织结构）。
- 原文功能：用 Step-based proof 把长推导拆成可审稿的“承重中间量 + route A/B + per-position min”结构。
- 关键证明技巧标签：step decomposition / per-position min / importance ratio decomposition。
- 与本文挂钩点：用于重写/增强本文 appendix 中与 prefix causal gate 相关的 proof aesthetics（Step/Bound/Case）。
- 归类：**O**。
- 本文落点：仅用于改写本文 proof 的结构与语言，不作为新结论迁移。

---

## 2. DPPO（Rethinking Trust Region in the LLM Regime / Divergence Proxies）

### DPPO-1：TV as Expectation of Ratio Residual（identity）

- 参考来源与位置：`tex/literature/DPPO/paper/method.tex`，Eq.~(tv_as_expectation)，L18–24（见 `\label{eq:tv_as_expectation}`）。
- 结果类型：Identity。
- 原文功能：解释 PPO ratio clipping 是对 TV divergence 的单样本 noisy proxy。
- 关键证明技巧标签：importance ratio as Radon–Nikodym derivative / expectation identity。
- 与本文挂钩点：对应本文 `tex/main/5-theory.tex` 的 `rem:tv-ratio-identity`（仅 statement + citation）。
- 归类：**O**（引用型 building block）。
- 本文落点：`tex/main/5-theory.tex`（cited supporting results / comparisons）；不在 Preliminaries 放 proof。
- 优势映射：为本文“prefix gate 是 ratio-based causal proxy”的解释提供出发点。

### DPPO-2：Bound on Sequence-Level TV Divergence（lemma）

- 参考来源与位置：`tex/literature/DPPO/paper/app.tex`，Lemma `lem:sequence_tv_bound`，L60–89。
- 结果类型：Lemma（building block）。
- 原文功能：把 sequence-level TV 上界为 per-step TV 的期望和（用于 finite-horizon remainder 控制）。
- 关键证明技巧标签：telescoping product identity / triangle inequality / integrate-out future。
- 与本文挂钩点：对应本文 `tex/main/5-theory.tex` 的 `rem:seq-tv-sum`（只陈述用途，不复现 proof）。
- 归类：**O**。
- 本文落点：statement-only + citation（不在本文附录复现 DPPO proof）。

### DPPO-3：Coarse-Graining Lower Bound for TV/KL（technique + statement）

- 参考来源与位置：`tex/literature/DPPO/paper/app.tex`，Section “Approximations as Lower Bounds of True Divergence”，约 L160+（该文件后半段）。
- 结果类型：Statement + proof technique（triangle inequality / log-sum inequality）。
- 原文功能：证明 Binary/Top-$K$ divergence 是 full-vocab divergence 的下界，并给出 equality 条件与 gap 上界思路。
- 关键证明技巧标签：triangle inequality / log-sum inequality / partitioning。
- 与本文挂钩点：对应本文 `tex/main/5-theory.tex` 的 `rem:coarse-grain-lower`、`rem:tv-gap`、`rem:kl-equality`。
- 归类：**O**。
- 本文落点：仅引用 statement + citation；本文新增内容必须是“prefix-causal gate 的新结论”，不能复述这些定理。

### DPPO-4：LLM regime trust-region remainder bound（theorem family）

- 参考来源与位置：`tex/literature/DPPO/paper/llm_bound.tex`，Theorem “Performance Difference Identity for LLMs” 与 “Policy Improvement Bound for LLMs”。
- 结果类型：Theorem family（LLM finite-horizon, $\gamma=1$）。
- 原文功能：在 LLM 设定下重写 PDI 并控制 remainder。
- 关键证明技巧标签：telescoping identity / conditioning / bounding future divergence。
- 与本文挂钩点：本文已采用类似 error decomposition 框架；DPPO 的价值主要在 proof technique 与叙事结构，而不是直接迁移定理。
- 归类：**O**（避免重复与结论等价的再陈述）。
- 本文落点：如需使用，仅在 Preliminaries/Related Work 做引用性陈述（statement + citation），不复制 proof。

