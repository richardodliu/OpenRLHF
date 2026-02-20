# REINFORCE Pro Max：论文设计与结构说明

本文档用于描述论文 **REINFORCE Pro Max** 的整体设计、章节结构、符号与宏约定，以及主要理论结论（定理/引理/命题）的陈述与证明分布位置。

> 本文档由历史 planning 资料（原 `tex/plan/` 下的 Markdown）整理合并而来，并以当前论文实际代码与结构为准：`tex/main.tex` 与 `tex/main/*.tex`。

---

## 1. 文档定位与维护原则

### 1.1 读者与用途

- **读者**：论文作者/合作者、未来维护者、需要对照实现的工程同学。
- **用途**：
  - 快速理解论文的“结构骨架”（每个 section 放什么、为什么这么组织）。
  - 快速定位“某个定义/定理/证明”在哪个 `.tex` 文件。
  - 确保符号体系与实现/参考文献保持一致，减少记号漂移。

### 1.2 不变项（不改科学结论）

- **不改变**论文的核心科学结论与贡献逻辑（方法实现以仓库中的代码与 `reinforce_pro_max/` 设计文档为准）。
- **允许改变**表达方式：符号规范化、证明步骤补全、结构重排（在不改变结论的前提下）。

---

## 2. `tex/` 目录布局（source of truth）

### 2.1 入口与公共配置

- `tex/main.tex`：论文入口（title/abstract + `\input{main/...}`）。
- `tex/env.tex`：LaTeX 包、超链接、列表/算法等环境配置（含 theorem 环境定义）。
- `tex/math.tex`：数学记号与常用宏（policy、divergence 等）。
- `tex/main/reference.bib`：BibTeX 数据库（`plainnat` + `natbib`）。
- `tex/literature/`：参考材料（gold standard），用于对齐记号与证明风格。

### 2.2 章节拆分文件

所有正文与附录均拆分在 `tex/main/` 下：

- `tex/main/1-intro.tex`：Introduction
- `tex/main/3-preliminaries.tex`：Preliminaries（含 Notation/Assumptions/Error decomposition）
- `tex/main/4-method.tex`：方法主体（REINFORCE Max + REINFORCE Pro）
- `tex/main/5-theory.tex`：Unified Framework（算法伪代码、统一视角、对比表）
- `tex/main/2-related.tex`：Related Work
- `tex/main/6-experiment.tex`：Experiments（结构性验证）
- `tex/main/7-conclusion.tex`：Conclusion
- `tex/main/appendix.tex`：Appendix（完整证明与技术补充）

> 注意：`tex/main/` 的文件名前缀数字是历史遗留，**不保证**与 LaTeX 的实际 section 编号一致；章节顺序以 `tex/main.tex` 的 `\input{...}` 为准。

---

## 3. 论文结构（以 `tex/main.tex` 为准）

### 3.1 阅读顺序与文件映射

| 阅读顺序 | 章节标题（`\section{...}`） | label | 文件 | 主要职责 |
|---:|---|---|---|---|
| 0 | Abstract | — | `tex/main.tex` | 给出问题、两大挑战、两组件与主要理论主张 |
| 1 | Introduction | `sec:introduction` | `tex/main/1-intro.tex` | 动机 + 两大挑战 + 贡献点与组织结构 |
| 2 | Preliminaries | `sec:preliminaries` | `tex/main/3-preliminaries.tex` | 统一符号、假设、error decomposition 与信赖域背景 |
| 3 | REINFORCE Max: Variance-Reduced Advantage Estimation | `sec:reinforce-max` | `tex/main/4-method.tex` | RLOO baseline + token expansion + 自适应非对称归一化 |
| 4 | REINFORCE Pro: Causal Off-Policy Correction | `sec:reinforce-pro` | `tex/main/4-method.tex` | prefix cumulative IS + tighter masking + trust-region proxy |
| 5 | The Unified Framework | `sec:unified` | `tex/main/5-theory.tex` | 统一视角、算法伪代码（Algorithm）、方法对比表 |
| 6 | Related Work | `sec:related` | `tex/main/2-related.tex` | critic-free RL 与 off-policy/信赖域相关工作定位 |
| 7 | Experiments | `sec:experiments` | `tex/main/6-experiment.tex` | 用受控模拟验证结构性主张（非大规模 benchmark） |
| 8 | Conclusion | `sec:conclusion` | `tex/main/7-conclusion.tex` | 总结贡献、范围边界、未来工作 |
| 9 | Appendix | — | `tex/main/appendix.tex` | 完整证明与补充：RLOO、$\gamma\!=\!1$、prefix theorem、uniform scale 等 |

### 3.2 结构设计动机（为何这样排）

- **Preliminaries** 先把三套 policy、两种 ratio（PPO ratio vs rollout mismatch）与 error decomposition 说清楚，避免后文“同名不同义”。
- 方法部分在同一文件 `4-method.tex` 内按“Max → Pro”组织，便于读者先理解 advantage 端的方差控制，再理解 off-policy 端的因果过滤。
- **Unified Framework** 单独成章，把两组件如何对应到 error decomposition 的两个因子进行统一叙述，并给出可执行的算法伪代码。

---

## 4. 核心设计（Method-level）

### 4.1 两大挑战（论文的“问题定义”）

1. **高方差 advantage 估计**：critic-free 场景下 advantage 只能来自 reward；均值 baseline 与全局归一化在“高度偏斜 reward”下会引入不稳定（符号翻转、尺度扭曲）。
2. **rollout 与训练的 off-policy mismatch**：推理引擎（如 vLLM）与训练框架的策略不一致，且误差会沿自回归因果结构累积放大。

### 4.2 REINFORCE Max（控制 advantage 因子）

由三步组成：

- **RLOO baseline**：对 prompt group（同一 prompt 的 $n$ 个 sample）做 leave-one-out baseline，避免样本与自身 baseline 相关（便于二阶矩分析）。
- **Token expansion**：在 sparse reward 且 $\gamma=1$ 时，$A_{i,t}$ 在序列内可取常数 shaped reward。
- **Adaptive asymmetric normalization**：用正/负不同缩放 $\alpha,\beta$，在不改变符号的前提下对非零 token 满足经验均值/方差约束，避免全局标准化引发的 sign flip。

### 4.3 REINFORCE Pro（控制 context shift / mismatch 因子）

核心是 **prefix cumulative IS**：

- 定义 per-token mismatch log-ratio：$\ell_t=\log\piold(y_t|c_t)-\log\piroll(y_t|c_t)$。
- 用 prefix 平均（几何均值的 log）构造因果统计量：$\exp(L_t/P_t)$，并据阈值区间 $[\lambda,\Lambda]$ 做 **prefix mask**。
- 与 token-level / seq-level 方法比较：prefix 过滤能在“早期偏离但后续单步都不出界”的模式下屏蔽更多 off-policy token（在明确充分条件下）。
- 与 trust-region 的连接：通过 KL chain rule 给出“prefix log-ratio 与 prefix KL”的期望恒等式，并把阈值化解释为 per-position sample-level proxy。

---

## 5. 理论结论与证明地图（Theorem/Proof Map）

下表用于快速定位“陈述在哪、证明在哪、作用是什么”。**label 以论文源码为准**。

| label | 类型 | 陈述位置 | 完整证明/补充位置 | 作用 |
|---|---|---|---|---|
| `eq:error-decomp` | Equation | `tex/main/3-preliminaries.tex` | — | 将 surrogate 误差分解为 advantage 因子 × context shift 因子 |
| `sec:assumptions` | Section | `tex/main/3-preliminaries.tex` | — | 支撑 error decomposition 与 IS/TV/KL 讨论的假设范围 |
| `def:rloo` / `eq:rloo-baseline` | Definition/Eq | `tex/main/4-method.tex` | `app:rloo-proof` | 定义 leave-one-out baseline 与 shaped reward |
| `prop:rloo-variance` | Proposition | `tex/main/appendix.tex` | 同处（含二阶矩分解） | 给出 unbiasedness、baseline independence、二阶矩分解等性质 |
| `sec:token-expand` / `eq:token-advantage` | Section/Eq | `tex/main/4-method.tex` | `app:gamma-one` | 说明 sparse reward 下 token expansion 的合理性（$\gamma=1$） |
| `def:adaptive-norm` | Definition | `tex/main/4-method.tex` | — | 定义 $\alpha/\beta$ 非对称归一化与经验约束（mean=0, var=1） |
| `eq:alpha-beta` / `prop:alpha-beta` | Eq/Proposition | `tex/main/4-method.tex` | — | 给出 $\alpha,\beta$ 的闭式解与等价约束写法 |
| `prop:gradient-direction` | Proposition | `tex/main/4-method.tex` | — | 证明该归一化保持梯度方向/不引入符号翻转 |
| `def:prefix-is` | Definition | `tex/main/4-method.tex` | — | 定义 prefix cumulative IS（action mask-aware 的 $L_t,P_t$） |
| `thm:prefix-tighter` | Theorem | `tex/main/4-method.tex` | `app:prefix-proof` | 在明确充分条件下，对比 token/seq 方法给出 tighter masking 结论 |
| `lem:cumsum-kl` | Lemma | `tex/main/4-method.tex` | — | KL chain rule 形式：prefix log-ratio 期望与 prefix KL 的恒等式 |
| `rem:prefix-proxy` | Remark | `tex/main/4-method.tex` | — | 将 prefix thresholding 解释为 trust-region style proxy（范围限定） |
| `cor:per-position-trust` | Corollary | `tex/main/4-method.tex` | — | 把 prefix 阈值化写成 per-position 的 sample-level 约束 |
| `eq:promax-loss` | Equation | `tex/main/4-method.tex` | — | 给出 Pro Max 的 loss 结构（mask × detached IS × PPOClip） |
| `alg:promax` | Algorithm | `tex/main/5-theory.tex` | — | 全流程伪代码，明确 rollout / baseline / norm / prefix mask / loss |
| `app:gamma-one` | Appendix section | `tex/main/appendix.tex` | 同处 | 形式化说明为何 $\gamma$ 取 1（与 sparse reward/推理任务一致） |
| `app:uniform-scale` | Appendix section | `tex/main/appendix.tex` | 同处 | uniform reward 组的可选 uniform scale 技巧与梯度方向解释 |

---

## 6. 符号、宏与引用规范（Notation & Macros）

### 6.1 统一符号（见 `sec:notation`）

论文把关键符号集中在 `tex/main/3-preliminaries.tex` 的 `\subsection{Notation}`：

- prompt：$x$；第 $i$ 个 rollout：$y^{(i)}$；位置上下文：$c_t=(x,y_{<t})$。
- 三套 policy：`\piroll`（rollout）、`\piold`（old actor）、`\pitheta`（current actor）。
- 两类 ratio：
  - PPO ratio：$\rho_t^{\mathrm{PPO}}=\pitheta/\piold$（**带梯度**）
  - mismatch ratio：$w_t=\piold/\piroll$，$\ell_t=\log w_t$（**作为 detached 系数/过滤统计量**）

### 6.2 宏的唯一来源

- 新增/修改数学符号：优先在 `tex/math.tex` 增补，并在正文中使用宏（避免各处临时定义导致不一致）。
- theorem/lemma/definition 等环境：统一在 `tex/env.tex` 定义。
- `tex/math_commands.tex`：历史遗留宏集合，包含 `\usepackage` 与高风险重定义（如覆盖 `\eqref`）。**不要直接 `\input{math_commands.tex}`**；若确需其中某些宏，请“挑选并迁移”到 `tex/math.tex` 后再使用。

### 6.3 交叉引用与 bib

- 交叉引用：使用 `\Cref{...}`（`cleveref`）统一格式。
- 文献引用：`natbib`，BibTeX 数据库为 `tex/main/reference.bib`。

---

## 7. 论文与实现对应关系（Paper ↔ Code）

论文是理论与结构性解释，工程实现与参数语义主要在以下位置对照：

### 7.1 设计文档（实现语义的“解释层”）

- `reinforce_pro_max/REINFORCE_MAX.md`：REINFORCE Max（RLOO + adaptive norm + uniform_scale）工程语义说明
- `reinforce_pro_max/REINFORCE_PRO.md`：REINFORCE Pro（prefix cumulative mask）工程语义说明

### 7.2 关键代码入口（实现层）

- `openrlhf/trainer/ppo_utils/experience_maker.py`：advantage estimator（含 `reinforce_max`、RLOO、归一化等）
- `openrlhf/models/loss.py`：vLLM IS correction（含 `reinforce_pro` 的 prefix cumulative mask）
- `openrlhf/cli/train_ppo_ray.py`：命令行参数（`--advantage_estimator reinforce_max`、`--vllm_is_correction_type reinforce_pro` 等）

> 维护建议：当论文符号/阈值/mask 定义变更时，优先检查上述三个路径是否仍一致；若不一致，应以实现与 `reinforce_pro_max/*.md` 为准更新论文表述。

---

## 8. 构建与发布（编译、零 warning、自动推送）

- 推荐编译入口：仓库根目录 `compile.sh`（执行 `pdflatex → bibtex → pdflatex` 多轮直到引用稳定）。
- 约束：脚本会在日志中检测 `LaTeX Warning / Package Warning / Overfull|Underfull \\hbox / pdfTeX warning / BibTeX Warning--`，若出现则退出失败（避免把 warning 版本推到远端）。
- 若只想本地编译、不提交不推送：`PUSH=0 bash compile.sh`

---

## 9. 变更说明（由 `tex/plan/` 合并而来）

历史上 `tex/plan/` 下存在两类 planning 文档：

1. **工程拆分计划**：把单个 `main.tex` 拆成 `tex/main/*.tex` 子文件。该目标已经完成，当前结构以 `tex/main.tex` 为准。
2. **论文写作大纲**：早期版本曾引用旧文件名（如 `reinforce_pro_max.tex`）与旧章节划分。其“结构化表达与定理地图”的有用部分已吸收到本文档中，过时内容已移除。

从现在起，论文设计与结构的唯一维护入口为本文件：`tex/PAPER_DESIGN.md`。

