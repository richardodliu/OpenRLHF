# `tex/literature/` 学习索引（算法实现、数学原理、证明路线）

最后更新：2026-02-21

这份文档用于“写/改我们自己的论文”时快速回忆并复用已有工作的理论链路，尤其是：

- rollout(采样) 与 training(训练前向) 不一致导致的 off-policy / mismatch
- importance sampling (IS) 的 bias-variance 权衡
- token-level vs sequence-level vs prefix-level 的修正与 gating/masking
- trust region 视角下的改进保证与可实现的 proxy

刻意不做的事情：

- 不总结实验设置、曲线与指标（最多给出原文的章节入口）
- 不复写长篇证明细节（只给 proof roadmap + 关键 lemma/theorem 的位置与作用）

---

## 快速索引（目录里到底有什么）


| 名称                                           | 载体                | 主题关键词                                                        | 理论/证明密度              | 入口文件                                                                                         |
| -------------------------------------------- | ----------------- | ------------------------------------------------------------ | -------------------- | -------------------------------------------------------------------------------------------- |
| TRM (Trust Region Masking)                   | paper `.tex`      | 长序列下 trust region bound 非空化，sequence-level masking           | 高                    | `tex/literature/TRM/main_arxiv.tex`                                                          |
| GSPO (Group Sequence Policy Optimization)    | paper `.tex`      | sequence likelihood ratio，sequence-level clipping，与 GRPO 的差异 | 中                    | `tex/literature/GSPO/colm2024_conference.tex`                                                |
| IcePop (Every Step Evolves / Ring-1T report) | paper `.tex`      | training-inference mismatch，token-level filtering/校正         | 中（附录有定理+证明）          | `tex/literature/IcePop/main.tex`                                                             |
| CISPO (MiniMax-M1 report, 含 `\method{}`)    | paper `.tex`      | 重要性权重截断、工程化的 RL scaling 配方                                   | 低-中（更偏 recipe）       | `tex/literature/CISPO/main.tex`                                                              |
| DPPO (Divergence PPO)                        | paper `.tex`      | 用分布散度替代 ratio clip 的 trust region，Binary/Top-K divergence 近似 | 中-高（含 theorem + proof） | `tex/literature/DPPO/example_paper.tex`                                                      |
| TIS（off-policy mismatch + truncated IS）      | tech report `.md` | sampler/learner mismatch，TIS 修正项                             | 中                    | `tex/literature/TIS.md`                                                                      |
| MIS（mismatch -> collapse + 序列级修正观点）          | tech report `.md` | token-level bias、seq-level 修正、MIS                            | 中                    | `tex/literature/MIS.md`                                                                      |
| IcePop（博客稿）                                  | tech report `.md` | MoE mismatch，IcePop objective 与直觉                            | 中                    | `tex/literature/IcePop.md`                                                                   |
| Policy Gradient Intro（LLM reasoning）         | tech report `.md` | policy gradient theorem，surrogate objective（autodiff 实现）     | 中-高（含 theorem/推导）    | `tex/literature/Brief Introduction of Policy Gradient In LLM Reasoning.md`                   |
| On-Policy Distillation (OPD)                 | tech report `.md` | OPD = reverse KL，等价 entropy-regularized RL，policy gradient   | 高（含 theorem + proof） | `tex/literature/Theory of On-Policy Distillation.md`                                         |
| Policy Entropy Convergence Note              | tech report `.md` | NPG/KL-regularized update 下策略熵变化，协方差表达式与直觉解释 | 中（推导为主）             | `tex/literature/How Does RL Policy Entropy Converge During Iteration.md`                     |
| 熵减收敛笔记（skydownacai）                      | tech report `.md` | entropy 与梯度/Reverse-KL 位移的关系（含不等式推导）                     | 中（推导+不等式证明）         | `tex/literature/RL训练中为什么熵减往往意味着训练收敛.md`                                           |
| Theory Part 1                                | tech report `.md` | SGA lemma，bias vs variance，TV vs chi^2，TRPO 连接               | 高                    | `tex/literature/Theory/1-Why Off-Policy Breaks RL An SGA Analysis Framework.md`              |
| Theory Part 2                                | tech report `.md` | Seq-IS/Token-IS 的系统性 bias-variance 分析                        | 高                    | `tex/literature/Theory/2-Applying the SGA Framework Token v.s. Sequence-level Correction.md` |
| Theory Part 3                                | tech report `.md` | Seq-MIS，Geo-Mask，hard trust region via masking               | 高                    | `tex/literature/Theory/3-Trust Region Optimization via Sequence Masking.md`                  |


---

## 推荐阅读顺序（从“问题”到“证明”）

1. `tex/literature/TIS.md`：先建立 sampler/learner mismatch 的基本模型与 TIS 修正项。
2. `tex/literature/Theory/1-Why Off-Policy Breaks RL An SGA Analysis Framework.md`：用 SGA lemma 把“训练会崩”拆成 bias 与 variance 两条主因。
3. `tex/literature/Theory/2-Applying the SGA Framework Token v.s. Sequence-level Correction.md`：理解 token-level 与 sequence-level IS 在长序列下的结构性 tradeoff。
4. `tex/literature/Theory/3-Trust Region Optimization via Sequence Masking.md`：理解 Seq-MIS 与 Geo-Mask 作为 hard trust region 的实现路径。
5. `tex/literature/TRM/main_arxiv.tex`：看最系统的 bound 家族与“为什么必须 sequence-level gate 才能让 bound 非空化”。
6. `tex/literature/GSPO/colm2024_conference.tex`：看 sequence-level ratio 与 clipping 如何落到一个可训练的 surrogate。
7. `tex/literature/IcePop/main.tex` 与 `tex/literature/IcePop.md`：看 token-level filtering 在 MoE mismatch 下的具体形式与理论化表达。
8. `tex/literature/Brief Introduction of Policy Gradient In LLM Reasoning.md`：如果需要从零把 policy gradient/surrogate objective 写清楚，这份笔记可直接复用定理与推导。
9. `tex/literature/Theory of On-Policy Distillation.md`：如果论文里涉及“蒸馏视角/OPD”，用这份笔记快速对齐目标函数与 policy gradient 形式。
10. `tex/literature/How Does RL Policy Entropy Converge During Iteration.md`：如果你需要解释“为什么/何时 policy entropy 会下降或上升”，用这里的协方差表达式给出一阶近似的定量直觉（与 NPG/KL-regularized 更新对齐）。
11. `tex/literature/RL训练中为什么熵减往往意味着训练收敛.md`：如果你需要从 entropy 角度解释“为什么训练会逐步收敛/变慢”，这里给了两条不等式（梯度范数与 Reverse-KL 位移上界）以及一段完整推导链。

---

## 统一术语与符号对照（跨文献对齐）

为了减少“同一个东西不同名字”的认知切换，后续卡片统一用下面这组抽象：

- $x$：prompt / query
- $y = (y_1,...,y_T)$：response（token 序列）
- $c_t = (x, y_{<t})$：prefix/context（LLM 生成的 state）
- 行为策略 / 采样策略：$\mu$ 或 $\pi_{\mathrm{roll}}$（rollout policy / sampler policy）
- 目标策略 / 训练策略：$\pi$ 或 $\pi_\theta$（learner policy / training policy）
- per-token ratio：$\rho_t = \pi(y_t|c_t) / \mu(y_t|c_t)$
- sequence ratio：$\rho(y) = \pi(y|x) / \mu(y|x) = \prod_t \rho_t$
- geometric mean ratio（长度归一）：$\rho_{\mathrm{geo}}(y) = \rho(y)^{1/T} = \exp( (1/T)\sum_t \log \rho_t )$

一个关键提醒（对齐 OpenRLHF 实现很重要）：

- “PPO ratio”通常指 $\pi_{\mathrm{current}} / \pi_{\mathrm{old}}$（用于 clip）
- “mismatch correction ratio”通常指 $\pi_{\mathrm{old}} / \pi_{\mathrm{roll}}$（用于纠正 rollout backend 与训练 backend 的不一致）
这两种 ratio 在实现里往往同时存在，不能混为一谈。

---

## 关系图谱：TRM vs Seq-MIS vs Geo-Mask（以及与 OpenRLHF 的映射）

这部分的目标是回答一个具体问题：**TRM 与 MIS 的关系是什么？它们是否等价？如果不等价，差异点在哪里？**

结论先行（可核查，后面给证据链）：

- TRM 与 (Seq-)MIS 都属于“hard trust region / rejection/masking”家族：当样本被判定为不可信时直接丢弃（梯度贡献为 0）。
- 它们不等价，核心差异在于 gate 统计量的选择：
  - TRM：用 $max_t D_{\mathrm{KL}}(\pi_{\mathrm{roll}}(\cdot|c_t) || \pi_\theta(\cdot|c_t))$ 这类 **worst-case token divergence** 做 gate（`tex/literature/TRM/main_arxiv.tex` 的 TRM 定义与 Algorithm）。
  - Seq-MIS：用序列重要性比 $\rho(y)$ 是否超过阈值 $C$ 做 gate（`tex/literature/Theory/3-Trust Region Optimization via Sequence Masking.md` 的 Seq-MIS 定义）。
  - Geo-Mask：用 $\rho_{\mathrm{geo}}(y)$（几何均值/平均 log-ratio）做 length-invariant gate（同一个 Theory Part 3 文档的 Geo-Mask 定义）。
- 从“长序列可用性”的角度：
  - TRM 的阈值 $\delta$ 是 length-invariant（不依赖 $T$），这是它的显式设计目标之一。
  - Seq-MIS 的 $\rho(y)=\prod\rho_t$ 是 extensive quantity，会天然随 $T$ 指数级漂移，导致结构性长度偏置；Theory Part 3 专门引入 Geo-Mask 来修正这一点。
  - 因而“TRM vs MIS”更准确的关系是：TRM 与 Geo-Mask 在“length-invariant trust region gate”的方向更一致，而 Seq-MIS 是 hard gate 但存在 length bias 需要额外处理。

### 方法对照表（家族图谱）


| 方法                               | Gate 粒度       | Gate 统计量                              | 是否 length-invariant | 是否乘 IS 权重  | 主要控制/解决的问题                                   |
| -------------------------------- | ------------- | ------------------------------------- | ------------------- | ---------- | -------------------------------------------- |
| Token-TIS                        | token         | $\rho_t$ 截断/clip                         | 是（逐 token）          | 是          | 缓解方差爆炸，但不处理 state distribution mismatch      |
| IcePop                           | token         | $\rho_t$ 超界则置 0（或保留系数）                   | 是（逐 token）          | 是/系数式      | 丢弃“不健康 token 更新”，抑制 mismatch 噪声              |
| Seq-IS                           | sequence      | $\rho(y)$                                | 否（extensive）        | 是          | 理论无偏，但长序列方差指数爆炸                              |
| Seq-TIS                          | sequence      | $\rho(y)$ 截断                             | 否（extensive）        | 是          | 在无偏与方差之间折中（仍有 length bias）                   |
| Seq-MIS                          | sequence      | $\rho(y)$ gate（拒绝）                       | 否（extensive）        | 是          | Hard trust region via rejection，过滤 OOD 高权重样本 |
| Geo-Mask                         | sequence(样本级) | $\rho_{\mathrm{geo}}(y)$ gate（双侧）                   | 是（按平均/几何均值）         | 否（纯过滤）     | 长序列下的 length-invariant hard trust region     |
| DPPO                             | token         | $D(\mu(\cdot\mid c_t) \Vert \pi(\cdot\mid c_t))$ gate + PPO 方向性 | 是（逐 token）          | 是          | 用 divergence 做 trust region，避免 ratio clip 的长尾词表病态 |
| TRM                              | sequence      | $max_t D_{\mathrm{KL}}(\pi_{\mathrm{roll}}(\cdot\mid c_t) \Vert \pi_\theta(\cdot\mid c_t))$ gate | 是（阈值不随 T）          | 是          | 用 max-divergence gate 保证 bound 非空化           |
| Prefix gate（本仓库 `reinforce_pro`） | token(前缀因果)   | prefix 几何均值 $\exp(mean_{s\leq t} \log \rho_s)$ | 是（按前缀平均）            | 是（token 级） | 在因果结构下做“前缀过滤”，更细粒度的 gate                     |


### 证据链（在哪些文件里能直接看到这些定义）

- TRM 的 gate 与 masked surrogate：
  - `tex/literature/TRM/main_arxiv.tex` 的 `\section{Trust Region Masking}` 中直接给出 `M(x,y)` 的定义、$L_masked$ 形式、Algorithm 与 TRM Guarantee（见该节的 “The Masked Surrogate Objective / Masking Criterion and Implementation / Theoretical Guarantee”）。
- Seq-MIS 与 Geo-Mask 的定义与动机：
  - `tex/literature/Theory/3-Trust Region Optimization via Sequence Masking.md` 中有
    - “Definition: Sequence-Level Masked IS (Seq-MIS)”
    - “Definition: Geometric Sequence Masking (Geo-Mask)”
    - 以及为什么 $\rho(y)$ 会导致 length-dependent rejection bias 的分析。
- MIS.md 的 “MIS” 叙述（与 Theory Part 3 一致）：
  - `tex/literature/MIS.md` 中有 “Masked Importance Sampling (MIS)” 与 token-level vs sequence-level 的对照。

### TRM 与 MIS 的关系：三个可操作的判别维度

把 “TRM vs (Seq-)MIS vs Geo-Mask” 的关系落到工程/论文写作上，最实用的是三维判别：

- 维度 1：**gate 用的统计量是什么**
  - TRM：worst-case token divergence（max-KL / max-TV），目标是直接控制 bound 里出现的 $D^{tok,max}$。
  - Seq-MIS：sequence ratio $\rho(y)$（本质是 $\sum \log \rho_t$ 的指数化），更像“用 IS 权重大小作为 OOD 指示器”。
  - Geo-Mask：$\rho_{\mathrm{geo}}$（平均 log-ratio 的指数化），把 extensive quantity 变成 per-token 的 intensive quantity。
- 维度 2：**是否天然 length-invariant**
  - TRM：是（阈值按设计不依赖 $T$）。
  - Seq-MIS：不是（$\rho(y)$ 是乘积，长度越长越容易被拒）。
  - Geo-Mask：是（平均/几何均值消掉长度）。
- 维度 3：**实现时需要访问哪些概率信息**
  - TRM：理想形态需要 full-vocab logits 才能算每个位置的 $D_{\mathrm{KL}}(\pi_{\mathrm{roll}}||\pi_\theta)$；文中也给了 sample-based proxy（`k_2/k_3`）但仍要定义好 max/avg 约束如何计算。
  - Seq-MIS / Geo-Mask：只需要 sampled token 的 logprob 就能构造 $\log \rho_t$ 与其前缀/序列统计量（更贴近现有高吞吐 RL 系统的可用信息）。

---

## OpenRLHF 里的实现接口（用于把文献映射到代码）

这一节只回答“在本仓库里，怎么开这些修正/算法选项”，不做额外理论扩展。

### 配置入口（CLI flags）

在 `openrlhf/cli/train_ppo_ray.py`（关键行号方便跳转）：

- `--enable_vllm_is_correction`
- `--vllm_is_truncated_threshold low high`（默认 `[0.5, 5.0]`）
- `--vllm_is_correction_type tis|icepop|seq-mask-tis|reinforce_pro`
- `--policy_loss_type ppo|gspo`

建议直接看：

- `openrlhf/cli/train_ppo_ray.py:244`（`--enable_vllm_is_correction`）
- `openrlhf/cli/train_ppo_ray.py:245`（阈值）
- `openrlhf/cli/train_ppo_ray.py:252`（type choices + help）
- `openrlhf/cli/train_ppo_ray.py:376`（`--policy_loss_type`）

### 关键实现点（PolicyLoss）

在 `openrlhf/models/loss.py` 的 `PolicyLoss.forward()`（关键行号）：

- PPO/GSPO 的 surrogate 与 ratio 定义：
  - PPO：$ratio = \exp(log_probs - old_log_probs)$（$\pi_{\mathrm{current}} / \pi_{\mathrm{old}}$，见 `openrlhf/models/loss.py:122`）
  - GSPO：$ratio = \exp(mean_t (log_probs - old/rollout))$ 并广播到 token（sequence-level geometric mean ratio，见 `openrlhf/models/loss.py:125`）
- vLLM mismatch 的 IS correction（只在 `policy_loss_type == "ppo"` 时生效）：
  - $log_ratio = old_log_probs - rollout_log_probs$（$\pi_{\mathrm{old}} / \pi_{\mathrm{roll}}$，见 `openrlhf/models/loss.py:152` 起）
  - `tis`：token-level clamp
  - `icepop`：token-level filter（超阈值置 0）
  - `seq-mask-tis`：先用 $seq_is = \exp(mean_t log_ratio)$ 做 **sequence gate**，再乘 token-level $\exp(log_ratio)$
  - `reinforce_pro`：prefix 累积几何均值 gate（prefix-level causal filtering），再乘 token-level $\exp(log_ratio)$

### 文献到实现的“保守映射”（不强行 1:1）

这里的关键词是“保守”：只写我们能在代码里指认出来的对应关系。

- TIS（token-level truncated IS） -> `--enable_vllm_is_correction --vllm_is_correction_type tis`
- IcePop（token-level filtering） -> `--enable_vllm_is_correction --vllm_is_correction_type icepop`
- Geo-Mask-Token-IS 的混合形态 -> `--enable_vllm_is_correction --vllm_is_correction_type seq-mask-tis`
  - 解释：`seq-mask-tis` 的 gate 统计量是 $seq_is = \exp(mean log_ratio)$，就是 $\rho_{\mathrm{geo}}$ 风格；然后仍保留 token-level IS 系数。
- GSPO（sequence-level ratio/clipping） -> `--policy_loss_type gspo`
- TRM（max-KL gate）：
  - 本仓库当前的 `seq-mask-tis` 不是 TRM 的 max-KL gate（它是 geometric-mean gate）。
  - 若要实现 TRM，需要能计算或近似 $max_t D_{\mathrm{KL}}(\pi_{\mathrm{roll}}(\cdot|c_t) || \pi_\theta(\cdot|c_t))$ 这类统计量；当前 `vllm_is` 路径主要使用 sampled-token logprob 的 log-ratio，而不是 full-vocab KL。

---

## Papers（文件夹内 `.tex`）

### TRM: Trust Region Masking for Long-Horizon LLM RL

入口：`tex/literature/TRM/main_arxiv.tex`

#### 问题设定

- 目标：控制 long-horizon 下 surrogate 与真目标之间的误差 `|Error|`，并让 bound 不随 $T^2$ 爆炸到 vacuous。
- 关键现实问题：现代系统里 $\pi_{\mathrm{roll}} \ne \pi_\theta$（backend discrepancy、MoE routing discontinuity、distributed staleness），导致 off-policy mismatch 不是可选项。

#### 承重公式

- divergence 记号（在 `main_arxiv.tex` 的 divergence 定义块与 `\section{Theoretical Analysis}` 开头）：
  - $\epsilon = D_{\mathrm{TV}}^{tok,max}$，$\delta = D_{\mathrm{KL}}^{tok,max}$，以及 $D_{\mathrm{KL}}^{seq}$ / $D_{\mathrm{TV}}^{seq}$ / $\bar{D}_t$
- Error 的 PDI 分解（承重起点，后续所有 bound 都从这里出发）：
  - $Error = \sum_t ( E_{d_t^{\pi_\theta}}[g_t] - E_{d_t^{\pi_{\mathrm{roll}}}}[g_t] )$
- TRM 的 gate 与 masked surrogate（在 `\section{Trust Region Masking}`）：
  - $M(x,y) = I[ max_t D_{\mathrm{KL}}(\pi_{\mathrm{roll}}(\cdot|c_t) || \pi_\theta(\cdot|c_t)) <= \delta ]$
  - $L_masked = E_{(x,y)\sim\pi_{\mathrm{roll}}}[ M(x,y) \cdot A(x,y) \cdot \sum_t \rho_t ]$（被拒样本梯度贡献为 0）

#### 算法步骤

1. 用 $\pi_{\mathrm{roll}}$ rollout 得到 `(x,y)`，并记录必要的统计量（至少要能在每个 token 位置估计 $D_{\mathrm{KL}}(\pi_{\mathrm{roll}}(\cdot|c_t) || \pi_\theta(\cdot|c_t))$ 或其 proxy）。
2. 在训练端计算/近似 per-token divergence，并取 worst-case：$max_t D_{\mathrm{KL}}(...)$。
3. 构造 sequence gate：若 $max_t D_{\mathrm{KL}} <= \delta$ 则 $M(x,y)=1$，否则拒绝该序列（$M=0$）。
4. 用 $L_masked$ 做更新：只让被接受序列对梯度有贡献，并按 batch size 做归一化以稳定 step size。

#### 关键定理

- 关键结论是 “TRM Guarantee”：当 gate 确保样本满足需要的 divergence 前提时，可以把 unified bound 套进来得到非空化（non-vacuous）的改进保证。
- `main_arxiv.tex` 给出一组可组合的 bound 家族并最终取 min（unified）：
  - Advantage Bound 与 Context Shift 是两块承重 building block；
  - Pinsker-Marginal / Mixed / Coupling / Adaptive 等路径给出不同的 $T$ scaling，上界更紧的路径决定最终的 $min{...}$。

#### 理论结论与证明过程入口

- 证明路线（把它当作你自己写证明时的模板）：
  1. 从 PDI 写出 `Error` 的逐步差分形式。
  2. 用 $|E_P[f]-E_Q[f]| <= 2||f||_\infty D_{\mathrm{TV}}(P,Q)$ 把每项拆成 “advantage 上界 \times context shift 上界”。
  3. 分别建立 $||g_t||_\infty$ 的上界（由 token divergence 控制）与 $||d_t^{\pi_\theta}-d_t^{\pi_{\mathrm{roll}}}||_TV$ 的上界（coupling / Pinsker / data processing / seq-level divergence 等路径）。
  4. 组合得到不同 bound 家族，并取 $min{...}$ 形成 unified bound。
- 证明入口：`main_arxiv.tex` 后半部分直接包含 “Proofs of Foundational Lemmas / Proofs of Main Theorems / Proof of the Adaptive Bound / Sample-Based Estimators” 等章节。

#### 与 OpenRLHF 映射（可选）

- OpenRLHF 当前的 `seq-mask-tis` 是 geometric-mean gate（基于 $\exp(mean log_ratio)$），不等价于 TRM 的 max-KL gate。
- 若要在本仓库落地 TRM：核心是新增一个可计算/可近似的 $max_t D_{\mathrm{KL}}(\pi_{\mathrm{roll}}(\cdot|c_t) || \pi_\theta(\cdot|c_t))$ gate（通常需要 full-vocab logits 或论文中提到的 proxy 统计量）。

---

### GSPO: Group Sequence Policy Optimization

入口：`tex/literature/GSPO/colm2024_conference.tex`

#### 问题设定

- 观点：GRPO 把 token-level ratio 当作 importance sampling correction，本质上是在每个 time step 只用 1 个样本估计 next-token 分布的 correction，导致高方差噪声累积并被 clipping 放大。
- 关键主张：reward 是 sequence-level 的，所以 off-policy correction 与 clipping 也应当 sequence-level 化。

#### 承重公式

- GSPO objective：$J_GSPO(\theta) = E[ (1/G) \sum_i \min( s_i(\theta) \hat{A}_i, clip(s_i(\theta)) \hat{A}_i ) ]$
- sequence ratio（几何均值）：$s_i(\theta) = ( \pi_\theta(y_i|x) / \pi_{\mathrm{old}}(y_i|x) )^{1/|y_i|} = \exp( (1/|y_i|) \sum_t \log \pi_\theta/\pi_{\mathrm{old}} )$
- gradient analysis（对比 GRPO）：GSPO 的 token 梯度被同一个 $s_i(\theta)$ 等权重缩放，而 GRPO 是 token-wise 不等权缩放。

#### 算法步骤

1. 对每个 query $x$，从 $\pi_{\mathrm{old}}$ 采样一组 responses ${y_i}_{i=1}^G$，得到 sequence-level rewards `r(x,y_i)`。
2. 用 group baseline 构造 sequence advantage（常见是 group mean/std 归一化的 $\hat{A}_i$）。
3. 计算 sequence ratio 的几何均值 $s_i(\theta)$（等价于平均 log-ratio 的指数化）。
4. 用 sequence-level clipping 对 $s_i(\theta)$ 做截断，并对整条序列的 token 梯度做同一个权重缩放（而不是 token-wise 不等权缩放）。

#### 关键定理

- 无正式定理（theorem-proof）；该 paper 以“推导/分析”为主，核心承重是 objective 定义与梯度对比结论。

#### 理论结论与证明过程入口

- 关键推导入口：`tex/literature/GSPO/colm2024_conference.tex` 的 `\subsection{Gradient Analysis}`。
- 建议关注两件事：
  - GSPO 的梯度形式里，整条序列的 token log-likelihood 梯度共享同一个缩放系数 $s_i(\theta)$。
  - GRPO 的梯度形式里，每个 token 被各自的 $w_{i,t}(\theta)$ 缩放，从而引入更高的 token-level 噪声与累积不稳定性。

#### 与 OpenRLHF 映射（可选）

- `--policy_loss_type gspo` 会走 `openrlhf/models/loss.py` 的 GSPO 分支：$ratio = \exp(mean_t log_ratio)$（sequence geometric mean ratio）并广播到 token。
- 注意 OpenRLHF 里 GSPO 的 `log_ratio` 在启用 vLLM correction 时可能取 `log_probs - rollout_log_probs`（即把 behavior policy 视为 rollout policy），这与 paper 的 $\pi_{\mathrm{old}}$/$\pi_{\mathrm{current}}$ 视角有关，使用时需要明确你把谁当 behavior。

---

### IcePop (Every Step Evolves / Ring-1T report)

入口：

- 总入口：`tex/literature/IcePop/main.tex`
- 算法公式入口：`tex/literature/IcePop/sections/method/rl-algo.tex`
- 理论分析入口：`tex/literature/IcePop/sections/appendix.tex` 的 “Theoretical Analysis for IcePop”

#### 问题设定

- MoE + 长序列下，training backend 与 inference engine 的概率差异会被路由不稳定与自回归累积放大，导致训练崩溃。

#### 承重公式

- IcePop 的核心是一个 token-level 的过滤/校正因子（双侧阈值 $[\alpha, \beta]$），常用写法是对 mismatch ratio 做 gate：
  - $v = \pi_{\mathrm{train}}(old) / \pi_{\mathrm{infer}}(old)$（或等价的 “training vs inference” 的校准 ratio）
  - $\mathcal{M}(v; \alpha, \beta)$ 用于丢弃/抑制超界 token 的梯度贡献
- 目标函数形态入口：`tex/literature/IcePop/sections/method/rl-algo.tex`（$J_IcePop(\theta)$ 在 GRPO/PPO 的 token surrogate 外乘 $\mathcal{M}(\cdot;\alpha,\beta)$）。

#### 算法步骤

1. rollout 侧用 inference engine 生成数据（隐含行为策略 $\pi_{\mathrm{infer}}(old)$）。
2. 训练侧对同一序列用 training backend 重新计算（或对齐得到）所需概率量（隐含 $\pi_{\mathrm{train}}(old)$）。
3. 对每个 token 计算 mismatch ratio $v$，并用阈值 $[\alpha,\beta]$ 得到 mask/系数 $\mathcal{M}(v;\alpha,\beta)$。
4. 在 token-level surrogate 上乘以该 mask/系数，超界 token 的梯度贡献为 0（或被显著抑制），从而降低 mismatch 噪声的复利式累积。

#### 关键定理

- “Compounding probability discrepancy” 类定理：在一组局部条件下 mismatch（例如 KL）会按 $(1 + const*\mu)$ 形式增长，解释了为什么 mismatch 会自激放大并导致崩溃风险上升。

#### 理论结论与证明过程入口

- 理论分析与 proof 入口：`tex/literature/IcePop/sections/appendix.tex` 的 “Theoretical Analysis for IcePop”。
- 建议先读该 appendix 的定理陈述，再回看 `rl-algo.tex` 的 objective 形式，理解“为什么 token-level filter 能阻断复利效应”。

#### 与 OpenRLHF 映射（可选）

- `--enable_vllm_is_correction --vllm_is_correction_type icepop`
  - 对应 `openrlhf/models/loss.py` 的 `icepop` 分支：$vllm_is = \exp(old_log_probs - rollout_log_probs)$，超阈值置 0，再乘到 loss 上。

---

### CISPO（MiniMax-M1 report，含 `\method{}`）

入口：`tex/literature/CISPO/main.tex`（主要内容在 `intro.tex`、`cpt.tex` 等 $\input{...}$ 文件）

#### 问题设定

- 目标：在大规模/长序列 RL 中提升稳定性与样本效率，但不希望像 PPO/GRPO 那样通过 clip 或 mask “丢 token”（因为长序列下丢 token 会显著损失学习信号）。
- 核心观点：与其 clip token update（PPO ratio clipping 的思路），不如直接 clip importance sampling 权重（IS weight），并保持所有 token 都参与梯度。

#### 承重公式

- `tex/literature/CISPO/cpt.tex` 给出从 REINFORCE 出发的 offline-corrected 形式（Eq.~`eq:reinforce`）：
  - $J_REINFORCE(\theta) = E_{o\sim\pi_{\mathrm{old}}}[ \sum_t \mathrm{sg}(r_{i,t}(\theta)) \cdot \hat{A}_{i,t} \cdot \log \pi_\theta(o_{i,t}|...) ]$
- CISPO 的核心 objective（Eq.~`eq:CISPO`）：
  - $J_CISPO(\theta) = E[ \sum_{i,t} \mathrm{sg}( \hat{r}_{i,t}(\theta) ) \cdot \hat{A}_{i,t} \cdot \log \pi_\theta(o_{i,t}|...) ]$
  - 其中 $\hat{r}_{i,t}(\theta) = clip(r_{i,t}(\theta), 1-\epsilon_low^{IS}, 1+\epsilon_high^{IS})$（IS weight clipping，而不是 token update clipping）
- 统一视角（Eq.~`eq:unify`）：把 token-wise mask $M_{i,t}$ 显式写进来，可表示 PPO-style mask 与 CISPO-style weight clipping 在同一框架下的差异。

#### 算法步骤

1. 用 $\pi_{\mathrm{old}}$ 采样一组 responses（通常是 group 采样），并计算 group-based advantage（沿用 GRPO/DAPO 的优势估计习惯）。
2. 对每个 token 计算 importance ratio $r_{i,t}(\theta)$，并使用 stop-gradient $\mathrm{sg}(\cdot)$ 固定权重数值，避免引入额外梯度项。
3. 对 IS weight 做截断：$r_{i,t} -> \hat{r}_{i,t} = clip(r_{i,t}, low, high)$。
4. 用 $\mathrm{sg}(\hat{r}_{i,t}) \cdot \hat{A}_{i,t} \cdot \log \pi_\theta$ 累加所有 token 的梯度贡献（关键点：不 drop token）。

#### 关键定理

- 无正式定理（theorem-proof）；该材料的承重点在 “目标函数推导 + 统一视角下的 clipping/mask 形式化”。

#### 理论结论与证明过程入口

- 关键推导入口：`tex/literature/CISPO/cpt.tex` 的 `\subsection{Efficient RL Scaling with \method{}}`：
  - Eq.~`eq:reinforce`（REINFORCE offline corrected）
  - Eq.~`eq:CISPO`（CISPO objective + weight clipping）
  - “A General Formulation” 段落 + Eq.~`eq:unify`（把 mask 写成显式变量并对齐 PPO trust region mask）
- 这份 report 的重心是“可训练的 surrogate/recipe”，不是在 trust region bound 意义上给出改进保证；因此更适合在 related work 里作为“工程路线”对照项。

#### 与 OpenRLHF 映射（可选）

- OpenRLHF 当前的 vLLM mismatch 修正属于 “乘一个额外的 IS 权重到 loss 上”（见 `openrlhf/models/loss.py` 的 `enable_vllm_is_correction` 路径）。
- CISPO 的区别在于：它把 “IS weight clipping” 作为主稳定机制，并且强调不 drop token；若要在本仓库 1:1 复现，需要在 policy loss 里新增一条 “clip vllm_is / clip IS weight” 的 surrogate 变体，并对齐 stop-gradient 语义。

---

### DPPO: Divergence Proximal Policy Optimization（Rethinking the Trust Region in LLM RL）

入口：`tex/literature/DPPO/example_paper.tex`（正文分块在 `tex/literature/DPPO/paper/`）

建议按这条路线读：

- 算法与核心公式：`tex/literature/DPPO/paper/method.tex`
- LLM 场景下的 trust region 定理：`tex/literature/DPPO/paper/llm_bound.tex`
- 证明与近似散度的“lower bound”论证：`tex/literature/DPPO/paper/app.tex`
- 经典 TRPO 对照背景：`tex/literature/DPPO/paper/background.tex`

#### 问题设定

- 这篇的核心论点是：PPO 的 ratio clipping 在 LLM 的长尾词表下会出现结构性病态。
- 关键原因：PPO 用 sampled token 的 ratio $r_t$ 来近似约束“分布散度”，本质上是对真实 divergence 的 **single-sample Monte Carlo 估计**，在低概率 token 上会极不稳定（`intro.tex` + `method.tex` 的动机段落）。

#### 承重公式

- TV 与 ratio 的精确关系（把 PPO clip 解释成 “约束 TV 的单样本估计”）：
  - $D_{\mathrm{TV}}(\mu(\cdot|s_t) || \pi(\cdot|s_t)) = (1/2) E_{a\sim\mu}[ |r_t - 1| ]$（`method.tex` 的 Eq.~(tv_as_expectation)）。
- DPPO objective（仍是 PPO 风格的 per-token surrogate，但把 clip 换成 divergence gate）：
  - $L^{DPPO}_\mu(\pi) = E_{y\sim\mu}[ \sum_t M_t^{DPPO} \cdot r_t \cdot \hat{A}_t ]$（`method.tex` 的 Eq.~(dppo_obj)）。
- DPPO mask（只在“朝远离 trust region 的方向”且 divergence 超阈值时屏蔽更新，保留 PPO 的非对称结构）：
  - $M_t^{DPPO} = 0$ 当 $(\hat{A}_t>0 ∧ r_t>1 ∧ D>\delta)$ 或 $(\hat{A}_t<0 ∧ r_t<1 ∧ D>\delta)$，否则为 1（`method.tex` 的 Eq.~(dppo_mask)）。
- divergence 近似的理论定位：Binary/Top-K divergence 是 true divergence 的 lower bound（`app.tex` 的 `app:divergence_lower_bounds`）。

#### 算法步骤

1. 用 $\mu=\pi_{\mathrm{roll}}$ 采样序列，并在每个 token 位置保留必要概率信息（至少是 sampled token 的 $\mu(a_t|c_t)$ 与训练端的 $\pi(a_t|c_t)$；Top-K 版本还需要 top-K logprob）。
2. 在每个 token 位置估计/近似 divergence $D(\mu(\cdot|c_t) || \pi(\cdot|c_t))$：
  - Binary：把 vocab 分成 `{a_t}` 与 “other”，用两点分布计算 TV/KL（`method.tex` 的 Binary 小节）。
  - Top-K：把 vocab 压缩成 $TopK(\mu) ∪ {a_t} ∪ {other}$ 再算 divergence（`method.tex` 的 Top-K 小节）。
3. 用 DPPO 的方向性规则构造 $M_t^{DPPO}$：只在 “会把 ratio 推得更远且 divergence>\delta” 的 token 上屏蔽更新（`method.tex` 的 `dppo_mask`）。
4. 用 $L^{DPPO}$ 做更新：本质上是 “divergence-gated 的 PPO surrogate”，但 gate 统计量来自分布层面而不是单样本 ratio。

#### 关键定理

1. Theorem: Performance Difference Identity for LLMs（`llm_bound.tex`，label `lem:llm_identity`）
2. Theorem: Policy Improvement Bound for LLMs（`llm_bound.tex`，label `thm:llm_tr_bound`）
3. Divergence 近似为 lower bound（`app.tex`，label `app:divergence_lower_bounds`）

#### 理论结论与证明过程入口

1. LLM trust region 理论入口：`tex/literature/DPPO/paper/llm_bound.tex`
  - surrogate $L'_\mu(\pi)$（Eq.~(llm_surrogate)）
  - bound 主结论（Eq.~(llm-tv-bound)，label `thm:llm_tr_bound`）
2. proofs 入口：`tex/literature/DPPO/paper/app.tex`
  - `\section{Trust Region in LLMs}`（label `app:llm_tr_proof`）
  - `\section{A Tighter Bound}`（label `app:tighter_bound`，给出更 practical 的 linear-in-$T$ bound）
3. divergence 近似 lower bound 的 proof：`app.tex` 的 `\section{Approximations as Lower Bounds of True Divergence}`（label `app:divergence_lower_bounds`）

#### 与 OpenRLHF 映射（可选）

- OpenRLHF 当前实现的是 ratio-based 的 vLLM mismatch correction（`tis/icepop/seq-mask-tis/reinforce_pro`），没有 DPPO 这种 “divergence-gated PPO”。
- 若要落地 DPPO：
  - Binary-TV/KL 只需要 sampled token 的 $(\mu(a_t|c_t), \pi(a_t|c_t))$，可以在现有 “rollout 保存 logprob + train 重算 logprob” 的管线里实现。
  - Top-K 需要两侧 top-K logprob；论文提到 vLLM 常见限制 $K<=20$，这会直接约束可用的 Top-K divergence 精度。

---

## Technical Reports（`.md`）

### Theory Part 1：Why Off-Policy Breaks RL — SGA Analysis Framework

入口：`tex/literature/Theory/1-Why Off-Policy Breaks RL An SGA Analysis Framework.md`

#### 问题设定

- 讨论对象：当行为策略 $\mu$ 与目标策略 $\pi$ 不一致（off-policy）时，为什么 RL 训练会出现崩溃/停滞。
- 核心框架：把“算法好不好”先抽象成 “你喂给优化器的随机梯度估计器 $\hat{g}$ 的性质”，再用一个通用的 SGA Lemma 分解单步进展。
- 结论导向：off-policy 的失败不是单一原因，而是两条独立失败链路：Bias（方向错）与 Variance（噪声大）。

#### 承重公式

- SGA Lemma（承重结构是三项分解）：单步期望进展 = True progress + Bias term + Noise penalty（文档 `## 2. The SGA Lemma`）。
- 指标对齐（承重度量工具）：
  - Bias 的量级通常用 TV distance 控制（期望差上界）。
  - Variance 的量级通常由 IS 二阶矩主导，可用 $\chi^2$-divergence 表达（$E[\rho^2]$ 量级）。

#### 算法步骤

1. 把你的算法写成 “用行为分布 $\mu$ 的样本估计目标 $J(\pi)$ 的梯度”的形式：明确 estimator 是什么（是否用了 IS、是否 truncation、是否 normalization）。
2. 套用 SGA Lemma：把单步进展拆成三项，并把你关心的失败模式落在 Bias 或 Variance 上。
3. 用 TV/$\chi^2$ 两个工具分别界住 Bias/Variance：回答“偏差会不会把方向推错”“方差会不会迫使学习率变小到训练停滞”。

#### 关键定理

- SGA Lemma：给出通用的 optimization progress 分解（文档 `## 2`，并带 “Derivation of the SGA Lemma”）。
-（用于 trust region 连接的）TRPO lower bound / Performance Difference Lemma / Simulation Lemma：用于把 surrogate 与真目标之间的误差项写成可 bound 的形式（文档 `## 5`）。

#### 理论结论与证明过程入口

- SGA Lemma 的推导入口：`## 2. The SGA Lemma: Quantifying Off-Policy Effects` 下的 `Derivation of the SGA Lemma`（spoiler）。
- trust region 连接入口：`## 5. Connection to Trust Region Methods (PPO/TRPO)`，其中：
  - `### 5.3 The TRPO Lower Bound`（含 proof sketch）
  - `Simulation Lemma Proof` / `Proof by Induction`（给出 TV 随时间步增长的上界证明）

---

### Theory Part 2：Token vs Sequence-level Correction（系统性 bias-variance 对比）

入口：`tex/literature/Theory/2-Applying the SGA Framework Token v.s. Sequence-level Correction.md`

#### 问题设定

- 目标：在 Part 1 的 SGA 框架下，对比 “sequence-level 修正” 与 “token-level 修正” 在长序列生成任务中的系统性 tradeoff。
- 关注点：同一件事（off-policy correction）在不同粒度下会把问题从 “Bias” 转移到 “Variance”，或者反过来。

#### 承重公式

- 把目标统一成：$g = E_\pi[f(y)]$，而你真正能做的是在 $y\sim\mu$ 下构造 estimator $\hat{g}$ 来近似 $g$。
- Seq-IS estimator：$\hat{g}_seq = \rho(y) \cdot f(y)$，其中 $\rho(y)=\pi(y)/\mu(y)$。
- Token-IS / PPO 范式：把 $\rho(y)=\prod_t \rho_t$ 打断成 token-wise 权重，避免 $E_\mu[\rho(y)^2]$ 指数爆炸，但会引入 state occupancy mismatch 的 bias（通过 Simulation Lemma / hybrid argument 体现）。

#### 算法步骤

1. 列出要比较的 estimator（Seq-IS / SNIS / Token-IS / Token-TIS / Seq-TIS 等），明确其权重统计量（$\rho(y)$ 还是 $\rho_t$，以及是否 truncation）。
2. 对每个 estimator 用 SGA Lemma 分析 Bias（Term B）与 Variance（Term C）：
  - Bias：看 $E_\mu[\hat{g}] - E_\pi[f]$ 如何由 state distribution mismatch 累积出来。
  - Variance：看 $E_\mu[\rho^2]$（或等价的 $\chi^2$）如何随 $T$ 放大。
3. 得出一张“不可同时兼得”的 tradeoff 图：要么无偏但方差灾难（Seq-IS），要么可训练但有结构性 bias（Token-IS / PPO 风格）。

#### 关键定理

- Simulation Lemma / hybrid argument（在文档中用来证明 “state occupancy mismatch 误差会按 $O(T\cdot\Delta_max)$ 累积”）：它是 token-level 修正无法消除 bias 的数学根源之一。
- tower property / 条件期望展开：用于把 $E_\mu[\rho(y)^2]$ 展开成 per-token 的连乘，得到长序列下的指数级上界（解释 Seq-IS 方差灾难）。

#### 理论结论与证明过程入口

- 逐 estimator 的分析入口（这篇是 analysis-driven，不是 theorem-driven）：
  - `## Analysis 1: Sequence-Level Importance Sampling (Seq-IS)`（无偏但方差灾难）
  - `## Analysis 2: Naive & Token-Level IS`（方差改善但出现 bias）
  - `## Analysis 3: Token-Level Truncated IS (Token-TIS and The PPO Paradigm)`（PPO 范式的稳定化解释）
  - `## Trade-offs: Sequence-Level Truncated IS (Seq-TIS)`（sequence truncation 的折中）
- Simulation Lemma 的位置可用全文检索 “Simulation Lemma” 直接跳转（该段含证明/推导链路）。

---

### Theory Part 3：Trust Region Optimization via Sequence Masking（Seq-MIS / Geo-Mask）

入口：`tex/literature/Theory/3-Trust Region Optimization via Sequence Masking.md`

#### 问题设定

- 目标：解释并修复长序列/LLM-RL 场景下 “soft trust region（clipping）不够用” 的两类病态：
  - OOD high-weight samples：$\rho(y)$ 极大时，clipping 仍保留样本但把权重截到 $C$，仍会污染梯度。
  - length-dependent rejection bias：$\rho(y)=\prod_t \rho_t$ 随长度指数漂移，导致长序列系统性更容易被拒绝。

#### 承重公式

- Seq-MIS（hard trust region via rejection）：用 gate $I(\rho(y) <= C)$ 直接丢弃 OOD 样本（文档 `Definition: Sequence-Level Masked IS (Seq-MIS)`）。
- 关键诊断：sequence ratio $\rho(y)=\prod_t \rho_t$ 是 extensive quantity，长度越长越容易越界。
- Geo-Mask（几何均值 gate，length-invariant）：$\rho_{\mathrm{geo}}(y) = \exp(mean_t \log \rho_t)$，并用双侧阈值得到 hard gate（文档 `Definition: Geometric Sequence Masking (Geo-Mask)`）。
- 连接到 divergence：文档 `### Mathematical Foundation: Connection to Per-Token KL Divergence` 给出 $\rho_{\mathrm{geo}}$ 与 per-token KL 约束的联系（把 gate 从 ratio 解释回 trust region）。

#### 算法步骤

1. 从 trust region 框架出发：承认 surrogate 只在信任域内是有效近似，因此要对 OOD 样本做硬约束（hard reject）。
2. 构造 Seq-MIS：当 $\rho(y)$ 超过阈值 $C$ 时拒绝整个序列，从而避免 “clip 仍更新 OOD” 的问题。
3. 识别 Seq-MIS 的长度偏置：$\rho(y)$ 的乘积结构导致长序列更易被拒绝。
4. 用 Geo-Mask 修正：改用 $\rho_{\mathrm{geo}}$（平均 log-ratio 的指数化）做 gate，使阈值不随 $T$ 漂移。
5.（可选）将 Geo-Mask 与 token-level truncated IS 组合，形成 “过滤 + 校正” 的 hybrid estimator（文档 `Definition: Geo-Mask-Token-TIS`）。

#### 关键定理

- 无单一“主定理”，但有三组关键定义/推导链路共同承重：
  - Seq-MIS 与 Geo-Mask 的定义（hard gate 的统计量选择）。
  - length bias 的数学解释（为什么 $\rho(y)$ 会系统性拒绝长序列）。
  - $\rho_{\mathrm{geo}}$ 与 per-token KL 的联系（把 heuristic gate 拉回 trust region 语义）。

#### 理论结论与证明过程入口

- 读法建议（按病态 -> 解决方案）：
  1. `## 1. OOD High-Weight Samples: Why Rejection Outperforms Clipping`（Seq-MIS 动机与 bias-variance 分析）
  2. `## 2. Length-Dependent Rejection Bias`（解释为什么 $\rho(y)$ 带来系统性长度偏置）
  3. `## 3. Geometric Sequence Masking`（从 extensive 到 intensive，并给出与 per-token KL 的联系）
- 关键定义的入口可用全文检索直接跳：
  - “Definition: Sequence-Level Masked IS (Seq-MIS)”
  - “Definition: Geometric Sequence Masking (Geo-Mask)”
  - “Definition: Geo-Mask-Token-TIS”

---

### TIS.md：sampler/learner mismatch 与 truncated IS

入口：`tex/literature/TIS.md`

#### 问题设定

- 讨论对象：现代 LLM-RL 系统里 rollout（vLLM/SGLang）与 training forward（FSDP/Megatron）用不同 backend，导致同参不同分布：$\pi_{\mathrm{sampler}}(\theta) \ne \pi_{\mathrm{learner}}(\theta)$。
- 直接后果：即便你“以为”在做 on-policy RL，实际梯度形式会变成 off-policy：采样来自 $\pi_{\mathrm{sampler}}$，梯度来自 $\pi_{\mathrm{learner}}$。

#### 承重公式

- 混合系统下的 REINFORCE 形式（文档 `# The Mismatch Problem`）：
  - $E_{a\sim\pi_{\mathrm{sampler}}(\theta)}[ R(a) \cdot \nabla_\theta \log \pi_{\mathrm{learner}}(a,\theta) ]$
- importance sampling 修正（文档 `## Embrace the mismatch — Apply algorithm-level fix`）：
  - 乘 ratio $\pi_{\mathrm{learner}}(a,\theta) / \pi_{\mathrm{sampler}}(a,\theta)$ 进行分布校正
- truncated importance sampling（TIS）：
  - $\min( \pi_{\mathrm{learner}}/\pi_{\mathrm{sampler}}, C )$ 用于控制方差与数值不稳定

#### 算法步骤

1. rollout 侧明确行为策略是 $\pi_{\mathrm{sampler}}$（不是 $\pi_{\mathrm{learner}}$），并把 sampler logprob/logits 作为行为分布的观测。
2. 训练侧计算 $\pi_{\mathrm{learner}}$ 的对应概率量（至少是 sampled token 的 logprob）。
3. 构造 mismatch ratio：$\rho_mis = \pi_{\mathrm{learner}} / \pi_{\mathrm{sampler}}$。
4. 用 TIS 截断：$\rho_mis_trunc = clip_or_min(\rho_mis, C)$，并把它乘到 policy gradient / surrogate loss 上作为 correction（必要时可区分 “PPO ratio” 与 “mismatch ratio” 两套 ratio）。

#### 关键定理

- 无正式定理（theorem-proof）；该材料以现象诊断、推导与经验分析为主，关键承重点是 “为什么系统层面的 gap 让训练变成 off-policy” 以及 “TIS 为什么是一个可行的 bias-variance 折中”。

#### 理论结论与证明过程入口

- mismatch 的数学化入口：`# The Mismatch Problem`（把 on-policy 形式改写成 sampler/learner 两策略形式）。
- TIS 修正项入口：`## Embrace the mismatch — Apply algorithm-level fix`（给出 IS 与 truncated IS 的推导）。
- PPO 扩展入口：`### Extension to Other Algorithms`（展示两套 ratio 并存时如何写出 corrected surrogate）。
- 若你关心 “为什么某些 PPO-IS 形式效果差”：全文检索 “PPO-IS v.s. TIS”。

#### 与 OpenRLHF 映射（可选）

- `--enable_vllm_is_correction --vllm_is_correction_type tis`：对应 `openrlhf/models/loss.py` 的 token-level clamp（把 mismatch ratio 截断到阈值区间）。

---

### MIS.md：mismatch -> collapse 诊断 + MIS 概念

入口：`tex/literature/MIS.md`

#### 问题设定

- 讨论对象：training-inference mismatch（例如 vLLM sampler vs FSDP trainer）如何在 LLM-RL 中触发突然崩溃。
- 该文的关键价值不在实验细节，而在于把 mismatch 与 off-policy 的两类失败模式（bias/variance）串起来，并给出 IS / MIS 的系统性解释。

#### 承重公式

- mismatch 下的“实际更新”（文档 `# 2. A Fundamental Conflict...`）：
  - $E_{y\sim\pi_{\mathrm{vllm}}(\cdot|x)}[ R(x,y) \cdot \nabla \log \pi_{\mathrm{fsdp}}(y|x) ]$（同参不同分布，隐式 off-policy）
- 分布校正的序列级 IS 形式（文档 `4.2.1 A Principled Solution: Distribution Correction`）：
  - sequence-level ratio $\rho(y) = \pi(y|x) / \mu(y|x)$（并讨论其方差问题与 truncation）
- Token-level IS 的关键缺陷定位（文档 `The Source of Bias in Token-Level IS`）：
  - deterministic transition 导致 $d_\mu \ne d_\pi$（state occupancy mismatch），token-wise ratio 无法校正状态分布差异，从而引入结构性 bias。
- MIS 的核心形式（文档 `Masked Importance Sampling (MIS)`）：
  - 用 hard gate/mask 拒绝高权重（OOD）样本，典型形式是 sequence gate：$I(\rho(y) <= C)$。

#### 算法步骤

1. 先诊断 mismatch：把“rollout policy”和“training policy”分开写出来，明确你在优化哪个 $\pi$、数据来自哪个 $\mu$。
2. 尝试分布校正（IS/TIS）：用 ratio $\rho$ 做 correction，但要意识到 Seq-IS 方差可能爆炸，因此需要 truncation。
3. 进一步做 hard gate（MIS）：当样本明显 OOD（$\rho$ 超阈）时直接拒绝，避免 clipping 后仍污染梯度。
4. 对 token-level 修正保持警惕：它能控方差但不消除 state occupancy mismatch 的 bias，长序列下仍可能崩。

#### 关键定理

- 无正式定理（theorem-proof）；该文主要以 “推导 + 机制解释 + 现象链路” 承重。
- 但它给出两个对写论文非常有用的“可复用结论”：
  - Token-level IS 的 bias 来源：state occupancy mismatch（$d_\mu \ne d_\pi$）是不可被 token-wise ratio 消除的。
  - MIS 的必要性：对于 OOD high-weight samples，clipping 不等价于 rejection，后者在稳定性上更强。

#### 理论结论与证明过程入口

- 主要理论段落入口（与 MIS/TRM/Geo-Mask 关系最紧密）：
  - `4.2.1 A Principled Solution: Distribution Correction`（sequence-level IS / token-level IS 的对比）
  - `The Source of Bias in Token-Level IS`（明确写出 state occupancy mismatch 的来源）
  - `Masked Importance Sampling (MIS)` 与 `Token-Level MIS vs. Sequence-Level MIS`（给出 MIS 的形式化与粒度对比）

#### 与 OpenRLHF 映射（可选）

- OpenRLHF 的 `seq-mask-tis` 更接近 Geo-Mask 风格的 gate（$\exp(mean log_ratio)$），而不是 $\rho(y)$ 形式的 Seq-MIS。
- 若要做 Seq-MIS：需要显式构造 sequence ratio gate，并处理长度偏置（参考 Theory Part 3 对 $\rho_{\mathrm{geo}}$ 的修正思路）。

---

### IcePop.md：MoE mismatch 直觉 + IcePop objective

入口：`tex/literature/IcePop.md`

#### 问题设定

- 讨论对象：MoE 架构下 training-inference mismatch 为什么更严重，以及它如何在 on-policy RL 中触发崩溃。
- 关键观察：routing（TopK experts）对数值扰动很敏感，导致同参但不同 backend 时激活专家集合不同，进而放大 $\pi_{\mathrm{infer}}$ vs $\pi_{\mathrm{train}}$ 的概率差异。

#### 承重公式

- mismatch 的 policy gradient 写法（文档开头给出一条 “混合引擎” 更新式）：采样来自 $\pi_{\mathrm{infer}}(\theta)$，梯度来自 $\pi_{\mathrm{train}}(\theta)$。
- “Compounding Probability Discrepancy” Lemma（文档 `## What Effects Will It Bring to MoE RL?` 下）：
  - 定义 $\delta_t = D_{\mathrm{KL}}(\pi_{\mathrm{infer}}(\cdot;\theta_t) || \pi_{\mathrm{train}}(\cdot;\theta_t))$；
  - 在一组局部条件下，bias 会推动 mismatch 进一步变大（自激/复利式增长）。
- IcePop 的 double-sided mask：用双侧阈值把 discrepancy 过大的 token 从梯度里移除（“Discard All Noisy Gradient Updates” 的算法段落）。

#### 算法步骤

1. 明确行为策略与训练策略来自不同 backend：$\pi_{\mathrm{infer}}$ vs $\pi_{\mathrm{train}}$（即便参数相同也可能不同分布）。
2. 对每个 token 计算 mismatch 指标（可理解为某种 ratio 或 divergence proxy）。
3. 用双侧阈值生成 mask：当 mismatch 过大时把对应 token 的梯度贡献置 0。
4. 用过滤后的 token-level surrogate 做更新，目标是阻断 mismatch 的复利式放大路径。

#### 关键定理

- Lemma（Compounding Probability Discrepancy）：给出 mismatch 自激增长的一组充分条件与推导链路，是该文最“可复用”的理论片段。

#### 理论结论与证明过程入口

- 理论入口：`tex/literature/IcePop.md` 的 “Lemma (Compounding Probability Discrepancy)” 这一节（在 mismatch effects 段落之后）。
- 算法入口：`# Unleash MoE RL with IcePop: Discard All Noisy Gradient Updates!`（包含 mask 的直觉与规则化描述）。

---

### Brief Introduction of Policy Gradient In LLM Reasoning：PG 定理与 surrogate objective（写作级模板）

入口：`tex/literature/Brief Introduction of Policy Gradient In LLM Reasoning.md`

#### 问题设定

- 目标：给出一套“写作级可复用”的推导链，把 LLM 的序列生成问题写成 MDP / policy optimization，并推到可在 autodiff 框架里实现的 surrogate objective。

#### 承重公式

- 两种等价写法的对齐：
  - token policy：$\pi(a_t|s_t)$
  - sequence likelihood：$\pi(y|x) = \prod_t \pi(y_t|x,y_{<t})$
- log-likelihood 分解（承重恒等式）：$\log \pi(y|x) = \sum_t \log \pi(a_t|s_t)$
- surrogate objective 的关键形式：把采样分布固定在 $\pi_{\theta_k}$，并显式写 ratio $\pi_\theta/\pi_{\theta_k}$，使得 “对 surrogate 做梯度下降” 等价于 “做 policy gradient 上升”。

#### 算法步骤

1. 先在序列层面写出目标 $J(\theta) = E_{y\sim\pi_\theta}[R(y)]$，并用 log-derivative trick 写出 $\nabla_\theta J$。
2. 用 $\log \pi(y|x)=\sum_t \log \pi(a_t|s_t)$ 把序列级梯度展开成 token-level 求和形式（便于实现与解释 credit assignment）。
3. 选择一个可训练的 surrogate loss（固定采样分布、显式 ratio、可带 baseline），把它写成 “直接喂给 autograd 的 loss”。

#### 关键定理

- Policy Gradient Theorem（序列级）与 token-level 分解是该文的核心定理链路；文档用 theorem + 推导的方式直接给出。

#### 理论结论与证明过程入口

- 入口就是文档本身（它以 “theorem + 推导” 组织），建议按文档顺序读完 Notation -> PG theorem -> token 分解 -> surrogate objective。

---

### Theory of On-Policy Distillation：OPD 的目标函数与 policy gradient（含 proof）

入口：`tex/literature/Theory of On-Policy Distillation.md`

#### 问题设定

- 给定 prompt `x~D`，每个 $x$ 对应一个 teacher policy $\pi_{T(x)}$。
- 目标：最小化 $\mathrm{KL}(\pi_\theta(\cdot|x) || \pi_{T(x)}(\cdot|x))$（reverse KL），等价于最大化一个 entropy-regularized 的序列级回报。

#### 承重公式

- 目标函数把 OPD 解释为“奖励是 teacher logprob 的 RL”：
  - immediate reward $r(s_t,a_t) = \log \pi_{T(x)}(a_t|s_t)$（并带 entropy 项）
- 给出 OPD 的 policy gradient 形式，并列出多种 advantage 选择：
  - full-trajectory log ratio
  - log-ratio-to-go
  - baselined 版本
  - 以及把未来项写成 KL 的等价形式

#### 算法步骤

1. 写出 OPD objective（reverse KL），并把它改写成在 $y\sim\pi_\theta$ 下的期望形式。
2. 用 log-derivative trick 得到 $E[ advantage \cdot \nabla \log \pi_\theta ]$ 的 policy gradient 结构。
3. 选择 advantage 形态（full-trajectory / log-ratio-to-go / KL-to-go / baseline），并给出它们何时无偏、何时会引入 bias。

#### 关键定理

- Theorem 1（Policy Gradient of OPD）：给出 OPD 的无偏 policy gradient 形式，并说明 per-token advantage 需要包含未来项（log-ratio-to-go / KL-to-go 等）。
- Lemma 1（Expectation Exchange Under State Occupancy）：用于把期望在 state occupancy 下交换，支撑后续分解。
- Theorem 2（Gradient of Parameterized State Occupancy）：给出 $\sum_s f(s) \nabla d_\theta(s)$ 的解析形式，用于解释 “只用 immediate log ratio 会产生 bias” 的来源。
- Theorem 3（Optimality At The Stationary Point）：对不同 surrogate/advantage 选择的 stationary point 性质做比较。

#### 理论结论与证明过程入口

- 定理与 proof 都在文档本体内（不是外链 appendix），建议直接按顺序读：
  1. `## 1.2 Policy Gradient of OPD`（Theorem 1 + 分条 proof）
  2. `Lemma 1` 与 `Theorem 2`（解释 state occupancy 的梯度项为何导致 bias）
  3. `Theorem 3`（比较 immediate log ratio vs log-ratio-to-go 的最优性/稳定点性质）

---

### how does rl policy entropy converge during iteration：策略熵随迭代如何变化（NPG/KL-regularized 更新）

入口：`tex/literature/How Does RL Policy Entropy Converge During Iteration.md`

这份笔记回答一个很具体的问题：**在策略迭代/梯度迭代过程中，什么时候 entropy 会下降，什么时候 entropy 可能上升？**
它不是一篇完整 paper，但给了一个在写作与分析时很实用的“承重结论”：entropy 的一阶变化可以写成一个协方差。

#### 问题设定

- 离散动作空间，softmax policy（非 neural softmax，只讨论最简形态）。
- 用 KL-regularized 的 per-state update 解释 Natural Policy Gradient (NPG)：
  - $\pi_{k+1}(\cdot|s) = \arg\max_p E_{a\sim p}[Q^{\pi_k}(s,a)] - (1/\eta) \mathrm{KL}(p, \pi_k(\cdot|s))$
  - 等价形式：$\pi_{k+1}(a|s) \propto \pi_k(a|s) \exp(\eta A^{\pi_k}(s,a))$
- 一句与 RLHF 的对齐：在 bandit/单步设置下，上述更新就是 “reward 直接当 advantage” 的单步 NPG（笔记里也点明了这点）。

#### 承重公式

把 $H(\pi(\cdot|s))$ 视为参数 $\theta$ 的函数，做一阶近似：

- 一般 softmax 参数更新：
  - $H(\theta^{k+1}|s) - H(\theta^k|s) \approx -Cov_{a\sim\pi_k(\cdot|s)}( \log \pi_k(a|s), \theta^{k+1}_{s,a} - \theta^k_{s,a} )$
- 代入 NPG 更新 $\Delta\theta = \eta A$：
  - $H(\theta^{k+1}|s) - H(\theta^k|s) \approx -\eta \cdot Cov_{a\sim\pi_k(\cdot|s)}( \log \pi_k(a|s), A^{\pi_k}(s,a) )$

#### 算法步骤

1. 把 entropy 写成参数函数 $H(\theta|s)$，对 $\theta^{k+1}=\theta^k+\Delta\theta$ 做一阶展开。
2. 显式计算 $\nabla_\theta H$ 并与更新方向 $\Delta\theta$ 做内积，整理成协方差形式。
3. 代入 NPG 的 $\Delta\theta = \eta A$ 得到 entropy 变化与 $Cov(\log \pi, A)$ 的直接关系，从而给出 “何时熵减/熵增” 的可解释条件。

#### 关键定理

- 无正式定理（theorem-proof）；该材料是推导笔记，核心承重结论是 “entropy 一阶变化 = 协方差” 这一恒等式与其 NPG 特例。

#### 理论结论与证明过程入口

- 推导入口：笔记从 entropy 定义出发逐行计算 $\nabla_\theta H$，并整理成协方差表达式；不依赖额外 lemma。
- 直觉入口：结论可以读成：
  - $Cov(\log \pi, A) > 0$ 时 entropy 倾向下降（高概率动作也高 advantage，强化更确定的选择）。
  - $Cov(\log \pi, A) < 0$ 时 entropy 下降被抑制甚至上升（优势主要集中在低概率动作，出现探索/迁移概率质量的压力）。

---

### RL训练中为什么熵减往往意味着训练收敛：entropy -> gradient norm / KL 位移（skydownacai）

入口：`tex/literature/RL训练中为什么熵减往往意味着训练收敛.md`

这份笔记更像“读书笔记 + 推导草稿”：它试图从 softmax 参数化的曲率出发解释一个经验现象：**entropy 逐步降低时，训练往往进入收敛/变慢阶段**。

#### 问题设定

- 离散动作空间 + softmax policy（logits 参数化）。
- 讨论对象是“策略梯度/策略迭代一类更新”在 softmax 几何下的行为；其中第 1 个结果引用 EMPG paper。

#### 承重公式

- 不等式 1（引用 EMPG paper，并用 Renyi entropy 单调性推出信息熵版上界）：
  - $E_{a\sim\pi(\cdot|s)}[ ||\nabla_{z(s)} \log \pi(a|s)||_2^2 ] \leq 1 - \exp( -H(\pi(\cdot|s)) )$
- 不等式 2（reverse KL 的二阶近似上界，带完整推导）：
  - 对 $\Delta_s = z_{\theta^+}(s) - z_\theta(s)$，
  - $\mathrm{KL}( \pi_{z_{\theta^+}}(\cdot|s) || \pi_{z_\theta}(\cdot|s) ) \leq (|A|/2) \cdot ||\Delta_s||_\infty^2 \cdot (1 - \exp(-H(\pi_\theta(\cdot|s)))) + o(||\Delta_s||_2^2)$

#### 算法步骤

1. 把 entropy/梯度大小/策略位移都写到同一个 softmax logits 参数空间里（把 softmax 的“曲率”显式写出来）。
2. 用不等式 1 解释：entropy 越低，期望梯度范数越小，学习会变慢。
3. 用泰勒展开把 reverse KL 写成 Fisher 二次型，再把它上界到 $||\Delta||_\infty^2$ 与 entropy 因子（不等式 2）。

#### 关键定理

- 无正式 theorem-proof（它是笔记体），但有两条可复用的“承重结论”：
  - entropy 控制期望梯度范数上界（通过 $1-\exp(-H)$ 形式）
  - entropy 进入 reverse KL 位移上界（通过 Fisher 二次项与 $||\Delta||_\infty$）

#### 理论结论与证明过程入口

- 不等式 1：入口是笔记中 “## 1. Entropy 衰减，策略梯度幅度衰减” 段落（并指向 EMPG paper 链接）。
- 不等式 2：入口是 “## 2. Entropy 衰减，策略Reverse KL移动幅度上界衰减”：
  - 用泰勒展开得到 KL 的二阶项；
  - 识别 Hessian 为 Fisher；
  - 用 $||\Delta||_\infty$ + entropy 上界 Fisher 二次型（最后一步显式调用不等式 1）。
