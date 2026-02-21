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

| 名称 | 载体 | 主题关键词 | 理论/证明密度 | 入口文件 |
| --- | --- | --- | --- | --- |
| TRM (Trust Region Masking) | paper `.tex` | 长序列下 trust region bound 非空化，sequence-level masking | 高 | `tex/literature/TRM/main_arxiv.tex` |
| GSPO (Group Sequence Policy Optimization) | paper `.tex` | sequence likelihood ratio，sequence-level clipping，与 GRPO 的差异 | 中 | `tex/literature/GSPO/colm2024_conference.tex` |
| IcePop (Every Step Evolves / Ring-1T report) | paper `.tex` | training-inference mismatch，token-level filtering/校正 | 中（附录有定理+证明） | `tex/literature/IcePop/main.tex` |
| CISPO (MiniMax-M1 report, 含 `\\method{}`) | paper `.tex` | 重要性权重截断、工程化的 RL scaling 配方 | 低-中（更偏 recipe） | `tex/literature/CISPO/main.tex` |
| DPPO (example) | paper `.tex` | 示例/模板性质 | 低 | `tex/literature/DPPO/example_paper.tex` |
| TIS（off-policy mismatch + truncated IS） | tech report `.md` | sampler/learner mismatch，TIS 修正项 | 中 | `tex/literature/TIS.md` |
| MIS（mismatch -> collapse + 序列级修正观点） | tech report `.md` | token-level bias、seq-level 修正、MIS | 中 | `tex/literature/MIS.md` |
| IcePop（博客稿） | tech report `.md` | MoE mismatch，IcePop objective 与直觉 | 中 | `tex/literature/IcePop.md` |
| Policy Gradient Intro（LLM reasoning） | tech report `.md` | policy gradient theorem，surrogate objective（autodiff 实现） | 中-高（含 theorem/推导） | `tex/literature/Brief Introduction of Policy Gradient In LLM Reasoning.md` |
| On-Policy Distillation (OPD) | tech report `.md` | OPD = reverse KL，等价 entropy-regularized RL，policy gradient | 高（含 theorem + proof） | `tex/literature/Theory of On-Policy Distillation.md` |
| Theory Part 1 | tech report `.md` | SGA lemma，bias vs variance，TV vs chi^2，TRPO 连接 | 高 | `tex/literature/Theory/1-Why Off-Policy Breaks RL An SGA Analysis Framework.md` |
| Theory Part 2 | tech report `.md` | Seq-IS/Token-IS 的系统性 bias-variance 分析 | 高 | `tex/literature/Theory/2-Applying the SGA Framework Token v.s. Sequence-level Correction.md` |
| Theory Part 3 | tech report `.md` | Seq-MIS，Geo-Mask，hard trust region via masking | 高 | `tex/literature/Theory/3-Trust Region Optimization via Sequence Masking.md` |

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

---

## 统一术语与符号对照（跨文献对齐）

为了减少“同一个东西不同名字”的认知切换，后续卡片统一用下面这组抽象：

- `x`：prompt / query
- `y = (y_1,...,y_T)`：response（token 序列）
- `c_t = (x, y_{<t})`：prefix/context（LLM 生成的 state）
- 行为策略 / 采样策略：`μ` 或 `π_roll`（rollout policy / sampler policy）
- 目标策略 / 训练策略：`π` 或 `π_θ`（learner policy / training policy）
- per-token ratio：`ρ_t = π(y_t|c_t) / μ(y_t|c_t)`
- sequence ratio：`ρ(y) = π(y|x) / μ(y|x) = ∏_t ρ_t`
- geometric mean ratio（长度归一）：`ρ_geo(y) = ρ(y)^{1/T} = exp( (1/T)∑_t log ρ_t )`

一个关键提醒（对齐 OpenRLHF 实现很重要）：
- “PPO ratio”通常指 `π_current / π_old`（用于 clip）
- “mismatch correction ratio”通常指 `π_old / π_rollout`（用于纠正 rollout backend 与训练 backend 的不一致）
这两种 ratio 在实现里往往同时存在，不能混为一谈。

---

## 关系图谱：TRM vs Seq-MIS vs Geo-Mask（以及与 OpenRLHF 的映射）

这部分的目标是回答一个具体问题：**TRM 与 MIS 的关系是什么？它们是否等价？如果不等价，差异点在哪里？**

结论先行（可核查，后面给证据链）：
- TRM 与 (Seq-)MIS 都属于“hard trust region / rejection/masking”家族：当样本被判定为不可信时直接丢弃（梯度贡献为 0）。
- 它们不等价，核心差异在于 gate 统计量的选择：
  - TRM：用 `max_t D_KL(π_roll(·|c_t) || π_θ(·|c_t))` 这类 **worst-case token divergence** 做 gate（`tex/literature/TRM/main_arxiv.tex` 的 TRM 定义与 Algorithm）。
  - Seq-MIS：用序列重要性比 `ρ(y)` 是否超过阈值 `C` 做 gate（`tex/literature/Theory/3-Trust Region Optimization via Sequence Masking.md` 的 Seq-MIS 定义）。
  - Geo-Mask：用 `ρ_geo(y)`（几何均值/平均 log-ratio）做 length-invariant gate（同一个 Theory Part 3 文档的 Geo-Mask 定义）。
- 从“长序列可用性”的角度：
  - TRM 的阈值 `δ` 是 length-invariant（不依赖 `T`），这是它的显式设计目标之一。
  - Seq-MIS 的 `ρ(y)=∏ρ_t` 是 extensive quantity，会天然随 `T` 指数级漂移，导致结构性长度偏置；Theory Part 3 专门引入 Geo-Mask 来修正这一点。
  - 因而“TRM vs MIS”更准确的关系是：TRM 与 Geo-Mask 在“length-invariant trust region gate”的方向更一致，而 Seq-MIS 是 hard gate 但存在 length bias 需要额外处理。

### 方法对照表（家族图谱）

| 方法 | Gate 粒度 | Gate 统计量 | 是否 length-invariant | 是否乘 IS 权重 | 主要控制/解决的问题 |
| --- | --- | --- | --- | --- | --- |
| Token-TIS | token | `ρ_t` 截断/clip | 是（逐 token） | 是 | 缓解方差爆炸，但不处理 state distribution mismatch |
| IcePop | token | `ρ_t` 超界则置 0（或保留系数） | 是（逐 token） | 是/系数式 | 丢弃“不健康 token 更新”，抑制 mismatch 噪声 |
| Seq-IS | sequence | `ρ(y)` | 否（extensive） | 是 | 理论无偏，但长序列方差指数爆炸 |
| Seq-TIS | sequence | `ρ(y)` 截断 | 否（extensive） | 是 | 在无偏与方差之间折中（仍有 length bias） |
| Seq-MIS | sequence | `ρ(y)` gate（拒绝） | 否（extensive） | 是 | Hard trust region via rejection，过滤 OOD 高权重样本 |
| Geo-Mask | sequence(样本级) | `ρ_geo(y)` gate（双侧） | 是（按平均/几何均值） | 否（纯过滤） | 长序列下的 length-invariant hard trust region |
| TRM | sequence | `max_t D_KL(π_roll||π_θ)` gate | 是（阈值不随 T） | 间接（通过 surrogate） | 让 long-horizon trust region bound 非空化；必须序列级 gate |
| Prefix gate（本仓库 `reinforce_pro`） | token(前缀因果) | prefix 几何均值 `exp(mean_{s≤t} log ρ_s)` | 是（按前缀平均） | 是（token 级） | 在因果结构下做“前缀过滤”，更细粒度的 gate |

### 证据链（在哪些文件里能直接看到这些定义）

- TRM 的 gate 与 masked surrogate：
  - `tex/literature/TRM/main_arxiv.tex` 的 `\section{Trust Region Masking}` 中直接给出 `M(x,y)` 的定义、`L_masked` 形式、Algorithm 与 TRM Guarantee（见该节的 “The Masked Surrogate Objective / Masking Criterion and Implementation / Theoretical Guarantee”）。
- Seq-MIS 与 Geo-Mask 的定义与动机：
  - `tex/literature/Theory/3-Trust Region Optimization via Sequence Masking.md` 中有
    - “Definition: Sequence-Level Masked IS (Seq-MIS)”
    - “Definition: Geometric Sequence Masking (Geo-Mask)”
    - 以及为什么 `ρ(y)` 会导致 length-dependent rejection bias 的分析。
- MIS.md 的 “MIS” 叙述（与 Theory Part 3 一致）：
  - `tex/literature/MIS.md` 中有 “Masked Importance Sampling (MIS)” 与 token-level vs sequence-level 的对照。

### TRM 与 MIS 的关系：三个可操作的判别维度

把 “TRM vs (Seq-)MIS vs Geo-Mask” 的关系落到工程/论文写作上，最实用的是三维判别：

- 维度 1：**gate 用的统计量是什么**
  - TRM：worst-case token divergence（max-KL / max-TV），目标是直接控制 bound 里出现的 `D^{tok,max}`。
  - Seq-MIS：sequence ratio `ρ(y)`（本质是 `∑ log ρ_t` 的指数化），更像“用 IS 权重大小作为 OOD 指示器”。
  - Geo-Mask：`ρ_geo`（平均 log-ratio 的指数化），把 extensive quantity 变成 per-token 的 intensive quantity。
- 维度 2：**是否天然 length-invariant**
  - TRM：是（阈值按设计不依赖 `T`）。
  - Seq-MIS：不是（`ρ(y)` 是乘积，长度越长越容易被拒）。
  - Geo-Mask：是（平均/几何均值消掉长度）。
- 维度 3：**实现时需要访问哪些概率信息**
  - TRM：理想形态需要 full-vocab logits 才能算每个位置的 `D_KL(π_roll||π_θ)`；文中也给了 sample-based proxy（`k_2/k_3`）但仍要定义好 max/avg 约束如何计算。
  - Seq-MIS / Geo-Mask：只需要 sampled token 的 logprob 就能构造 `log ρ_t` 与其前缀/序列统计量（更贴近现有高吞吐 RL 系统的可用信息）。

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
  - PPO：`ratio = exp(log_probs - old_log_probs)`（`π_current / π_old`，见 `openrlhf/models/loss.py:122`）
  - GSPO：`ratio = exp(mean_t (log_probs - old/rollout))` 并广播到 token（sequence-level geometric mean ratio，见 `openrlhf/models/loss.py:125`）
- vLLM mismatch 的 IS correction（只在 `policy_loss_type == "ppo"` 时生效）：
  - `log_ratio = old_log_probs - rollout_log_probs`（`π_old / π_rollout`，见 `openrlhf/models/loss.py:152` 起）
  - `tis`：token-level clamp
  - `icepop`：token-level filter（超阈值置 0）
  - `seq-mask-tis`：先用 `seq_is = exp(mean_t log_ratio)` 做 **sequence gate**，再乘 token-level `exp(log_ratio)`
  - `reinforce_pro`：prefix 累积几何均值 gate（prefix-level causal filtering），再乘 token-level `exp(log_ratio)`

### 文献到实现的“保守映射”（不强行 1:1）

这里的关键词是“保守”：只写我们能在代码里指认出来的对应关系。

- TIS（token-level truncated IS） -> `--enable_vllm_is_correction --vllm_is_correction_type tis`
- IcePop（token-level filtering） -> `--enable_vllm_is_correction --vllm_is_correction_type icepop`
- Geo-Mask-Token-IS 的混合形态 -> `--enable_vllm_is_correction --vllm_is_correction_type seq-mask-tis`
  - 解释：`seq-mask-tis` 的 gate 统计量是 `seq_is = exp(mean log_ratio)`，就是 `ρ_geo` 风格；然后仍保留 token-level IS 系数。
- GSPO（sequence-level ratio/clipping） -> `--policy_loss_type gspo`
- TRM（max-KL gate）：
  - 本仓库当前的 `seq-mask-tis` 不是 TRM 的 max-KL gate（它是 geometric-mean gate）。
  - 若要实现 TRM，需要能计算或近似 `max_t D_KL(π_roll(·|c_t) || π_θ(·|c_t))` 这类统计量；当前 `vllm_is` 路径主要使用 sampled-token logprob 的 log-ratio，而不是 full-vocab KL。

---

## Papers（文件夹内 `.tex`）

### TRM: Trust Region Masking for Long-Horizon LLM RL

入口：`tex/literature/TRM/main_arxiv.tex`

#### 问题设定
- 目标：控制 long-horizon 下 surrogate 与真目标之间的误差 `|Error|`，并让 bound 不随 `T^2` 爆炸到 vacuous。
- 关键现实问题：现代系统里 `π_roll != π_θ`（backend discrepancy、MoE routing discontinuity、distributed staleness），导致 off-policy mismatch 不是可选项。

#### 核心定义/公式（承重公式）
- 最大 token divergence 与序列 divergence 的定义：`ε = D_TV^{tok,max}`，`δ = D_KL^{tok,max}`，以及 `D_KL^{seq}` / `D_TV^{seq}` / `\bar{D}_t`（见 `main_arxiv.tex` 的 divergence 定义块与 `\section{Theoretical Analysis}` 开头）。
- Error 的 PDI 分解（`Error = Σ_t (E_{d_t^{πθ}}[g_t] - E_{d_t^{πroll}}[g_t])`），后续所有 bound 都从这里出发。

#### 算法要点（TRM 本体）
- sequence mask：`M(x,y)= I[max_t KL(c_t) <= δ]`（用 worst-case token KL 做 gate）。
- masked surrogate：`L_masked = E_{πroll}[ M * A(x,y) * Σ_t ρ_t ]`（注意：拒绝样本贡献为 0，并且梯度按总 batch size 归一化）。
- 关键工程前提：TRM 需要能算出 per-token KL（文中强调可用 rollout 存储 logits + training 前向 logits “exact KL over vocab”）。

#### 理论贡献（这篇必须看的点）
- 两个 building blocks：
  - Advantage Bound（把 `g_t` 的 sup-norm 与 `ε/δ` 绑定）
  - Context Shift（把 `||d_t^{πθ} - d_t^{πroll}||_TV` 与 coupling / Pinsker / seq-level divergence 绑定）
- 主要 bound 家族：
  - Pinsker-Marginal（`O(T^{3/2})`）
  - Mixed（`O(T)`）
  - Coupling（TV route，capped 后可 `O(T)`）
  - Adaptive（用 `\bar{D}_t` 做 data-dependent tightening）
  - Unified（取 min）
- TRM Guarantee：当 gate 保证 `δ`（或其 proxy）被控制时，将 unified bound 替换进去得到非空化 guarantee。

#### 证明路线（Roadmap）
1. 从 PDI 写出 `Error` 的逐步差分形式（每一步是 “同一个函数在两个 context 分布上的期望差”）。
2. 用 `|E_P[f]-E_Q[f]| <= 2||f||_∞ D_TV(P,Q)` 把每项拆成 “advantage 上界 × context shift 上界”。
3. 分别证明：
   - `||g_t||_∞` 可以由 token divergence 上界（`ε` 或 `δ`）控制。
   - `||d_t^{πθ} - d_t^{πroll}||_TV` 可以由 coupling、Pinsker-on-marginal、data processing、Pinsker-on-seq 等多条路控制。
4. 组合不同上界路径得到不同 bound 家族，分析它们的 `T` scaling。
5. 用 `min{...}` 组成 unified bound，保证每个子 bound 单独成立时 min 也成立。
6. 设计 TRM gate，使“被接受样本”满足 needed 的 divergence 前提；再用 acceptance rate 作为经验信号判断全局前提是否近似成立。

#### Proof 在哪里
- `main_arxiv.tex` 本身就包含证明章：可以直接跳到 “Proofs of Foundational Lemmas / Proofs of Main Theorems / Proof of the Adaptive Bound / Sample-Based Estimators” 等章节（在文件后半部分）。

#### 与本仓库实现对应（仅映射能落地的部分）
- OpenRLHF 当前的 `seq-mask-tis` 是 “geometric mean gate”，不等价于 TRM 的 “max-KL gate”。
- 如果未来要补 TRM：需要新增一个能估计 `max_t D_KL(π_roll || π_θ)` 的 gate（至少要能从 rollout 拿到 full-vocab logits，或引入近似 `k_2/k_3` 统计量并做 max/avg 约束）。

---

### GSPO: Group Sequence Policy Optimization

入口：`tex/literature/GSPO/colm2024_conference.tex`

#### 问题设定
- 观点：GRPO 把 token-level ratio 当作 importance sampling correction，本质上是在每个 time step 只用 1 个样本估计 next-token 分布的 correction，导致高方差噪声累积并被 clipping 放大。
- 关键主张：reward 是 sequence-level 的，所以 off-policy correction 与 clipping 也应当 sequence-level 化。

#### 核心公式/算法
- GSPO objective：`J_GSPO(θ) = E[ (1/G) Σ_i min( s_i(θ) Â_i, clip(s_i(θ)) Â_i ) ]`
- sequence ratio（几何均值）：`s_i(θ) = ( π_θ(y_i|x) / π_old(y_i|x) )^{1/|y_i|} = exp( (1/|y_i|) Σ_t log π_θ/π_old )`
- gradient analysis（对比 GRPO）：GSPO 的 token 梯度被同一个 `s_i(θ)` 等权重缩放，而 GRPO 是 token-wise 不等权缩放。

#### 理论/证明形态
- 这篇更像“推导/分析”而非大量 theorem-proof：重点在 `\subsection{Gradient Analysis}` 的推导链与对比结论。

#### 与本仓库实现对应
- `--policy_loss_type gspo` 会走 `openrlhf/models/loss.py` 的 GSPO 分支：`ratio = exp(mean_t log_ratio)`（sequence geometric mean ratio）并广播到 token。
- 注意 OpenRLHF 里 GSPO 的 `log_ratio` 在启用 vLLM correction 时可能取 `log_probs - rollout_log_probs`（即把 behavior policy 视为 rollout policy），这与 paper 的 `π_old`/`π_current` 视角有关，使用时需要明确你把谁当 behavior。

---

### IcePop (Every Step Evolves / Ring-1T report)

入口：
- 总入口：`tex/literature/IcePop/main.tex`
- 算法公式入口：`tex/literature/IcePop/sections/method/rl-algo.tex`
- 理论分析入口：`tex/literature/IcePop/sections/appendix.tex` 的 “Theoretical Analysis for IcePop”

#### 问题设定
- MoE + 长序列下，training backend 与 inference engine 的概率差异会被路由不稳定与自回归累积放大，导致训练崩溃。

#### 核心算法（IcePop 的 token-level filtering/校正）
- 用校准 ratio `π_train(old)/π_infer(old)` 做 token-level 的 mask/系数（双侧阈值 `[α, β]`），超界 token 的梯度直接置 0。
- 目标函数形态在 `rl-algo.tex` 中给出：`J_IcePop(θ)` 是在 GRPO/PPO 的 token surrogate 外乘一个 `\mathcal{M}(\cdot;α,β)`。

#### 理论贡献与证明
- `sections/appendix.tex` 给出 “Compounding probability discrepancy” 定理与证明：在一组局部条件下 mismatch KL `δ_t` 会按 `(1 + const*μ)` 形式增长（说明 mismatch 具有自激/复利性质）。

#### 与本仓库实现对应
- `--enable_vllm_is_correction --vllm_is_correction_type icepop`
  - 对应 `openrlhf/models/loss.py` 的 `icepop` 分支：`vllm_is = exp(old_log_probs - rollout_log_probs)`，超阈值置 0，再乘到 loss 上。

---

### CISPO（MiniMax-M1 report，含 `\\method{}`）

入口：`tex/literature/CISPO/main.tex`（主要内容在 `intro.tex`、`cpt.tex` 等 `\input{...}` 文件）

这份资料偏“工程化 recipe + 算法描述”，理论/证明密度不高。与本文主题最相关的点是：
- 在 `tex/literature/CISPO/cpt.tex` 的 “Efficient RL Scaling with `\\method{}`” 里出现一种“放弃传统 trust region 约束，改为 clipping importance weights 来稳定训练”的路线。
- 可以把它作为“实践派路线”的参考，与 TRM/Theory 系列的“trust region / bound”路线形成对照。

---

### DPPO（example）

入口：`tex/literature/DPPO/example_paper.tex`

更像模板/示例文件，用于补齐引用与写作结构，不建议作为理论来源。

---

## Technical Reports（`.md`）

### Theory Part 1：Why Off-Policy Breaks RL — SGA Analysis Framework

入口：`tex/literature/Theory/1-Why Off-Policy Breaks RL An SGA Analysis Framework.md`

#### 核心贡献
- SGA Lemma 把单步期望进展拆成三项：
  - True progress（真梯度项）
  - Bias term（系统偏差方向项）
  - Noise penalty（方差/噪声惩罚项）
- 明确给出度量工具：
  - bias 用 TV distance 控制（期望差的上界）
  - variance 用 chi^2 divergence 控制（IS 二阶矩）
- 把“off-policy 会崩”解释为：不是一个单一问题，而是 bias/variance 两种失败模式会分别杀死 Term B 或 Term C。

#### 必看公式（建议在写论文/证明时直接复用的模板）
- SGA Lemma 的三项分解（Term A/B/C），用于把任何 estimator 的问题拆成：
  - 是否系统偏向（Bias）
  - 是否二阶矩爆炸（Var）
- “TV 控 bias、chi^2 控 var”的原因：
  - expectation difference bound 用 TV
  - importance sampling variance 由 `E[ρ^2]`（chi^2）主导

#### 证明/推导形态
- 以 smoothness + 期望展开给出 lemma 推导（偏数学分析风格），适合作为所有后续讨论的统一框架。

---

### Theory Part 2：Token vs Sequence-level Correction（系统性 bias-variance 对比）

入口：`tex/literature/Theory/2-Applying the SGA Framework Token v.s. Sequence-level Correction.md`

#### 主线结论
- Seq-IS：无偏，但 `E_μ[ρ(y)^2]` 在长序列下指数爆炸（方差灾难）。
- Token-IS（PPO/GRPO 风格）：方差多项式级，但会产生结构性 bias（核心是 deterministic transition 导致 state occupancy mismatch 不能被 token-wise ratio 修正）。
- SNIS、truncation 等是把“稳定性”与“有效样本量”重新拉回可用区间的统计工具，但不会消除根本 tradeoff。

#### 推导路线（读这篇时抓主线即可）
1. 先把目标统一写成 `g = E_π[f(y)]`，对比不同 estimator 的 `E_μ[ĝ]`（bias）与 `E_μ[ρ^2]`（variance）。
2. Seq-IS：用 tower property/逐步条件期望把 `E_μ[ρ(y)^2]` 展开成 per-token `1+chi^2` 的连乘，得到指数上界。
3. Token-IS：通过打断乘积把二阶矩从指数变成多项式，但代价是把 state distribution ratio 当成 1，从而引入 bias。
4. SNIS / truncation：用分母归一化或 clip 来避免单样本 dominate；这类方法引入可控 bias 或降低 ESS。

---

### Theory Part 3：Trust Region Optimization via Sequence Masking（Seq-MIS / Geo-Mask）

入口：`tex/literature/Theory/3-Trust Region Optimization via Sequence Masking.md`

#### 必看定义
- Seq-MIS：hard trust region via rejection（`I(ρ(y) <= C)`）
- Geo-Mask：用 `ρ_geo` 做 length-invariant gate（双侧阈值，且可以作为纯过滤，不乘 ρ）

#### 关键洞见（与 TRM 关系紧密）
- “Seq ratio 是 extensive quantity”会带来结构性长度偏置；Geo-Mask 通过几何均值把约束变成 per-token 的 intensive quantity，从而 length-invariant。
- 这与 TRM “阈值不随 T 变化”的设计目标在精神上对齐，但两者 gate 统计量不同（ratio-based vs divergence-based）。

#### 证明/推导主线
1. 从 TRPO 下界/“surrogate 只在 trust region 内成立”出发，解释为什么要 hard reject。
2. 给出 Seq-MIS：把 `ρ(y)` 超界样本当作 OOD，直接拒绝以避免 “clip 仍然更新 OOD” 的污染。
3. 指出 Seq-MIS 的结构性长度偏置来自 `ρ(y)=∏ρ_t` 的 extensive 性质。
4. 引入 Geo-Mask：改用 `ρ_geo = exp(mean log ρ_t)` 作为 gate，得到 length-invariant 的 hard trust region。
5. 解释为何 Geo-Mask 可以作为纯过滤（不乘 IS 权重），以及如何与 token-level IS 组合成 hybrid estimator。

---

### TIS.md：sampler/learner mismatch 与 truncated IS

入口：`tex/literature/TIS.md`

#### 核心内容
- 从 REINFORCE 更新式出发，把混合系统写成 `E_{a~π_sampler}[ R(a) ∇ log π_learner(a) ]`，指出这是 off-policy。
- 给出最朴素的修正：乘 `π_learner/π_sampler`，以及 truncated importance sampling `min(π_learner/π_sampler, C)`。
- 讨论如何把同样思想迁移到 PPO 的 surrogate 上（注意这里会出现 “PPO ratio” 与 “mismatch ratio” 两套 ratio）。

#### 与本仓库实现对应
- `--vllm_is_correction_type tis`：token-level clamp（与文中 truncated IS 对齐）。

---

### MIS.md：mismatch -> collapse 诊断 + MIS 概念

入口：`tex/literature/MIS.md`

#### 与 TRM/MIS 关系最相关的段落
- 明确指出 token-level 修正无法解决 state occupancy mismatch（deterministic transition 使得轨迹一旦分叉后续状态分布完全不同）。
- 给出 MIS 作为 “mask sequences where ρ(y) > C” 的序列级 hard gate，并强调 token-level MIS 依然会崩。
- 该文的 “MIS/Seq-MIS/Geo-Mask” 观点与 Theory Part 2/3 形成一条连续的理论链。

#### 与本仓库实现对应（保守）
- 当前 `seq-mask-tis` 更接近 Geo-Mask 风格的 gate，而不是 `ρ(y)` 形式的 Seq-MIS；要做 Seq-MIS 需要显式构造 sequence ratio gate（并处理长度偏置问题）。

---

### IcePop.md：MoE mismatch 直觉 + IcePop objective

入口：`tex/literature/IcePop.md`

#### 核心内容
- 更偏“讲清楚为什么 MoE mismatch 会更糟”，并用一个 compounding discrepancy 的 lemma/条件解释为何 mismatch 会自激增长。
- 给出 IcePop 的目标函数与 mask 形式，方便快速对照实现。

---

### Brief Introduction of Policy Gradient In LLM Reasoning：PG 定理与 surrogate objective（写作级模板）

入口：`tex/literature/Brief Introduction of Policy Gradient In LLM Reasoning.md`

#### 适用场景
- 当你需要在论文里从“序列生成是 MDP”写到“policy gradient theorem/token-level 分解/如何落到 autograd 的 surrogate loss”，这份文档基本就是可复用的模板。

#### 核心内容（按证明链路组织）
- Notation & Objective：把 LLM policy 既写成 token policy `π(·|s)`，也写成 sequence likelihood `π(y|x)=∏ π(y_t|x,y_{<t})`。
- Policy Gradient Theorem（序列级）与 token-level 分解：
  - 先给 `∇_θ J(θ)` 的序列级表达，再用 `log π(y|x)=Σ_t log π(a_t|s_t)` 推出 token-level 形式。
- Surrogate objective（为了自动微分框架实现）：
  - 解释为什么要把采样分布固定在 `π_{θ_k}`，并在 loss 里显式写 `π_θ/π_{θ_k}` 作为 ratio，从而让 “对 loss 做一次 exact gradient descent” 等价于 “做一次 policy gradient ascent”。

#### Proof 在哪里
- 文档直接以内嵌 theorem + 推导的方式给出（适合引用其推导结构，而不是引用具体措辞）。

---

### Theory of On-Policy Distillation：OPD 的目标函数与 policy gradient（含 proof）

入口：`tex/literature/Theory of On-Policy Distillation.md`

#### 问题设定
- 给定 prompt `x~D`，每个 `x` 对应一个 teacher policy `π_{T(x)}`。
- 目标：最小化 `KL(π_θ(·|x) || π_{T(x)}(·|x))`（reverse KL），等价于最大化一个 entropy-regularized 的序列级回报。

#### 核心公式/结论
- 目标函数把 OPD 解释为“奖励是 teacher logprob 的 RL”：
  - immediate reward `r(s_t,a_t) = log π_{T(x)}(a_t|s_t)`（并带 entropy 项）
- 给出 OPD 的 policy gradient 形式，并列出多种 advantage 选择：
  - full-trajectory log ratio
  - log-ratio-to-go
  - baselined 版本
  - 以及把未来项写成 KL 的等价形式

#### 证明路线（Roadmap）
1. 从 `KL(π_θ || π_T)` 的定义出发，把优化写成对 `E_{y~π_θ}` 的期望。
2. 用 log-derivative trick 得到 `E[ advantage * ∇ log π_θ ]` 结构。
3. 通过条件期望/塔式性质把 “未来 log ratio” 与 “未来 KL” 联系起来，得到多种等价 advantage 形式。
4. 说明 baseline 不改变期望梯度（经典 `E[b(s)∇logπ]=0`）。
