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

## 1. DPPO（Rethinking Trust Region in the LLM Regime / Divergence Proxies）

> 本节目标：把 DPPO 里**所有带 proof/长推导的理论内容**做“可迁移/可引用”视角的整理，方便判断哪些值得进入本文（合规见 `tex/plan.md`）。

### DPPO-0：Proof Inventory（完整性检查）

| ID | 类型 | 来源 | 对应 label / 位置 | 是否有显式 proof |
|---|---|---|---|---|
| DPPO-1 | Theorem | `tex/literature/DPPO/paper/llm_bound.tex` | `lem:llm_identity` | ✅（见 `app.tex`） |
| DPPO-2 | Lemma | `tex/literature/DPPO/paper/app.tex` | `lem:sequence_tv_bound` | ✅（同处） |
| DPPO-3 | Theorem | `tex/literature/DPPO/paper/llm_bound.tex` | `thm:llm_tr_bound` | ✅（见 `app.tex`） |
| DPPO-4 | Derivation（tighter bound） | `tex/literature/DPPO/paper/app.tex` | `app:tighter_bound` | ✅（推导段落） |
| DPPO-5 | Derivation（梯度同构） | `tex/literature/DPPO/paper/app.tex` | `app:compare_classical_rl` | ✅（推导段落） |
| DPPO-6 | Proof（coarse-grain lower bound） | `tex/literature/DPPO/paper/app.tex` | `app:divergence_lower_bounds` | ✅（TV/KL 两段 proof） |
| DPPO-7 | Identity（TV=E|r-1|/2） | `tex/literature/DPPO/paper/method.tex` | `eq:tv_as_expectation` | ⚠️（可 3 行推导） |
| DPPO-8 | Cited theorem（TRPO/TV bound） | `tex/literature/DPPO/paper/background.tex` | `schulman2015trust` | ⚠️（外部结果，DPPO 只引用） |

> 说明：`app.tex` 后半部分还有大量实验与工程细节（稳定性实验设定、clipped tokens 等），不属于“证明资产”，本节不强行纳入。

---

### DPPO-1：Performance Difference Identity for LLMs（`lem:llm_identity`）

- **Source**：`tex/literature/DPPO/paper/llm_bound.tex`（statement），`tex/literature/DPPO/paper/app.tex`（proof: “Proof of Performance Difference Identity”）。
- **Statement（paraphrase）**：在有限时域、sequence-level reward（$\gamma=1$）下，
  \[
    \mathcal{J}(\pi_{\mathrm{train}})-\mathcal{J}(\pi_{\mathrm{roll}})
    =
    L_{\pi_{\mathrm{roll}}}'(\pi_{\mathrm{train}}) - \Delta(\pi_{\mathrm{roll}},\pi_{\mathrm{train}}),
  \]
  其中 $L'$ 是把“未来比率项”置 1 得到的一阶 surrogate，而 $\Delta$ 收集了未来 ratio 的高阶 remainder（包含 $\prod_{j>t} \rho_j$）。
- **Proof outline（用于判断可迁移性）**
  1. 把性能差写成对所有序列 $y$ 的求和：$\sum_y (\pi_{\mathrm{train}}(y)-\pi_{\mathrm{roll}}(y))R(y)$。
  2. 用**乘积差的 telescoping identity**把 $\pi_{\mathrm{train}}(y)-\pi_{\mathrm{roll}}(y)$ 展开为对 $t$ 的求和：前缀用 rollout、当前项用差值、后缀用 train（这一步是整个 proof 的承重）。
  3. 重新整理为 $\E_{y\sim \pi_{\mathrm{roll}}}[\cdot]$ 的形式，出现 $(\rho_t-1)\prod_{j>t}\rho_j$。
  4. 在期望里“加减 1”（把 $\prod_{j>t}\rho_j$ 分解成 $1 + (\prod_{j>t}\rho_j-1)$），从而分离出 surrogate $L'$ 与 remainder $\Delta$。
  5. 得到 theorem 的最终形式。
- **Technique tags**：telescoping（product difference）/ change-of-measure / “future ratio term” remainder。
- **U/O/C**：**O**（经典 LLM regime PDI 变体；本文可**引用 statement**但不能复现其 proof）。
- **本文落点建议**：
  - 若要在本文使用：Preliminaries/Related Work 给 statement-only + citation 即可。
  - 若要做本文“prefix-causal”延伸：必须改写 remainder 的结构（从 suffix ratio 变成 prefix causal measure），并给**本文原创 proof**（附录）。

---

### DPPO-2：Bound on Sequence-Level TV Divergence（`lem:sequence_tv_bound`）

- **Source**：`tex/literature/DPPO/paper/app.tex`（lemma + proof）。
- **Statement（paraphrase）**：两策略诱导的 length-$N$ 序列分布的 TV，可被逐步 token-level TV 的期望和控制：
  \[
    D_{\mathrm{TV}}(\pi_{\mathrm{roll},N} \| \pi_{\mathrm{train},N})
    \le \sum_{t=1}^{N}\E_{s_t\sim \pi_{\mathrm{roll}}}\big[D_{\mathrm{TV}}(\pi_{\mathrm{roll}}(\cdot|s_t)\|\pi_{\mathrm{train}}(\cdot|s_t))\big].
  \]
- **Proof outline**
  1. 从 $2D_{\mathrm{TV}}(P\|Q)=\sum_y|P(y)-Q(y)|$ 出发，$P,Q$ 是两条序列概率乘积。
  2. 用乘积差 telescoping identity：$\prod a_t - \prod b_t = \sum_t (\prod_{k<t}a_k)(a_t-b_t)(\prod_{j>t}b_j)$。
  3. 对绝对值用 triangle inequality，上界为 $\sum_t$ 的和。
  4. 对每个 $t$，把后缀 $\prod_{j>t} \pi_{\mathrm{train}}(\cdot)$ 的求和积分掉（$\sum \pi =1$），只剩前缀 rollout 的权重。
  5. 内层 $\sum_{y_t}|\pi_{\mathrm{roll}}-\pi_{\mathrm{train}}|$ 恰是 $2D_{\mathrm{TV}}$；外层前缀求和变成对 $s_t$ 的 rollout 期望。
- **Technique tags**：telescoping / triangle inequality / integrate-out future（把 suffix 归一化掉）。
- **U/O/C**：**O**（通用 building block；本文只需引用）。
- **本文落点建议**：适合做一个 cited remark（解释“sequence TV 可由 per-step TV 累积控制”），但不要在本文复现 proof。

---

### DPPO-3：Policy Improvement Bound for LLMs（`thm:llm_tr_bound`）

- **Source**：`tex/literature/DPPO/paper/llm_bound.tex`（statement），`tex/literature/DPPO/paper/app.tex`（proof: “Proof of Policy Improvement Bound”）。
- **Statement（paraphrase）**：把 remainder $\Delta$ 上界化后得到一个 trust-region 风格下界：
  \[
    \mathcal{J}(\pi_{\mathrm{train}})-\mathcal{J}(\pi_{\mathrm{roll}})
    \ge
    L_{\pi_{\mathrm{roll}}}'(\pi_{\mathrm{train}})
    - O\!\left(\xi\,T(T-1)\,[D_{\mathrm{TV}}^{\max}]^2\right),
  \]
  其中 $\xi=\max_y|R(y)|$，$D_{\mathrm{TV}}^{\max}$ 是单步 TV 的 worst-case 上界。
- **Proof outline**
  1. 从 DPPO-1 的 exact identity：$\mathcal{J}(\pi_{\mathrm{train}})-\mathcal{J}(\pi_{\mathrm{roll}})=L'-\Delta$。
  2. 用 $|R|\le \xi$ 与 triangle inequality：把 $|\Delta|$ 上界为 $\sum_t \E[|\rho_t-1|\cdot |1-\rho_{>t}|]$。
  3. 用 tower property 条件在 $y_{\le t}$ 上，把“未来项”的绝对值期望识别为 $2D_{\mathrm{TV}}$（未来 trajectory 分布的 TV）。
  4. 对未来 trajectory TV 用 DPPO-2（sequence-TV 由 per-step TV 累积）+ 再用 $D_{\mathrm{TV}}^{\max}$ 上界每一步，从而得到 $(T-t)D_{\mathrm{TV}}^{\max}$。
  5. 对当前步 $\E_{y_t\sim \pi_{\mathrm{roll}}}[|\rho_t-1|]=2D_{\mathrm{TV}}(\pi_{\mathrm{roll}}(\cdot|s_t)\|\pi_{\mathrm{train}}(\cdot|s_t))\le 2D_{\mathrm{TV}}^{\max}$。
  6. 汇总：$\sum_t (T-t)$ 给出 $T(T-1)/2$，得到 $\Delta \lesssim \xi T(T-1)[D_{\mathrm{TV}}^{\max}]^2$（常数按原文）。
- **Technique tags**：exact identity + remainder / tower property / TV-as-expectation / worst-case max TV。
- **U/O/C**：**O**（LLM trust region bound 的一个实现版本；本文可引用但不复现 proof）。
- **本文落点建议**：
  - 可以用作 Related Work / 背景对照：为什么 $T^2$ 依赖会松。
  - 若本文需要 tight bound，建议引用其 proof technique，并在 prefix gate 语境下写“effective horizon / acceptance-weighted”版本（本文原创）。

---

### DPPO-4：A Tighter Policy Improvement Bound（`app:tighter_bound`）

- **Source**：`tex/literature/DPPO/paper/app.tex`（derivation）。
- **核心点（paraphrase）**：用 $D_{\mathrm{TV}}\le 1$ 替换“未来 TV 用 $(T-t)D_{\mathrm{TV}}^{\max}$”这一松的 worst-case，从而把 $T^2$ 变为对 per-step TV 的线性和：
  \[
    \Delta \;\lesssim\; 4\xi\,\E_{y\sim \pi_{\mathrm{roll}}}\Big[\sum_{t}D_{\mathrm{TV}}(\pi_{\mathrm{roll}}(\cdot|s_t)\|\pi_{\mathrm{train}}(\cdot|s_t))\Big].
  \]
  然后与二次 bound 取 min 得到 composite bound。
- **推导 outline**
  1. 从中间不等式（把 $\Delta$ 写成 $\E[|\rho_t-1|\cdot \E|1-\rho_{>t}|]$）。
  2. 把 $\E|1-\rho_{>t}| = 2D_{\mathrm{TV}}(\text{future traj}) \le 2$（因为 TV $\le 1$）。
  3. 剩下 $\sum_t \E|\rho_t-1|$，再用 $\E|\rho_t-1|=2D_{\mathrm{TV}}(\text{token})$。
  4. 得到线性于“累计 per-token TV”的 bound；与 $T^2$ bound 取 min。
- **Technique tags**：use TV≤1 to kill horizon / “linear-in-sum-divergence” remainder control。
- **U/O/C**：**O**（非常好的“松紧开关”技巧，但属于引用性推导）。
- **本文落点建议**：可作为“长视野下 $T^2$ bound 很松”的对照解释；若本文需要 tight bound，建议在 prefix gate 语境下写“effective horizon / acceptance-weighted”版本（本文原创）。

---

### DPPO-5：Comparing Surrogate Objectives with Classical RL（`app:compare_classical_rl`）

- **Source**：`tex/literature/DPPO/paper/app.tex`（derivation）。
- **目标（paraphrase）**：说明 LLM surrogate 的梯度形式与 classical TRPO surrogate 的梯度同构（把 $\rho^{\pi}$ 的 state 期望换成 trajectory 上的 $\sum_t$）。
- **推导 outline**
  1. 用恒等式 $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$。
  2. classical surrogate：$\nabla L = \E_{s,a\sim \pi_{\mathrm{roll}}}\big[\frac{\pi_\theta}{\pi_{\mathrm{roll}}}\nabla\log\pi_\theta \cdot A\big]$。
  3. LLM surrogate：$\nabla L' = \E_{y\sim \pi_{\mathrm{roll}}}\big[\sum_t \frac{\pi_\theta(y_t|s_t)}{\pi_{\mathrm{roll}}(y_t|s_t)}\nabla\log\pi_\theta(y_t|s_t)\cdot R(y)\big]$（$-1$ 项梯度为 0）。
  4. 若把 $R(y)$（或 $R(y)-V(x)$）看作 sequence-level advantage，则两者结构一致。
- **Technique tags**：score function trick / surrogate-gradient isomorphic mapping。
- **U/O/C**：**O**（解释性推导；适合引用，不适合搬运进本文作为“新结论”）。
- **本文落点建议**：可作为 Related Work 中“LLM surrogate 合法性”的一句话支持，但应 statement-only + citation。

---

### DPPO-6：Approximations as Lower Bounds of True Divergence（`app:divergence_lower_bounds`）

- **Source**：`tex/literature/DPPO/paper/app.tex`（TV/KL 两段 proof + equality 条件）。
- **Statement（paraphrase）**：对任意 vocabulary partition $\mathcal{C}$，在 coarsened space 上计算的 divergence 是 full-vocab divergence 的 lower bound。
  - TV：由 triangle inequality 得到 $D_{\mathrm{TV}}^{\mathcal{C}} \le D_{\mathrm{TV}}$。
  - KL：由 log-sum inequality 得到 $D_{\mathrm{KL}}^{\mathcal{C}} \le D_{\mathrm{KL}}$。
- **Proof outline（TV）**
  1. partition 后的差值是组内差值之和：$\Delta(C_j)=\sum_{a\in C_j}(\pi_{\mathrm{roll}}(a)-\pi_{\mathrm{train}}(a))$。
  2. 用 $|\sum u_a| \le \sum |u_a|$（triangle inequality）逐组放松。
  3. 对所有组求和，得到 coarsened TV ≤ full TV。
  4. equality 条件：每个组内差值同号（否则会有抵消导致 strict inequality）。
- **Proof outline（KL）**
  1. 对每个组应用 log-sum inequality：$\sum x_a\log(x_a/y_a)\ge (\sum x_a)\log(\sum x_a/\sum y_a)$。
  2. 把组内 token 的 KL 下界化为组级 KL，最后对组求和得到 coarsened KL ≤ full KL。
  3. equality 条件：组内 ratio $x_a/y_a$ 为常数。
- **Technique tags**：partition / triangle inequality / log-sum inequality / lower bound & equality characterization。
- **U/O/C**：**O**（引用型工具定理；本文若要用，必须只引用 statement）。
- **本文落点建议**：可在 Preliminaries 用 remark 引用“Binary/Top-K 是 lower bound”，但不能复现 proof；若要做 prefix-level coarse-grain，需要本文新增点与 proof。

---

### DPPO-7：TV as expectation of ratio residual（`eq:tv_as_expectation`）

- **Source**：`tex/literature/DPPO/paper/method.tex`（Eq. `eq:tv_as_expectation`）。
- **Statement（paraphrase）**：令 $r(a)=\frac{\pi_{\mathrm{train}}(a|s)}{\pi_{\mathrm{roll}}(a|s)}$，则
  \[
    D_{\mathrm{TV}}(\pi_{\mathrm{roll}}(\cdot|s)\|\pi_{\mathrm{train}}(\cdot|s))
    = \tfrac{1}{2}\E_{a\sim \pi_{\mathrm{roll}}(\cdot|s)}[|r(a)-1|].
  \]
- **3 行推导 outline**
  1. $D_{\mathrm{TV}}=\tfrac12 \sum_a |\pi_{\mathrm{roll}}(a)-\pi_{\mathrm{train}}(a)|$。
  2. 代入 $\pi_{\mathrm{train}}(a)=\pi_{\mathrm{roll}}(a)\,r(a)$：得到 $\tfrac12\sum_a \pi_{\mathrm{roll}}(a)|r(a)-1|$。
  3. 识别为期望 $\tfrac12 \E_{\pi_{\mathrm{roll}}}[|r-1|]$。
- **Technique tags**：RN derivative / TV-as-expectation identity。
- **U/O/C**：**O**（引用性恒等式）。
- **本文落点建议**：适合做一句 cited remark：解释“ratio clipping 是 TV 的单样本 proxy”。

---

### DPPO-8：Classical trust region bound（`schulman2015trust`）

- **Source**：`tex/literature/DPPO/paper/background.tex`（TRPO/TV bound theorem）。
- **说明**：这是外部经典结果（TRPO / Achiam 等），DPPO 论文把它作为背景；我们在本文中若要用，必须按 `tex/plan.md`：**statement-only + citation**，不要复现 proof。
- **Technique tags（可借鉴）**：performance difference identity / coupling vs Pinsker / max-divergence penalty。
- **U/O/C**：**O**（引用型）。

---

## 2. TRM（Trust Region Masking for Long-Horizon LLM-RL）

本节目标：把 `tex/literature/TRM/main_arxiv.tex` 中所有**定理/引理/命题/推导 proof**按“可迁移”视角拆解成 proof outline，方便：
1) 检查 theory.md 是否遗漏了承重结论；2) 判断哪些技巧可用于本文 prefix-causal trust region 的原创证明。

### TRM-0：Proof Inventory（完整性检查）

| ID | 类型 | 来源 | 对应 label / 位置 | 是否有显式 proof |
|---|---|---|---|---|
| TRM-1 | Lemma | `tex/literature/TRM/main_arxiv.tex` | `lem:kl-chain-app` | ✅ |
| TRM-2 | Corollary | `tex/literature/TRM/main_arxiv.tex` | `cor:kl-chain-bound` | ✅ |
| TRM-3 | Lemma | `tex/literature/TRM/main_arxiv.tex` | `lem:martingale-app` | ✅ |
| TRM-4 | Lemma | `tex/literature/TRM/main_arxiv.tex` | `lem:advantage-bound` | ✅（full proof） |
| TRM-5 | Lemma | `tex/literature/TRM/main_arxiv.tex` | `lem:context-shift` | ✅（full proof） |
| TRM-6 | Theorem | `tex/literature/TRM/main_arxiv.tex` | `thm:pinsker-marginal` | ✅（proof section） |
| TRM-7 | Theorem | `tex/literature/TRM/main_arxiv.tex` | `thm:mixed` | ✅（proof section） |
| TRM-8 | Theorem | `tex/literature/TRM/main_arxiv.tex` | `thm:coupling` | ✅（proof section） |
| TRM-9 | Theorem | `tex/literature/TRM/main_arxiv.tex` | `thm:adaptive` | ✅（`app:proof-adaptive`，step-wise） |
| TRM-10 | Theorem | `tex/literature/TRM/main_arxiv.tex` | `thm:unified` | ✅（直接取 min） |
| TRM-11 | Proposition | `tex/literature/TRM/main_arxiv.tex` | `prop:no-pure-seq` | ✅ |
| TRM-12 | Proposition | `tex/literature/TRM/main_arxiv.tex` | `prop:token-masking-fails` | ⚠️（陈述为主，证明=一句话） |
| TRM-13 | Theorem | `tex/literature/TRM/main_arxiv.tex` | `thm:trm-guarantee` | ✅（短 proof） |
| TRM-14 | Derivation | `tex/literature/TRM/main_arxiv.tex` | `app:k3`（$k_2,k_3$） | ✅（推导/验证） |
| TRM-15 | Proposition | `tex/literature/TRM/main_arxiv.tex` | `prop:ln-trm-guarantee` | ✅ |

---

### TRM-1：KL Chain Rule（`lem:kl-chain-app`）

- **Source**：`tex/literature/TRM/main_arxiv.tex`，Appendix “Proofs of Foundational Lemmas”。
- **Statement（paraphrase）**：rollout vs train 的 prefix context 分布 KL 能拆成逐 token 条件 KL 的期望和：
  \[
    D_{\mathrm{KL}}(d_t^{\pi_{\mathrm{roll}}}\|d_t^{\pi_\theta})
    =\sum_{s=1}^{t-1}\E_{c_s\sim d_s^{\pi_{\mathrm{roll}}}}\!\big[D_{\mathrm{KL}}(\pi_{\mathrm{roll}}(\cdot|c_s)\|\pi_\theta(\cdot|c_s))\big].
  \]
- **Proof outline**
  1. 写出 $d_t^\pi(x,y_{<t}) = P(x)\prod_{s<t}\pi(y_s|c_s)$（trajectory factorization）。
  2. KL 定义：$\E_{d_t^{\pi_{\mathrm{roll}}}}[\log \frac{P\prod \pi_{\mathrm{roll}}}{P\prod \pi_\theta}]$，$P(x)$ 相消。
  3. log 把乘积变成求和：$\sum_{s<t}\log \frac{\pi_{\mathrm{roll}}(y_s|c_s)}{\pi_\theta(y_s|c_s)}$。
  4. 对每个 $s$，把 $y_{s+1:t-1}$ marginalize 掉，变为对 $c_s\sim d_s^{\pi_{\mathrm{roll}}}$ 的期望；内层期望正是 token-level KL。
- **Technique tags**：trajectory factorization / log product→sum / marginalization。
- **U/O/C**：**O**（标准链式分解；可引用）。
- **本文落点建议**：若本文需要 KL-route 的 context shift bound，可以 statement-only 引用该 chain rule。

---

### TRM-2：KL chain bound（`cor:kl-chain-bound`）

- **Source**：同上（corollary + short proof）。
- **Statement（paraphrase）**：若单步 token KL 的 worst-case 上界为 $\delta$，则
  \[
    D_{\mathrm{KL}}(d_t^{\pi_{\mathrm{roll}}}\|d_t^{\pi_\theta})\le (t-1)\delta,\qquad
    D_{\mathrm{KL}}^{\mathrm{seq}}\le T\delta.
  \]
- **Proof outline**：每项期望 $\le \max \le \delta$，直接求和。
- **U/O/C**：**O**。

---

### TRM-3：Martingale Property（`lem:martingale-app`）

- **Statement（paraphrase）**：在 rollout policy 下，advantage 的条件期望为 0：
  \[
    \E_{y_t\sim \pi_{\mathrm{roll}}(\cdot|c_t)}[A_t^{\pi_{\mathrm{roll}}}(c_t,y_t)]=0.
  \]
- **Proof outline**：按定义 $A_t=Q_t-V_t$，而 $V_t=\E_{y_t\sim\pi_{\mathrm{roll}}}[Q_t]$，相消。
- **Technique tags**：definition unwind / tower property（隐含）。
- **U/O/C**：**O**（building block）。

---

### TRM-4：Advantage Bound（`lem:advantage-bound`，full proof）

- **Statement（paraphrase）**：当 $R\in[0,1]$ 时，$g_t(c_t)=\E_{y_t\sim\pi_\theta}[A_t^{\pi_{\mathrm{roll}}}]$ 的无穷范数由 token divergence 控制：
  \[
    \|g_t\|_\infty \le 2\min\big(1,\epsilon,\sqrt{\delta/2}\big).
  \]
- **Proof outline**
  1. 用 TRM-3：$g_t=\E_{\pi_\theta}[A_t]-\E_{\pi_{\mathrm{roll}}}[A_t]$。
  2. 展开成 $\sum_{y_t}(\pi_\theta-\pi_{\mathrm{roll}})A_t$。
  3. 用 $R\in[0,1]$ 推出 $|A_t|\le 1$（$Q,V\in[0,1]$）。
  4. 于是 $|g_t|\le \sum_{y_t}|\pi_\theta-\pi_{\mathrm{roll}}| = 2D_{\mathrm{TV}}(\pi_\theta(\cdot|c_t),\pi_{\mathrm{roll}}(\cdot|c_t))$。
  5. 再用 $D_{\mathrm{TV}}\le 1$ 与 Pinsker：$D_{\mathrm{TV}}\le \sqrt{D_{\mathrm{KL}}/2}$，得到 cap 形式。
- **Technique tags**：advantage as signed measure / TV bound / Pinsker cap。
- **U/O/C**：**O**（可引用 building block；proof technique 可借鉴）。

---

### TRM-5：Context Shift（`lem:context-shift`，full proof）

- **Statement（paraphrase）**：prefix context visitation 分布的 TV shift 有 5 条 route 上界（trivial / coupling / Pinsker-on-marginal-KL / data-processing / Pinsker-on-seq-KL），最后取 min。
- **Proof outline（五条 route）**
  1. **Trivial**：TV $\le 1$。
  2. **Coupling（induction）**：
     - base：$t=1$ 时分布相同；
     - step：写 $d_{t+1}^\pi = d_t^\pi\cdot \pi(\cdot|c_t)$，对差值用 triangle inequality 拆成“前缀差 + 条件分布差”，得到递推：
       \[
       \|d_{t+1}^{\pi_\theta}-d_{t+1}^{\pi_{\mathrm{roll}}}\|_{\mathrm{TV}}
       \le \|d_t^{\pi_\theta}-d_t^{\pi_{\mathrm{roll}}}\|_{\mathrm{TV}} + \E_{c_t\sim d_t^{\pi_{\mathrm{roll}}}}[D_{\mathrm{TV}}(\pi_\theta(\cdot|c_t),\pi_{\mathrm{roll}}(\cdot|c_t))].
       \]
     - 用 $\Dtvtok\le \epsilon$ 得到 $(t-1)\epsilon$。
  3. **Pinsker on marginal KL**：用 TRM-1 + TRM-2 得到 $D_{\mathrm{KL}}(d_t^{\pi_{\mathrm{roll}}}\|d_t^{\pi_\theta})\le (t-1)\delta$，再 Pinsker。
  4. **Data processing（TV）**：$d_t$ 是 full trajectory 分布的边缘分布，marginalization 不增 TV，所以 $\|d_t^\theta-d_t^{\mathrm{roll}}\|_{\mathrm{TV}}\le D_{\mathrm{TV}}^{\mathrm{seq}}$。
  5. **Pinsker on sequence KL**：用 $D_{\mathrm{KL}}(d_t^{\mathrm{roll}}\|d_t^\theta)\le D_{\mathrm{KL}}^{\mathrm{seq}}$ 再 Pinsker。
- **Technique tags**：product measure TV recursion / KL chain rule / data processing / multi-route min。
- **U/O/C**：**O**（典型“context shift factor” control）。

---

### TRM-6：Pinsker-Marginal Bounds（`thm:pinsker-marginal`）

- **核心结构**：把 master decomposition
  \[
    |\mathrm{Error}|\le \sum_t 2\|g_t\|_\infty\cdot \|d_t^\theta-d_t^{\mathrm{roll}}\|_{\mathrm{TV}}
  \]
  中两因子分别走：
  - advantage factor：用 $\|g_t\|_\infty\le 2\min(1,\sqrt{\delta/2})$（KL route）或 $2\min(1,\epsilon)$（TV route）；
  - context shift：用 Pinsker-on-marginal-KL：$\min(1,\sqrt{(t-1)\delta/2})$。
- **Proof outline**：直接 substitution + 求和；小散度 regime 用 $\sum_{k=0}^{T-1}\sqrt{k}\le \int_0^T\sqrt{x}dx$ 得到 $O(T^{3/2}\delta)$。
- **Technique tags**：plug-in bound / integral approximation for sums。
- **U/O/C**：**O**（可引用 bound family，不复现 proof）。

---

### TRM-7：Mixed Bounds（`thm:mixed`）

- **核心结构**：用 sequence-level divergence 给一个**不随 t 增长**的 context shift cap（通过 data processing 或 “marginal KL ≤ seq KL”）。
- **Proof outline**：
  1. context shift：$\|d_t^\theta-d_t^{\mathrm{roll}}\|_{\mathrm{TV}}\le \min(1,\sqrt{D_{\mathrm{KL}}^{\mathrm{seq}}/2})$ 或 $\min(1,D_{\mathrm{TV}}^{\mathrm{seq}})$。
  2. advantage factor 取 KL 或 TV 路径。
  3. 求和得到 $O(T)$ scaling。
- **Technique tags**：uniform-in-t cap / data processing。
- **U/O/C**：**O**。

---

### TRM-8：Coupling Bound（`thm:coupling`）

- **核心结构**：纯 TV 路径（避免 KL/Pinsker），但 context shift 随 $(t-1)\epsilon$ 线性增长；靠 $\min(1,\cdot)$ cap 在大 $T$ 时转为 $O(T)$。
- **Proof outline**：master decomposition + advantage bound（TV）+ context shift（coupling）+ 两个 regime 分析（cap 是否激活）。
- **Technique tags**：coupling recursion + cap-induced regime split。
- **U/O/C**：**O**。

---

### TRM-9：Adaptive Bound（`thm:adaptive`）与其 step-wise proof（`app:proof-adaptive`）

- **Source**：`tex/literature/TRM/main_arxiv.tex`，Appendix “Proof of the Adaptive Bound”。
- **为什么值得单独记录**：这套 proof aesthetics（Step 1–5）是 TRM 的“黄金模板”，适合迁移到本文 prefix-causal trust region 的附录证明写法。
- **Step 1（`lem:exact-identity`）Exact Error Identity**
  - 关键：对重要性比率序列 $(\rho_1,\dots,\rho_T)$ 用 telescoping：
    \[
      \prod_{t=1}^T\rho_t-1 = \sum_{t=1}^T(\rho_t-1)\prod_{j>t}\rho_j.
    \]
  - 从而把 $J(\pi_\theta)-J(\pi_{\mathrm{roll}})$ 写成 $L' - \Delta$，其中 $\Delta$ 含 $(\rho_t-1)(1-\prod_{j>t}\rho_j)$。
- **Step 2 Tower property factorization（把前缀与后缀分离）**
  - 设 $A_t=|\rho_t-1|$，$B_t=|1-\prod_{j>t}\rho_j|$。
  - 条件在 $c_{t+1}$ 上后，$A_t$ 变成确定量；$B_t$ 的期望可识别为未来 trajectory 的 TV：
    \[
      \E[B_t|c_{t+1}] = 2D_{\mathrm{TV}}(P^{\pi_{\mathrm{roll}}}(\cdot|c_{t+1})\|P^{\pi_\theta}(\cdot|c_{t+1})).
    \]
  - 同时 $\E_{y_t\sim \pi_{\mathrm{roll}}}[|\rho_t-1|]=2D_{\mathrm{TV}}(\pi_{\mathrm{roll}}(\cdot|c_t)\|\pi_\theta(\cdot|c_t))$，汇总得到 $2\bar D_t$。
- **Step 3 Bounding future-trajectory TV（Route A/B/C + min）**
  - A：trivial $\le 1$；
  - B：Pinsker + future KL chain rule $\le \sqrt{(T-t)\delta/2}$；
  - C：coupling $\le (T-t)\epsilon$；
  - 取 min 得到 per-position 的未来 TV cap。
- **Step 4 Combine**
  - 把 $2\bar D_t$ 与未来 cap 相乘并对 $t$ 求和，得到
    \[
      B_{\mathrm{Adap}}^{*}
      =
      4\sum_{t=1}^T \bar D_t\cdot \min\!\Big(1,(T-t)\epsilon,\sqrt{(T-t)\delta/2}\Big).
    \]
- **Step 5 Relationship / strictness**
  - 说明设置 $\bar D_t$ 为 worst-case 时可 recover 先前 bounds；非均匀散度下严格更紧。
- **Technique tags**：importance ratio error identity / tower property / “future TV as expectation of |1-ratio|” / per-position min（route selection）。
- **U/O/C**：**O**（proof technique 极具参考价值；结论本身应引用而非搬运）。
- **本文落点建议**：重点迁移 Step-based proof aesthetics 与“per-position min + effective horizon”叙事，不迁移原结论。

---

### TRM-10：Unified Bound（`thm:unified`）

- **Statement（paraphrase）**：因为每条 bound family 独立成立，所以取最小值仍是 valid bound；并用 $L(\pi_\theta)-B^*$ 给出 monotonic improvement 的 sufficient condition。
- **Proof outline**：逻辑性（min of valid upper bounds is valid）。
- **U/O/C**：**O**。

---

### TRM-11：No “pure sequence” control for max token KL（`prop:no-pure-seq`）

- **Statement（paraphrase）**：不存在仅依赖 $\Dklseq$ 的函数 $f$ 能上界 $\Dkltokmax$（即“平均小不代表最大也小”）。
- **Proof outline（counterexample）**
  1. 构造一个罕见 context $c^*$（概率 $\epsilon$）；
  2. 令 divergence 只集中在 $c^*$：$\DKL(c^*)=1$，其他为 0；
  3. 则 $\Dkltokmax=1$ 恒定，而 $\Dklseq=\epsilon\to 0$。
- **Technique tags**：adversarial concentration / impossibility via rare event。
- **U/O/C**：**O**（很好的“为什么必须 max-based”叙事支撑，但属于引用性命题）。
- **本文落点建议**：如要用来支撑“必须控制 worst-case”，建议 statement-only + citation；避免在本文复现 proof。

---

### TRM-12：Token masking preserves vacuous bounds（`prop:token-masking-fails`）

- **Statement（paraphrase）**：token-level masking 改变的是梯度/优化方向，而不改变 rollout 与 train 的真实分布差异；因此 $\Dkltokmax$ 不会因 token masking 而下降，理论保证仍不可用。
- **Proof sketch（基本一句话）**：$\Dkltokmax$ 是由 $(\pi_\theta,\pi_{\mathrm{roll}})$ 的生成过程决定的分布量；在不改变 $\pi_\theta$ 分布本身（只在 loss 上乘 mask）的情况下，该 divergence 不会变。
- **Technique tags**：“objective masking ≠ distribution change”。
- **U/O/C**：**O**（叙事性命题）。

---

### TRM-13：TRM Guarantee（`thm:trm-guarantee`）

- **Statement（paraphrase）**：TRM 的 mask 通过 max-KL criterion 保证接受的样本满足 per-token divergence 上界；如果更强的“全局上界”成立，则可直接把统一误差界用于 monotonic improvement 条件。
- **Proof outline**：Part (1)(2) 是 by construction；Part (3) 直接套用 `thm:unified`（把 $\Dkltokmax$ 替换成阈值 $\delta$）。
- **U/O/C**：**O**（与本文机制相似，但要合规引用）。

---

### TRM-14：Sample-based estimators $k_2,k_3$（`app:k3`）

- **$k_3(\rho)=\rho-1-\log\rho$ 的验证 outline**
  1. rollout 下 $\E[\rho]=\E[\pi_\theta/\pi_{\mathrm{roll}}]=1$（RN derivative）。
  2. $\E[-\log\rho]=\E[\log(\pi_{\mathrm{roll}}/\pi_\theta)]=D_{\mathrm{KL}}(\pi_{\mathrm{roll}}\|\pi_\theta)$。
  3. 所以 $\E[k_3]=\E[\rho-1]-\E[\log\rho]=D_{\mathrm{KL}}$。
  4. 非负性来自 Jensen（$x-1-\log x\ge 0$）。
- **$k_2(\rho)=\tfrac12(\log\rho)^2$ 的角色**：对 max-filtering 友好（对 $\rho\to 0$ 与 $\rho\to\infty$ 对称），但不是 KL 的无偏估计。
- **Technique tags**：unbiased estimator / convexity / RN derivative。
- **U/O/C**：**O**（工程实现提示；理论上可引用其“无偏性”推导但不复现）。

---

### TRM-15：Length-Neutral TRM（LN-TRM）Guarantee（`prop:ln-trm-guarantee`）

- **Source**：`tex/literature/TRM/main_arxiv.tex`，Appendix “Length-Neutral Trust Region Masking”。
- **Statement（paraphrase）**：构造一个加权序列误差分数
  \[
    W(y)=\sum_{t=1}^{T}|\rho_t-1|\cdot w_t,\quad
    w_t=\min\big(1,(T-t)\epsilon,\sqrt{(T-t)\delta/2}\big),
  \]
  则在全局 divergence 上界成立时，$|\mathrm{Error}|\le 2\E_{\pi_{\mathrm{roll}}}[W(y)]$；对被接受轨迹，$W(y)$ 有显式上界（由阈值 $\delta_W$ 与 $Z(T)=\sum_t w_t$ 给出）。
- **Proof outline**：直接引用 TRM-9（Adaptive bound proof）中的最终形式；线性期望可把 $\sum_t$ 写成 $\E[W(y)]$；接受条件给出 $W(y)\le \delta_W Z(T)$。
- **Technique tags**：effective horizon weighting / length bias mitigation / reuse of Adaptive proof。
- **U/O/C**：**O**（可借鉴“effective horizon”的叙事，但本文要避免直接迁移其 theorem）。

---

## 3. Coverage Checklist（缺失项检查 & 论文落点筛选）

### 3.1 DPPO coverage（本次已补齐）

| Source item | 证明/推导是否已写入本文件 | 对应条目 |
|---|---|---|
| `lem:llm_identity`（PDI for LLMs） | ✅ | DPPO-1 |
| `lem:sequence_tv_bound`（sequence TV bound） | ✅ | DPPO-2 |
| `thm:llm_tr_bound`（policy improvement bound） | ✅ | DPPO-3 |
| `app:tighter_bound`（linear-in-T variant） | ✅ | DPPO-4 |
| `app:compare_classical_rl`（梯度同构） | ✅ | DPPO-5 |
| `app:divergence_lower_bounds`（TV/KL lower bound proofs） | ✅ | DPPO-6 |
| `eq:tv_as_expectation`（TV identity） | ✅ | DPPO-7 |
| `schulman2015trust`（背景定理） | ✅（说明性记录） | DPPO-8 |

### 3.2 TRM coverage（本次已补齐）

| Source item | 证明/推导是否已写入本文件 | 对应条目 |
|---|---|---|
| `lem:kl-chain-app` + `cor:kl-chain-bound` | ✅ | TRM-1 / TRM-2 |
| `lem:martingale-app` | ✅ | TRM-3 |
| `lem:advantage-bound`（full proof） | ✅ | TRM-4 |
| `lem:context-shift`（full proof） | ✅ | TRM-5 |
| `thm:pinsker-marginal` | ✅ | TRM-6 |
| `thm:mixed` | ✅ | TRM-7 |
| `thm:coupling` | ✅ | TRM-8 |
| `thm:adaptive` + `lem:exact-identity`（step-wise proof） | ✅ | TRM-9 |
| `thm:unified` | ✅ | TRM-10 |
| `prop:no-pure-seq` | ✅ | TRM-11 |
| `prop:token-masking-fails` | ✅（sketch） | TRM-12 |
| `thm:trm-guarantee` | ✅ | TRM-13 |
| `app:k3`（$k_2,k_3$ 推导） | ✅ | TRM-14 |
| `prop:ln-trm-guarantee` | ✅ | TRM-15 |

### 3.3 论文用途快速分桶（用于决定“该不该进正文/附录”）

- **可直接引用（Preliminaries/Related Work，statement-only + citation）**：
  - DPPO-1–DPPO-8（均为引用型，不复现 proof）
  - TRM-1–TRM-15（同理）
- **值得迁移其 proof technique（在本文设定下做原创 theorem + appendix proof）**：
  - TRM-9（Adaptive bound 的 step-wise proof 模板）
  - DPPO-1/DPPO-2（telescoping + integrate-out future 的结构）
  - DPPO-6（coarse-grain lower bound 的不等式套路：triangle/log-sum）
- **谨慎/避免作为本文承重结论**：
  - 任何“外部论文 theorem + proof”都不应出现在本文附录；只能引用 statement，或做本文扩展并给本文 proof（见 `tex/plan.md`）。
