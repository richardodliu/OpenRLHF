# TRM/DPPO 风格化重写与 TRM/DPPO 理论借鉴总计划

本文件是 `tex/plan/1.md`、`tex/plan/2.md`、`tex/plan/3.md` 的整合权威版本，用于指导论文 `tex/main.tex`（及其 `tex/main/*.tex`）的数学严谨性增强、证明结构与排版风格统一，以及对 DPPO 理论模块的系统性迁移与挂钩。

## 1. 目标与产物

### 1.1 总目标

1. 数学正确性优先：论文中每个主张都可追溯到清晰的定义、引理、定理与证明链路；证明中的等式与不等式转换都有合法依据。
2. 风格主对齐 TRM：定理陈述、证明语言、段落切分、Step/Case/Bound 组织方式、公式隔离策略，尽量复刻 `tex/literature/TRM/main_arxiv.tex` 的审稿友好写法。
3. 结构借鉴 DPPO：正文避免堆砌细推导，只保留关键定义、承重公式与高层逻辑；细推导、长证明与技术变体集中在附录，并使用统一模板组织。
4. 保守符号策略：保留现有核心符号体系（例如 $\pi_{\mathrm{roll}}$、$\pi_\theta$、$\mathcal{L}$、token/sequence/prefix 三层统计量），只修复不一致、增强可读性与严谨性，不做大规模“重命名式改论文”。

### 1.2 产物清单

1. 一份可执行的 Style Spec 与写作规范（本文件第 4 节）。
2. 一份可复用的诊断模板与缺陷清单格式（本文件第 5 节）。
3. 一份逐文件实施清单，覆盖正文与附录（本文件第 6 节）。
4. 一份 TRM/DPPO 理论借鉴、去重门禁与新增理论证明流程（本文件第 7 节）。
5. 明确的验收标准与自动化检索项（本文件第 8 节）。
6. 提交与迭代策略（本文件第 9 节）。

## 2. 输入材料与合并原则

### 2.1 输入来源

1. `tex/plan/1.md`：全局 TRM 主对齐与 DPPO 组织方式的论文重写计划（Style Spec、诊断、逐文件重写、附录证明模板）。
2. `tex/plan/2.md`：DPPO 迁移第一版，聚焦 DPPO 的 1+2，并强调必须做非平凡延伸与挂钩。
3. `tex/plan/3.md`：DPPO 迁移升级版，引入 U/O/C 分类、去重与矛盾门禁、结果池与结构落点。

### 2.2 合并优先级与去重规则

1. DPPO 迁移相关内容以 `tex/plan/3.md` 为主干，`tex/plan/2.md` 仅用于补充更清晰的解释性措辞与桥接动机。
2. 论文整体重写与 TRM 风格对齐以 `tex/plan/1.md` 为主干。
3. 若出现冲突，优先保留约束更强、验收更明确、门禁更严格的版本。
4. 对同一事项仅保留一处“最终口径”，避免平行表述导致执行歧义。

## 3. 硬约束与红线

### 3.1 论文结构红线

1. Preliminaries 的功能区分：Preliminaries 只放定义、记号、假设、引用公式与必要的背景性陈述，不放需要本论文附录证明后才能使用的原创/延伸结论。
2. 若某个结论完全来自引用论文：只给出本文需要使用的陈述与明确出处说明，不在本文附录给 proof。
3. 若某个结论需要在本文设定下做延伸或补充才能使用：放到 Method/Theory，并在附录证明。
4. Method/Theory 中引入的新结论必须与本文方法强绑定，用于突出或解释本文的核心优势；不满足该要求的内容不得进入正文结论链路（最多作为背景引用，且不占篇幅）。
5. Preliminaries 不得出现“必须先证明才能使用”的结论型内容：不允许出现 “Proof deferred to Appendix / 见附录证明 / proof in appendix” 等表述；凡需要本文证明链路支撑的内容必须移至 Method/Theory + Appendix。

### 3.2 DPPO 迁移红线

1. 禁止重复：TRM 或本文草稿中已存在同等强度、同等假设、同等结论的结果，不再以新 theorem/proposition/lemma 形式出现。
2. 禁止矛盾：DPPO 与 TRM/本文草稿在同一设定下结论冲突的表述，必须删除或改为明确的条件化说明。
3. 禁止只换符号搬运：若 DPPO 结论与已存在结论完全等价，且不承担结构性作用，则不引入；若其 proof 技巧更优，则用于改写现有 proof，但不新增命题。
4. 贡献归属清晰：迁移部分必须标注 “following / adapted from DPPO” 并引用；本文自己的延伸明确使用 “we further derive”。
5. 学术合规门禁：迁移与改写必须遵循本文件第 3.5 节的学术规范与贡献边界；任何“复制证明/复制表述/符号替换式搬运”一票否决。

### 3.3 写作与排版红线

1. 不在句子中夹杂复杂推导：超过“一行定义/一行等式”的数学内容必须用 display math 分块呈现。
2. 证明必须结构化：使用 Step/Case/Bound 形式组织；推导使用 `align`/`align*` 并逐行给出理由短语。
3. 交叉引用规范：方程统一用 `Eq.~\eqref{...}`，定理/引理/定义/备注/附录统一用 `\Cref{...}`。

### 3.4 文档写作规范

1. 数学符号必须使用数学环境，例如行内写作 $\pi$ 或 $\mathcal{L}$；需要独立成行时使用行间公式块或 `align` 等环境。不得使用 code span 表示数学符号。
2. 命令与路径使用 code span，例如 `PUSH=0 ./compile.sh`、`tex/main/4-method.tex`。
3. 禁止 Unicode 数学符号误入文本，例如直接输入希腊字母 pi 字符；必须在数学环境内使用 `\pi`，例如 $\pi$。

### 3.5 学术规范与贡献边界（强制）

1. 正文新增结论的适配性要求：凡在 Method/Theory 中新增的结论型断言（lemma/theorem/proposition/corollary/remark 等），必须能够从数学理论上突出或解释本文方法的核心优势，或作为本文主定理链路的承重步骤；否则不得进入正文，只能作为背景引用或直接删除。
2. 正文与附录的分工：对本文原创或在本文设定下必要延伸（即非纯引用）的结论，正文只保留清晰陈述、必要解释与证明入口；证明过程与数学细节必须放在附录，并以参考论文（TRM/DPPO 等）的证明写作与排版规范为标杆组织推导。
3. 严禁复制与符号替换式搬运：禁止直接复制参考论文的结论陈述、证明推导、文字表述或段落结构；禁止仅做符号替换、变量重命名、重排等形式化改写来冒充新内容。这类行为属于严重学术不端，必须在执行中作为一票否决项。
4. 引用他人成果的合规方式：若需要使用参考论文的结论，应在 Preliminaries 中仅引用本文需要使用的那部分陈述，并给出明确引用与出处说明；不得在本文（包括附录）复现其证明过程。若必须在本文设定下扩展该结论，则必须明确写出新增假设、扩展点与差异，并给出本文自己的证明。
5. 核心目标约束：参考论文的主要价值在于提供证明方法与技术细节，用来辅助说明本文方法的优势与机制。大量直接搬运参考论文结果会削弱本文核心贡献，应严格控制引用结果的数量与篇幅，并保证论文叙事的主语始终是本文方法与其可证明的优势。

### 3.6 Section 职能分工（强制）

1. Preliminaries：只用于定义、记号、假设与引用外部论文的必要结论片段（statement-only + citation）。Preliminaries 的所有内容必须做到“无需依赖本文附录证明即可直接使用”。
2. Preliminaries 禁止项：不得出现本文原创或本文延伸结论；不得出现需要本文附录证明才能成立的命题；不得出现 proof 指向或延迟证明的叙事。
3. 外部结论的呈现方式：外部论文结论在 Preliminaries 中只允许以 `definition` 或 `remark` 的形式引用（推荐 remark 标题含 “Cited result”），并在文字中明确说明出处。不得在本文（包括附录）复现该外部结论的证明过程。
4. Method/Theory：承载本文原创或在本文设定下必要延伸才能成立的结论，并必须服务于本文方法优势叙事（参见第 3.5 节）。正文只保留清晰陈述、必要解释与证明入口。
5. Appendix：只包含本文原创/延伸结论的完整证明与推导细节；不得出现纯引用结果的 proof 复现。

## 4. 风格与记号规范抽取

### 4.1 参考材料阅读范围

1. TRM 主文本与附录证明：`tex/literature/TRM/main_arxiv.tex`。
2. DPPO 的组织与附录风格示例：`tex/literature/DPPO/example_paper.tex`。
3. DPPO 论文正文与附录入口文件：`tex/literature/DPPO/paper/`。
4. GSPO 的 ratio 与长度归一化推导写法：`tex/literature/GSPO/colm2024_conference.tex`。
5. Theory 三篇（用于理解 off-policy 与校正机制叙事）：`tex/literature/Theory/` 下的三份文档。

### 4.2 定理与证明语言模板

1. 证明开头：一句话说明要证什么，并给出证明路线，例如 “We establish Bounds 1–k and conclude by combining them.”。
2. 证明主体组织：使用 “Step k” 或 “Bound k” 的显式标题，必要时拆成 Case A/B 或 Part (a)/(b)/(c)。
3. 推导书写：集中使用 `align`/`align*`，每行只做一次等式或不等式变形。
4. 行尾理由短语：在每一步末尾用短语标注依据，例如 “(tower property)”, “(Pinsker)”, “(definition of total variation)”, “(log-sum inequality)”, “(triangle inequality)”。
5. 结尾：一句话 “This completes the proof.”。

### 4.3 公式与解释隔离策略

1. 先写桥接句：说明接下来给出的公式是什么、为什么需要、将用于哪里。
2. 再给 display 公式。
3. 再写解释句：解释符号含义、边界条件与直观意义。
4. 禁止在一句话里同时出现多个求和、条件期望、max/min 与多重下标的链式推导。

### 4.4 标题与术语大小写策略

1. 正文中统一使用小写连字符：token-level、sequence-level、prefix-level、off-policy。
2. 标题使用 Title Case，但保留连字符：Token-Level、Sequence-Level、Prefix-Level。
3. 术语名首次出现时给出全称：Trust Region Masking (TRM)。

### 4.5 宏与符号管理策略

1. 以 `tex/math_commands.tex` 作为符号库参考，不直接 `\input`，避免危险重定义（尤其是 `\eqref`）。
2. 论文实际用到的宏收敛到 `tex/math.tex`，任何新增符号先定义宏再使用。
3. 指示函数建议统一宏，例如 `\ind` 或 `\mathbb{I}` 的单一写法，避免混用产生视觉噪声。

## 5. 诊断与缺陷清单模板

### 5.1 诊断范围

1. `tex/main.tex` 及其 `\input` 的所有 `tex/main/*.tex`。
2. 与宏与环境相关的：`tex/env.tex`、`tex/math.tex`。
3. 与方法实现对应的工程说明（若需要映射）：`reinforce_pro_max/`。

### 5.2 缺陷类别与优先级

1. Rigor：定义域/假设缺失、等价性与近似性表述不精确、proof sketch 与 full proof 不一致、因果结构与上下文漂移链条不闭合。
2. Notation：环境类型与 label 前缀不一致、同一对象多种写法、宏未统一。
3. Typesetting：复杂公式与解释混排、长公式 overfull 风险、proof 缺 Step/Case/Bound 结构。

### 5.3 缺陷条目格式

每条缺陷必须按以下模板记录并逐条关闭：

1. 位置：`tex/main/<file>.tex:<line>`。
2. 类型：Rigor / Notation / Typesetting。
3. 问题描述：一句话说明问题。
4. 修复策略：一句话说明将如何改。
5. 影响范围：是否会触及 labels、宏、附录结构或引用。

## 6. 逐文件实施清单

### 6.1 宏与环境层

#### `tex/math.tex`

1. 建立“论文实际使用的最小宏集合”，避免引入未使用宏。
2. 检查 prefix 相关的视觉一致性，必要时统一使用 $\mathrm{pre}$ 的上标风格。
3. 指示函数统一为单一宏或单一符号风格。

#### `tex/env.tex`

1. 确保常用包齐全且不冲突：`amsmath`、`amssymb`、`amsthm`、`mathtools`、`thmtools`、`cleveref`。
2. 环境集合保持标准：theorem/lemma/proposition/corollary/definition/assumption/remark。
3. 证明结构靠内容组织，不新增自造环境名。

### 6.2 正文重写与结构统一

#### `tex/main/1-intro.tex`

1. 用 TRM 风格组织引言：问题设定、挑战、贡献列表。
2. 贡献列表每条一句话，避免口语化与不可验证主张。

#### `tex/main/3-preliminaries.tex`

目标：像 TRM 那样按层级组织为 “设定与记号 → surrogate → error decomposition → divergence building blocks → existing methods”。

1. Existing IS Correction Methods 仅保留定义与统一记号，不做对比评论。
2. 对 $\mathcal{L}_{\pi_{\mathrm{roll}}}$ 与实现 surrogate 的关系，严格区分 “等价改写” 与 “实现近似”。
3. 所有复杂等式使用 `align`，并在每一步末尾标注理由短语。
4. Section 职能门禁：Preliminaries 只允许 definition/notation/assumption/cited statements。若需要引用外部结论，仅给出本文使用的 statement + citation，不写 proof，也不写 “proof in appendix”。凡需要本文证明链路支撑的内容必须移至 Method/Theory，并在 Appendix 给本文自己的 proof（参见第 3.6 节）。

#### `tex/main/4-method.tex`

1. REINFORCE Max：正文只保留最终 closed-form 结果，例如 $(\alpha,\beta)$，推导移至附录。
2. REINFORCE Max：对 “response length” 相关讨论，写成严谨的 remark 结构，并给出可引用的推导入口。
3. REINFORCE Pro：将 “limitations of existing methods” 组织成并列小节，分别解释 token-level 与 sequence-level 的失效机制。
4. 核心定理正文保留 theorem statement 与精炼 proof sketch，完整证明放附录。

#### `tex/main/5-theory.tex`

1. 保持对比结构拆分清晰，例如分别比较 advantage estimation methods 与 IS correction methods。
2. 每个对比段落使用 “Definition → Mechanism → Consequence” 三段式，每段最多一个承重 display 公式。
3. 算法部分采用 TRM 风格解释：line-by-line 或机制说明，避免与公式混排。
4. 新增 theorem family 的承载区：参考 TRM 的 Theoretical Analysis 组织方式（building blocks + bound families），在 Theory 中新增与 Prefix Causal Trust Region 强绑定的理论模块，并确保每个新增结论都有“优势映射”解释段与 Appendix proof 入口。

### 6.3 附录证明重写模板

#### `tex/main/appendix.tex`

目标：附录是“可审稿的证明仓库”，所有证明遵循统一模板。

每个 proof 强制使用以下结构：

1. Goal：一句话说明要证明什么，并引用 `\Cref{...}`。
2. Setup：集中列出将使用的定义、阈值与符号，使用 display math 给出。
3. Derivation：使用 `align`/`align*` 逐行推导，每行标注理由短语。
4. Case/Part：若主定理有 Part (a)/(b)/(c)，proof 必须按 Part 分块，并在块内再用 Step/Bound。
5. Conclusion：一句话结束证明。
6. 新增结论的证明必须是本文原创推导：可以学习 TRM/DPPO 的 proof techniques 与写作结构，但不得复现其证明过程或推导顺序；必须在 proof 开头明确列出本文设定下的符号与假设，并在每一步推导标注理由短语。

## 7. TRM/DPPO 理论借鉴与新增证明（服务 Prefix Causal Trust Region 优势）

本章目标不是“迁移并复现”参考论文结论，而是系统学习 TRM/DPPO 的定理设计与证明组织方式（proof techniques 与 proof aesthetics），并据此在不重复、不矛盾、且严格学术合规的前提下，新增与本文方法强绑定的理论证明，从数学层面更有力地说明 Prefix Causal Trust Region 的核心优势。

### 7.1 扫描范围与 Theorem Mining（强制）

1. TRM：`tex/literature/TRM/main_arxiv.tex`（重点关注 Theoretical Analysis 与 Appendix proofs 的组织方式、Step/Bound 推导结构与承重不等式链）。
2. DPPO：`tex/literature/DPPO/paper/`（重点关注 trust region in LLMs、mismatch、以及 sequence-level divergence 与 surrogate error 的证明套路）。
3. 本文草稿：`tex/main/3-preliminaries.tex`、`tex/main/4-method.tex`、`tex/main/5-theory.tex`、`tex/main/appendix.tex`。

Theorem mining 的产出必须记录为一个“候选结果清单”，每条至少包含：

1. 参考来源与位置：TRM/DPPO 文件 + section/label/行号。
2. 结果类型：Identity / Lemma / Theorem / Proposition / Proof technique。
3. 该结果在原文中的功能：building block / main guarantee / extension / discussion scaffold。
4. 关键证明技巧标签：例如 telescoping identity、tower property factorization、coupling、Pinsker、KL chain rule、data processing、martingale 等。
5. 与本文方法的可挂钩点：该技巧如何用于构造或证明“prefix-causal”相关的新结论。

### 7.2 U/O/C 分类门禁

对 TRM/DPPO 的每个候选结果（lemma/theorem/identity/关键推导）必须归类为：

1. Class U：Unique and Useful，可新增，但必须与本文方法强绑定，并在正文中实际被引用或使用；同时必须能写出明确的优势映射（该结果如何突出或解释本文优势）。
2. Class O：Overlapping，仅用于改写 proof/叙事，不新增 theorem/lemma。
3. Class C：Conflicting or Risky，禁止引入；必要时只能写成条件化 remark 并强调设定差异，且不得形成公式级强断言。

每条候选结果必须记录：

1. 参考原位置：TRM/DPPO 文件与 section 或 label。
2. 类型：Identity / Lemma / Theorem / Proof technique。
3. 归类：U/O/C。
4. 本文落点：Preliminaries / Method / Theory / Appendix。
5. 作用：支撑哪一段论证链，或导出哪条本文自己的延伸推论。
6. 是否需要 prefix/causal extension：是/否。
7. 优势映射：该结果如何从数学理论上突出或解释本文方法的核心优势（一句话写清）。
8. 新增点或差异点：与 TRM/DPPO 对应结果相比，本文新结论在对象、假设或 bound 形式上的关键差异是什么（必须明确写出，避免符号替换式搬运）。

### 7.3 新增理论证明（强制，正文 + 附录）

为提升论文的数学含量与学术程度，必须在不改变本文核心方法逻辑的前提下，新增一组与 Prefix Causal Trust Region 强绑定的理论结论与证明。该新增工作必须满足：

1. 新增结论必须服务于本文方法优势叙事：每个新增 theorem/lemma/proposition 需要在正文给出 “Why this matters for our method” 的解释段，明确它在本文链路中的作用。
2. 参考论文只提供“证明技术与组织风格”：可以学习其 proof techniques 与 proof aesthetics，但不得复制其定理陈述与证明推导结构，更不得复现其证明过程（参见第 3.5 与第 3.6 节）。
3. Section 分工一致：新增结论必须放在 Method/Theory；证明过程与数学细节放在 Appendix（TRM 风格 Step/Bound/Case + `align` 逐行推导）。

新增理论证明的优先模块（建议按优先级从高到低推进）：

1. Prefix-causal 机制优势的理论模块：围绕“因果结构”给出可引用的 lemma/theorem，使其明确区分 token-level/sequence-level gate 的局限，并说明 prefix-causal gate 在结构上为何更匹配长链路的上下文传播。
2. Prefix 统计量与 divergence proxy 的桥接模块：在清晰条件下，将 prefix 累积统计量与可解释的 divergence proxy 联系起来，作为“为何这种 gate 合理”的数学入口，并明确其局限（proxy、条件化、不等价也要说清楚）。
3. 误差界或稳定性界的定理族模块：学习 TRM 的 theorem family 组织方式（building blocks + bound families + caps），在本文设定下形成与 prefix-causal gate 直接相关的误差界或稳定性界，并在附录给完整 proof。

### 7.4 引用型 building blocks：强制优先引入 DPPO 的 1+2（仅陈述 + citation）

#### 7.4.1 DPPO-2：TV 与 ratio 的期望恒等式

目标：迁移并作为 “single-sample proxy 与 divergence 的关系” 的承重背景结论，并避免只换符号复述。

statement 级别（示例写法，最终按本文符号体系落地）：

$$
D_{\mathrm{TV}}\bigl(\pi_{\mathrm{roll}}(\cdot \mid c_t)\,\|\,\pi_\theta(\cdot \mid c_t)\bigr)
\;=\; \tfrac12\,\mathbb{E}_{y_t\sim \pi_{\mathrm{roll}}(\cdot\mid c_t)}\bigl[|\rho_t-1|\bigr].
$$

放置与证明策略：

1. 若该结果作为纯引用背景：在 Preliminaries 以 statement-only 的 `remark` 引用（带 citation），并说明该结论在本文中的用途；不得在本文附录重复其原始 proof。
2. 若本文需要在 prefix/causal 设定下做延伸：延伸版本必须放在 Method/Theory，并在 Appendix 给出本文自己的 proof；Preliminaries 只保留对 DPPO 原结论的引用性 remark（参见第 3.6 节）。

非平凡延伸要求：

1. 至少给出一个桥接 lemma，在可验证条件下将本文使用的 log-ratio 或 prefix 几何均值 proxy 与 $|\rho_t-1|$ 视角建立条件化联系。
2. 必须清楚说明该桥接是 proxy 与不等式关系，而不是“等价于控制 TV”。

#### 7.4.2 DPPO-1：Partition divergence lower bound（coarse-graining 下界）

目标：将 coarse-graining 的 TV/KL 下界作为 “binary/top-k proxy 的 principled 基础”，并避免与 TRM 的结论重复。

放置与证明策略：

1. 若该结论作为纯引用背景：在 Preliminaries 只给 general partition 的 statement（必要时给出 binary/top-k 作为随附陈述），并给 citation；不得在 Appendix 复现 DPPO proof。
2. 若本文需扩展到 prefix 聚合 proxy：扩展版本必须放在 Method/Theory，并在 Appendix 给出本文自己的 proof；Preliminaries 只保留对 DPPO 原结论的引用性 statement + citation（参见第 3.6 节）。

非平凡延伸要求：

1. 给出 sampled-token 的 binary proxy 与 ratio 的等价表达推论，作为解释 “ratio 很大但 divergence 贡献可能很小” 的数学入口。
2. 给出 prefix aggregation 推论：将 per-step lower bound 沿 prefix 聚合，形成可解释的 proxy family，并明确其局限性。

### 7.5 其它可借鉴结果池（TRM/DPPO，优先服务新增 theorem family）

以下结果按默认策略处理，并在迁移前做 U/O/C 去重判定：

1. TRM 的 building blocks 与证明模板：例如 advantage bound、context shift、KL chain rule、exact error identity、tower property factorization、adaptive bound 的 proof 组织方式。原则上不复现其结论，而是学习其证明结构来证明“prefix-causal”设定下的新结论或新定理族。
2. DPPO 的 sequence-level divergence 控制 lemma：若与本文设定可耦合且不与 TRM/本文重复，可作为 cited building block；若需要在 prefix-causal 设定下扩展，则扩展版本必须成为本文新结论并给出本文 proof。
3. mismatch 与稳定性相关的讨论与不等式链：优先作为本文新增结论的 proof 技巧来源，不应仅以引用堆砌的方式进入正文。
4. approximation gap、取等条件与 proxy 紧致性分析：仅当能直接服务于“为何 prefix-causal gate 在实践中更稳/更有效”的解释链路时才引入，否则不引入。

### 7.6 桥接到本文方法叙事（强制）

TRM/DPPO 借鉴与新增理论证明不得成为装饰性背景，必须在 `tex/main/5-theory.tex` 与 `tex/main/4-method.tex` 的相关段落中形成“方法层解释”的桥接段，至少包含：

1. 用 TV identity 解释 ratio clipping 的统计学定位，强调 single-sample proxy 与期望的差异。
2. 用 partition lower bound 解释 binary/top-k divergence proxy 的 principled 性与局限。
3. 若引入 sequence-TV chain bound，则用于把 sequence drift 与 per-position budget 关联，并自然回到 prefix causal gate 的动机。
4. 对每个新增 theorem family，必须写明其服务的核心优势点，并给出读者可追踪的 proof 入口（Appendix label）。

## 8. 验收标准与自动化检索

### 8.1 编译闸门

1. 运行 `PUSH=0 ./compile.sh`。
2. 验收：无 LaTeX 报错，重点关注 undefined references、tabular alignment errors、符号替换导致的解释不一致。

### 8.2 排版与写作规范验收

1. 复杂数学内容使用 display math，不夹在句子里。
2. proof 均具备 Step/Case/Bound 结构，并使用 `align` 展开推导。
3. token-level、sequence-level、prefix-level 的大小写与连字符策略全篇一致。

### 8.3 记号与宏一致性验收

1. 宏集中在 `tex/math.tex`，不直接 `\input` `tex/math_commands.tex`。
2. label 前缀与环境一致，例如 remark 对应 `rem:` 前缀。
3. 禁止 Unicode 数学符号：检索 `rg -n $'\\u03c0' tex`，结果必须为 0。

### 8.4 学术合规验收（人工检查）

1. 对每个 Method/Theory 新增结论：必须能指出其服务的“优势映射”段落位置，并说明它在本文论证链路中承担的作用。
2. 对每个引用自参考论文的结论：只出现陈述与 citation，不出现 proof 复现；附录中不得出现对参考论文证明过程的逐步复刻。
3. 抽查关键段落，确保不存在“符号替换式搬运”：若发现与参考论文在段落结构、推导顺序与表述上高度一致，则必须重写或移除。
4. 检查引用占比与叙事主语：引用性结果不得遮蔽本文贡献；正文应以本文方法的优势与可证明机制为主线。

### 8.5 Section 职能分工验收（自动化检索）

以下检索作为结构门禁，确保 Preliminaries 不承载需要本文证明的结论：

1. Preliminaries 不包含 theorem 类环境：运行 `rg -n "\\\\begin\\{(theorem|lemma|proposition|corollary)\\}" tex/main/3-preliminaries.tex`，结果必须为 0。
2. Preliminaries 不包含 proof 指向或延迟证明叙事：运行 `rg -n "Proof|proof|Appendix|附录|defer|deferred|见附录" tex/main/3-preliminaries.tex`，结果必须为 0。

### 8.6 新增理论证明验收（人工 + 结构化检查）

1. 每个新增 theorem/lemma/proposition 必须满足第 7.2 节记录要求：具有明确的优势映射与新增点或差异点；正文中必须出现对应的解释段，并指向 Appendix proof 入口。
2. 每个新增 theorem family 必须在 `tex/main/appendix.tex` 中提供完整 proof（TRM 风格 Step/Bound/Case + `align` 逐行推导），并在每一步标注推导依据；不得用 “proof sketch” 代替必要细节。
3. 抽查证明过程的原创性：不得出现与 TRM/DPPO proof 在推导顺序、关键中间式与段落结构上高度一致的复刻；若存在相似，必须重写推导路径或将其降级为纯引用 statement（且不写 proof）。
4. 结果密度与贡献聚焦：新增理论证明应当直接服务 Prefix Causal Trust Region 的核心优势叙事，不得为“增加定理数量”而引入与本文方法弱相关的结论。

## 9. 提交与迭代策略

目标：可审阅、可回滚、每步都有编译通过证据。

建议切分为两条主线提交序列：

1. Style and structure rewrite：按文件或章节拆分提交。
2. TRM/DPPO theory borrowing：先做 theorem mining 与候选清单，再做与 Prefix Causal Trust Region 强绑定的新增 theorem family，最后再补充引用型 building blocks 与必要的扩展推论；每个阶段单独提交。

每个 commit 的门禁：

1. 改动后必须通过 `PUSH=0 ./compile.sh`。
2. 提交信息必须明确范围，例如 `tex: ...`、`docs: ...`。

## 10. 维护规则

1. 本文件 `tex/plan.md` 是唯一权威计划文档。
2. 后续新增计划只允许在本文件对应章节追加或修订，不新增平行的 `plan/*.md`。
3. 对计划的修改需要同时更新硬约束章节中的红线口径、验收标准章节中的可执行检查项，以及提交策略章节中的切分建议（若受影响）。
