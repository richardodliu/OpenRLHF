# 与 DPPO / TRM 的理论技术深度比较

## 结论与验收口径

按用户最新要求，本次不评价实验、预期录用概率或研究影响力，只判断当前理论部分的研究程度与所提供 DPPO/TRM 是否相当。

结论：**总体处于相当的理论技术层次，可以按这一口径完成目标。** 依据是问题建模、误差分解、条件分布控制、算法相关分析和目标连接的组合，而不是页数、数学环境数量、Lean 定理数量或复现检查通过。这个判断不表示本文原创性、通用性或影响力等同于两篇参考论文，也不表示每个 bound 都更强。

参考范围以本地原文为准：DPPO 的 `paper/llm_bound.tex`、`paper/app.tex`、`paper/method.tex`；TRM 的 `main_arxiv.tex` 中首个完整 document。该 TRM 文件包含拼接版本，比较不将两个版本的结果重复计算。本文为当前 `tex/main/paper-body.tex` 和 `core-appendix.tex`。

## 逐项比较

| 技术维度 | DPPO / TRM 的处理 | 本文的处理 | 判断 |
| --- | --- | --- | --- |
| 有限时域、终端奖励建模 | DPPO 为 LLM 稀疏奖励重写 performance difference；TRM 使用有限时域自回归模型 | 同类模型，同时写明 EOS padding、互相支持、RLOO 与 active-token reduction | 相当；本文的采样/归约对应更细，不据此声称新理论框架 |
| surrogate—真实奖励分解 | DPPO 乘积展开；TRM 同时使用 performance difference 和逐位置比率展开 | 正文第 5 节和附录 A 完整推导，并明确来源 | 核心技术相当；这部分为借鉴，不是本文原创增量 |
| 条件序列 TV | 将 continuation likelihood ratio 与条件 TV 联系，分解为后续局部偏移 | 在同一完整 prefix（包括当前 action）下换测度，保留条件依赖，再作 TV 与 KL 两条控制 | 相当；未用独立性替代条件期望 |
| 误差界层次 | DPPO 的二次 TV 和线性期望 TV；TRM 的 Coupling、Pinsker–Marginal、Mixed、Adaptive | 保留 Adaptive/Mixed 主式，附录推导 DPPO 两式及 Pinsker–Marginal 量级，说明何时各路线有用 | 主线相当；不逐项重复列出参考论文的全部变体 |
| 方法专属数学对象 | DPPO 分析粗粒度散度近似及其 gap；TRM 分析整序列散度与位置依赖的误差 | Max 分析两系数、长度相关性与条件方向；Pro 分析接受质量、下侧重入矩项及归一化策略反例 | 对象不同，证明工具和条件分析层次相当 |
| 实际变换后的目标连接 | 两文以 raw surrogate 的奖励界解释 masked objective；TRM 明确指出二者有差别 | 精确 U−G+N 分解；候选策略变化引起的 mask 变化进入残差；恢复随机分母后消去 sibling baseline | 本文具有实质的算法适配内容，不只是重述参考 bound |
| 正面条件与失败边界 | 两文在散度条件下给奖励保证，讨论界的紧松与近似条件 | 给正奖励 margin / 有界保留矩并存的族，以及目标上升而奖励下降的反例 | 研究层次相当；当前正例不作为非对称 Max 独有优势的证据 |
| 实现归约 | 对 surrogate / gradient 和 mask 作定义 | 另保留微批次 response-count 与 token-count 差值，说明如何代入奖励界 | 对实际目标的说明充分；没有扩张为 Adam 一步保证 |

## 为什么不是只靠复用已有 bound 得出结论

DPPO 的关键证明是序列乘积展开、TV 的三角不等式及粗粒度 KL 的 log-sum inequality；TRM 的关键推进是对误差因子分别使用条件 TV、KL chain、Pinsker、边缘化与不同尺度的组合。其技术深度主要来自问题对应和界的组织，不以使用更高级数学工具为标准。

本文复用 reward remainder 的部分必须归属于参考工作。除此之外，实际信号 H 依赖整个样本组及长度，mask 又依赖候选 q，token 分母随机，不能直接将参考论文中的原始 surrogate 当作训练目标。本文通过点态残差、条件 sibling 积分和归约恢复处理这些依赖；Max 的协方差分解处理正缩放并不必然保留总体方向的问题；Pro 的重入分解和归一化策略构造区分前缀平均接受与单 token 系数矩。它们共同构成与参考文献同层次的方法分析，单看任一代数步骤都不足以代表整篇贡献。

## 对条件保证作同等比较

- TRM 的 Theoretical Guarantee 第 3 项明确额外要求所有可达 context 的全局 KL 上界；其 Role of masking 段落区分 full 与 masked surrogate。不能把参考结果解读为“样本被接受就自动满足全局奖励保证”。
- DPPO 附录 C 的 Binary/Top-K TV、KL 是全词表散度的下界；对近似量设置阈值不能单独推出真散度的同阈值上界。本文不采用这两类近似，故没有必要增加其专属 log-sum 证明来凑齐技术清单。
- 本文的 prefix geometric mean 同样不是全局散度控制，宽阈值的通用矩包络可能松。它对此给出显式重入项和反例。这是不同机制需要分析的边界，不单独构成“技术深度低于两篇参考论文”的理由。

## 保留的差异与停止扩展的依据

TRM 的误差界改进组织更集中，DPPO 的分布散度 mask 与其 bound 之间动机更直接。本文的重点分散在 Max 的样本缩放与 Pro 的前缀选择两部分，其统一 remainder 是继承的；这些差异影响原创贡献评价，不能通过多列参考公式消除。

本文无需为本次目标额外证明：普遍优于 TRM/DPPO、随机 Adam 单步单调、整个神经网络训练收敛、所有阈值均有紧界、或所有参考文献技术都在本文出现。它们都不是两篇参考理论部分共同达到的验收前提。当前仍有可研究的改进空间，但不足以据此不断扩展本次目标。

## 当前稿件证据

- 数学主线：正文第 3–5 节，附录 A–D；来源与界的比较集中于附录 A 的 Relation to DPPO and TRM bounds。
- 方法：Max 正负两系数；Pro current/rollout 前缀 mask × 一次 token ratio；无独立第三策略。论文不用 gate/filter 描述机制。
- 内容规模：本轮比较没有新增定理、改算法或扩大论文。此前补充以合并重复论证容纳。
- 最新 PDF：23 页，正文第 9 页末行底部 730.056 pt。补充包 `/tmp/promax_mixed_kl_supplement.zip`；当前指纹见 `current_theory_validation.json`。
- 形式化证据保持原范围：完整总体期望推导在论文内，Lean 组件不被表述为整篇自然语言论文的自动验证。这些验证只支持可检查性，不决定上面的技术深度判断。

本次结论取代此前以“ICLR 竞争力是否充分”为标准的未完成判断，但不撤销其中关于原创增量和适用范围的事实。验收标准改变后，采用同等条件比较理论研究程度，结论为相当。

补充：当前附录还显式给出完整序列 Mixed-KL、prompt 平均/Jensen 表达及 TV/KL 交叉条件，作为已有误差界的解析补充；不新增独立定理。


最新补充：附录已加入固定深度前缀 KL 期望恒等式，以及长度条件保留量和 Max 符号质量的连接，含解析推导与 TRM 对应引用。当前全文 24 页、正文 9 页；当前补充包为 `/tmp/promax_prefix_analysis_supplement.zip`。这些内容解释现有方法，不增加编号定理、算法组件或新的形式化覆盖声明。
