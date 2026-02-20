# `4-method.tex` 审稿意见

## 总评
本章近期已修复多处关键问题（包括 Part (a) 非对称阈值推导、Part (b) 条件双边化、Part (c) strict-violation 情形下与 sequence-level 的计数冲突、以及 `eq:threshold-kl`/Corollary 的 mask 一致性）。当前剩余风险主要集中在 “prefix IS 与 Adaptive bound 的连接” 这一段仍存在从 sample-level 过滤到 population-level KL/TV 约束的严格性缺口，容易被读者误解为已给出可直接用于误差界的定理化结论。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P1] 类型: 证明
  位置: `tex/main/4-method.tex:208`
  问题: `thm:prefix-adaptive` 将 “prefix IS 的 sample-level 阈值过滤” 与 “population-level 的 KL/TV 量（KL chain rule + Pinsker）” 放在同一证明链中叙述，且出现了 “On the accepted subset ... applying Pinsker to the population-level KL yields ...” 这类表述。严格来说，逐轨迹的约束 $|L_t|/P_t\\in[\\log\\lambda,\\log\\Lambda]$ 并不推出 $\\DKL(d_{t+1}^{\\piroll}\\|d_{t+1}^{\\piold})$ 或 $\\|d_{t+1}^{\\piold}-d_{t+1}^{\\piroll}\\|_{\\mathrm{TV}}$ 在全分布意义下变小；而对 “过滤后的有效训练分布” 也未形式化定义其对应的 KL/TV 量与可检验上界。
  影响: “与 Adaptive bound 的连接” 容易被读者误读为已给出严格的误差界控制结论，进而抬高摘要/引言中关于 proxy 的主张强度。
  建议: 将 Part (c) 收敛为严格可证的两点：1) 期望恒等式（KL chain rule）；2) sample-level 过滤等价于对 $L_t/P_t$ 的区间约束。关于“过滤后有效分布的 KL/TV 被控制”的部分，建议改写为讨论性 remark，或显式引入并证明关于条件分布/截断分布的界（需额外假设，如接受概率下界或对数似然比上界）。

## 逻辑问题
- 当前未发现未修复问题。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P1: 重新收敛 `thm:prefix-adaptive` 的可证内容，避免 sample-level 过滤与 population-level KL/TV 量混用造成的“定理强度过度解读”。
