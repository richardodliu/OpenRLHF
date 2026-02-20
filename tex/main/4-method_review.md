# `4-method.tex` 审稿意见

## 总评
本章近期已修复多处关键问题（包括 Part (a) 非对称阈值推导、Part (c) strict-violation 情形下与 sequence-level 的计数冲突、以及 `eq:threshold-kl`/Corollary 的 mask 一致性）。当前剩余问题主要有两处：Part (b) 证明条件与定理条件仍未完全同构，以及 `\pi_{old}` 与 `\pi_\theta` 的桥接仍停留在“引用后续章节说明”层面。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P1] 类型: 证明
  位置: `tex/main/4-method.tex:179`
  问题: Part (b) 证明在早期 token 的通过条件写为 `M_t^{\mathrm{prefix}} = 1` when `\epsilon < \log\Lambda`，未与定理假设 `0 \le \epsilon \le \min(\log\Lambda, |\log\lambda|)` 完整同构；缺少 `|\log\lambda|` 一侧的对应论证。
  影响: 证明文本的适用域小于定理陈述域，严格性不足。
  建议: 将 Part (b) 的通过条件改为双边阈值版本（或分上下侧分别证明），确保与定理假设一一对应。

- [P1] 类型: 证明
  位置: `tex/main/4-method.tex:196`
  问题: `thm:prefix-adaptive` 已将 context shift 写为 `\|d_t^{\piold} - d_t^{\piroll}\|_{TV}`，并在括号中引用 `\Cref{sec:unified}` 说明 `\piold \to \pitheta` 的 gap；但该处仍缺本章内可检验的传递界或明确“仅控制 `\piold/\piroll`”的作用域定理化表述。
  影响: 本章 theorem 与全局误差分解目标项（涉及 `\pitheta`）之间仍存在形式化衔接缺口。
  建议: 在本章补充显式 bridge 引理（含 PPO clip/step-size 条件），或把 theorem 结论明确限定为 `\piold` 层面的控制结论。

## 逻辑问题
- 当前未发现未修复问题。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P1: 对齐 Part (b) 证明条件与定理条件（补全 `|\log\lambda|` 侧约束）。
2. P1: 补齐 `\pi_{old}\rightarrow\pi_\theta` 的形式化桥接，或收紧 theorem 作用域表述。
