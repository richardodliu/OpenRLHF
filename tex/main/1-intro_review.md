# `1-intro.tex` 审稿意见

## 总评
引言整体已与正文主线对齐，术语与作用域限定较完整。当前剩余风险主要来自“与 Adaptive bound 的连接”在引言贡献中的表述强度仍可能超过方法章目前严格可证的内容。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P1] 类型: 证明
  位置: `tex/main/1-intro.tex:23`
  问题: 贡献 4 将 “prefix IS 阈值过滤” 表述为对 trust region Adaptive bound 的“practical proxy / via KL chain rule”的连接。当前方法章 `thm:prefix-adaptive` 更接近结构性对应与期望恒等式（KL chain rule），但从 sample-level 过滤直接推出 population-level KL/TV 控制仍未严格闭环。
  影响: 贡献表述可能被读者理解为已给出可直接代入误差界的定理化结论，主张强度偏高。
  建议: 将引言措辞收敛到“structural correspondence / expectation identity + heuristic proxy”，或在方法章补全对“过滤后有效分布”的可检验界后再恢复更强表述。

## 逻辑问题
- 当前未发现未修复问题。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P1: 让贡献 4 关于 “proxy / Adaptive bound connection” 的表述强度与 `4-method` 当前严格可证内容严格一致。
