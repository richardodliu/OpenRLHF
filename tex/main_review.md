# `main.tex` 审稿意见

## 总评
主控结构（章节组织、引用接入、附录衔接）已完整。当前主要风险来自摘要中若干“理论连接/误差分解控制”式主张对 `3-preliminaries`、`4-method`、`5-theory` 的关键证明链依赖。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P1] 类型: 证明
  位置: `tex/main.tex:16`
  问题: 摘要中除 “tighter masking” 外，还包含 “prefix cumulative IS 与 Adaptive bound 的连接/近似” 以及 “context distribution shift is controlled via sample-level filtering” 等更强的理论含义陈述；这些表述当前仍依赖上游未闭环点（主要是 `4-method` 中 sample-level 过滤与 population-level KL/TV 的严格关系、以及 `5-theory` 中 `\\pi_{old}\\to\\pi_\\theta` 的形式化桥接）。
  影响: 摘要可能被解读为已给出可直接用于误差界的严格控制结论，主张强度超过当前可严格证明范围。
  建议: 将摘要相关句子收敛到“structural correspondence / expectation identity + heuristic proxy”的强度，或在 `4/5` 章补齐可检验界后再恢复更强结论；具体缺陷见对应章节 review。

## 逻辑问题
- 当前未发现未修复问题。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P1: 让摘要中关于 “Adaptive bound connection / context shift control” 的结论强度与 `4/5` 章当前可证范围严格一致。
