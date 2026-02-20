# `1-intro.tex` 审稿意见

## 总评
引言整体已与正文主线对齐，术语与作用域限定较完整。剩余风险主要来自贡献 3 对 `thm:prefix-tighter` 完整正确性的依赖。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P1] 类型: 证明
  位置: `tex/main/1-intro.tex:21`
  问题: 贡献 3 虽已改为 “can provide tighter masking in analyzed patterns”，但仍直接依赖 `thm:prefix-tighter`。该定理当前仍有关键证明缺口（Part (c) 单调性断言、Part (a)/(b) 条件不完备、以及与 `\pi_\theta` 误差项的桥接不足）。
  影响: 引言贡献陈述与当前可严格证明范围仍存在耦合风险。
  建议: 增加更明确限定语（如 “under currently proved sufficient conditions in Theorem 4.x”）或在方法章补齐证明后再维持当前表述。

## 逻辑问题
- 当前未发现未修复问题。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P1: 让贡献 3 的结论范围与 `4-method` 当前可证条件严格一致。
