# `main.tex` 审稿意见

## 总评
主控结构（章节组织、引用接入、附录衔接）已完整。当前主要风险仍来自方法章 `thm:prefix-tighter` 的严格证明缺口对摘要结论的反向依赖。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P1] 类型: 证明
  位置: `tex/main.tex:66`
  问题: 摘要中“在 analyzed deviation patterns 下可提供更紧 masking”依赖 `thm:prefix-tighter` 的完整成立；该定理当前仍有关键证明缺口（Part (c) 单调性断言、Part (a)/(b) 条件不完备、以及与 `\pi_\theta` 目标项的桥接不足）。
  影响: 摘要结论的可证范围受上游定理缺口限制。
  建议: 在 `4-method` 完整补证前，进一步限定为“conditional on currently proved theorem cases”。

## 逻辑问题
- 当前未发现未修复问题。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P1: 让摘要中 “tighter masking” 的结论强度与 `4-method` 实际可证范围严格一致。
