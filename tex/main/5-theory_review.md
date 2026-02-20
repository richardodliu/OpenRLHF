# `5-theory.tex` 审稿意见

## 总评
本章叙述结构清晰，但仍有一处理论强度与数学可证性不完全一致的陈述。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- 当前未发现未修复问题。

## 逻辑问题
- [P1] 类型: 逻辑
  位置: `tex/main/5-theory.tex:11`
  问题: 文中将“`\Var[\hat{A}]=1`”表述为“bounding the effective advantage magnitude and preventing gradient explosion from outlier advantages”。方差归一化只能约束二阶矩，不直接给出幅值上界，也不能单独保证“prevent explosion”。
  影响: 该句容易被解读为严格稳定性保证，而当前理论只支持“降低尺度/改善数值稳定性倾向”。
  建议: 改为条件化表述（如“stabilizes scale in expectation / mitigates outlier impact”），避免写成确定性防爆结论。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P1: 将 `Var=1` 的效果从“确定性防止梯度爆炸”降级为“统计尺度稳定化”表述。
