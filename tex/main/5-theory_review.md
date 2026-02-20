# `5-theory.tex` 审稿意见

## 总评
本章叙述结构清晰，但仍有两处理论强度与数学可证性不完全一致的陈述：一处是 `\pi_{old}` 与 `\pi_\theta` 的桥接，另一处是方差归一化的效果表述过强。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P1] 类型: 证明
  位置: `tex/main/5-theory.tex:13`
  问题: 本章将 context shift 控制叙述为对 `\|d_t^{\pi_\theta}-d_t^{\pi_{roll}}\|_{TV}` 的改进，但上游 prefix IS 直接约束的是 `\pi_{old}/\pi_{roll}`。当前缺少 `\pi_{old}\to\pi_\theta` 的传递假设与界。
  影响: “统一框架同时控制误差分解两因子”的证明链仍有关键中间环节缺失。
  建议: 增加桥接假设（如 PPO 小步更新）并给出可检验不等式，或将本章结论显式限定到 `\pi_{old}` 作用域。

## 逻辑问题
- [P1] 类型: 逻辑
  位置: `tex/main/5-theory.tex:11`
  问题: 文中将“`\Var[\hat{A}]=1`”表述为“bounding the effective advantage magnitude and preventing gradient explosion from outlier advantages”。方差归一化只能约束二阶矩，不直接给出幅值上界，也不能单独保证“prevent explosion”。
  影响: 该句容易被解读为严格稳定性保证，而当前理论只支持“降低尺度/改善数值稳定性倾向”。
  建议: 改为条件化表述（如“stabilizes scale in expectation / mitigates outlier impact”），避免写成确定性防爆结论。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P1: 补齐 `\pi_{old}` 与 `\pi_\theta` 的理论桥接，避免跨策略变量直接替代。
2. P1: 将 `Var=1` 的效果从“确定性防止梯度爆炸”降级为“统计尺度稳定化”表述。
