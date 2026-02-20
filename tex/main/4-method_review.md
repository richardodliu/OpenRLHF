# `4-method.tex` 审稿意见

## 总评
本章已修复多处历史问题（包括 seq-mask 计数分情况、loss 方向、显式补充部分阈值条件）。当前剩余问题集中在三类：`thm:prefix-tighter` 的陈述-证明一致性、`thm:prefix-adaptive` 中 sample-level 公式与阈值界限处理不一致、以及 `\pi_{old}` 到 `\pi_\theta` 的理论桥接缺失。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P0] 类型: 证明
  位置: `tex/main/4-method.tex:165`
  问题: Theorem (c) 仍写“`|L_t/P_t|` non-decreasing”，该结论不成立。即便所有 `\ell_s` 同号且均超过阈值，运行平均值也可能下降（只能保证在额外充分条件下保持阈值同侧）。
  影响: 定理陈述包含可被反例否定的数学断言。
  建议: 将 “non-decreasing” 改为“stays on the same threshold side under stated sufficient conditions”，或给出真正可证的单调性条件。

- [P0] 类型: 证明
  位置: `tex/main/4-method.tex:205`
  问题: Eq.~\eqref{eq:threshold-kl} 写成 `\frac{1}{P_t}\sum_{s=1}^t \ell_s`，与定义 `L_t=\sum_{s=1}^t \ell_s m_s` 不一致；Corollary 同样遗漏 `m_s`（`tex/main/4-method.tex:237`）。
  影响: prefix 约束的核心等价式与定义不一致，在存在非 action token/padding 时定理不成立。
  建议: 统一改为带 mask 的形式：`\frac{1}{P_t}\sum_{s=1}^t \ell_s m_s`。

- [P1] 类型: 证明
  位置: `tex/main/4-method.tex:153`
  问题: Part (a) 的掩码长度结论 `\lfloor(|\ell_1|-\epsilon)/\max(\cdot)\rfloor` 与证明给出的条件 `|\ell_1| > P_t(\log\Lambda+\epsilon)-\epsilon` 不一致，缺严格推导对应关系。
  影响: theorem statement 与 proof 存在公式级不匹配。
  建议: 统一 statement/proof 使用同一可证上界，或将长度结论改为条件不等式而非闭式计数。

- [P1] 类型: 证明
  位置: `tex/main/4-method.tex:159`
  问题: Part (b) 未声明 `\epsilon` 与阈值区间关系；若 `\epsilon` 超出阈值边界，`t<t^*` 的早期 token 也可能被 mask，与“最多从 `t^*` 开始受影响”的叙述不一致。
  影响: Part (b) 结论适用域不清晰，可能被误用于不满足条件的场景。
  建议: 增加显式条件（如 `\epsilon \le \min(\log\Lambda, |\log\lambda|)`）或改写为条件化结论。

- [P1] 类型: 证明
  位置: `tex/main/4-method.tex:196`
  问题: `thm:prefix-adaptive` 讨论的 context shift 项是 `\|d_t^{\pi_\theta}-d_t^{\pi_{roll}}\|_{TV}`，但 prefix IS 约束直接作用于 `\log(\pi_{old}/\pi_{roll})`。当前未给出从 `\pi_{old}` 到 `\pi_\theta` 的桥接假设与不等式链。
  影响: theorem 对目标误差分解项“可控性”的论证缺关键中间步骤。
  建议: 增加 `\pi_\theta` 与 `\pi_{old}` 的小步约束（如 trust-region / PPO clip 条件）并给出传递界，或将结论限定为对 `\pi_{old}` 的控制。

## 逻辑问题
- [P1] 类型: 逻辑
  位置: `tex/main/4-method.tex:230`
  问题: 文中将 accepted sample 约束写为 `|L_t| \le t\cdot|\log\Lambda|`，该绝对值界在非对称阈值下不成立（例如 `|\log\lambda|>\log\Lambda`）。
  影响: “sample-level 过滤 -> 有效 KL 受控”的解释链在非对称阈值设置下失真。
  建议: 改为 `|L_t| \le t\cdot\max(|\log\lambda|,\log\Lambda)`，或按上下侧分别给界。

- [P1] 类型: 逻辑
  位置: `tex/main/4-method.tex:165`
  问题: Part (c) 文字写“each individually exceeds threshold”，但公式使用非严格不等号（`\ell_s \ge \log\Lambda` / `\ell_s \le \log\lambda`）且阈值区间是闭区间，边界点会被接受而非 mask。
  影响: 语义（exceeds）与公式（含等号）不一致，导致读者误解定理适用域。
  建议: 统一为严格越界符号（`>`/`<`）或把文字改成“reaches or exceeds boundary”并重写对应结论。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P0: 修复 `thm:prefix-tighter` Part (c) 中错误的“non-decreasing”断言。
2. P0: 修复 Eq.~\eqref{eq:threshold-kl} / Corollary 与 `L_t` 定义不一致（补上 `m_s`）。
3. P1: 补齐 `\pi_{old}` 到 `\pi_\theta` 的理论桥接，并对齐 Part (a)/(b) 的适用条件。
