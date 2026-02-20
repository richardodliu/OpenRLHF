# `4-method.tex` 审稿意见

## 总评
本章较前版本已完成多项关键修复（例如 `eq:threshold-kl`/Corollary 的 mask 一致性、非对称阈值上界写法等）。当前剩余问题集中在四处：`thm:prefix-tighter` 的严格适用域、Part (b)/(c) 的陈述-证明一致性、以及 `\pi_{old}` 到 `\pi_\theta` 的理论桥接。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P0] 类型: 证明
  位置: `tex/main/4-method.tex:177`
  问题: Part (a) 的证明将“越界”写成 `|L_t/P_t|` 相对 `[-\log\Lambda,\log\Lambda]` 的条件，并推导 `P_t\log\Lambda` 界；这默认了对称阈值，而定理阈值是一般的非对称区间 `[\log\lambda,\log\Lambda]`。
  影响: 当 `|\log\lambda| \neq \log\Lambda` 时，Part (a) 结论的可证范围与陈述不一致。
  建议: 按上下侧分别给出条件，或统一改为 `\max(|\log\lambda|,\log\Lambda)` / 分段不等式推导。

- [P0] 类型: 证明
  位置: `tex/main/4-method.tex:168`
  问题: Part (c) 在严格越界假设下先给出“`|\{t:M_t^{prefix}=0\}| \le |\{t:M_t^{seq}=0\}|`（seq rejection 时 all masked）”，但证明又写“if sequence is accepted, prefix masks more tokens”（`tex/main/4-method.tex:181`）。后者与该假设下 seq 必拒绝的逻辑冲突。
  影响: 定理陈述与证明文本存在内部不一致，削弱结论可审计性。
  建议: 将“seq accepted”分支移到“更一般同号但不逐点越界”的子情形，或在严格越界情形中删除该分支并给出一致结论。

- [P1] 类型: 证明
  位置: `tex/main/4-method.tex:157`
  问题: Part (a) 文字称“mask 一个 contiguous block，且长度随 `|\ell_1|` 增大、随 `\epsilon` 减小”，但当前证明仅给出一个足够条件下的初始区间不等式，未给出对“块长度单调关系”的完整证明。
  影响: 读者易将经验性趋势解读为严格定理结论。
  建议: 将该句降级为“under stated sufficient condition”并给出精确可证定义（如最大满足不等式的 `t`）。

- [P1] 类型: 证明
  位置: `tex/main/4-method.tex:179`
  问题: Part (b) 证明在早期 token 的通过条件写为 `M_t^{\mathrm{prefix}} = 1` when `\epsilon < \log\Lambda`，未与定理条件 `0 \le \epsilon \le \min(\log\Lambda, |\log\lambda|)` 完整同构；下侧阈值 `|\log\lambda|` 的对应论证缺失。
  影响: 证明文本条件域与定理陈述条件域不完全一致，降低可审计性。
  建议: 将 Part (b) 的通过条件改为双边阈值版本（或分上下侧分别证明），确保与定理假设完全对齐。

- [P1] 类型: 证明
  位置: `tex/main/4-method.tex:196`
  问题: `thm:prefix-adaptive` 讨论目标误差项 `\|d_t^{\pi_\theta}-d_t^{\pi_{roll}}\|_{TV}`，但 prefix IS 直接约束的是 `\pi_{old}/\pi_{roll}`。当前仅给启发式文字（triangle inequality + clipping narrative），缺少从 PPO clip 条件到状态分布 TV 距离的可检验不等式链。
  影响: theorem 对“控制目标误差项”的严格性不足。
  建议: 增加显式桥接引理（含步长/ratio/覆盖条件），或将本节结论明确限制为 `\pi_{old}` 层面的控制。

## 逻辑问题
- 当前未发现未修复问题。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P0: 修复 Part (a) 对非对称阈值 `[\lambda,\Lambda]` 的严格推导（避免默认对称阈值）。
2. P0: 消除 Part (c) 陈述与证明的分情况冲突，统一 strict-over-threshold 情形下的 seq/prefix 关系。
3. P1: 对齐 Part (b) 证明条件与定理条件（补全 `|\log\lambda|` 侧约束）。
4. P1: 补齐 `\pi_{old}\rightarrow\pi_\theta` 的形式化桥接，避免仅靠叙述性解释。
