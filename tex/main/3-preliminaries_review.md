# `3-preliminaries.tex` 审稿意见

## 总评
本章相较旧版已有实质改进（已补充 surrogate 替换说明与 coupling 引理条件），但仍存在两个核心理论风险：surrogate 化简与性能差分 surrogate 的等价关系未严格闭环、以及 `Context Shift Propagation` 的证明链仍有关键论证缺口。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P0] 类型: 证明
  位置: `tex/main/3-preliminaries.tex:39`
  问题: 文中已补充“trajectory-level advantage substitution”说明，但仍将 `A_t^{\piroll}` 的 per-step surrogate 与 REINFORCE 无偏梯度估计放在同一段直接衔接；“对策略梯度无偏”并不自动推出“与该 surrogate 目标等价”。
  影响: “理论目标（PDI surrogate）-实现损失（REINFORCE 估计）”之间仍存在未形式化的桥接缺口。
  建议: 显式拆分两层结论：1) surrogate 近似/替代关系；2) gradient estimator 无偏性；并给出对应条件与推导。

- [P0] 类型: 证明
  位置: `tex/main/3-preliminaries.tex:110`
  问题: `Context Shift Propagation` 的归纳步构造了“一条后续每步两策略取同概率 token 的路径”，并以此让乘积比值化简为 base case 的常数（`tex/main/3-preliminaries.tex:110`-`114`）。但“support overlap / 同词表”并不推出“存在 token 使两分布在该 token 上概率相等”，两分布可以在所有 token 上概率均不同，从而该路径未必存在；因此当前归纳论证不成立。
  影响: 该引理被 `4-method` 用于论证 token-level IS 忽略因果结构时的“后续 context shift 必然存在”，证明链会被上游断点传导影响。
  建议: 改为不依赖“逐步概率相等 token”的构造：例如从已知的某个前缀事件在 $t=s+1$ 时概率不同出发，利用“对该前缀的所有长度-$t$ 扩展概率和等于该前缀概率”的守恒关系（两策略在后续同支持下扩展集合一致），推出必存在某个扩展前缀在 $t$ 时仍概率不同，从而 $\|d_t^\pi-d_t^{\pi'}\|_{\mathrm{TV}}>0$。

## 逻辑问题
- 当前未发现未修复问题。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P0: 为 surrogate 化简补全严格条件与推导，或改回标准逐时刻形式。
2. P0: 修复 `Context Shift Propagation` 归纳步的构造漏洞，给出不依赖“逐步概率相等 token”的严格证明。
