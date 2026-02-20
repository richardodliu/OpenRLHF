# `3-preliminaries.tex` 审稿意见

## 总评
本章相较旧版已有实质改进（已补充 surrogate 替换说明与 coupling 引理条件），但仍存在两个核心理论风险：surrogate 化简与性能差分 surrogate 的等价关系未严格闭环、以及 `Context Shift Propagation` 证明链仍有关键前提缺口。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P0] 类型: 证明
  位置: `tex/main/3-preliminaries.tex:39`
  问题: 文中已补充“trajectory-level advantage substitution”说明，但仍将 `A_t^{\piroll}` 的 per-step surrogate 与 REINFORCE 无偏梯度估计放在同一段直接衔接；“对策略梯度无偏”并不自动推出“与该 surrogate 目标等价”。
  影响: “理论目标（PDI surrogate）-实现损失（REINFORCE 估计）”之间仍存在未形式化的桥接缺口。
  建议: 显式拆分两层结论：1) surrogate 近似/替代关系；2) gradient estimator 无偏性；并给出对应条件与推导。

- [P0] 类型: 证明
  位置: `tex/main/3-preliminaries.tex:94`
  问题: `Context Shift Propagation` 的证明中，base case 使用了 `d_s^{\pi}(c_s)=d_s^{\pi'}(c_s)`（`tex/main/3-preliminaries.tex:108`），但当前假设只要求“在位置 `s` 的某个可达 context 处 token 分布不同”，并未保证两策略在 `s` 前完全一致。
  影响: `4-method` 中关于 token-level IS 局限性的理论依托不够稳固。
  建议: 明确补充“`s` 为首次分歧位置”这类先验条件，或重写 base case（避免依赖该等式）。

- [P1] 类型: 证明
  位置: `tex/main/3-preliminaries.tex:114`
  问题: 归纳步由 `d_{s+1}^{\pi}(c_s,v^*) \neq d_{s+1}^{\pi'}(c_s,v^*)` 与“后续乘子均为正”直接推出 `d_t^{\pi}(c_t^*) \neq d_t^{\pi'}(c_t^*)`，但两侧后续乘子本身可以不同，当前论证未排除乘积偶然相等的情形。
  影响: 归纳链条不完整，导致结论严格性不足。
  建议: 增加同一路径下的可控比值界/下界构造，或改为基于 Radon-Nikodym 比值递推的证明方式。

## 逻辑问题
- 当前未发现未修复问题。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P0: 为 surrogate 化简补全严格条件与推导，或改回标准逐时刻形式。
2. P0: 修复 `Context Shift Propagation` base case 的前提缺口（或补充首次分歧假设）。
3. P1: 补齐归纳步中的不等式链，避免“正乘子下差异必保留”的跳步推理。
