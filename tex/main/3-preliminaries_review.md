# `3-preliminaries.tex` 审稿意见

## 总评
本章相较旧版已有实质改进（已补充 surrogate 替换说明与 coupling 引理条件），但仍存在两个核心理论风险：surrogate 化简与性能差分 surrogate 的等价关系未严格闭环、context shift 传播证明仍偏直觉化。

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
  问题: `Context Shift Propagation` 已加入可达性与支持条件，但证明仍以“joint 不同 -> marginal 不同”直觉论证为主，缺少可验证的不等式链或构造式下界。
  影响: `4-method` 中关于 token-level IS 局限性的理论依托不够稳固。
  建议: 补充严格证明（例如显式构造一个未来上下文事件，其在两策略下概率差严格大于 0）。

## 逻辑问题
- [P1] 类型: 逻辑
  位置: `tex/main/3-preliminaries.tex:98`
  问题: 证明中“joint 分布不同即可推出对应 marginal 必不同”的推理缺关键条件（需要指定映射与不可抵消性）；当前文字未排除边际求和后的抵消情形。
  影响: 引理证明链存在跳步，降低可审计性。
  建议: 增加从 prefix-joint 到 `d_t` 的映射写法，并明确哪一类事件保证差异在边际下保留。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P0: 为 surrogate 化简补全严格条件与推导，或改回标准逐时刻形式。
2. P0: 重写 `Context Shift Propagation` 的结论条件与证明链。
3. P1: 补齐 joint-to-marginal 推导步骤，避免逻辑跳跃。
