# `main.tex` 审稿意见

## 总评
主控结构（章节组织、引用接入、附录衔接）已完整。当前主要风险来自理论主张对 `3-preliminaries`、`4-method`、`5-theory` 的关键证明链依赖。

## 结构不完整
- 当前未发现未修复问题。

## 证明不严谨
- [P1] 类型: 证明
  位置: `tex/main.tex:16`
  问题: 摘要中“prefix cumulative IS can provide tighter masking ... in three analyzed deviation patterns”依赖上游理论链完整成立；当前仍有未闭环项（含 preliminaries 的 coupling 证明链缺口、method 的 Part (b) 条件同构问题、以及 unified 中 `\pi_{old}\!\to\!\pi_\theta` 桥接不足）。
  影响: 摘要结论的可证范围受上游定理缺口限制。
  建议: 在 `3-preliminaries` / `4-method` / `5-theory` 完整补证前，将摘要进一步限定为“under the currently proved sufficient cases/conditions”；具体缺陷请见对应章节 review。

## 逻辑问题
- 当前未发现未修复问题。

## 表达缺陷
- 当前未发现未修复问题。

## 高优先级修改清单（P0/P1/P2）
1. P1: 让摘要中 “tighter masking” 的结论强度与 `3/4/5` 章当前可证范围严格一致。
