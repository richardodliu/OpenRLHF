# 拆分 main.tex 到 main/ 子文件

## Context
将 `tex/main.tex` 的各 section 内容拆到 `tex/main/` 下已有的空文件中，main.tex 保留 preamble + `\input{}` 引用。

## 拆分映射

| main.tex 行范围 | 目标文件 | 内容 |
|---|---|---|
| 84-107 | `main/1-intro.tex` | Section 1: Introduction |
| 113-182 | `main/2-related.tex` | Section 2: Background and Problem Setup |
| 188-329 | `main/3-premilary.tex` | Section 3: REINFORCE Max |
| 335-494 | `main/4-method.tex` | Section 4: REINFORCE Pro |
| 500-566 | `main/5-theory.tex` | Section 5: The Unified Framework |
| 572-575 | `main/6-experiment.tex` | Section 6: Experiments |
| 581-584 | `main/7-conclusion.tex` | Section 7: Conclusion |
| 590-672 | `main/reference.bib` | Bibliography (thebibliography 环境) |
| 680-777 | `main/appendix.tex` | Appendix A-C |

## main.tex 改造
保留 L1-78（preamble + abstract），将各 section 替换为 `\input{main/X.tex}`。

## 修改文件
- `tex/main.tex` — 改为 \input 引用
- `tex/main/1-intro.tex` ~ `tex/main/appendix.tex` — 填入对应内容
- `tex/main/reference.bib` — 填入 thebibliography 内容
