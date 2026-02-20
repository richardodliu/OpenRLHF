# Plan: 撰写 REINFORCE Pro Max 理论论文

## Context

用户需要在 `tex/reinforce_pro_max.tex` 中撰写一篇 RL theory 论文，从理论上证明 REINFORCE Pro Max 算法的优越性。参考论文 `tex/main_arxiv.tex`（Trust Region Masking）的 LaTeX 风格和符号体系。参考文献使用 `tex/ref.bib`（用户已添加）。

## 论文标题

**REINFORCE Pro Max: Variance-Optimal Policy Gradient with Asymmetric Normalization for LLM Reasoning**

## 论文结构

### 主体部分

1. **Abstract** (~150 words)
   - 问题：LLM 推理任务中策略梯度方差高、off-policy mismatch
   - 三个创新：RLOO baseline、自适应非对称归一化、前缀累积 IS
   - 理论贡献：每个组件的方差缩减证明

2. **Section 1: Introduction**
   - 动机：LLM 推理（数学、代码）使用 0/1 奖励的 RL，梯度方差是瓶颈
   - 现有方法的三个问题
   - 贡献列表（3 个定理 + 算法）

3. **Section 2: Background and Problem Setup**
   - 2.1 自回归生成与目标函数（复用 main_arxiv.tex 符号）
   - 2.2 策略梯度与 Baseline
   - 2.3 现有方法：REINFORCE++, GRPO, RLOO, PPO

4. **Section 3: RLOO Baseline — Variance-Optimal Leave-One-Out Estimation**
   - 3.1 符号与设定
   - 3.2 **Theorem 1**: RLOO 是独立线性无偏 baseline 中方差最小的
   - 3.3 与控制变量的联系

5. **Section 4: Adaptive Asymmetric Normalization**
   - 4.1 动机：偏斜奖励分布（推理任务中大多数回答错误）
   - 4.2 α/β 归一化定义
   - 4.3 **Theorem 2**: 唯一性与最优性证明
   - 4.4 **Theorem 3**: 梯度方差缩减（per-group vs global）

6. **Section 5: Prefix Cumulative Importance Sampling**
   - 5.1 Off-policy mismatch 回顾
   - 5.2 前缀累积 IS 定义（geometric/arithmetic）
   - 5.3 **Theorem 4**: 因果结构保持 + 近似误差界
   - 5.4 与 token-level IS 和 sequence-level IS 的关系

7. **Section 6: The REINFORCE Pro Max Algorithm**
   - 6.1 算法伪代码（Algorithm 1）
   - 6.2 **Theorem 5**: 组合方差界
   - 6.3 Uniform Scale 模式
   - 6.4 与现有方法的关系表

8. **Section 7: Conclusion**

### 附录

- **Appendix A**: Theorem 1 完整证明
- **Appendix B**: Theorem 2-3 完整证明
- **Appendix C**: Theorem 4 完整证明
- **Appendix D**: 数值稳定性分析

## 核心定理

### Theorem 1: RLOO 最优独立 Baseline

**设定**: 给定 prompt q，采样 n 个 i.i.d. 回答，奖励 r_i，方差 σ²。

**结论**:
- RLOO baseline b_i^LOO = Σ_{j≠i} r_j / (n-1) 是独立于 o_i 的
- 在所有独立线性无偏 baseline 中，RLOO 的 baseline 方差最小：Var[b_i^LOO] = σ²/(n-1)
- 关键恒等式：r_i - r̄ = (n-1)/n · (r_i - b_i^LOO)

### Theorem 2: 非对称归一化的唯一性

**设定**: 给定 advantage 向量，分为正集 P 和负集 N。

**结论**: 在约束 (1) 均值=0, (2) 方差=1, (3) 符号保持 下，α/β 解是唯一的：
- α = √(N / (Q⁺ + (S⁺/S⁻)²Q⁻))
- β = -α · S⁺/S⁻

### Theorem 3: Per-Group 归一化保持组内排序

**结论**: Per-group 非对称归一化保持同一 prompt 组内 advantage 的相对排序，而全局归一化在组间奖励尺度不同时会破坏排序。

### Theorem 4: 前缀累积 IS 的因果性与误差界

**结论**:
- (a) 因果一致性：位置 t 的梯度仅依赖 ℓ_1,...,ℓ_t
- (b) 插值性：t=1 退化为 token-level，t=T 退化为 sequence-level
- (c) 近似误差界：|L^prefix - L^tok| = O(Tδ)

### Theorem 5: 组合方差界

**结论**: REINFORCE Pro Max 的梯度方差 ≤ (1/n)·((n-2)/(n-1))·σ²_R·F(θ) + O(δ)

## 需要修改/创建的文件

- `tex/reinforce_pro_max.tex` — 新论文（主要工作）
- `tex/ref.bib` — 添加新引用条目（RLOO, REINFORCE++, GRPO 等）

## 新增参考文献

需要在 ref.bib 中添加：
- Kool et al. 2019 (RLOO/LOORF)
- Hu 2025 (REINFORCE++)
- Ahmadian et al. 2024 (REINFORCE revisited)
- Greensmith et al. 2004 (variance reduction)
- GSPO paper

## LaTeX 风格

- 复用 main_arxiv.tex 的 preamble（packages, theorem environments, custom commands）
- 添加新的 custom commands（如 \piref, \rloo 等）
- 使用相同的 documentclass[11pt]{article}
- 引用 ref.bib

## 验证方式

1. 在 tex/ 目录下运行 `pdflatex reinforce_pro_max.tex && bibtex reinforce_pro_max && pdflatex reinforce_pro_max.tex && pdflatex reinforce_pro_max.tex` 确认编译通过
2. 检查所有定理的数学公式正确性
3. 确认引用完整
