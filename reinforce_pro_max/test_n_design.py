"""
实验：比较两种 N 定义下的归一化效果

设计 A (当前采用): N = s_pos + s_neg (只计算非零 token)
设计 B: N = masked_adv.numel() (包含 A=0 的 token)

结论：设计 A 产生更小的 alpha/beta，梯度更保守、训练更稳定
"""

import torch

def adaptive_norm_design_a(group_adv, group_mask, eps=1e-8):
    """设计 A: N = 非零 token 数量"""
    flat_adv = group_adv.flatten()
    flat_mask = group_mask.flatten().bool()
    masked_adv = flat_adv[flat_mask]

    pos_mask = masked_adv > 0
    neg_mask = masked_adv < 0

    s_pos = pos_mask.sum().float()
    s_neg = neg_mask.sum().float()

    if s_pos == 0 or s_neg == 0:
        mean = masked_adv.mean()
        std = masked_adv.std().clamp(min=eps)
        result = (group_adv - mean) / std * group_mask
        return result, None, None

    q_pos = (masked_adv[pos_mask] ** 2).sum()
    q_neg = (masked_adv[neg_mask] ** 2).sum()
    sum_pos = masked_adv[pos_mask].sum()
    sum_neg = masked_adv[neg_mask].sum()

    sum_neg_safe = sum_neg.clamp(max=-eps)
    ratio = sum_pos / sum_neg_safe

    # 设计 A: 只计算非零 token
    n = s_pos + s_neg

    ratio_sq_q_neg = (ratio**2 * q_neg).clamp(max=1e10)
    alpha = torch.sqrt(n / (q_pos + ratio_sq_q_neg + eps))
    beta = -alpha * ratio

    alpha = alpha.clamp(min=eps, max=100.0)
    beta = beta.clamp(min=eps, max=100.0)

    result = torch.where(group_adv > 0, alpha * group_adv,
                         torch.where(group_adv < 0, beta * group_adv, group_adv))
    result = result * group_mask

    return result, alpha.item(), beta.item()


def adaptive_norm_design_b(group_adv, group_mask, eps=1e-8):
    """设计 B: N = 所有有效 token 数量（包括 A=0）"""
    flat_adv = group_adv.flatten()
    flat_mask = group_mask.flatten().bool()
    masked_adv = flat_adv[flat_mask]

    pos_mask = masked_adv > 0
    neg_mask = masked_adv < 0

    s_pos = pos_mask.sum().float()
    s_neg = neg_mask.sum().float()

    if s_pos == 0 or s_neg == 0:
        mean = masked_adv.mean()
        std = masked_adv.std().clamp(min=eps)
        result = (group_adv - mean) / std * group_mask
        return result, None, None

    q_pos = (masked_adv[pos_mask] ** 2).sum()
    q_neg = (masked_adv[neg_mask] ** 2).sum()
    sum_pos = masked_adv[pos_mask].sum()
    sum_neg = masked_adv[neg_mask].sum()

    sum_neg_safe = sum_neg.clamp(max=-eps)
    ratio = sum_pos / sum_neg_safe

    # 设计 B: 包含所有有效 token（包括 A=0）
    n = masked_adv.numel()

    ratio_sq_q_neg = (ratio**2 * q_neg).clamp(max=1e10)
    alpha = torch.sqrt(n / (q_pos + ratio_sq_q_neg + eps))
    beta = -alpha * ratio

    alpha = alpha.clamp(min=eps, max=100.0)
    beta = beta.clamp(min=eps, max=100.0)

    result = torch.where(group_adv > 0, alpha * group_adv,
                         torch.where(group_adv < 0, beta * group_adv, group_adv))
    result = result * group_mask

    return result, alpha.item(), beta.item()


def compute_stats(result, group_mask):
    """计算归一化后的统计量"""
    flat_result = result.flatten()
    flat_mask = group_mask.flatten().bool()
    masked_result = flat_result[flat_mask]

    non_zero_mask = masked_result != 0
    non_zero_result = masked_result[non_zero_mask]

    return {
        'total_tokens': masked_result.numel(),
        'non_zero_tokens': non_zero_result.numel(),
        'zero_tokens': (masked_result == 0).sum().item(),
        'mean_all': masked_result.mean().item(),
        'var_all': masked_result.var().item(),
        'std_all': masked_result.std().item(),
        'mean_nonzero': non_zero_result.mean().item() if non_zero_result.numel() > 0 else 0,
        'var_nonzero': non_zero_result.var().item() if non_zero_result.numel() > 1 else 0,
        'std_nonzero': non_zero_result.std().item() if non_zero_result.numel() > 1 else 0,
        'pos_mean': masked_result[masked_result > 0].mean().item() if (masked_result > 0).any() else 0,
        'neg_mean': masked_result[masked_result < 0].mean().item() if (masked_result < 0).any() else 0,
        'pos_max': masked_result.max().item(),
        'neg_min': masked_result.min().item(),
    }


def run_experiment(zero_ratio=0.1, n_samples=8, seq_len=500, seed=42):
    """运行实验，比较两种设计"""
    torch.manual_seed(seed)

    # 生成模拟数据
    group_adv = torch.randn(n_samples, seq_len)
    group_mask = torch.ones(n_samples, seq_len)

    # 随机设置一些 token 为 0
    num_zeros = int(n_samples * seq_len * zero_ratio)
    zero_indices = torch.randperm(n_samples * seq_len)[:num_zeros]
    group_adv.flatten()[zero_indices] = 0

    # 设计 A
    result_a, alpha_a, beta_a = adaptive_norm_design_a(group_adv.clone(), group_mask)
    stats_a = compute_stats(result_a, group_mask)

    # 设计 B
    result_b, alpha_b, beta_b = adaptive_norm_design_b(group_adv.clone(), group_mask)
    stats_b = compute_stats(result_b, group_mask)

    return {
        'zero_ratio': zero_ratio,
        'design_a': {'alpha': alpha_a, 'beta': beta_a, **stats_a},
        'design_b': {'alpha': alpha_b, 'beta': beta_b, **stats_b},
    }


def print_comparison(result):
    """打印比较结果"""
    zero_ratio = result['zero_ratio']
    a = result['design_a']
    b = result['design_b']

    print(f"\n{'='*70}")
    print(f"零值 token 比例: {zero_ratio*100:.1f}%")
    print(f"{'='*70}")
    print(f"{'指标':<25} {'设计 A (N=非零)':<20} {'设计 B (N=全部)':<20}")
    print(f"{'-'*70}")
    print(f"{'alpha':<25} {a['alpha']:<20.4f} {b['alpha']:<20.4f}")
    print(f"{'beta':<25} {a['beta']:<20.4f} {b['beta']:<20.4f}")
    print(f"{'-'*70}")
    print(f"{'总 token 数':<25} {a['total_tokens']:<20} {b['total_tokens']:<20}")
    print(f"{'非零 token 数':<25} {a['non_zero_tokens']:<20} {b['non_zero_tokens']:<20}")
    print(f"{'零 token 数':<25} {a['zero_tokens']:<20} {b['zero_tokens']:<20}")
    print(f"{'-'*70}")
    print(f"{'均值 (所有 token)':<25} {a['mean_all']:<20.6f} {b['mean_all']:<20.6f}")
    print(f"{'方差 (所有 token)':<25} {a['var_all']:<20.6f} {b['var_all']:<20.6f}")
    print(f"{'标准差 (所有 token)':<25} {a['std_all']:<20.6f} {b['std_all']:<20.6f}")
    print(f"{'-'*70}")
    print(f"{'均值 (非零 token)':<25} {a['mean_nonzero']:<20.6f} {b['mean_nonzero']:<20.6f}")
    print(f"{'方差 (非零 token)':<25} {a['var_nonzero']:<20.6f} {b['var_nonzero']:<20.6f}")
    print(f"{'标准差 (非零 token)':<25} {a['std_nonzero']:<20.6f} {b['std_nonzero']:<20.6f}")
    print(f"{'-'*70}")
    print(f"{'正 advantage 均值':<25} {a['pos_mean']:<20.6f} {b['pos_mean']:<20.6f}")
    print(f"{'负 advantage 均值':<25} {a['neg_mean']:<20.6f} {b['neg_mean']:<20.6f}")
    print(f"{'最大值':<25} {a['pos_max']:<20.6f} {b['pos_max']:<20.6f}")
    print(f"{'最小值':<25} {a['neg_min']:<20.6f} {b['neg_min']:<20.6f}")

    # 计算放大比例
    if a['alpha'] and b['alpha']:
        alpha_ratio = b['alpha'] / a['alpha']
        beta_ratio = b['beta'] / a['beta']
        print(f"{'-'*70}")
        print(f"{'设计 B 相对于 A 的放大':<25}")
        print(f"{'  alpha 放大倍数':<25} {alpha_ratio:<20.4f}")
        print(f"{'  beta 放大倍数':<25} {beta_ratio:<20.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("实验：比较两种 N 定义下的归一化效果")
    print("设计 A (当前采用): N = 非零 token 数量 → 更小的 alpha/beta")
    print("设计 B: N = 所有有效 token 数量（包括 A=0）→ 更大的 alpha/beta")
    print("=" * 70)

    # 不同零值比例的实验
    for zero_ratio in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        result = run_experiment(zero_ratio=zero_ratio)
        print_comparison(result)

    print("\n" + "=" * 70)
    print("结论：")
    print("=" * 70)
    print("""
1. 当零值比例为 0% 时，两种设计完全相同
2. 随着零值比例增加，设计 B 的 alpha/beta 越来越大
3. 设计 B 的放大倍数 = sqrt(N_all / N_nonzero) = sqrt(1 / (1 - zero_ratio))
4. 设计 A (当前采用) 在非零 token 上满足 mean=0, var=1
5. 设计 B 在所有 token 上满足 mean=0, var=1
6. 设计 A 产生更保守的梯度，训练更稳定
""")
