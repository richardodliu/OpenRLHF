"""Reproducible finite-pattern and Monte Carlo checks for the ProMax paper.

This script deliberately uses synthetic rewards and log-ratio traces.  It does
not claim language-model or RLVR performance; it checks the structural
properties stated in tex/main/6-experiment.tex with a fixed random seed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def prefix_masks(log_ratios: torch.Tensor, low: float, high: float) -> dict[str, list[int]]:
    """Return token, sequence and prefix rejection positions (1-based)."""
    prefix_mean = torch.cumsum(log_ratios, 0) / torch.arange(1, log_ratios.numel() + 1)
    sequence_mean = log_ratios.mean()
    token = (log_ratios < math.log(low)) | (log_ratios > math.log(high))
    sequence_reject = bool(sequence_mean < math.log(low) or sequence_mean > math.log(high))
    prefix = (prefix_mean < math.log(low)) | (prefix_mean > math.log(high))
    return {
        "token": (torch.where(token)[0] + 1).tolist(),
        "sequence": (list(range(1, log_ratios.numel() + 1)) if sequence_reject else []),
        "prefix": (torch.where(prefix)[0] + 1).tolist(),
    }


def deterministic_patterns(T: int = 32, low: float = 0.9, high: float = 1.1) -> dict:
    """Reproduce the early, late and sustained deviations in the manuscript."""
    tau_plus = math.log(high)
    patterns = {
        "early": torch.tensor([8 * tau_plus] + [0.0] * (T - 1)),
        "late": torch.tensor([0.0] * 23 + [32 * tau_plus] + [0.0] * 8),
        "drift": torch.full((T,), 2 * tau_plus),
    }
    return {name: prefix_masks(values, low, high) for name, values in patterns.items()}


def rloo_and_mean(rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = rewards.numel()
    # The ordinary mean baseline includes the sample itself; the RLOO
    # baseline excludes it.  Their shaped signals differ by (n-1)/n.
    mean_baseline = rewards.mean()
    rloo_baseline = (rewards.sum() - rewards) / (n - 1)
    return rewards - mean_baseline, rewards - rloo_baseline


def adaptive_ideal(values: torch.Tensor) -> torch.Tensor:
    """Ideal real-valued alpha/beta normalization used by the paper's proof."""
    positive = values[values > 0]
    negative = values[values < 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return values.clone()
    s_pos = positive.sum()
    s_neg = negative.sum()
    q_pos = (positive**2).sum()
    q_neg = (negative**2).sum()
    count = positive.numel() + negative.numel()
    ratio = s_pos / s_neg
    alpha = torch.sqrt(torch.tensor(float(count), dtype=values.dtype) / (q_pos + ratio**2 * q_neg))
    beta = -alpha * ratio
    return torch.where(values > 0, alpha * values, torch.where(values < 0, beta * values, values))


def sign_flip_statistics(raw: torch.Tensor, normalized: torch.Tensor) -> dict:
    """Count sign changes with nonzero raw active tokens as the denominator."""
    eligible = raw.ne(0)
    count = int(eligible.sum())
    flips = int(((normalized.sign() != raw.sign()) & eligible).sum())
    return {"eligible_tokens": count, "flipped_tokens": flips, "rate": flips / count if count else None}


def monte_carlo(groups: int, n: int, seed: int) -> dict:
    if groups < 1 or n < 2:
        raise ValueError("groups must be positive and samples_per_prompt must be at least 2")
    generator = torch.Generator().manual_seed(seed)
    success = torch.rand((groups, n), generator=generator) < 0.15
    perturb = (torch.rand((groups, n), generator=generator) * 0.2) - 0.1
    rewards = torch.where(success, 1.0 + perturb, perturb).clamp(-0.1, 1.1)
    lengths = torch.where(
        success,
        torch.randint(224, 289, (groups, n), generator=generator),
        torch.randint(480, 545, (groups, n), generator=generator),
    )

    mean_adv, rloo_adv = [], []
    adaptive_values, global_values = [], []
    for reward_row, length_row in zip(rewards, lengths):
        mean_signal, rloo_signal = rloo_and_mean(reward_row)
        mean_adv.append(mean_signal)
        rloo_adv.append(rloo_signal)
        expanded = rloo_signal.repeat_interleave(length_row)
        adaptive_values.append(adaptive_ideal(expanded))
        global_values.append((expanded - expanded.mean()) / expanded.std(unbiased=False).clamp_min(1e-12))

    mean_adv = torch.stack(mean_adv)
    rloo_adv = torch.stack(rloo_adv)
    adaptive_values = torch.cat(adaptive_values)
    global_values = torch.cat(global_values)
    raw_rewards = rewards.flatten()
    mean_baselines = rewards.mean(1, keepdim=True).expand_as(rewards)
    rloo_baselines = (rewards.sum(1, keepdim=True) - rewards) / (n - 1)

    def correlation(x: torch.Tensor, y: torch.Tensor) -> float:
        return float(torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1])

    # Sign flips are counted only where the raw signal is nonzero.
    raw_token_signal = torch.cat([r.repeat_interleave(l) for r, l in zip(rloo_adv, lengths)])
    adaptive_flips = sign_flip_statistics(raw_token_signal, adaptive_values)
    global_flips = sign_flip_statistics(raw_token_signal, global_values)
    return {
        "groups": groups,
        "samples_per_prompt": n,
        "seed": seed,
        "mean_baseline_reward_correlation": correlation(raw_rewards, mean_baselines),
        "rloo_baseline_reward_correlation": correlation(raw_rewards, rloo_baselines),
        "active_tokens": raw_token_signal.numel(),
        "nonzero_active_tokens": adaptive_flips["eligible_tokens"],
        "adaptive_sign_flipped_tokens": adaptive_flips["flipped_tokens"],
        "global_sign_flipped_tokens": global_flips["flipped_tokens"],
        "adaptive_sign_flip_rate": adaptive_flips["rate"],
        "global_sign_flip_rate": global_flips["rate"],
        "adaptive_nonzero_mean": float(adaptive_values[raw_token_signal.ne(0)].mean()),
        "adaptive_nonzero_population_variance": float(
            adaptive_values[raw_token_signal.ne(0)].var(unbiased=False)
        ),
        "rloo_group_sum_max_abs": float(rloo_adv.sum(1).abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=10000)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(3)
    result = {
        "runtime": {"torch_version": torch.__version__, "threads": torch.get_num_threads(),
                    "device": "cpu", "dtype": "float32"},
        "deterministic_patterns": deterministic_patterns(),
        "monte_carlo": monte_carlo(args.groups, args.samples_per_prompt, args.seed),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
