"""Check loss and gradients against the explicit unclipped token IS objective."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    os.environ.pop('PROMAX_STUDY_METRICS', None)
    import torch
    from openrlhf.models.loss import PolicyLoss
    dtype = torch.float64
    rollout = torch.full((1, 6), -4., dtype=dtype)
    old = rollout + torch.tensor([[math.log(4), math.log(.25), math.log(.25), math.log(4), 0, 0]], dtype=dtype)
    values = rollout + torch.tensor([[math.log(4), math.log(4), math.log(.25), math.log(.25), math.log(1.1), math.log(.9)]], dtype=dtype)
    advantage = torch.tensor([[1., -1., 1., -1., 1., -1.]], dtype=dtype)
    mask = torch.ones_like(advantage)
    checked = []
    for pro in [False, True]:
        for scaled in [False, True]:
            a = advantage * torch.where(advantage > 0, 1.7 if scaled else 1., .6 if scaled else 1.)
            current = values.clone().requires_grad_()
            fn = PolicyLoss(clip_eps_low=.2, clip_eps_high=.28, policy_loss_type='token_is',
                            enable_vllm_is_correction=pro, vllm_is_correction_type='reinforce_pro',
                            vllm_is_truncated_threshold=[.5, 2.])
            actual = fn(current, old, a, mask, rollout)[0]
            grad = torch.autograd.grad(actual, current)[0]
            reference = values.clone().requires_grad_()
            ratio = (reference-rollout).exp()
            expected_tokens = -ratio*a
            if pro:
                prefix = ((reference.detach()-rollout).cumsum(-1)/torch.arange(1,7,dtype=dtype)).exp()
                expected_tokens *= ((prefix >= .5) & (prefix <= 2.)).to(dtype)
            expected = expected_tokens.mean()
            expected_grad = torch.autograd.grad(expected, reference)[0]
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(grad, expected_grad)
            changed_old = fn(values, old+2, a, mask, rollout)[0]
            torch.testing.assert_close(actual.detach(), changed_old)
            checked.append({'pro':pro,'scaled_advantages':scaled,'loss':actual.item()})
    # User-selected interval: below, inside, above, and causal reentry.
    bounds = [.8, 1.25]
    small_ratio = torch.tensor([[.5, 2., 4., .25]], dtype=torch.float32)
    rollout32 = torch.zeros_like(small_ratio)
    current32 = small_ratio.log().requires_grad_()
    ones = torch.ones_like(small_ratio)
    narrow = PolicyLoss(policy_loss_type='token_is', enable_vllm_is_correction=True,
                        vllm_is_correction_type='reinforce_pro', vllm_is_truncated_threshold=bounds)
    actual = narrow(current32, rollout32, ones, ones, rollout32)[0]
    prefix = (current32.detach().cumsum(-1)/torch.arange(1,5)).exp()
    accepted = (prefix >= bounds[0]) & (prefix <= bounds[1])
    assert accepted.tolist() == [[False, True, False, True]], accepted
    expected = -(small_ratio*accepted).mean()
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, current32)[0]
    torch.testing.assert_close(gradient, -small_ratio*accepted/4)
    changed_future = current32.detach().clone();changed_future[0,-1] += 1
    changed_prefix = (changed_future.cumsum(-1)/torch.arange(1,5)).exp()
    torch.testing.assert_close(prefix[:,:-1], changed_prefix[:,:-1])
    checked.append({'thresholds':bounds,'prefix_acceptance':accepted.tolist(),'causal_reentry':True})
    try:
        fn(values, old, advantage, mask, None)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing rollout logprobs must fail')
    print(json.dumps({'status':'passed','loss_and_gradient_cases':checked,
                      'objective_and_mask_independent_of_cached_old':True,'missing_rollout_rejected':True},indent=2))


if __name__ == '__main__':
    main()
