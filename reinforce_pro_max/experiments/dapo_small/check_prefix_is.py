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
