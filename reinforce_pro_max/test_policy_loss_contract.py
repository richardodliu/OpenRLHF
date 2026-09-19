"""CPU checks of actual loss.py/utils.py against independent scalar formulas.

Load the complete source modules under a private package name to avoid the
models/__init__.py imports of model/accelerator dependencies. No function is
copied or mocked. This is loss-level verification, not a Ray/vLLM integration
or a proof of floating-point correctness.
"""

import importlib.util
import math
from pathlib import Path
import sys
import types

import pytest
import torch


def load_policy_loss():
    root = Path(__file__).resolve().parents[1] / "openrlhf" / "models"
    package_name = "_promax_actual_models_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    for name in ("utils", "loss"):
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(full_name, root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.loss"].PolicyLoss


PolicyLoss = load_policy_loss()


def test_dual_clip_invalidates_standard_ppo_nonnegative_deduction():
    # Same rational counterexample as ProMaxObjective.lean, evaluated through
    # the real prefix-IS branch. The cached old/rollout ratio is one, so the
    # prefix gate accepts even though the current/old ratio is four.
    old = torch.full((1, 1), -5.0, dtype=torch.float64)
    current = (old + math.log(4)).requires_grad_()
    advantage = -torch.ones_like(old)
    mask = torch.ones_like(old)
    losses, gradients = [], []
    for dual_clip in (None, 3.0):
        objective = PolicyLoss(
            dual_clip=dual_clip, enable_vllm_is_correction=True,
            vllm_is_correction_type="reinforce_pro",
            vllm_is_truncated_threshold=[0.5, 5.0],
        )
        loss = objective(current, old, advantage, mask, old)[0]
        losses.append(loss.item())
        gradients.append(torch.autograd.grad(loss, current)[0].item())
    assert losses == pytest.approx([4.0, 3.0])
    assert gradients == pytest.approx([4.0, 0.0])
    raw_maximization_objective = -4.0
    assert raw_maximization_objective + losses[1] == pytest.approx(-1.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("token_mean", [True, False])
@pytest.mark.parametrize("mode", ["reinforce_pro", "seq-mask-tis", "icepop", "tis"])
def test_actual_loss_gradient_and_diagnostics(mode, token_mean, dtype):
    # Distinct current/old and old/rollout ratios catch accidental substitution.
    # Prefix product returns to one after the first rejected token; masked
    # positions include an internal gap to exercise active-token counting.
    w = [[2.0, 0.5, 1.0, 9.0], [1.0, 7.0, 1.5, 0.8]]
    r = [[0.7, 1.1, 1.4, 1.05], [1.3, 0.9, 0.6, 1.1]]
    adv = [[1.0, -2.0, 3.0, 4.0], [-1.0, 8.0, -2.0, 1.0]]
    active = [[1, 1, 1, 0], [1, 0, 1, 0]]
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    old = torch.full((2, 4), -5.0, dtype=dtype)
    rollout = (old - torch.tensor(w, dtype=dtype).log()).requires_grad_()
    current = (old + torch.tensor(r, dtype=dtype).log()).requires_grad_()
    mask = torch.tensor(active, dtype=dtype)
    objective = PolicyLoss(
        token_level_loss=token_mean, enable_vllm_is_correction=True,
        vllm_is_correction_type=mode, vllm_is_truncated_threshold=[0.8, 1.2],
    )
    loss, clip_fraction, ppo_kl, vllm_kl = objective(
        current, old, torch.tensor(adv, dtype=dtype), mask, rollout,
    )
    expected_loss = 0.0
    expected_gradient = torch.zeros_like(current)
    log_r, log_w, clipped = [], [], []
    total = sum(map(sum, active))
    for i in range(2):
        prefix = []
        seq = math.exp(math.fsum(math.log(w[i][j]) for j in range(4) if active[i][j]) / sum(active[i]))
        for j in range(4):
            if not active[i][j]:
                continue
            prefix.append(math.log(w[i][j]))
            if mode == "tis":
                correction = min(1.2, max(0.8, w[i][j]))
            else:
                statistic = {"reinforce_pro": math.exp(math.fsum(prefix) / len(prefix)),
                             "seq-mask-tis": seq, "icepop": w[i][j]}[mode]
                correction = w[i][j] if 0.8 <= statistic <= 1.2 else 0.0
            raw = r[i][j] * adv[i][j]
            capped = min(1.2, max(0.8, r[i][j])) * adv[i][j]
            weight = 1 / total if token_mean else 1 / (2 * sum(active[i]))
            expected_loss -= weight * correction * min(raw, capped)
            if raw < capped or 0.8 < r[i][j] < 1.2:
                expected_gradient[i, j] = -weight * correction * raw
            log_r.append(math.log(r[i][j]))
            log_w.append(math.log(w[i][j]))
            clipped.append(capped < raw)
    loss.backward()
    assert rollout.grad is None, "IS correction must not differentiate through rollout logprobs"
    assert loss.item() == pytest.approx(expected_loss, abs=tolerance)
    torch.testing.assert_close(current.grad, expected_gradient, atol=tolerance, rtol=tolerance)
    assert ppo_kl.item() == pytest.approx(-math.fsum(log_r) / total, abs=tolerance)
    assert vllm_kl.item() == pytest.approx(-math.fsum(log_w) / total, abs=tolerance)
    assert clip_fraction.item() == pytest.approx(sum(clipped) / total)


def test_response_weighted_microbatches_do_not_reconstruct_token_mean():
    # Exact example corresponding to dynamic_token_weight_counterexample in Lean.
    # Contributions sum to 0 and 3, while active lengths are 1 and 3.
    objective = PolicyLoss(token_level_loss=True)
    logp = torch.full((2, 3), -5.0, dtype=torch.float64)
    advantages = torch.tensor([[0., 0., 0.], [-1., -1., -1.]], dtype=torch.float64)
    mask = torch.tensor([[1., 0., 0.], [1., 1., 1.]], dtype=torch.float64)
    pooled = objective(logp, logp, advantages, mask)[0]
    pieces = [objective(logp[i:i+1], logp[i:i+1], advantages[i:i+1], mask[i:i+1])[0]
              for i in range(2)]
    response_weighted = (pieces[0] + pieces[1]) / 2
    token_weighted = (pieces[0] + 3 * pieces[1]) / 4
    assert pooled.item() == 0.75
    assert response_weighted.item() == 0.5
    assert token_weighted.item() == pooled.item()
