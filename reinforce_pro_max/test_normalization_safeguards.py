"""Actual normalization examples corresponding to the exact-real moment formulas.

Uses normal package imports. CPU numerical checks complement, and do not
replace, formal/NormalizationSafeguards.lean or distributed training tests.
"""

import math

import pytest
import torch

from openrlhf.trainer.ppo_utils.experience_maker import _adaptive_token_normalization_single_group


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stabilizer_variance_deficit(dtype):
    # P=1,N=2,kappa=1/2,Q+=1,Q-=4: D=2, D_c=2.1.
    advantage = torch.tensor([[1.0, -2.0, 0.0, 123.0]], dtype=dtype)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=dtype)
    actual = _adaptive_token_normalization_single_group(advantage, mask, eps=0.1)
    expected = torch.tensor([[math.sqrt(20 / 21), -math.sqrt(20 / 21), 0, 0]], dtype=dtype)
    torch.testing.assert_close(actual, expected)
    nonzero = actual[0, :2]
    assert nonzero.mean().item() == pytest.approx(0, abs=1e-6)
    assert nonzero.var(unbiased=False).item() == pytest.approx(20 / 21, abs=1e-6)
    # Including the active zero token changes the denominator; the theorem
    # explicitly uses nonzero active tokens, not all active tokens.
    assert actual[0, :3].var(unbiased=False).item() == pytest.approx(40 / 63, abs=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_product_cap_can_increase_variance(dtype):
    # D=8e8, D_c=4e8+1e8+eps: variance approximately 1.6, not <=1.
    advantage = torch.tensor([[20000.0, -20000.0]], dtype=dtype)
    actual = _adaptive_token_normalization_single_group(advantage, torch.ones_like(advantage))
    expected_variance = 800000000 / (500000000 + 1e-8)
    assert actual.mean().item() == pytest.approx(0, abs=1e-6)
    assert actual.var(unbiased=False).item() == pytest.approx(expected_variance, abs=1e-6)
    assert actual.var(unbiased=False).item() > 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_upper_factor_clamp_changes_mean_and_variance(dtype):
    # Both ideal factors exceed 10: final factors both equal 10. Unequal
    # magnitudes then break the zero mean, while preserving both signs.
    advantage = torch.tensor([[1e-4, -1e-3]], dtype=dtype)
    actual = _adaptive_token_normalization_single_group(advantage, torch.ones_like(advantage))
    torch.testing.assert_close(actual, 10 * advantage)
    assert actual.mean().item() == pytest.approx(-0.0045, abs=1e-8)
    assert actual.var(unbiased=False).item() == pytest.approx(0.00003025, abs=1e-9)
    assert torch.equal(actual.sign(), advantage.sign())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lower_factor_clamp_changes_mean_and_variance(dtype):
    # Both computed factors lie below eps=0.1; final values are 10,-20.
    advantage = torch.tensor([[100.0, -200.0]], dtype=dtype)
    actual = _adaptive_token_normalization_single_group(advantage, torch.ones_like(advantage), eps=0.1)
    torch.testing.assert_close(actual, torch.tensor([[10.0, -20.0]], dtype=dtype))
    assert actual.mean().item() == pytest.approx(-5)
    assert actual.var(unbiased=False).item() == pytest.approx(225)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_small_signed_sum_fallback_is_identity(dtype):
    advantage = torch.tensor([[1e-10, -1.0]], dtype=dtype)
    actual = _adaptive_token_normalization_single_group(advantage, torch.ones_like(advantage))
    assert torch.equal(actual, advantage)
