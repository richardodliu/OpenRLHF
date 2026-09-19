"""Regression checks for the paper's synthetic measurement protocol."""

import torch

from reinforce_pro_max.structural_experiment import deterministic_patterns, sign_flip_statistics


def test_sign_flip_denominator_excludes_raw_zeros():
    raw = torch.tensor([0.0, 1.0, -1.0])
    normalized = torch.tensor([2.0, -1.0, -1.0])
    assert sign_flip_statistics(raw, normalized) == {
        "eligible_tokens": 2, "flipped_tokens": 1, "rate": 0.5
    }


def test_sign_flip_empty_denominator_is_undefined():
    assert sign_flip_statistics(torch.zeros(2), torch.ones(2)) == {
        "eligible_tokens": 0, "flipped_tokens": 0, "rate": None
    }


def test_erasing_nonzero_signal_counts_as_sign_change():
    assert sign_flip_statistics(torch.tensor([1.0, -1.0]), torch.zeros(2))["rate"] == 1.0


def test_paper_masking_positions_and_closed_boundary():
    patterns = deterministic_patterns()
    assert patterns["early"] == {"token": [1], "sequence": [], "prefix": list(range(1, 8))}
    assert patterns["late"] == {"token": [24], "sequence": [], "prefix": list(range(24, 32))}
    assert patterns["drift"] == {key: list(range(1, 33)) for key in ("token", "sequence", "prefix")}
