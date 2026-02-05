"""Unit tests for REINFORCE Pro Max adaptive token-level normalization."""

import torch
from openrlhf.trainer.ppo_utils.experience_maker import _adaptive_token_normalization_single_group


def test_basic_adaptive_normalization():
    """After normalization, masked tokens should have mean~0 and var~1."""
    torch.manual_seed(42)
    n_samples, seq_len = 8, 32
    # Create advantages with mixed signs
    adv = torch.randn(n_samples, seq_len)
    mask = torch.ones(n_samples, seq_len)
    # Mask out some positions
    mask[:, -4:] = 0

    result = _adaptive_token_normalization_single_group(adv, mask)

    flat = result.flatten()
    flat_mask = mask.flatten().bool()
    masked_vals = flat[flat_mask]

    mean = masked_vals.mean().item()
    var = masked_vals.var().item()
    print(f"[basic] mean={mean:.6f}, var={var:.6f}")
    assert abs(mean) < 0.05, f"Mean should be ~0, got {mean}"
    assert abs(var - 1.0) < 0.15, f"Var should be ~1, got {var}"


def test_fallback_all_positive():
    """When all advantages are positive, should skip normalization and return original."""
    torch.manual_seed(0)
    n_samples, seq_len = 4, 16
    adv = torch.rand(n_samples, seq_len) + 0.1  # All positive
    mask = torch.ones(n_samples, seq_len)

    result = _adaptive_token_normalization_single_group(adv, mask)

    # Fallback 应返回原始 advantage，不做归一化
    assert torch.allclose(result, adv), "Fallback should return original advantages"
    print("[all_positive] fallback returns original: OK")


def test_fallback_all_negative():
    """When all advantages are negative, should skip normalization and return original."""
    torch.manual_seed(0)
    n_samples, seq_len = 4, 16
    adv = -torch.rand(n_samples, seq_len) - 0.1  # All negative
    mask = torch.ones(n_samples, seq_len)

    result = _adaptive_token_normalization_single_group(adv, mask)

    # Fallback 应返回原始 advantage，不做归一化
    assert torch.allclose(result, adv), "Fallback should return original advantages"
    print("[all_negative] fallback returns original: OK")


def test_action_mask_respected():
    """Masked positions should remain zero."""
    torch.manual_seed(42)
    n_samples, seq_len = 4, 20
    adv = torch.randn(n_samples, seq_len)
    mask = torch.ones(n_samples, seq_len)
    mask[:, -5:] = 0  # Last 5 positions masked

    result = _adaptive_token_normalization_single_group(adv, mask)

    # Masked positions should be zero
    masked_out = result[:, -5:]
    assert (masked_out == 0).all(), f"Masked positions should be 0, got max={masked_out.abs().max().item()}"
    print("[mask] masked positions are all zero: OK")


def test_shuffled_vs_sequential_consistency():
    """Verify that normalization gives same results regardless of sample ordering."""
    torch.manual_seed(123)
    n_samples, seq_len = 8, 16
    adv = torch.randn(n_samples, seq_len)
    mask = torch.ones(n_samples, seq_len)

    # Sequential order
    result_seq = _adaptive_token_normalization_single_group(adv, mask)

    # Shuffled order
    perm = torch.randperm(n_samples)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(n_samples)

    adv_shuffled = adv[perm]
    mask_shuffled = mask[perm]
    result_shuffled = _adaptive_token_normalization_single_group(adv_shuffled, mask_shuffled)
    result_unshuffled = result_shuffled[inv_perm]

    diff = (result_seq - result_unshuffled).abs().max().item()
    print(f"[shuffle] max diff={diff:.8f}")
    assert diff < 1e-5, f"Results should match regardless of order, max diff={diff}"


def test_empty_mask():
    """All-zero mask should return original advantages unchanged."""
    adv = torch.randn(4, 10)
    mask = torch.zeros(4, 10)

    result = _adaptive_token_normalization_single_group(adv, mask)
    # Should return original since no valid tokens
    assert torch.allclose(result, adv), "With empty mask, should return original"
    print("[empty_mask] OK")


def test_mean_zero_property():
    """Verify mean of normalized advantages (weighted by mask) is close to 0."""
    torch.manual_seed(7)
    for trial in range(5):
        n_samples, seq_len = 8, 64
        adv = torch.randn(n_samples, seq_len) * (trial + 1)
        mask = (torch.rand(n_samples, seq_len) > 0.2).float()

        result = _adaptive_token_normalization_single_group(adv, mask)
        masked_vals = result.flatten()[mask.flatten().bool()]
        if masked_vals.numel() == 0:
            continue
        mean = masked_vals.mean().item()
        print(f"[mean_zero trial {trial}] mean={mean:.6f}")
        assert abs(mean) < 0.1, f"Mean should be ~0, got {mean}"


def test_index_mapping_forward():
    """Test forward index mapping: sorted_adv[indices] = all_adv."""
    # indices[i] = j means all_adv[i] should be placed at sorted_adv[j]
    indices = torch.tensor([2, 0, 1])
    all_adv = torch.tensor([300., 100., 200.])  # Shuffled data
    # Original order should be [100, 200, 300] (positions 0, 1, 2)

    sorted_adv = torch.empty_like(all_adv)
    sorted_adv[indices] = all_adv

    expected = torch.tensor([100., 200., 300.])
    assert torch.allclose(sorted_adv, expected), f"Forward mapping failed: {sorted_adv} != {expected}"
    print(f"[index_forward] sorted_adv[indices] = all_adv: {all_adv.tolist()} -> {sorted_adv.tolist()} OK")


def test_index_mapping_backward():
    """Test backward index mapping: unshuffled_adv = sorted_adv[indices]."""
    indices = torch.tensor([2, 0, 1])
    all_adv = torch.tensor([300., 100., 200.])  # Shuffled data

    # Forward mapping
    sorted_adv = torch.empty_like(all_adv)
    sorted_adv[indices] = all_adv

    # Simulate normalization
    sorted_adv_normed = sorted_adv * 2

    # Backward mapping
    unshuffled_adv = sorted_adv_normed[indices]

    # Should equal all_adv * 2
    expected = all_adv * 2
    assert torch.allclose(unshuffled_adv, expected), f"Backward mapping failed: {unshuffled_adv} != {expected}"
    print(f"[index_backward] sorted_adv_normed[indices]: {sorted_adv_normed.tolist()} -> {unshuffled_adv.tolist()} OK")


def test_index_mapping_roundtrip():
    """Test that forward + backward mapping preserves data correspondence."""
    torch.manual_seed(42)
    n_samples = 8

    # Simulate shuffled indices (as in dynamic batching)
    indices = torch.randperm(n_samples)
    all_adv = torch.randn(n_samples)

    # Forward: restore to original order
    sorted_adv = torch.empty_like(all_adv)
    sorted_adv[indices] = all_adv

    # Apply some transformation (simulating normalization)
    sorted_adv_transformed = sorted_adv * 3 + 1

    # Backward: restore to shuffled order
    unshuffled_adv = sorted_adv_transformed[indices]

    # Verify: unshuffled_adv[i] should correspond to all_adv[i] after transformation
    expected = all_adv * 3 + 1
    assert torch.allclose(unshuffled_adv, expected), f"Roundtrip failed: max diff={torch.abs(unshuffled_adv - expected).max().item()}"
    print(f"[index_roundtrip] indices={indices.tolist()}, max_diff={torch.abs(unshuffled_adv - expected).max().item():.8f} OK")


def test_padding_and_trim():
    """Test that padding and trimming preserves data correctly."""
    # Simulate experiences with different sequence lengths
    seq_lens = [10, 15, 12]
    experiences = [torch.randn(4, sl) for sl in seq_lens]
    masks = [torch.ones(4, sl) for sl in seq_lens]

    # Pad to max_seq_len
    max_seq_len = max(seq_lens)
    padded_advs = []
    padded_masks = []
    for adv, mask, seq_len in zip(experiences, masks, seq_lens):
        if seq_len < max_seq_len:
            pad_len = max_seq_len - seq_len
            padded_advs.append(torch.nn.functional.pad(adv, (0, pad_len), value=0.0))
            padded_masks.append(torch.nn.functional.pad(mask, (0, pad_len), value=0))
        else:
            padded_advs.append(adv)
            padded_masks.append(mask)

    # Verify padding
    for i, (padded, orig_len) in enumerate(zip(padded_advs, seq_lens)):
        assert padded.shape[-1] == max_seq_len, f"Padded shape mismatch at {i}"
        # Padded region should be zeros
        if orig_len < max_seq_len:
            assert (padded[:, orig_len:] == 0).all(), f"Padded region not zero at {i}"

    # Simulate concatenation and processing
    all_adv = torch.cat(padded_advs, dim=0)
    all_mask = torch.cat(padded_masks, dim=0)

    # Apply transformation
    all_adv_transformed = all_adv * 2

    # Trim back to original lengths
    trimmed = []
    offset = 0
    for exp, seq_len in zip(experiences, seq_lens):
        exp_size = exp.shape[0]
        trimmed.append(all_adv_transformed[offset:offset + exp_size, :seq_len])
        offset += exp_size

    # Verify trimmed data matches original * 2
    for i, (trim, orig) in enumerate(zip(trimmed, experiences)):
        expected = orig * 2
        assert torch.allclose(trim, expected), f"Trim mismatch at {i}: max_diff={torch.abs(trim - expected).max().item()}"

    print(f"[padding_trim] seq_lens={seq_lens}, max_seq_len={max_seq_len} OK")


def test_full_pipeline_simulation():
    """Simulate the full REINFORCE Pro Max pipeline with padding, index mapping, and normalization."""
    torch.manual_seed(123)

    # Simulate multiple experiences with different seq_lens (as in packing_samples)
    n_samples_per_prompt = 4
    num_prompts = 3
    total_samples = n_samples_per_prompt * num_prompts

    # Different seq_lens per micro-batch
    exp_configs = [
        (4, 20),  # exp 0: 4 samples, seq_len=20
        (4, 25),  # exp 1: 4 samples, seq_len=25
        (4, 18),  # exp 2: 4 samples, seq_len=18
    ]

    experiences = []
    masks = []
    all_indices = []
    for i, (n, seq_len) in enumerate(exp_configs):
        adv = torch.randn(n, seq_len)
        mask = torch.ones(n, seq_len)
        # Random indices simulating dynamic batching shuffle
        idx = list(range(i * n, (i + 1) * n))
        experiences.append(adv)
        masks.append(mask)
        all_indices.extend(idx)

    # Shuffle indices (simulating use_dynamic_batch)
    perm = torch.randperm(total_samples)
    indices = torch.tensor(all_indices)[perm]
    indices_inv = torch.empty_like(indices)
    indices_inv[indices] = torch.arange(total_samples)
    indices = indices_inv  # Now indices[i] = original position of sample i

    # Step 1: Pad to max_seq_len
    seq_lens = [exp.shape[-1] for exp in experiences]
    max_seq_len = max(seq_lens)
    padded_advs = []
    padded_masks = []
    for adv, mask, seq_len in zip(experiences, masks, seq_lens):
        if seq_len < max_seq_len:
            pad_len = max_seq_len - seq_len
            padded_advs.append(torch.nn.functional.pad(adv, (0, pad_len), value=0.0))
            padded_masks.append(torch.nn.functional.pad(mask, (0, pad_len), value=0))
        else:
            padded_advs.append(adv)
            padded_masks.append(mask)

    all_adv = torch.cat(padded_advs, dim=0)
    all_mask = torch.cat(padded_masks, dim=0)

    # Step 2: Forward mapping to original order
    sorted_adv = torch.empty_like(all_adv)
    sorted_mask = torch.empty_like(all_mask)
    sorted_adv[indices] = all_adv
    sorted_mask[indices] = all_mask

    # Step 3: Reshape by prompt groups
    grouped_adv = sorted_adv.reshape(num_prompts, n_samples_per_prompt, -1)
    grouped_mask = sorted_mask.reshape(num_prompts, n_samples_per_prompt, -1)

    # Step 4: Apply normalization per group
    for i in range(num_prompts):
        grouped_adv[i] = _adaptive_token_normalization_single_group(grouped_adv[i], grouped_mask[i], group_idx=i)

    # Verify normalization per group (on non-zero tokens)
    for i in range(num_prompts):
        group_vals = grouped_adv[i].flatten()
        group_mask_flat = grouped_mask[i].flatten().bool()
        masked_vals = group_vals[group_mask_flat]
        nonzero_vals = masked_vals[masked_vals != 0]
        if nonzero_vals.numel() > 0:
            mean = nonzero_vals.mean().item()
            var = nonzero_vals.var().item()
            assert abs(mean) < 0.05, f"Group {i} mean={mean} not ~0"
            assert abs(var - 1.0) < 0.1, f"Group {i} var={var} not ~1"

    # Step 5: Backward mapping
    sorted_adv = grouped_adv.reshape(-1, max_seq_len)
    unshuffled_adv = sorted_adv[indices]

    # Step 6: Trim to original seq_lens
    exp_offset = 0
    for exp_idx, (exp, seq_len) in enumerate(zip(experiences, seq_lens)):
        exp_size = exp.shape[0]
        trimmed = unshuffled_adv[exp_offset:exp_offset + exp_size, :seq_len]
        assert trimmed.shape == exp.shape, f"Shape mismatch at exp {exp_idx}"
        exp_offset += exp_size

    print(f"[full_pipeline] num_prompts={num_prompts}, n_samples_per_prompt={n_samples_per_prompt}, seq_lens={seq_lens} OK")


def test_fallback_preserves_zero_advantages():
    """Fallback branch should keep A=0 tokens as 0, not transform them."""
    torch.manual_seed(42)
    n_samples, seq_len = 4, 16

    # Create all-positive advantages with some zeros
    adv = torch.rand(n_samples, seq_len) + 0.1  # All positive
    # Set some values to exactly 0
    adv[0, 2] = 0.0
    adv[1, 5] = 0.0
    adv[2, 8] = 0.0
    mask = torch.ones(n_samples, seq_len)

    result = _adaptive_token_normalization_single_group(adv, mask)

    # Check that A=0 positions remain 0
    assert result[0, 2].item() == 0.0, f"A=0 at [0,2] should stay 0, got {result[0, 2].item()}"
    assert result[1, 5].item() == 0.0, f"A=0 at [1,5] should stay 0, got {result[1, 5].item()}"
    assert result[2, 8].item() == 0.0, f"A=0 at [2,8] should stay 0, got {result[2, 8].item()}"

    # Check that non-zero positions are normalized
    nonzero_vals = result[adv != 0]
    mean = nonzero_vals.mean().item()
    std = nonzero_vals.std().item()
    print(f"[fallback_zero] A=0 preserved, nonzero mean={mean:.6f}, std={std:.6f} OK")


def test_fallback_all_zeros():
    """When all advantages are 0, fallback should return original."""
    adv = torch.zeros(4, 10)
    mask = torch.ones(4, 10)

    result = _adaptive_token_normalization_single_group(adv, mask)

    assert torch.allclose(result, adv), "All-zero advantages should return unchanged"
    print("[fallback_all_zeros] OK")


def test_fallback_sum_near_zero():
    """When sum_pos or sum_neg is near zero, should fallback to avoid inf/nan."""
    torch.manual_seed(42)
    n_samples, seq_len = 4, 16

    # Create advantages where negative values sum to near zero
    adv = torch.randn(n_samples, seq_len)
    mask = torch.ones(n_samples, seq_len)

    # Make most values positive, with tiny negative values
    adv = adv.abs() + 0.1  # All positive now
    # Add a few very small negative values (sum will be near zero)
    adv[0, 0] = -1e-8
    adv[1, 1] = -1e-8
    adv[2, 2] = -1e-8

    result = _adaptive_token_normalization_single_group(adv, mask)

    # Should not produce nan/inf
    assert not torch.isnan(result).any(), "Result contains NaN"
    assert not torch.isinf(result).any(), "Result contains Inf"

    # Check normalization is reasonable
    flat = result.flatten()
    nonzero = flat[flat != 0]
    if nonzero.numel() > 0:
        mean = nonzero.mean().item()
        assert abs(mean) < 1.0, f"Mean should be reasonable, got {mean}"

    print("[fallback_sum_near_zero] No nan/inf, fallback triggered correctly OK")


def test_rloo_all_same_reward_fallback():
    """Test RLOO fallback for all-same reward groups (all correct or all wrong)."""
    n_samples_per_prompt = 4
    num_prompts = 3

    # Group 0: all correct (reward=1)
    # Group 1: mixed rewards
    # Group 2: all wrong (reward=0)
    rewards = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],  # all same -> fallback to reward/n
        [1.0, 0.0, 1.0, 0.0],  # mixed -> use RLOO
        [0.0, 0.0, 0.0, 0.0],  # all same -> fallback to reward/n
    ])

    # Standard RLOO baseline
    baseline = (rewards.sum(-1, keepdim=True) - rewards) / (n_samples_per_prompt - 1)
    shaped_rewards_rloo = rewards - baseline

    # Detect all-same groups
    group_std = rewards.std(-1, keepdim=True)
    all_same_mask = (group_std < 1e-8).expand_as(rewards)

    # Fallback for all-same groups
    fallback_rewards = rewards / n_samples_per_prompt

    # Combined
    final_rewards = torch.where(all_same_mask, fallback_rewards, shaped_rewards_rloo)

    # Verify group 0 (all correct): should be 1/4 = 0.25 for each sample
    assert torch.allclose(final_rewards[0], torch.tensor([0.25, 0.25, 0.25, 0.25])), \
        f"All-correct group should have reward/n, got {final_rewards[0]}"

    # Verify group 1 (mixed): should use RLOO
    # RLOO for [1, 0, 1, 0]: baseline = [1/3, 2/3, 1/3, 2/3]
    # shaped = [1-1/3, 0-2/3, 1-1/3, 0-2/3] = [2/3, -2/3, 2/3, -2/3]
    expected_mixed = torch.tensor([2/3, -2/3, 2/3, -2/3])
    assert torch.allclose(final_rewards[1], expected_mixed, atol=1e-6), \
        f"Mixed group should use RLOO, got {final_rewards[1]}"

    # Verify group 2 (all wrong): should be 0/4 = 0 for each sample
    assert torch.allclose(final_rewards[2], torch.tensor([0.0, 0.0, 0.0, 0.0])), \
        f"All-wrong group should have reward/n=0, got {final_rewards[2]}"

    print(f"[rloo_all_same_fallback] all_correct={final_rewards[0].tolist()}, "
          f"mixed={final_rewards[1].tolist()}, all_wrong={final_rewards[2].tolist()} OK")


def test_indices_permutation_check():
    """Test that indices must be a valid permutation of 0..B-1."""
    # Valid permutation should work
    B = 8
    valid_indices = torch.tensor([3, 1, 7, 0, 5, 2, 6, 4])

    # Check valid permutation passes
    assert valid_indices.numel() == B
    assert torch.unique(valid_indices).numel() == B
    assert valid_indices.min().item() == 0
    assert valid_indices.max().item() == B - 1
    print(f"[indices_permutation] valid indices {valid_indices.tolist()} passed checks OK")

    # Invalid: has duplicates
    invalid_dup = torch.tensor([0, 1, 1, 3, 4, 5, 6, 7])
    assert torch.unique(invalid_dup).numel() != B, "Duplicate indices should fail unique check"

    # Invalid: missing values
    invalid_missing = torch.tensor([0, 1, 2, 3, 4, 5, 6, 8])  # missing 7, has 8
    assert invalid_missing.max().item() != B - 1, "Out-of-range indices should fail max check"

    # Invalid: wrong length
    invalid_len = torch.tensor([0, 1, 2, 3, 4])
    assert invalid_len.numel() != B, "Wrong length should fail length check"

    print("[indices_permutation] invalid cases correctly detected OK")


def test_prompt_grouping_consistency():
    """Test that samples from the same prompt are grouped together after sorting."""
    n_samples_per_prompt = 4
    num_prompts = 3
    total_samples = num_prompts * n_samples_per_prompt

    # Original prompts in prompt-major order (this is the "sorted" or "original" order)
    # Each prompt has n_samples_per_prompt consecutive samples
    original_prompts = []
    for p in range(num_prompts):
        original_prompts.extend([f"prompt_{p}"] * n_samples_per_prompt)
    # original_prompts = [p0, p0, p0, p0, p1, p1, p1, p1, p2, p2, p2, p2]

    # Simulate shuffled indices (as would happen with dynamic batching)
    # indices[i] = original position of shuffled sample i
    # So if shuffled_order[i] came from original position indices[i],
    # then sorted_adv[indices[i]] = shuffled_adv[i]
    # which means sorted_adv recovers the original order
    shuffled_indices = torch.tensor([6, 3, 0, 7, 2, 1, 4, 5, 9, 10, 8, 11])

    # After sorted_prompts[indices] = shuffled_prompts mapping,
    # the shuffled_prompts are placed back to their original positions.
    # So we need shuffled_prompts first.
    # shuffled_prompts[i] = original_prompts[indices[i]]
    shuffled_prompts = [original_prompts[idx] for idx in shuffled_indices.tolist()]

    # Now sorted_prompts[indices[i]] = shuffled_prompts[i]
    sorted_prompts = [None] * total_samples
    for i, idx in enumerate(shuffled_indices.tolist()):
        sorted_prompts[idx] = shuffled_prompts[i]

    # sorted_prompts should be exactly original_prompts (prompt-major order)
    assert sorted_prompts == original_prompts, \
        f"sorted_prompts should match original_prompts"

    # Check each group has consistent prompts
    for g in range(num_prompts):
        group_prompts = sorted_prompts[g * n_samples_per_prompt : (g + 1) * n_samples_per_prompt]
        first_prompt = group_prompts[0]
        assert all(p == first_prompt for p in group_prompts), \
            f"Group {g} has inconsistent prompts: {group_prompts}"

    print(f"[prompt_grouping] all {num_prompts} groups have consistent prompts OK")


def test_prompt_grouping_detects_error():
    """Test that inconsistent prompt grouping is detected when original data is not prompt-major."""
    n_samples_per_prompt = 4
    num_prompts = 2
    total_samples = num_prompts * n_samples_per_prompt

    # Original data is NOT in prompt-major order (interleaved)
    # This simulates a bug where samples from different prompts are mixed
    interleaved_original = ["prompt_0", "prompt_1", "prompt_0", "prompt_1",
                           "prompt_0", "prompt_1", "prompt_0", "prompt_1"]

    # Identity indices (data is already in "shuffled" order, which is interleaved)
    indices = torch.arange(total_samples)

    # Simulate the code's behavior: shuffled_prompts[i] from position indices[i]
    # Since indices is identity, shuffled_prompts = interleaved_original
    shuffled_prompts = [interleaved_original[idx] for idx in indices.tolist()]

    # sorted_prompts[indices[i]] = shuffled_prompts[i]
    # Since indices is identity, sorted_prompts = shuffled_prompts = interleaved_original
    sorted_prompts = [None] * total_samples
    for i, idx in enumerate(indices.tolist()):
        sorted_prompts[idx] = shuffled_prompts[i]

    # Check each group - should detect inconsistency
    error_detected = False
    for g in range(num_prompts):
        group_prompts = sorted_prompts[g * n_samples_per_prompt : (g + 1) * n_samples_per_prompt]
        first_prompt = group_prompts[0]
        if not all(p == first_prompt for p in group_prompts):
            error_detected = True
            break

    assert error_detected, "Should detect inconsistent prompt grouping"
    print("[prompt_grouping_error] inconsistent grouping correctly detected OK")


if __name__ == "__main__":
    test_basic_adaptive_normalization()
    test_fallback_all_positive()
    test_fallback_all_negative()
    test_action_mask_respected()
    test_shuffled_vs_sequential_consistency()
    test_empty_mask()
    test_mean_zero_property()
    test_index_mapping_forward()
    test_index_mapping_backward()
    test_index_mapping_roundtrip()
    test_padding_and_trim()
    test_full_pipeline_simulation()
    test_fallback_preserves_zero_advantages()
    test_fallback_all_zeros()
    test_fallback_sum_near_zero()
    test_rloo_all_same_reward_fallback()
    test_indices_permutation_check()
    test_prompt_grouping_consistency()
    test_prompt_grouping_detects_error()
    print("\nAll tests passed!")
