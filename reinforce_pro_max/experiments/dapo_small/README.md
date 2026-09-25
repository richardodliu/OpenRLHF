# DAPO data and reward snapshot

This directory stores the dataset and reward function used for the small DAPO
experiment. It contains inputs and validation records, not experiment results.

## Files

| File | Contents |
| --- | --- |
| `distinct-prompts-with-rewards.parquet` | Full public deduplicated release: 17,398 exact-unique prompts and their original reward labels. |
| `train-3200.jsonl` | Fixed 3,200-example training subset with `question` and `label` fields. |
| `mathverify_dapo.py` | Reward function with final line-start `Answer:` section extraction. |
| `source.json` | Public source URL, pinned revision, size and SHA-256. |
| `training-selection.json` | Exclusions, zero-based source row indices, seed and training-file hash. |
| `reward-validation.json` | Validation results for this exact reward-file hash. |
| `SHA256SUMS` | Checksums of data, reward code and JSON records. |

## Dataset provenance and selection

The original dataset is [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k).
The bundled Parquet is the public
[YouJiacheng deduplicated release](https://huggingface.co/datasets/YouJiacheng/DAPO-Math-17k-dedup),
pinned at revision `6e26a33abdabd3e6aaa1d742326b790758f7dbc5`.
Consult these source repositories for dataset attribution and licensing.
This is a pinned input for this experiment; it is not a claim that TRM used
this exact release or subset.

Deduplication here means exact prompt-text deduplication, not semantic
deduplication. All 17,398 bundled prompt/label pairs were found in the existing
local DAPO conversion, which contained 17,917 records. That conversion's SHA-256
was `b94a18b981121c5c9adcd22f86f3dbb7a6c2e241193e4e547f07075daec40b8b`.

For the training subset, seven source rows whose questions had conflicting
labels in that conversion were excluded. Ten further rows exceeded the
1,024-token prompt limit under the experiment's Qwen2.5-Math-7B preprocessing.
From the remaining 17,381 rows, `random.Random(42).sample(eligible, 3200)` selected
the subset, sorted by source row index before serialization. The manifest records
all selected and excluded indices; the bundled subset is the authoritative input.
Prompts, including their original `Answer:` instructions, and ground-truth labels
were preserved without rewriting.

The AIME25 overlap check removed only the known DAPO instruction wrapper and
compared NFKC/whitespace-normalized question text. It found no exact overlaps;
this is not a semantic contamination audit.

## Reward behavior

Relative to the original `mathverify_v3.py`, only `extract_answer` was changed:

1. If a line-start `Answer:` marker exists, restrict the passage to the text after
   its last occurrence (case-insensitive, with whitespace allowed).
2. Continue the existing boxed-answer extraction on that passage.
3. If no box is present but an `Answer:` section exists, use its first nonempty
   line after stripping surrounding whitespace.
4. Continue the original mathematical equivalence checking and reward logic.

Without an `Answer:` marker, the original boxed-answer behavior is retained.
The existing final-200-character extraction window is unchanged: an `Answer:`
marker outside that window is not seen by the extractor during reward scoring.
Correct, incorrect and unextractable answers still receive `1`, `-0.5` and `-1`,
respectively. For example, `\boxed{35}\nAnswer: \boxed{34}` is graded using `34`.

Dependencies are `torch`, `sympy` and `pylatexenc`. From the repository root,
the relevant training arguments are:

```bash
--prompt_data reinforce_pro_max/experiments/dapo_small/train-3200.jsonl \
--input_key question --label_key label \
--remote_rm_url reinforce_pro_max/experiments/dapo_small/mathverify_dapo.py
```

The recorded reward validation passed 22 targeted cases, five unchanged legacy
cases and correct-answer emission checks for all 3,200 training labels. These
checks validate parsing and grading behavior, not model performance.

Verify the archived bytes from this directory with `sha256sum -c SHA256SUMS`.

## Standalone AIME25 evaluation

`aime25.jsonl` contains the 960 AIME25 records used in this experiment: 30
questions repeated 32 times, preserving source messages, answers, order and
original row indices. `evaluation-source.json` records the LUFFY source commit
and checksums. LUFFY is the data source only; evaluation does not import its code
or require a checkout of that repository.

`evaluate_aime25.py` reads these local records and calls this directory's
`mathverify_dapo.py:reward_func` directly. Correctness is `reward == 1`, with the
same answer extraction, final-200-character window and mathematical grading as
training. Both `Answer:` and boxed answers are accepted. This replaces the
previously planned OAT grader; reported scores use this shared reward function.

Generation retains temperature 1.0, top-p 0.95, seed 42 plus the selected row's
position, context length 4,096 and generation cap 3,072. The source system message
is removed before applying the model's own chat template, as in the existing
evaluation setup. All 32 responses contribute to avg@32 (sample-mean accuracy),
not pass@32. The runner records data, reward, evaluator and model hashes and saves
generated text and individual scores before producing aggregate results.

In an environment with `vllm`, `transformers`, `torch`, `sympy` and `pylatexenc`:

```bash
python reinforce_pro_max/experiments/dapo_small/evaluate_aime25.py \
  --model /path/to/completed/model --output /path/to/evaluation
```

The script defaults to data and reward files beside itself. Add `--validate-only`
to check tokenization, repetition counts and grading without loading model weights
or generating responses. Standard evaluation uses four GPUs for tensor parallelism.

## Four-arm component comparison

`study-arms.json` records the current configuration. Each arm starts from the same
Qwen2.5-Math-7B initial model, uses seed 42 and runs 100 updates on the same subset.
All four arms use token-level PPO clipping with the differentiable ratio
`exp(current_training_logprob - rollout_inference_logprob)` and clip interval
`[0.8, 1.28]`, matching the TRM clipping baseline's interval.

| Arm | Advantage processing | Pro mask | Extra IS multiplier |
| --- | --- | --- | --- |
| baseline | RLOO + global normalization | No | No |
| max_only | Max sign-dependent scaling | No | No |
| pro_only | RLOO + global normalization | Yes | No |
| pro_max | Max sign-dependent scaling | Yes | No |

Pro uses the existing detached causal prefix geometric mean of cached-old /
rollout probabilities, with thresholds `[0.5, 5]`. It adds only a mask; the common
PPO ratio already accounts for rollout probabilities. Max replaces the baseline's
global normalization with its own scaling, so its contrast measures that full
advantage-processing change, not just one scalar in isolation.

`rollout_clip.patch` records the changes against the study's frozen training source,
including collection of rollout log probabilities for all arms. The implementation
uses `policy_loss_type=ppo_rollout`; existing `ppo` behavior is unchanged.
`check_rollout_clip.py --source /path/to/patched/source` checks both loss values and
gradients, with and without the mask and with different advantage scales.

`run_four_arm_study.py --run-dir /path/to/prepared/study` runs the current plan,
waits for the preceding study's lock and preserves that study's outputs. It needs
the prepared study directory (source, configuration, data and validation records).
The former RLOO + ICEPOP run is an archived additional reference, not the baseline
for this four-arm comparison. `run_pro_only_extension.py` is the superseded legacy
queue extension and must not be used with the current plan.

All four arms use the same reward function and local AIME25 evaluation. Comparisons
include Max vs baseline, Pro vs baseline, Pro Max vs Max, and Pro Max vs Pro.
A single training seed and 100 updates per arm scope the conclusions to this run.
